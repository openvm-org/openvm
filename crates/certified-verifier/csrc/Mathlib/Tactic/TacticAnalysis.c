// Lean compiler output
// Module: Mathlib.Tactic.TacticAnalysis
// Imports: public import Init public meta import Init public meta import Lean.Util.Heartbeats public meta import Lean.Server.InfoUtils public meta import Mathlib.Lean.Elab.Tactic.Meta public meta import Lean.Compiler.IR.CompilerM public import Lean.Elab.Command public import Mathlib.Lean.ContextInfo
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
lean_object* l_Lean_Environment_evalConst___redArg(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Elab_Info_stx(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getHeadInfo_x3f(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Elab_PartialContextInfo_mergeIntoOuter_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_instMonadCommandElabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_instMonadCommandElabM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Info_updateContext_x3f(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toList___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_Name_cmp(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Environment_evalConstCheck___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_io_user_error(lean_object*);
lean_object* l_Lean_registerPersistentEnvExtensionUnsafe___redArg(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_decl_get_sorry_dep(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
uint8_t l_Lean_instBEqAttributeKind_beq(uint8_t, uint8_t);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticAnalysis"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(85, 245, 235, 249, 235, 40, 236, 31)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "enable the tactic analysis framework"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_tacticAnalysis;
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1;
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCodeCapturingInfoTree(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCodeCapturingInfoTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "TacticAnalysis"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(73, 121, 71, 227, 64, 118, 131, 70)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(130, 35, 171, 250, 33, 29, 167, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_Entry_import(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_Entry_import___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__0(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticAnalysisExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(73, 121, 71, 227, 64, 118, 131, 70)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(175, 82, 203, 218, 43, 234, 107, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 0, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysisExt;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(107, 67, 254, 234, 65, 174, 209, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "invalid attribute 'tacticAnalysis', declaration is in an imported module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "invalid attribute 'tacticAnalysis', must be global"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "tacticAnalysis: missing option name."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(156, 189, 6, 141, 204, 166, 99, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(61, 26, 41, 213, 39, 24, 236, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 218, 209, 51, 236, 87, 247, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(199, 223, 127, 56, 192, 119, 159, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(246, 234, 131, 180, 189, 103, 6, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__12_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(199, 74, 254, 201, 91, 13, 251, 190)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__12_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__12_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__13_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__12_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(10, 37, 132, 162, 125, 64, 223, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__13_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__13_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__14_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__13_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(31, 122, 215, 120, 78, 70, 159, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__14_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__14_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__15_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__14_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(124, 17, 31, 163, 1, 146, 184, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__15_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__15_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__16_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__16_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__17_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__17_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__17_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__18_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__18_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__19_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__19_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__19_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__20_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__20_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__21_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__21_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__22_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(192, 159, 170, 95, 241, 182, 12, 227)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__22_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__22_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__23_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed, .m_arity = 9, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__22_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__23_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__23_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__24_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__22_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__24_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__24_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__25_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "adds a tacticAnalysis pass"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__25_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__25_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__26_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__26_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__27_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__27_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(187, 150, 238, 148, 228, 221, 116, 224)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(101, 218, 47, 72, 64, 31, 83, 55)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(238, 60, 149, 138, 55, 149, 59, 229)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "by"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(33, 100, 221, 244, 231, 185, 222, 214)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "withAnnotateState"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(27, 100, 151, 108, 10, 177, 75, 150)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "anyGoals"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(168, 19, 163, 3, 232, 106, 175, 32)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__31_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__29_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__33_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Command_instMonadCommandElabM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Command_instMonadCommandElabM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "unexpected context-free info tree node"};
static const lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "_private.Lean.Server.InfoUtils.0.Lean.Elab.InfoTree.visitM.go"};
static const lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Lean.Server.InfoUtils"};
static const lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPasses(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPasses___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__1___boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(73, 121, 71, 227, 64, 118, 131, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(26, 26, 45, 40, 120, 33, 89, 132)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1736486024____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1736486024____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_skip_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_skip_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_continue_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_continue_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_accept_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_accept_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "original tactic '"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__1;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "' failed: "};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__1;
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "seq1"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(242, 140, 137, 56, 141, 11, 143, 117)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_TacticAnalysis_runPass_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_TacticAnalysis_runPass_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__1_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "internal error in tactic analysis: accepted an empty sequence."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "done"};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 161, 179, 82, 204, 87, 48, 123)}};
static const lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_Config_ofComplex(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "dummy"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(85, 245, 235, 249, 235, 40, 236, 31)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(223, 133, 125, 150, 169, 194, 15, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_tacticAnalysis_dummy;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_));
v___x_47_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_));
v___x_48_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0(v___x_46_, v___x_47_, v___x_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4____boxed(lean_object* v_a_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_();
return v_res_50_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__0(void){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = l_instMonadEIO(lean_box(0));
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__0, &lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__0_once, _init_lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__0);
v___x_53_ = l_StateRefT_x27_instMonad___redArg(v___x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode(lean_object* v_i_56_, lean_object* v_goal_57_, lean_object* v_code_58_, lean_object* v_a_59_, lean_object* v_a_60_){
_start:
{
lean_object* v___x_62_; lean_object* v_toApplicative_63_; lean_object* v_toFunctor_64_; lean_object* v_toSeq_65_; lean_object* v_toSeqLeft_66_; lean_object* v_toSeqRight_67_; lean_object* v___f_68_; lean_object* v___f_69_; lean_object* v___f_70_; lean_object* v___f_71_; lean_object* v___x_72_; lean_object* v___f_73_; lean_object* v___f_74_; lean_object* v___f_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v_ctxI_79_; lean_object* v_tacI_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_62_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1, &lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1_once, _init_lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1);
v_toApplicative_63_ = lean_ctor_get(v___x_62_, 0);
v_toFunctor_64_ = lean_ctor_get(v_toApplicative_63_, 0);
v_toSeq_65_ = lean_ctor_get(v_toApplicative_63_, 2);
v_toSeqLeft_66_ = lean_ctor_get(v_toApplicative_63_, 3);
v_toSeqRight_67_ = lean_ctor_get(v_toApplicative_63_, 4);
v___f_68_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__2));
v___f_69_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__3));
lean_inc_ref_n(v_toFunctor_64_, 2);
v___f_70_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_70_, 0, v_toFunctor_64_);
v___f_71_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_71_, 0, v_toFunctor_64_);
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v___f_70_);
lean_ctor_set(v___x_72_, 1, v___f_71_);
lean_inc(v_toSeqRight_67_);
v___f_73_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_73_, 0, v_toSeqRight_67_);
lean_inc(v_toSeqLeft_66_);
v___f_74_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_74_, 0, v_toSeqLeft_66_);
lean_inc(v_toSeq_65_);
v___f_75_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_75_, 0, v_toSeq_65_);
v___x_76_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_76_, 0, v___x_72_);
lean_ctor_set(v___x_76_, 1, v___f_68_);
lean_ctor_set(v___x_76_, 2, v___f_75_);
lean_ctor_set(v___x_76_, 3, v___f_74_);
lean_ctor_set(v___x_76_, 4, v___f_73_);
v___x_77_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v___f_69_);
v___x_78_ = l_StateRefT_x27_instMonad___redArg(v___x_77_);
v_ctxI_79_ = lean_ctor_get(v_i_56_, 0);
lean_inc_ref(v_ctxI_79_);
v_tacI_80_ = lean_ctor_get(v_i_56_, 1);
lean_inc_ref(v_tacI_80_);
lean_dec_ref(v_i_56_);
v___x_81_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 4);
lean_closure_set(v___x_81_, 0, lean_box(0));
lean_closure_set(v___x_81_, 1, lean_box(0));
lean_closure_set(v___x_81_, 2, v___x_78_);
lean_closure_set(v___x_81_, 3, lean_box(0));
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, lean_box(0));
lean_ctor_set(v___x_82_, 1, v___x_81_);
v___x_83_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode(v_ctxI_79_, v_tacI_80_, v_goal_57_, v_code_58_, v___x_82_, v_a_59_, v_a_60_);
lean_dec_ref(v_tacI_80_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___boxed(lean_object* v_i_84_, lean_object* v_goal_85_, lean_object* v_code_86_, lean_object* v_a_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode(v_i_84_, v_goal_85_, v_code_86_, v_a_87_, v_a_88_);
lean_dec(v_a_88_);
lean_dec_ref(v_a_87_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCodeCapturingInfoTree(lean_object* v_i_91_, lean_object* v_goal_92_, lean_object* v_code_93_, lean_object* v_a_94_, lean_object* v_a_95_){
_start:
{
lean_object* v_ctxI_97_; lean_object* v_tacI_98_; lean_object* v___x_99_; 
v_ctxI_97_ = lean_ctor_get(v_i_91_, 0);
lean_inc_ref(v_ctxI_97_);
v_tacI_98_ = lean_ctor_get(v_i_91_, 1);
lean_inc_ref(v_tacI_98_);
lean_dec_ref(v_i_91_);
v___x_99_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCodeCapturingInfoTree(v_ctxI_97_, v_tacI_98_, v_goal_92_, v_code_93_, v_a_94_, v_a_95_);
lean_dec_ref(v_tacI_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCodeCapturingInfoTree___boxed(lean_object* v_i_100_, lean_object* v_goal_101_, lean_object* v_code_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCodeCapturingInfoTree(v_i_100_, v_goal_101_, v_code_102_, v_a_103_, v_a_104_);
lean_dec(v_a_104_);
lean_dec_ref(v_a_103_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1(lean_object* v_e_114_, lean_object* v_env_115_, lean_object* v_opts_116_){
_start:
{
lean_object* v_declName_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v_declName_117_ = lean_ctor_get(v_e_114_, 0);
lean_inc(v_declName_117_);
lean_dec_ref(v_e_114_);
v___x_118_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3));
v___x_119_ = l_Lean_Environment_evalConstCheck___redArg(v_env_115_, v_opts_116_, v___x_118_, v_declName_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___boxed(lean_object* v_e_120_, lean_object* v_env_121_, lean_object* v_opts_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1(v_e_120_, v_env_121_, v_opts_122_);
lean_dec_ref(v_opts_122_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__4(lean_object* v_e_124_, lean_object* v_env_125_, lean_object* v_opts_126_){
_start:
{
lean_object* v_optionName_127_; uint8_t v___x_128_; lean_object* v___x_129_; 
v_optionName_127_ = lean_ctor_get(v_e_124_, 1);
v___x_128_ = 1;
v___x_129_ = l_Lean_Environment_evalConst___redArg(v_env_125_, v_opts_126_, v_optionName_127_, v___x_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__4___boxed(lean_object* v_e_130_, lean_object* v_env_131_, lean_object* v_opts_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__4(v_e_130_, v_env_131_, v_opts_132_);
lean_dec_ref(v_opts_132_);
lean_dec_ref(v_env_131_);
lean_dec_ref(v_e_130_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg(lean_object* v_e_134_){
_start:
{
if (lean_obj_tag(v_e_134_) == 0)
{
lean_object* v_a_136_; lean_object* v___x_138_; uint8_t v_isShared_139_; uint8_t v_isSharedCheck_144_; 
v_a_136_ = lean_ctor_get(v_e_134_, 0);
v_isSharedCheck_144_ = !lean_is_exclusive(v_e_134_);
if (v_isSharedCheck_144_ == 0)
{
v___x_138_ = v_e_134_;
v_isShared_139_ = v_isSharedCheck_144_;
goto v_resetjp_137_;
}
else
{
lean_inc(v_a_136_);
lean_dec(v_e_134_);
v___x_138_ = lean_box(0);
v_isShared_139_ = v_isSharedCheck_144_;
goto v_resetjp_137_;
}
v_resetjp_137_:
{
lean_object* v___x_140_; lean_object* v___x_142_; 
v___x_140_ = lean_mk_io_user_error(v_a_136_);
if (v_isShared_139_ == 0)
{
lean_ctor_set_tag(v___x_138_, 1);
lean_ctor_set(v___x_138_, 0, v___x_140_);
v___x_142_ = v___x_138_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v___x_140_);
v___x_142_ = v_reuseFailAlloc_143_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
return v___x_142_;
}
}
}
else
{
lean_object* v_a_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_152_; 
v_a_145_ = lean_ctor_get(v_e_134_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v_e_134_);
if (v_isSharedCheck_152_ == 0)
{
v___x_147_ = v_e_134_;
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_a_145_);
lean_dec(v_e_134_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___x_150_; 
if (v_isShared_148_ == 0)
{
lean_ctor_set_tag(v___x_147_, 0);
v___x_150_ = v___x_147_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 1, 0);
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
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg___boxed(lean_object* v_e_153_, lean_object* v_a_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg(v_e_153_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0(lean_object* v_00_u03b1_156_, lean_object* v_e_157_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg(v_e_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___boxed(lean_object* v_00_u03b1_160_, lean_object* v_e_161_, lean_object* v_a_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0(v_00_u03b1_160_, v_e_161_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_Entry_import(lean_object* v_e_164_, lean_object* v_a_165_){
_start:
{
lean_object* v_env_167_; lean_object* v_opts_168_; lean_object* v_declName_169_; lean_object* v_optionName_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_209_; 
v_env_167_ = lean_ctor_get(v_a_165_, 0);
v_opts_168_ = lean_ctor_get(v_a_165_, 1);
v_declName_169_ = lean_ctor_get(v_e_164_, 0);
v_optionName_170_ = lean_ctor_get(v_e_164_, 1);
v_isSharedCheck_209_ = !lean_is_exclusive(v_e_164_);
if (v_isSharedCheck_209_ == 0)
{
v___x_172_ = v_e_164_;
v_isShared_173_ = v_isSharedCheck_209_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_optionName_170_);
lean_inc(v_declName_169_);
lean_dec(v_e_164_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_209_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_174_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_Entry_import_unsafe__1___closed__3));
lean_inc_ref(v_env_167_);
v___x_175_ = l_Lean_Environment_evalConstCheck___redArg(v_env_167_, v_opts_168_, v___x_174_, v_declName_169_);
v___x_176_ = lp_mathlib_IO_ofExcept___at___00Mathlib_TacticAnalysis_Entry_import_spec__0___redArg(v___x_175_);
if (lean_obj_tag(v___x_176_) == 0)
{
lean_object* v_a_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_200_; 
v_a_177_ = lean_ctor_get(v___x_176_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_176_);
if (v_isSharedCheck_200_ == 0)
{
v___x_179_ = v___x_176_;
v_isShared_180_ = v_isSharedCheck_200_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_a_177_);
lean_dec(v___x_176_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_200_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___y_182_; uint8_t v___x_189_; lean_object* v___x_190_; 
v___x_189_ = 1;
v___x_190_ = l_Lean_Environment_evalConst___redArg(v_env_167_, v_opts_168_, v_optionName_170_, v___x_189_);
lean_dec(v_optionName_170_);
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v___x_191_; 
lean_dec_ref_known(v___x_190_, 1);
v___x_191_ = lean_box(0);
v___y_182_ = v___x_191_;
goto v___jp_181_;
}
else
{
lean_object* v_a_192_; lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_199_; 
v_a_192_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_199_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_199_ == 0)
{
v___x_194_ = v___x_190_;
v_isShared_195_ = v_isSharedCheck_199_;
goto v_resetjp_193_;
}
else
{
lean_inc(v_a_192_);
lean_dec(v___x_190_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_199_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v___x_197_; 
if (v_isShared_195_ == 0)
{
v___x_197_ = v___x_194_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v_a_192_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
v___y_182_ = v___x_197_;
goto v___jp_181_;
}
}
}
v___jp_181_:
{
lean_object* v___x_184_; 
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 1, v___y_182_);
lean_ctor_set(v___x_172_, 0, v_a_177_);
v___x_184_ = v___x_172_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_a_177_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v___y_182_);
v___x_184_ = v_reuseFailAlloc_188_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
lean_object* v___x_186_; 
if (v_isShared_180_ == 0)
{
lean_ctor_set(v___x_179_, 0, v___x_184_);
v___x_186_ = v___x_179_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_184_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
return v___x_186_;
}
}
}
}
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_del_object(v___x_172_);
lean_dec(v_optionName_170_);
v_a_201_ = lean_ctor_get(v___x_176_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_176_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_176_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_176_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_Entry_import___boxed(lean_object* v_e_210_, lean_object* v_a_211_, lean_object* v_a_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_Mathlib_TacticAnalysis_Entry_import(v_e_210_, v_a_211_);
lean_dec_ref(v_a_211_);
return v_res_213_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___lam__0(lean_object* v_a_214_, lean_object* v_b_215_){
_start:
{
lean_object* v_declName_216_; lean_object* v_optionName_217_; lean_object* v_declName_218_; lean_object* v_optionName_219_; uint8_t v___x_220_; 
v_declName_216_ = lean_ctor_get(v_a_214_, 0);
v_optionName_217_ = lean_ctor_get(v_a_214_, 1);
v_declName_218_ = lean_ctor_get(v_b_215_, 0);
v_optionName_219_ = lean_ctor_get(v_b_215_, 1);
v___x_220_ = l_Lean_Name_cmp(v_declName_216_, v_declName_218_);
if (v___x_220_ == 1)
{
uint8_t v___x_221_; 
v___x_221_ = l_Lean_Name_cmp(v_optionName_217_, v_optionName_219_);
return v___x_221_;
}
else
{
return v___x_220_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___lam__0___boxed(lean_object* v_a_222_, lean_object* v_b_223_){
_start:
{
uint8_t v_res_224_; lean_object* v_r_225_; 
v_res_224_ = lp_mathlib_Mathlib_TacticAnalysis_instOrdEntry___lam__0(v_a_222_, v_b_223_);
lean_dec_ref(v_b_223_);
lean_dec_ref(v_a_222_);
v_r_225_ = lean_box(v_res_224_);
return v_r_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object* v_x_228_){
_start:
{
lean_object* v_fst_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v_fst_229_ = lean_ctor_get(v_x_228_, 0);
lean_inc(v_fst_229_);
lean_dec_ref(v_x_228_);
v___x_230_ = l_List_reverse___redArg(v_fst_229_);
v___x_231_ = lean_array_mk(v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object* v_x_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_box(0);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object* v_x_234_){
_start:
{
lean_object* v_res_235_; 
v_res_235_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(v_x_234_);
lean_dec_ref(v_x_234_);
return v_res_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object* v_x_236_, lean_object* v_s_237_){
_start:
{
lean_object* v_fst_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
v_fst_238_ = lean_ctor_get(v_s_237_, 0);
lean_inc(v_fst_238_);
lean_dec_ref(v_s_237_);
v___x_239_ = l_List_reverse___redArg(v_fst_238_);
v___x_240_ = lean_array_mk(v___x_239_);
lean_inc_ref_n(v___x_240_, 2);
v___x_241_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
lean_ctor_set(v___x_241_, 2, v___x_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object* v_x_242_, lean_object* v_s_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__2_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(v_x_242_, v_s_243_);
lean_dec_ref(v_x_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__3_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object* v_x_245_, lean_object* v_x_246_){
_start:
{
lean_object* v_fst_247_; lean_object* v_snd_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_265_; 
v_fst_247_ = lean_ctor_get(v_x_245_, 0);
v_snd_248_ = lean_ctor_get(v_x_245_, 1);
v_isSharedCheck_265_ = !lean_is_exclusive(v_x_245_);
if (v_isSharedCheck_265_ == 0)
{
v___x_250_ = v_x_245_;
v_isShared_251_ = v_isSharedCheck_265_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_snd_248_);
lean_inc(v_fst_247_);
lean_dec(v_x_245_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_265_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v_fst_252_; lean_object* v_snd_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_264_; 
v_fst_252_ = lean_ctor_get(v_x_246_, 0);
v_snd_253_ = lean_ctor_get(v_x_246_, 1);
v_isSharedCheck_264_ = !lean_is_exclusive(v_x_246_);
if (v_isSharedCheck_264_ == 0)
{
v___x_255_ = v_x_246_;
v_isShared_256_ = v_isSharedCheck_264_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_snd_253_);
lean_inc(v_fst_252_);
lean_dec(v_x_246_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_264_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_258_; 
if (v_isShared_251_ == 0)
{
lean_ctor_set_tag(v___x_250_, 1);
lean_ctor_set(v___x_250_, 1, v_fst_247_);
lean_ctor_set(v___x_250_, 0, v_fst_252_);
v___x_258_ = v___x_250_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_fst_252_);
lean_ctor_set(v_reuseFailAlloc_263_, 1, v_fst_247_);
v___x_258_ = v_reuseFailAlloc_263_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
lean_object* v___x_259_; lean_object* v___x_261_; 
v___x_259_ = lean_array_push(v_snd_248_, v_snd_253_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 1, v___x_259_);
lean_ctor_set(v___x_255_, 0, v___x_258_);
v___x_261_ = v___x_255_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v___x_258_);
lean_ctor_set(v_reuseFailAlloc_262_, 1, v___x_259_);
v___x_261_ = v_reuseFailAlloc_262_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
return v___x_261_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1(lean_object* v_as_266_, size_t v_i_267_, size_t v_stop_268_, lean_object* v_b_269_){
_start:
{
uint8_t v___x_270_; 
v___x_270_ = lean_usize_dec_eq(v_i_267_, v_stop_268_);
if (v___x_270_ == 0)
{
lean_object* v___x_271_; lean_object* v___x_272_; size_t v___x_273_; size_t v___x_274_; 
v___x_271_ = lean_array_uget_borrowed(v_as_266_, v_i_267_);
v___x_272_ = l_Array_append___redArg(v_b_269_, v___x_271_);
v___x_273_ = ((size_t)1ULL);
v___x_274_ = lean_usize_add(v_i_267_, v___x_273_);
v_i_267_ = v___x_274_;
v_b_269_ = v___x_272_;
goto _start;
}
else
{
return v_b_269_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1___boxed(lean_object* v_as_276_, lean_object* v_i_277_, lean_object* v_stop_278_, lean_object* v_b_279_){
_start:
{
size_t v_i_boxed_280_; size_t v_stop_boxed_281_; lean_object* v_res_282_; 
v_i_boxed_280_ = lean_unbox_usize(v_i_277_);
lean_dec(v_i_277_);
v_stop_boxed_281_ = lean_unbox_usize(v_stop_278_);
lean_dec(v_stop_278_);
v_res_282_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1(v_as_276_, v_i_boxed_280_, v_stop_boxed_281_, v_b_279_);
lean_dec_ref(v_as_276_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__0(size_t v_sz_283_, size_t v_i_284_, lean_object* v_bs_285_, lean_object* v___y_286_){
_start:
{
uint8_t v___x_288_; 
v___x_288_ = lean_usize_dec_lt(v_i_284_, v_sz_283_);
if (v___x_288_ == 0)
{
lean_object* v___x_289_; 
v___x_289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_289_, 0, v_bs_285_);
return v___x_289_;
}
else
{
lean_object* v_v_290_; lean_object* v___x_291_; 
v_v_290_ = lean_array_uget_borrowed(v_bs_285_, v_i_284_);
lean_inc(v_v_290_);
v___x_291_ = lp_mathlib_Mathlib_TacticAnalysis_Entry_import(v_v_290_, v___y_286_);
if (lean_obj_tag(v___x_291_) == 0)
{
lean_object* v_a_292_; lean_object* v___x_293_; lean_object* v_bs_x27_294_; size_t v___x_295_; size_t v___x_296_; lean_object* v___x_297_; 
v_a_292_ = lean_ctor_get(v___x_291_, 0);
lean_inc(v_a_292_);
lean_dec_ref_known(v___x_291_, 1);
v___x_293_ = lean_unsigned_to_nat(0u);
v_bs_x27_294_ = lean_array_uset(v_bs_285_, v_i_284_, v___x_293_);
v___x_295_ = ((size_t)1ULL);
v___x_296_ = lean_usize_add(v_i_284_, v___x_295_);
v___x_297_ = lean_array_uset(v_bs_x27_294_, v_i_284_, v_a_292_);
v_i_284_ = v___x_296_;
v_bs_285_ = v___x_297_;
goto _start;
}
else
{
lean_object* v_a_299_; lean_object* v___x_301_; uint8_t v_isShared_302_; uint8_t v_isSharedCheck_306_; 
lean_dec_ref(v_bs_285_);
v_a_299_ = lean_ctor_get(v___x_291_, 0);
v_isSharedCheck_306_ = !lean_is_exclusive(v___x_291_);
if (v_isSharedCheck_306_ == 0)
{
v___x_301_ = v___x_291_;
v_isShared_302_ = v_isSharedCheck_306_;
goto v_resetjp_300_;
}
else
{
lean_inc(v_a_299_);
lean_dec(v___x_291_);
v___x_301_ = lean_box(0);
v_isShared_302_ = v_isSharedCheck_306_;
goto v_resetjp_300_;
}
v_resetjp_300_:
{
lean_object* v___x_304_; 
if (v_isShared_302_ == 0)
{
v___x_304_ = v___x_301_;
goto v_reusejp_303_;
}
else
{
lean_object* v_reuseFailAlloc_305_; 
v_reuseFailAlloc_305_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_305_, 0, v_a_299_);
v___x_304_ = v_reuseFailAlloc_305_;
goto v_reusejp_303_;
}
v_reusejp_303_:
{
return v___x_304_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__0___boxed(lean_object* v_sz_307_, lean_object* v_i_308_, lean_object* v_bs_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
size_t v_sz_boxed_312_; size_t v_i_boxed_313_; lean_object* v_res_314_; 
v_sz_boxed_312_ = lean_unbox_usize(v_sz_307_);
lean_dec(v_sz_307_);
v_i_boxed_313_ = lean_unbox_usize(v_i_308_);
lean_dec(v_i_308_);
v_res_314_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__0(v_sz_boxed_312_, v_i_boxed_313_, v_bs_309_, v___y_310_);
lean_dec_ref(v___y_310_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object* v_localEntries_317_, lean_object* v_s_318_, lean_object* v___y_319_){
_start:
{
lean_object* v___y_322_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; uint8_t v___x_346_; 
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_));
v___x_345_ = lean_array_get_size(v_s_318_);
v___x_346_ = lean_nat_dec_lt(v___x_343_, v___x_345_);
if (v___x_346_ == 0)
{
v___y_322_ = v___x_344_;
goto v___jp_321_;
}
else
{
uint8_t v___x_347_; 
v___x_347_ = lean_nat_dec_le(v___x_345_, v___x_345_);
if (v___x_347_ == 0)
{
if (v___x_346_ == 0)
{
v___y_322_ = v___x_344_;
goto v___jp_321_;
}
else
{
size_t v___x_348_; size_t v___x_349_; lean_object* v___x_350_; 
v___x_348_ = ((size_t)0ULL);
v___x_349_ = lean_usize_of_nat(v___x_345_);
v___x_350_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1(v_s_318_, v___x_348_, v___x_349_, v___x_344_);
v___y_322_ = v___x_350_;
goto v___jp_321_;
}
}
else
{
size_t v___x_351_; size_t v___x_352_; lean_object* v___x_353_; 
v___x_351_ = ((size_t)0ULL);
v___x_352_ = lean_usize_of_nat(v___x_345_);
v___x_353_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__1(v_s_318_, v___x_351_, v___x_352_, v___x_344_);
v___y_322_ = v___x_353_;
goto v___jp_321_;
}
}
v___jp_321_:
{
size_t v_sz_323_; size_t v___x_324_; lean_object* v___x_325_; 
v_sz_323_ = lean_array_size(v___y_322_);
v___x_324_ = ((size_t)0ULL);
v___x_325_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2__spec__0(v_sz_323_, v___x_324_, v___y_322_, v___y_319_);
if (lean_obj_tag(v___x_325_) == 0)
{
lean_object* v_a_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_334_; 
v_a_326_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_334_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_334_ == 0)
{
v___x_328_ = v___x_325_;
v_isShared_329_ = v_isSharedCheck_334_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_a_326_);
lean_dec(v___x_325_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_334_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_330_; lean_object* v___x_332_; 
v___x_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_330_, 0, v_localEntries_317_);
lean_ctor_set(v___x_330_, 1, v_a_326_);
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 0, v___x_330_);
v___x_332_ = v___x_328_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v___x_330_);
v___x_332_ = v_reuseFailAlloc_333_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
return v___x_332_;
}
}
}
else
{
lean_object* v_a_335_; lean_object* v___x_337_; uint8_t v_isShared_338_; uint8_t v_isSharedCheck_342_; 
lean_dec(v_localEntries_317_);
v_a_335_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_342_ == 0)
{
v___x_337_ = v___x_325_;
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
else
{
lean_inc(v_a_335_);
lean_dec(v___x_325_);
v___x_337_ = lean_box(0);
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
v_resetjp_336_:
{
lean_object* v___x_340_; 
if (v_isShared_338_ == 0)
{
v___x_340_ = v___x_337_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_a_335_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object* v_localEntries_354_, lean_object* v_s_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__4_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(v_localEntries_354_, v_s_355_, v___y_356_);
lean_dec_ref(v___y_356_);
lean_dec_ref(v_s_355_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(lean_object* v___x_359_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_361_, 0, v___x_359_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object* v___x_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__5_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(v___x_362_);
return v_res_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_396_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__11_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_));
v___x_397_ = l_Lean_registerPersistentEnvExtensionUnsafe___redArg(v___x_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2____boxed(lean_object* v_a_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_();
return v_res_399_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_400_ = lean_box(0);
v___x_401_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_402_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
lean_ctor_set(v___x_402_, 1, v___x_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg(){
_start:
{
lean_object* v___x_404_; lean_object* v___x_405_; 
v___x_404_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___closed__0);
v___x_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_405_, 0, v___x_404_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v___y_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg();
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg();
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1(v_00_u03b1_413_, v___y_414_, v___y_415_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
return v_res_417_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_418_; 
v___x_418_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_418_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_419_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__0);
v___x_420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
return v___x_420_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__1);
v___x_422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v___x_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg(lean_object* v_env_423_, lean_object* v___y_424_){
_start:
{
lean_object* v___x_426_; lean_object* v_nextMacroScope_427_; lean_object* v_ngen_428_; lean_object* v_auxDeclNGen_429_; lean_object* v_traceState_430_; lean_object* v_messages_431_; lean_object* v_infoState_432_; lean_object* v_snapshotTasks_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_444_; 
v___x_426_ = lean_st_ref_take(v___y_424_);
v_nextMacroScope_427_ = lean_ctor_get(v___x_426_, 1);
v_ngen_428_ = lean_ctor_get(v___x_426_, 2);
v_auxDeclNGen_429_ = lean_ctor_get(v___x_426_, 3);
v_traceState_430_ = lean_ctor_get(v___x_426_, 4);
v_messages_431_ = lean_ctor_get(v___x_426_, 6);
v_infoState_432_ = lean_ctor_get(v___x_426_, 7);
v_snapshotTasks_433_ = lean_ctor_get(v___x_426_, 8);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_426_);
if (v_isSharedCheck_444_ == 0)
{
lean_object* v_unused_445_; lean_object* v_unused_446_; 
v_unused_445_ = lean_ctor_get(v___x_426_, 5);
lean_dec(v_unused_445_);
v_unused_446_ = lean_ctor_get(v___x_426_, 0);
lean_dec(v_unused_446_);
v___x_435_ = v___x_426_;
v_isShared_436_ = v_isSharedCheck_444_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_snapshotTasks_433_);
lean_inc(v_infoState_432_);
lean_inc(v_messages_431_);
lean_inc(v_traceState_430_);
lean_inc(v_auxDeclNGen_429_);
lean_inc(v_ngen_428_);
lean_inc(v_nextMacroScope_427_);
lean_dec(v___x_426_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_444_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v___x_437_; lean_object* v___x_439_; 
v___x_437_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___closed__2);
if (v_isShared_436_ == 0)
{
lean_ctor_set(v___x_435_, 5, v___x_437_);
lean_ctor_set(v___x_435_, 0, v_env_423_);
v___x_439_ = v___x_435_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_env_423_);
lean_ctor_set(v_reuseFailAlloc_443_, 1, v_nextMacroScope_427_);
lean_ctor_set(v_reuseFailAlloc_443_, 2, v_ngen_428_);
lean_ctor_set(v_reuseFailAlloc_443_, 3, v_auxDeclNGen_429_);
lean_ctor_set(v_reuseFailAlloc_443_, 4, v_traceState_430_);
lean_ctor_set(v_reuseFailAlloc_443_, 5, v___x_437_);
lean_ctor_set(v_reuseFailAlloc_443_, 6, v_messages_431_);
lean_ctor_set(v_reuseFailAlloc_443_, 7, v_infoState_432_);
lean_ctor_set(v_reuseFailAlloc_443_, 8, v_snapshotTasks_433_);
v___x_439_ = v_reuseFailAlloc_443_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_440_ = lean_st_ref_set(v___y_424_, v___x_439_);
v___x_441_ = lean_box(0);
v___x_442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
return v___x_442_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v_env_447_, lean_object* v___y_448_, lean_object* v___y_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg(v_env_447_, v___y_448_);
lean_dec(v___y_448_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2(lean_object* v_env_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v___x_455_; 
v___x_455_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg(v_env_451_, v___y_453_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___boxed(lean_object* v_env_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2(v_env_456_, v___y_457_, v___y_458_);
lean_dec(v___y_458_);
lean_dec_ref(v___y_457_);
return v_res_460_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_461_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_462_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__0);
v___x_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_463_, 0, v___x_462_);
return v___x_463_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2(void){
_start:
{
lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_464_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1);
v___x_465_ = lean_unsigned_to_nat(0u);
v___x_466_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_466_, 0, v___x_465_);
lean_ctor_set(v___x_466_, 1, v___x_465_);
lean_ctor_set(v___x_466_, 2, v___x_465_);
lean_ctor_set(v___x_466_, 3, v___x_465_);
lean_ctor_set(v___x_466_, 4, v___x_464_);
lean_ctor_set(v___x_466_, 5, v___x_464_);
lean_ctor_set(v___x_466_, 6, v___x_464_);
lean_ctor_set(v___x_466_, 7, v___x_464_);
lean_ctor_set(v___x_466_, 8, v___x_464_);
lean_ctor_set(v___x_466_, 9, v___x_464_);
return v___x_466_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_467_ = lean_unsigned_to_nat(32u);
v___x_468_ = lean_mk_empty_array_with_capacity(v___x_467_);
v___x_469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_469_, 0, v___x_468_);
return v___x_469_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__4(void){
_start:
{
size_t v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_470_ = ((size_t)5ULL);
v___x_471_ = lean_unsigned_to_nat(0u);
v___x_472_ = lean_unsigned_to_nat(32u);
v___x_473_ = lean_mk_empty_array_with_capacity(v___x_472_);
v___x_474_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__3);
v___x_475_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_475_, 0, v___x_474_);
lean_ctor_set(v___x_475_, 1, v___x_473_);
lean_ctor_set(v___x_475_, 2, v___x_471_);
lean_ctor_set(v___x_475_, 3, v___x_471_);
lean_ctor_set_usize(v___x_475_, 4, v___x_470_);
return v___x_475_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_476_ = lean_box(1);
v___x_477_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__4);
v___x_478_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__1);
v___x_479_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_479_, 0, v___x_478_);
lean_ctor_set(v___x_479_, 1, v___x_477_);
lean_ctor_set(v___x_479_, 2, v___x_476_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_msgData_480_, lean_object* v___y_481_, lean_object* v___y_482_){
_start:
{
lean_object* v___x_484_; lean_object* v_env_485_; lean_object* v_options_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_484_ = lean_st_ref_get(v___y_482_);
v_env_485_ = lean_ctor_get(v___x_484_, 0);
lean_inc_ref(v_env_485_);
lean_dec(v___x_484_);
v_options_486_ = lean_ctor_get(v___y_481_, 2);
v___x_487_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2);
v___x_488_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5);
lean_inc_ref(v_options_486_);
v___x_489_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_489_, 0, v_env_485_);
lean_ctor_set(v___x_489_, 1, v___x_487_);
lean_ctor_set(v___x_489_, 2, v___x_488_);
lean_ctor_set(v___x_489_, 3, v_options_486_);
v___x_490_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
lean_ctor_set(v___x_490_, 1, v_msgData_480_);
v___x_491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_491_, 0, v___x_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_msgData_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0(v_msgData_492_, v___y_493_, v___y_494_);
lean_dec(v___y_494_);
lean_dec_ref(v___y_493_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(lean_object* v_msg_497_, lean_object* v___y_498_, lean_object* v___y_499_){
_start:
{
lean_object* v_ref_501_; lean_object* v___x_502_; lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_511_; 
v_ref_501_ = lean_ctor_get(v___y_498_, 5);
v___x_502_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0(v_msg_497_, v___y_498_, v___y_499_);
v_a_503_ = lean_ctor_get(v___x_502_, 0);
v_isSharedCheck_511_ = !lean_is_exclusive(v___x_502_);
if (v_isSharedCheck_511_ == 0)
{
v___x_505_ = v___x_502_;
v_isShared_506_ = v_isSharedCheck_511_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_dec(v___x_502_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_511_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_507_; lean_object* v___x_509_; 
lean_inc(v_ref_501_);
v___x_507_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_507_, 0, v_ref_501_);
lean_ctor_set(v___x_507_, 1, v_a_503_);
if (v_isShared_506_ == 0)
{
lean_ctor_set_tag(v___x_505_, 1);
lean_ctor_set(v___x_505_, 0, v___x_507_);
v___x_509_ = v___x_505_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v___x_507_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_msg_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(v_msg_512_, v___y_513_, v___y_514_);
lean_dec(v___y_514_);
lean_dec_ref(v___y_513_);
return v_res_516_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_527_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_528_ = l_Lean_stringToMessageData(v___x_527_);
return v___x_528_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_530_; lean_object* v___x_531_; 
v___x_530_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_531_ = l_Lean_stringToMessageData(v___x_530_);
return v___x_531_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_533_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_534_ = l_Lean_stringToMessageData(v___x_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(lean_object* v___x_535_, lean_object* v___x_536_, lean_object* v___x_537_, lean_object* v_declName_538_, lean_object* v_stx_539_, uint8_t v_kind_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
lean_object* v___x_544_; uint8_t v___x_545_; 
v___x_544_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
lean_inc(v_stx_539_);
v___x_545_ = l_Lean_Syntax_isOfKind(v_stx_539_, v___x_544_);
if (v___x_545_ == 0)
{
lean_object* v___x_546_; 
lean_dec(v_stx_539_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_546_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg();
return v___x_546_;
}
else
{
lean_object* v___x_547_; uint8_t v___x_548_; 
v___x_547_ = l_Lean_Syntax_getArg(v_stx_539_, v___x_535_);
v___x_548_ = l_Lean_Syntax_matchesIdent(v___x_547_, v___x_536_);
lean_dec(v___x_547_);
if (v___x_548_ == 0)
{
lean_object* v___x_549_; 
lean_dec(v_stx_539_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_549_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg();
return v___x_549_;
}
else
{
lean_object* v___x_550_; lean_object* v___x_551_; uint8_t v___x_552_; 
v___x_550_ = lean_unsigned_to_nat(1u);
v___x_551_ = l_Lean_Syntax_getArg(v_stx_539_, v___x_550_);
lean_dec(v_stx_539_);
lean_inc(v___x_551_);
v___x_552_ = l_Lean_Syntax_matchesNull(v___x_551_, v___x_535_);
if (v___x_552_ == 0)
{
uint8_t v___x_553_; 
lean_inc(v___x_551_);
v___x_553_ = l_Lean_Syntax_matchesNull(v___x_551_, v___x_550_);
if (v___x_553_ == 0)
{
lean_object* v___x_554_; 
lean_dec(v___x_551_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_554_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__1___redArg();
return v___x_554_;
}
else
{
lean_object* v___x_555_; lean_object* v___y_557_; lean_object* v___y_558_; lean_object* v___y_559_; uint8_t v___y_560_; lean_object* v___y_591_; lean_object* v___y_592_; lean_object* v___y_593_; lean_object* v___y_596_; lean_object* v___y_597_; lean_object* v___y_598_; uint8_t v___y_599_; lean_object* v___y_603_; lean_object* v___y_604_; uint8_t v___x_608_; uint8_t v___x_609_; 
v___x_555_ = l_Lean_Syntax_getArg(v___x_551_, v___x_535_);
lean_dec(v___x_551_);
v___x_608_ = 0;
v___x_609_ = l_Lean_instBEqAttributeKind_beq(v_kind_540_, v___x_608_);
if (v___x_609_ == 0)
{
lean_object* v___x_610_; lean_object* v___x_611_; 
lean_dec(v___x_555_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_610_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_611_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(v___x_610_, v___y_541_, v___y_542_);
return v___x_611_;
}
else
{
v___y_603_ = v___y_541_;
v___y_604_ = v___y_542_;
goto v___jp_602_;
}
v___jp_556_:
{
if (v___y_560_ == 0)
{
lean_object* v___x_561_; lean_object* v_env_562_; lean_object* v_options_563_; lean_object* v_ref_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
v___x_561_ = lean_st_ref_get(v___y_557_);
v_env_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc_ref(v_env_562_);
lean_dec(v___x_561_);
v_options_563_ = lean_ctor_get(v___y_558_, 2);
v_ref_564_ = lean_ctor_get(v___y_558_, 5);
v___x_565_ = l_Lean_Syntax_getId(v___x_555_);
lean_dec(v___x_555_);
v___x_566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_566_, 0, v_declName_538_);
lean_ctor_set(v___x_566_, 1, v___x_565_);
lean_inc_ref(v_options_563_);
v___x_567_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_567_, 0, v_env_562_);
lean_ctor_set(v___x_567_, 1, v_options_563_);
lean_inc_ref(v___x_566_);
v___x_568_ = lp_mathlib_Mathlib_TacticAnalysis_Entry_import(v___x_566_, v___x_567_);
lean_dec_ref_known(v___x_567_, 2);
if (lean_obj_tag(v___x_568_) == 0)
{
lean_object* v_a_569_; lean_object* v___x_570_; lean_object* v_toEnvExtension_571_; lean_object* v_asyncMode_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; 
v_a_569_ = lean_ctor_get(v___x_568_, 0);
lean_inc(v_a_569_);
lean_dec_ref_known(v___x_568_, 1);
v___x_570_ = lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysisExt;
v_toEnvExtension_571_ = lean_ctor_get(v___x_570_, 0);
v_asyncMode_572_ = lean_ctor_get(v_toEnvExtension_571_, 2);
v___x_573_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_566_);
lean_ctor_set(v___x_573_, 1, v_a_569_);
v___x_574_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_570_, v___y_559_, v___x_573_, v_asyncMode_572_, v___x_537_);
v___x_575_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__2___redArg(v___x_574_, v___y_557_);
return v___x_575_;
}
else
{
lean_object* v_a_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_587_; 
lean_dec_ref_known(v___x_566_, 2);
lean_dec_ref(v___y_559_);
lean_dec(v___x_537_);
v_a_576_ = lean_ctor_get(v___x_568_, 0);
v_isSharedCheck_587_ = !lean_is_exclusive(v___x_568_);
if (v_isSharedCheck_587_ == 0)
{
v___x_578_ = v___x_568_;
v_isShared_579_ = v_isSharedCheck_587_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_a_576_);
lean_dec(v___x_568_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_587_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_585_; 
v___x_580_ = lean_io_error_to_string(v_a_576_);
v___x_581_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_581_, 0, v___x_580_);
v___x_582_ = l_Lean_MessageData_ofFormat(v___x_581_);
lean_inc(v_ref_564_);
v___x_583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_583_, 0, v_ref_564_);
lean_ctor_set(v___x_583_, 1, v___x_582_);
if (v_isShared_579_ == 0)
{
lean_ctor_set(v___x_578_, 0, v___x_583_);
v___x_585_ = v___x_578_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_586_; 
v_reuseFailAlloc_586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_586_, 0, v___x_583_);
v___x_585_ = v_reuseFailAlloc_586_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
return v___x_585_;
}
}
}
}
else
{
lean_object* v___x_588_; lean_object* v___x_589_; 
lean_dec_ref(v___y_559_);
lean_dec(v___x_555_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_588_ = lean_box(0);
v___x_589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_589_, 0, v___x_588_);
return v___x_589_;
}
}
v___jp_590_:
{
lean_object* v___x_594_; 
lean_inc(v_declName_538_);
lean_inc_ref(v___y_591_);
v___x_594_ = lean_decl_get_sorry_dep(v___y_591_, v_declName_538_);
if (lean_obj_tag(v___x_594_) == 0)
{
v___y_557_ = v___y_593_;
v___y_558_ = v___y_592_;
v___y_559_ = v___y_591_;
v___y_560_ = v___x_552_;
goto v___jp_556_;
}
else
{
lean_dec_ref_known(v___x_594_, 1);
v___y_557_ = v___y_593_;
v___y_558_ = v___y_592_;
v___y_559_ = v___y_591_;
v___y_560_ = v___x_553_;
goto v___jp_556_;
}
}
v___jp_595_:
{
if (v___y_599_ == 0)
{
lean_object* v___x_600_; lean_object* v___x_601_; 
lean_dec_ref(v___y_598_);
lean_dec(v___x_555_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_600_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_601_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(v___x_600_, v___y_596_, v___y_597_);
return v___x_601_;
}
else
{
v___y_591_ = v___y_598_;
v___y_592_ = v___y_596_;
v___y_593_ = v___y_597_;
goto v___jp_590_;
}
}
v___jp_602_:
{
lean_object* v___x_605_; lean_object* v_env_606_; lean_object* v___x_607_; 
v___x_605_ = lean_st_ref_get(v___y_604_);
v_env_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc_ref(v_env_606_);
lean_dec(v___x_605_);
v___x_607_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_606_, v_declName_538_);
if (lean_obj_tag(v___x_607_) == 0)
{
v___y_596_ = v___y_603_;
v___y_597_ = v___y_604_;
v___y_598_ = v_env_606_;
v___y_599_ = v___x_553_;
goto v___jp_595_;
}
else
{
lean_dec_ref_known(v___x_607_, 1);
v___y_596_ = v___y_603_;
v___y_597_ = v___y_604_;
v___y_598_ = v_env_606_;
v___y_599_ = v___x_552_;
goto v___jp_595_;
}
}
}
}
else
{
lean_object* v___x_612_; lean_object* v___x_613_; 
lean_dec(v___x_551_);
lean_dec(v_declName_538_);
lean_dec(v___x_537_);
v___x_612_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_613_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(v___x_612_, v___y_541_, v___y_542_);
return v___x_613_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed(lean_object* v___x_614_, lean_object* v___x_615_, lean_object* v___x_616_, lean_object* v_declName_617_, lean_object* v_stx_618_, lean_object* v_kind_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
uint8_t v_kind_boxed_623_; lean_object* v_res_624_; 
v_kind_boxed_623_ = lean_unbox(v_kind_619_);
v_res_624_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(v___x_614_, v___x_615_, v___x_616_, v_declName_617_, v_stx_618_, v_kind_boxed_623_, v___y_620_, v___y_621_);
lean_dec(v___y_621_);
lean_dec_ref(v___y_620_);
lean_dec(v___x_615_);
lean_dec(v___x_614_);
return v_res_624_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_626_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_627_ = l_Lean_stringToMessageData(v___x_626_);
return v___x_627_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_629_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_630_ = l_Lean_stringToMessageData(v___x_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(lean_object* v___x_631_, lean_object* v_decl_632_, lean_object* v___y_633_, lean_object* v___y_634_){
_start:
{
lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_636_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_637_ = l_Lean_MessageData_ofName(v___x_631_);
v___x_638_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_636_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
v___x_639_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_640_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_640_, 0, v___x_638_);
lean_ctor_set(v___x_640_, 1, v___x_639_);
v___x_641_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(v___x_640_, v___y_633_, v___y_634_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed(lean_object* v___x_642_, lean_object* v_decl_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_){
_start:
{
lean_object* v_res_647_; 
v_res_647_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___lam__1_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(v___x_642_, v_decl_643_, v___y_644_, v___y_645_);
lean_dec(v___y_645_);
lean_dec_ref(v___y_644_);
lean_dec(v_decl_643_);
return v_res_647_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__16_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v___x_688_ = lean_unsigned_to_nat(2535838111u);
v___x_689_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__15_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_690_ = l_Lean_Name_num___override(v___x_689_, v___x_688_);
return v___x_690_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__18_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_692_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__17_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_693_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__16_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__16_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__16_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_694_ = l_Lean_Name_str___override(v___x_693_, v___x_692_);
return v___x_694_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__20_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v___x_696_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__19_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_697_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__18_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__18_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__18_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_698_ = l_Lean_Name_str___override(v___x_697_, v___x_696_);
return v___x_698_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__21_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; 
v___x_699_ = lean_unsigned_to_nat(2u);
v___x_700_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__20_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__20_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__20_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_701_ = l_Lean_Name_num___override(v___x_700_, v___x_699_);
return v___x_701_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__26_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_711_ = 1;
v___x_712_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__25_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_713_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__22_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_714_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__21_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__21_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__21_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_715_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_715_, 0, v___x_714_);
lean_ctor_set(v___x_715_, 1, v___x_713_);
lean_ctor_set(v___x_715_, 2, v___x_712_);
lean_ctor_set_uint8(v___x_715_, sizeof(void*)*3, v___x_711_);
return v___x_715_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__27_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_716_; lean_object* v___f_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___f_716_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__24_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___f_717_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__23_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_));
v___x_718_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__26_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__26_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__26_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_719_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
lean_ctor_set(v___x_719_, 1, v___f_717_);
lean_ctor_set(v___x_719_, 2, v___f_716_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_721_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__27_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__27_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__27_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_);
v___x_722_ = l_Lean_registerBuiltinAttribute(v___x_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2____boxed(lean_object* v_a_723_){
_start:
{
lean_object* v_res_724_; 
v_res_724_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_();
return v_res_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_725_, lean_object* v_msg_726_, lean_object* v___y_727_, lean_object* v___y_728_){
_start:
{
lean_object* v___x_730_; 
v___x_730_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___redArg(v_msg_726_, v___y_727_, v___y_728_);
return v___x_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_731_, lean_object* v_msg_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_){
_start:
{
lean_object* v_res_736_; 
v_res_736_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0(v_00_u03b1_731_, v_msg_732_, v___y_733_, v___y_734_);
lean_dec(v___y_734_);
lean_dec_ref(v___y_733_);
return v_res_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__0(lean_object* v_x_737_, lean_object* v_x_738_, lean_object* v_x_739_, lean_object* v___y_740_, lean_object* v___y_741_){
_start:
{
uint8_t v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v___x_743_ = 1;
v___x_744_ = lean_box(v___x_743_);
v___x_745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__0___boxed(lean_object* v_x_746_, lean_object* v_x_747_, lean_object* v_x_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_){
_start:
{
lean_object* v_res_752_; 
v_res_752_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__0(v_x_746_, v_x_747_, v_x_748_, v___y_749_, v___y_750_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec_ref(v_x_748_);
lean_dec_ref(v_x_747_);
lean_dec_ref(v_x_746_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__0(size_t v_sz_753_, size_t v_i_754_, lean_object* v_bs_755_){
_start:
{
uint8_t v___x_756_; 
v___x_756_ = lean_usize_dec_lt(v_i_754_, v_sz_753_);
if (v___x_756_ == 0)
{
return v_bs_755_;
}
else
{
lean_object* v_v_757_; lean_object* v_ctxI_758_; lean_object* v_tacI_759_; lean_object* v___x_761_; uint8_t v_isShared_762_; uint8_t v_isSharedCheck_772_; 
v_v_757_ = lean_array_uget(v_bs_755_, v_i_754_);
v_ctxI_758_ = lean_ctor_get(v_v_757_, 0);
v_tacI_759_ = lean_ctor_get(v_v_757_, 1);
v_isSharedCheck_772_ = !lean_is_exclusive(v_v_757_);
if (v_isSharedCheck_772_ == 0)
{
v___x_761_ = v_v_757_;
v_isShared_762_ = v_isSharedCheck_772_;
goto v_resetjp_760_;
}
else
{
lean_inc(v_tacI_759_);
lean_inc(v_ctxI_758_);
lean_dec(v_v_757_);
v___x_761_ = lean_box(0);
v_isShared_762_ = v_isSharedCheck_772_;
goto v_resetjp_760_;
}
v_resetjp_760_:
{
lean_object* v___x_763_; lean_object* v_bs_x27_764_; lean_object* v___x_766_; 
v___x_763_ = lean_unsigned_to_nat(0u);
v_bs_x27_764_ = lean_array_uset(v_bs_755_, v_i_754_, v___x_763_);
if (v_isShared_762_ == 0)
{
v___x_766_ = v___x_761_;
goto v_reusejp_765_;
}
else
{
lean_object* v_reuseFailAlloc_771_; 
v_reuseFailAlloc_771_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_771_, 0, v_ctxI_758_);
lean_ctor_set(v_reuseFailAlloc_771_, 1, v_tacI_759_);
v___x_766_ = v_reuseFailAlloc_771_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
size_t v___x_767_; size_t v___x_768_; lean_object* v___x_769_; 
lean_ctor_set_uint8(v___x_766_, sizeof(void*)*2, v___x_756_);
v___x_767_ = ((size_t)1ULL);
v___x_768_ = lean_usize_add(v_i_754_, v___x_767_);
v___x_769_ = lean_array_uset(v_bs_x27_764_, v_i_754_, v___x_766_);
v_i_754_ = v___x_768_;
v_bs_755_ = v___x_769_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__0___boxed(lean_object* v_sz_773_, lean_object* v_i_774_, lean_object* v_bs_775_){
_start:
{
size_t v_sz_boxed_776_; size_t v_i_boxed_777_; lean_object* v_res_778_; 
v_sz_boxed_776_ = lean_unbox_usize(v_sz_773_);
lean_dec(v_sz_773_);
v_i_boxed_777_ = lean_unbox_usize(v_i_774_);
lean_dec(v_i_774_);
v_res_778_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__0(v_sz_boxed_776_, v_i_boxed_777_, v_bs_775_);
return v_res_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__3(size_t v_sz_779_, size_t v_i_780_, lean_object* v_bs_781_){
_start:
{
uint8_t v___x_782_; 
v___x_782_ = lean_usize_dec_lt(v_i_780_, v_sz_779_);
if (v___x_782_ == 0)
{
return v_bs_781_;
}
else
{
lean_object* v_v_783_; lean_object* v___x_784_; lean_object* v_bs_x27_785_; size_t v_sz_786_; size_t v___x_787_; lean_object* v___x_788_; size_t v___x_789_; size_t v___x_790_; lean_object* v___x_791_; 
v_v_783_ = lean_array_uget(v_bs_781_, v_i_780_);
v___x_784_ = lean_unsigned_to_nat(0u);
v_bs_x27_785_ = lean_array_uset(v_bs_781_, v_i_780_, v___x_784_);
v_sz_786_ = lean_array_size(v_v_783_);
v___x_787_ = ((size_t)0ULL);
v___x_788_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__0(v_sz_786_, v___x_787_, v_v_783_);
v___x_789_ = ((size_t)1ULL);
v___x_790_ = lean_usize_add(v_i_780_, v___x_789_);
v___x_791_ = lean_array_uset(v_bs_x27_785_, v_i_780_, v___x_788_);
v_i_780_ = v___x_790_;
v_bs_781_ = v___x_791_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__3___boxed(lean_object* v_sz_793_, lean_object* v_i_794_, lean_object* v_bs_795_){
_start:
{
size_t v_sz_boxed_796_; size_t v_i_boxed_797_; lean_object* v_res_798_; 
v_sz_boxed_796_ = lean_unbox_usize(v_sz_793_);
lean_dec(v_sz_793_);
v_i_boxed_797_ = lean_unbox_usize(v_i_794_);
lean_dec(v_i_794_);
v_res_798_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__3(v_sz_boxed_796_, v_i_boxed_797_, v_bs_795_);
return v_res_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__4(size_t v_sz_799_, size_t v_i_800_, lean_object* v_bs_801_){
_start:
{
uint8_t v___x_802_; 
v___x_802_ = lean_usize_dec_lt(v_i_800_, v_sz_799_);
if (v___x_802_ == 0)
{
return v_bs_801_;
}
else
{
lean_object* v_v_803_; lean_object* v_snd_804_; lean_object* v___x_805_; lean_object* v_bs_x27_806_; size_t v___x_807_; size_t v___x_808_; lean_object* v___x_809_; 
v_v_803_ = lean_array_uget_borrowed(v_bs_801_, v_i_800_);
v_snd_804_ = lean_ctor_get(v_v_803_, 1);
lean_inc(v_snd_804_);
v___x_805_ = lean_unsigned_to_nat(0u);
v_bs_x27_806_ = lean_array_uset(v_bs_801_, v_i_800_, v___x_805_);
v___x_807_ = ((size_t)1ULL);
v___x_808_ = lean_usize_add(v_i_800_, v___x_807_);
v___x_809_ = lean_array_uset(v_bs_x27_806_, v_i_800_, v_snd_804_);
v_i_800_ = v___x_808_;
v_bs_801_ = v___x_809_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__4___boxed(lean_object* v_sz_811_, lean_object* v_i_812_, lean_object* v_bs_813_){
_start:
{
size_t v_sz_boxed_814_; size_t v_i_boxed_815_; lean_object* v_res_816_; 
v_sz_boxed_814_ = lean_unbox_usize(v_sz_811_);
lean_dec(v_sz_811_);
v_i_boxed_815_ = lean_unbox_usize(v_i_812_);
lean_dec(v_i_812_);
v_res_816_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__4(v_sz_boxed_814_, v_i_boxed_815_, v_bs_813_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__1(lean_object* v_a_817_, lean_object* v_a_818_){
_start:
{
if (lean_obj_tag(v_a_817_) == 0)
{
lean_object* v___x_819_; 
v___x_819_ = lean_array_to_list(v_a_818_);
return v___x_819_;
}
else
{
lean_object* v_head_820_; 
v_head_820_ = lean_ctor_get(v_a_817_, 0);
if (lean_obj_tag(v_head_820_) == 0)
{
lean_object* v_tail_821_; 
v_tail_821_ = lean_ctor_get(v_a_817_, 1);
lean_inc(v_tail_821_);
lean_dec_ref_known(v_a_817_, 2);
v_a_817_ = v_tail_821_;
goto _start;
}
else
{
lean_object* v_tail_823_; lean_object* v_val_824_; lean_object* v___x_825_; 
lean_inc_ref(v_head_820_);
v_tail_823_ = lean_ctor_get(v_a_817_, 1);
lean_inc(v_tail_823_);
lean_dec_ref_known(v_a_817_, 2);
v_val_824_ = lean_ctor_get(v_head_820_, 0);
lean_inc(v_val_824_);
lean_dec_ref_known(v_head_820_, 1);
v___x_825_ = lean_array_push(v_a_818_, v_val_824_);
v_a_817_ = v_tail_823_;
v_a_818_ = v___x_825_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2(lean_object* v_as_827_, size_t v_i_828_, size_t v_stop_829_, lean_object* v_b_830_){
_start:
{
lean_object* v___y_832_; uint8_t v___x_836_; 
v___x_836_ = lean_usize_dec_eq(v_i_828_, v_stop_829_);
if (v___x_836_ == 0)
{
lean_object* v___x_837_; lean_object* v_fst_838_; 
v___x_837_ = lean_array_uget_borrowed(v_as_827_, v_i_828_);
v_fst_838_ = lean_ctor_get(v___x_837_, 0);
if (lean_obj_tag(v_fst_838_) == 0)
{
v___y_832_ = v_b_830_;
goto v___jp_831_;
}
else
{
lean_object* v_val_839_; lean_object* v___x_840_; 
v_val_839_ = lean_ctor_get(v_fst_838_, 0);
lean_inc(v_val_839_);
v___x_840_ = lean_array_push(v_b_830_, v_val_839_);
v___y_832_ = v___x_840_;
goto v___jp_831_;
}
}
else
{
return v_b_830_;
}
v___jp_831_:
{
size_t v___x_833_; size_t v___x_834_; 
v___x_833_ = ((size_t)1ULL);
v___x_834_ = lean_usize_add(v_i_828_, v___x_833_);
v_i_828_ = v___x_834_;
v_b_830_ = v___y_832_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2___boxed(lean_object* v_as_841_, lean_object* v_i_842_, lean_object* v_stop_843_, lean_object* v_b_844_){
_start:
{
size_t v_i_boxed_845_; size_t v_stop_boxed_846_; lean_object* v_res_847_; 
v_i_boxed_845_ = lean_unbox_usize(v_i_842_);
lean_dec(v_i_842_);
v_stop_boxed_846_ = lean_unbox_usize(v_stop_843_);
lean_dec(v_stop_843_);
v_res_847_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2(v_as_841_, v_i_boxed_845_, v_stop_boxed_846_, v_b_844_);
lean_dec_ref(v_as_841_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2(lean_object* v_as_850_, lean_object* v_start_851_, lean_object* v_stop_852_){
_start:
{
lean_object* v___x_853_; uint8_t v___x_854_; 
v___x_853_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2___closed__0));
v___x_854_ = lean_nat_dec_lt(v_start_851_, v_stop_852_);
if (v___x_854_ == 0)
{
return v___x_853_;
}
else
{
lean_object* v___x_855_; uint8_t v___x_856_; 
v___x_855_ = lean_array_get_size(v_as_850_);
v___x_856_ = lean_nat_dec_le(v_stop_852_, v___x_855_);
if (v___x_856_ == 0)
{
uint8_t v___x_857_; 
v___x_857_ = lean_nat_dec_lt(v_start_851_, v___x_855_);
if (v___x_857_ == 0)
{
return v___x_853_;
}
else
{
size_t v___x_858_; size_t v___x_859_; lean_object* v___x_860_; 
v___x_858_ = lean_usize_of_nat(v_start_851_);
v___x_859_ = lean_usize_of_nat(v___x_855_);
v___x_860_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2(v_as_850_, v___x_858_, v___x_859_, v___x_853_);
return v___x_860_;
}
}
else
{
size_t v___x_861_; size_t v___x_862_; lean_object* v___x_863_; 
v___x_861_ = lean_usize_of_nat(v_start_851_);
v___x_862_ = lean_usize_of_nat(v_stop_852_);
v___x_863_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2_spec__2(v_as_850_, v___x_861_, v___x_862_, v___x_853_);
return v___x_863_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2___boxed(lean_object* v_as_864_, lean_object* v_start_865_, lean_object* v_stop_866_){
_start:
{
lean_object* v_res_867_; 
v_res_867_ = lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2(v_as_864_, v_start_865_, v_stop_866_);
lean_dec(v_stop_866_);
lean_dec(v_start_865_);
lean_dec_ref(v_as_864_);
return v_res_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5(lean_object* v_as_868_, size_t v_i_869_, size_t v_stop_870_, lean_object* v_b_871_){
_start:
{
uint8_t v___x_872_; 
v___x_872_ = lean_usize_dec_eq(v_i_869_, v_stop_870_);
if (v___x_872_ == 0)
{
lean_object* v___x_873_; lean_object* v___x_874_; size_t v___x_875_; size_t v___x_876_; 
v___x_873_ = lean_array_uget_borrowed(v_as_868_, v_i_869_);
v___x_874_ = l_Array_append___redArg(v_b_871_, v___x_873_);
v___x_875_ = ((size_t)1ULL);
v___x_876_ = lean_usize_add(v_i_869_, v___x_875_);
v_i_869_ = v___x_876_;
v_b_871_ = v___x_874_;
goto _start;
}
else
{
return v_b_871_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5___boxed(lean_object* v_as_878_, lean_object* v_i_879_, lean_object* v_stop_880_, lean_object* v_b_881_){
_start:
{
size_t v_i_boxed_882_; size_t v_stop_boxed_883_; lean_object* v_res_884_; 
v_i_boxed_882_ = lean_unbox_usize(v_i_879_);
lean_dec(v_i_879_);
v_stop_boxed_883_ = lean_unbox_usize(v_stop_880_);
lean_dec(v_stop_880_);
v_res_884_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5(v_as_878_, v_i_boxed_882_, v_stop_boxed_883_, v_b_881_);
lean_dec_ref(v_as_878_);
return v_res_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1(lean_object* v___x_971_, lean_object* v___x_972_, lean_object* v_ctx_973_, lean_object* v_i_974_, lean_object* v___c_975_, lean_object* v_cs_976_, lean_object* v___y_977_, lean_object* v___y_978_){
_start:
{
lean_object* v___y_981_; uint8_t v___y_982_; lean_object* v___y_983_; lean_object* v___y_989_; lean_object* v___y_990_; lean_object* v___y_994_; lean_object* v___y_995_; lean_object* v___y_999_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___y_1009_; size_t v_sz_1047_; size_t v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; uint8_t v___x_1052_; 
v___x_1003_ = lean_mk_empty_array_with_capacity(v___x_971_);
v___x_1004_ = lp_mathlib_List_filterMapTR_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__1(v_cs_976_, v___x_1003_);
v___x_1005_ = lean_array_mk(v___x_1004_);
v___x_1006_ = lean_array_get_size(v___x_1005_);
v___x_1007_ = lp_mathlib_Array_filterMapM___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__2(v___x_1005_, v___x_971_, v___x_1006_);
v_sz_1047_ = lean_array_size(v___x_1005_);
v___x_1048_ = ((size_t)0ULL);
v___x_1049_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__4(v_sz_1047_, v___x_1048_, v___x_1005_);
v___x_1050_ = lean_mk_empty_array_with_capacity(v___x_971_);
v___x_1051_ = lean_array_get_size(v___x_1049_);
v___x_1052_ = lean_nat_dec_lt(v___x_971_, v___x_1051_);
if (v___x_1052_ == 0)
{
lean_dec_ref(v___x_1049_);
v___y_1009_ = v___x_1050_;
goto v___jp_1008_;
}
else
{
uint8_t v___x_1053_; 
v___x_1053_ = lean_nat_dec_le(v___x_1051_, v___x_1051_);
if (v___x_1053_ == 0)
{
if (v___x_1052_ == 0)
{
lean_dec_ref(v___x_1049_);
v___y_1009_ = v___x_1050_;
goto v___jp_1008_;
}
else
{
size_t v___x_1054_; lean_object* v___x_1055_; 
v___x_1054_ = lean_usize_of_nat(v___x_1051_);
v___x_1055_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5(v___x_1049_, v___x_1048_, v___x_1054_, v___x_1050_);
lean_dec_ref(v___x_1049_);
v___y_1009_ = v___x_1055_;
goto v___jp_1008_;
}
}
else
{
size_t v___x_1056_; lean_object* v___x_1057_; 
v___x_1056_ = lean_usize_of_nat(v___x_1051_);
v___x_1057_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__5(v___x_1049_, v___x_1048_, v___x_1056_, v___x_1050_);
lean_dec_ref(v___x_1049_);
v___y_1009_ = v___x_1057_;
goto v___jp_1008_;
}
}
v___jp_980_:
{
lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; 
v___x_984_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_984_, 0, v_ctx_973_);
lean_ctor_set(v___x_984_, 1, v___y_981_);
lean_ctor_set_uint8(v___x_984_, sizeof(void*)*2, v___y_982_);
v___x_985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_985_, 0, v___x_984_);
v___x_986_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_986_, 0, v___x_985_);
lean_ctor_set(v___x_986_, 1, v___y_983_);
v___x_987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_987_, 0, v___x_986_);
return v___x_987_;
}
v___jp_988_:
{
lean_object* v___x_991_; lean_object* v___x_992_; 
v___x_991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_991_, 0, v___y_990_);
lean_ctor_set(v___x_991_, 1, v___y_989_);
v___x_992_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_992_, 0, v___x_991_);
return v___x_992_;
}
v___jp_993_:
{
lean_object* v___x_996_; lean_object* v___x_997_; 
lean_inc(v___y_994_);
v___x_996_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_996_, 0, v___y_994_);
lean_ctor_set(v___x_996_, 1, v___y_995_);
v___x_997_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_997_, 0, v___x_996_);
return v___x_997_;
}
v___jp_998_:
{
lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_1000_ = lean_box(0);
v___x_1001_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1001_, 0, v___x_1000_);
lean_ctor_set(v___x_1001_, 1, v___y_999_);
v___x_1002_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1001_);
return v___x_1002_;
}
v___jp_1008_:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; uint8_t v___x_1013_; 
v___x_1010_ = l_Lean_Elab_Info_stx(v_i_974_);
lean_inc(v___x_1010_);
v___x_1011_ = l_Lean_Syntax_getKind(v___x_1010_);
v___x_1012_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__9));
lean_inc(v___x_1011_);
lean_inc_ref(v___x_972_);
v___x_1013_ = l_List_elem___redArg(v___x_972_, v___x_1011_, v___x_1012_);
if (v___x_1013_ == 0)
{
lean_object* v___x_1014_; 
v___x_1014_ = l_Lean_Syntax_getHeadInfo_x3f(v___x_1010_);
lean_dec(v___x_1010_);
if (lean_obj_tag(v___x_1014_) == 1)
{
lean_object* v_val_1015_; lean_object* v___x_1017_; uint8_t v_isShared_1018_; uint8_t v_isSharedCheck_1042_; 
v_val_1015_ = lean_ctor_get(v___x_1014_, 0);
v_isSharedCheck_1042_ = !lean_is_exclusive(v___x_1014_);
if (v_isSharedCheck_1042_ == 0)
{
v___x_1017_ = v___x_1014_;
v_isShared_1018_ = v_isSharedCheck_1042_;
goto v_resetjp_1016_;
}
else
{
lean_inc(v_val_1015_);
lean_dec(v___x_1014_);
v___x_1017_ = lean_box(0);
v_isShared_1018_ = v_isSharedCheck_1042_;
goto v_resetjp_1016_;
}
v_resetjp_1016_:
{
if (lean_obj_tag(v_val_1015_) == 0)
{
lean_object* v___x_1019_; uint8_t v___x_1020_; 
lean_dec_ref_known(v_val_1015_, 4);
v___x_1019_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__24));
lean_inc(v___x_1011_);
lean_inc_ref(v___x_972_);
v___x_1020_ = l_List_elem___redArg(v___x_972_, v___x_1011_, v___x_1019_);
if (v___x_1020_ == 0)
{
lean_object* v___x_1021_; uint8_t v___x_1022_; 
v___x_1021_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__27));
lean_inc(v___x_1011_);
lean_inc_ref(v___x_972_);
v___x_1022_ = l_List_elem___redArg(v___x_972_, v___x_1011_, v___x_1021_);
if (v___x_1022_ == 0)
{
lean_del_object(v___x_1017_);
lean_dec_ref(v___x_1007_);
if (lean_obj_tag(v_i_974_) == 0)
{
lean_object* v_i_1023_; lean_object* v___x_1024_; uint8_t v___x_1025_; 
v_i_1023_ = lean_ctor_get(v_i_974_, 0);
lean_inc_ref(v_i_1023_);
lean_dec_ref_known(v_i_974_, 1);
v___x_1024_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__33));
v___x_1025_ = l_List_elem___redArg(v___x_972_, v___x_1011_, v___x_1024_);
if (v___x_1025_ == 0)
{
v___y_981_ = v_i_1023_;
v___y_982_ = v___x_1022_;
v___y_983_ = v___y_1009_;
goto v___jp_980_;
}
else
{
size_t v_sz_1026_; size_t v___x_1027_; lean_object* v___x_1028_; 
v_sz_1026_ = lean_array_size(v___y_1009_);
v___x_1027_ = ((size_t)0ULL);
v___x_1028_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__3(v_sz_1026_, v___x_1027_, v___y_1009_);
v___y_981_ = v_i_1023_;
v___y_982_ = v___x_1022_;
v___y_983_ = v___x_1028_;
goto v___jp_980_;
}
}
else
{
lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
lean_dec(v___x_1011_);
lean_dec_ref(v_i_974_);
lean_dec_ref(v_ctx_973_);
lean_dec_ref(v___x_972_);
v___x_1029_ = lean_box(0);
v___x_1030_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1029_);
lean_ctor_set(v___x_1030_, 1, v___y_1009_);
v___x_1031_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1031_, 0, v___x_1030_);
return v___x_1031_;
}
}
else
{
lean_object* v___x_1032_; uint8_t v___x_1033_; 
lean_dec(v___x_1011_);
lean_dec_ref(v_i_974_);
lean_dec_ref(v_ctx_973_);
lean_dec_ref(v___x_972_);
v___x_1032_ = lean_array_get_size(v___x_1007_);
v___x_1033_ = lean_nat_dec_lt(v___x_971_, v___x_1032_);
if (v___x_1033_ == 0)
{
lean_object* v___x_1034_; 
lean_del_object(v___x_1017_);
lean_dec_ref(v___x_1007_);
v___x_1034_ = lean_box(0);
v___y_989_ = v___y_1009_;
v___y_990_ = v___x_1034_;
goto v___jp_988_;
}
else
{
lean_object* v___x_1035_; lean_object* v___x_1037_; 
v___x_1035_ = lean_array_fget(v___x_1007_, v___x_971_);
lean_dec_ref(v___x_1007_);
if (v_isShared_1018_ == 0)
{
lean_ctor_set(v___x_1017_, 0, v___x_1035_);
v___x_1037_ = v___x_1017_;
goto v_reusejp_1036_;
}
else
{
lean_object* v_reuseFailAlloc_1038_; 
v_reuseFailAlloc_1038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1038_, 0, v___x_1035_);
v___x_1037_ = v_reuseFailAlloc_1038_;
goto v_reusejp_1036_;
}
v_reusejp_1036_:
{
v___y_989_ = v___y_1009_;
v___y_990_ = v___x_1037_;
goto v___jp_988_;
}
}
}
}
else
{
lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; 
lean_del_object(v___x_1017_);
lean_dec(v___x_1011_);
lean_dec_ref(v___x_1007_);
lean_dec_ref(v_i_974_);
lean_dec_ref(v_ctx_973_);
lean_dec_ref(v___x_972_);
v___x_1039_ = lean_box(0);
v___x_1040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1040_, 0, v___x_1039_);
lean_ctor_set(v___x_1040_, 1, v___y_1009_);
v___x_1041_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
return v___x_1041_;
}
}
else
{
lean_del_object(v___x_1017_);
lean_dec(v_val_1015_);
lean_dec(v___x_1011_);
lean_dec_ref(v___x_1007_);
lean_dec_ref(v_i_974_);
lean_dec_ref(v_ctx_973_);
lean_dec_ref(v___x_972_);
v___y_999_ = v___y_1009_;
goto v___jp_998_;
}
}
}
else
{
lean_dec(v___x_1014_);
lean_dec(v___x_1011_);
lean_dec_ref(v___x_1007_);
lean_dec_ref(v_i_974_);
lean_dec_ref(v_ctx_973_);
lean_dec_ref(v___x_972_);
v___y_999_ = v___y_1009_;
goto v___jp_998_;
}
}
else
{
lean_object* v___x_1043_; lean_object* v___x_1044_; uint8_t v___x_1045_; 
lean_dec(v___x_1011_);
lean_dec(v___x_1010_);
lean_dec_ref(v_i_974_);
lean_dec_ref(v_ctx_973_);
lean_dec_ref(v___x_972_);
v___x_1043_ = lean_box(0);
v___x_1044_ = lean_array_get_size(v___x_1007_);
v___x_1045_ = lean_nat_dec_eq(v___x_1044_, v___x_971_);
if (v___x_1045_ == 0)
{
lean_object* v___x_1046_; 
v___x_1046_ = lean_array_push(v___y_1009_, v___x_1007_);
v___y_994_ = v___x_1043_;
v___y_995_ = v___x_1046_;
goto v___jp_993_;
}
else
{
lean_dec_ref(v___x_1007_);
v___y_994_ = v___x_1043_;
v___y_995_ = v___y_1009_;
goto v___jp_993_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___boxed(lean_object* v___x_1058_, lean_object* v___x_1059_, lean_object* v_ctx_1060_, lean_object* v_i_1061_, lean_object* v___c_1062_, lean_object* v_cs_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_){
_start:
{
lean_object* v_res_1067_; 
v_res_1067_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1(v___x_1058_, v___x_1059_, v_ctx_1060_, v_i_1061_, v___c_1062_, v_cs_1063_, v___y_1064_, v___y_1065_);
lean_dec(v___y_1065_);
lean_dec_ref(v___y_1064_);
lean_dec_ref(v___c_1062_);
lean_dec(v___x_1058_);
return v_res_1067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg(lean_object* v_msg_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_){
_start:
{
lean_object* v___x_1074_; lean_object* v_toApplicative_1075_; lean_object* v_toFunctor_1076_; lean_object* v_toSeq_1077_; lean_object* v_toSeqLeft_1078_; lean_object* v_toSeqRight_1079_; lean_object* v___f_1080_; lean_object* v___f_1081_; lean_object* v___f_1082_; lean_object* v___f_1083_; lean_object* v___x_1084_; lean_object* v___f_1085_; lean_object* v___f_1086_; lean_object* v___f_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_3570__overap_1092_; lean_object* v___x_1093_; 
v___x_1074_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1, &lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1_once, _init_lp_mathlib_Mathlib_TacticAnalysis_TacticNode_runTacticCode___closed__1);
v_toApplicative_1075_ = lean_ctor_get(v___x_1074_, 0);
v_toFunctor_1076_ = lean_ctor_get(v_toApplicative_1075_, 0);
v_toSeq_1077_ = lean_ctor_get(v_toApplicative_1075_, 2);
v_toSeqLeft_1078_ = lean_ctor_get(v_toApplicative_1075_, 3);
v_toSeqRight_1079_ = lean_ctor_get(v_toApplicative_1075_, 4);
v___f_1080_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__0));
v___f_1081_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___closed__1));
lean_inc_ref_n(v_toFunctor_1076_, 2);
v___f_1082_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1082_, 0, v_toFunctor_1076_);
v___f_1083_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1083_, 0, v_toFunctor_1076_);
v___x_1084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1084_, 0, v___f_1082_);
lean_ctor_set(v___x_1084_, 1, v___f_1083_);
lean_inc(v_toSeqRight_1079_);
v___f_1085_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1085_, 0, v_toSeqRight_1079_);
lean_inc(v_toSeqLeft_1078_);
v___f_1086_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1086_, 0, v_toSeqLeft_1078_);
lean_inc(v_toSeq_1077_);
v___f_1087_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1087_, 0, v_toSeq_1077_);
v___x_1088_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1088_, 0, v___x_1084_);
lean_ctor_set(v___x_1088_, 1, v___f_1080_);
lean_ctor_set(v___x_1088_, 2, v___f_1087_);
lean_ctor_set(v___x_1088_, 3, v___f_1086_);
lean_ctor_set(v___x_1088_, 4, v___f_1085_);
v___x_1089_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1089_, 0, v___x_1088_);
lean_ctor_set(v___x_1089_, 1, v___f_1081_);
v___x_1090_ = lean_box(0);
v___x_1091_ = l_instInhabitedOfMonad___redArg(v___x_1089_, v___x_1090_);
v___x_3570__overap_1092_ = lean_panic_fn_borrowed(v___x_1091_, v_msg_1070_);
lean_dec(v___x_1091_);
lean_inc(v___y_1072_);
lean_inc_ref(v___y_1071_);
v___x_1093_ = lean_apply_3(v___x_3570__overap_1092_, v___y_1071_, v___y_1072_, lean_box(0));
return v___x_1093_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg___boxed(lean_object* v_msg_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_){
_start:
{
lean_object* v_res_1098_; 
v_res_1098_ = lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg(v_msg_1094_, v___y_1095_, v___y_1096_);
lean_dec(v___y_1096_);
lean_dec_ref(v___y_1095_);
return v_res_1098_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; 
v___x_1102_ = ((lean_object*)(lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__2));
v___x_1103_ = lean_unsigned_to_nat(21u);
v___x_1104_ = lean_unsigned_to_nat(65u);
v___x_1105_ = ((lean_object*)(lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__1));
v___x_1106_ = ((lean_object*)(lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__0));
v___x_1107_ = l_mkPanicMessageWithDecl(v___x_1106_, v___x_1105_, v___x_1104_, v___x_1103_, v___x_1102_);
return v___x_1107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg(lean_object* v_preNode_1108_, lean_object* v_postNode_1109_, lean_object* v_x_1110_, lean_object* v_x_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
switch(lean_obj_tag(v_x_1111_))
{
case 0:
{
lean_object* v_i_1115_; lean_object* v_t_1116_; lean_object* v___x_1117_; 
v_i_1115_ = lean_ctor_get(v_x_1111_, 0);
lean_inc_ref(v_i_1115_);
v_t_1116_ = lean_ctor_get(v_x_1111_, 1);
lean_inc_ref(v_t_1116_);
lean_dec_ref_known(v_x_1111_, 2);
v___x_1117_ = l_Lean_Elab_PartialContextInfo_mergeIntoOuter_x3f(v_i_1115_, v_x_1110_);
v_x_1110_ = v___x_1117_;
v_x_1111_ = v_t_1116_;
goto _start;
}
case 1:
{
if (lean_obj_tag(v_x_1110_) == 0)
{
lean_object* v___x_1119_; lean_object* v___x_1120_; 
lean_dec_ref_known(v_x_1111_, 2);
lean_dec_ref(v_postNode_1109_);
lean_dec_ref(v_preNode_1108_);
v___x_1119_ = lean_obj_once(&lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__3, &lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__3_once, _init_lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___closed__3);
v___x_1120_ = lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg(v___x_1119_, v___y_1112_, v___y_1113_);
return v___x_1120_;
}
else
{
lean_object* v_i_1121_; lean_object* v_children_1122_; lean_object* v_val_1123_; lean_object* v___x_1124_; 
v_i_1121_ = lean_ctor_get(v_x_1111_, 0);
lean_inc_ref_n(v_i_1121_, 2);
v_children_1122_ = lean_ctor_get(v_x_1111_, 1);
lean_inc_ref_n(v_children_1122_, 2);
lean_dec_ref_known(v_x_1111_, 2);
v_val_1123_ = lean_ctor_get(v_x_1110_, 0);
lean_inc_n(v_val_1123_, 2);
lean_inc_ref(v_preNode_1108_);
lean_inc(v___y_1113_);
lean_inc_ref(v___y_1112_);
v___x_1124_ = lean_apply_6(v_preNode_1108_, v_val_1123_, v_i_1121_, v_children_1122_, v___y_1112_, v___y_1113_, lean_box(0));
if (lean_obj_tag(v___x_1124_) == 0)
{
lean_object* v_a_1125_; uint8_t v___x_1126_; 
v_a_1125_ = lean_ctor_get(v___x_1124_, 0);
lean_inc(v_a_1125_);
lean_dec_ref_known(v___x_1124_, 1);
v___x_1126_ = lean_unbox(v_a_1125_);
lean_dec(v_a_1125_);
if (v___x_1126_ == 0)
{
lean_object* v___x_1128_; uint8_t v_isShared_1129_; uint8_t v_isSharedCheck_1151_; 
lean_dec_ref(v_preNode_1108_);
v_isSharedCheck_1151_ = !lean_is_exclusive(v_x_1110_);
if (v_isSharedCheck_1151_ == 0)
{
lean_object* v_unused_1152_; 
v_unused_1152_ = lean_ctor_get(v_x_1110_, 0);
lean_dec(v_unused_1152_);
v___x_1128_ = v_x_1110_;
v_isShared_1129_ = v_isSharedCheck_1151_;
goto v_resetjp_1127_;
}
else
{
lean_dec(v_x_1110_);
v___x_1128_ = lean_box(0);
v_isShared_1129_ = v_isSharedCheck_1151_;
goto v_resetjp_1127_;
}
v_resetjp_1127_:
{
lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1130_ = lean_box(0);
lean_inc(v___y_1113_);
lean_inc_ref(v___y_1112_);
v___x_1131_ = lean_apply_7(v_postNode_1109_, v_val_1123_, v_i_1121_, v_children_1122_, v___x_1130_, v___y_1112_, v___y_1113_, lean_box(0));
if (lean_obj_tag(v___x_1131_) == 0)
{
lean_object* v_a_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1142_; 
v_a_1132_ = lean_ctor_get(v___x_1131_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1131_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1134_ = v___x_1131_;
v_isShared_1135_ = v_isSharedCheck_1142_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_a_1132_);
lean_dec(v___x_1131_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1142_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1137_; 
if (v_isShared_1129_ == 0)
{
lean_ctor_set(v___x_1128_, 0, v_a_1132_);
v___x_1137_ = v___x_1128_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_a_1132_);
v___x_1137_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
lean_object* v___x_1139_; 
if (v_isShared_1135_ == 0)
{
lean_ctor_set(v___x_1134_, 0, v___x_1137_);
v___x_1139_ = v___x_1134_;
goto v_reusejp_1138_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v___x_1137_);
v___x_1139_ = v_reuseFailAlloc_1140_;
goto v_reusejp_1138_;
}
v_reusejp_1138_:
{
return v___x_1139_;
}
}
}
}
else
{
lean_object* v_a_1143_; lean_object* v___x_1145_; uint8_t v_isShared_1146_; uint8_t v_isSharedCheck_1150_; 
lean_del_object(v___x_1128_);
v_a_1143_ = lean_ctor_get(v___x_1131_, 0);
v_isSharedCheck_1150_ = !lean_is_exclusive(v___x_1131_);
if (v_isSharedCheck_1150_ == 0)
{
v___x_1145_ = v___x_1131_;
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
else
{
lean_inc(v_a_1143_);
lean_dec(v___x_1131_);
v___x_1145_ = lean_box(0);
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
v_resetjp_1144_:
{
lean_object* v___x_1148_; 
if (v_isShared_1146_ == 0)
{
v___x_1148_ = v___x_1145_;
goto v_reusejp_1147_;
}
else
{
lean_object* v_reuseFailAlloc_1149_; 
v_reuseFailAlloc_1149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1149_, 0, v_a_1143_);
v___x_1148_ = v_reuseFailAlloc_1149_;
goto v_reusejp_1147_;
}
v_reusejp_1147_:
{
return v___x_1148_;
}
}
}
}
}
else
{
lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; 
v___x_1153_ = l_Lean_Elab_Info_updateContext_x3f(v_x_1110_, v_i_1121_);
v___x_1154_ = l_Lean_PersistentArray_toList___redArg(v_children_1122_);
v___x_1155_ = lean_box(0);
lean_inc_ref(v_postNode_1109_);
v___x_1156_ = lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg(v_preNode_1108_, v_postNode_1109_, v___x_1153_, v___x_1154_, v___x_1155_, v___y_1112_, v___y_1113_);
if (lean_obj_tag(v___x_1156_) == 0)
{
lean_object* v_a_1157_; lean_object* v___x_1158_; 
v_a_1157_ = lean_ctor_get(v___x_1156_, 0);
lean_inc(v_a_1157_);
lean_dec_ref_known(v___x_1156_, 1);
lean_inc(v___y_1113_);
lean_inc_ref(v___y_1112_);
v___x_1158_ = lean_apply_7(v_postNode_1109_, v_val_1123_, v_i_1121_, v_children_1122_, v_a_1157_, v___y_1112_, v___y_1113_, lean_box(0));
if (lean_obj_tag(v___x_1158_) == 0)
{
lean_object* v_a_1159_; lean_object* v___x_1161_; uint8_t v_isShared_1162_; uint8_t v_isSharedCheck_1167_; 
v_a_1159_ = lean_ctor_get(v___x_1158_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1158_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1161_ = v___x_1158_;
v_isShared_1162_ = v_isSharedCheck_1167_;
goto v_resetjp_1160_;
}
else
{
lean_inc(v_a_1159_);
lean_dec(v___x_1158_);
v___x_1161_ = lean_box(0);
v_isShared_1162_ = v_isSharedCheck_1167_;
goto v_resetjp_1160_;
}
v_resetjp_1160_:
{
lean_object* v___x_1163_; lean_object* v___x_1165_; 
v___x_1163_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1163_, 0, v_a_1159_);
if (v_isShared_1162_ == 0)
{
lean_ctor_set(v___x_1161_, 0, v___x_1163_);
v___x_1165_ = v___x_1161_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v___x_1163_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
}
else
{
lean_object* v_a_1168_; lean_object* v___x_1170_; uint8_t v_isShared_1171_; uint8_t v_isSharedCheck_1175_; 
v_a_1168_ = lean_ctor_get(v___x_1158_, 0);
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1158_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1170_ = v___x_1158_;
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
else
{
lean_inc(v_a_1168_);
lean_dec(v___x_1158_);
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
else
{
lean_object* v_a_1176_; lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1183_; 
lean_dec(v_val_1123_);
lean_dec_ref(v_children_1122_);
lean_dec_ref(v_i_1121_);
lean_dec_ref(v_postNode_1109_);
v_a_1176_ = lean_ctor_get(v___x_1156_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_1156_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1178_ = v___x_1156_;
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
else
{
lean_inc(v_a_1176_);
lean_dec(v___x_1156_);
v___x_1178_ = lean_box(0);
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
v_resetjp_1177_:
{
lean_object* v___x_1181_; 
if (v_isShared_1179_ == 0)
{
v___x_1181_ = v___x_1178_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_a_1176_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
return v___x_1181_;
}
}
}
}
}
else
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1191_; 
lean_dec(v_val_1123_);
lean_dec_ref(v_children_1122_);
lean_dec_ref_known(v_x_1110_, 1);
lean_dec_ref(v_i_1121_);
lean_dec_ref(v_postNode_1109_);
lean_dec_ref(v_preNode_1108_);
v_a_1184_ = lean_ctor_get(v___x_1124_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v___x_1124_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1186_ = v___x_1124_;
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_1124_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1189_; 
if (v_isShared_1187_ == 0)
{
v___x_1189_ = v___x_1186_;
goto v_reusejp_1188_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_a_1184_);
v___x_1189_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1188_;
}
v_reusejp_1188_:
{
return v___x_1189_;
}
}
}
}
}
default: 
{
lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1199_; 
lean_dec(v_x_1110_);
lean_dec_ref(v_postNode_1109_);
lean_dec_ref(v_preNode_1108_);
v_isSharedCheck_1199_ = !lean_is_exclusive(v_x_1111_);
if (v_isSharedCheck_1199_ == 0)
{
lean_object* v_unused_1200_; 
v_unused_1200_ = lean_ctor_get(v_x_1111_, 0);
lean_dec(v_unused_1200_);
v___x_1193_ = v_x_1111_;
v_isShared_1194_ = v_isSharedCheck_1199_;
goto v_resetjp_1192_;
}
else
{
lean_dec(v_x_1111_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1199_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1195_; lean_object* v___x_1197_; 
v___x_1195_ = lean_box(0);
if (v_isShared_1194_ == 0)
{
lean_ctor_set_tag(v___x_1193_, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1195_);
v___x_1197_ = v___x_1193_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v___x_1195_);
v___x_1197_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
return v___x_1197_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg(lean_object* v_preNode_1201_, lean_object* v_postNode_1202_, lean_object* v___x_1203_, lean_object* v_x_1204_, lean_object* v_x_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
if (lean_obj_tag(v_x_1204_) == 0)
{
lean_object* v___x_1209_; lean_object* v___x_1210_; 
lean_dec(v___x_1203_);
lean_dec_ref(v_postNode_1202_);
lean_dec_ref(v_preNode_1201_);
v___x_1209_ = l_List_reverse___redArg(v_x_1205_);
v___x_1210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1210_, 0, v___x_1209_);
return v___x_1210_;
}
else
{
lean_object* v_head_1211_; lean_object* v_tail_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1230_; 
v_head_1211_ = lean_ctor_get(v_x_1204_, 0);
v_tail_1212_ = lean_ctor_get(v_x_1204_, 1);
v_isSharedCheck_1230_ = !lean_is_exclusive(v_x_1204_);
if (v_isSharedCheck_1230_ == 0)
{
v___x_1214_ = v_x_1204_;
v_isShared_1215_ = v_isSharedCheck_1230_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_tail_1212_);
lean_inc(v_head_1211_);
lean_dec(v_x_1204_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1230_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
lean_object* v___x_1216_; 
lean_inc(v___x_1203_);
lean_inc_ref(v_postNode_1202_);
lean_inc_ref(v_preNode_1201_);
v___x_1216_ = lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg(v_preNode_1201_, v_postNode_1202_, v___x_1203_, v_head_1211_, v___y_1206_, v___y_1207_);
if (lean_obj_tag(v___x_1216_) == 0)
{
lean_object* v_a_1217_; lean_object* v___x_1219_; 
v_a_1217_ = lean_ctor_get(v___x_1216_, 0);
lean_inc(v_a_1217_);
lean_dec_ref_known(v___x_1216_, 1);
if (v_isShared_1215_ == 0)
{
lean_ctor_set(v___x_1214_, 1, v_x_1205_);
lean_ctor_set(v___x_1214_, 0, v_a_1217_);
v___x_1219_ = v___x_1214_;
goto v_reusejp_1218_;
}
else
{
lean_object* v_reuseFailAlloc_1221_; 
v_reuseFailAlloc_1221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1221_, 0, v_a_1217_);
lean_ctor_set(v_reuseFailAlloc_1221_, 1, v_x_1205_);
v___x_1219_ = v_reuseFailAlloc_1221_;
goto v_reusejp_1218_;
}
v_reusejp_1218_:
{
v_x_1204_ = v_tail_1212_;
v_x_1205_ = v___x_1219_;
goto _start;
}
}
else
{
lean_object* v_a_1222_; lean_object* v___x_1224_; uint8_t v_isShared_1225_; uint8_t v_isSharedCheck_1229_; 
lean_del_object(v___x_1214_);
lean_dec(v_tail_1212_);
lean_dec(v_x_1205_);
lean_dec(v___x_1203_);
lean_dec_ref(v_postNode_1202_);
lean_dec_ref(v_preNode_1201_);
v_a_1222_ = lean_ctor_get(v___x_1216_, 0);
v_isSharedCheck_1229_ = !lean_is_exclusive(v___x_1216_);
if (v_isSharedCheck_1229_ == 0)
{
v___x_1224_ = v___x_1216_;
v_isShared_1225_ = v_isSharedCheck_1229_;
goto v_resetjp_1223_;
}
else
{
lean_inc(v_a_1222_);
lean_dec(v___x_1216_);
v___x_1224_ = lean_box(0);
v_isShared_1225_ = v_isSharedCheck_1229_;
goto v_resetjp_1223_;
}
v_resetjp_1223_:
{
lean_object* v___x_1227_; 
if (v_isShared_1225_ == 0)
{
v___x_1227_ = v___x_1224_;
goto v_reusejp_1226_;
}
else
{
lean_object* v_reuseFailAlloc_1228_; 
v_reuseFailAlloc_1228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1228_, 0, v_a_1222_);
v___x_1227_ = v_reuseFailAlloc_1228_;
goto v_reusejp_1226_;
}
v_reusejp_1226_:
{
return v___x_1227_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg___boxed(lean_object* v_preNode_1231_, lean_object* v_postNode_1232_, lean_object* v___x_1233_, lean_object* v_x_1234_, lean_object* v_x_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v_res_1239_; 
v_res_1239_ = lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg(v_preNode_1231_, v_postNode_1232_, v___x_1233_, v_x_1234_, v_x_1235_, v___y_1236_, v___y_1237_);
lean_dec(v___y_1237_);
lean_dec_ref(v___y_1236_);
return v_res_1239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg___boxed(lean_object* v_preNode_1240_, lean_object* v_postNode_1241_, lean_object* v_x_1242_, lean_object* v_x_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_){
_start:
{
lean_object* v_res_1247_; 
v_res_1247_ = lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg(v_preNode_1240_, v_postNode_1241_, v_x_1242_, v_x_1243_, v___y_1244_, v___y_1245_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
return v_res_1247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(lean_object* v_tree_1255_, lean_object* v_a_1256_, lean_object* v_a_1257_){
_start:
{
lean_object* v___x_1259_; lean_object* v_env_1260_; lean_object* v_ngen_1261_; lean_object* v_fileMap_1262_; lean_object* v___f_1263_; lean_object* v___x_1264_; lean_object* v___f_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; 
v___x_1259_ = lean_st_ref_get(v_a_1257_);
v_env_1260_ = lean_ctor_get(v___x_1259_, 0);
lean_inc_ref(v_env_1260_);
v_ngen_1261_ = lean_ctor_get(v___x_1259_, 6);
lean_inc_ref(v_ngen_1261_);
lean_dec(v___x_1259_);
v_fileMap_1262_ = lean_ctor_get(v_a_1256_, 1);
v___f_1263_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__0));
v___x_1264_ = lean_box(0);
v___f_1265_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__2));
v___x_1266_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2);
v___x_1267_ = l_Lean_Options_empty;
v___x_1268_ = lean_box(0);
v___x_1269_ = lean_box(0);
lean_inc_ref(v_fileMap_1262_);
v___x_1270_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_1270_, 0, v_env_1260_);
lean_ctor_set(v___x_1270_, 1, v___x_1264_);
lean_ctor_set(v___x_1270_, 2, v_fileMap_1262_);
lean_ctor_set(v___x_1270_, 3, v___x_1266_);
lean_ctor_set(v___x_1270_, 4, v___x_1267_);
lean_ctor_set(v___x_1270_, 5, v___x_1268_);
lean_ctor_set(v___x_1270_, 6, v___x_1269_);
lean_ctor_set(v___x_1270_, 7, v_ngen_1261_);
v___x_1271_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___closed__3));
v___x_1272_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1270_);
lean_ctor_set(v___x_1272_, 1, v___x_1264_);
lean_ctor_set(v___x_1272_, 2, v___x_1271_);
v___x_1273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1273_, 0, v___x_1272_);
v___x_1274_ = lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg(v___f_1263_, v___f_1265_, v___x_1273_, v_tree_1255_, v_a_1256_, v_a_1257_);
if (lean_obj_tag(v___x_1274_) == 0)
{
lean_object* v_a_1275_; lean_object* v___x_1277_; uint8_t v_isShared_1278_; uint8_t v_isSharedCheck_1287_; 
v_a_1275_ = lean_ctor_get(v___x_1274_, 0);
v_isSharedCheck_1287_ = !lean_is_exclusive(v___x_1274_);
if (v_isSharedCheck_1287_ == 0)
{
v___x_1277_ = v___x_1274_;
v_isShared_1278_ = v_isSharedCheck_1287_;
goto v_resetjp_1276_;
}
else
{
lean_inc(v_a_1275_);
lean_dec(v___x_1274_);
v___x_1277_ = lean_box(0);
v_isShared_1278_ = v_isSharedCheck_1287_;
goto v_resetjp_1276_;
}
v_resetjp_1276_:
{
if (lean_obj_tag(v_a_1275_) == 0)
{
lean_object* v___x_1280_; 
if (v_isShared_1278_ == 0)
{
lean_ctor_set(v___x_1277_, 0, v___x_1271_);
v___x_1280_ = v___x_1277_;
goto v_reusejp_1279_;
}
else
{
lean_object* v_reuseFailAlloc_1281_; 
v_reuseFailAlloc_1281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1281_, 0, v___x_1271_);
v___x_1280_ = v_reuseFailAlloc_1281_;
goto v_reusejp_1279_;
}
v_reusejp_1279_:
{
return v___x_1280_;
}
}
else
{
lean_object* v_val_1282_; lean_object* v_snd_1283_; lean_object* v___x_1285_; 
v_val_1282_ = lean_ctor_get(v_a_1275_, 0);
lean_inc(v_val_1282_);
lean_dec_ref_known(v_a_1275_, 1);
v_snd_1283_ = lean_ctor_get(v_val_1282_, 1);
lean_inc(v_snd_1283_);
lean_dec(v_val_1282_);
if (v_isShared_1278_ == 0)
{
lean_ctor_set(v___x_1277_, 0, v_snd_1283_);
v___x_1285_ = v___x_1277_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1286_; 
v_reuseFailAlloc_1286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1286_, 0, v_snd_1283_);
v___x_1285_ = v_reuseFailAlloc_1286_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
return v___x_1285_;
}
}
}
}
else
{
lean_object* v_a_1288_; lean_object* v___x_1290_; uint8_t v_isShared_1291_; uint8_t v_isSharedCheck_1295_; 
v_a_1288_ = lean_ctor_get(v___x_1274_, 0);
v_isSharedCheck_1295_ = !lean_is_exclusive(v___x_1274_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1290_ = v___x_1274_;
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
else
{
lean_inc(v_a_1288_);
lean_dec(v___x_1274_);
v___x_1290_ = lean_box(0);
v_isShared_1291_ = v_isSharedCheck_1295_;
goto v_resetjp_1289_;
}
v_resetjp_1289_:
{
lean_object* v___x_1293_; 
if (v_isShared_1291_ == 0)
{
v___x_1293_ = v___x_1290_;
goto v_reusejp_1292_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v_a_1288_);
v___x_1293_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1292_;
}
v_reusejp_1292_:
{
return v___x_1293_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___boxed(lean_object* v_tree_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_, lean_object* v_a_1299_){
_start:
{
lean_object* v_res_1300_; 
v_res_1300_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(v_tree_1296_, v_a_1297_, v_a_1298_);
lean_dec(v_a_1298_);
lean_dec_ref(v_a_1297_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7(lean_object* v_00_u03b1_1301_, lean_object* v_msg_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_){
_start:
{
lean_object* v___x_1306_; 
v___x_1306_ = lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___redArg(v_msg_1302_, v___y_1303_, v___y_1304_);
return v___x_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7___boxed(lean_object* v_00_u03b1_1307_, lean_object* v_msg_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_mathlib_panic___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__7(v_00_u03b1_1307_, v_msg_1308_, v___y_1309_, v___y_1310_);
lean_dec(v___y_1310_);
lean_dec_ref(v___y_1309_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6(lean_object* v_00_u03b1_1313_, lean_object* v_preNode_1314_, lean_object* v_postNode_1315_, lean_object* v_x_1316_, lean_object* v_x_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_){
_start:
{
lean_object* v___x_1321_; 
v___x_1321_ = lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___redArg(v_preNode_1314_, v_postNode_1315_, v_x_1316_, v_x_1317_, v___y_1318_, v___y_1319_);
return v___x_1321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6___boxed(lean_object* v_00_u03b1_1322_, lean_object* v_preNode_1323_, lean_object* v_postNode_1324_, lean_object* v_x_1325_, lean_object* v_x_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_){
_start:
{
lean_object* v_res_1330_; 
v_res_1330_ = lp_mathlib___private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6(v_00_u03b1_1322_, v_preNode_1323_, v_postNode_1324_, v_x_1325_, v_x_1326_, v___y_1327_, v___y_1328_);
lean_dec(v___y_1328_);
lean_dec_ref(v___y_1327_);
return v_res_1330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8(lean_object* v_00_u03b1_1331_, lean_object* v_preNode_1332_, lean_object* v_postNode_1333_, lean_object* v___x_1334_, lean_object* v_x_1335_, lean_object* v_x_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_){
_start:
{
lean_object* v___x_1340_; 
v___x_1340_ = lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___redArg(v_preNode_1332_, v_postNode_1333_, v___x_1334_, v_x_1335_, v_x_1336_, v___y_1337_, v___y_1338_);
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8___boxed(lean_object* v_00_u03b1_1341_, lean_object* v_preNode_1342_, lean_object* v_postNode_1343_, lean_object* v___x_1344_, lean_object* v_x_1345_, lean_object* v_x_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_){
_start:
{
lean_object* v_res_1350_; 
v_res_1350_ = lp_mathlib_List_mapM_loop___at___00__private_Lean_Server_InfoUtils_0__Lean_Elab_InfoTree_visitM_go___at___00Mathlib_TacticAnalysis_findTacticSeqs_spec__6_spec__8(v_00_u03b1_1341_, v_preNode_1342_, v_postNode_1343_, v___x_1344_, v_x_1345_, v_x_1346_, v___y_1347_, v___y_1348_);
lean_dec(v___y_1348_);
lean_dec_ref(v___y_1347_);
return v_res_1350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg(lean_object* v_o_1351_, lean_object* v___y_1352_){
_start:
{
lean_object* v___x_1354_; lean_object* v_env_1355_; lean_object* v___x_1356_; lean_object* v_toEnvExtension_1357_; lean_object* v_asyncMode_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v_merged_1362_; lean_object* v___x_1364_; uint8_t v_isShared_1365_; uint8_t v_isSharedCheck_1370_; 
v___x_1354_ = lean_st_ref_get(v___y_1352_);
v_env_1355_ = lean_ctor_get(v___x_1354_, 0);
lean_inc_ref(v_env_1355_);
lean_dec(v___x_1354_);
v___x_1356_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1357_ = lean_ctor_get(v___x_1356_, 0);
v_asyncMode_1358_ = lean_ctor_get(v_toEnvExtension_1357_, 2);
v___x_1359_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1360_ = lean_box(0);
v___x_1361_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1359_, v___x_1356_, v_env_1355_, v_asyncMode_1358_, v___x_1360_);
v_merged_1362_ = lean_ctor_get(v___x_1361_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1361_);
if (v_isSharedCheck_1370_ == 0)
{
lean_object* v_unused_1371_; 
v_unused_1371_ = lean_ctor_get(v___x_1361_, 1);
lean_dec(v_unused_1371_);
v___x_1364_ = v___x_1361_;
v_isShared_1365_ = v_isSharedCheck_1370_;
goto v_resetjp_1363_;
}
else
{
lean_inc(v_merged_1362_);
lean_dec(v___x_1361_);
v___x_1364_ = lean_box(0);
v_isShared_1365_ = v_isSharedCheck_1370_;
goto v_resetjp_1363_;
}
v_resetjp_1363_:
{
lean_object* v___x_1367_; 
if (v_isShared_1365_ == 0)
{
lean_ctor_set(v___x_1364_, 1, v_merged_1362_);
lean_ctor_set(v___x_1364_, 0, v_o_1351_);
v___x_1367_ = v___x_1364_;
goto v_reusejp_1366_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_o_1351_);
lean_ctor_set(v_reuseFailAlloc_1369_, 1, v_merged_1362_);
v___x_1367_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1366_;
}
v_reusejp_1366_:
{
lean_object* v___x_1368_; 
v___x_1368_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1368_, 0, v___x_1367_);
return v___x_1368_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg___boxed(lean_object* v_o_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_){
_start:
{
lean_object* v_res_1375_; 
v_res_1375_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg(v_o_1372_, v___y_1373_);
lean_dec(v___y_1373_);
return v_res_1375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0(lean_object* v___y_1376_, lean_object* v___y_1377_){
_start:
{
lean_object* v___x_1379_; lean_object* v_scopes_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v_opts_1383_; lean_object* v___x_1384_; 
v___x_1379_ = lean_st_ref_get(v___y_1377_);
v_scopes_1380_ = lean_ctor_get(v___x_1379_, 2);
lean_inc(v_scopes_1380_);
lean_dec(v___x_1379_);
v___x_1381_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1382_ = l_List_head_x21___redArg(v___x_1381_, v_scopes_1380_);
lean_dec(v_scopes_1380_);
v_opts_1383_ = lean_ctor_get(v___x_1382_, 1);
lean_inc_ref(v_opts_1383_);
lean_dec(v___x_1382_);
v___x_1384_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg(v_opts_1383_, v___y_1377_);
return v___x_1384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0___boxed(lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_){
_start:
{
lean_object* v_res_1388_; 
v_res_1388_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0(v___y_1385_, v___y_1386_);
lean_dec(v___y_1386_);
lean_dec_ref(v___y_1385_);
return v_res_1388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4(lean_object* v_a_1389_, lean_object* v_as_1390_, size_t v_i_1391_, size_t v_stop_1392_, lean_object* v_b_1393_){
_start:
{
lean_object* v___y_1395_; uint8_t v___x_1399_; 
v___x_1399_ = lean_usize_dec_eq(v_i_1391_, v_stop_1392_);
if (v___x_1399_ == 0)
{
lean_object* v___x_1400_; lean_object* v_opt_1401_; 
v___x_1400_ = lean_array_uget_borrowed(v_as_1390_, v_i_1391_);
v_opt_1401_ = lean_ctor_get(v___x_1400_, 1);
if (lean_obj_tag(v_opt_1401_) == 1)
{
lean_object* v_val_1402_; uint8_t v___x_1403_; 
v_val_1402_ = lean_ctor_get(v_opt_1401_, 0);
v___x_1403_ = l_Lean_Linter_getLinterValue(v_val_1402_, v_a_1389_);
if (v___x_1403_ == 0)
{
v___y_1395_ = v_b_1393_;
goto v___jp_1394_;
}
else
{
lean_object* v___x_1404_; 
lean_inc(v___x_1400_);
v___x_1404_ = lean_array_push(v_b_1393_, v___x_1400_);
v___y_1395_ = v___x_1404_;
goto v___jp_1394_;
}
}
else
{
v___y_1395_ = v_b_1393_;
goto v___jp_1394_;
}
}
else
{
return v_b_1393_;
}
v___jp_1394_:
{
size_t v___x_1396_; size_t v___x_1397_; 
v___x_1396_ = ((size_t)1ULL);
v___x_1397_ = lean_usize_add(v_i_1391_, v___x_1396_);
v_i_1391_ = v___x_1397_;
v_b_1393_ = v___y_1395_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4___boxed(lean_object* v_a_1405_, lean_object* v_as_1406_, lean_object* v_i_1407_, lean_object* v_stop_1408_, lean_object* v_b_1409_){
_start:
{
size_t v_i_boxed_1410_; size_t v_stop_boxed_1411_; lean_object* v_res_1412_; 
v_i_boxed_1410_ = lean_unbox_usize(v_i_1407_);
lean_dec(v_i_1407_);
v_stop_boxed_1411_ = lean_unbox_usize(v_stop_1408_);
lean_dec(v_stop_1408_);
v_res_1412_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4(v_a_1405_, v_as_1406_, v_i_boxed_1410_, v_stop_boxed_1411_, v_b_1409_);
lean_dec_ref(v_as_1406_);
lean_dec_ref(v_a_1405_);
return v_res_1412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__1(lean_object* v_a_1413_, lean_object* v_as_1414_, size_t v_sz_1415_, size_t v_i_1416_, lean_object* v_b_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_){
_start:
{
uint8_t v___x_1421_; 
v___x_1421_ = lean_usize_dec_lt(v_i_1416_, v_sz_1415_);
if (v___x_1421_ == 0)
{
lean_object* v___x_1422_; 
lean_dec_ref(v_a_1413_);
v___x_1422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1422_, 0, v_b_1417_);
return v___x_1422_;
}
else
{
lean_object* v_a_1423_; lean_object* v_toConfig_1424_; lean_object* v___x_1425_; 
v_a_1423_ = lean_array_uget_borrowed(v_as_1414_, v_i_1416_);
v_toConfig_1424_ = lean_ctor_get(v_a_1423_, 0);
lean_inc_ref(v_toConfig_1424_);
lean_inc(v___y_1419_);
lean_inc_ref(v___y_1418_);
lean_inc_ref(v_a_1413_);
v___x_1425_ = lean_apply_4(v_toConfig_1424_, v_a_1413_, v___y_1418_, v___y_1419_, lean_box(0));
if (lean_obj_tag(v___x_1425_) == 0)
{
lean_object* v___x_1426_; size_t v___x_1427_; size_t v___x_1428_; 
lean_dec_ref_known(v___x_1425_, 1);
v___x_1426_ = lean_box(0);
v___x_1427_ = ((size_t)1ULL);
v___x_1428_ = lean_usize_add(v_i_1416_, v___x_1427_);
v_i_1416_ = v___x_1428_;
v_b_1417_ = v___x_1426_;
goto _start;
}
else
{
lean_dec_ref(v_a_1413_);
return v___x_1425_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__1___boxed(lean_object* v_a_1430_, lean_object* v_as_1431_, lean_object* v_sz_1432_, lean_object* v_i_1433_, lean_object* v_b_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_){
_start:
{
size_t v_sz_boxed_1438_; size_t v_i_boxed_1439_; lean_object* v_res_1440_; 
v_sz_boxed_1438_ = lean_unbox_usize(v_sz_1432_);
lean_dec(v_sz_1432_);
v_i_boxed_1439_ = lean_unbox_usize(v_i_1433_);
lean_dec(v_i_1433_);
v_res_1440_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__1(v_a_1430_, v_as_1431_, v_sz_boxed_1438_, v_i_boxed_1439_, v_b_1434_, v___y_1435_, v___y_1436_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec_ref(v_as_1431_);
return v_res_1440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(lean_object* v___y_1441_, lean_object* v_as_1442_, size_t v_sz_1443_, size_t v_i_1444_, lean_object* v_b_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_){
_start:
{
uint8_t v___x_1449_; 
v___x_1449_ = lean_usize_dec_lt(v_i_1444_, v_sz_1443_);
if (v___x_1449_ == 0)
{
lean_object* v___x_1450_; 
v___x_1450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1450_, 0, v_b_1445_);
return v___x_1450_;
}
else
{
lean_object* v___x_1451_; lean_object* v_a_1452_; size_t v_sz_1453_; size_t v___x_1454_; lean_object* v___x_1455_; 
v___x_1451_ = lean_box(0);
v_a_1452_ = lean_array_uget_borrowed(v_as_1442_, v_i_1444_);
v_sz_1453_ = lean_array_size(v___y_1441_);
v___x_1454_ = ((size_t)0ULL);
lean_inc(v_a_1452_);
v___x_1455_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__1(v_a_1452_, v___y_1441_, v_sz_1453_, v___x_1454_, v___x_1451_, v___y_1446_, v___y_1447_);
if (lean_obj_tag(v___x_1455_) == 0)
{
size_t v___x_1456_; size_t v___x_1457_; 
lean_dec_ref_known(v___x_1455_, 1);
v___x_1456_ = ((size_t)1ULL);
v___x_1457_ = lean_usize_add(v_i_1444_, v___x_1456_);
v_i_1444_ = v___x_1457_;
v_b_1445_ = v___x_1451_;
goto _start;
}
else
{
return v___x_1455_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2___boxed(lean_object* v___y_1459_, lean_object* v_as_1460_, lean_object* v_sz_1461_, lean_object* v_i_1462_, lean_object* v_b_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_){
_start:
{
size_t v_sz_boxed_1467_; size_t v_i_boxed_1468_; lean_object* v_res_1469_; 
v_sz_boxed_1467_ = lean_unbox_usize(v_sz_1461_);
lean_dec(v_sz_1461_);
v_i_boxed_1468_ = lean_unbox_usize(v_i_1462_);
lean_dec(v_i_1462_);
v_res_1469_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(v___y_1459_, v_as_1460_, v_sz_boxed_1467_, v_i_boxed_1468_, v_b_1463_, v___y_1464_, v___y_1465_);
lean_dec(v___y_1465_);
lean_dec_ref(v___y_1464_);
lean_dec_ref(v_as_1460_);
lean_dec_ref(v___y_1459_);
return v_res_1469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8(lean_object* v___y_1473_, lean_object* v_as_1474_, size_t v_sz_1475_, size_t v_i_1476_, lean_object* v_b_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_){
_start:
{
uint8_t v___x_1481_; 
v___x_1481_ = lean_usize_dec_lt(v_i_1476_, v_sz_1475_);
if (v___x_1481_ == 0)
{
lean_object* v___x_1482_; 
v___x_1482_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1482_, 0, v_b_1477_);
return v___x_1482_;
}
else
{
lean_object* v_a_1483_; lean_object* v___x_1484_; 
lean_dec_ref(v_b_1477_);
v_a_1483_ = lean_array_uget_borrowed(v_as_1474_, v_i_1476_);
lean_inc(v_a_1483_);
v___x_1484_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(v_a_1483_, v___y_1478_, v___y_1479_);
if (lean_obj_tag(v___x_1484_) == 0)
{
lean_object* v_a_1485_; lean_object* v___x_1486_; size_t v_sz_1487_; size_t v___x_1488_; lean_object* v___x_1489_; 
v_a_1485_ = lean_ctor_get(v___x_1484_, 0);
lean_inc(v_a_1485_);
lean_dec_ref_known(v___x_1484_, 1);
v___x_1486_ = lean_box(0);
v_sz_1487_ = lean_array_size(v_a_1485_);
v___x_1488_ = ((size_t)0ULL);
v___x_1489_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(v___y_1473_, v_a_1485_, v_sz_1487_, v___x_1488_, v___x_1486_, v___y_1478_, v___y_1479_);
lean_dec(v_a_1485_);
if (lean_obj_tag(v___x_1489_) == 0)
{
lean_object* v___x_1490_; size_t v___x_1491_; size_t v___x_1492_; 
lean_dec_ref_known(v___x_1489_, 1);
v___x_1490_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___closed__0));
v___x_1491_ = ((size_t)1ULL);
v___x_1492_ = lean_usize_add(v_i_1476_, v___x_1491_);
v_i_1476_ = v___x_1492_;
v_b_1477_ = v___x_1490_;
goto _start;
}
else
{
lean_object* v_a_1494_; lean_object* v___x_1496_; uint8_t v_isShared_1497_; uint8_t v_isSharedCheck_1501_; 
v_a_1494_ = lean_ctor_get(v___x_1489_, 0);
v_isSharedCheck_1501_ = !lean_is_exclusive(v___x_1489_);
if (v_isSharedCheck_1501_ == 0)
{
v___x_1496_ = v___x_1489_;
v_isShared_1497_ = v_isSharedCheck_1501_;
goto v_resetjp_1495_;
}
else
{
lean_inc(v_a_1494_);
lean_dec(v___x_1489_);
v___x_1496_ = lean_box(0);
v_isShared_1497_ = v_isSharedCheck_1501_;
goto v_resetjp_1495_;
}
v_resetjp_1495_:
{
lean_object* v___x_1499_; 
if (v_isShared_1497_ == 0)
{
v___x_1499_ = v___x_1496_;
goto v_reusejp_1498_;
}
else
{
lean_object* v_reuseFailAlloc_1500_; 
v_reuseFailAlloc_1500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1500_, 0, v_a_1494_);
v___x_1499_ = v_reuseFailAlloc_1500_;
goto v_reusejp_1498_;
}
v_reusejp_1498_:
{
return v___x_1499_;
}
}
}
}
else
{
lean_object* v_a_1502_; lean_object* v___x_1504_; uint8_t v_isShared_1505_; uint8_t v_isSharedCheck_1509_; 
v_a_1502_ = lean_ctor_get(v___x_1484_, 0);
v_isSharedCheck_1509_ = !lean_is_exclusive(v___x_1484_);
if (v_isSharedCheck_1509_ == 0)
{
v___x_1504_ = v___x_1484_;
v_isShared_1505_ = v_isSharedCheck_1509_;
goto v_resetjp_1503_;
}
else
{
lean_inc(v_a_1502_);
lean_dec(v___x_1484_);
v___x_1504_ = lean_box(0);
v_isShared_1505_ = v_isSharedCheck_1509_;
goto v_resetjp_1503_;
}
v_resetjp_1503_:
{
lean_object* v___x_1507_; 
if (v_isShared_1505_ == 0)
{
v___x_1507_ = v___x_1504_;
goto v_reusejp_1506_;
}
else
{
lean_object* v_reuseFailAlloc_1508_; 
v_reuseFailAlloc_1508_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1508_, 0, v_a_1502_);
v___x_1507_ = v_reuseFailAlloc_1508_;
goto v_reusejp_1506_;
}
v_reusejp_1506_:
{
return v___x_1507_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___boxed(lean_object* v___y_1510_, lean_object* v_as_1511_, lean_object* v_sz_1512_, lean_object* v_i_1513_, lean_object* v_b_1514_, lean_object* v___y_1515_, lean_object* v___y_1516_, lean_object* v___y_1517_){
_start:
{
size_t v_sz_boxed_1518_; size_t v_i_boxed_1519_; lean_object* v_res_1520_; 
v_sz_boxed_1518_ = lean_unbox_usize(v_sz_1512_);
lean_dec(v_sz_1512_);
v_i_boxed_1519_ = lean_unbox_usize(v_i_1513_);
lean_dec(v_i_1513_);
v_res_1520_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8(v___y_1510_, v_as_1511_, v_sz_boxed_1518_, v_i_boxed_1519_, v_b_1514_, v___y_1515_, v___y_1516_);
lean_dec(v___y_1516_);
lean_dec_ref(v___y_1515_);
lean_dec_ref(v_as_1511_);
lean_dec_ref(v___y_1510_);
return v_res_1520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6(lean_object* v___y_1521_, lean_object* v_as_1522_, size_t v_sz_1523_, size_t v_i_1524_, lean_object* v_b_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_){
_start:
{
uint8_t v___x_1529_; 
v___x_1529_ = lean_usize_dec_lt(v_i_1524_, v_sz_1523_);
if (v___x_1529_ == 0)
{
lean_object* v___x_1530_; 
v___x_1530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1530_, 0, v_b_1525_);
return v___x_1530_;
}
else
{
lean_object* v_a_1531_; lean_object* v___x_1532_; 
lean_dec_ref(v_b_1525_);
v_a_1531_ = lean_array_uget_borrowed(v_as_1522_, v_i_1524_);
lean_inc(v_a_1531_);
v___x_1532_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(v_a_1531_, v___y_1526_, v___y_1527_);
if (lean_obj_tag(v___x_1532_) == 0)
{
lean_object* v_a_1533_; lean_object* v___x_1534_; size_t v_sz_1535_; size_t v___x_1536_; lean_object* v___x_1537_; 
v_a_1533_ = lean_ctor_get(v___x_1532_, 0);
lean_inc(v_a_1533_);
lean_dec_ref_known(v___x_1532_, 1);
v___x_1534_ = lean_box(0);
v_sz_1535_ = lean_array_size(v_a_1533_);
v___x_1536_ = ((size_t)0ULL);
v___x_1537_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(v___y_1521_, v_a_1533_, v_sz_1535_, v___x_1536_, v___x_1534_, v___y_1526_, v___y_1527_);
lean_dec(v_a_1533_);
if (lean_obj_tag(v___x_1537_) == 0)
{
lean_object* v___x_1538_; size_t v___x_1539_; size_t v___x_1540_; lean_object* v___x_1541_; 
lean_dec_ref_known(v___x_1537_, 1);
v___x_1538_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8___closed__0));
v___x_1539_ = ((size_t)1ULL);
v___x_1540_ = lean_usize_add(v_i_1524_, v___x_1539_);
v___x_1541_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6_spec__8(v___y_1521_, v_as_1522_, v_sz_1523_, v___x_1540_, v___x_1538_, v___y_1526_, v___y_1527_);
return v___x_1541_;
}
else
{
lean_object* v_a_1542_; lean_object* v___x_1544_; uint8_t v_isShared_1545_; uint8_t v_isSharedCheck_1549_; 
v_a_1542_ = lean_ctor_get(v___x_1537_, 0);
v_isSharedCheck_1549_ = !lean_is_exclusive(v___x_1537_);
if (v_isSharedCheck_1549_ == 0)
{
v___x_1544_ = v___x_1537_;
v_isShared_1545_ = v_isSharedCheck_1549_;
goto v_resetjp_1543_;
}
else
{
lean_inc(v_a_1542_);
lean_dec(v___x_1537_);
v___x_1544_ = lean_box(0);
v_isShared_1545_ = v_isSharedCheck_1549_;
goto v_resetjp_1543_;
}
v_resetjp_1543_:
{
lean_object* v___x_1547_; 
if (v_isShared_1545_ == 0)
{
v___x_1547_ = v___x_1544_;
goto v_reusejp_1546_;
}
else
{
lean_object* v_reuseFailAlloc_1548_; 
v_reuseFailAlloc_1548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1548_, 0, v_a_1542_);
v___x_1547_ = v_reuseFailAlloc_1548_;
goto v_reusejp_1546_;
}
v_reusejp_1546_:
{
return v___x_1547_;
}
}
}
}
else
{
lean_object* v_a_1550_; lean_object* v___x_1552_; uint8_t v_isShared_1553_; uint8_t v_isSharedCheck_1557_; 
v_a_1550_ = lean_ctor_get(v___x_1532_, 0);
v_isSharedCheck_1557_ = !lean_is_exclusive(v___x_1532_);
if (v_isSharedCheck_1557_ == 0)
{
v___x_1552_ = v___x_1532_;
v_isShared_1553_ = v_isSharedCheck_1557_;
goto v_resetjp_1551_;
}
else
{
lean_inc(v_a_1550_);
lean_dec(v___x_1532_);
v___x_1552_ = lean_box(0);
v_isShared_1553_ = v_isSharedCheck_1557_;
goto v_resetjp_1551_;
}
v_resetjp_1551_:
{
lean_object* v___x_1555_; 
if (v_isShared_1553_ == 0)
{
v___x_1555_ = v___x_1552_;
goto v_reusejp_1554_;
}
else
{
lean_object* v_reuseFailAlloc_1556_; 
v_reuseFailAlloc_1556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1556_, 0, v_a_1550_);
v___x_1555_ = v_reuseFailAlloc_1556_;
goto v_reusejp_1554_;
}
v_reusejp_1554_:
{
return v___x_1555_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6___boxed(lean_object* v___y_1558_, lean_object* v_as_1559_, lean_object* v_sz_1560_, lean_object* v_i_1561_, lean_object* v_b_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_){
_start:
{
size_t v_sz_boxed_1566_; size_t v_i_boxed_1567_; lean_object* v_res_1568_; 
v_sz_boxed_1566_ = lean_unbox_usize(v_sz_1560_);
lean_dec(v_sz_1560_);
v_i_boxed_1567_ = lean_unbox_usize(v_i_1561_);
lean_dec(v_i_1561_);
v_res_1568_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6(v___y_1558_, v_as_1559_, v_sz_boxed_1566_, v_i_boxed_1567_, v_b_1562_, v___y_1563_, v___y_1564_);
lean_dec(v___y_1564_);
lean_dec_ref(v___y_1563_);
lean_dec_ref(v_as_1559_);
lean_dec_ref(v___y_1558_);
return v_res_1568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4(lean_object* v_init_1569_, lean_object* v___y_1570_, lean_object* v_n_1571_, lean_object* v_b_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_){
_start:
{
if (lean_obj_tag(v_n_1571_) == 0)
{
lean_object* v_cs_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; size_t v_sz_1579_; size_t v___x_1580_; lean_object* v___x_1581_; 
v_cs_1576_ = lean_ctor_get(v_n_1571_, 0);
v___x_1577_ = lean_box(0);
v___x_1578_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1578_, 0, v___x_1577_);
lean_ctor_set(v___x_1578_, 1, v_b_1572_);
v_sz_1579_ = lean_array_size(v_cs_1576_);
v___x_1580_ = ((size_t)0ULL);
v___x_1581_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__5(v_init_1569_, v___y_1570_, v_cs_1576_, v_sz_1579_, v___x_1580_, v___x_1578_, v___y_1573_, v___y_1574_);
if (lean_obj_tag(v___x_1581_) == 0)
{
lean_object* v_a_1582_; lean_object* v___x_1584_; uint8_t v_isShared_1585_; uint8_t v_isSharedCheck_1596_; 
v_a_1582_ = lean_ctor_get(v___x_1581_, 0);
v_isSharedCheck_1596_ = !lean_is_exclusive(v___x_1581_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1584_ = v___x_1581_;
v_isShared_1585_ = v_isSharedCheck_1596_;
goto v_resetjp_1583_;
}
else
{
lean_inc(v_a_1582_);
lean_dec(v___x_1581_);
v___x_1584_ = lean_box(0);
v_isShared_1585_ = v_isSharedCheck_1596_;
goto v_resetjp_1583_;
}
v_resetjp_1583_:
{
lean_object* v_fst_1586_; 
v_fst_1586_ = lean_ctor_get(v_a_1582_, 0);
if (lean_obj_tag(v_fst_1586_) == 0)
{
lean_object* v_snd_1587_; lean_object* v___x_1588_; lean_object* v___x_1590_; 
v_snd_1587_ = lean_ctor_get(v_a_1582_, 1);
lean_inc(v_snd_1587_);
lean_dec(v_a_1582_);
v___x_1588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1588_, 0, v_snd_1587_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v___x_1588_);
v___x_1590_ = v___x_1584_;
goto v_reusejp_1589_;
}
else
{
lean_object* v_reuseFailAlloc_1591_; 
v_reuseFailAlloc_1591_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1591_, 0, v___x_1588_);
v___x_1590_ = v_reuseFailAlloc_1591_;
goto v_reusejp_1589_;
}
v_reusejp_1589_:
{
return v___x_1590_;
}
}
else
{
lean_object* v_val_1592_; lean_object* v___x_1594_; 
lean_inc_ref(v_fst_1586_);
lean_dec(v_a_1582_);
v_val_1592_ = lean_ctor_get(v_fst_1586_, 0);
lean_inc(v_val_1592_);
lean_dec_ref_known(v_fst_1586_, 1);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v_val_1592_);
v___x_1594_ = v___x_1584_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_val_1592_);
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
else
{
lean_object* v_a_1597_; lean_object* v___x_1599_; uint8_t v_isShared_1600_; uint8_t v_isSharedCheck_1604_; 
v_a_1597_ = lean_ctor_get(v___x_1581_, 0);
v_isSharedCheck_1604_ = !lean_is_exclusive(v___x_1581_);
if (v_isSharedCheck_1604_ == 0)
{
v___x_1599_ = v___x_1581_;
v_isShared_1600_ = v_isSharedCheck_1604_;
goto v_resetjp_1598_;
}
else
{
lean_inc(v_a_1597_);
lean_dec(v___x_1581_);
v___x_1599_ = lean_box(0);
v_isShared_1600_ = v_isSharedCheck_1604_;
goto v_resetjp_1598_;
}
v_resetjp_1598_:
{
lean_object* v___x_1602_; 
if (v_isShared_1600_ == 0)
{
v___x_1602_ = v___x_1599_;
goto v_reusejp_1601_;
}
else
{
lean_object* v_reuseFailAlloc_1603_; 
v_reuseFailAlloc_1603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1603_, 0, v_a_1597_);
v___x_1602_ = v_reuseFailAlloc_1603_;
goto v_reusejp_1601_;
}
v_reusejp_1601_:
{
return v___x_1602_;
}
}
}
}
else
{
lean_object* v_vs_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; size_t v_sz_1608_; size_t v___x_1609_; lean_object* v___x_1610_; 
v_vs_1605_ = lean_ctor_get(v_n_1571_, 0);
v___x_1606_ = lean_box(0);
v___x_1607_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1607_, 0, v___x_1606_);
lean_ctor_set(v___x_1607_, 1, v_b_1572_);
v_sz_1608_ = lean_array_size(v_vs_1605_);
v___x_1609_ = ((size_t)0ULL);
v___x_1610_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__6(v___y_1570_, v_vs_1605_, v_sz_1608_, v___x_1609_, v___x_1607_, v___y_1573_, v___y_1574_);
if (lean_obj_tag(v___x_1610_) == 0)
{
lean_object* v_a_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1625_; 
v_a_1611_ = lean_ctor_get(v___x_1610_, 0);
v_isSharedCheck_1625_ = !lean_is_exclusive(v___x_1610_);
if (v_isSharedCheck_1625_ == 0)
{
v___x_1613_ = v___x_1610_;
v_isShared_1614_ = v_isSharedCheck_1625_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_a_1611_);
lean_dec(v___x_1610_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1625_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
lean_object* v_fst_1615_; 
v_fst_1615_ = lean_ctor_get(v_a_1611_, 0);
if (lean_obj_tag(v_fst_1615_) == 0)
{
lean_object* v_snd_1616_; lean_object* v___x_1617_; lean_object* v___x_1619_; 
v_snd_1616_ = lean_ctor_get(v_a_1611_, 1);
lean_inc(v_snd_1616_);
lean_dec(v_a_1611_);
v___x_1617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1617_, 0, v_snd_1616_);
if (v_isShared_1614_ == 0)
{
lean_ctor_set(v___x_1613_, 0, v___x_1617_);
v___x_1619_ = v___x_1613_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v___x_1617_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
return v___x_1619_;
}
}
else
{
lean_object* v_val_1621_; lean_object* v___x_1623_; 
lean_inc_ref(v_fst_1615_);
lean_dec(v_a_1611_);
v_val_1621_ = lean_ctor_get(v_fst_1615_, 0);
lean_inc(v_val_1621_);
lean_dec_ref_known(v_fst_1615_, 1);
if (v_isShared_1614_ == 0)
{
lean_ctor_set(v___x_1613_, 0, v_val_1621_);
v___x_1623_ = v___x_1613_;
goto v_reusejp_1622_;
}
else
{
lean_object* v_reuseFailAlloc_1624_; 
v_reuseFailAlloc_1624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1624_, 0, v_val_1621_);
v___x_1623_ = v_reuseFailAlloc_1624_;
goto v_reusejp_1622_;
}
v_reusejp_1622_:
{
return v___x_1623_;
}
}
}
}
else
{
lean_object* v_a_1626_; lean_object* v___x_1628_; uint8_t v_isShared_1629_; uint8_t v_isSharedCheck_1633_; 
v_a_1626_ = lean_ctor_get(v___x_1610_, 0);
v_isSharedCheck_1633_ = !lean_is_exclusive(v___x_1610_);
if (v_isSharedCheck_1633_ == 0)
{
v___x_1628_ = v___x_1610_;
v_isShared_1629_ = v_isSharedCheck_1633_;
goto v_resetjp_1627_;
}
else
{
lean_inc(v_a_1626_);
lean_dec(v___x_1610_);
v___x_1628_ = lean_box(0);
v_isShared_1629_ = v_isSharedCheck_1633_;
goto v_resetjp_1627_;
}
v_resetjp_1627_:
{
lean_object* v___x_1631_; 
if (v_isShared_1629_ == 0)
{
v___x_1631_ = v___x_1628_;
goto v_reusejp_1630_;
}
else
{
lean_object* v_reuseFailAlloc_1632_; 
v_reuseFailAlloc_1632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1632_, 0, v_a_1626_);
v___x_1631_ = v_reuseFailAlloc_1632_;
goto v_reusejp_1630_;
}
v_reusejp_1630_:
{
return v___x_1631_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__5(lean_object* v_init_1634_, lean_object* v___y_1635_, lean_object* v_as_1636_, size_t v_sz_1637_, size_t v_i_1638_, lean_object* v_b_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_){
_start:
{
uint8_t v___x_1643_; 
v___x_1643_ = lean_usize_dec_lt(v_i_1638_, v_sz_1637_);
if (v___x_1643_ == 0)
{
lean_object* v___x_1644_; 
v___x_1644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1644_, 0, v_b_1639_);
return v___x_1644_;
}
else
{
lean_object* v_snd_1645_; lean_object* v___x_1647_; uint8_t v_isShared_1648_; uint8_t v_isSharedCheck_1679_; 
v_snd_1645_ = lean_ctor_get(v_b_1639_, 1);
v_isSharedCheck_1679_ = !lean_is_exclusive(v_b_1639_);
if (v_isSharedCheck_1679_ == 0)
{
lean_object* v_unused_1680_; 
v_unused_1680_ = lean_ctor_get(v_b_1639_, 0);
lean_dec(v_unused_1680_);
v___x_1647_ = v_b_1639_;
v_isShared_1648_ = v_isSharedCheck_1679_;
goto v_resetjp_1646_;
}
else
{
lean_inc(v_snd_1645_);
lean_dec(v_b_1639_);
v___x_1647_ = lean_box(0);
v_isShared_1648_ = v_isSharedCheck_1679_;
goto v_resetjp_1646_;
}
v_resetjp_1646_:
{
lean_object* v_a_1649_; lean_object* v___x_1650_; 
v_a_1649_ = lean_array_uget_borrowed(v_as_1636_, v_i_1638_);
lean_inc(v_snd_1645_);
v___x_1650_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4(v_init_1634_, v___y_1635_, v_a_1649_, v_snd_1645_, v___y_1640_, v___y_1641_);
if (lean_obj_tag(v___x_1650_) == 0)
{
lean_object* v_a_1651_; lean_object* v___x_1653_; uint8_t v_isShared_1654_; uint8_t v_isSharedCheck_1670_; 
v_a_1651_ = lean_ctor_get(v___x_1650_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1650_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1653_ = v___x_1650_;
v_isShared_1654_ = v_isSharedCheck_1670_;
goto v_resetjp_1652_;
}
else
{
lean_inc(v_a_1651_);
lean_dec(v___x_1650_);
v___x_1653_ = lean_box(0);
v_isShared_1654_ = v_isSharedCheck_1670_;
goto v_resetjp_1652_;
}
v_resetjp_1652_:
{
if (lean_obj_tag(v_a_1651_) == 0)
{
lean_object* v___x_1655_; lean_object* v___x_1657_; 
v___x_1655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1655_, 0, v_a_1651_);
if (v_isShared_1648_ == 0)
{
lean_ctor_set(v___x_1647_, 0, v___x_1655_);
v___x_1657_ = v___x_1647_;
goto v_reusejp_1656_;
}
else
{
lean_object* v_reuseFailAlloc_1661_; 
v_reuseFailAlloc_1661_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1661_, 0, v___x_1655_);
lean_ctor_set(v_reuseFailAlloc_1661_, 1, v_snd_1645_);
v___x_1657_ = v_reuseFailAlloc_1661_;
goto v_reusejp_1656_;
}
v_reusejp_1656_:
{
lean_object* v___x_1659_; 
if (v_isShared_1654_ == 0)
{
lean_ctor_set(v___x_1653_, 0, v___x_1657_);
v___x_1659_ = v___x_1653_;
goto v_reusejp_1658_;
}
else
{
lean_object* v_reuseFailAlloc_1660_; 
v_reuseFailAlloc_1660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1660_, 0, v___x_1657_);
v___x_1659_ = v_reuseFailAlloc_1660_;
goto v_reusejp_1658_;
}
v_reusejp_1658_:
{
return v___x_1659_;
}
}
}
else
{
lean_object* v_a_1662_; lean_object* v___x_1663_; lean_object* v___x_1665_; 
lean_del_object(v___x_1653_);
lean_dec(v_snd_1645_);
v_a_1662_ = lean_ctor_get(v_a_1651_, 0);
lean_inc(v_a_1662_);
lean_dec_ref_known(v_a_1651_, 1);
v___x_1663_ = lean_box(0);
if (v_isShared_1648_ == 0)
{
lean_ctor_set(v___x_1647_, 1, v_a_1662_);
lean_ctor_set(v___x_1647_, 0, v___x_1663_);
v___x_1665_ = v___x_1647_;
goto v_reusejp_1664_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v___x_1663_);
lean_ctor_set(v_reuseFailAlloc_1669_, 1, v_a_1662_);
v___x_1665_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1664_;
}
v_reusejp_1664_:
{
size_t v___x_1666_; size_t v___x_1667_; 
v___x_1666_ = ((size_t)1ULL);
v___x_1667_ = lean_usize_add(v_i_1638_, v___x_1666_);
v_i_1638_ = v___x_1667_;
v_b_1639_ = v___x_1665_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1671_; lean_object* v___x_1673_; uint8_t v_isShared_1674_; uint8_t v_isSharedCheck_1678_; 
lean_del_object(v___x_1647_);
lean_dec(v_snd_1645_);
v_a_1671_ = lean_ctor_get(v___x_1650_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1650_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1673_ = v___x_1650_;
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___x_1650_);
v___x_1673_ = lean_box(0);
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
v_resetjp_1672_:
{
lean_object* v___x_1676_; 
if (v_isShared_1674_ == 0)
{
v___x_1676_ = v___x_1673_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1677_; 
v_reuseFailAlloc_1677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1677_, 0, v_a_1671_);
v___x_1676_ = v_reuseFailAlloc_1677_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
return v___x_1676_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__5___boxed(lean_object* v_init_1681_, lean_object* v___y_1682_, lean_object* v_as_1683_, lean_object* v_sz_1684_, lean_object* v_i_1685_, lean_object* v_b_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_){
_start:
{
size_t v_sz_boxed_1690_; size_t v_i_boxed_1691_; lean_object* v_res_1692_; 
v_sz_boxed_1690_ = lean_unbox_usize(v_sz_1684_);
lean_dec(v_sz_1684_);
v_i_boxed_1691_ = lean_unbox_usize(v_i_1685_);
lean_dec(v_i_1685_);
v_res_1692_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4_spec__5(v_init_1681_, v___y_1682_, v_as_1683_, v_sz_boxed_1690_, v_i_boxed_1691_, v_b_1686_, v___y_1687_, v___y_1688_);
lean_dec(v___y_1688_);
lean_dec_ref(v___y_1687_);
lean_dec_ref(v_as_1683_);
lean_dec_ref(v___y_1682_);
return v_res_1692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4___boxed(lean_object* v_init_1693_, lean_object* v___y_1694_, lean_object* v_n_1695_, lean_object* v_b_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_){
_start:
{
lean_object* v_res_1700_; 
v_res_1700_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4(v_init_1693_, v___y_1694_, v_n_1695_, v_b_1696_, v___y_1697_, v___y_1698_);
lean_dec(v___y_1698_);
lean_dec_ref(v___y_1697_);
lean_dec_ref(v_n_1695_);
lean_dec_ref(v___y_1694_);
return v_res_1700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8(lean_object* v___y_1704_, lean_object* v_as_1705_, size_t v_sz_1706_, size_t v_i_1707_, lean_object* v_b_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_){
_start:
{
uint8_t v___x_1712_; 
v___x_1712_ = lean_usize_dec_lt(v_i_1707_, v_sz_1706_);
if (v___x_1712_ == 0)
{
lean_object* v___x_1713_; 
v___x_1713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1713_, 0, v_b_1708_);
return v___x_1713_;
}
else
{
lean_object* v_a_1714_; lean_object* v___x_1715_; 
lean_dec_ref(v_b_1708_);
v_a_1714_ = lean_array_uget_borrowed(v_as_1705_, v_i_1707_);
lean_inc(v_a_1714_);
v___x_1715_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(v_a_1714_, v___y_1709_, v___y_1710_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v_a_1716_; lean_object* v___x_1717_; size_t v_sz_1718_; size_t v___x_1719_; lean_object* v___x_1720_; 
v_a_1716_ = lean_ctor_get(v___x_1715_, 0);
lean_inc(v_a_1716_);
lean_dec_ref_known(v___x_1715_, 1);
v___x_1717_ = lean_box(0);
v_sz_1718_ = lean_array_size(v_a_1716_);
v___x_1719_ = ((size_t)0ULL);
v___x_1720_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(v___y_1704_, v_a_1716_, v_sz_1718_, v___x_1719_, v___x_1717_, v___y_1709_, v___y_1710_);
lean_dec(v_a_1716_);
if (lean_obj_tag(v___x_1720_) == 0)
{
lean_object* v___x_1721_; size_t v___x_1722_; size_t v___x_1723_; 
lean_dec_ref_known(v___x_1720_, 1);
v___x_1721_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___closed__0));
v___x_1722_ = ((size_t)1ULL);
v___x_1723_ = lean_usize_add(v_i_1707_, v___x_1722_);
v_i_1707_ = v___x_1723_;
v_b_1708_ = v___x_1721_;
goto _start;
}
else
{
lean_object* v_a_1725_; lean_object* v___x_1727_; uint8_t v_isShared_1728_; uint8_t v_isSharedCheck_1732_; 
v_a_1725_ = lean_ctor_get(v___x_1720_, 0);
v_isSharedCheck_1732_ = !lean_is_exclusive(v___x_1720_);
if (v_isSharedCheck_1732_ == 0)
{
v___x_1727_ = v___x_1720_;
v_isShared_1728_ = v_isSharedCheck_1732_;
goto v_resetjp_1726_;
}
else
{
lean_inc(v_a_1725_);
lean_dec(v___x_1720_);
v___x_1727_ = lean_box(0);
v_isShared_1728_ = v_isSharedCheck_1732_;
goto v_resetjp_1726_;
}
v_resetjp_1726_:
{
lean_object* v___x_1730_; 
if (v_isShared_1728_ == 0)
{
v___x_1730_ = v___x_1727_;
goto v_reusejp_1729_;
}
else
{
lean_object* v_reuseFailAlloc_1731_; 
v_reuseFailAlloc_1731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1731_, 0, v_a_1725_);
v___x_1730_ = v_reuseFailAlloc_1731_;
goto v_reusejp_1729_;
}
v_reusejp_1729_:
{
return v___x_1730_;
}
}
}
}
else
{
lean_object* v_a_1733_; lean_object* v___x_1735_; uint8_t v_isShared_1736_; uint8_t v_isSharedCheck_1740_; 
v_a_1733_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_1740_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_1740_ == 0)
{
v___x_1735_ = v___x_1715_;
v_isShared_1736_ = v_isSharedCheck_1740_;
goto v_resetjp_1734_;
}
else
{
lean_inc(v_a_1733_);
lean_dec(v___x_1715_);
v___x_1735_ = lean_box(0);
v_isShared_1736_ = v_isSharedCheck_1740_;
goto v_resetjp_1734_;
}
v_resetjp_1734_:
{
lean_object* v___x_1738_; 
if (v_isShared_1736_ == 0)
{
v___x_1738_ = v___x_1735_;
goto v_reusejp_1737_;
}
else
{
lean_object* v_reuseFailAlloc_1739_; 
v_reuseFailAlloc_1739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1739_, 0, v_a_1733_);
v___x_1738_ = v_reuseFailAlloc_1739_;
goto v_reusejp_1737_;
}
v_reusejp_1737_:
{
return v___x_1738_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___boxed(lean_object* v___y_1741_, lean_object* v_as_1742_, lean_object* v_sz_1743_, lean_object* v_i_1744_, lean_object* v_b_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_){
_start:
{
size_t v_sz_boxed_1749_; size_t v_i_boxed_1750_; lean_object* v_res_1751_; 
v_sz_boxed_1749_ = lean_unbox_usize(v_sz_1743_);
lean_dec(v_sz_1743_);
v_i_boxed_1750_ = lean_unbox_usize(v_i_1744_);
lean_dec(v_i_1744_);
v_res_1751_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8(v___y_1741_, v_as_1742_, v_sz_boxed_1749_, v_i_boxed_1750_, v_b_1745_, v___y_1746_, v___y_1747_);
lean_dec(v___y_1747_);
lean_dec_ref(v___y_1746_);
lean_dec_ref(v_as_1742_);
lean_dec_ref(v___y_1741_);
return v_res_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5(lean_object* v___y_1752_, lean_object* v_as_1753_, size_t v_sz_1754_, size_t v_i_1755_, lean_object* v_b_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_){
_start:
{
uint8_t v___x_1760_; 
v___x_1760_ = lean_usize_dec_lt(v_i_1755_, v_sz_1754_);
if (v___x_1760_ == 0)
{
lean_object* v___x_1761_; 
v___x_1761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1761_, 0, v_b_1756_);
return v___x_1761_;
}
else
{
lean_object* v_a_1762_; lean_object* v___x_1763_; 
lean_dec_ref(v_b_1756_);
v_a_1762_ = lean_array_uget_borrowed(v_as_1753_, v_i_1755_);
lean_inc(v_a_1762_);
v___x_1763_ = lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs(v_a_1762_, v___y_1757_, v___y_1758_);
if (lean_obj_tag(v___x_1763_) == 0)
{
lean_object* v_a_1764_; lean_object* v___x_1765_; size_t v_sz_1766_; size_t v___x_1767_; lean_object* v___x_1768_; 
v_a_1764_ = lean_ctor_get(v___x_1763_, 0);
lean_inc(v_a_1764_);
lean_dec_ref_known(v___x_1763_, 1);
v___x_1765_ = lean_box(0);
v_sz_1766_ = lean_array_size(v_a_1764_);
v___x_1767_ = ((size_t)0ULL);
v___x_1768_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPasses_spec__2(v___y_1752_, v_a_1764_, v_sz_1766_, v___x_1767_, v___x_1765_, v___y_1757_, v___y_1758_);
lean_dec(v_a_1764_);
if (lean_obj_tag(v___x_1768_) == 0)
{
lean_object* v___x_1769_; size_t v___x_1770_; size_t v___x_1771_; lean_object* v___x_1772_; 
lean_dec_ref_known(v___x_1768_, 1);
v___x_1769_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8___closed__0));
v___x_1770_ = ((size_t)1ULL);
v___x_1771_ = lean_usize_add(v_i_1755_, v___x_1770_);
v___x_1772_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5_spec__8(v___y_1752_, v_as_1753_, v_sz_1754_, v___x_1771_, v___x_1769_, v___y_1757_, v___y_1758_);
return v___x_1772_;
}
else
{
lean_object* v_a_1773_; lean_object* v___x_1775_; uint8_t v_isShared_1776_; uint8_t v_isSharedCheck_1780_; 
v_a_1773_ = lean_ctor_get(v___x_1768_, 0);
v_isSharedCheck_1780_ = !lean_is_exclusive(v___x_1768_);
if (v_isSharedCheck_1780_ == 0)
{
v___x_1775_ = v___x_1768_;
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
else
{
lean_inc(v_a_1773_);
lean_dec(v___x_1768_);
v___x_1775_ = lean_box(0);
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
v_resetjp_1774_:
{
lean_object* v___x_1778_; 
if (v_isShared_1776_ == 0)
{
v___x_1778_ = v___x_1775_;
goto v_reusejp_1777_;
}
else
{
lean_object* v_reuseFailAlloc_1779_; 
v_reuseFailAlloc_1779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1779_, 0, v_a_1773_);
v___x_1778_ = v_reuseFailAlloc_1779_;
goto v_reusejp_1777_;
}
v_reusejp_1777_:
{
return v___x_1778_;
}
}
}
}
else
{
lean_object* v_a_1781_; lean_object* v___x_1783_; uint8_t v_isShared_1784_; uint8_t v_isSharedCheck_1788_; 
v_a_1781_ = lean_ctor_get(v___x_1763_, 0);
v_isSharedCheck_1788_ = !lean_is_exclusive(v___x_1763_);
if (v_isSharedCheck_1788_ == 0)
{
v___x_1783_ = v___x_1763_;
v_isShared_1784_ = v_isSharedCheck_1788_;
goto v_resetjp_1782_;
}
else
{
lean_inc(v_a_1781_);
lean_dec(v___x_1763_);
v___x_1783_ = lean_box(0);
v_isShared_1784_ = v_isSharedCheck_1788_;
goto v_resetjp_1782_;
}
v_resetjp_1782_:
{
lean_object* v___x_1786_; 
if (v_isShared_1784_ == 0)
{
v___x_1786_ = v___x_1783_;
goto v_reusejp_1785_;
}
else
{
lean_object* v_reuseFailAlloc_1787_; 
v_reuseFailAlloc_1787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1787_, 0, v_a_1781_);
v___x_1786_ = v_reuseFailAlloc_1787_;
goto v_reusejp_1785_;
}
v_reusejp_1785_:
{
return v___x_1786_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5___boxed(lean_object* v___y_1789_, lean_object* v_as_1790_, lean_object* v_sz_1791_, lean_object* v_i_1792_, lean_object* v_b_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_){
_start:
{
size_t v_sz_boxed_1797_; size_t v_i_boxed_1798_; lean_object* v_res_1799_; 
v_sz_boxed_1797_ = lean_unbox_usize(v_sz_1791_);
lean_dec(v_sz_1791_);
v_i_boxed_1798_ = lean_unbox_usize(v_i_1792_);
lean_dec(v_i_1792_);
v_res_1799_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5(v___y_1789_, v_as_1790_, v_sz_boxed_1797_, v_i_boxed_1798_, v_b_1793_, v___y_1794_, v___y_1795_);
lean_dec(v___y_1795_);
lean_dec_ref(v___y_1794_);
lean_dec_ref(v_as_1790_);
lean_dec_ref(v___y_1789_);
return v_res_1799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3(lean_object* v___y_1800_, lean_object* v_t_1801_, lean_object* v_init_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_){
_start:
{
lean_object* v_root_1806_; lean_object* v_tail_1807_; lean_object* v___x_1808_; 
v_root_1806_ = lean_ctor_get(v_t_1801_, 0);
v_tail_1807_ = lean_ctor_get(v_t_1801_, 1);
v___x_1808_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__4(v_init_1802_, v___y_1800_, v_root_1806_, v_init_1802_, v___y_1803_, v___y_1804_);
if (lean_obj_tag(v___x_1808_) == 0)
{
lean_object* v_a_1809_; lean_object* v___x_1811_; uint8_t v_isShared_1812_; uint8_t v_isSharedCheck_1845_; 
v_a_1809_ = lean_ctor_get(v___x_1808_, 0);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1808_);
if (v_isSharedCheck_1845_ == 0)
{
v___x_1811_ = v___x_1808_;
v_isShared_1812_ = v_isSharedCheck_1845_;
goto v_resetjp_1810_;
}
else
{
lean_inc(v_a_1809_);
lean_dec(v___x_1808_);
v___x_1811_ = lean_box(0);
v_isShared_1812_ = v_isSharedCheck_1845_;
goto v_resetjp_1810_;
}
v_resetjp_1810_:
{
if (lean_obj_tag(v_a_1809_) == 0)
{
lean_object* v_a_1813_; lean_object* v___x_1815_; 
v_a_1813_ = lean_ctor_get(v_a_1809_, 0);
lean_inc(v_a_1813_);
lean_dec_ref_known(v_a_1809_, 1);
if (v_isShared_1812_ == 0)
{
lean_ctor_set(v___x_1811_, 0, v_a_1813_);
v___x_1815_ = v___x_1811_;
goto v_reusejp_1814_;
}
else
{
lean_object* v_reuseFailAlloc_1816_; 
v_reuseFailAlloc_1816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1816_, 0, v_a_1813_);
v___x_1815_ = v_reuseFailAlloc_1816_;
goto v_reusejp_1814_;
}
v_reusejp_1814_:
{
return v___x_1815_;
}
}
else
{
lean_object* v_a_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; size_t v_sz_1820_; size_t v___x_1821_; lean_object* v___x_1822_; 
lean_del_object(v___x_1811_);
v_a_1817_ = lean_ctor_get(v_a_1809_, 0);
lean_inc(v_a_1817_);
lean_dec_ref_known(v_a_1809_, 1);
v___x_1818_ = lean_box(0);
v___x_1819_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1819_, 0, v___x_1818_);
lean_ctor_set(v___x_1819_, 1, v_a_1817_);
v_sz_1820_ = lean_array_size(v_tail_1807_);
v___x_1821_ = ((size_t)0ULL);
v___x_1822_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3_spec__5(v___y_1800_, v_tail_1807_, v_sz_1820_, v___x_1821_, v___x_1819_, v___y_1803_, v___y_1804_);
if (lean_obj_tag(v___x_1822_) == 0)
{
lean_object* v_a_1823_; lean_object* v___x_1825_; uint8_t v_isShared_1826_; uint8_t v_isSharedCheck_1836_; 
v_a_1823_ = lean_ctor_get(v___x_1822_, 0);
v_isSharedCheck_1836_ = !lean_is_exclusive(v___x_1822_);
if (v_isSharedCheck_1836_ == 0)
{
v___x_1825_ = v___x_1822_;
v_isShared_1826_ = v_isSharedCheck_1836_;
goto v_resetjp_1824_;
}
else
{
lean_inc(v_a_1823_);
lean_dec(v___x_1822_);
v___x_1825_ = lean_box(0);
v_isShared_1826_ = v_isSharedCheck_1836_;
goto v_resetjp_1824_;
}
v_resetjp_1824_:
{
lean_object* v_fst_1827_; 
v_fst_1827_ = lean_ctor_get(v_a_1823_, 0);
if (lean_obj_tag(v_fst_1827_) == 0)
{
lean_object* v_snd_1828_; lean_object* v___x_1830_; 
v_snd_1828_ = lean_ctor_get(v_a_1823_, 1);
lean_inc(v_snd_1828_);
lean_dec(v_a_1823_);
if (v_isShared_1826_ == 0)
{
lean_ctor_set(v___x_1825_, 0, v_snd_1828_);
v___x_1830_ = v___x_1825_;
goto v_reusejp_1829_;
}
else
{
lean_object* v_reuseFailAlloc_1831_; 
v_reuseFailAlloc_1831_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1831_, 0, v_snd_1828_);
v___x_1830_ = v_reuseFailAlloc_1831_;
goto v_reusejp_1829_;
}
v_reusejp_1829_:
{
return v___x_1830_;
}
}
else
{
lean_object* v_val_1832_; lean_object* v___x_1834_; 
lean_inc_ref(v_fst_1827_);
lean_dec(v_a_1823_);
v_val_1832_ = lean_ctor_get(v_fst_1827_, 0);
lean_inc(v_val_1832_);
lean_dec_ref_known(v_fst_1827_, 1);
if (v_isShared_1826_ == 0)
{
lean_ctor_set(v___x_1825_, 0, v_val_1832_);
v___x_1834_ = v___x_1825_;
goto v_reusejp_1833_;
}
else
{
lean_object* v_reuseFailAlloc_1835_; 
v_reuseFailAlloc_1835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1835_, 0, v_val_1832_);
v___x_1834_ = v_reuseFailAlloc_1835_;
goto v_reusejp_1833_;
}
v_reusejp_1833_:
{
return v___x_1834_;
}
}
}
}
else
{
lean_object* v_a_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1844_; 
v_a_1837_ = lean_ctor_get(v___x_1822_, 0);
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1822_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1839_ = v___x_1822_;
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_a_1837_);
lean_dec(v___x_1822_);
v___x_1839_ = lean_box(0);
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
v_resetjp_1838_:
{
lean_object* v___x_1842_; 
if (v_isShared_1840_ == 0)
{
v___x_1842_ = v___x_1839_;
goto v_reusejp_1841_;
}
else
{
lean_object* v_reuseFailAlloc_1843_; 
v_reuseFailAlloc_1843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1843_, 0, v_a_1837_);
v___x_1842_ = v_reuseFailAlloc_1843_;
goto v_reusejp_1841_;
}
v_reusejp_1841_:
{
return v___x_1842_;
}
}
}
}
}
}
else
{
lean_object* v_a_1846_; lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_1853_; 
v_a_1846_ = lean_ctor_get(v___x_1808_, 0);
v_isSharedCheck_1853_ = !lean_is_exclusive(v___x_1808_);
if (v_isSharedCheck_1853_ == 0)
{
v___x_1848_ = v___x_1808_;
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
else
{
lean_inc(v_a_1846_);
lean_dec(v___x_1808_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
lean_object* v___x_1851_; 
if (v_isShared_1849_ == 0)
{
v___x_1851_ = v___x_1848_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v_a_1846_);
v___x_1851_ = v_reuseFailAlloc_1852_;
goto v_reusejp_1850_;
}
v_reusejp_1850_:
{
return v___x_1851_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3___boxed(lean_object* v___y_1854_, lean_object* v_t_1855_, lean_object* v_init_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_){
_start:
{
lean_object* v_res_1860_; 
v_res_1860_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3(v___y_1854_, v_t_1855_, v_init_1856_, v___y_1857_, v___y_1858_);
lean_dec(v___y_1858_);
lean_dec_ref(v___y_1857_);
lean_dec_ref(v_t_1855_);
lean_dec_ref(v___y_1854_);
return v_res_1860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPasses(lean_object* v_configs_1861_, lean_object* v_trees_1862_, lean_object* v_a_1863_, lean_object* v_a_1864_){
_start:
{
lean_object* v___x_1866_; lean_object* v_a_1867_; lean_object* v___x_1869_; uint8_t v_isShared_1870_; uint8_t v_isSharedCheck_1901_; 
v___x_1866_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0(v_a_1863_, v_a_1864_);
v_a_1867_ = lean_ctor_get(v___x_1866_, 0);
v_isSharedCheck_1901_ = !lean_is_exclusive(v___x_1866_);
if (v_isSharedCheck_1901_ == 0)
{
v___x_1869_ = v___x_1866_;
v_isShared_1870_ = v_isSharedCheck_1901_;
goto v_resetjp_1868_;
}
else
{
lean_inc(v_a_1867_);
lean_dec(v___x_1866_);
v___x_1869_ = lean_box(0);
v_isShared_1870_ = v_isSharedCheck_1901_;
goto v_resetjp_1868_;
}
v_resetjp_1868_:
{
lean_object* v___y_1872_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; uint8_t v___x_1893_; 
v___x_1890_ = lean_unsigned_to_nat(0u);
v___x_1891_ = lean_array_get_size(v_configs_1861_);
v___x_1892_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn___closed__7_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_));
v___x_1893_ = lean_nat_dec_lt(v___x_1890_, v___x_1891_);
if (v___x_1893_ == 0)
{
lean_dec(v_a_1867_);
v___y_1872_ = v___x_1892_;
goto v___jp_1871_;
}
else
{
uint8_t v___x_1894_; 
v___x_1894_ = lean_nat_dec_le(v___x_1891_, v___x_1891_);
if (v___x_1894_ == 0)
{
if (v___x_1893_ == 0)
{
lean_dec(v_a_1867_);
v___y_1872_ = v___x_1892_;
goto v___jp_1871_;
}
else
{
size_t v___x_1895_; size_t v___x_1896_; lean_object* v___x_1897_; 
v___x_1895_ = ((size_t)0ULL);
v___x_1896_ = lean_usize_of_nat(v___x_1891_);
v___x_1897_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4(v_a_1867_, v_configs_1861_, v___x_1895_, v___x_1896_, v___x_1892_);
lean_dec(v_a_1867_);
v___y_1872_ = v___x_1897_;
goto v___jp_1871_;
}
}
else
{
size_t v___x_1898_; size_t v___x_1899_; lean_object* v___x_1900_; 
v___x_1898_ = ((size_t)0ULL);
v___x_1899_ = lean_usize_of_nat(v___x_1891_);
v___x_1900_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_TacticAnalysis_runPasses_spec__4(v_a_1867_, v_configs_1861_, v___x_1898_, v___x_1899_, v___x_1892_);
lean_dec(v_a_1867_);
v___y_1872_ = v___x_1900_;
goto v___jp_1871_;
}
}
v___jp_1871_:
{
lean_object* v___x_1873_; lean_object* v___x_1874_; uint8_t v___x_1875_; 
v___x_1873_ = lean_array_get_size(v___y_1872_);
v___x_1874_ = lean_unsigned_to_nat(0u);
v___x_1875_ = lean_nat_dec_eq(v___x_1873_, v___x_1874_);
if (v___x_1875_ == 0)
{
lean_object* v___x_1876_; lean_object* v___x_1877_; 
lean_del_object(v___x_1869_);
v___x_1876_ = lean_box(0);
v___x_1877_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_TacticAnalysis_runPasses_spec__3(v___y_1872_, v_trees_1862_, v___x_1876_, v_a_1863_, v_a_1864_);
lean_dec_ref(v___y_1872_);
if (lean_obj_tag(v___x_1877_) == 0)
{
lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1884_; 
v_isSharedCheck_1884_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1884_ == 0)
{
lean_object* v_unused_1885_; 
v_unused_1885_ = lean_ctor_get(v___x_1877_, 0);
lean_dec(v_unused_1885_);
v___x_1879_ = v___x_1877_;
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
else
{
lean_dec(v___x_1877_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1882_; 
if (v_isShared_1880_ == 0)
{
lean_ctor_set(v___x_1879_, 0, v___x_1876_);
v___x_1882_ = v___x_1879_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v___x_1876_);
v___x_1882_ = v_reuseFailAlloc_1883_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
return v___x_1882_;
}
}
}
else
{
return v___x_1877_;
}
}
else
{
lean_object* v___x_1886_; lean_object* v___x_1888_; 
lean_dec_ref(v___y_1872_);
v___x_1886_ = lean_box(0);
if (v_isShared_1870_ == 0)
{
lean_ctor_set(v___x_1869_, 0, v___x_1886_);
v___x_1888_ = v___x_1869_;
goto v_reusejp_1887_;
}
else
{
lean_object* v_reuseFailAlloc_1889_; 
v_reuseFailAlloc_1889_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1889_, 0, v___x_1886_);
v___x_1888_ = v_reuseFailAlloc_1889_;
goto v_reusejp_1887_;
}
v_reusejp_1887_:
{
return v___x_1888_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPasses___boxed(lean_object* v_configs_1902_, lean_object* v_trees_1903_, lean_object* v_a_1904_, lean_object* v_a_1905_, lean_object* v_a_1906_){
_start:
{
lean_object* v_res_1907_; 
v_res_1907_ = lp_mathlib_Mathlib_TacticAnalysis_runPasses(v_configs_1902_, v_trees_1903_, v_a_1904_, v_a_1905_);
lean_dec(v_a_1905_);
lean_dec_ref(v_a_1904_);
lean_dec_ref(v_trees_1903_);
lean_dec_ref(v_configs_1902_);
return v_res_1907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0(lean_object* v_o_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_){
_start:
{
lean_object* v___x_1912_; 
v___x_1912_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___redArg(v_o_1908_, v___y_1910_);
return v___x_1912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0___boxed(lean_object* v_o_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
lean_object* v_res_1917_; 
v_res_1917_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Mathlib_TacticAnalysis_runPasses_spec__0_spec__0(v_o_1913_, v___y_1914_, v___y_1915_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
return v_res_1917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg(lean_object* v___y_1918_){
_start:
{
lean_object* v___x_1920_; lean_object* v_infoState_1921_; lean_object* v_trees_1922_; lean_object* v___x_1923_; 
v___x_1920_ = lean_st_ref_get(v___y_1918_);
v_infoState_1921_ = lean_ctor_get(v___x_1920_, 8);
lean_inc_ref(v_infoState_1921_);
lean_dec(v___x_1920_);
v_trees_1922_ = lean_ctor_get(v_infoState_1921_, 2);
lean_inc_ref(v_trees_1922_);
lean_dec_ref(v_infoState_1921_);
v___x_1923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1923_, 0, v_trees_1922_);
return v___x_1923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg___boxed(lean_object* v___y_1924_, lean_object* v___y_1925_){
_start:
{
lean_object* v_res_1926_; 
v_res_1926_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg(v___y_1924_);
lean_dec(v___y_1924_);
return v_res_1926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0(lean_object* v___y_1927_, lean_object* v___y_1928_){
_start:
{
lean_object* v___x_1930_; 
v___x_1930_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg(v___y_1928_);
return v___x_1930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___boxed(lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_){
_start:
{
lean_object* v_res_1934_; 
v_res_1934_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0(v___y_1931_, v___y_1932_);
lean_dec(v___y_1932_);
lean_dec_ref(v___y_1931_);
return v_res_1934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg(lean_object* v_category_1935_, lean_object* v_opts_1936_, lean_object* v_act_1937_, lean_object* v_decl_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_){
_start:
{
lean_object* v___x_1942_; lean_object* v___x_1943_; 
lean_inc(v___y_1940_);
lean_inc_ref(v___y_1939_);
v___x_1942_ = lean_apply_2(v_act_1937_, v___y_1939_, v___y_1940_);
v___x_1943_ = l_Lean_profileitIOUnsafe___redArg(v_category_1935_, v_opts_1936_, v___x_1942_, v_decl_1938_);
return v___x_1943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg___boxed(lean_object* v_category_1944_, lean_object* v_opts_1945_, lean_object* v_act_1946_, lean_object* v_decl_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_){
_start:
{
lean_object* v_res_1951_; 
v_res_1951_ = lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg(v_category_1944_, v_opts_1945_, v_act_1946_, v_decl_1947_, v___y_1948_, v___y_1949_);
lean_dec(v___y_1949_);
lean_dec_ref(v___y_1948_);
lean_dec_ref(v_opts_1945_);
lean_dec_ref(v_category_1944_);
return v_res_1951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1(lean_object* v_00_u03b1_1952_, lean_object* v_category_1953_, lean_object* v_opts_1954_, lean_object* v_act_1955_, lean_object* v_decl_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_){
_start:
{
lean_object* v___x_1960_; 
v___x_1960_ = lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg(v_category_1953_, v_opts_1954_, v_act_1955_, v_decl_1956_, v___y_1957_, v___y_1958_);
return v___x_1960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___boxed(lean_object* v_00_u03b1_1961_, lean_object* v_category_1962_, lean_object* v_opts_1963_, lean_object* v_act_1964_, lean_object* v_decl_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_){
_start:
{
lean_object* v_res_1969_; 
v_res_1969_ = lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1(v_00_u03b1_1961_, v_category_1962_, v_opts_1963_, v_act_1964_, v_decl_1965_, v___y_1966_, v___y_1967_);
lean_dec(v___y_1967_);
lean_dec_ref(v___y_1966_);
lean_dec_ref(v_opts_1963_);
lean_dec_ref(v_category_1962_);
return v_res_1969_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1970_; 
v___x_1970_ = l_Array_instInhabited(lean_box(0));
return v___x_1970_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; 
v___x_1971_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__0, &lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__0);
v___x_1972_ = lean_box(0);
v___x_1973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1973_, 0, v___x_1972_);
lean_ctor_set(v___x_1973_, 1, v___x_1971_);
return v___x_1973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0(lean_object* v___y_1974_, lean_object* v___y_1975_){
_start:
{
lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v_a_1979_; lean_object* v_env_1980_; lean_object* v___x_1981_; lean_object* v_toEnvExtension_1982_; lean_object* v_asyncMode_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v_snd_1987_; lean_object* v___x_1988_; 
v___x_1977_ = lean_st_ref_get(v___y_1975_);
v___x_1978_ = lp_mathlib_Lean_Elab_getInfoTrees___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__0___redArg(v___y_1975_);
v_a_1979_ = lean_ctor_get(v___x_1978_, 0);
lean_inc(v_a_1979_);
lean_dec_ref(v___x_1978_);
v_env_1980_ = lean_ctor_get(v___x_1977_, 0);
lean_inc_ref(v_env_1980_);
lean_dec(v___x_1977_);
v___x_1981_ = lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysisExt;
v_toEnvExtension_1982_ = lean_ctor_get(v___x_1981_, 0);
v_asyncMode_1983_ = lean_ctor_get(v_toEnvExtension_1982_, 2);
v___x_1984_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__1, &lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___closed__1);
v___x_1985_ = lean_box(0);
v___x_1986_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1984_, v___x_1981_, v_env_1980_, v_asyncMode_1983_, v___x_1985_);
v_snd_1987_ = lean_ctor_get(v___x_1986_, 1);
lean_inc(v_snd_1987_);
lean_dec(v___x_1986_);
v___x_1988_ = lp_mathlib_Mathlib_TacticAnalysis_runPasses(v_snd_1987_, v_a_1979_, v___y_1974_, v___y_1975_);
lean_dec(v_a_1979_);
lean_dec(v_snd_1987_);
return v___x_1988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0___boxed(lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_){
_start:
{
lean_object* v_res_1992_; 
v_res_1992_ = lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__0(v___y_1989_, v___y_1990_);
lean_dec(v___y_1990_);
lean_dec_ref(v___y_1989_);
return v_res_1992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__1(lean_object* v___f_1993_, lean_object* v_stx_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_){
_start:
{
lean_object* v___x_1998_; lean_object* v_messages_1999_; uint8_t v___x_2000_; 
v___x_1998_ = lean_st_ref_get(v___y_1996_);
v_messages_1999_ = lean_ctor_get(v___x_1998_, 1);
lean_inc_ref(v_messages_1999_);
lean_dec(v___x_1998_);
v___x_2000_ = l_Lean_MessageLog_hasErrors(v_messages_1999_);
lean_dec_ref(v_messages_1999_);
if (v___x_2000_ == 0)
{
lean_object* v___x_2001_; lean_object* v_scopes_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v_opts_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; 
v___x_2001_ = lean_st_ref_get(v___y_1996_);
v_scopes_2002_ = lean_ctor_get(v___x_2001_, 2);
lean_inc(v_scopes_2002_);
lean_dec(v___x_2001_);
v___x_2003_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2004_ = l_List_head_x21___redArg(v___x_2003_, v_scopes_2002_);
lean_dec(v_scopes_2002_);
v_opts_2005_ = lean_ctor_get(v___x_2004_, 1);
lean_inc_ref(v_opts_2005_);
lean_dec(v___x_2004_);
v___x_2006_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_));
v___x_2007_ = lean_box(0);
v___x_2008_ = lp_mathlib_Lean_profileitM___at___00Mathlib_TacticAnalysis_tacticAnalysis_spec__1___redArg(v___x_2006_, v_opts_2005_, v___f_1993_, v___x_2007_, v___y_1995_, v___y_1996_);
lean_dec_ref(v_opts_2005_);
return v___x_2008_;
}
else
{
lean_object* v___x_2009_; lean_object* v___x_2010_; 
lean_dec_ref(v___f_1993_);
v___x_2009_ = lean_box(0);
v___x_2010_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2010_, 0, v___x_2009_);
return v___x_2010_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__1___boxed(lean_object* v___f_2011_, lean_object* v_stx_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_, lean_object* v___y_2015_){
_start:
{
lean_object* v_res_2016_; 
v_res_2016_ = lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis___lam__1(v___f_2011_, v_stx_2012_, v___y_2013_, v___y_2014_);
lean_dec(v___y_2014_);
lean_dec_ref(v___y_2013_);
lean_dec(v_stx_2012_);
return v_res_2016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1736486024____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2031_; lean_object* v___x_2032_; 
v___x_2031_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysis));
v___x_2032_ = l_Lean_Elab_Command_addLinter(v___x_2031_);
return v___x_2032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1736486024____hygCtx___hyg_2____boxed(lean_object* v_a_2033_){
_start:
{
lean_object* v_res_2034_; 
v_res_2034_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1736486024____hygCtx___hyg_2_();
return v_res_2034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___redArg(lean_object* v_x_2035_){
_start:
{
switch(lean_obj_tag(v_x_2035_))
{
case 0:
{
lean_object* v___x_2036_; 
v___x_2036_ = lean_unsigned_to_nat(0u);
return v___x_2036_;
}
case 1:
{
lean_object* v___x_2037_; 
v___x_2037_ = lean_unsigned_to_nat(1u);
return v___x_2037_;
}
default: 
{
lean_object* v___x_2038_; 
v___x_2038_ = lean_unsigned_to_nat(2u);
return v___x_2038_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___redArg___boxed(lean_object* v_x_2039_){
_start:
{
lean_object* v_res_2040_; 
v_res_2040_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___redArg(v_x_2039_);
lean_dec(v_x_2039_);
return v_res_2040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx(lean_object* v_ctx_2041_, lean_object* v_x_2042_){
_start:
{
lean_object* v___x_2043_; 
v___x_2043_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___redArg(v_x_2042_);
return v___x_2043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx___boxed(lean_object* v_ctx_2044_, lean_object* v_x_2045_){
_start:
{
lean_object* v_res_2046_; 
v_res_2046_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorIdx(v_ctx_2044_, v_x_2045_);
lean_dec(v_x_2045_);
return v_res_2046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(lean_object* v_t_2047_, lean_object* v_k_2048_){
_start:
{
if (lean_obj_tag(v_t_2047_) == 0)
{
return v_k_2048_;
}
else
{
lean_object* v_context_2049_; lean_object* v___x_2050_; 
v_context_2049_ = lean_ctor_get(v_t_2047_, 0);
lean_inc(v_context_2049_);
lean_dec(v_t_2047_);
v___x_2050_ = lean_apply_1(v_k_2048_, v_context_2049_);
return v___x_2050_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim(lean_object* v_ctx_2051_, lean_object* v_motive_2052_, lean_object* v_ctorIdx_2053_, lean_object* v_t_2054_, lean_object* v_h_2055_, lean_object* v_k_2056_){
_start:
{
lean_object* v___x_2057_; 
v___x_2057_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2054_, v_k_2056_);
return v___x_2057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___boxed(lean_object* v_ctx_2058_, lean_object* v_motive_2059_, lean_object* v_ctorIdx_2060_, lean_object* v_t_2061_, lean_object* v_h_2062_, lean_object* v_k_2063_){
_start:
{
lean_object* v_res_2064_; 
v_res_2064_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim(v_ctx_2058_, v_motive_2059_, v_ctorIdx_2060_, v_t_2061_, v_h_2062_, v_k_2063_);
lean_dec(v_ctorIdx_2060_);
return v_res_2064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_skip_elim___redArg(lean_object* v_t_2065_, lean_object* v_skip_2066_){
_start:
{
lean_object* v___x_2067_; 
v___x_2067_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2065_, v_skip_2066_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_skip_elim(lean_object* v_ctx_2068_, lean_object* v_motive_2069_, lean_object* v_t_2070_, lean_object* v_h_2071_, lean_object* v_skip_2072_){
_start:
{
lean_object* v___x_2073_; 
v___x_2073_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2070_, v_skip_2072_);
return v___x_2073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_continue_elim___redArg(lean_object* v_t_2074_, lean_object* v_continue_2075_){
_start:
{
lean_object* v___x_2076_; 
v___x_2076_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2074_, v_continue_2075_);
return v___x_2076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_continue_elim(lean_object* v_ctx_2077_, lean_object* v_motive_2078_, lean_object* v_t_2079_, lean_object* v_h_2080_, lean_object* v_continue_2081_){
_start:
{
lean_object* v___x_2082_; 
v___x_2082_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2079_, v_continue_2081_);
return v___x_2082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_accept_elim___redArg(lean_object* v_t_2083_, lean_object* v_accept_2084_){
_start:
{
lean_object* v___x_2085_; 
v___x_2085_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2083_, v_accept_2084_);
return v___x_2085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_accept_elim(lean_object* v_ctx_2086_, lean_object* v_motive_2087_, lean_object* v_t_2088_, lean_object* v_h_2089_, lean_object* v_accept_2090_){
_start:
{
lean_object* v___x_2091_; 
v___x_2091_ = lp_mathlib_Mathlib_TacticAnalysis_TriggerCondition_ctorElim___redArg(v_t_2088_, v_accept_2090_);
return v___x_2091_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___redArg(lean_object* v_inst_2092_, lean_object* v_x_2093_, lean_object* v_x_2094_){
_start:
{
switch(lean_obj_tag(v_x_2093_))
{
case 0:
{
lean_dec_ref(v_inst_2092_);
if (lean_obj_tag(v_x_2094_) == 0)
{
uint8_t v___x_2095_; 
v___x_2095_ = 1;
return v___x_2095_;
}
else
{
uint8_t v___x_2096_; 
lean_dec(v_x_2094_);
v___x_2096_ = 0;
return v___x_2096_;
}
}
case 1:
{
if (lean_obj_tag(v_x_2094_) == 1)
{
lean_object* v_context_2097_; lean_object* v_context_2098_; lean_object* v___x_2099_; uint8_t v___x_2100_; 
v_context_2097_ = lean_ctor_get(v_x_2093_, 0);
lean_inc(v_context_2097_);
lean_dec_ref_known(v_x_2093_, 1);
v_context_2098_ = lean_ctor_get(v_x_2094_, 0);
lean_inc(v_context_2098_);
lean_dec_ref_known(v_x_2094_, 1);
v___x_2099_ = lean_apply_2(v_inst_2092_, v_context_2097_, v_context_2098_);
v___x_2100_ = lean_unbox(v___x_2099_);
return v___x_2100_;
}
else
{
uint8_t v___x_2101_; 
lean_dec_ref_known(v_x_2093_, 1);
lean_dec(v_x_2094_);
lean_dec_ref(v_inst_2092_);
v___x_2101_ = 0;
return v___x_2101_;
}
}
default: 
{
if (lean_obj_tag(v_x_2094_) == 2)
{
lean_object* v_context_2102_; lean_object* v_context_2103_; lean_object* v___x_2104_; uint8_t v___x_2105_; 
v_context_2102_ = lean_ctor_get(v_x_2093_, 0);
lean_inc(v_context_2102_);
lean_dec_ref_known(v_x_2093_, 1);
v_context_2103_ = lean_ctor_get(v_x_2094_, 0);
lean_inc(v_context_2103_);
lean_dec_ref_known(v_x_2094_, 1);
v___x_2104_ = lean_apply_2(v_inst_2092_, v_context_2102_, v_context_2103_);
v___x_2105_ = lean_unbox(v___x_2104_);
return v___x_2105_;
}
else
{
uint8_t v___x_2106_; 
lean_dec_ref_known(v_x_2093_, 1);
lean_dec(v_x_2094_);
lean_dec_ref(v_inst_2092_);
v___x_2106_ = 0;
return v___x_2106_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___redArg___boxed(lean_object* v_inst_2107_, lean_object* v_x_2108_, lean_object* v_x_2109_){
_start:
{
uint8_t v_res_2110_; lean_object* v_r_2111_; 
v_res_2110_ = lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___redArg(v_inst_2107_, v_x_2108_, v_x_2109_);
v_r_2111_ = lean_box(v_res_2110_);
return v_r_2111_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq(lean_object* v_ctx_2112_, lean_object* v_inst_2113_, lean_object* v_x_2114_, lean_object* v_x_2115_){
_start:
{
uint8_t v___x_2116_; 
v___x_2116_ = lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___redArg(v_inst_2113_, v_x_2114_, v_x_2115_);
return v___x_2116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___boxed(lean_object* v_ctx_2117_, lean_object* v_inst_2118_, lean_object* v_x_2119_, lean_object* v_x_2120_){
_start:
{
uint8_t v_res_2121_; lean_object* v_r_2122_; 
v_res_2121_ = lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq(v_ctx_2117_, v_inst_2118_, v_x_2119_, v_x_2120_);
v_r_2122_ = lean_box(v_res_2121_);
return v_r_2122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition___redArg(lean_object* v_inst_2123_){
_start:
{
lean_object* v___x_2124_; 
v___x_2124_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___boxed), 4, 2);
lean_closure_set(v___x_2124_, 0, lean_box(0));
lean_closure_set(v___x_2124_, 1, v_inst_2123_);
return v___x_2124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition(lean_object* v_ctx_2125_, lean_object* v_inst_2126_){
_start:
{
lean_object* v___x_2127_; 
v___x_2127_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_TacticAnalysis_instBEqTriggerCondition_beq___boxed), 4, 2);
lean_closure_set(v___x_2127_, 0, lean_box(0));
lean_closure_set(v___x_2127_, 1, v_inst_2126_);
return v___x_2127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg(lean_object* v_x_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_){
_start:
{
lean_object* v___x_2132_; lean_object* v___x_2133_; 
v___x_2132_ = lean_io_get_num_heartbeats();
lean_inc(v___y_2130_);
lean_inc_ref(v___y_2129_);
v___x_2133_ = lean_apply_3(v_x_2128_, v___y_2129_, v___y_2130_, lean_box(0));
if (lean_obj_tag(v___x_2133_) == 0)
{
lean_object* v_a_2134_; lean_object* v___x_2136_; uint8_t v_isShared_2137_; uint8_t v_isSharedCheck_2144_; 
v_a_2134_ = lean_ctor_get(v___x_2133_, 0);
v_isSharedCheck_2144_ = !lean_is_exclusive(v___x_2133_);
if (v_isSharedCheck_2144_ == 0)
{
v___x_2136_ = v___x_2133_;
v_isShared_2137_ = v_isSharedCheck_2144_;
goto v_resetjp_2135_;
}
else
{
lean_inc(v_a_2134_);
lean_dec(v___x_2133_);
v___x_2136_ = lean_box(0);
v_isShared_2137_ = v_isSharedCheck_2144_;
goto v_resetjp_2135_;
}
v_resetjp_2135_:
{
lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2142_; 
v___x_2138_ = lean_io_get_num_heartbeats();
v___x_2139_ = lean_nat_sub(v___x_2138_, v___x_2132_);
lean_dec(v___x_2132_);
lean_dec(v___x_2138_);
v___x_2140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2140_, 0, v_a_2134_);
lean_ctor_set(v___x_2140_, 1, v___x_2139_);
if (v_isShared_2137_ == 0)
{
lean_ctor_set(v___x_2136_, 0, v___x_2140_);
v___x_2142_ = v___x_2136_;
goto v_reusejp_2141_;
}
else
{
lean_object* v_reuseFailAlloc_2143_; 
v_reuseFailAlloc_2143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2143_, 0, v___x_2140_);
v___x_2142_ = v_reuseFailAlloc_2143_;
goto v_reusejp_2141_;
}
v_reusejp_2141_:
{
return v___x_2142_;
}
}
}
else
{
lean_object* v_a_2145_; lean_object* v___x_2147_; uint8_t v_isShared_2148_; uint8_t v_isSharedCheck_2152_; 
lean_dec(v___x_2132_);
v_a_2145_ = lean_ctor_get(v___x_2133_, 0);
v_isSharedCheck_2152_ = !lean_is_exclusive(v___x_2133_);
if (v_isSharedCheck_2152_ == 0)
{
v___x_2147_ = v___x_2133_;
v_isShared_2148_ = v_isSharedCheck_2152_;
goto v_resetjp_2146_;
}
else
{
lean_inc(v_a_2145_);
lean_dec(v___x_2133_);
v___x_2147_ = lean_box(0);
v_isShared_2148_ = v_isSharedCheck_2152_;
goto v_resetjp_2146_;
}
v_resetjp_2146_:
{
lean_object* v___x_2150_; 
if (v_isShared_2148_ == 0)
{
v___x_2150_ = v___x_2147_;
goto v_reusejp_2149_;
}
else
{
lean_object* v_reuseFailAlloc_2151_; 
v_reuseFailAlloc_2151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2151_, 0, v_a_2145_);
v___x_2150_ = v_reuseFailAlloc_2151_;
goto v_reusejp_2149_;
}
v_reusejp_2149_:
{
return v___x_2150_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg___boxed(lean_object* v_x_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_){
_start:
{
lean_object* v_res_2157_; 
v_res_2157_ = lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg(v_x_2153_, v___y_2154_, v___y_2155_);
lean_dec(v___y_2155_);
lean_dec_ref(v___y_2154_);
return v_res_2157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1(lean_object* v_00_u03b1_2158_, lean_object* v_x_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_){
_start:
{
lean_object* v___x_2163_; 
v___x_2163_ = lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg(v_x_2159_, v___y_2160_, v___y_2161_);
return v___x_2163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___boxed(lean_object* v_00_u03b1_2164_, lean_object* v_x_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_){
_start:
{
lean_object* v_res_2169_; 
v_res_2169_ = lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1(v_00_u03b1_2164_, v_x_2165_, v___y_2166_, v___y_2167_);
lean_dec(v___y_2167_);
lean_dec_ref(v___y_2166_);
return v_res_2169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg(lean_object* v___y_2170_){
_start:
{
lean_object* v___x_2172_; lean_object* v_env_2173_; lean_object* v___x_2174_; lean_object* v_mainModule_2175_; lean_object* v___x_2176_; 
v___x_2172_ = lean_st_ref_get(v___y_2170_);
v_env_2173_ = lean_ctor_get(v___x_2172_, 0);
lean_inc_ref(v_env_2173_);
lean_dec(v___x_2172_);
v___x_2174_ = l_Lean_Environment_header(v_env_2173_);
lean_dec_ref(v_env_2173_);
v_mainModule_2175_ = lean_ctor_get(v___x_2174_, 0);
lean_inc(v_mainModule_2175_);
lean_dec_ref(v___x_2174_);
v___x_2176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2176_, 0, v_mainModule_2175_);
return v___x_2176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg___boxed(lean_object* v___y_2177_, lean_object* v___y_2178_){
_start:
{
lean_object* v_res_2179_; 
v_res_2179_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg(v___y_2177_);
lean_dec(v___y_2177_);
return v_res_2179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2(lean_object* v___y_2180_, lean_object* v___y_2181_){
_start:
{
lean_object* v___x_2183_; 
v___x_2183_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg(v___y_2181_);
return v___x_2183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___boxed(lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_){
_start:
{
lean_object* v_res_2187_; 
v_res_2187_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2(v___y_2184_, v___y_2185_);
lean_dec(v___y_2185_);
lean_dec_ref(v___y_2184_);
return v_res_2187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__0(lean_object* v___y_2188_, lean_object* v___y_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_){
_start:
{
lean_object* v___x_2194_; 
v___x_2194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2194_, 0, v___y_2188_);
return v___x_2194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__0___boxed(lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_){
_start:
{
lean_object* v_res_2201_; 
v_res_2201_ = lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__0(v___y_2195_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_);
lean_dec(v___y_2199_);
lean_dec_ref(v___y_2198_);
lean_dec(v___y_2197_);
lean_dec_ref(v___y_2196_);
return v_res_2201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__1(lean_object* v_goalsBefore_2202_, lean_object* v_____r_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_){
_start:
{
lean_object* v___x_2207_; lean_object* v___x_2208_; 
v___x_2207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2207_, 0, v_goalsBefore_2202_);
v___x_2208_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2208_, 0, v___x_2207_);
return v___x_2208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__1___boxed(lean_object* v_goalsBefore_2209_, lean_object* v_____r_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_, lean_object* v___y_2213_){
_start:
{
lean_object* v_res_2214_; 
v_res_2214_ = lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__1(v_goalsBefore_2209_, v_____r_2210_, v___y_2211_, v___y_2212_);
lean_dec(v___y_2212_);
lean_dec_ref(v___y_2211_);
return v_res_2214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg(lean_object* v_msgData_2215_, lean_object* v___y_2216_){
_start:
{
lean_object* v___x_2218_; lean_object* v_env_2219_; lean_object* v___x_2220_; lean_object* v_scopes_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v_opts_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; 
v___x_2218_ = lean_st_ref_get(v___y_2216_);
v_env_2219_ = lean_ctor_get(v___x_2218_, 0);
lean_inc_ref(v_env_2219_);
lean_dec(v___x_2218_);
v___x_2220_ = lean_st_ref_get(v___y_2216_);
v_scopes_2221_ = lean_ctor_get(v___x_2220_, 2);
lean_inc(v_scopes_2221_);
lean_dec(v___x_2220_);
v___x_2222_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2223_ = l_List_head_x21___redArg(v___x_2222_, v_scopes_2221_);
lean_dec(v_scopes_2221_);
v_opts_2224_ = lean_ctor_get(v___x_2223_, 1);
lean_inc_ref(v_opts_2224_);
lean_dec(v___x_2223_);
v___x_2225_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__2);
v___x_2226_ = lean_unsigned_to_nat(32u);
v___x_2227_ = lean_mk_empty_array_with_capacity(v___x_2226_);
lean_dec_ref(v___x_2227_);
v___x_2228_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2__spec__0_spec__0___closed__5);
v___x_2229_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2229_, 0, v_env_2219_);
lean_ctor_set(v___x_2229_, 1, v___x_2225_);
lean_ctor_set(v___x_2229_, 2, v___x_2228_);
lean_ctor_set(v___x_2229_, 3, v_opts_2224_);
v___x_2230_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2230_, 0, v___x_2229_);
lean_ctor_set(v___x_2230_, 1, v_msgData_2215_);
v___x_2231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2231_, 0, v___x_2230_);
return v___x_2231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_){
_start:
{
lean_object* v_res_2235_; 
v_res_2235_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg(v_msgData_2232_, v___y_2233_);
lean_dec(v___y_2233_);
return v_res_2235_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0(uint8_t v___y_2237_, uint8_t v_suppressElabErrors_2238_, lean_object* v_x_2239_){
_start:
{
if (lean_obj_tag(v_x_2239_) == 1)
{
lean_object* v_pre_2240_; 
v_pre_2240_ = lean_ctor_get(v_x_2239_, 0);
if (lean_obj_tag(v_pre_2240_) == 0)
{
lean_object* v_str_2241_; lean_object* v___x_2242_; uint8_t v___x_2243_; 
v_str_2241_ = lean_ctor_get(v_x_2239_, 1);
v___x_2242_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___closed__0));
v___x_2243_ = lean_string_dec_eq(v_str_2241_, v___x_2242_);
if (v___x_2243_ == 0)
{
return v___y_2237_;
}
else
{
return v_suppressElabErrors_2238_;
}
}
else
{
return v___y_2237_;
}
}
else
{
return v___y_2237_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___boxed(lean_object* v___y_2244_, lean_object* v_suppressElabErrors_2245_, lean_object* v_x_2246_){
_start:
{
uint8_t v___y_6697__boxed_2247_; uint8_t v_suppressElabErrors_boxed_2248_; uint8_t v_res_2249_; lean_object* v_r_2250_; 
v___y_6697__boxed_2247_ = lean_unbox(v___y_2244_);
v_suppressElabErrors_boxed_2248_ = lean_unbox(v_suppressElabErrors_2245_);
v_res_2249_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0(v___y_6697__boxed_2247_, v_suppressElabErrors_boxed_2248_, v_x_2246_);
lean_dec(v_x_2246_);
v_r_2250_ = lean_box(v_res_2249_);
return v_r_2250_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__5(lean_object* v_opts_2251_, lean_object* v_opt_2252_){
_start:
{
lean_object* v_name_2253_; lean_object* v_defValue_2254_; lean_object* v_map_2255_; lean_object* v___x_2256_; 
v_name_2253_ = lean_ctor_get(v_opt_2252_, 0);
v_defValue_2254_ = lean_ctor_get(v_opt_2252_, 1);
v_map_2255_ = lean_ctor_get(v_opts_2251_, 0);
v___x_2256_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2255_, v_name_2253_);
if (lean_obj_tag(v___x_2256_) == 0)
{
uint8_t v___x_2257_; 
v___x_2257_ = lean_unbox(v_defValue_2254_);
return v___x_2257_;
}
else
{
lean_object* v_val_2258_; 
v_val_2258_ = lean_ctor_get(v___x_2256_, 0);
lean_inc(v_val_2258_);
lean_dec_ref_known(v___x_2256_, 1);
if (lean_obj_tag(v_val_2258_) == 1)
{
uint8_t v_v_2259_; 
v_v_2259_ = lean_ctor_get_uint8(v_val_2258_, 0);
lean_dec_ref_known(v_val_2258_, 0);
return v_v_2259_;
}
else
{
uint8_t v___x_2260_; 
lean_dec(v_val_2258_);
v___x_2260_ = lean_unbox(v_defValue_2254_);
return v___x_2260_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__5___boxed(lean_object* v_opts_2261_, lean_object* v_opt_2262_){
_start:
{
uint8_t v_res_2263_; lean_object* v_r_2264_; 
v_res_2263_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__5(v_opts_2261_, v_opt_2262_);
lean_dec_ref(v_opt_2262_);
lean_dec_ref(v_opts_2261_);
v_r_2264_ = lean_box(v_res_2263_);
return v_r_2264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3(lean_object* v_ref_2266_, lean_object* v_msgData_2267_, uint8_t v_severity_2268_, uint8_t v_isSilent_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_){
_start:
{
uint8_t v___y_2274_; lean_object* v___y_2275_; lean_object* v___y_2276_; lean_object* v___y_2277_; lean_object* v___y_2278_; lean_object* v___y_2279_; uint8_t v___y_2280_; lean_object* v___y_2281_; uint8_t v___y_2338_; uint8_t v___y_2339_; lean_object* v___y_2340_; uint8_t v___y_2341_; lean_object* v___y_2342_; uint8_t v___y_2366_; uint8_t v___y_2367_; lean_object* v___y_2368_; uint8_t v___y_2369_; lean_object* v___y_2370_; uint8_t v___y_2374_; uint8_t v___y_2375_; uint8_t v___y_2376_; uint8_t v___x_2391_; uint8_t v___y_2393_; uint8_t v___y_2394_; uint8_t v___y_2395_; uint8_t v___y_2397_; uint8_t v___x_2409_; 
v___x_2391_ = 2;
v___x_2409_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2268_, v___x_2391_);
if (v___x_2409_ == 0)
{
v___y_2397_ = v___x_2409_;
goto v___jp_2396_;
}
else
{
uint8_t v___x_2410_; 
lean_inc_ref(v_msgData_2267_);
v___x_2410_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2267_);
v___y_2397_ = v___x_2410_;
goto v___jp_2396_;
}
v___jp_2273_:
{
lean_object* v___x_2282_; 
v___x_2282_ = l_Lean_Elab_Command_getScope___redArg(v___y_2281_);
if (lean_obj_tag(v___x_2282_) == 0)
{
lean_object* v_a_2283_; lean_object* v___x_2284_; 
v_a_2283_ = lean_ctor_get(v___x_2282_, 0);
lean_inc(v_a_2283_);
lean_dec_ref_known(v___x_2282_, 1);
v___x_2284_ = l_Lean_Elab_Command_getScope___redArg(v___y_2281_);
if (lean_obj_tag(v___x_2284_) == 0)
{
lean_object* v_a_2285_; lean_object* v___x_2287_; uint8_t v_isShared_2288_; uint8_t v_isSharedCheck_2320_; 
v_a_2285_ = lean_ctor_get(v___x_2284_, 0);
v_isSharedCheck_2320_ = !lean_is_exclusive(v___x_2284_);
if (v_isSharedCheck_2320_ == 0)
{
v___x_2287_ = v___x_2284_;
v_isShared_2288_ = v_isSharedCheck_2320_;
goto v_resetjp_2286_;
}
else
{
lean_inc(v_a_2285_);
lean_dec(v___x_2284_);
v___x_2287_ = lean_box(0);
v_isShared_2288_ = v_isSharedCheck_2320_;
goto v_resetjp_2286_;
}
v_resetjp_2286_:
{
lean_object* v___x_2289_; lean_object* v_currNamespace_2290_; lean_object* v_openDecls_2291_; lean_object* v_env_2292_; lean_object* v_messages_2293_; lean_object* v_scopes_2294_; lean_object* v_usedQuotCtxts_2295_; lean_object* v_nextMacroScope_2296_; lean_object* v_maxRecDepth_2297_; lean_object* v_ngen_2298_; lean_object* v_auxDeclNGen_2299_; lean_object* v_infoState_2300_; lean_object* v_traceState_2301_; lean_object* v_snapshotTasks_2302_; lean_object* v_prevLinterStates_2303_; lean_object* v___x_2305_; uint8_t v_isShared_2306_; uint8_t v_isSharedCheck_2319_; 
v___x_2289_ = lean_st_ref_take(v___y_2281_);
v_currNamespace_2290_ = lean_ctor_get(v_a_2283_, 2);
lean_inc(v_currNamespace_2290_);
lean_dec(v_a_2283_);
v_openDecls_2291_ = lean_ctor_get(v_a_2285_, 3);
lean_inc(v_openDecls_2291_);
lean_dec(v_a_2285_);
v_env_2292_ = lean_ctor_get(v___x_2289_, 0);
v_messages_2293_ = lean_ctor_get(v___x_2289_, 1);
v_scopes_2294_ = lean_ctor_get(v___x_2289_, 2);
v_usedQuotCtxts_2295_ = lean_ctor_get(v___x_2289_, 3);
v_nextMacroScope_2296_ = lean_ctor_get(v___x_2289_, 4);
v_maxRecDepth_2297_ = lean_ctor_get(v___x_2289_, 5);
v_ngen_2298_ = lean_ctor_get(v___x_2289_, 6);
v_auxDeclNGen_2299_ = lean_ctor_get(v___x_2289_, 7);
v_infoState_2300_ = lean_ctor_get(v___x_2289_, 8);
v_traceState_2301_ = lean_ctor_get(v___x_2289_, 9);
v_snapshotTasks_2302_ = lean_ctor_get(v___x_2289_, 10);
v_prevLinterStates_2303_ = lean_ctor_get(v___x_2289_, 11);
v_isSharedCheck_2319_ = !lean_is_exclusive(v___x_2289_);
if (v_isSharedCheck_2319_ == 0)
{
v___x_2305_ = v___x_2289_;
v_isShared_2306_ = v_isSharedCheck_2319_;
goto v_resetjp_2304_;
}
else
{
lean_inc(v_prevLinterStates_2303_);
lean_inc(v_snapshotTasks_2302_);
lean_inc(v_traceState_2301_);
lean_inc(v_infoState_2300_);
lean_inc(v_auxDeclNGen_2299_);
lean_inc(v_ngen_2298_);
lean_inc(v_maxRecDepth_2297_);
lean_inc(v_nextMacroScope_2296_);
lean_inc(v_usedQuotCtxts_2295_);
lean_inc(v_scopes_2294_);
lean_inc(v_messages_2293_);
lean_inc(v_env_2292_);
lean_dec(v___x_2289_);
v___x_2305_ = lean_box(0);
v_isShared_2306_ = v_isSharedCheck_2319_;
goto v_resetjp_2304_;
}
v_resetjp_2304_:
{
lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2312_; 
v___x_2307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2307_, 0, v_currNamespace_2290_);
lean_ctor_set(v___x_2307_, 1, v_openDecls_2291_);
v___x_2308_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2308_, 0, v___x_2307_);
lean_ctor_set(v___x_2308_, 1, v___y_2276_);
lean_inc_ref(v___y_2277_);
lean_inc_ref(v___y_2278_);
v___x_2309_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2309_, 0, v___y_2278_);
lean_ctor_set(v___x_2309_, 1, v___y_2279_);
lean_ctor_set(v___x_2309_, 2, v___y_2275_);
lean_ctor_set(v___x_2309_, 3, v___y_2277_);
lean_ctor_set(v___x_2309_, 4, v___x_2308_);
lean_ctor_set_uint8(v___x_2309_, sizeof(void*)*5, v___y_2280_);
lean_ctor_set_uint8(v___x_2309_, sizeof(void*)*5 + 1, v___y_2274_);
lean_ctor_set_uint8(v___x_2309_, sizeof(void*)*5 + 2, v_isSilent_2269_);
v___x_2310_ = l_Lean_MessageLog_add(v___x_2309_, v_messages_2293_);
if (v_isShared_2306_ == 0)
{
lean_ctor_set(v___x_2305_, 1, v___x_2310_);
v___x_2312_ = v___x_2305_;
goto v_reusejp_2311_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v_env_2292_);
lean_ctor_set(v_reuseFailAlloc_2318_, 1, v___x_2310_);
lean_ctor_set(v_reuseFailAlloc_2318_, 2, v_scopes_2294_);
lean_ctor_set(v_reuseFailAlloc_2318_, 3, v_usedQuotCtxts_2295_);
lean_ctor_set(v_reuseFailAlloc_2318_, 4, v_nextMacroScope_2296_);
lean_ctor_set(v_reuseFailAlloc_2318_, 5, v_maxRecDepth_2297_);
lean_ctor_set(v_reuseFailAlloc_2318_, 6, v_ngen_2298_);
lean_ctor_set(v_reuseFailAlloc_2318_, 7, v_auxDeclNGen_2299_);
lean_ctor_set(v_reuseFailAlloc_2318_, 8, v_infoState_2300_);
lean_ctor_set(v_reuseFailAlloc_2318_, 9, v_traceState_2301_);
lean_ctor_set(v_reuseFailAlloc_2318_, 10, v_snapshotTasks_2302_);
lean_ctor_set(v_reuseFailAlloc_2318_, 11, v_prevLinterStates_2303_);
v___x_2312_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2311_;
}
v_reusejp_2311_:
{
lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2316_; 
v___x_2313_ = lean_st_ref_set(v___y_2281_, v___x_2312_);
v___x_2314_ = lean_box(0);
if (v_isShared_2288_ == 0)
{
lean_ctor_set(v___x_2287_, 0, v___x_2314_);
v___x_2316_ = v___x_2287_;
goto v_reusejp_2315_;
}
else
{
lean_object* v_reuseFailAlloc_2317_; 
v_reuseFailAlloc_2317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2317_, 0, v___x_2314_);
v___x_2316_ = v_reuseFailAlloc_2317_;
goto v_reusejp_2315_;
}
v_reusejp_2315_:
{
return v___x_2316_;
}
}
}
}
}
else
{
lean_object* v_a_2321_; lean_object* v___x_2323_; uint8_t v_isShared_2324_; uint8_t v_isSharedCheck_2328_; 
lean_dec(v_a_2283_);
lean_dec_ref(v___y_2279_);
lean_dec_ref(v___y_2276_);
lean_dec(v___y_2275_);
v_a_2321_ = lean_ctor_get(v___x_2284_, 0);
v_isSharedCheck_2328_ = !lean_is_exclusive(v___x_2284_);
if (v_isSharedCheck_2328_ == 0)
{
v___x_2323_ = v___x_2284_;
v_isShared_2324_ = v_isSharedCheck_2328_;
goto v_resetjp_2322_;
}
else
{
lean_inc(v_a_2321_);
lean_dec(v___x_2284_);
v___x_2323_ = lean_box(0);
v_isShared_2324_ = v_isSharedCheck_2328_;
goto v_resetjp_2322_;
}
v_resetjp_2322_:
{
lean_object* v___x_2326_; 
if (v_isShared_2324_ == 0)
{
v___x_2326_ = v___x_2323_;
goto v_reusejp_2325_;
}
else
{
lean_object* v_reuseFailAlloc_2327_; 
v_reuseFailAlloc_2327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2327_, 0, v_a_2321_);
v___x_2326_ = v_reuseFailAlloc_2327_;
goto v_reusejp_2325_;
}
v_reusejp_2325_:
{
return v___x_2326_;
}
}
}
}
else
{
lean_object* v_a_2329_; lean_object* v___x_2331_; uint8_t v_isShared_2332_; uint8_t v_isSharedCheck_2336_; 
lean_dec_ref(v___y_2279_);
lean_dec_ref(v___y_2276_);
lean_dec(v___y_2275_);
v_a_2329_ = lean_ctor_get(v___x_2282_, 0);
v_isSharedCheck_2336_ = !lean_is_exclusive(v___x_2282_);
if (v_isSharedCheck_2336_ == 0)
{
v___x_2331_ = v___x_2282_;
v_isShared_2332_ = v_isSharedCheck_2336_;
goto v_resetjp_2330_;
}
else
{
lean_inc(v_a_2329_);
lean_dec(v___x_2282_);
v___x_2331_ = lean_box(0);
v_isShared_2332_ = v_isSharedCheck_2336_;
goto v_resetjp_2330_;
}
v_resetjp_2330_:
{
lean_object* v___x_2334_; 
if (v_isShared_2332_ == 0)
{
v___x_2334_ = v___x_2331_;
goto v_reusejp_2333_;
}
else
{
lean_object* v_reuseFailAlloc_2335_; 
v_reuseFailAlloc_2335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2335_, 0, v_a_2329_);
v___x_2334_ = v_reuseFailAlloc_2335_;
goto v_reusejp_2333_;
}
v_reusejp_2333_:
{
return v___x_2334_;
}
}
}
}
v___jp_2337_:
{
lean_object* v_fileName_2343_; lean_object* v_fileMap_2344_; uint8_t v_suppressElabErrors_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v_a_2348_; lean_object* v___x_2350_; uint8_t v_isShared_2351_; uint8_t v_isSharedCheck_2364_; 
v_fileName_2343_ = lean_ctor_get(v___y_2270_, 0);
v_fileMap_2344_ = lean_ctor_get(v___y_2270_, 1);
v_suppressElabErrors_2345_ = lean_ctor_get_uint8(v___y_2270_, sizeof(void*)*10);
v___x_2346_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2267_);
v___x_2347_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg(v___x_2346_, v___y_2271_);
v_a_2348_ = lean_ctor_get(v___x_2347_, 0);
v_isSharedCheck_2364_ = !lean_is_exclusive(v___x_2347_);
if (v_isSharedCheck_2364_ == 0)
{
v___x_2350_ = v___x_2347_;
v_isShared_2351_ = v_isSharedCheck_2364_;
goto v_resetjp_2349_;
}
else
{
lean_inc(v_a_2348_);
lean_dec(v___x_2347_);
v___x_2350_ = lean_box(0);
v_isShared_2351_ = v_isSharedCheck_2364_;
goto v_resetjp_2349_;
}
v_resetjp_2349_:
{
lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; 
lean_inc_ref_n(v_fileMap_2344_, 2);
v___x_2352_ = l_Lean_FileMap_toPosition(v_fileMap_2344_, v___y_2340_);
lean_dec(v___y_2340_);
v___x_2353_ = l_Lean_FileMap_toPosition(v_fileMap_2344_, v___y_2342_);
lean_dec(v___y_2342_);
v___x_2354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2354_, 0, v___x_2353_);
v___x_2355_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___closed__0));
if (v_suppressElabErrors_2345_ == 0)
{
lean_del_object(v___x_2350_);
v___y_2274_ = v___y_2339_;
v___y_2275_ = v___x_2354_;
v___y_2276_ = v_a_2348_;
v___y_2277_ = v___x_2355_;
v___y_2278_ = v_fileName_2343_;
v___y_2279_ = v___x_2352_;
v___y_2280_ = v___y_2341_;
v___y_2281_ = v___y_2271_;
goto v___jp_2273_;
}
else
{
lean_object* v___x_2356_; lean_object* v___x_2357_; lean_object* v___f_2358_; uint8_t v___x_2359_; 
v___x_2356_ = lean_box(v___y_2338_);
v___x_2357_ = lean_box(v_suppressElabErrors_2345_);
v___f_2358_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2358_, 0, v___x_2356_);
lean_closure_set(v___f_2358_, 1, v___x_2357_);
lean_inc(v_a_2348_);
v___x_2359_ = l_Lean_MessageData_hasTag(v___f_2358_, v_a_2348_);
if (v___x_2359_ == 0)
{
lean_object* v___x_2360_; lean_object* v___x_2362_; 
lean_dec_ref_known(v___x_2354_, 1);
lean_dec_ref(v___x_2352_);
lean_dec(v_a_2348_);
v___x_2360_ = lean_box(0);
if (v_isShared_2351_ == 0)
{
lean_ctor_set(v___x_2350_, 0, v___x_2360_);
v___x_2362_ = v___x_2350_;
goto v_reusejp_2361_;
}
else
{
lean_object* v_reuseFailAlloc_2363_; 
v_reuseFailAlloc_2363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2363_, 0, v___x_2360_);
v___x_2362_ = v_reuseFailAlloc_2363_;
goto v_reusejp_2361_;
}
v_reusejp_2361_:
{
return v___x_2362_;
}
}
else
{
lean_del_object(v___x_2350_);
v___y_2274_ = v___y_2339_;
v___y_2275_ = v___x_2354_;
v___y_2276_ = v_a_2348_;
v___y_2277_ = v___x_2355_;
v___y_2278_ = v_fileName_2343_;
v___y_2279_ = v___x_2352_;
v___y_2280_ = v___y_2341_;
v___y_2281_ = v___y_2271_;
goto v___jp_2273_;
}
}
}
}
v___jp_2365_:
{
lean_object* v___x_2371_; 
v___x_2371_ = l_Lean_Syntax_getTailPos_x3f(v___y_2368_, v___y_2369_);
lean_dec(v___y_2368_);
if (lean_obj_tag(v___x_2371_) == 0)
{
lean_inc(v___y_2370_);
v___y_2338_ = v___y_2366_;
v___y_2339_ = v___y_2367_;
v___y_2340_ = v___y_2370_;
v___y_2341_ = v___y_2369_;
v___y_2342_ = v___y_2370_;
goto v___jp_2337_;
}
else
{
lean_object* v_val_2372_; 
v_val_2372_ = lean_ctor_get(v___x_2371_, 0);
lean_inc(v_val_2372_);
lean_dec_ref_known(v___x_2371_, 1);
v___y_2338_ = v___y_2366_;
v___y_2339_ = v___y_2367_;
v___y_2340_ = v___y_2370_;
v___y_2341_ = v___y_2369_;
v___y_2342_ = v_val_2372_;
goto v___jp_2337_;
}
}
v___jp_2373_:
{
lean_object* v___x_2377_; 
v___x_2377_ = l_Lean_Elab_Command_getRef___redArg(v___y_2270_);
if (lean_obj_tag(v___x_2377_) == 0)
{
lean_object* v_a_2378_; lean_object* v_ref_2379_; lean_object* v___x_2380_; 
v_a_2378_ = lean_ctor_get(v___x_2377_, 0);
lean_inc(v_a_2378_);
lean_dec_ref_known(v___x_2377_, 1);
v_ref_2379_ = l_Lean_replaceRef(v_ref_2266_, v_a_2378_);
lean_dec(v_a_2378_);
v___x_2380_ = l_Lean_Syntax_getPos_x3f(v_ref_2379_, v___y_2375_);
if (lean_obj_tag(v___x_2380_) == 0)
{
lean_object* v___x_2381_; 
v___x_2381_ = lean_unsigned_to_nat(0u);
v___y_2366_ = v___y_2374_;
v___y_2367_ = v___y_2376_;
v___y_2368_ = v_ref_2379_;
v___y_2369_ = v___y_2375_;
v___y_2370_ = v___x_2381_;
goto v___jp_2365_;
}
else
{
lean_object* v_val_2382_; 
v_val_2382_ = lean_ctor_get(v___x_2380_, 0);
lean_inc(v_val_2382_);
lean_dec_ref_known(v___x_2380_, 1);
v___y_2366_ = v___y_2374_;
v___y_2367_ = v___y_2376_;
v___y_2368_ = v_ref_2379_;
v___y_2369_ = v___y_2375_;
v___y_2370_ = v_val_2382_;
goto v___jp_2365_;
}
}
else
{
lean_object* v_a_2383_; lean_object* v___x_2385_; uint8_t v_isShared_2386_; uint8_t v_isSharedCheck_2390_; 
lean_dec_ref(v_msgData_2267_);
v_a_2383_ = lean_ctor_get(v___x_2377_, 0);
v_isSharedCheck_2390_ = !lean_is_exclusive(v___x_2377_);
if (v_isSharedCheck_2390_ == 0)
{
v___x_2385_ = v___x_2377_;
v_isShared_2386_ = v_isSharedCheck_2390_;
goto v_resetjp_2384_;
}
else
{
lean_inc(v_a_2383_);
lean_dec(v___x_2377_);
v___x_2385_ = lean_box(0);
v_isShared_2386_ = v_isSharedCheck_2390_;
goto v_resetjp_2384_;
}
v_resetjp_2384_:
{
lean_object* v___x_2388_; 
if (v_isShared_2386_ == 0)
{
v___x_2388_ = v___x_2385_;
goto v_reusejp_2387_;
}
else
{
lean_object* v_reuseFailAlloc_2389_; 
v_reuseFailAlloc_2389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2389_, 0, v_a_2383_);
v___x_2388_ = v_reuseFailAlloc_2389_;
goto v_reusejp_2387_;
}
v_reusejp_2387_:
{
return v___x_2388_;
}
}
}
}
v___jp_2392_:
{
if (v___y_2395_ == 0)
{
v___y_2374_ = v___y_2393_;
v___y_2375_ = v___y_2394_;
v___y_2376_ = v_severity_2268_;
goto v___jp_2373_;
}
else
{
v___y_2374_ = v___y_2393_;
v___y_2375_ = v___y_2394_;
v___y_2376_ = v___x_2391_;
goto v___jp_2373_;
}
}
v___jp_2396_:
{
if (v___y_2397_ == 0)
{
lean_object* v___x_2398_; lean_object* v_scopes_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; lean_object* v_opts_2402_; uint8_t v___x_2403_; uint8_t v___x_2404_; 
v___x_2398_ = lean_st_ref_get(v___y_2271_);
v_scopes_2399_ = lean_ctor_get(v___x_2398_, 2);
lean_inc(v_scopes_2399_);
lean_dec(v___x_2398_);
v___x_2400_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2401_ = l_List_head_x21___redArg(v___x_2400_, v_scopes_2399_);
lean_dec(v_scopes_2399_);
v_opts_2402_ = lean_ctor_get(v___x_2401_, 1);
lean_inc_ref(v_opts_2402_);
lean_dec(v___x_2401_);
v___x_2403_ = 1;
v___x_2404_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2268_, v___x_2403_);
if (v___x_2404_ == 0)
{
lean_dec_ref(v_opts_2402_);
v___y_2393_ = v___y_2397_;
v___y_2394_ = v___y_2397_;
v___y_2395_ = v___x_2404_;
goto v___jp_2392_;
}
else
{
lean_object* v___x_2405_; uint8_t v___x_2406_; 
v___x_2405_ = l_Lean_warningAsError;
v___x_2406_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__5(v_opts_2402_, v___x_2405_);
lean_dec_ref(v_opts_2402_);
v___y_2393_ = v___y_2397_;
v___y_2394_ = v___y_2397_;
v___y_2395_ = v___x_2406_;
goto v___jp_2392_;
}
}
else
{
lean_object* v___x_2407_; lean_object* v___x_2408_; 
lean_dec_ref(v_msgData_2267_);
v___x_2407_ = lean_box(0);
v___x_2408_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2408_, 0, v___x_2407_);
return v___x_2408_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3___boxed(lean_object* v_ref_2411_, lean_object* v_msgData_2412_, lean_object* v_severity_2413_, lean_object* v_isSilent_2414_, lean_object* v___y_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_){
_start:
{
uint8_t v_severity_boxed_2418_; uint8_t v_isSilent_boxed_2419_; lean_object* v_res_2420_; 
v_severity_boxed_2418_ = lean_unbox(v_severity_2413_);
v_isSilent_boxed_2419_ = lean_unbox(v_isSilent_2414_);
v_res_2420_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3(v_ref_2411_, v_msgData_2412_, v_severity_boxed_2418_, v_isSilent_boxed_2419_, v___y_2415_, v___y_2416_);
lean_dec(v___y_2416_);
lean_dec_ref(v___y_2415_);
lean_dec(v_ref_2411_);
return v_res_2420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0(lean_object* v_msgData_2421_, uint8_t v_severity_2422_, uint8_t v_isSilent_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v___x_2427_; 
v___x_2427_ = l_Lean_Elab_Command_getRef___redArg(v___y_2424_);
if (lean_obj_tag(v___x_2427_) == 0)
{
lean_object* v_a_2428_; lean_object* v___x_2429_; 
v_a_2428_ = lean_ctor_get(v___x_2427_, 0);
lean_inc(v_a_2428_);
lean_dec_ref_known(v___x_2427_, 1);
v___x_2429_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3(v_a_2428_, v_msgData_2421_, v_severity_2422_, v_isSilent_2423_, v___y_2424_, v___y_2425_);
lean_dec(v_a_2428_);
return v___x_2429_;
}
else
{
lean_object* v_a_2430_; lean_object* v___x_2432_; uint8_t v_isShared_2433_; uint8_t v_isSharedCheck_2437_; 
lean_dec_ref(v_msgData_2421_);
v_a_2430_ = lean_ctor_get(v___x_2427_, 0);
v_isSharedCheck_2437_ = !lean_is_exclusive(v___x_2427_);
if (v_isSharedCheck_2437_ == 0)
{
v___x_2432_ = v___x_2427_;
v_isShared_2433_ = v_isSharedCheck_2437_;
goto v_resetjp_2431_;
}
else
{
lean_inc(v_a_2430_);
lean_dec(v___x_2427_);
v___x_2432_ = lean_box(0);
v_isShared_2433_ = v_isSharedCheck_2437_;
goto v_resetjp_2431_;
}
v_resetjp_2431_:
{
lean_object* v___x_2435_; 
if (v_isShared_2433_ == 0)
{
v___x_2435_ = v___x_2432_;
goto v_reusejp_2434_;
}
else
{
lean_object* v_reuseFailAlloc_2436_; 
v_reuseFailAlloc_2436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2436_, 0, v_a_2430_);
v___x_2435_ = v_reuseFailAlloc_2436_;
goto v_reusejp_2434_;
}
v_reusejp_2434_:
{
return v___x_2435_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0___boxed(lean_object* v_msgData_2438_, lean_object* v_severity_2439_, lean_object* v_isSilent_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_){
_start:
{
uint8_t v_severity_boxed_2444_; uint8_t v_isSilent_boxed_2445_; lean_object* v_res_2446_; 
v_severity_boxed_2444_ = lean_unbox(v_severity_2439_);
v_isSilent_boxed_2445_ = lean_unbox(v_isSilent_2440_);
v_res_2446_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0(v_msgData_2438_, v_severity_boxed_2444_, v_isSilent_boxed_2445_, v___y_2441_, v___y_2442_);
lean_dec(v___y_2442_);
lean_dec_ref(v___y_2441_);
return v_res_2446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0(lean_object* v_msgData_2447_, lean_object* v___y_2448_, lean_object* v___y_2449_){
_start:
{
uint8_t v___x_2451_; uint8_t v___x_2452_; lean_object* v___x_2453_; 
v___x_2451_ = 1;
v___x_2452_ = 0;
v___x_2453_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0(v_msgData_2447_, v___x_2451_, v___x_2452_, v___y_2448_, v___y_2449_);
return v___x_2453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0___boxed(lean_object* v_msgData_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_){
_start:
{
lean_object* v_res_2458_; 
v_res_2458_ = lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0(v_msgData_2454_, v___y_2455_, v___y_2456_);
lean_dec(v___y_2456_);
lean_dec_ref(v___y_2455_);
return v_res_2458_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__1(void){
_start:
{
lean_object* v___x_2460_; lean_object* v___x_2461_; 
v___x_2460_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__0));
v___x_2461_ = l_Lean_stringToMessageData(v___x_2460_);
return v___x_2461_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__3(void){
_start:
{
lean_object* v___x_2463_; lean_object* v___x_2464_; 
v___x_2463_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__2));
v___x_2464_ = l_Lean_stringToMessageData(v___x_2463_);
return v___x_2464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2(lean_object* v___f_2465_, lean_object* v_ctxI_2466_, lean_object* v_tacI_2467_, lean_object* v_head_2468_, lean_object* v___x_2469_, lean_object* v___f_2470_, uint8_t v_mayFail_2471_, lean_object* v___y_2472_, lean_object* v___y_2473_){
_start:
{
lean_object* v___y_2476_; lean_object* v___x_2494_; lean_object* v___x_2495_; 
v___x_2494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2494_, 0, lean_box(0));
lean_ctor_set(v___x_2494_, 1, v___f_2465_);
lean_inc(v___x_2469_);
v___x_2495_ = lp_mathlib_Lean_Elab_ContextInfo_runTacticCode(v_ctxI_2466_, v_tacI_2467_, v_head_2468_, v___x_2469_, v___x_2494_, v___y_2472_, v___y_2473_);
if (lean_obj_tag(v___x_2495_) == 0)
{
lean_object* v_a_2496_; lean_object* v___x_2498_; uint8_t v_isShared_2499_; uint8_t v_isSharedCheck_2503_; 
lean_dec(v___y_2473_);
lean_dec_ref(v___y_2472_);
lean_dec_ref(v___f_2470_);
lean_dec(v___x_2469_);
v_a_2496_ = lean_ctor_get(v___x_2495_, 0);
v_isSharedCheck_2503_ = !lean_is_exclusive(v___x_2495_);
if (v_isSharedCheck_2503_ == 0)
{
v___x_2498_ = v___x_2495_;
v_isShared_2499_ = v_isSharedCheck_2503_;
goto v_resetjp_2497_;
}
else
{
lean_inc(v_a_2496_);
lean_dec(v___x_2495_);
v___x_2498_ = lean_box(0);
v_isShared_2499_ = v_isSharedCheck_2503_;
goto v_resetjp_2497_;
}
v_resetjp_2497_:
{
lean_object* v___x_2501_; 
if (v_isShared_2499_ == 0)
{
v___x_2501_ = v___x_2498_;
goto v_reusejp_2500_;
}
else
{
lean_object* v_reuseFailAlloc_2502_; 
v_reuseFailAlloc_2502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2502_, 0, v_a_2496_);
v___x_2501_ = v_reuseFailAlloc_2502_;
goto v_reusejp_2500_;
}
v_reusejp_2500_:
{
return v___x_2501_;
}
}
}
else
{
lean_object* v_a_2504_; lean_object* v___x_2506_; uint8_t v_isShared_2507_; uint8_t v_isSharedCheck_2533_; 
v_a_2504_ = lean_ctor_get(v___x_2495_, 0);
v_isSharedCheck_2533_ = !lean_is_exclusive(v___x_2495_);
if (v_isSharedCheck_2533_ == 0)
{
v___x_2506_ = v___x_2495_;
v_isShared_2507_ = v_isSharedCheck_2533_;
goto v_resetjp_2505_;
}
else
{
lean_inc(v_a_2504_);
lean_dec(v___x_2495_);
v___x_2506_ = lean_box(0);
v_isShared_2507_ = v_isSharedCheck_2533_;
goto v_resetjp_2505_;
}
v_resetjp_2505_:
{
uint8_t v___x_2527_; 
v___x_2527_ = l_Lean_Exception_isInterrupt(v_a_2504_);
if (v___x_2527_ == 0)
{
lean_del_object(v___x_2506_);
if (v_mayFail_2471_ == 0)
{
goto v___jp_2508_;
}
else
{
if (v___x_2527_ == 0)
{
lean_object* v___x_2528_; lean_object* v___x_2529_; 
lean_dec(v_a_2504_);
lean_dec(v___x_2469_);
v___x_2528_ = lean_box(0);
v___x_2529_ = lean_apply_4(v___f_2470_, v___x_2528_, v___y_2472_, v___y_2473_, lean_box(0));
v___y_2476_ = v___x_2529_;
goto v___jp_2475_;
}
else
{
goto v___jp_2508_;
}
}
}
else
{
lean_object* v___x_2531_; 
lean_dec(v___y_2473_);
lean_dec_ref(v___y_2472_);
lean_dec_ref(v___f_2470_);
lean_dec(v___x_2469_);
if (v_isShared_2507_ == 0)
{
v___x_2531_ = v___x_2506_;
goto v_reusejp_2530_;
}
else
{
lean_object* v_reuseFailAlloc_2532_; 
v_reuseFailAlloc_2532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2532_, 0, v_a_2504_);
v___x_2531_ = v_reuseFailAlloc_2532_;
goto v_reusejp_2530_;
}
v_reusejp_2530_:
{
return v___x_2531_;
}
}
v___jp_2508_:
{
lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; 
v___x_2509_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__1, &lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__1);
v___x_2510_ = l_Lean_MessageData_ofSyntax(v___x_2469_);
v___x_2511_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2511_, 0, v___x_2509_);
lean_ctor_set(v___x_2511_, 1, v___x_2510_);
v___x_2512_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__3, &lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__3_once, _init_lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___closed__3);
v___x_2513_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2513_, 0, v___x_2511_);
lean_ctor_set(v___x_2513_, 1, v___x_2512_);
v___x_2514_ = l_Lean_Exception_toMessageData(v_a_2504_);
v___x_2515_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2515_, 0, v___x_2513_);
lean_ctor_set(v___x_2515_, 1, v___x_2514_);
v___x_2516_ = lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0(v___x_2515_, v___y_2472_, v___y_2473_);
if (lean_obj_tag(v___x_2516_) == 0)
{
lean_object* v_a_2517_; lean_object* v___x_2518_; 
v_a_2517_ = lean_ctor_get(v___x_2516_, 0);
lean_inc(v_a_2517_);
lean_dec_ref_known(v___x_2516_, 1);
v___x_2518_ = lean_apply_4(v___f_2470_, v_a_2517_, v___y_2472_, v___y_2473_, lean_box(0));
v___y_2476_ = v___x_2518_;
goto v___jp_2475_;
}
else
{
lean_object* v_a_2519_; lean_object* v___x_2521_; uint8_t v_isShared_2522_; uint8_t v_isSharedCheck_2526_; 
lean_dec(v___y_2473_);
lean_dec_ref(v___y_2472_);
lean_dec_ref(v___f_2470_);
v_a_2519_ = lean_ctor_get(v___x_2516_, 0);
v_isSharedCheck_2526_ = !lean_is_exclusive(v___x_2516_);
if (v_isSharedCheck_2526_ == 0)
{
v___x_2521_ = v___x_2516_;
v_isShared_2522_ = v_isSharedCheck_2526_;
goto v_resetjp_2520_;
}
else
{
lean_inc(v_a_2519_);
lean_dec(v___x_2516_);
v___x_2521_ = lean_box(0);
v_isShared_2522_ = v_isSharedCheck_2526_;
goto v_resetjp_2520_;
}
v_resetjp_2520_:
{
lean_object* v___x_2524_; 
if (v_isShared_2522_ == 0)
{
v___x_2524_ = v___x_2521_;
goto v_reusejp_2523_;
}
else
{
lean_object* v_reuseFailAlloc_2525_; 
v_reuseFailAlloc_2525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2525_, 0, v_a_2519_);
v___x_2524_ = v_reuseFailAlloc_2525_;
goto v_reusejp_2523_;
}
v_reusejp_2523_:
{
return v___x_2524_;
}
}
}
}
}
}
v___jp_2475_:
{
if (lean_obj_tag(v___y_2476_) == 0)
{
lean_object* v_a_2477_; lean_object* v___x_2479_; uint8_t v_isShared_2480_; uint8_t v_isSharedCheck_2485_; 
v_a_2477_ = lean_ctor_get(v___y_2476_, 0);
v_isSharedCheck_2485_ = !lean_is_exclusive(v___y_2476_);
if (v_isSharedCheck_2485_ == 0)
{
v___x_2479_ = v___y_2476_;
v_isShared_2480_ = v_isSharedCheck_2485_;
goto v_resetjp_2478_;
}
else
{
lean_inc(v_a_2477_);
lean_dec(v___y_2476_);
v___x_2479_ = lean_box(0);
v_isShared_2480_ = v_isSharedCheck_2485_;
goto v_resetjp_2478_;
}
v_resetjp_2478_:
{
lean_object* v_a_2481_; lean_object* v___x_2483_; 
v_a_2481_ = lean_ctor_get(v_a_2477_, 0);
lean_inc(v_a_2481_);
lean_dec(v_a_2477_);
if (v_isShared_2480_ == 0)
{
lean_ctor_set(v___x_2479_, 0, v_a_2481_);
v___x_2483_ = v___x_2479_;
goto v_reusejp_2482_;
}
else
{
lean_object* v_reuseFailAlloc_2484_; 
v_reuseFailAlloc_2484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2484_, 0, v_a_2481_);
v___x_2483_ = v_reuseFailAlloc_2484_;
goto v_reusejp_2482_;
}
v_reusejp_2482_:
{
return v___x_2483_;
}
}
}
else
{
lean_object* v_a_2486_; lean_object* v___x_2488_; uint8_t v_isShared_2489_; uint8_t v_isSharedCheck_2493_; 
v_a_2486_ = lean_ctor_get(v___y_2476_, 0);
v_isSharedCheck_2493_ = !lean_is_exclusive(v___y_2476_);
if (v_isSharedCheck_2493_ == 0)
{
v___x_2488_ = v___y_2476_;
v_isShared_2489_ = v_isSharedCheck_2493_;
goto v_resetjp_2487_;
}
else
{
lean_inc(v_a_2486_);
lean_dec(v___y_2476_);
v___x_2488_ = lean_box(0);
v_isShared_2489_ = v_isSharedCheck_2493_;
goto v_resetjp_2487_;
}
v_resetjp_2487_:
{
lean_object* v___x_2491_; 
if (v_isShared_2489_ == 0)
{
v___x_2491_ = v___x_2488_;
goto v_reusejp_2490_;
}
else
{
lean_object* v_reuseFailAlloc_2492_; 
v_reuseFailAlloc_2492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2492_, 0, v_a_2486_);
v___x_2491_ = v_reuseFailAlloc_2492_;
goto v_reusejp_2490_;
}
v_reusejp_2490_:
{
return v___x_2491_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___boxed(lean_object* v___f_2534_, lean_object* v_ctxI_2535_, lean_object* v_tacI_2536_, lean_object* v_head_2537_, lean_object* v___x_2538_, lean_object* v___f_2539_, lean_object* v_mayFail_2540_, lean_object* v___y_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_){
_start:
{
uint8_t v_mayFail_boxed_2544_; lean_object* v_res_2545_; 
v_mayFail_boxed_2544_ = lean_unbox(v_mayFail_2540_);
v_res_2545_ = lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2(v___f_2534_, v_ctxI_2535_, v_tacI_2536_, v_head_2537_, v___x_2538_, v___f_2539_, v_mayFail_boxed_2544_, v___y_2541_, v___y_2542_);
lean_dec_ref(v_tacI_2536_);
return v_res_2545_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__1(void){
_start:
{
lean_object* v___x_2547_; 
v___x_2547_ = l_Array_mkArray0(lean_box(0));
return v___x_2547_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq(lean_object* v_config_2554_, lean_object* v_tacticSeq_2555_, lean_object* v_i_2556_, lean_object* v_ctx_2557_, lean_object* v_a_2558_, lean_object* v_a_2559_){
_start:
{
lean_object* v___x_2564_; 
v___x_2564_ = l_Lean_Elab_Command_getRef___redArg(v_a_2558_);
if (lean_obj_tag(v___x_2564_) == 0)
{
lean_object* v_a_2565_; lean_object* v_fileName_2566_; lean_object* v_fileMap_2567_; lean_object* v_currRecDepth_2568_; lean_object* v_cmdPos_2569_; lean_object* v_macroStack_2570_; lean_object* v_quotContext_x3f_2571_; lean_object* v_currMacroScope_2572_; lean_object* v_snap_x3f_2573_; lean_object* v_cancelTk_x3f_2574_; uint8_t v_suppressElabErrors_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v_ref_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; 
v_a_2565_ = lean_ctor_get(v___x_2564_, 0);
lean_inc(v_a_2565_);
lean_dec_ref_known(v___x_2564_, 1);
v_fileName_2566_ = lean_ctor_get(v_a_2558_, 0);
v_fileMap_2567_ = lean_ctor_get(v_a_2558_, 1);
v_currRecDepth_2568_ = lean_ctor_get(v_a_2558_, 2);
v_cmdPos_2569_ = lean_ctor_get(v_a_2558_, 3);
v_macroStack_2570_ = lean_ctor_get(v_a_2558_, 4);
v_quotContext_x3f_2571_ = lean_ctor_get(v_a_2558_, 5);
v_currMacroScope_2572_ = lean_ctor_get(v_a_2558_, 6);
v_snap_x3f_2573_ = lean_ctor_get(v_a_2558_, 8);
v_cancelTk_x3f_2574_ = lean_ctor_get(v_a_2558_, 9);
v_suppressElabErrors_2575_ = lean_ctor_get_uint8(v_a_2558_, sizeof(void*)*10);
v___x_2576_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__17));
v___x_2577_ = lean_box(2);
lean_inc_ref(v_tacticSeq_2555_);
v___x_2578_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2578_, 0, v___x_2577_);
lean_ctor_set(v___x_2578_, 1, v___x_2576_);
lean_ctor_set(v___x_2578_, 2, v_tacticSeq_2555_);
v_ref_2579_ = l_Lean_replaceRef(v___x_2578_, v_a_2565_);
lean_dec(v_a_2565_);
lean_dec_ref_known(v___x_2578_, 3);
lean_inc(v_cancelTk_x3f_2574_);
lean_inc(v_snap_x3f_2573_);
lean_inc(v_currMacroScope_2572_);
lean_inc(v_quotContext_x3f_2571_);
lean_inc(v_macroStack_2570_);
lean_inc(v_cmdPos_2569_);
lean_inc(v_currRecDepth_2568_);
lean_inc_ref(v_fileMap_2567_);
lean_inc_ref(v_fileName_2566_);
v___x_2580_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_2580_, 0, v_fileName_2566_);
lean_ctor_set(v___x_2580_, 1, v_fileMap_2567_);
lean_ctor_set(v___x_2580_, 2, v_currRecDepth_2568_);
lean_ctor_set(v___x_2580_, 3, v_cmdPos_2569_);
lean_ctor_set(v___x_2580_, 4, v_macroStack_2570_);
lean_ctor_set(v___x_2580_, 5, v_quotContext_x3f_2571_);
lean_ctor_set(v___x_2580_, 6, v_currMacroScope_2572_);
lean_ctor_set(v___x_2580_, 7, v_ref_2579_);
lean_ctor_set(v___x_2580_, 8, v_snap_x3f_2573_);
lean_ctor_set(v___x_2580_, 9, v_cancelTk_x3f_2574_);
lean_ctor_set_uint8(v___x_2580_, sizeof(void*)*10, v_suppressElabErrors_2575_);
v___x_2581_ = l_Lean_Elab_Command_getRef___redArg(v___x_2580_);
if (lean_obj_tag(v___x_2581_) == 0)
{
lean_object* v_a_2582_; lean_object* v___x_2583_; 
v_a_2582_ = lean_ctor_get(v___x_2581_, 0);
lean_inc(v_a_2582_);
lean_dec_ref_known(v___x_2581_, 1);
v___x_2583_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___x_2580_);
if (lean_obj_tag(v___x_2583_) == 0)
{
lean_object* v___f_2584_; uint8_t v___x_2585_; lean_object* v___x_2586_; 
lean_dec_ref_known(v___x_2583_, 1);
v___f_2584_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__0));
v___x_2585_ = 0;
v___x_2586_ = l_Lean_SourceInfo_fromRef(v_a_2582_, v___x_2585_);
lean_dec(v_a_2582_);
if (lean_obj_tag(v_quotContext_x3f_2571_) == 0)
{
lean_object* v___x_2651_; 
v___x_2651_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg(v_a_2559_);
lean_dec_ref(v___x_2651_);
goto v___jp_2587_;
}
else
{
goto v___jp_2587_;
}
v___jp_2587_:
{
lean_object* v_tacI_2588_; lean_object* v_goalsBefore_2589_; 
v_tacI_2588_ = lean_ctor_get(v_i_2556_, 1);
lean_inc_ref(v_tacI_2588_);
v_goalsBefore_2589_ = lean_ctor_get(v_tacI_2588_, 2);
if (lean_obj_tag(v_goalsBefore_2589_) == 1)
{
lean_object* v_tail_2590_; 
v_tail_2590_ = lean_ctor_get(v_goalsBefore_2589_, 1);
if (lean_obj_tag(v_tail_2590_) == 0)
{
lean_object* v_ctxI_2591_; uint8_t v_mayFail_2592_; lean_object* v_head_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___f_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v___f_2603_; lean_object* v___x_2604_; 
v_ctxI_2591_ = lean_ctor_get(v_i_2556_, 0);
lean_inc_ref_n(v_ctxI_2591_, 2);
v_mayFail_2592_ = lean_ctor_get_uint8(v_i_2556_, sizeof(void*)*2);
lean_dec_ref(v_i_2556_);
v_head_2593_ = lean_ctor_get(v_goalsBefore_2589_, 0);
lean_inc_n(v_head_2593_, 2);
v___x_2594_ = lean_obj_once(&lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__1, &lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__1_once, _init_lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__1);
v___x_2595_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_findTacticSeqs___lam__1___closed__10));
v___x_2596_ = l_Lean_Syntax_SepArray_ofElems(v___x_2595_, v_tacticSeq_2555_);
lean_dec_ref(v_tacticSeq_2555_);
v___x_2597_ = l_Array_append___redArg(v___x_2594_, v___x_2596_);
lean_dec_ref(v___x_2596_);
lean_inc(v___x_2586_);
v___x_2598_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2598_, 0, v___x_2586_);
lean_ctor_set(v___x_2598_, 1, v___x_2576_);
lean_ctor_set(v___x_2598_, 2, v___x_2597_);
lean_inc_ref(v_goalsBefore_2589_);
v___f_2599_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__1___boxed), 5, 1);
lean_closure_set(v___f_2599_, 0, v_goalsBefore_2589_);
v___x_2600_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___closed__3));
v___x_2601_ = l_Lean_Syntax_node1(v___x_2586_, v___x_2600_, v___x_2598_);
v___x_2602_ = lean_box(v_mayFail_2592_);
lean_inc(v___x_2601_);
lean_inc_ref(v_tacI_2588_);
v___f_2603_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___lam__2___boxed), 10, 7);
lean_closure_set(v___f_2603_, 0, v___f_2584_);
lean_closure_set(v___f_2603_, 1, v_ctxI_2591_);
lean_closure_set(v___f_2603_, 2, v_tacI_2588_);
lean_closure_set(v___f_2603_, 3, v_head_2593_);
lean_closure_set(v___f_2603_, 4, v___x_2601_);
lean_closure_set(v___f_2603_, 5, v___f_2599_);
lean_closure_set(v___f_2603_, 6, v___x_2602_);
v___x_2604_ = lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg(v___f_2603_, v___x_2580_, v_a_2559_);
if (lean_obj_tag(v___x_2604_) == 0)
{
lean_object* v_a_2605_; lean_object* v_fst_2606_; lean_object* v_snd_2607_; lean_object* v_test_2608_; lean_object* v_tell_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; 
v_a_2605_ = lean_ctor_get(v___x_2604_, 0);
lean_inc(v_a_2605_);
lean_dec_ref_known(v___x_2604_, 1);
v_fst_2606_ = lean_ctor_get(v_a_2605_, 0);
lean_inc(v_fst_2606_);
v_snd_2607_ = lean_ctor_get(v_a_2605_, 1);
lean_inc(v_snd_2607_);
lean_dec(v_a_2605_);
v_test_2608_ = lean_ctor_get(v_config_2554_, 1);
lean_inc_ref(v_test_2608_);
v_tell_2609_ = lean_ctor_get(v_config_2554_, 2);
lean_inc_ref(v_tell_2609_);
lean_dec_ref(v_config_2554_);
v___x_2610_ = lean_apply_4(v_test_2608_, v_ctxI_2591_, v_tacI_2588_, v_ctx_2557_, v_head_2593_);
v___x_2611_ = lp_mathlib_Lean_withHeartbeats___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__1___redArg(v___x_2610_, v___x_2580_, v_a_2559_);
if (lean_obj_tag(v___x_2611_) == 0)
{
lean_object* v_a_2612_; lean_object* v_fst_2613_; lean_object* v_snd_2614_; lean_object* v___x_2615_; 
v_a_2612_ = lean_ctor_get(v___x_2611_, 0);
lean_inc(v_a_2612_);
lean_dec_ref_known(v___x_2611_, 1);
v_fst_2613_ = lean_ctor_get(v_a_2612_, 0);
lean_inc(v_fst_2613_);
v_snd_2614_ = lean_ctor_get(v_a_2612_, 1);
lean_inc(v_snd_2614_);
lean_dec(v_a_2612_);
lean_inc(v_a_2559_);
lean_inc_ref(v___x_2580_);
v___x_2615_ = lean_apply_8(v_tell_2609_, v___x_2601_, v_fst_2606_, v_snd_2607_, v_fst_2613_, v_snd_2614_, v___x_2580_, v_a_2559_, lean_box(0));
if (lean_obj_tag(v___x_2615_) == 0)
{
lean_object* v_a_2616_; lean_object* v___x_2618_; uint8_t v_isShared_2619_; uint8_t v_isSharedCheck_2626_; 
v_a_2616_ = lean_ctor_get(v___x_2615_, 0);
v_isSharedCheck_2626_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2626_ == 0)
{
v___x_2618_ = v___x_2615_;
v_isShared_2619_ = v_isSharedCheck_2626_;
goto v_resetjp_2617_;
}
else
{
lean_inc(v_a_2616_);
lean_dec(v___x_2615_);
v___x_2618_ = lean_box(0);
v_isShared_2619_ = v_isSharedCheck_2626_;
goto v_resetjp_2617_;
}
v_resetjp_2617_:
{
if (lean_obj_tag(v_a_2616_) == 1)
{
lean_object* v_val_2620_; lean_object* v___x_2621_; 
lean_del_object(v___x_2618_);
v_val_2620_ = lean_ctor_get(v_a_2616_, 0);
lean_inc(v_val_2620_);
lean_dec_ref_known(v_a_2616_, 1);
v___x_2621_ = lp_mathlib_Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0(v_val_2620_, v___x_2580_, v_a_2559_);
lean_dec_ref_known(v___x_2580_, 10);
return v___x_2621_;
}
else
{
lean_object* v___x_2622_; lean_object* v___x_2624_; 
lean_dec(v_a_2616_);
lean_dec_ref_known(v___x_2580_, 10);
v___x_2622_ = lean_box(0);
if (v_isShared_2619_ == 0)
{
lean_ctor_set(v___x_2618_, 0, v___x_2622_);
v___x_2624_ = v___x_2618_;
goto v_reusejp_2623_;
}
else
{
lean_object* v_reuseFailAlloc_2625_; 
v_reuseFailAlloc_2625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2625_, 0, v___x_2622_);
v___x_2624_ = v_reuseFailAlloc_2625_;
goto v_reusejp_2623_;
}
v_reusejp_2623_:
{
return v___x_2624_;
}
}
}
}
else
{
lean_object* v_a_2627_; lean_object* v___x_2629_; uint8_t v_isShared_2630_; uint8_t v_isSharedCheck_2634_; 
lean_dec_ref_known(v___x_2580_, 10);
v_a_2627_ = lean_ctor_get(v___x_2615_, 0);
v_isSharedCheck_2634_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2634_ == 0)
{
v___x_2629_ = v___x_2615_;
v_isShared_2630_ = v_isSharedCheck_2634_;
goto v_resetjp_2628_;
}
else
{
lean_inc(v_a_2627_);
lean_dec(v___x_2615_);
v___x_2629_ = lean_box(0);
v_isShared_2630_ = v_isSharedCheck_2634_;
goto v_resetjp_2628_;
}
v_resetjp_2628_:
{
lean_object* v___x_2632_; 
if (v_isShared_2630_ == 0)
{
v___x_2632_ = v___x_2629_;
goto v_reusejp_2631_;
}
else
{
lean_object* v_reuseFailAlloc_2633_; 
v_reuseFailAlloc_2633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2633_, 0, v_a_2627_);
v___x_2632_ = v_reuseFailAlloc_2633_;
goto v_reusejp_2631_;
}
v_reusejp_2631_:
{
return v___x_2632_;
}
}
}
}
else
{
lean_object* v_a_2635_; lean_object* v___x_2637_; uint8_t v_isShared_2638_; uint8_t v_isSharedCheck_2642_; 
lean_dec_ref(v_tell_2609_);
lean_dec(v_snd_2607_);
lean_dec(v_fst_2606_);
lean_dec(v___x_2601_);
lean_dec_ref_known(v___x_2580_, 10);
v_a_2635_ = lean_ctor_get(v___x_2611_, 0);
v_isSharedCheck_2642_ = !lean_is_exclusive(v___x_2611_);
if (v_isSharedCheck_2642_ == 0)
{
v___x_2637_ = v___x_2611_;
v_isShared_2638_ = v_isSharedCheck_2642_;
goto v_resetjp_2636_;
}
else
{
lean_inc(v_a_2635_);
lean_dec(v___x_2611_);
v___x_2637_ = lean_box(0);
v_isShared_2638_ = v_isSharedCheck_2642_;
goto v_resetjp_2636_;
}
v_resetjp_2636_:
{
lean_object* v___x_2640_; 
if (v_isShared_2638_ == 0)
{
v___x_2640_ = v___x_2637_;
goto v_reusejp_2639_;
}
else
{
lean_object* v_reuseFailAlloc_2641_; 
v_reuseFailAlloc_2641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2641_, 0, v_a_2635_);
v___x_2640_ = v_reuseFailAlloc_2641_;
goto v_reusejp_2639_;
}
v_reusejp_2639_:
{
return v___x_2640_;
}
}
}
}
else
{
lean_object* v_a_2643_; lean_object* v___x_2645_; uint8_t v_isShared_2646_; uint8_t v_isSharedCheck_2650_; 
lean_dec(v___x_2601_);
lean_dec(v_head_2593_);
lean_dec_ref(v_ctxI_2591_);
lean_dec_ref(v_tacI_2588_);
lean_dec_ref_known(v___x_2580_, 10);
lean_dec(v_ctx_2557_);
lean_dec_ref(v_config_2554_);
v_a_2643_ = lean_ctor_get(v___x_2604_, 0);
v_isSharedCheck_2650_ = !lean_is_exclusive(v___x_2604_);
if (v_isSharedCheck_2650_ == 0)
{
v___x_2645_ = v___x_2604_;
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
else
{
lean_inc(v_a_2643_);
lean_dec(v___x_2604_);
v___x_2645_ = lean_box(0);
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
v_resetjp_2644_:
{
lean_object* v___x_2648_; 
if (v_isShared_2646_ == 0)
{
v___x_2648_ = v___x_2645_;
goto v_reusejp_2647_;
}
else
{
lean_object* v_reuseFailAlloc_2649_; 
v_reuseFailAlloc_2649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2649_, 0, v_a_2643_);
v___x_2648_ = v_reuseFailAlloc_2649_;
goto v_reusejp_2647_;
}
v_reusejp_2647_:
{
return v___x_2648_;
}
}
}
}
else
{
lean_dec_ref(v_tacI_2588_);
lean_dec(v___x_2586_);
lean_dec_ref_known(v___x_2580_, 10);
lean_dec(v_ctx_2557_);
lean_dec_ref(v_i_2556_);
lean_dec_ref(v_tacticSeq_2555_);
lean_dec_ref(v_config_2554_);
goto v___jp_2561_;
}
}
else
{
lean_dec_ref(v_tacI_2588_);
lean_dec(v___x_2586_);
lean_dec_ref_known(v___x_2580_, 10);
lean_dec(v_ctx_2557_);
lean_dec_ref(v_i_2556_);
lean_dec_ref(v_tacticSeq_2555_);
lean_dec_ref(v_config_2554_);
goto v___jp_2561_;
}
}
}
else
{
lean_object* v_a_2652_; lean_object* v___x_2654_; uint8_t v_isShared_2655_; uint8_t v_isSharedCheck_2659_; 
lean_dec(v_a_2582_);
lean_dec_ref_known(v___x_2580_, 10);
lean_dec(v_ctx_2557_);
lean_dec_ref(v_i_2556_);
lean_dec_ref(v_tacticSeq_2555_);
lean_dec_ref(v_config_2554_);
v_a_2652_ = lean_ctor_get(v___x_2583_, 0);
v_isSharedCheck_2659_ = !lean_is_exclusive(v___x_2583_);
if (v_isSharedCheck_2659_ == 0)
{
v___x_2654_ = v___x_2583_;
v_isShared_2655_ = v_isSharedCheck_2659_;
goto v_resetjp_2653_;
}
else
{
lean_inc(v_a_2652_);
lean_dec(v___x_2583_);
v___x_2654_ = lean_box(0);
v_isShared_2655_ = v_isSharedCheck_2659_;
goto v_resetjp_2653_;
}
v_resetjp_2653_:
{
lean_object* v___x_2657_; 
if (v_isShared_2655_ == 0)
{
v___x_2657_ = v___x_2654_;
goto v_reusejp_2656_;
}
else
{
lean_object* v_reuseFailAlloc_2658_; 
v_reuseFailAlloc_2658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2658_, 0, v_a_2652_);
v___x_2657_ = v_reuseFailAlloc_2658_;
goto v_reusejp_2656_;
}
v_reusejp_2656_:
{
return v___x_2657_;
}
}
}
}
else
{
lean_object* v_a_2660_; lean_object* v___x_2662_; uint8_t v_isShared_2663_; uint8_t v_isSharedCheck_2667_; 
lean_dec_ref_known(v___x_2580_, 10);
lean_dec(v_ctx_2557_);
lean_dec_ref(v_i_2556_);
lean_dec_ref(v_tacticSeq_2555_);
lean_dec_ref(v_config_2554_);
v_a_2660_ = lean_ctor_get(v___x_2581_, 0);
v_isSharedCheck_2667_ = !lean_is_exclusive(v___x_2581_);
if (v_isSharedCheck_2667_ == 0)
{
v___x_2662_ = v___x_2581_;
v_isShared_2663_ = v_isSharedCheck_2667_;
goto v_resetjp_2661_;
}
else
{
lean_inc(v_a_2660_);
lean_dec(v___x_2581_);
v___x_2662_ = lean_box(0);
v_isShared_2663_ = v_isSharedCheck_2667_;
goto v_resetjp_2661_;
}
v_resetjp_2661_:
{
lean_object* v___x_2665_; 
if (v_isShared_2663_ == 0)
{
v___x_2665_ = v___x_2662_;
goto v_reusejp_2664_;
}
else
{
lean_object* v_reuseFailAlloc_2666_; 
v_reuseFailAlloc_2666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2666_, 0, v_a_2660_);
v___x_2665_ = v_reuseFailAlloc_2666_;
goto v_reusejp_2664_;
}
v_reusejp_2664_:
{
return v___x_2665_;
}
}
}
}
else
{
lean_object* v_a_2668_; lean_object* v___x_2670_; uint8_t v_isShared_2671_; uint8_t v_isSharedCheck_2675_; 
lean_dec(v_ctx_2557_);
lean_dec_ref(v_i_2556_);
lean_dec_ref(v_tacticSeq_2555_);
lean_dec_ref(v_config_2554_);
v_a_2668_ = lean_ctor_get(v___x_2564_, 0);
v_isSharedCheck_2675_ = !lean_is_exclusive(v___x_2564_);
if (v_isSharedCheck_2675_ == 0)
{
v___x_2670_ = v___x_2564_;
v_isShared_2671_ = v_isSharedCheck_2675_;
goto v_resetjp_2669_;
}
else
{
lean_inc(v_a_2668_);
lean_dec(v___x_2564_);
v___x_2670_ = lean_box(0);
v_isShared_2671_ = v_isSharedCheck_2675_;
goto v_resetjp_2669_;
}
v_resetjp_2669_:
{
lean_object* v___x_2673_; 
if (v_isShared_2671_ == 0)
{
v___x_2673_ = v___x_2670_;
goto v_reusejp_2672_;
}
else
{
lean_object* v_reuseFailAlloc_2674_; 
v_reuseFailAlloc_2674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2674_, 0, v_a_2668_);
v___x_2673_ = v_reuseFailAlloc_2674_;
goto v_reusejp_2672_;
}
v_reusejp_2672_:
{
return v___x_2673_;
}
}
}
v___jp_2561_:
{
lean_object* v___x_2562_; lean_object* v___x_2563_; 
v___x_2562_ = lean_box(0);
v___x_2563_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2563_, 0, v___x_2562_);
return v___x_2563_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq___boxed(lean_object* v_config_2676_, lean_object* v_tacticSeq_2677_, lean_object* v_i_2678_, lean_object* v_ctx_2679_, lean_object* v_a_2680_, lean_object* v_a_2681_, lean_object* v_a_2682_){
_start:
{
lean_object* v_res_2683_; 
v_res_2683_ = lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq(v_config_2676_, v_tacticSeq_2677_, v_i_2678_, v_ctx_2679_, v_a_2680_, v_a_2681_);
lean_dec(v_a_2681_);
lean_dec_ref(v_a_2680_);
return v_res_2683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4(lean_object* v_msgData_2684_, lean_object* v___y_2685_, lean_object* v___y_2686_){
_start:
{
lean_object* v___x_2688_; 
v___x_2688_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___redArg(v_msgData_2684_, v___y_2686_);
return v___x_2688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4___boxed(lean_object* v_msgData_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_, lean_object* v___y_2692_){
_start:
{
lean_object* v_res_2693_; 
v_res_2693_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3_spec__4(v_msgData_2689_, v___y_2690_, v___y_2691_);
lean_dec(v___y_2691_);
lean_dec_ref(v___y_2690_);
return v_res_2693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_TacticAnalysis_runPass_spec__0(lean_object* v_ref_2694_, lean_object* v_msgData_2695_, lean_object* v___y_2696_, lean_object* v___y_2697_){
_start:
{
uint8_t v___x_2699_; uint8_t v___x_2700_; lean_object* v___x_2701_; 
v___x_2699_ = 1;
v___x_2700_ = 0;
v___x_2701_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__0_spec__0_spec__3(v_ref_2694_, v_msgData_2695_, v___x_2699_, v___x_2700_, v___y_2696_, v___y_2697_);
return v___x_2701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_TacticAnalysis_runPass_spec__0___boxed(lean_object* v_ref_2702_, lean_object* v_msgData_2703_, lean_object* v___y_2704_, lean_object* v___y_2705_, lean_object* v___y_2706_){
_start:
{
lean_object* v_res_2707_; 
v_res_2707_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_TacticAnalysis_runPass_spec__0(v_ref_2702_, v_msgData_2703_, v___y_2704_, v___y_2705_);
lean_dec(v___y_2705_);
lean_dec_ref(v___y_2704_);
lean_dec(v_ref_2702_);
return v_res_2707_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__4(void){
_start:
{
lean_object* v___x_2717_; lean_object* v___x_2718_; 
v___x_2717_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__3));
v___x_2718_ = l_Lean_stringToMessageData(v___x_2717_);
return v___x_2718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1(lean_object* v_config_2719_, lean_object* v_as_2720_, size_t v_sz_2721_, size_t v_i_2722_, lean_object* v_b_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_){
_start:
{
lean_object* v_a_2728_; uint8_t v___x_2732_; 
v___x_2732_ = lean_usize_dec_lt(v_i_2722_, v_sz_2721_);
if (v___x_2732_ == 0)
{
lean_object* v___x_2733_; 
lean_dec_ref(v_config_2719_);
v___x_2733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2733_, 0, v_b_2723_);
return v___x_2733_;
}
else
{
lean_object* v_snd_2734_; lean_object* v_fst_2735_; lean_object* v___x_2737_; uint8_t v_isShared_2738_; uint8_t v_isSharedCheck_2806_; 
v_snd_2734_ = lean_ctor_get(v_b_2723_, 1);
v_fst_2735_ = lean_ctor_get(v_b_2723_, 0);
v_isSharedCheck_2806_ = !lean_is_exclusive(v_b_2723_);
if (v_isSharedCheck_2806_ == 0)
{
v___x_2737_ = v_b_2723_;
v_isShared_2738_ = v_isSharedCheck_2806_;
goto v_resetjp_2736_;
}
else
{
lean_inc(v_snd_2734_);
lean_inc(v_fst_2735_);
lean_dec(v_b_2723_);
v___x_2737_ = lean_box(0);
v_isShared_2738_ = v_isSharedCheck_2806_;
goto v_resetjp_2736_;
}
v_resetjp_2736_:
{
lean_object* v_fst_2739_; lean_object* v_snd_2740_; lean_object* v___x_2742_; uint8_t v_isShared_2743_; uint8_t v_isSharedCheck_2805_; 
v_fst_2739_ = lean_ctor_get(v_snd_2734_, 0);
v_snd_2740_ = lean_ctor_get(v_snd_2734_, 1);
v_isSharedCheck_2805_ = !lean_is_exclusive(v_snd_2734_);
if (v_isSharedCheck_2805_ == 0)
{
v___x_2742_ = v_snd_2734_;
v_isShared_2743_ = v_isSharedCheck_2805_;
goto v_resetjp_2741_;
}
else
{
lean_inc(v_snd_2740_);
lean_inc(v_fst_2739_);
lean_dec(v_snd_2734_);
v___x_2742_ = lean_box(0);
v_isShared_2743_ = v_isSharedCheck_2805_;
goto v_resetjp_2741_;
}
v_resetjp_2741_:
{
lean_object* v_acc_2744_; lean_object* v___y_2746_; lean_object* v___y_2747_; lean_object* v___x_2754_; lean_object* v_a_2755_; lean_object* v_firstInfo_2757_; lean_object* v___y_2758_; lean_object* v___y_2759_; 
v_acc_2744_ = lean_box(0);
v___x_2754_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__2));
v_a_2755_ = lean_array_uget_borrowed(v_as_2720_, v_i_2722_);
if (lean_obj_tag(v_fst_2739_) == 0)
{
lean_object* v___x_2804_; 
lean_inc(v_a_2755_);
v___x_2804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2804_, 0, v_a_2755_);
v_firstInfo_2757_ = v___x_2804_;
v___y_2758_ = v___y_2724_;
v___y_2759_ = v___y_2725_;
goto v___jp_2756_;
}
else
{
v_firstInfo_2757_ = v_fst_2739_;
v___y_2758_ = v___y_2724_;
v___y_2759_ = v___y_2725_;
goto v___jp_2756_;
}
v___jp_2745_:
{
lean_object* v___x_2749_; 
if (v_isShared_2743_ == 0)
{
lean_ctor_set(v___x_2742_, 1, v___y_2746_);
lean_ctor_set(v___x_2742_, 0, v___y_2747_);
v___x_2749_ = v___x_2742_;
goto v_reusejp_2748_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v___y_2747_);
lean_ctor_set(v_reuseFailAlloc_2753_, 1, v___y_2746_);
v___x_2749_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2748_;
}
v_reusejp_2748_:
{
lean_object* v___x_2751_; 
if (v_isShared_2738_ == 0)
{
lean_ctor_set(v___x_2737_, 1, v___x_2749_);
lean_ctor_set(v___x_2737_, 0, v_acc_2744_);
v___x_2751_ = v___x_2737_;
goto v_reusejp_2750_;
}
else
{
lean_object* v_reuseFailAlloc_2752_; 
v_reuseFailAlloc_2752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2752_, 0, v_acc_2744_);
lean_ctor_set(v_reuseFailAlloc_2752_, 1, v___x_2749_);
v___x_2751_ = v_reuseFailAlloc_2752_;
goto v_reusejp_2750_;
}
v_reusejp_2750_:
{
v_a_2728_ = v___x_2751_;
goto v___jp_2727_;
}
}
}
v___jp_2756_:
{
lean_object* v_tacI_2760_; lean_object* v_toElabInfo_2761_; lean_object* v_stx_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2802_; 
v_tacI_2760_ = lean_ctor_get(v_a_2755_, 1);
v_toElabInfo_2761_ = lean_ctor_get(v_tacI_2760_, 0);
lean_inc_ref(v_toElabInfo_2761_);
v_stx_2762_ = lean_ctor_get(v_toElabInfo_2761_, 1);
v_isSharedCheck_2802_ = !lean_is_exclusive(v_toElabInfo_2761_);
if (v_isSharedCheck_2802_ == 0)
{
lean_object* v_unused_2803_; 
v_unused_2803_ = lean_ctor_get(v_toElabInfo_2761_, 0);
lean_dec(v_unused_2803_);
v___x_2764_ = v_toElabInfo_2761_;
v_isShared_2765_ = v_isSharedCheck_2802_;
goto v_resetjp_2763_;
}
else
{
lean_inc(v_stx_2762_);
lean_dec(v_toElabInfo_2761_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2802_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v_trigger_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; 
v_trigger_2766_ = lean_ctor_get(v_config_2719_, 0);
lean_inc_n(v_stx_2762_, 2);
v___x_2767_ = lean_array_push(v_snd_2740_, v_stx_2762_);
lean_inc_ref(v_trigger_2766_);
v___x_2768_ = lean_apply_2(v_trigger_2766_, v_fst_2735_, v_stx_2762_);
switch(lean_obj_tag(v___x_2768_))
{
case 0:
{
lean_dec_ref(v___x_2767_);
lean_del_object(v___x_2764_);
lean_dec(v_stx_2762_);
lean_dec(v_firstInfo_2757_);
lean_del_object(v___x_2742_);
lean_del_object(v___x_2737_);
v_a_2728_ = v___x_2754_;
goto v___jp_2727_;
}
case 1:
{
lean_object* v_context_2769_; lean_object* v___x_2771_; uint8_t v_isShared_2772_; uint8_t v_isSharedCheck_2780_; 
lean_dec(v_stx_2762_);
lean_del_object(v___x_2742_);
lean_del_object(v___x_2737_);
v_context_2769_ = lean_ctor_get(v___x_2768_, 0);
v_isSharedCheck_2780_ = !lean_is_exclusive(v___x_2768_);
if (v_isSharedCheck_2780_ == 0)
{
v___x_2771_ = v___x_2768_;
v_isShared_2772_ = v_isSharedCheck_2780_;
goto v_resetjp_2770_;
}
else
{
lean_inc(v_context_2769_);
lean_dec(v___x_2768_);
v___x_2771_ = lean_box(0);
v_isShared_2772_ = v_isSharedCheck_2780_;
goto v_resetjp_2770_;
}
v_resetjp_2770_:
{
lean_object* v___x_2774_; 
if (v_isShared_2772_ == 0)
{
v___x_2774_ = v___x_2771_;
goto v_reusejp_2773_;
}
else
{
lean_object* v_reuseFailAlloc_2779_; 
v_reuseFailAlloc_2779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2779_, 0, v_context_2769_);
v___x_2774_ = v_reuseFailAlloc_2779_;
goto v_reusejp_2773_;
}
v_reusejp_2773_:
{
lean_object* v___x_2776_; 
if (v_isShared_2765_ == 0)
{
lean_ctor_set(v___x_2764_, 1, v___x_2767_);
lean_ctor_set(v___x_2764_, 0, v_firstInfo_2757_);
v___x_2776_ = v___x_2764_;
goto v_reusejp_2775_;
}
else
{
lean_object* v_reuseFailAlloc_2778_; 
v_reuseFailAlloc_2778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2778_, 0, v_firstInfo_2757_);
lean_ctor_set(v_reuseFailAlloc_2778_, 1, v___x_2767_);
v___x_2776_ = v_reuseFailAlloc_2778_;
goto v_reusejp_2775_;
}
v_reusejp_2775_:
{
lean_object* v___x_2777_; 
v___x_2777_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2777_, 0, v___x_2774_);
lean_ctor_set(v___x_2777_, 1, v___x_2776_);
v_a_2728_ = v___x_2777_;
goto v___jp_2727_;
}
}
}
}
default: 
{
lean_del_object(v___x_2764_);
if (lean_obj_tag(v_firstInfo_2757_) == 1)
{
lean_object* v_context_2781_; lean_object* v_val_2782_; lean_object* v___x_2783_; 
lean_dec(v_stx_2762_);
v_context_2781_ = lean_ctor_get(v___x_2768_, 0);
lean_inc(v_context_2781_);
lean_dec_ref_known(v___x_2768_, 1);
v_val_2782_ = lean_ctor_get(v_firstInfo_2757_, 0);
lean_inc(v_val_2782_);
lean_inc_ref(v___x_2767_);
lean_inc_ref(v_config_2719_);
v___x_2783_ = lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq(v_config_2719_, v___x_2767_, v_val_2782_, v_context_2781_, v___y_2758_, v___y_2759_);
if (lean_obj_tag(v___x_2783_) == 0)
{
lean_dec_ref_known(v___x_2783_, 1);
v___y_2746_ = v___x_2767_;
v___y_2747_ = v_firstInfo_2757_;
goto v___jp_2745_;
}
else
{
lean_object* v_a_2784_; lean_object* v___x_2786_; uint8_t v_isShared_2787_; uint8_t v_isSharedCheck_2791_; 
lean_dec_ref_known(v_firstInfo_2757_, 1);
lean_dec_ref(v___x_2767_);
lean_del_object(v___x_2742_);
lean_del_object(v___x_2737_);
lean_dec_ref(v_config_2719_);
v_a_2784_ = lean_ctor_get(v___x_2783_, 0);
v_isSharedCheck_2791_ = !lean_is_exclusive(v___x_2783_);
if (v_isSharedCheck_2791_ == 0)
{
v___x_2786_ = v___x_2783_;
v_isShared_2787_ = v_isSharedCheck_2791_;
goto v_resetjp_2785_;
}
else
{
lean_inc(v_a_2784_);
lean_dec(v___x_2783_);
v___x_2786_ = lean_box(0);
v_isShared_2787_ = v_isSharedCheck_2791_;
goto v_resetjp_2785_;
}
v_resetjp_2785_:
{
lean_object* v___x_2789_; 
if (v_isShared_2787_ == 0)
{
v___x_2789_ = v___x_2786_;
goto v_reusejp_2788_;
}
else
{
lean_object* v_reuseFailAlloc_2790_; 
v_reuseFailAlloc_2790_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2790_, 0, v_a_2784_);
v___x_2789_ = v_reuseFailAlloc_2790_;
goto v_reusejp_2788_;
}
v_reusejp_2788_:
{
return v___x_2789_;
}
}
}
}
else
{
lean_object* v___x_2792_; lean_object* v___x_2793_; 
lean_dec_ref_known(v___x_2768_, 1);
v___x_2792_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__4);
v___x_2793_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_TacticAnalysis_runPass_spec__0(v_stx_2762_, v___x_2792_, v___y_2758_, v___y_2759_);
lean_dec(v_stx_2762_);
if (lean_obj_tag(v___x_2793_) == 0)
{
lean_dec_ref_known(v___x_2793_, 1);
v___y_2746_ = v___x_2767_;
v___y_2747_ = v_firstInfo_2757_;
goto v___jp_2745_;
}
else
{
lean_object* v_a_2794_; lean_object* v___x_2796_; uint8_t v_isShared_2797_; uint8_t v_isSharedCheck_2801_; 
lean_dec_ref(v___x_2767_);
lean_dec(v_firstInfo_2757_);
lean_del_object(v___x_2742_);
lean_del_object(v___x_2737_);
lean_dec_ref(v_config_2719_);
v_a_2794_ = lean_ctor_get(v___x_2793_, 0);
v_isSharedCheck_2801_ = !lean_is_exclusive(v___x_2793_);
if (v_isSharedCheck_2801_ == 0)
{
v___x_2796_ = v___x_2793_;
v_isShared_2797_ = v_isSharedCheck_2801_;
goto v_resetjp_2795_;
}
else
{
lean_inc(v_a_2794_);
lean_dec(v___x_2793_);
v___x_2796_ = lean_box(0);
v_isShared_2797_ = v_isSharedCheck_2801_;
goto v_resetjp_2795_;
}
v_resetjp_2795_:
{
lean_object* v___x_2799_; 
if (v_isShared_2797_ == 0)
{
v___x_2799_ = v___x_2796_;
goto v_reusejp_2798_;
}
else
{
lean_object* v_reuseFailAlloc_2800_; 
v_reuseFailAlloc_2800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2800_, 0, v_a_2794_);
v___x_2799_ = v_reuseFailAlloc_2800_;
goto v_reusejp_2798_;
}
v_reusejp_2798_:
{
return v___x_2799_;
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
}
v___jp_2727_:
{
size_t v___x_2729_; size_t v___x_2730_; 
v___x_2729_ = ((size_t)1ULL);
v___x_2730_ = lean_usize_add(v_i_2722_, v___x_2729_);
v_i_2722_ = v___x_2730_;
v_b_2723_ = v_a_2728_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___boxed(lean_object* v_config_2807_, lean_object* v_as_2808_, lean_object* v_sz_2809_, lean_object* v_i_2810_, lean_object* v_b_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_){
_start:
{
size_t v_sz_boxed_2815_; size_t v_i_boxed_2816_; lean_object* v_res_2817_; 
v_sz_boxed_2815_ = lean_unbox_usize(v_sz_2809_);
lean_dec(v_sz_2809_);
v_i_boxed_2816_ = lean_unbox_usize(v_i_2810_);
lean_dec(v_i_2810_);
v_res_2817_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1(v_config_2807_, v_as_2808_, v_sz_boxed_2815_, v_i_boxed_2816_, v_b_2811_, v___y_2812_, v___y_2813_);
lean_dec(v___y_2813_);
lean_dec_ref(v___y_2812_);
lean_dec_ref(v_as_2808_);
return v_res_2817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPass(lean_object* v_config_2824_, lean_object* v_seq_2825_, lean_object* v_a_2826_, lean_object* v_a_2827_){
_start:
{
lean_object* v___x_2829_; size_t v_sz_2830_; size_t v___x_2831_; lean_object* v___x_2832_; 
v___x_2829_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1___closed__2));
v_sz_2830_ = lean_array_size(v_seq_2825_);
v___x_2831_ = ((size_t)0ULL);
lean_inc_ref(v_config_2824_);
v___x_2832_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_TacticAnalysis_runPass_spec__1(v_config_2824_, v_seq_2825_, v_sz_2830_, v___x_2831_, v___x_2829_, v_a_2826_, v_a_2827_);
if (lean_obj_tag(v___x_2832_) == 0)
{
lean_object* v_a_2833_; lean_object* v___x_2834_; 
v_a_2833_ = lean_ctor_get(v___x_2832_, 0);
lean_inc(v_a_2833_);
lean_dec_ref_known(v___x_2832_, 1);
v___x_2834_ = l_Lean_Elab_Command_getRef___redArg(v_a_2826_);
if (lean_obj_tag(v___x_2834_) == 0)
{
lean_object* v_a_2835_; lean_object* v___x_2836_; 
v_a_2835_ = lean_ctor_get(v___x_2834_, 0);
lean_inc(v_a_2835_);
lean_dec_ref_known(v___x_2834_, 1);
v___x_2836_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_2826_);
if (lean_obj_tag(v___x_2836_) == 0)
{
lean_object* v___x_2838_; uint8_t v_isShared_2839_; uint8_t v_isSharedCheck_2872_; 
v_isSharedCheck_2872_ = !lean_is_exclusive(v___x_2836_);
if (v_isSharedCheck_2872_ == 0)
{
lean_object* v_unused_2873_; 
v_unused_2873_ = lean_ctor_get(v___x_2836_, 0);
lean_dec(v_unused_2873_);
v___x_2838_ = v___x_2836_;
v_isShared_2839_ = v_isSharedCheck_2872_;
goto v_resetjp_2837_;
}
else
{
lean_dec(v___x_2836_);
v___x_2838_ = lean_box(0);
v_isShared_2839_ = v_isSharedCheck_2872_;
goto v_resetjp_2837_;
}
v_resetjp_2837_:
{
lean_object* v_snd_2840_; lean_object* v_fst_2841_; lean_object* v_fst_2842_; lean_object* v_snd_2843_; lean_object* v___x_2845_; uint8_t v_isShared_2846_; uint8_t v_isSharedCheck_2871_; 
v_snd_2840_ = lean_ctor_get(v_a_2833_, 1);
lean_inc(v_snd_2840_);
v_fst_2841_ = lean_ctor_get(v_a_2833_, 0);
lean_inc(v_fst_2841_);
lean_dec(v_a_2833_);
v_fst_2842_ = lean_ctor_get(v_snd_2840_, 0);
v_snd_2843_ = lean_ctor_get(v_snd_2840_, 1);
v_isSharedCheck_2871_ = !lean_is_exclusive(v_snd_2840_);
if (v_isSharedCheck_2871_ == 0)
{
v___x_2845_ = v_snd_2840_;
v_isShared_2846_ = v_isSharedCheck_2871_;
goto v_resetjp_2844_;
}
else
{
lean_inc(v_snd_2843_);
lean_inc(v_fst_2842_);
lean_dec(v_snd_2840_);
v___x_2845_ = lean_box(0);
v_isShared_2846_ = v_isSharedCheck_2871_;
goto v_resetjp_2844_;
}
v_resetjp_2844_:
{
lean_object* v_quotContext_x3f_2847_; uint8_t v___x_2848_; lean_object* v___x_2849_; 
v_quotContext_x3f_2847_ = lean_ctor_get(v_a_2826_, 5);
v___x_2848_ = 0;
v___x_2849_ = l_Lean_SourceInfo_fromRef(v_a_2835_, v___x_2848_);
lean_dec(v_a_2835_);
if (lean_obj_tag(v_quotContext_x3f_2847_) == 0)
{
lean_object* v___x_2870_; 
v___x_2870_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_TacticAnalysis_testTacticSeq_spec__2___redArg(v_a_2827_);
lean_dec_ref(v___x_2870_);
goto v___jp_2850_;
}
else
{
goto v___jp_2850_;
}
v___jp_2850_:
{
lean_object* v_trigger_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2855_; 
v_trigger_2851_ = lean_ctor_get(v_config_2824_, 0);
v___x_2852_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__0));
v___x_2853_ = ((lean_object*)(lp_mathlib_Mathlib_TacticAnalysis_runPass___closed__1));
lean_inc(v___x_2849_);
if (v_isShared_2846_ == 0)
{
lean_ctor_set_tag(v___x_2845_, 2);
lean_ctor_set(v___x_2845_, 1, v___x_2852_);
lean_ctor_set(v___x_2845_, 0, v___x_2849_);
v___x_2855_ = v___x_2845_;
goto v_reusejp_2854_;
}
else
{
lean_object* v_reuseFailAlloc_2869_; 
v_reuseFailAlloc_2869_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2869_, 0, v___x_2849_);
lean_ctor_set(v_reuseFailAlloc_2869_, 1, v___x_2852_);
v___x_2855_ = v_reuseFailAlloc_2869_;
goto v_reusejp_2854_;
}
v_reusejp_2854_:
{
lean_object* v___x_2856_; lean_object* v___x_2857_; 
v___x_2856_ = l_Lean_Syntax_node1(v___x_2849_, v___x_2853_, v___x_2855_);
lean_inc_ref(v_trigger_2851_);
v___x_2857_ = lean_apply_2(v_trigger_2851_, v_fst_2841_, v___x_2856_);
if (lean_obj_tag(v___x_2857_) == 2)
{
if (lean_obj_tag(v_fst_2842_) == 1)
{
lean_object* v_context_2858_; lean_object* v_val_2859_; lean_object* v___x_2860_; 
lean_del_object(v___x_2838_);
v_context_2858_ = lean_ctor_get(v___x_2857_, 0);
lean_inc(v_context_2858_);
lean_dec_ref_known(v___x_2857_, 1);
v_val_2859_ = lean_ctor_get(v_fst_2842_, 0);
lean_inc(v_val_2859_);
lean_dec_ref_known(v_fst_2842_, 1);
v___x_2860_ = lp_mathlib_Mathlib_TacticAnalysis_testTacticSeq(v_config_2824_, v_snd_2843_, v_val_2859_, v_context_2858_, v_a_2826_, v_a_2827_);
return v___x_2860_;
}
else
{
lean_object* v___x_2861_; lean_object* v___x_2863_; 
lean_dec_ref_known(v___x_2857_, 1);
lean_dec(v_snd_2843_);
lean_dec(v_fst_2842_);
lean_dec_ref(v_config_2824_);
v___x_2861_ = lean_box(0);
if (v_isShared_2839_ == 0)
{
lean_ctor_set(v___x_2838_, 0, v___x_2861_);
v___x_2863_ = v___x_2838_;
goto v_reusejp_2862_;
}
else
{
lean_object* v_reuseFailAlloc_2864_; 
v_reuseFailAlloc_2864_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2864_, 0, v___x_2861_);
v___x_2863_ = v_reuseFailAlloc_2864_;
goto v_reusejp_2862_;
}
v_reusejp_2862_:
{
return v___x_2863_;
}
}
}
else
{
lean_object* v___x_2865_; lean_object* v___x_2867_; 
lean_dec(v___x_2857_);
lean_dec(v_snd_2843_);
lean_dec(v_fst_2842_);
lean_dec_ref(v_config_2824_);
v___x_2865_ = lean_box(0);
if (v_isShared_2839_ == 0)
{
lean_ctor_set(v___x_2838_, 0, v___x_2865_);
v___x_2867_ = v___x_2838_;
goto v_reusejp_2866_;
}
else
{
lean_object* v_reuseFailAlloc_2868_; 
v_reuseFailAlloc_2868_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2868_, 0, v___x_2865_);
v___x_2867_ = v_reuseFailAlloc_2868_;
goto v_reusejp_2866_;
}
v_reusejp_2866_:
{
return v___x_2867_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2874_; lean_object* v___x_2876_; uint8_t v_isShared_2877_; uint8_t v_isSharedCheck_2881_; 
lean_dec(v_a_2835_);
lean_dec(v_a_2833_);
lean_dec_ref(v_config_2824_);
v_a_2874_ = lean_ctor_get(v___x_2836_, 0);
v_isSharedCheck_2881_ = !lean_is_exclusive(v___x_2836_);
if (v_isSharedCheck_2881_ == 0)
{
v___x_2876_ = v___x_2836_;
v_isShared_2877_ = v_isSharedCheck_2881_;
goto v_resetjp_2875_;
}
else
{
lean_inc(v_a_2874_);
lean_dec(v___x_2836_);
v___x_2876_ = lean_box(0);
v_isShared_2877_ = v_isSharedCheck_2881_;
goto v_resetjp_2875_;
}
v_resetjp_2875_:
{
lean_object* v___x_2879_; 
if (v_isShared_2877_ == 0)
{
v___x_2879_ = v___x_2876_;
goto v_reusejp_2878_;
}
else
{
lean_object* v_reuseFailAlloc_2880_; 
v_reuseFailAlloc_2880_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2880_, 0, v_a_2874_);
v___x_2879_ = v_reuseFailAlloc_2880_;
goto v_reusejp_2878_;
}
v_reusejp_2878_:
{
return v___x_2879_;
}
}
}
}
else
{
lean_object* v_a_2882_; lean_object* v___x_2884_; uint8_t v_isShared_2885_; uint8_t v_isSharedCheck_2889_; 
lean_dec(v_a_2833_);
lean_dec_ref(v_config_2824_);
v_a_2882_ = lean_ctor_get(v___x_2834_, 0);
v_isSharedCheck_2889_ = !lean_is_exclusive(v___x_2834_);
if (v_isSharedCheck_2889_ == 0)
{
v___x_2884_ = v___x_2834_;
v_isShared_2885_ = v_isSharedCheck_2889_;
goto v_resetjp_2883_;
}
else
{
lean_inc(v_a_2882_);
lean_dec(v___x_2834_);
v___x_2884_ = lean_box(0);
v_isShared_2885_ = v_isSharedCheck_2889_;
goto v_resetjp_2883_;
}
v_resetjp_2883_:
{
lean_object* v___x_2887_; 
if (v_isShared_2885_ == 0)
{
v___x_2887_ = v___x_2884_;
goto v_reusejp_2886_;
}
else
{
lean_object* v_reuseFailAlloc_2888_; 
v_reuseFailAlloc_2888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2888_, 0, v_a_2882_);
v___x_2887_ = v_reuseFailAlloc_2888_;
goto v_reusejp_2886_;
}
v_reusejp_2886_:
{
return v___x_2887_;
}
}
}
}
else
{
lean_object* v_a_2890_; lean_object* v___x_2892_; uint8_t v_isShared_2893_; uint8_t v_isSharedCheck_2897_; 
lean_dec_ref(v_config_2824_);
v_a_2890_ = lean_ctor_get(v___x_2832_, 0);
v_isSharedCheck_2897_ = !lean_is_exclusive(v___x_2832_);
if (v_isSharedCheck_2897_ == 0)
{
v___x_2892_ = v___x_2832_;
v_isShared_2893_ = v_isSharedCheck_2897_;
goto v_resetjp_2891_;
}
else
{
lean_inc(v_a_2890_);
lean_dec(v___x_2832_);
v___x_2892_ = lean_box(0);
v_isShared_2893_ = v_isSharedCheck_2897_;
goto v_resetjp_2891_;
}
v_resetjp_2891_:
{
lean_object* v___x_2895_; 
if (v_isShared_2893_ == 0)
{
v___x_2895_ = v___x_2892_;
goto v_reusejp_2894_;
}
else
{
lean_object* v_reuseFailAlloc_2896_; 
v_reuseFailAlloc_2896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2896_, 0, v_a_2890_);
v___x_2895_ = v_reuseFailAlloc_2896_;
goto v_reusejp_2894_;
}
v_reusejp_2894_:
{
return v___x_2895_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_runPass___boxed(lean_object* v_config_2898_, lean_object* v_seq_2899_, lean_object* v_a_2900_, lean_object* v_a_2901_, lean_object* v_a_2902_){
_start:
{
lean_object* v_res_2903_; 
v_res_2903_ = lp_mathlib_Mathlib_TacticAnalysis_runPass(v_config_2898_, v_seq_2899_, v_a_2900_, v_a_2901_);
lean_dec(v_a_2901_);
lean_dec_ref(v_a_2900_);
lean_dec_ref(v_seq_2899_);
return v_res_2903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_TacticAnalysis_Config_ofComplex(lean_object* v_config_2904_){
_start:
{
lean_object* v___x_2905_; 
v___x_2905_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_TacticAnalysis_runPass___boxed), 5, 1);
lean_closure_set(v___x_2905_, 0, v_config_2904_);
return v___x_2905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; 
v___x_2917_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__1_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_));
v___x_2918_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn___closed__2_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_));
v___x_2919_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4__spec__0(v___x_2917_, v___x_2918_, v___x_2917_);
return v___x_2919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4____boxed(lean_object* v_a_2920_){
_start:
{
lean_object* v_res_2921_; 
v_res_2921_ = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_();
return v_res_2921_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_ContextInfo(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TacticAnalysis(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_ContextInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_Heartbeats(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* runtime_initialize_Lean_Compiler_IR_CompilerM(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_TacticAnalysis(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_Heartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Compiler_IR_CompilerM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_3788891571____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_tacticAnalysis = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_tacticAnalysis);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_358552301____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysisExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_TacticAnalysis_tacticAnalysisExt);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_2535838111____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__Mathlib_TacticAnalysis_initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1736486024____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_TacticAnalysis_0__initFn_00___x40_Mathlib_Tactic_TacticAnalysis_1223302409____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_tacticAnalysis_dummy = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_tacticAnalysis_dummy);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Util_Heartbeats(uint8_t builtin);
lean_object* initialize_Lean_Server_InfoUtils(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* initialize_Lean_Compiler_IR_CompilerM(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_ContextInfo(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_TacticAnalysis(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_Heartbeats(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_InfoUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Compiler_IR_CompilerM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_ContextInfo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TacticAnalysis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_TacticAnalysis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_TacticAnalysis(builtin);
}
#ifdef __cplusplus
}
#endif
