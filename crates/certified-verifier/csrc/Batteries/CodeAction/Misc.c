// Lean compiler output
// Module: Batteries.CodeAction.Misc
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Induction public meta import Batteries.Lean.Position public meta import Batteries.CodeAction.Attr public meta import Lean.Server.CodeActions.Provider
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
lean_object* l_Lean_FileMap_utf8RangeToLspRange(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_WorkspaceEdit_ofTextEdit(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_findIndentAndIsStart(lean_object*, lean_object*);
lean_object* l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_getStructureFields(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_getFieldInfo_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkDefaultFnOfProjFn(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
uint8_t l_Lean_Syntax_instBEqRange_beq(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getNumArgs(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
lean_object* l_List_tail_x21___redArg(lean_object*);
lean_object* l_Lean_FileMap_lspPosToUtf8Pos(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTrailingSize(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
uint8_t l_Lean_Name_hasNum(lean_object*);
uint8_t l_Lean_Name_isInternal(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_getString_x21(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_hasMacroScopes(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_Server_Snapshots_Snapshot_env(lean_object*);
lean_object* l_Lean_Parser_getTokenTable(lean_object*);
lean_object* l_Lean_Data_Trie_find_x3f___redArg(lean_object*, lean_object*);
extern uint32_t l_Lean_idBeginEscape;
extern uint32_t l_Lean_idEndEscape;
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Syntax_setArgs(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_binderInfo(lean_object*);
uint8_t l_Lean_BinderInfo_isExplicit(uint8_t);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedLocalDecl_default;
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_Range_includes(lean_object*, lean_object*, uint8_t, uint8_t);
uint8_t l_Lean_Syntax_isNone(lean_object*);
extern lean_object* l_instInhabitedError;
lean_object* l_instInhabitedEIO___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_updatePrefix(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_findStack_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_PartialContextInfo_mergeIntoOuter_x3f(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_TermInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkCasesOnName(lean_object*);
lean_object* l_Lean_mkRecName(lean_object*);
extern lean_object* l_Lean_tactic_customEliminators;
lean_object* l_Lean_Meta_getCustomEliminator_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_InfoTree_findInfo_x3f(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isStructure(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_FileMap_utf8PosToLspPos(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_findStack_x3f___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findStack_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_findStack_x3f___lam__1(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findStack_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findStack_x3f(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__2_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__3_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "elabSyntheticHole"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__4_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "elabSorry"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__5_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "sorry"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__6_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__7_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "  "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__5_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__6_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__7_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "where"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___boxed(lean_object**);
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Generate a "};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "skeleton for the structure under construction."};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "quickfix"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "(maximal) "};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__8_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "(minimal) "};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__9_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_CodeAction_instanceStub_spec__4(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_CodeAction_instanceStub_spec__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0_value;
static const lean_array_object lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getExplicitArgs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getAllArgs(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " _"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Batteries.CodeAction.Misc"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__0 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__0_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Batteries.CodeAction.eqnStub"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__1 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__1_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "bad inductive"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__2 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__3;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "| ."};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__4 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__4_value;
static const lean_array_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__5 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__5_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__6 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___boxed(lean_object**);
static const lean_string_object lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Generate a list of equations for a recursive definition."};
static const lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*10 + 0, .m_other = 10, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__0_value),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "by\n"};
static const lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "done"};
static const lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Start a tactic proof."};
static const lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Remove tactics after 'no goals'"};
static const lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_getElimExprNames_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__0;
static const lean_array_object lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_CodeAction_getElimExprNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_getElimExprNames___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_findTermInfo_x3f___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findTermInfo_x3f___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findTermInfo_x3f(lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0 = (const lean_object*)&lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findTermInfoWithCtx_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Batteries_CodeAction_casesExpand_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Batteries_CodeAction_casesExpand_spec__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__2;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__3;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__4;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__5;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__6 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__6_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__8 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__8_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__9;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__10 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__10_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__11;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__12 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__12_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__13;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__14 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__14_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__15;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__16 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__16_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__17;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__18 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__18_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__19;
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__1;
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ih"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(179, 60, 57, 160, 49, 55, 75, 124)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "_ih"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "| "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__1;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__2;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "inductionAlt"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__1_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(249, 42, 154, 153, 222, 213, 69, 136)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "inductionAltLHS"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__3 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__3_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(58, 206, 3, 35, 121, 94, 13, 140)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__5 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__5_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value_aux_1),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__3_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__7 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__7_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__8 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__8_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__9 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__9_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__10 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "elimTarget"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 63, 46, 91, 99, 29, 205, 171)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Generate an explicit pattern match for '"};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__1_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "'."};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__2_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inductionAlts"};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(249, 186, 227, 253, 35, 189, 199, 190)}};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(197, 49, 98, 208, 150, 151, 163, 74)}};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "induction"};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(231, 196, 247, 144, 178, 6, 178, 16)}};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8_value;
static const lean_array_object lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__9 = (const lean_object*)&lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 6, .m_data = "· done"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___closed__0 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___lam__0___boxed(lean_object**);
LEAN_EXPORT uint8_t lp_batteries_List_elem___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_elem___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__1___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__0_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Add subgoals"};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*10 + 0, .m_other = 10, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__1_value),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__2 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(113, 161, 179, 82, 204, 87, 48, 123)}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticSorry"};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__4 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__4_value),LEAN_SCALAR_PTR_LITERAL(254, 186, 126, 140, 105, 148, 113, 102)}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value;
static const lean_string_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticAdmit"};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__6 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__6_value),LEAN_SCALAR_PTR_LITERAL(124, 238, 151, 162, 2, 58, 5, 140)}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__8 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__5_value),((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__9 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__3_value),((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__10 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__10_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
if (lean_obj_tag(v_x_2_) == 0)
{
uint8_t v___x_3_; 
v___x_3_ = 1;
return v___x_3_;
}
else
{
uint8_t v___x_4_; 
v___x_4_ = 0;
return v___x_4_;
}
}
else
{
if (lean_obj_tag(v_x_2_) == 0)
{
uint8_t v___x_5_; 
v___x_5_ = 0;
return v___x_5_;
}
else
{
lean_object* v_val_6_; lean_object* v_val_7_; uint8_t v___x_8_; 
v_val_6_ = lean_ctor_get(v_x_1_, 0);
v_val_7_ = lean_ctor_get(v_x_2_, 0);
v___x_8_ = l_Lean_Syntax_instBEqRange_beq(v_val_6_, v_val_7_);
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0___boxed(lean_object* v_x_9_, lean_object* v_x_10_){
_start:
{
uint8_t v_res_11_; lean_object* v_r_12_; 
v_res_11_ = lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0(v_x_9_, v_x_10_);
lean_dec(v_x_10_);
lean_dec(v_x_9_);
v_r_12_ = lean_box(v_res_11_);
return v_r_12_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_findStack_x3f___lam__0(lean_object* v_target_13_, uint8_t v___x_14_, lean_object* v___x_15_, lean_object* v_s_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; uint8_t v___x_19_; 
lean_inc(v_s_16_);
v___x_17_ = l_Lean_Syntax_getKind(v_s_16_);
v___x_18_ = l_Lean_Syntax_getKind(v_target_13_);
v___x_19_ = lean_name_eq(v___x_17_, v___x_18_);
lean_dec(v___x_18_);
lean_dec(v___x_17_);
if (v___x_19_ == 0)
{
lean_dec(v_s_16_);
return v___x_14_;
}
else
{
lean_object* v___x_20_; uint8_t v___x_21_; 
v___x_20_ = l_Lean_Syntax_getRange_x3f(v_s_16_, v___x_14_);
lean_dec(v_s_16_);
v___x_21_ = lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0(v___x_20_, v___x_15_);
lean_dec(v___x_20_);
return v___x_21_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findStack_x3f___lam__0___boxed(lean_object* v_target_22_, lean_object* v___x_23_, lean_object* v___x_24_, lean_object* v_s_25_){
_start:
{
uint8_t v___x_169__boxed_26_; uint8_t v_res_27_; lean_object* v_r_28_; 
v___x_169__boxed_26_ = lean_unbox(v___x_23_);
v_res_27_ = lp_batteries_Batteries_CodeAction_findStack_x3f___lam__0(v_target_22_, v___x_169__boxed_26_, v___x_24_, v_s_25_);
lean_dec(v___x_24_);
v_r_28_ = lean_box(v_res_27_);
return v_r_28_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_findStack_x3f___lam__1(uint8_t v___x_29_, lean_object* v_val_30_, lean_object* v_x_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = l_Lean_Syntax_getRange_x3f(v_x_31_, v___x_29_);
if (lean_obj_tag(v___x_32_) == 0)
{
return v___x_29_;
}
else
{
lean_object* v_val_33_; uint8_t v___x_34_; 
v_val_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc(v_val_33_);
lean_dec_ref_known(v___x_32_, 1);
v___x_34_ = l_Lean_Syntax_Range_includes(v_val_33_, v_val_30_, v___x_29_, v___x_29_);
lean_dec(v_val_33_);
return v___x_34_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findStack_x3f___lam__1___boxed(lean_object* v___x_35_, lean_object* v_val_36_, lean_object* v_x_37_){
_start:
{
uint8_t v___x_185__boxed_38_; uint8_t v_res_39_; lean_object* v_r_40_; 
v___x_185__boxed_38_ = lean_unbox(v___x_35_);
v_res_39_ = lp_batteries_Batteries_CodeAction_findStack_x3f___lam__1(v___x_185__boxed_38_, v_val_36_, v_x_37_);
lean_dec(v_x_37_);
lean_dec_ref(v_val_36_);
v_r_40_ = lean_box(v_res_39_);
return v_r_40_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findStack_x3f(lean_object* v_root_41_, lean_object* v_target_42_){
_start:
{
uint8_t v___x_43_; lean_object* v___x_44_; 
v___x_43_ = 0;
v___x_44_ = l_Lean_Syntax_getRange_x3f(v_target_42_, v___x_43_);
if (lean_obj_tag(v___x_44_) == 0)
{
lean_object* v___x_45_; 
lean_dec(v_target_42_);
lean_dec(v_root_41_);
v___x_45_ = lean_box(0);
return v___x_45_;
}
else
{
lean_object* v_val_46_; lean_object* v___x_47_; lean_object* v___f_48_; lean_object* v___x_49_; lean_object* v___f_50_; lean_object* v___x_51_; 
v_val_46_ = lean_ctor_get(v___x_44_, 0);
lean_inc(v_val_46_);
v___x_47_ = lean_box(v___x_43_);
v___f_48_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_findStack_x3f___lam__0___boxed), 4, 3);
lean_closure_set(v___f_48_, 0, v_target_42_);
lean_closure_set(v___f_48_, 1, v___x_47_);
lean_closure_set(v___f_48_, 2, v___x_44_);
v___x_49_ = lean_box(v___x_43_);
v___f_50_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_findStack_x3f___lam__1___boxed), 3, 2);
lean_closure_set(v___f_50_, 0, v___x_49_);
lean_closure_set(v___f_50_, 1, v_val_46_);
v___x_51_ = l_Lean_Syntax_findStack_x3f(v_root_41_, v___f_50_, v___f_48_);
return v___x_51_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString(lean_object* v_x_60_, lean_object* v_x_61_){
_start:
{
if (lean_obj_tag(v_x_60_) == 1)
{
lean_object* v_pre_62_; 
v_pre_62_ = lean_ctor_get(v_x_60_, 0);
if (lean_obj_tag(v_pre_62_) == 1)
{
lean_object* v_pre_63_; 
v_pre_63_ = lean_ctor_get(v_pre_62_, 0);
if (lean_obj_tag(v_pre_63_) == 1)
{
lean_object* v_pre_64_; 
v_pre_64_ = lean_ctor_get(v_pre_63_, 0);
if (lean_obj_tag(v_pre_64_) == 1)
{
lean_object* v_pre_65_; 
v_pre_65_ = lean_ctor_get(v_pre_64_, 0);
if (lean_obj_tag(v_pre_65_) == 0)
{
lean_object* v_str_66_; lean_object* v_str_67_; lean_object* v_str_68_; lean_object* v_str_69_; lean_object* v___x_70_; uint8_t v___x_71_; 
v_str_66_ = lean_ctor_get(v_x_60_, 1);
v_str_67_ = lean_ctor_get(v_pre_62_, 1);
v_str_68_ = lean_ctor_get(v_pre_63_, 1);
v_str_69_ = lean_ctor_get(v_pre_64_, 1);
v___x_70_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__0));
v___x_71_ = lean_string_dec_eq(v_str_69_, v___x_70_);
if (v___x_71_ == 0)
{
lean_object* v___x_72_; 
v___x_72_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_72_;
}
else
{
lean_object* v___x_73_; uint8_t v___x_74_; 
v___x_73_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__2));
v___x_74_ = lean_string_dec_eq(v_str_68_, v___x_73_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; 
v___x_75_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_75_;
}
else
{
lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_76_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__3));
v___x_77_ = lean_string_dec_eq(v_str_67_, v___x_76_);
if (v___x_77_ == 0)
{
lean_object* v___x_78_; 
v___x_78_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_78_;
}
else
{
lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_79_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__4));
v___x_80_ = lean_string_dec_eq(v_str_66_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_81_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__5));
v___x_82_ = lean_string_dec_eq(v_str_66_, v___x_81_);
if (v___x_82_ == 0)
{
lean_object* v___x_83_; 
v___x_83_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_83_;
}
else
{
lean_object* v___x_84_; 
v___x_84_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__6));
return v___x_84_;
}
}
else
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__7));
v___x_86_ = lean_string_append(v___x_85_, v_x_61_);
return v___x_86_;
}
}
}
}
}
else
{
lean_object* v___x_87_; 
v___x_87_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_87_;
}
}
else
{
lean_object* v___x_88_; 
v___x_88_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_88_;
}
}
else
{
lean_object* v___x_89_; 
v___x_89_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_89_;
}
}
else
{
lean_object* v___x_90_; 
v___x_90_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_90_;
}
}
else
{
lean_object* v___x_91_; 
v___x_91_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__1));
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_holeKindToHoleString___boxed(lean_object* v_x_92_, lean_object* v_x_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_batteries_Batteries_CodeAction_holeKindToHoleString(v_x_92_, v_x_93_);
lean_dec_ref(v_x_93_);
lean_dec(v_x_92_);
return v_res_94_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable_spec__0(lean_object* v_fieldInfo_95_, lean_object* v_env_96_, lean_object* v_x_97_){
_start:
{
if (lean_obj_tag(v_x_97_) == 0)
{
uint8_t v___x_98_; 
lean_dec_ref(v_env_96_);
lean_dec_ref(v_fieldInfo_95_);
v___x_98_ = 0;
return v___x_98_;
}
else
{
lean_object* v_head_99_; lean_object* v_tail_100_; lean_object* v_fieldName_101_; lean_object* v___x_102_; lean_object* v___x_103_; uint8_t v___x_104_; uint8_t v___x_105_; 
v_head_99_ = lean_ctor_get(v_x_97_, 0);
lean_inc(v_head_99_);
v_tail_100_ = lean_ctor_get(v_x_97_, 1);
lean_inc(v_tail_100_);
lean_dec_ref_known(v_x_97_, 2);
v_fieldName_101_ = lean_ctor_get(v_fieldInfo_95_, 0);
lean_inc(v_fieldName_101_);
v___x_102_ = l_Lean_Name_append(v_head_99_, v_fieldName_101_);
v___x_103_ = l_Lean_mkDefaultFnOfProjFn(v___x_102_);
v___x_104_ = 1;
lean_inc_ref(v_env_96_);
v___x_105_ = l_Lean_Environment_contains(v_env_96_, v___x_103_, v___x_104_);
if (v___x_105_ == 0)
{
v_x_97_ = v_tail_100_;
goto _start;
}
else
{
lean_dec(v_tail_100_);
lean_dec_ref(v_env_96_);
lean_dec_ref(v_fieldInfo_95_);
return v___x_105_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable_spec__0___boxed(lean_object* v_fieldInfo_107_, lean_object* v_env_108_, lean_object* v_x_109_){
_start:
{
uint8_t v_res_110_; lean_object* v_r_111_; 
v_res_110_ = lp_batteries_List_any___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable_spec__0(v_fieldInfo_107_, v_env_108_, v_x_109_);
v_r_111_ = lean_box(v_res_110_);
return v_r_111_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable(lean_object* v_env_112_, lean_object* v_fieldInfo_113_, lean_object* v_stack_114_){
_start:
{
lean_object* v_autoParam_x3f_115_; 
v_autoParam_x3f_115_ = lean_ctor_get(v_fieldInfo_113_, 3);
if (lean_obj_tag(v_autoParam_x3f_115_) == 0)
{
lean_object* v_projFn_116_; lean_object* v___x_117_; uint8_t v___x_118_; uint8_t v___x_119_; 
v_projFn_116_ = lean_ctor_get(v_fieldInfo_113_, 1);
lean_inc(v_projFn_116_);
v___x_117_ = l_Lean_mkDefaultFnOfProjFn(v_projFn_116_);
v___x_118_ = 1;
lean_inc_ref(v_env_112_);
v___x_119_ = l_Lean_Environment_contains(v_env_112_, v___x_117_, v___x_118_);
if (v___x_119_ == 0)
{
uint8_t v___x_120_; 
v___x_120_ = lp_batteries_List_any___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable_spec__0(v_fieldInfo_113_, v_env_112_, v_stack_114_);
return v___x_120_;
}
else
{
lean_dec(v_stack_114_);
lean_dec_ref(v_fieldInfo_113_);
lean_dec_ref(v_env_112_);
return v___x_119_;
}
}
else
{
uint8_t v___x_121_; 
lean_dec(v_stack_114_);
lean_dec_ref(v_fieldInfo_113_);
lean_dec_ref(v_env_112_);
v___x_121_ = 1;
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable___boxed(lean_object* v_env_122_, lean_object* v_fieldInfo_123_, lean_object* v_stack_124_){
_start:
{
uint8_t v_res_125_; lean_object* v_r_126_; 
v_res_125_ = lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable(v_env_122_, v_fieldInfo_123_, v_stack_124_);
v_r_126_ = lean_box(v_res_125_);
return v_r_126_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0(lean_object* v_env_127_, lean_object* v_structName_128_, lean_object* v_stack_129_, lean_object* v_as_130_, size_t v_i_131_, size_t v_stop_132_, lean_object* v_b_133_){
_start:
{
lean_object* v___y_135_; uint8_t v___x_139_; 
v___x_139_ = lean_usize_dec_eq(v_i_131_, v_stop_132_);
if (v___x_139_ == 0)
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = lean_array_uget_borrowed(v_as_130_, v_i_131_);
lean_inc(v___x_140_);
lean_inc(v_structName_128_);
lean_inc_ref(v_env_127_);
v___x_141_ = l_Lean_getFieldInfo_x3f(v_env_127_, v_structName_128_, v___x_140_);
if (lean_obj_tag(v___x_141_) == 1)
{
lean_object* v_val_142_; lean_object* v_subobject_x3f_143_; 
v_val_142_ = lean_ctor_get(v___x_141_, 0);
lean_inc(v_val_142_);
lean_dec_ref_known(v___x_141_, 1);
v_subobject_x3f_143_ = lean_ctor_get(v_val_142_, 2);
if (lean_obj_tag(v_subobject_x3f_143_) == 1)
{
lean_object* v_val_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
lean_inc_ref(v_subobject_x3f_143_);
lean_dec(v_val_142_);
v_val_144_ = lean_ctor_get(v_subobject_x3f_143_, 0);
lean_inc(v_val_144_);
lean_dec_ref_known(v_subobject_x3f_143_, 1);
lean_inc(v_stack_129_);
lean_inc(v_structName_128_);
v___x_145_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_145_, 0, v_structName_128_);
lean_ctor_set(v___x_145_, 1, v_stack_129_);
lean_inc_ref(v_env_127_);
v___x_146_ = lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields(v_env_127_, v_val_144_, v_b_133_, v___x_145_);
v___y_135_ = v___x_146_;
goto v___jp_134_;
}
else
{
uint8_t v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
lean_inc(v_stack_129_);
lean_inc_ref(v_env_127_);
v___x_147_ = lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_isAutofillable(v_env_127_, v_val_142_, v_stack_129_);
v___x_148_ = lean_box(v___x_147_);
lean_inc(v___x_140_);
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_140_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = lean_array_push(v_b_133_, v___x_149_);
v___y_135_ = v___x_150_;
goto v___jp_134_;
}
}
else
{
lean_dec(v___x_141_);
v___y_135_ = v_b_133_;
goto v___jp_134_;
}
}
else
{
lean_dec(v_stack_129_);
lean_dec(v_structName_128_);
lean_dec_ref(v_env_127_);
return v_b_133_;
}
v___jp_134_:
{
size_t v___x_136_; size_t v___x_137_; 
v___x_136_ = ((size_t)1ULL);
v___x_137_ = lean_usize_add(v_i_131_, v___x_136_);
v_i_131_ = v___x_137_;
v_b_133_ = v___y_135_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields(lean_object* v_env_151_, lean_object* v_structName_152_, lean_object* v_fields_153_, lean_object* v_stack_154_){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
lean_inc(v_structName_152_);
lean_inc_ref(v_env_151_);
v___x_155_ = l_Lean_getStructureFields(v_env_151_, v_structName_152_);
v___x_156_ = lean_unsigned_to_nat(0u);
v___x_157_ = lean_array_get_size(v___x_155_);
v___x_158_ = lean_nat_dec_lt(v___x_156_, v___x_157_);
if (v___x_158_ == 0)
{
lean_dec_ref(v___x_155_);
lean_dec(v_stack_154_);
lean_dec(v_structName_152_);
lean_dec_ref(v_env_151_);
return v_fields_153_;
}
else
{
uint8_t v___x_159_; 
v___x_159_ = lean_nat_dec_le(v___x_157_, v___x_157_);
if (v___x_159_ == 0)
{
if (v___x_158_ == 0)
{
lean_dec_ref(v___x_155_);
lean_dec(v_stack_154_);
lean_dec(v_structName_152_);
lean_dec_ref(v_env_151_);
return v_fields_153_;
}
else
{
size_t v___x_160_; size_t v___x_161_; lean_object* v___x_162_; 
v___x_160_ = ((size_t)0ULL);
v___x_161_ = lean_usize_of_nat(v___x_157_);
v___x_162_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0(v_env_151_, v_structName_152_, v_stack_154_, v___x_155_, v___x_160_, v___x_161_, v_fields_153_);
lean_dec_ref(v___x_155_);
return v___x_162_;
}
}
else
{
size_t v___x_163_; size_t v___x_164_; lean_object* v___x_165_; 
v___x_163_ = ((size_t)0ULL);
v___x_164_ = lean_usize_of_nat(v___x_157_);
v___x_165_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0(v_env_151_, v_structName_152_, v_stack_154_, v___x_155_, v___x_163_, v___x_164_, v_fields_153_);
lean_dec_ref(v___x_155_);
return v___x_165_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0___boxed(lean_object* v_env_166_, lean_object* v_structName_167_, lean_object* v_stack_168_, lean_object* v_as_169_, lean_object* v_i_170_, lean_object* v_stop_171_, lean_object* v_b_172_){
_start:
{
size_t v_i_boxed_173_; size_t v_stop_boxed_174_; lean_object* v_res_175_; 
v_i_boxed_173_ = lean_unbox_usize(v_i_170_);
lean_dec(v_i_170_);
v_stop_boxed_174_ = lean_unbox_usize(v_stop_171_);
lean_dec(v_stop_171_);
v_res_175_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields_spec__0(v_env_166_, v_structName_167_, v_stack_168_, v_as_169_, v_i_boxed_173_, v_stop_boxed_174_, v_b_172_);
lean_dec_ref(v_as_169_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(lean_object* v___y_176_){
_start:
{
lean_object* v_doc_178_; lean_object* v___x_179_; 
v_doc_178_ = lean_ctor_get(v___y_176_, 1);
lean_inc_ref(v_doc_178_);
v___x_179_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_179_, 0, v_doc_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0___boxed(lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v___y_180_);
lean_dec_ref(v___y_180_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(lean_object* v_msg_183_){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = lean_unsigned_to_nat(0u);
v___x_185_ = lean_panic_fn_borrowed(v___x_184_, v_msg_183_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(lean_object* v_x_186_, lean_object* v_x_187_){
_start:
{
lean_object* v_zero_188_; uint8_t v_isZero_189_; 
v_zero_188_ = lean_unsigned_to_nat(0u);
v_isZero_189_ = lean_nat_dec_eq(v_x_186_, v_zero_188_);
if (v_isZero_189_ == 1)
{
lean_dec(v_x_186_);
return v_x_187_;
}
else
{
uint32_t v___x_190_; lean_object* v_one_191_; lean_object* v_n_192_; lean_object* v___x_193_; 
v___x_190_ = 32;
v_one_191_ = lean_unsigned_to_nat(1u);
v_n_192_ = lean_nat_sub(v_x_186_, v_one_191_);
lean_dec(v_x_186_);
v___x_193_ = lean_string_push(v_x_187_, v___x_190_);
v_x_186_ = v_n_192_;
v_x_187_ = v___x_193_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3(uint8_t v___x_198_, lean_object* v___x_199_, lean_object* v___x_200_, uint8_t v_minimal_201_, lean_object* v_as_202_, size_t v_sz_203_, size_t v_i_204_, lean_object* v_b_205_){
_start:
{
lean_object* v_a_208_; uint8_t v___x_212_; 
v___x_212_ = lean_usize_dec_lt(v_i_204_, v_sz_203_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; 
v___x_213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_213_, 0, v_b_205_);
return v___x_213_;
}
else
{
lean_object* v_a_214_; lean_object* v_fst_215_; lean_object* v_snd_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_254_; 
v_a_214_ = lean_array_uget(v_as_202_, v_i_204_);
v_fst_215_ = lean_ctor_get(v_a_214_, 0);
v_snd_216_ = lean_ctor_get(v_a_214_, 1);
v_isSharedCheck_254_ = !lean_is_exclusive(v_a_214_);
if (v_isSharedCheck_254_ == 0)
{
v___x_218_ = v_a_214_;
v_isShared_219_ = v_isSharedCheck_254_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_snd_216_);
lean_inc(v_fst_215_);
lean_dec(v_a_214_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_254_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v_str_221_; uint8_t v_first_222_; lean_object* v_fst_234_; lean_object* v_snd_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_253_; 
v_fst_234_ = lean_ctor_get(v_b_205_, 0);
v_snd_235_ = lean_ctor_get(v_b_205_, 1);
v_isSharedCheck_253_ = !lean_is_exclusive(v_b_205_);
if (v_isSharedCheck_253_ == 0)
{
v___x_237_ = v_b_205_;
v_isShared_238_ = v_isSharedCheck_253_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_snd_235_);
lean_inc(v_fst_234_);
lean_dec(v_b_205_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_253_;
goto v_resetjp_236_;
}
v___jp_220_:
{
lean_object* v_elaborator_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_232_; 
v_elaborator_223_ = lean_ctor_get(v___x_199_, 0);
v___x_224_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_215_, v___x_198_);
v___x_225_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__0));
lean_inc_ref(v___x_224_);
v___x_226_ = lean_string_append(v___x_224_, v___x_225_);
v___x_227_ = lp_batteries_Batteries_CodeAction_holeKindToHoleString(v_elaborator_223_, v___x_224_);
lean_dec_ref(v___x_224_);
v___x_228_ = lean_string_append(v___x_226_, v___x_227_);
lean_dec_ref(v___x_227_);
v___x_229_ = lean_string_append(v_str_221_, v___x_228_);
lean_dec_ref(v___x_228_);
v___x_230_ = lean_box(v_first_222_);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 1, v___x_230_);
lean_ctor_set(v___x_218_, 0, v___x_229_);
v___x_232_ = v___x_218_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v___x_229_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v___x_230_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
v_a_208_ = v___x_232_;
goto v___jp_207_;
}
}
v_resetjp_236_:
{
uint8_t v___y_240_; 
if (v_minimal_201_ == 0)
{
lean_del_object(v___x_237_);
lean_dec(v_snd_216_);
v___y_240_ = v_minimal_201_;
goto v___jp_239_;
}
else
{
uint8_t v___x_248_; 
v___x_248_ = lean_unbox(v_snd_216_);
if (v___x_248_ == 0)
{
uint8_t v___x_249_; 
lean_del_object(v___x_237_);
v___x_249_ = lean_unbox(v_snd_216_);
lean_dec(v_snd_216_);
v___y_240_ = v___x_249_;
goto v___jp_239_;
}
else
{
lean_object* v___x_251_; 
lean_del_object(v___x_218_);
lean_dec(v_snd_216_);
lean_dec(v_fst_215_);
if (v_isShared_238_ == 0)
{
v___x_251_ = v___x_237_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v_fst_234_);
lean_ctor_set(v_reuseFailAlloc_252_, 1, v_snd_235_);
v___x_251_ = v_reuseFailAlloc_252_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
v_a_208_ = v___x_251_;
goto v___jp_207_;
}
}
}
v___jp_239_:
{
uint8_t v___x_241_; 
v___x_241_ = lean_unbox(v_snd_235_);
if (v___x_241_ == 0)
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_242_ = lean_string_append(v_fst_234_, v___x_200_);
v___x_243_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__1));
v___x_244_ = lean_string_append(v___x_242_, v___x_243_);
v___x_245_ = lean_unbox(v_snd_235_);
lean_dec(v_snd_235_);
v_str_221_ = v___x_244_;
v_first_222_ = v___x_245_;
goto v___jp_220_;
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; 
lean_dec(v_snd_235_);
v___x_246_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__2));
v___x_247_ = lean_string_append(v_fst_234_, v___x_246_);
v_str_221_ = v___x_247_;
v_first_222_ = v___y_240_;
goto v___jp_220_;
}
}
}
}
}
v___jp_207_:
{
size_t v___x_209_; size_t v___x_210_; 
v___x_209_ = ((size_t)1ULL);
v___x_210_ = lean_usize_add(v_i_204_, v___x_209_);
v_i_204_ = v___x_210_;
v_b_205_ = v_a_208_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___boxed(lean_object* v___x_255_, lean_object* v___x_256_, lean_object* v___x_257_, lean_object* v_minimal_258_, lean_object* v_as_259_, lean_object* v_sz_260_, lean_object* v_i_261_, lean_object* v_b_262_, lean_object* v___y_263_){
_start:
{
uint8_t v___x_8558__boxed_264_; uint8_t v_minimal_boxed_265_; size_t v_sz_boxed_266_; size_t v_i_boxed_267_; lean_object* v_res_268_; 
v___x_8558__boxed_264_ = lean_unbox(v___x_255_);
v_minimal_boxed_265_ = lean_unbox(v_minimal_258_);
v_sz_boxed_266_ = lean_unbox_usize(v_sz_260_);
lean_dec(v_sz_260_);
v_i_boxed_267_ = lean_unbox_usize(v_i_261_);
lean_dec(v_i_261_);
v_res_268_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3(v___x_8558__boxed_264_, v___x_256_, v___x_257_, v_minimal_boxed_265_, v_as_259_, v_sz_boxed_266_, v_i_boxed_267_, v_b_262_);
lean_dec_ref(v_as_259_);
lean_dec_ref(v___x_257_);
lean_dec_ref(v___x_256_);
return v_res_268_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_272_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__2));
v___x_273_ = lean_unsigned_to_nat(14u);
v___x_274_ = lean_unsigned_to_nat(22u);
v___x_275_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__1));
v___x_276_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__0));
v___x_277_ = l_mkPanicMessageWithDecl(v___x_276_, v___x_275_, v___x_274_, v___x_273_, v___x_272_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0(lean_object* v___x_283_, lean_object* v___y_284_, lean_object* v_text_285_, lean_object* v___x_286_, lean_object* v___x_287_, lean_object* v___x_288_, lean_object* v___x_289_, lean_object* v___x_290_, lean_object* v___x_291_, lean_object* v___x_292_, lean_object* v___x_293_, lean_object* v_a_294_, lean_object* v_stx_295_, lean_object* v___x_296_, uint8_t v___x_297_, lean_object* v_toElabInfo_298_, uint8_t v_minimal_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___y_303_; lean_object* v___y_304_; lean_object* v___y_305_; lean_object* v_str_315_; lean_object* v_fst_322_; lean_object* v_snd_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_365_; 
v_fst_322_ = lean_ctor_get(v___x_283_, 0);
v_snd_323_ = lean_ctor_get(v___x_283_, 1);
v_isSharedCheck_365_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_365_ == 0)
{
v___x_325_ = v___x_283_;
v_isShared_326_ = v_isSharedCheck_365_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_snd_323_);
lean_inc(v_fst_322_);
lean_dec(v___x_283_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_365_;
goto v_resetjp_324_;
}
v___jp_302_:
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_306_, 0, v___y_284_);
lean_ctor_set(v___x_306_, 1, v___y_305_);
v___x_307_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_285_, v___x_306_);
v___x_308_ = lean_box(0);
lean_inc_n(v___x_286_, 2);
v___x_309_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_309_, 0, v___x_307_);
lean_ctor_set(v___x_309_, 1, v___y_304_);
lean_ctor_set(v___x_309_, 2, v___x_308_);
lean_ctor_set(v___x_309_, 3, v___x_286_);
v___x_310_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___y_303_, v___x_309_);
v___x_311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
v___x_312_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_312_, 0, v___x_286_);
lean_ctor_set(v___x_312_, 1, v___x_286_);
lean_ctor_set(v___x_312_, 2, v___x_287_);
lean_ctor_set(v___x_312_, 3, v___x_288_);
lean_ctor_set(v___x_312_, 4, v___x_289_);
lean_ctor_set(v___x_312_, 5, v___x_290_);
lean_ctor_set(v___x_312_, 6, v___x_291_);
lean_ctor_set(v___x_312_, 7, v___x_311_);
lean_ctor_set(v___x_312_, 8, v___x_292_);
lean_ctor_set(v___x_312_, 9, v___x_293_);
v___x_313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_313_, 0, v___x_312_);
return v___x_313_;
}
v___jp_314_:
{
lean_object* v___x_316_; uint8_t v___x_317_; lean_object* v___x_318_; 
v___x_316_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_294_);
v___x_317_ = 0;
v___x_318_ = l_Lean_Syntax_getTailPos_x3f(v_stx_295_, v___x_317_);
if (lean_obj_tag(v___x_318_) == 0)
{
lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_319_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_320_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_319_);
v___y_303_ = v___x_316_;
v___y_304_ = v_str_315_;
v___y_305_ = v___x_320_;
goto v___jp_302_;
}
else
{
lean_object* v_val_321_; 
v_val_321_ = lean_ctor_get(v___x_318_, 0);
lean_inc(v_val_321_);
lean_dec_ref_known(v___x_318_, 1);
v___y_303_ = v___x_316_;
v___y_304_ = v_str_315_;
v___y_305_ = v_val_321_;
goto v___jp_302_;
}
}
v_resetjp_324_:
{
lean_object* v___x_327_; lean_object* v___x_328_; uint8_t v___y_330_; lean_object* v___y_331_; uint8_t v___y_332_; lean_object* v___y_359_; 
v___x_327_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4));
v___x_328_ = lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(v_fst_322_, v___x_327_);
if (lean_obj_tag(v___y_300_) == 0)
{
goto v___jp_362_;
}
else
{
if (v___x_297_ == 0)
{
goto v___jp_362_;
}
else
{
lean_object* v___x_364_; 
v___x_364_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__8));
v___y_359_ = v___x_364_;
goto v___jp_358_;
}
}
v___jp_329_:
{
lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_333_ = lean_box(v___y_332_);
lean_inc_ref(v___y_331_);
if (v_isShared_326_ == 0)
{
lean_ctor_set(v___x_325_, 1, v___x_333_);
lean_ctor_set(v___x_325_, 0, v___y_331_);
v___x_335_ = v___x_325_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v___y_331_);
lean_ctor_set(v_reuseFailAlloc_357_, 1, v___x_333_);
v___x_335_ = v_reuseFailAlloc_357_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
size_t v_sz_336_; size_t v___x_337_; lean_object* v___x_338_; 
v_sz_336_ = lean_array_size(v___x_296_);
v___x_337_ = ((size_t)0ULL);
v___x_338_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3(v___x_297_, v_toElabInfo_298_, v___x_328_, v_minimal_299_, v___x_296_, v_sz_336_, v___x_337_, v___x_335_);
if (lean_obj_tag(v___x_338_) == 0)
{
lean_object* v_a_339_; 
v_a_339_ = lean_ctor_get(v___x_338_, 0);
lean_inc(v_a_339_);
lean_dec_ref_known(v___x_338_, 1);
if (v___y_330_ == 0)
{
lean_object* v_fst_340_; 
lean_dec_ref(v___x_328_);
lean_dec(v_snd_323_);
v_fst_340_ = lean_ctor_get(v_a_339_, 0);
lean_inc(v_fst_340_);
lean_dec(v_a_339_);
v_str_315_ = v_fst_340_;
goto v___jp_314_;
}
else
{
uint8_t v___x_341_; 
v___x_341_ = lean_unbox(v_snd_323_);
lean_dec(v_snd_323_);
if (v___x_341_ == 0)
{
lean_object* v_fst_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v_fst_342_ = lean_ctor_get(v_a_339_, 0);
lean_inc(v_fst_342_);
lean_dec(v_a_339_);
v___x_343_ = lean_string_append(v_fst_342_, v___x_328_);
lean_dec_ref(v___x_328_);
v___x_344_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__5));
v___x_345_ = lean_string_append(v___x_343_, v___x_344_);
v_str_315_ = v___x_345_;
goto v___jp_314_;
}
else
{
lean_object* v_fst_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec_ref(v___x_328_);
v_fst_346_ = lean_ctor_get(v_a_339_, 0);
lean_inc(v_fst_346_);
lean_dec(v_a_339_);
v___x_347_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__6));
v___x_348_ = lean_string_append(v_fst_346_, v___x_347_);
v_str_315_ = v___x_348_;
goto v___jp_314_;
}
}
}
else
{
lean_object* v_a_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_356_; 
lean_dec_ref(v___x_328_);
lean_dec(v_snd_323_);
lean_dec_ref(v_a_294_);
lean_dec(v___x_293_);
lean_dec(v___x_292_);
lean_dec(v___x_291_);
lean_dec(v___x_290_);
lean_dec(v___x_289_);
lean_dec(v___x_288_);
lean_dec_ref(v___x_287_);
lean_dec(v___x_286_);
lean_dec_ref(v_text_285_);
lean_dec(v___y_284_);
v_a_349_ = lean_ctor_get(v___x_338_, 0);
v_isSharedCheck_356_ = !lean_is_exclusive(v___x_338_);
if (v_isSharedCheck_356_ == 0)
{
v___x_351_ = v___x_338_;
v_isShared_352_ = v_isSharedCheck_356_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_a_349_);
lean_dec(v___x_338_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_356_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
lean_object* v___x_354_; 
if (v_isShared_352_ == 0)
{
v___x_354_ = v___x_351_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v_a_349_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
}
}
v___jp_358_:
{
if (lean_obj_tag(v___y_300_) == 0)
{
if (v___x_297_ == 0)
{
v___y_330_ = v___x_297_;
v___y_331_ = v___y_359_;
v___y_332_ = v___x_297_;
goto v___jp_329_;
}
else
{
uint8_t v___x_360_; 
v___x_360_ = lean_unbox(v_snd_323_);
v___y_330_ = v___x_297_;
v___y_331_ = v___y_359_;
v___y_332_ = v___x_360_;
goto v___jp_329_;
}
}
else
{
uint8_t v___x_361_; 
v___x_361_ = 0;
v___y_330_ = v___x_361_;
v___y_331_ = v___y_359_;
v___y_332_ = v___x_361_;
goto v___jp_329_;
}
}
v___jp_362_:
{
lean_object* v___x_363_; 
v___x_363_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__7));
v___y_359_ = v___x_363_;
goto v___jp_358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___boxed(lean_object** _args){
lean_object* v___x_366_ = _args[0];
lean_object* v___y_367_ = _args[1];
lean_object* v_text_368_ = _args[2];
lean_object* v___x_369_ = _args[3];
lean_object* v___x_370_ = _args[4];
lean_object* v___x_371_ = _args[5];
lean_object* v___x_372_ = _args[6];
lean_object* v___x_373_ = _args[7];
lean_object* v___x_374_ = _args[8];
lean_object* v___x_375_ = _args[9];
lean_object* v___x_376_ = _args[10];
lean_object* v_a_377_ = _args[11];
lean_object* v_stx_378_ = _args[12];
lean_object* v___x_379_ = _args[13];
lean_object* v___x_380_ = _args[14];
lean_object* v_toElabInfo_381_ = _args[15];
lean_object* v_minimal_382_ = _args[16];
lean_object* v___y_383_ = _args[17];
lean_object* v___y_384_ = _args[18];
_start:
{
uint8_t v___x_8697__boxed_385_; uint8_t v_minimal_boxed_386_; lean_object* v_res_387_; 
v___x_8697__boxed_385_ = lean_unbox(v___x_380_);
v_minimal_boxed_386_ = lean_unbox(v_minimal_382_);
v_res_387_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0(v___x_366_, v___y_367_, v_text_368_, v___x_369_, v___x_370_, v___x_371_, v___x_372_, v___x_373_, v___x_374_, v___x_375_, v___x_376_, v_a_377_, v_stx_378_, v___x_379_, v___x_8697__boxed_385_, v_toElabInfo_381_, v_minimal_boxed_386_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v_toElabInfo_381_);
lean_dec_ref(v___x_379_);
lean_dec(v_stx_378_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1(lean_object* v_snap_404_, lean_object* v_toElabInfo_405_, lean_object* v_a_406_, lean_object* v___x_407_, uint8_t v___x_408_, lean_object* v___x_409_, uint8_t v___y_410_, uint8_t v_minimal_411_){
_start:
{
lean_object* v___x_412_; lean_object* v___y_414_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___y_418_; lean_object* v___y_419_; lean_object* v___y_420_; lean_object* v___y_421_; lean_object* v___y_422_; lean_object* v___y_423_; lean_object* v___y_424_; lean_object* v___y_436_; lean_object* v___y_437_; lean_object* v___y_438_; lean_object* v___y_439_; lean_object* v___y_440_; lean_object* v___y_441_; lean_object* v___y_442_; lean_object* v___y_443_; lean_object* v___y_444_; lean_object* v___y_445_; lean_object* v___x_451_; lean_object* v___y_453_; 
v___x_412_ = lean_box(0);
v___x_451_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__0));
if (v___y_410_ == 0)
{
if (v_minimal_411_ == 0)
{
lean_object* v___x_475_; 
v___x_475_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__8));
v___y_453_ = v___x_475_;
goto v___jp_452_;
}
else
{
lean_object* v___x_476_; 
v___x_476_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__9));
v___y_453_ = v___x_476_;
goto v___jp_452_;
}
}
else
{
lean_object* v___x_477_; 
v___x_477_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10));
v___y_453_ = v___x_477_;
goto v___jp_452_;
}
v___jp_413_:
{
lean_object* v_toEditableDocumentCore_425_; lean_object* v_meta_426_; lean_object* v_text_427_; lean_object* v_source_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___y_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v_toEditableDocumentCore_425_ = lean_ctor_get(v_a_406_, 0);
v_meta_426_ = lean_ctor_get(v_toEditableDocumentCore_425_, 0);
v_text_427_ = lean_ctor_get(v_meta_426_, 3);
lean_inc_ref(v_text_427_);
v_source_428_ = lean_ctor_get(v_text_427_, 0);
lean_inc_ref(v_source_428_);
v___x_429_ = lp_batteries_Lean_findIndentAndIsStart(v_source_428_, v___y_424_);
v___x_430_ = lean_box(v___x_408_);
v___x_431_ = lean_box(v_minimal_411_);
lean_inc(v___y_420_);
v___y_432_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___boxed), 19, 18);
lean_closure_set(v___y_432_, 0, v___x_429_);
lean_closure_set(v___y_432_, 1, v___y_424_);
lean_closure_set(v___y_432_, 2, v_text_427_);
lean_closure_set(v___y_432_, 3, v___x_412_);
lean_closure_set(v___y_432_, 4, v___y_416_);
lean_closure_set(v___y_432_, 5, v___y_420_);
lean_closure_set(v___y_432_, 6, v___y_421_);
lean_closure_set(v___y_432_, 7, v___y_419_);
lean_closure_set(v___y_432_, 8, v___y_423_);
lean_closure_set(v___y_432_, 9, v___y_422_);
lean_closure_set(v___y_432_, 10, v___y_415_);
lean_closure_set(v___y_432_, 11, v_a_406_);
lean_closure_set(v___y_432_, 12, v___y_418_);
lean_closure_set(v___y_432_, 13, v___x_407_);
lean_closure_set(v___y_432_, 14, v___x_430_);
lean_closure_set(v___y_432_, 15, v_toElabInfo_405_);
lean_closure_set(v___y_432_, 16, v___x_431_);
lean_closure_set(v___y_432_, 17, v___y_414_);
v___x_433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_433_, 0, v___y_432_);
v___x_434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_434_, 0, v___y_417_);
lean_ctor_set(v___x_434_, 1, v___x_433_);
return v___x_434_;
}
v___jp_435_:
{
uint8_t v___x_446_; lean_object* v___x_447_; 
v___x_446_ = 0;
v___x_447_ = l_Lean_Syntax_getPos_x3f(v___y_438_, v___x_446_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_449_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_448_);
v___y_414_ = v___y_445_;
v___y_415_ = v___y_436_;
v___y_416_ = v___y_437_;
v___y_417_ = v___y_444_;
v___y_418_ = v___y_438_;
v___y_419_ = v___y_440_;
v___y_420_ = v___y_439_;
v___y_421_ = v___y_441_;
v___y_422_ = v___y_442_;
v___y_423_ = v___y_443_;
v___y_424_ = v___x_449_;
goto v___jp_413_;
}
else
{
lean_object* v_val_450_; 
v_val_450_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_val_450_);
lean_dec_ref_known(v___x_447_, 1);
v___y_414_ = v___y_445_;
v___y_415_ = v___y_436_;
v___y_416_ = v___y_437_;
v___y_417_ = v___y_444_;
v___y_418_ = v___y_438_;
v___y_419_ = v___y_440_;
v___y_420_ = v___y_439_;
v___y_421_ = v___y_441_;
v___y_422_ = v___y_442_;
v___y_423_ = v___y_443_;
v___y_424_ = v_val_450_;
goto v___jp_413_;
}
}
v___jp_452_:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v_stx_461_; lean_object* v_stx_462_; lean_object* v___x_463_; 
v___x_454_ = lean_string_append(v___x_451_, v___y_453_);
v___x_455_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__1));
v___x_456_ = lean_string_append(v___x_454_, v___x_455_);
v___x_457_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3));
v___x_458_ = lean_box(v_minimal_411_);
v___x_459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_459_, 0, v___x_458_);
lean_inc_ref(v___x_459_);
lean_inc_ref(v___x_456_);
v___x_460_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_460_, 0, v___x_412_);
lean_ctor_set(v___x_460_, 1, v___x_412_);
lean_ctor_set(v___x_460_, 2, v___x_456_);
lean_ctor_set(v___x_460_, 3, v___x_457_);
lean_ctor_set(v___x_460_, 4, v___x_412_);
lean_ctor_set(v___x_460_, 5, v___x_459_);
lean_ctor_set(v___x_460_, 6, v___x_412_);
lean_ctor_set(v___x_460_, 7, v___x_412_);
lean_ctor_set(v___x_460_, 8, v___x_412_);
lean_ctor_set(v___x_460_, 9, v___x_412_);
v_stx_461_ = lean_ctor_get(v_snap_404_, 0);
lean_inc(v_stx_461_);
lean_dec_ref(v_snap_404_);
v_stx_462_ = lean_ctor_get(v_toElabInfo_405_, 1);
lean_inc(v_stx_462_);
v___x_463_ = lp_batteries_Batteries_CodeAction_findStack_x3f(v_stx_461_, v_stx_462_);
if (lean_obj_tag(v___x_463_) == 0)
{
lean_inc(v_stx_462_);
v___y_436_ = v___x_412_;
v___y_437_ = v___x_456_;
v___y_438_ = v_stx_462_;
v___y_439_ = v___x_457_;
v___y_440_ = v___x_459_;
v___y_441_ = v___x_412_;
v___y_442_ = v___x_412_;
v___y_443_ = v___x_412_;
v___y_444_ = v___x_460_;
v___y_445_ = v___x_412_;
goto v___jp_435_;
}
else
{
lean_object* v_val_464_; 
v_val_464_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_val_464_);
lean_dec_ref_known(v___x_463_, 1);
if (lean_obj_tag(v_val_464_) == 1)
{
lean_object* v_tail_465_; 
v_tail_465_ = lean_ctor_get(v_val_464_, 1);
lean_inc(v_tail_465_);
lean_dec_ref_known(v_val_464_, 2);
if (lean_obj_tag(v_tail_465_) == 1)
{
lean_object* v_head_466_; lean_object* v_fst_467_; lean_object* v___x_468_; lean_object* v___x_469_; uint8_t v___x_470_; 
v_head_466_ = lean_ctor_get(v_tail_465_, 0);
lean_inc(v_head_466_);
lean_dec_ref_known(v_tail_465_, 2);
v_fst_467_ = lean_ctor_get(v_head_466_, 0);
lean_inc_n(v_fst_467_, 2);
lean_dec(v_head_466_);
v___x_468_ = l_Lean_Syntax_getKind(v_fst_467_);
v___x_469_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__7));
v___x_470_ = lean_name_eq(v___x_468_, v___x_469_);
lean_dec(v___x_468_);
if (v___x_470_ == 0)
{
lean_dec(v_fst_467_);
lean_inc(v_stx_462_);
v___y_436_ = v___x_412_;
v___y_437_ = v___x_456_;
v___y_438_ = v_stx_462_;
v___y_439_ = v___x_457_;
v___y_440_ = v___x_459_;
v___y_441_ = v___x_412_;
v___y_442_ = v___x_412_;
v___y_443_ = v___x_412_;
v___y_444_ = v___x_460_;
v___y_445_ = v___x_412_;
goto v___jp_435_;
}
else
{
lean_object* v___x_471_; uint8_t v___x_472_; lean_object* v___x_473_; 
v___x_471_ = l_Lean_Syntax_getArg(v_fst_467_, v___x_409_);
lean_dec(v_fst_467_);
v___x_472_ = 0;
v___x_473_ = l_Lean_Syntax_getPos_x3f(v___x_471_, v___x_472_);
lean_dec(v___x_471_);
if (lean_obj_tag(v___x_473_) == 0)
{
lean_inc(v_stx_462_);
v___y_436_ = v___x_412_;
v___y_437_ = v___x_456_;
v___y_438_ = v_stx_462_;
v___y_439_ = v___x_457_;
v___y_440_ = v___x_459_;
v___y_441_ = v___x_412_;
v___y_442_ = v___x_412_;
v___y_443_ = v___x_412_;
v___y_444_ = v___x_460_;
v___y_445_ = v___x_473_;
goto v___jp_435_;
}
else
{
lean_object* v_val_474_; 
v_val_474_ = lean_ctor_get(v___x_473_, 0);
lean_inc(v_val_474_);
lean_inc(v_stx_462_);
v___y_414_ = v___x_473_;
v___y_415_ = v___x_412_;
v___y_416_ = v___x_456_;
v___y_417_ = v___x_460_;
v___y_418_ = v_stx_462_;
v___y_419_ = v___x_459_;
v___y_420_ = v___x_457_;
v___y_421_ = v___x_412_;
v___y_422_ = v___x_412_;
v___y_423_ = v___x_412_;
v___y_424_ = v_val_474_;
goto v___jp_413_;
}
}
}
else
{
lean_dec(v_tail_465_);
lean_inc(v_stx_462_);
v___y_436_ = v___x_412_;
v___y_437_ = v___x_456_;
v___y_438_ = v_stx_462_;
v___y_439_ = v___x_457_;
v___y_440_ = v___x_459_;
v___y_441_ = v___x_412_;
v___y_442_ = v___x_412_;
v___y_443_ = v___x_412_;
v___y_444_ = v___x_460_;
v___y_445_ = v___x_412_;
goto v___jp_435_;
}
}
else
{
lean_dec(v_val_464_);
lean_inc(v_stx_462_);
v___y_436_ = v___x_412_;
v___y_437_ = v___x_456_;
v___y_438_ = v_stx_462_;
v___y_439_ = v___x_457_;
v___y_440_ = v___x_459_;
v___y_441_ = v___x_412_;
v___y_442_ = v___x_412_;
v___y_443_ = v___x_412_;
v___y_444_ = v___x_460_;
v___y_445_ = v___x_412_;
goto v___jp_435_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___boxed(lean_object* v_snap_478_, lean_object* v_toElabInfo_479_, lean_object* v_a_480_, lean_object* v___x_481_, lean_object* v___x_482_, lean_object* v___x_483_, lean_object* v___y_484_, lean_object* v_minimal_485_){
_start:
{
uint8_t v___x_8947__boxed_486_; uint8_t v___y_8949__boxed_487_; uint8_t v_minimal_boxed_488_; lean_object* v_res_489_; 
v___x_8947__boxed_486_ = lean_unbox(v___x_482_);
v___y_8949__boxed_487_ = lean_unbox(v___y_484_);
v_minimal_boxed_488_ = lean_unbox(v_minimal_485_);
v_res_489_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1(v_snap_478_, v_toElabInfo_479_, v_a_480_, v___x_481_, v___x_8947__boxed_486_, v___x_483_, v___y_8949__boxed_487_, v_minimal_boxed_488_);
lean_dec(v___x_483_);
return v_res_489_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_CodeAction_instanceStub_spec__4(lean_object* v_as_490_, size_t v_i_491_, size_t v_stop_492_){
_start:
{
uint8_t v___x_493_; 
v___x_493_ = lean_usize_dec_eq(v_i_491_, v_stop_492_);
if (v___x_493_ == 0)
{
lean_object* v___x_494_; lean_object* v_snd_495_; uint8_t v___x_496_; 
v___x_494_ = lean_array_uget_borrowed(v_as_490_, v_i_491_);
v_snd_495_ = lean_ctor_get(v___x_494_, 1);
v___x_496_ = lean_unbox(v_snd_495_);
if (v___x_496_ == 0)
{
size_t v___x_497_; size_t v___x_498_; 
v___x_497_ = ((size_t)1ULL);
v___x_498_ = lean_usize_add(v_i_491_, v___x_497_);
v_i_491_ = v___x_498_;
goto _start;
}
else
{
uint8_t v___x_500_; 
v___x_500_ = lean_unbox(v_snd_495_);
return v___x_500_;
}
}
else
{
uint8_t v___x_501_; 
v___x_501_ = 0;
return v___x_501_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_CodeAction_instanceStub_spec__4___boxed(lean_object* v_as_502_, lean_object* v_i_503_, lean_object* v_stop_504_){
_start:
{
size_t v_i_boxed_505_; size_t v_stop_boxed_506_; uint8_t v_res_507_; lean_object* v_r_508_; 
v_i_boxed_505_ = lean_unbox_usize(v_i_503_);
lean_dec(v_i_503_);
v_stop_boxed_506_ = lean_unbox_usize(v_stop_504_);
lean_dec(v_stop_504_);
v_res_507_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_CodeAction_instanceStub_spec__4(v_as_502_, v_i_boxed_505_, v_stop_boxed_506_);
lean_dec_ref(v_as_502_);
v_r_508_ = lean_box(v_res_507_);
return v_r_508_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg(lean_object* v_snap_513_, lean_object* v_ctx_514_, lean_object* v_info_515_, lean_object* v_a_516_){
_start:
{
lean_object* v_expectedType_x3f_518_; 
v_expectedType_x3f_518_ = lean_ctor_get(v_info_515_, 2);
if (lean_obj_tag(v_expectedType_x3f_518_) == 1)
{
lean_object* v_toElabInfo_519_; lean_object* v_val_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v_toElabInfo_519_ = lean_ctor_get(v_info_515_, 0);
lean_inc_ref(v_toElabInfo_519_);
v_val_520_ = lean_ctor_get(v_expectedType_x3f_518_, 0);
lean_inc(v_val_520_);
v___x_521_ = lean_alloc_closure((void*)(l_Lean_Meta_whnf___boxed), 6, 1);
lean_closure_set(v___x_521_, 0, v_val_520_);
v___x_522_ = l_Lean_Elab_TermInfo_runMetaM___redArg(v_info_515_, v_ctx_514_, v___x_521_);
if (lean_obj_tag(v___x_522_) == 0)
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_575_; 
v_a_523_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_575_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_575_ == 0)
{
v___x_525_ = v___x_522_;
v_isShared_526_ = v_isSharedCheck_575_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___x_522_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_575_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_527_; 
v___x_527_ = l_Lean_Expr_getAppFn(v_a_523_);
lean_dec(v_a_523_);
if (lean_obj_tag(v___x_527_) == 4)
{
lean_object* v_declName_528_; lean_object* v___x_529_; uint8_t v___x_530_; 
v_declName_528_ = lean_ctor_get(v___x_527_, 0);
lean_inc_n(v_declName_528_, 2);
lean_dec_ref_known(v___x_527_, 2);
v___x_529_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_513_);
lean_inc_ref(v___x_529_);
v___x_530_ = l_Lean_isStructure(v___x_529_, v_declName_528_);
if (v___x_530_ == 0)
{
lean_object* v___x_531_; lean_object* v___x_533_; 
lean_dec_ref(v___x_529_);
lean_dec(v_declName_528_);
lean_dec_ref(v_toElabInfo_519_);
lean_dec_ref(v_snap_513_);
v___x_531_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_526_ == 0)
{
lean_ctor_set(v___x_525_, 0, v___x_531_);
v___x_533_ = v___x_525_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v___x_531_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
else
{
lean_object* v___x_535_; lean_object* v_a_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_570_; 
v___x_535_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v_a_516_);
v_a_536_ = lean_ctor_get(v___x_535_, 0);
v_isSharedCheck_570_ = !lean_is_exclusive(v___x_535_);
if (v_isSharedCheck_570_ == 0)
{
v___x_538_ = v___x_535_;
v_isShared_539_ = v_isSharedCheck_570_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_a_536_);
lean_dec(v___x_535_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_570_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; uint8_t v___y_545_; uint8_t v___y_556_; lean_object* v___x_564_; uint8_t v___x_565_; 
v___x_540_ = lean_unsigned_to_nat(0u);
v___x_541_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__1));
v___x_542_ = lean_box(0);
v___x_543_ = lp_batteries___private_Batteries_CodeAction_Misc_0__Batteries_CodeAction_instanceStub_collectFields(v___x_529_, v_declName_528_, v___x_541_, v___x_542_);
v___x_564_ = lean_array_get_size(v___x_543_);
v___x_565_ = lean_nat_dec_lt(v___x_540_, v___x_564_);
if (v___x_565_ == 0)
{
v___y_556_ = v___x_530_;
goto v___jp_555_;
}
else
{
if (v___x_565_ == 0)
{
v___y_556_ = v___x_530_;
goto v___jp_555_;
}
else
{
size_t v___x_566_; size_t v___x_567_; uint8_t v___x_568_; 
v___x_566_ = ((size_t)0ULL);
v___x_567_ = lean_usize_of_nat(v___x_564_);
v___x_568_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_CodeAction_instanceStub_spec__4(v___x_543_, v___x_566_, v___x_567_);
if (v___x_568_ == 0)
{
v___y_556_ = v___x_530_;
goto v___jp_555_;
}
else
{
uint8_t v___x_569_; 
lean_del_object(v___x_525_);
v___x_569_ = 0;
v___y_545_ = v___x_569_;
goto v___jp_544_;
}
}
}
v___jp_544_:
{
lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_553_; 
lean_inc_ref(v___x_543_);
lean_inc(v_a_536_);
lean_inc_ref(v_toElabInfo_519_);
lean_inc_ref(v_snap_513_);
v___x_546_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1(v_snap_513_, v_toElabInfo_519_, v_a_536_, v___x_543_, v___x_530_, v___x_540_, v___y_545_, v___x_530_);
v___x_547_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1(v_snap_513_, v_toElabInfo_519_, v_a_536_, v___x_543_, v___x_530_, v___x_540_, v___y_545_, v___y_545_);
v___x_548_ = lean_unsigned_to_nat(2u);
v___x_549_ = lean_mk_empty_array_with_capacity(v___x_548_);
v___x_550_ = lean_array_push(v___x_549_, v___x_546_);
v___x_551_ = lean_array_push(v___x_550_, v___x_547_);
if (v_isShared_539_ == 0)
{
lean_ctor_set(v___x_538_, 0, v___x_551_);
v___x_553_ = v___x_538_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___x_551_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
v___jp_555_:
{
if (v___y_556_ == 0)
{
lean_del_object(v___x_525_);
v___y_545_ = v___y_556_;
goto v___jp_544_;
}
else
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_562_; 
lean_del_object(v___x_538_);
v___x_557_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1(v_snap_513_, v_toElabInfo_519_, v_a_536_, v___x_543_, v___x_530_, v___x_540_, v___y_556_, v___x_530_);
v___x_558_ = lean_unsigned_to_nat(1u);
v___x_559_ = lean_mk_empty_array_with_capacity(v___x_558_);
v___x_560_ = lean_array_push(v___x_559_, v___x_557_);
if (v_isShared_526_ == 0)
{
lean_ctor_set(v___x_525_, 0, v___x_560_);
v___x_562_ = v___x_525_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_560_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
}
}
else
{
lean_object* v___x_571_; lean_object* v___x_573_; 
lean_dec_ref(v___x_527_);
lean_dec_ref(v_toElabInfo_519_);
lean_dec_ref(v_snap_513_);
v___x_571_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_526_ == 0)
{
lean_ctor_set(v___x_525_, 0, v___x_571_);
v___x_573_ = v___x_525_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_571_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
}
}
else
{
lean_object* v_a_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_584_; 
lean_dec_ref(v_toElabInfo_519_);
lean_dec_ref(v_snap_513_);
v_a_576_ = lean_ctor_get(v___x_522_, 0);
v_isSharedCheck_584_ = !lean_is_exclusive(v___x_522_);
if (v_isSharedCheck_584_ == 0)
{
v___x_578_ = v___x_522_;
v_isShared_579_ = v_isSharedCheck_584_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_a_576_);
lean_dec(v___x_522_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_584_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
lean_object* v___x_580_; lean_object* v___x_582_; 
v___x_580_ = l_Lean_Server_RequestError_ofIoError(v_a_576_);
if (v_isShared_579_ == 0)
{
lean_ctor_set(v___x_578_, 0, v___x_580_);
v___x_582_ = v___x_578_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v___x_580_);
v___x_582_ = v_reuseFailAlloc_583_;
goto v_reusejp_581_;
}
v_reusejp_581_:
{
return v___x_582_;
}
}
}
}
else
{
lean_object* v___x_585_; lean_object* v___x_586_; 
lean_dec_ref(v_info_515_);
lean_dec_ref(v_ctx_514_);
lean_dec_ref(v_snap_513_);
v___x_585_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_586_, 0, v___x_585_);
return v___x_586_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___redArg___boxed(lean_object* v_snap_587_, lean_object* v_ctx_588_, lean_object* v_info_589_, lean_object* v_a_590_, lean_object* v_a_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg(v_snap_587_, v_ctx_588_, v_info_589_, v_a_590_);
lean_dec_ref(v_a_590_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub(lean_object* v_x_593_, lean_object* v_snap_594_, lean_object* v_ctx_595_, lean_object* v_info_596_, lean_object* v_a_597_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_batteries_Batteries_CodeAction_instanceStub___redArg(v_snap_594_, v_ctx_595_, v_info_596_, v_a_597_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_instanceStub___boxed(lean_object* v_x_600_, lean_object* v_snap_601_, lean_object* v_ctx_602_, lean_object* v_info_603_, lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_batteries_Batteries_CodeAction_instanceStub(v_x_600_, v_snap_601_, v_ctx_602_, v_info_603_, v_a_604_);
lean_dec_ref(v_a_604_);
lean_dec_ref(v_x_600_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getExplicitArgs(lean_object* v_x_607_, lean_object* v_x_608_){
_start:
{
if (lean_obj_tag(v_x_607_) == 7)
{
lean_object* v_binderName_609_; lean_object* v_body_610_; uint8_t v_binderInfo_611_; uint8_t v___x_612_; 
v_binderName_609_ = lean_ctor_get(v_x_607_, 0);
lean_inc(v_binderName_609_);
v_body_610_ = lean_ctor_get(v_x_607_, 2);
lean_inc_ref(v_body_610_);
v_binderInfo_611_ = lean_ctor_get_uint8(v_x_607_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_607_, 3);
v___x_612_ = l_Lean_BinderInfo_isExplicit(v_binderInfo_611_);
if (v___x_612_ == 0)
{
lean_dec(v_binderName_609_);
v_x_607_ = v_body_610_;
goto _start;
}
else
{
lean_object* v___x_614_; 
v___x_614_ = lean_array_push(v_x_608_, v_binderName_609_);
v_x_607_ = v_body_610_;
v_x_608_ = v___x_614_;
goto _start;
}
}
else
{
lean_dec_ref(v_x_607_);
return v_x_608_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getAllArgs(lean_object* v_x_616_, lean_object* v_x_617_){
_start:
{
if (lean_obj_tag(v_x_616_) == 7)
{
lean_object* v_binderName_618_; lean_object* v_body_619_; lean_object* v___x_620_; 
v_binderName_618_ = lean_ctor_get(v_x_616_, 0);
lean_inc(v_binderName_618_);
v_body_619_ = lean_ctor_get(v_x_616_, 2);
lean_inc_ref(v_body_619_);
lean_dec_ref_known(v_x_616_, 3);
v___x_620_ = lean_array_push(v_x_617_, v_binderName_618_);
v_x_616_ = v_body_619_;
v_x_617_ = v___x_620_;
goto _start;
}
else
{
lean_dec_ref(v_x_616_);
return v_x_617_;
}
}
}
static lean_object* _init_lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___closed__0(void){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_622_ = l_instInhabitedError;
v___x_623_ = lean_alloc_closure((void*)(l_instInhabitedEIO___aux__1___boxed), 4, 3);
lean_closure_set(v___x_623_, 0, lean_box(0));
lean_closure_set(v___x_623_, 1, lean_box(0));
lean_closure_set(v___x_623_, 2, v___x_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0(lean_object* v_msg_624_){
_start:
{
lean_object* v___x_626_; lean_object* v___x_7149__overap_627_; lean_object* v___x_628_; 
v___x_626_ = lean_obj_once(&lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___closed__0, &lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___closed__0_once, _init_lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___closed__0);
v___x_7149__overap_627_ = lean_panic_fn_borrowed(v___x_626_, v_msg_624_);
v___x_628_ = lean_apply_1(v___x_7149__overap_627_, lean_box(0));
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0___boxed(lean_object* v_msg_629_, lean_object* v___y_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0(v_msg_629_);
return v_res_631_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1(lean_object* v_as_633_, size_t v_sz_634_, size_t v_i_635_, lean_object* v_b_636_){
_start:
{
lean_object* v___y_639_; uint8_t v___x_646_; 
v___x_646_ = lean_usize_dec_lt(v_i_635_, v_sz_634_);
if (v___x_646_ == 0)
{
lean_object* v___x_647_; 
v___x_647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_647_, 0, v_b_636_);
return v___x_647_;
}
else
{
lean_object* v_a_648_; uint8_t v___x_649_; 
v_a_648_ = lean_array_uget_borrowed(v_as_633_, v_i_635_);
v___x_649_ = l_Lean_Name_hasNum(v_a_648_);
if (v___x_649_ == 0)
{
uint8_t v___x_650_; 
v___x_650_ = l_Lean_Name_isInternal(v_a_648_);
if (v___x_650_ == 0)
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; 
v___x_651_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__2));
lean_inc(v_a_648_);
v___x_652_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_a_648_, v___x_646_);
v___x_653_ = lean_string_append(v___x_651_, v___x_652_);
lean_dec_ref(v___x_652_);
v___y_639_ = v___x_653_;
goto v___jp_638_;
}
else
{
goto v___jp_644_;
}
}
else
{
goto v___jp_644_;
}
}
v___jp_638_:
{
lean_object* v___x_640_; size_t v___x_641_; size_t v___x_642_; 
v___x_640_ = lean_string_append(v_b_636_, v___y_639_);
lean_dec_ref(v___y_639_);
v___x_641_ = ((size_t)1ULL);
v___x_642_ = lean_usize_add(v_i_635_, v___x_641_);
v_i_635_ = v___x_642_;
v_b_636_ = v___x_640_;
goto _start;
}
v___jp_644_:
{
lean_object* v___x_645_; 
v___x_645_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___closed__0));
v___y_639_ = v___x_645_;
goto v___jp_638_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___boxed(lean_object* v_as_654_, lean_object* v_sz_655_, lean_object* v_i_656_, lean_object* v_b_657_, lean_object* v___y_658_){
_start:
{
size_t v_sz_boxed_659_; size_t v_i_boxed_660_; lean_object* v_res_661_; 
v_sz_boxed_659_ = lean_unbox_usize(v_sz_655_);
lean_dec(v_sz_655_);
v_i_boxed_660_ = lean_unbox_usize(v_i_656_);
lean_dec(v_i_656_);
v_res_661_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1(v_as_654_, v_sz_boxed_659_, v_i_boxed_660_, v_b_657_);
lean_dec_ref(v_as_654_);
return v_res_661_;
}
}
static lean_object* _init_lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v___x_665_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__2));
v___x_666_ = lean_unsigned_to_nat(57u);
v___x_667_ = lean_unsigned_to_nat(187u);
v___x_668_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__1));
v___x_669_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__0));
v___x_670_ = l_mkPanicMessageWithDecl(v___x_669_, v___x_668_, v___x_667_, v___x_666_, v___x_665_);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg(lean_object* v_snap_675_, lean_object* v___x_676_, lean_object* v___x_677_, lean_object* v_as_x27_678_, lean_object* v_b_679_){
_start:
{
if (lean_obj_tag(v_as_x27_678_) == 0)
{
lean_object* v___x_681_; 
v___x_681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_681_, 0, v_b_679_);
return v___x_681_;
}
else
{
lean_object* v_head_682_; lean_object* v_tail_683_; uint8_t v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v_head_682_ = lean_ctor_get(v_as_x27_678_, 0);
v_tail_683_ = lean_ctor_get(v_as_x27_678_, 1);
v___x_696_ = 0;
v___x_697_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_675_);
lean_inc(v_head_682_);
v___x_698_ = l_Lean_Environment_find_x3f(v___x_697_, v_head_682_, v___x_696_);
if (lean_obj_tag(v___x_698_) == 1)
{
lean_object* v_val_699_; 
v_val_699_ = lean_ctor_get(v___x_698_, 0);
lean_inc(v_val_699_);
lean_dec_ref_known(v___x_698_, 1);
if (lean_obj_tag(v_val_699_) == 6)
{
lean_object* v_val_700_; lean_object* v_toConstantVal_701_; lean_object* v_type_702_; lean_object* v___x_703_; lean_object* v___x_704_; uint8_t v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; size_t v_sz_713_; size_t v___x_714_; lean_object* v___x_715_; 
v_val_700_ = lean_ctor_get(v_val_699_, 0);
lean_inc_ref(v_val_700_);
lean_dec_ref_known(v_val_699_, 1);
v_toConstantVal_701_ = lean_ctor_get(v_val_700_, 0);
lean_inc_ref(v_toConstantVal_701_);
lean_dec_ref(v_val_700_);
v_type_702_ = lean_ctor_get(v_toConstantVal_701_, 2);
lean_inc_ref(v_type_702_);
lean_dec_ref(v_toConstantVal_701_);
v___x_703_ = lean_box(0);
lean_inc(v_head_682_);
v___x_704_ = l_Lean_Name_updatePrefix(v_head_682_, v___x_703_);
v___x_705_ = 1;
v___x_706_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_704_, v___x_705_);
v___x_707_ = lean_string_append(v_b_679_, v___x_676_);
v___x_708_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__4));
v___x_709_ = lean_string_append(v___x_708_, v___x_706_);
v___x_710_ = lean_string_append(v___x_707_, v___x_709_);
lean_dec_ref(v___x_709_);
v___x_711_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__5));
v___x_712_ = lp_batteries_Batteries_CodeAction_getExplicitArgs(v_type_702_, v___x_711_);
v_sz_713_ = lean_array_size(v___x_712_);
v___x_714_ = ((size_t)0ULL);
v___x_715_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1(v___x_712_, v_sz_713_, v___x_714_, v___x_710_);
lean_dec_ref(v___x_712_);
if (lean_obj_tag(v___x_715_) == 0)
{
lean_object* v_a_716_; lean_object* v_elaborator_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; 
v_a_716_ = lean_ctor_get(v___x_715_, 0);
lean_inc(v_a_716_);
lean_dec_ref_known(v___x_715_, 1);
v_elaborator_717_ = lean_ctor_get(v___x_677_, 0);
v___x_718_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__6));
v___x_719_ = lp_batteries_Batteries_CodeAction_holeKindToHoleString(v_elaborator_717_, v___x_706_);
lean_dec_ref(v___x_706_);
v___x_720_ = lean_string_append(v___x_718_, v___x_719_);
lean_dec_ref(v___x_719_);
v___x_721_ = lean_string_append(v_a_716_, v___x_720_);
lean_dec_ref(v___x_720_);
v_as_x27_678_ = v_tail_683_;
v_b_679_ = v___x_721_;
goto _start;
}
else
{
lean_dec_ref(v___x_706_);
return v___x_715_;
}
}
else
{
lean_dec(v_val_699_);
goto v___jp_684_;
}
}
else
{
lean_dec(v___x_698_);
goto v___jp_684_;
}
v___jp_684_:
{
lean_object* v___x_685_; lean_object* v___x_686_; 
v___x_685_ = lean_obj_once(&lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__3, &lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__3_once, _init_lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__3);
v___x_686_ = lp_batteries_panic___at___00Batteries_CodeAction_eqnStub_spec__0(v___x_685_);
if (lean_obj_tag(v___x_686_) == 0)
{
lean_dec_ref_known(v___x_686_, 1);
v_as_x27_678_ = v_tail_683_;
goto _start;
}
else
{
lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_695_; 
lean_dec_ref(v_b_679_);
v_a_688_ = lean_ctor_get(v___x_686_, 0);
v_isSharedCheck_695_ = !lean_is_exclusive(v___x_686_);
if (v_isSharedCheck_695_ == 0)
{
v___x_690_ = v___x_686_;
v_isShared_691_ = v_isSharedCheck_695_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_686_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_695_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___x_693_; 
if (v_isShared_691_ == 0)
{
v___x_693_ = v___x_690_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v_a_688_);
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
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___boxed(lean_object* v_snap_723_, lean_object* v___x_724_, lean_object* v___x_725_, lean_object* v_as_x27_726_, lean_object* v_b_727_, lean_object* v___y_728_){
_start:
{
lean_object* v_res_729_; 
v_res_729_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg(v_snap_723_, v___x_724_, v___x_725_, v_as_x27_726_, v_b_727_);
lean_dec(v_as_x27_726_);
lean_dec_ref(v___x_725_);
lean_dec_ref(v___x_724_);
lean_dec_ref(v_snap_723_);
return v_res_729_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0(lean_object* v___x_731_, lean_object* v___y_732_, lean_object* v_text_733_, lean_object* v___x_734_, lean_object* v___x_735_, lean_object* v___x_736_, lean_object* v___x_737_, lean_object* v___x_738_, lean_object* v___x_739_, lean_object* v___x_740_, lean_object* v___x_741_, lean_object* v_val_742_, lean_object* v_snap_743_, lean_object* v_toElabInfo_744_, lean_object* v_a_745_, lean_object* v_stx_746_, uint8_t v___x_747_){
_start:
{
lean_object* v___y_750_; lean_object* v___y_751_; lean_object* v___y_752_; lean_object* v_fst_761_; lean_object* v_snd_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___y_766_; uint8_t v___x_784_; 
v_fst_761_ = lean_ctor_get(v___x_731_, 0);
lean_inc(v_fst_761_);
v_snd_762_ = lean_ctor_get(v___x_731_, 1);
lean_inc(v_snd_762_);
lean_dec_ref(v___x_731_);
v___x_763_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___closed__0));
v___x_764_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4));
v___x_784_ = lean_unbox(v_snd_762_);
lean_dec(v_snd_762_);
if (v___x_784_ == 0)
{
lean_object* v___x_785_; lean_object* v___x_786_; 
v___x_785_ = lean_unsigned_to_nat(2u);
v___x_786_ = lean_nat_add(v_fst_761_, v___x_785_);
lean_dec(v_fst_761_);
v___y_766_ = v___x_786_;
goto v___jp_765_;
}
else
{
v___y_766_ = v_fst_761_;
goto v___jp_765_;
}
v___jp_749_:
{
lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v___x_753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_753_, 0, v___y_732_);
lean_ctor_set(v___x_753_, 1, v___y_752_);
v___x_754_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_733_, v___x_753_);
v___x_755_ = lean_box(0);
lean_inc_n(v___x_734_, 2);
v___x_756_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_756_, 0, v___x_754_);
lean_ctor_set(v___x_756_, 1, v___y_751_);
lean_ctor_set(v___x_756_, 2, v___x_755_);
lean_ctor_set(v___x_756_, 3, v___x_734_);
v___x_757_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___y_750_, v___x_756_);
v___x_758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_758_, 0, v___x_757_);
v___x_759_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_759_, 0, v___x_734_);
lean_ctor_set(v___x_759_, 1, v___x_734_);
lean_ctor_set(v___x_759_, 2, v___x_735_);
lean_ctor_set(v___x_759_, 3, v___x_736_);
lean_ctor_set(v___x_759_, 4, v___x_737_);
lean_ctor_set(v___x_759_, 5, v___x_738_);
lean_ctor_set(v___x_759_, 6, v___x_739_);
lean_ctor_set(v___x_759_, 7, v___x_758_);
lean_ctor_set(v___x_759_, 8, v___x_740_);
lean_ctor_set(v___x_759_, 9, v___x_741_);
v___x_760_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_760_, 0, v___x_759_);
return v___x_760_;
}
v___jp_765_:
{
lean_object* v_ctors_767_; lean_object* v___x_768_; lean_object* v___x_769_; 
v_ctors_767_ = lean_ctor_get(v_val_742_, 4);
v___x_768_ = lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(v___y_766_, v___x_764_);
v___x_769_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg(v_snap_743_, v___x_768_, v_toElabInfo_744_, v_ctors_767_, v___x_763_);
lean_dec_ref(v___x_768_);
if (lean_obj_tag(v___x_769_) == 0)
{
lean_object* v_a_770_; lean_object* v___x_771_; lean_object* v___x_772_; 
v_a_770_ = lean_ctor_get(v___x_769_, 0);
lean_inc(v_a_770_);
lean_dec_ref_known(v___x_769_, 1);
v___x_771_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_745_);
v___x_772_ = l_Lean_Syntax_getTailPos_x3f(v_stx_746_, v___x_747_);
if (lean_obj_tag(v___x_772_) == 0)
{
lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_773_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_774_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_773_);
v___y_750_ = v___x_771_;
v___y_751_ = v_a_770_;
v___y_752_ = v___x_774_;
goto v___jp_749_;
}
else
{
lean_object* v_val_775_; 
v_val_775_ = lean_ctor_get(v___x_772_, 0);
lean_inc(v_val_775_);
lean_dec_ref_known(v___x_772_, 1);
v___y_750_ = v___x_771_;
v___y_751_ = v_a_770_;
v___y_752_ = v_val_775_;
goto v___jp_749_;
}
}
else
{
lean_object* v_a_776_; lean_object* v___x_778_; uint8_t v_isShared_779_; uint8_t v_isSharedCheck_783_; 
lean_dec_ref(v_a_745_);
lean_dec(v___x_741_);
lean_dec(v___x_740_);
lean_dec(v___x_739_);
lean_dec(v___x_738_);
lean_dec(v___x_737_);
lean_dec(v___x_736_);
lean_dec_ref(v___x_735_);
lean_dec(v___x_734_);
lean_dec_ref(v_text_733_);
lean_dec(v___y_732_);
v_a_776_ = lean_ctor_get(v___x_769_, 0);
v_isSharedCheck_783_ = !lean_is_exclusive(v___x_769_);
if (v_isSharedCheck_783_ == 0)
{
v___x_778_ = v___x_769_;
v_isShared_779_ = v_isSharedCheck_783_;
goto v_resetjp_777_;
}
else
{
lean_inc(v_a_776_);
lean_dec(v___x_769_);
v___x_778_ = lean_box(0);
v_isShared_779_ = v_isSharedCheck_783_;
goto v_resetjp_777_;
}
v_resetjp_777_:
{
lean_object* v___x_781_; 
if (v_isShared_779_ == 0)
{
v___x_781_ = v___x_778_;
goto v_reusejp_780_;
}
else
{
lean_object* v_reuseFailAlloc_782_; 
v_reuseFailAlloc_782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_782_, 0, v_a_776_);
v___x_781_ = v_reuseFailAlloc_782_;
goto v_reusejp_780_;
}
v_reusejp_780_:
{
return v___x_781_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___boxed(lean_object** _args){
lean_object* v___x_787_ = _args[0];
lean_object* v___y_788_ = _args[1];
lean_object* v_text_789_ = _args[2];
lean_object* v___x_790_ = _args[3];
lean_object* v___x_791_ = _args[4];
lean_object* v___x_792_ = _args[5];
lean_object* v___x_793_ = _args[6];
lean_object* v___x_794_ = _args[7];
lean_object* v___x_795_ = _args[8];
lean_object* v___x_796_ = _args[9];
lean_object* v___x_797_ = _args[10];
lean_object* v_val_798_ = _args[11];
lean_object* v_snap_799_ = _args[12];
lean_object* v_toElabInfo_800_ = _args[13];
lean_object* v_a_801_ = _args[14];
lean_object* v_stx_802_ = _args[15];
lean_object* v___x_803_ = _args[16];
lean_object* v___y_804_ = _args[17];
_start:
{
uint8_t v___x_7804__boxed_805_; lean_object* v_res_806_; 
v___x_7804__boxed_805_ = lean_unbox(v___x_803_);
v_res_806_ = lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0(v___x_787_, v___y_788_, v_text_789_, v___x_790_, v___x_791_, v___x_792_, v___x_793_, v___x_794_, v___x_795_, v___x_796_, v___x_797_, v_val_798_, v_snap_799_, v_toElabInfo_800_, v_a_801_, v_stx_802_, v___x_7804__boxed_805_);
lean_dec(v_stx_802_);
lean_dec_ref(v_toElabInfo_800_);
lean_dec_ref(v_snap_799_);
lean_dec_ref(v_val_798_);
return v_res_806_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg(lean_object* v_snap_812_, lean_object* v_ctx_813_, lean_object* v_info_814_, lean_object* v_a_815_){
_start:
{
lean_object* v_expectedType_x3f_820_; 
v_expectedType_x3f_820_ = lean_ctor_get(v_info_814_, 2);
if (lean_obj_tag(v_expectedType_x3f_820_) == 1)
{
lean_object* v_toElabInfo_821_; lean_object* v_val_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v_toElabInfo_821_ = lean_ctor_get(v_info_814_, 0);
lean_inc_ref(v_toElabInfo_821_);
v_val_822_ = lean_ctor_get(v_expectedType_x3f_820_, 0);
lean_inc(v_val_822_);
v___x_823_ = lean_alloc_closure((void*)(l_Lean_Meta_whnf___boxed), 6, 1);
lean_closure_set(v___x_823_, 0, v_val_822_);
lean_inc_ref(v_ctx_813_);
lean_inc_ref(v_info_814_);
v___x_824_ = l_Lean_Elab_TermInfo_runMetaM___redArg(v_info_814_, v_ctx_813_, v___x_823_);
if (lean_obj_tag(v___x_824_) == 0)
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_899_; 
v_a_825_ = lean_ctor_get(v___x_824_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_824_);
if (v_isSharedCheck_899_ == 0)
{
v___x_827_ = v___x_824_;
v_isShared_828_ = v_isSharedCheck_899_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_824_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_899_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
if (lean_obj_tag(v_a_825_) == 7)
{
lean_object* v_binderType_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
lean_del_object(v___x_827_);
v_binderType_829_ = lean_ctor_get(v_a_825_, 1);
lean_inc_ref(v_binderType_829_);
lean_dec_ref_known(v_a_825_, 3);
v___x_830_ = lean_alloc_closure((void*)(l_Lean_Meta_whnf___boxed), 6, 1);
lean_closure_set(v___x_830_, 0, v_binderType_829_);
v___x_831_ = l_Lean_Elab_TermInfo_runMetaM___redArg(v_info_814_, v_ctx_813_, v___x_830_);
if (lean_obj_tag(v___x_831_) == 0)
{
lean_object* v_a_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_885_; 
v_a_832_ = lean_ctor_get(v___x_831_, 0);
v_isSharedCheck_885_ = !lean_is_exclusive(v___x_831_);
if (v_isSharedCheck_885_ == 0)
{
v___x_834_ = v___x_831_;
v_isShared_835_ = v_isSharedCheck_885_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_a_832_);
lean_dec(v___x_831_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_885_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
lean_object* v___x_836_; 
v___x_836_ = l_Lean_Expr_getAppFn(v_a_832_);
lean_dec(v_a_832_);
if (lean_obj_tag(v___x_836_) == 4)
{
lean_object* v_declName_837_; lean_object* v___x_838_; uint8_t v___x_839_; lean_object* v___x_840_; 
lean_del_object(v___x_834_);
v_declName_837_ = lean_ctor_get(v___x_836_, 0);
lean_inc(v_declName_837_);
lean_dec_ref_known(v___x_836_, 2);
v___x_838_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_812_);
v___x_839_ = 0;
v___x_840_ = l_Lean_Environment_find_x3f(v___x_838_, v_declName_837_, v___x_839_);
if (lean_obj_tag(v___x_840_) == 1)
{
lean_object* v_val_841_; lean_object* v___x_843_; uint8_t v_isShared_844_; uint8_t v_isSharedCheck_880_; 
v_val_841_ = lean_ctor_get(v___x_840_, 0);
v_isSharedCheck_880_ = !lean_is_exclusive(v___x_840_);
if (v_isSharedCheck_880_ == 0)
{
v___x_843_ = v___x_840_;
v_isShared_844_ = v_isSharedCheck_880_;
goto v_resetjp_842_;
}
else
{
lean_inc(v_val_841_);
lean_dec(v___x_840_);
v___x_843_ = lean_box(0);
v_isShared_844_ = v_isSharedCheck_880_;
goto v_resetjp_842_;
}
v_resetjp_842_:
{
if (lean_obj_tag(v_val_841_) == 5)
{
lean_object* v_val_845_; lean_object* v___x_846_; lean_object* v_a_847_; lean_object* v___x_849_; uint8_t v_isShared_850_; uint8_t v_isSharedCheck_879_; 
v_val_845_ = lean_ctor_get(v_val_841_, 0);
lean_inc_ref(v_val_845_);
lean_dec_ref_known(v_val_841_, 1);
v___x_846_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v_a_815_);
v_a_847_ = lean_ctor_get(v___x_846_, 0);
v_isSharedCheck_879_ = !lean_is_exclusive(v___x_846_);
if (v_isSharedCheck_879_ == 0)
{
v___x_849_ = v___x_846_;
v_isShared_850_ = v_isSharedCheck_879_;
goto v_resetjp_848_;
}
else
{
lean_inc(v_a_847_);
lean_dec(v___x_846_);
v___x_849_ = lean_box(0);
v_isShared_850_ = v_isSharedCheck_879_;
goto v_resetjp_848_;
}
v_resetjp_848_:
{
lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v_stx_855_; lean_object* v___y_857_; lean_object* v___x_875_; 
v___x_851_ = lean_box(0);
v___x_852_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__0));
v___x_853_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3));
v___x_854_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_eqnStub___redArg___closed__1));
v_stx_855_ = lean_ctor_get(v_toElabInfo_821_, 1);
lean_inc(v_stx_855_);
v___x_875_ = l_Lean_Syntax_getPos_x3f(v_stx_855_, v___x_839_);
if (lean_obj_tag(v___x_875_) == 0)
{
lean_object* v___x_876_; lean_object* v___x_877_; 
v___x_876_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_877_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_876_);
v___y_857_ = v___x_877_;
goto v___jp_856_;
}
else
{
lean_object* v_val_878_; 
v_val_878_ = lean_ctor_get(v___x_875_, 0);
lean_inc(v_val_878_);
lean_dec_ref_known(v___x_875_, 1);
v___y_857_ = v_val_878_;
goto v___jp_856_;
}
v___jp_856_:
{
lean_object* v_toEditableDocumentCore_858_; lean_object* v_meta_859_; lean_object* v_text_860_; lean_object* v_source_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___y_864_; lean_object* v___x_866_; 
v_toEditableDocumentCore_858_ = lean_ctor_get(v_a_847_, 0);
v_meta_859_ = lean_ctor_get(v_toEditableDocumentCore_858_, 0);
v_text_860_ = lean_ctor_get(v_meta_859_, 3);
lean_inc_ref(v_text_860_);
v_source_861_ = lean_ctor_get(v_text_860_, 0);
lean_inc_ref(v_source_861_);
v___x_862_ = lp_batteries_Lean_findIndentAndIsStart(v_source_861_, v___y_857_);
v___x_863_ = lean_box(v___x_839_);
v___y_864_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_eqnStub___redArg___lam__0___boxed), 18, 17);
lean_closure_set(v___y_864_, 0, v___x_862_);
lean_closure_set(v___y_864_, 1, v___y_857_);
lean_closure_set(v___y_864_, 2, v_text_860_);
lean_closure_set(v___y_864_, 3, v___x_851_);
lean_closure_set(v___y_864_, 4, v___x_852_);
lean_closure_set(v___y_864_, 5, v___x_853_);
lean_closure_set(v___y_864_, 6, v___x_851_);
lean_closure_set(v___y_864_, 7, v___x_851_);
lean_closure_set(v___y_864_, 8, v___x_851_);
lean_closure_set(v___y_864_, 9, v___x_851_);
lean_closure_set(v___y_864_, 10, v___x_851_);
lean_closure_set(v___y_864_, 11, v_val_845_);
lean_closure_set(v___y_864_, 12, v_snap_812_);
lean_closure_set(v___y_864_, 13, v_toElabInfo_821_);
lean_closure_set(v___y_864_, 14, v_a_847_);
lean_closure_set(v___y_864_, 15, v_stx_855_);
lean_closure_set(v___y_864_, 16, v___x_863_);
if (v_isShared_844_ == 0)
{
lean_ctor_set(v___x_843_, 0, v___y_864_);
v___x_866_ = v___x_843_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v___y_864_);
v___x_866_ = v_reuseFailAlloc_874_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_872_; 
v___x_867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_867_, 0, v___x_854_);
lean_ctor_set(v___x_867_, 1, v___x_866_);
v___x_868_ = lean_unsigned_to_nat(1u);
v___x_869_ = lean_mk_empty_array_with_capacity(v___x_868_);
v___x_870_ = lean_array_push(v___x_869_, v___x_867_);
if (v_isShared_850_ == 0)
{
lean_ctor_set(v___x_849_, 0, v___x_870_);
v___x_872_ = v___x_849_;
goto v_reusejp_871_;
}
else
{
lean_object* v_reuseFailAlloc_873_; 
v_reuseFailAlloc_873_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_873_, 0, v___x_870_);
v___x_872_ = v_reuseFailAlloc_873_;
goto v_reusejp_871_;
}
v_reusejp_871_:
{
return v___x_872_;
}
}
}
}
}
else
{
lean_del_object(v___x_843_);
lean_dec(v_val_841_);
lean_dec_ref(v_toElabInfo_821_);
lean_dec_ref(v_snap_812_);
goto v___jp_817_;
}
}
}
else
{
lean_dec(v___x_840_);
lean_dec_ref(v_toElabInfo_821_);
lean_dec_ref(v_snap_812_);
goto v___jp_817_;
}
}
else
{
lean_object* v___x_881_; lean_object* v___x_883_; 
lean_dec_ref(v___x_836_);
lean_dec_ref(v_toElabInfo_821_);
lean_dec_ref(v_snap_812_);
v___x_881_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_835_ == 0)
{
lean_ctor_set(v___x_834_, 0, v___x_881_);
v___x_883_ = v___x_834_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v___x_881_);
v___x_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
return v___x_883_;
}
}
}
}
else
{
lean_object* v_a_886_; lean_object* v___x_888_; uint8_t v_isShared_889_; uint8_t v_isSharedCheck_894_; 
lean_dec_ref(v_toElabInfo_821_);
lean_dec_ref(v_snap_812_);
v_a_886_ = lean_ctor_get(v___x_831_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_831_);
if (v_isSharedCheck_894_ == 0)
{
v___x_888_ = v___x_831_;
v_isShared_889_ = v_isSharedCheck_894_;
goto v_resetjp_887_;
}
else
{
lean_inc(v_a_886_);
lean_dec(v___x_831_);
v___x_888_ = lean_box(0);
v_isShared_889_ = v_isSharedCheck_894_;
goto v_resetjp_887_;
}
v_resetjp_887_:
{
lean_object* v___x_890_; lean_object* v___x_892_; 
v___x_890_ = l_Lean_Server_RequestError_ofIoError(v_a_886_);
if (v_isShared_889_ == 0)
{
lean_ctor_set(v___x_888_, 0, v___x_890_);
v___x_892_ = v___x_888_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v___x_890_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
else
{
lean_object* v___x_895_; lean_object* v___x_897_; 
lean_dec(v_a_825_);
lean_dec_ref(v_toElabInfo_821_);
lean_dec_ref(v_info_814_);
lean_dec_ref(v_ctx_813_);
lean_dec_ref(v_snap_812_);
v___x_895_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_828_ == 0)
{
lean_ctor_set(v___x_827_, 0, v___x_895_);
v___x_897_ = v___x_827_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v___x_895_);
v___x_897_ = v_reuseFailAlloc_898_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
return v___x_897_;
}
}
}
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_908_; 
lean_dec_ref(v_toElabInfo_821_);
lean_dec_ref(v_info_814_);
lean_dec_ref(v_ctx_813_);
lean_dec_ref(v_snap_812_);
v_a_900_ = lean_ctor_get(v___x_824_, 0);
v_isSharedCheck_908_ = !lean_is_exclusive(v___x_824_);
if (v_isSharedCheck_908_ == 0)
{
v___x_902_ = v___x_824_;
v_isShared_903_ = v_isSharedCheck_908_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_824_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_908_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_904_; lean_object* v___x_906_; 
v___x_904_ = l_Lean_Server_RequestError_ofIoError(v_a_900_);
if (v_isShared_903_ == 0)
{
lean_ctor_set(v___x_902_, 0, v___x_904_);
v___x_906_ = v___x_902_;
goto v_reusejp_905_;
}
else
{
lean_object* v_reuseFailAlloc_907_; 
v_reuseFailAlloc_907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_907_, 0, v___x_904_);
v___x_906_ = v_reuseFailAlloc_907_;
goto v_reusejp_905_;
}
v_reusejp_905_:
{
return v___x_906_;
}
}
}
}
else
{
lean_object* v___x_909_; lean_object* v___x_910_; 
lean_dec_ref(v_info_814_);
lean_dec_ref(v_ctx_813_);
lean_dec_ref(v_snap_812_);
v___x_909_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_910_, 0, v___x_909_);
return v___x_910_;
}
v___jp_817_:
{
lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_818_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
return v___x_819_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___redArg___boxed(lean_object* v_snap_911_, lean_object* v_ctx_912_, lean_object* v_info_913_, lean_object* v_a_914_, lean_object* v_a_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_batteries_Batteries_CodeAction_eqnStub___redArg(v_snap_911_, v_ctx_912_, v_info_913_, v_a_914_);
lean_dec_ref(v_a_914_);
return v_res_916_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub(lean_object* v_x_917_, lean_object* v_snap_918_, lean_object* v_ctx_919_, lean_object* v_info_920_, lean_object* v_a_921_){
_start:
{
lean_object* v___x_923_; 
v___x_923_ = lp_batteries_Batteries_CodeAction_eqnStub___redArg(v_snap_918_, v_ctx_919_, v_info_920_, v_a_921_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_eqnStub___boxed(lean_object* v_x_924_, lean_object* v_snap_925_, lean_object* v_ctx_926_, lean_object* v_info_927_, lean_object* v_a_928_, lean_object* v_a_929_){
_start:
{
lean_object* v_res_930_; 
v_res_930_ = lp_batteries_Batteries_CodeAction_eqnStub(v_x_924_, v_snap_925_, v_ctx_926_, v_info_927_, v_a_928_);
lean_dec_ref(v_a_928_);
lean_dec_ref(v_x_924_);
return v_res_930_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2(lean_object* v_snap_931_, lean_object* v___x_932_, lean_object* v___x_933_, lean_object* v_as_934_, lean_object* v_as_x27_935_, lean_object* v_b_936_, lean_object* v_a_937_){
_start:
{
lean_object* v___x_939_; 
v___x_939_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg(v_snap_931_, v___x_932_, v___x_933_, v_as_x27_935_, v_b_936_);
return v___x_939_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___boxed(lean_object* v_snap_940_, lean_object* v___x_941_, lean_object* v___x_942_, lean_object* v_as_943_, lean_object* v_as_x27_944_, lean_object* v_b_945_, lean_object* v_a_946_, lean_object* v___y_947_){
_start:
{
lean_object* v_res_948_; 
v_res_948_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2(v_snap_940_, v___x_941_, v___x_942_, v_as_943_, v_as_x27_944_, v_b_945_, v_a_946_);
lean_dec(v_as_x27_944_);
lean_dec(v_as_943_);
lean_dec_ref(v___x_942_);
lean_dec_ref(v___x_941_);
lean_dec_ref(v_snap_940_);
return v_res_948_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg(lean_object* v_info_952_, lean_object* v_a_953_){
_start:
{
lean_object* v___y_956_; lean_object* v___y_957_; lean_object* v___y_958_; lean_object* v___y_959_; lean_object* v___y_960_; lean_object* v___y_961_; lean_object* v___y_962_; lean_object* v___y_963_; lean_object* v___y_964_; lean_object* v___y_965_; lean_object* v___y_966_; lean_object* v_toElabInfo_985_; lean_object* v_stx_986_; uint8_t v___x_987_; lean_object* v___y_989_; lean_object* v___x_1006_; 
v_toElabInfo_985_ = lean_ctor_get(v_info_952_, 0);
v_stx_986_ = lean_ctor_get(v_toElabInfo_985_, 1);
v___x_987_ = 0;
v___x_1006_ = l_Lean_Syntax_getPos_x3f(v_stx_986_, v___x_987_);
if (lean_obj_tag(v___x_1006_) == 0)
{
lean_object* v___x_1007_; lean_object* v___x_1008_; 
v___x_1007_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_1008_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_1007_);
v___y_989_ = v___x_1008_;
goto v___jp_988_;
}
else
{
lean_object* v_val_1009_; 
v_val_1009_ = lean_ctor_get(v___x_1006_, 0);
lean_inc(v_val_1009_);
lean_dec_ref_known(v___x_1006_, 1);
v___y_989_ = v_val_1009_;
goto v___jp_988_;
}
v___jp_955_:
{
lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; 
v___x_967_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_967_, 0, v___y_962_);
lean_ctor_set(v___x_967_, 1, v___y_966_);
v___x_968_ = l_Lean_FileMap_utf8RangeToLspRange(v___y_957_, v___x_967_);
v___x_969_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__0));
v___x_970_ = lean_unsigned_to_nat(2u);
v___x_971_ = lean_nat_add(v___y_959_, v___x_970_);
lean_dec(v___y_959_);
v___x_972_ = lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(v___x_971_, v___x_969_);
v___x_973_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__1));
v___x_974_ = lean_string_append(v___x_972_, v___x_973_);
v___x_975_ = lean_box(0);
lean_inc_n(v___y_964_, 3);
v___x_976_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_976_, 0, v___x_968_);
lean_ctor_set(v___x_976_, 1, v___x_974_);
lean_ctor_set(v___x_976_, 2, v___x_975_);
lean_ctor_set(v___x_976_, 3, v___y_964_);
v___x_977_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___y_965_, v___x_976_);
v___x_978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_978_, 0, v___x_977_);
lean_inc(v___y_963_);
lean_inc(v___y_956_);
lean_inc(v___y_961_);
lean_inc(v___y_958_);
lean_inc_ref(v___y_960_);
v___x_979_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_979_, 0, v___y_964_);
lean_ctor_set(v___x_979_, 1, v___y_964_);
lean_ctor_set(v___x_979_, 2, v___y_960_);
lean_ctor_set(v___x_979_, 3, v___y_958_);
lean_ctor_set(v___x_979_, 4, v___y_961_);
lean_ctor_set(v___x_979_, 5, v___y_956_);
lean_ctor_set(v___x_979_, 6, v___y_963_);
lean_ctor_set(v___x_979_, 7, v___x_978_);
lean_ctor_set(v___x_979_, 8, v___x_975_);
lean_ctor_set(v___x_979_, 9, v___x_975_);
v___x_980_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_980_, 0, v___x_979_);
lean_ctor_set(v___x_980_, 1, v___x_975_);
v___x_981_ = lean_unsigned_to_nat(1u);
v___x_982_ = lean_mk_empty_array_with_capacity(v___x_981_);
v___x_983_ = lean_array_push(v___x_982_, v___x_980_);
v___x_984_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_984_, 0, v___x_983_);
return v___x_984_;
}
v___jp_988_:
{
lean_object* v___x_990_; lean_object* v_a_991_; lean_object* v_toEditableDocumentCore_992_; lean_object* v_meta_993_; lean_object* v_text_994_; lean_object* v_source_995_; lean_object* v___x_996_; lean_object* v_fst_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_990_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v_a_953_);
v_a_991_ = lean_ctor_get(v___x_990_, 0);
lean_inc(v_a_991_);
lean_dec_ref(v___x_990_);
v_toEditableDocumentCore_992_ = lean_ctor_get(v_a_991_, 0);
v_meta_993_ = lean_ctor_get(v_toEditableDocumentCore_992_, 0);
v_text_994_ = lean_ctor_get(v_meta_993_, 3);
lean_inc_ref(v_text_994_);
v_source_995_ = lean_ctor_get(v_text_994_, 0);
lean_inc_ref(v_source_995_);
v___x_996_ = lp_batteries_Lean_findIndentAndIsStart(v_source_995_, v___y_989_);
v_fst_997_ = lean_ctor_get(v___x_996_, 0);
lean_inc(v_fst_997_);
lean_dec_ref(v___x_996_);
v___x_998_ = lean_box(0);
v___x_999_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_startTacticStub___redArg___closed__2));
v___x_1000_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3));
v___x_1001_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_991_);
v___x_1002_ = l_Lean_Syntax_getTailPos_x3f(v_stx_986_, v___x_987_);
if (lean_obj_tag(v___x_1002_) == 0)
{
lean_object* v___x_1003_; lean_object* v___x_1004_; 
v___x_1003_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_1004_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_1003_);
v___y_956_ = v___x_998_;
v___y_957_ = v_text_994_;
v___y_958_ = v___x_1000_;
v___y_959_ = v_fst_997_;
v___y_960_ = v___x_999_;
v___y_961_ = v___x_998_;
v___y_962_ = v___y_989_;
v___y_963_ = v___x_998_;
v___y_964_ = v___x_998_;
v___y_965_ = v___x_1001_;
v___y_966_ = v___x_1004_;
goto v___jp_955_;
}
else
{
lean_object* v_val_1005_; 
v_val_1005_ = lean_ctor_get(v___x_1002_, 0);
lean_inc(v_val_1005_);
lean_dec_ref_known(v___x_1002_, 1);
v___y_956_ = v___x_998_;
v___y_957_ = v_text_994_;
v___y_958_ = v___x_1000_;
v___y_959_ = v_fst_997_;
v___y_960_ = v___x_999_;
v___y_961_ = v___x_998_;
v___y_962_ = v___y_989_;
v___y_963_ = v___x_998_;
v___y_964_ = v___x_998_;
v___y_965_ = v___x_1001_;
v___y_966_ = v_val_1005_;
goto v___jp_955_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___redArg___boxed(lean_object* v_info_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_){
_start:
{
lean_object* v_res_1013_; 
v_res_1013_ = lp_batteries_Batteries_CodeAction_startTacticStub___redArg(v_info_1010_, v_a_1011_);
lean_dec_ref(v_a_1011_);
lean_dec_ref(v_info_1010_);
return v_res_1013_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub(lean_object* v_x_1014_, lean_object* v_x_1015_, lean_object* v_x_1016_, lean_object* v_info_1017_, lean_object* v_a_1018_){
_start:
{
lean_object* v___x_1020_; 
v___x_1020_ = lp_batteries_Batteries_CodeAction_startTacticStub___redArg(v_info_1017_, v_a_1018_);
return v___x_1020_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_startTacticStub___boxed(lean_object* v_x_1021_, lean_object* v_x_1022_, lean_object* v_x_1023_, lean_object* v_info_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_){
_start:
{
lean_object* v_res_1027_; 
v_res_1027_ = lp_batteries_Batteries_CodeAction_startTacticStub(v_x_1021_, v_x_1022_, v_x_1023_, v_info_1024_, v_a_1025_);
lean_dec_ref(v_a_1025_);
lean_dec_ref(v_info_1024_);
lean_dec_ref(v_x_1023_);
lean_dec_ref(v_x_1022_);
lean_dec_ref(v_x_1021_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg(lean_object* v_stk_1029_, lean_object* v_node_1030_, lean_object* v_a_1031_){
_start:
{
if (lean_obj_tag(v_node_1030_) == 1)
{
lean_object* v_i_1039_; 
v_i_1039_ = lean_ctor_get(v_node_1030_, 0);
lean_inc_ref(v_i_1039_);
lean_dec_ref_known(v_node_1030_, 2);
if (lean_obj_tag(v_i_1039_) == 0)
{
lean_object* v_i_1040_; lean_object* v___x_1042_; uint8_t v_isShared_1043_; uint8_t v_isSharedCheck_1135_; 
v_i_1040_ = lean_ctor_get(v_i_1039_, 0);
v_isSharedCheck_1135_ = !lean_is_exclusive(v_i_1039_);
if (v_isSharedCheck_1135_ == 0)
{
v___x_1042_ = v_i_1039_;
v_isShared_1043_ = v_isSharedCheck_1135_;
goto v_resetjp_1041_;
}
else
{
lean_inc(v_i_1040_);
lean_dec(v_i_1039_);
v___x_1042_ = lean_box(0);
v_isShared_1043_ = v_isSharedCheck_1135_;
goto v_resetjp_1041_;
}
v_resetjp_1041_:
{
lean_object* v_goalsBefore_1044_; uint8_t v___x_1045_; 
v_goalsBefore_1044_ = lean_ctor_get(v_i_1040_, 2);
lean_inc(v_goalsBefore_1044_);
lean_dec_ref(v_i_1040_);
v___x_1045_ = l_List_isEmpty___redArg(v_goalsBefore_1044_);
lean_dec(v_goalsBefore_1044_);
if (v___x_1045_ == 0)
{
lean_object* v___x_1046_; lean_object* v___x_1048_; 
lean_dec(v_stk_1029_);
v___x_1046_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_1043_ == 0)
{
lean_ctor_set(v___x_1042_, 0, v___x_1046_);
v___x_1048_ = v___x_1042_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v___x_1046_);
v___x_1048_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
return v___x_1048_;
}
}
else
{
if (lean_obj_tag(v_stk_1029_) == 1)
{
lean_object* v_tail_1050_; 
v_tail_1050_ = lean_ctor_get(v_stk_1029_, 1);
lean_inc(v_tail_1050_);
lean_dec_ref_known(v_stk_1029_, 2);
if (lean_obj_tag(v_tail_1050_) == 1)
{
lean_object* v_head_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1133_; 
v_head_1051_ = lean_ctor_get(v_tail_1050_, 0);
v_isSharedCheck_1133_ = !lean_is_exclusive(v_tail_1050_);
if (v_isSharedCheck_1133_ == 0)
{
lean_object* v_unused_1134_; 
v_unused_1134_ = lean_ctor_get(v_tail_1050_, 1);
lean_dec(v_unused_1134_);
v___x_1053_ = v_tail_1050_;
v_isShared_1054_ = v_isSharedCheck_1133_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_head_1051_);
lean_dec(v_tail_1050_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1133_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v_fst_1055_; lean_object* v_snd_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1132_; 
v_fst_1055_ = lean_ctor_get(v_head_1051_, 0);
v_snd_1056_ = lean_ctor_get(v_head_1051_, 1);
v_isSharedCheck_1132_ = !lean_is_exclusive(v_head_1051_);
if (v_isSharedCheck_1132_ == 0)
{
v___x_1058_ = v_head_1051_;
v_isShared_1059_ = v_isSharedCheck_1132_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_snd_1056_);
lean_inc(v_fst_1055_);
lean_dec(v_head_1051_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1132_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
uint8_t v___x_1060_; lean_object* v___x_1061_; 
v___x_1060_ = 0;
v___x_1061_ = l_Lean_Syntax_getTailPos_x3f(v_fst_1055_, v___x_1060_);
if (lean_obj_tag(v___x_1061_) == 1)
{
lean_object* v_val_1062_; lean_object* v___x_1064_; uint8_t v_isShared_1065_; uint8_t v_isSharedCheck_1127_; 
v_val_1062_ = lean_ctor_get(v___x_1061_, 0);
v_isSharedCheck_1127_ = !lean_is_exclusive(v___x_1061_);
if (v_isSharedCheck_1127_ == 0)
{
v___x_1064_ = v___x_1061_;
v_isShared_1065_ = v_isSharedCheck_1127_;
goto v_resetjp_1063_;
}
else
{
lean_inc(v_val_1062_);
lean_dec(v___x_1061_);
v___x_1064_ = lean_box(0);
v_isShared_1065_ = v_isSharedCheck_1127_;
goto v_resetjp_1063_;
}
v_resetjp_1063_:
{
lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; 
v___x_1066_ = l_Lean_Syntax_getArgs(v_fst_1055_);
v___x_1067_ = lean_unsigned_to_nat(0u);
v___x_1068_ = l_Array_toSubarray___redArg(v___x_1066_, v___x_1067_, v_snd_1056_);
v___x_1069_ = l_Subarray_copy___redArg(v___x_1068_);
v___x_1070_ = l_Lean_Syntax_setArgs(v_fst_1055_, v___x_1069_);
v___x_1071_ = l_Lean_Syntax_getTailPos_x3f(v___x_1070_, v___x_1060_);
lean_dec(v___x_1070_);
if (lean_obj_tag(v___x_1071_) == 1)
{
lean_object* v_val_1072_; lean_object* v___x_1074_; uint8_t v_isShared_1075_; uint8_t v_isSharedCheck_1122_; 
lean_del_object(v___x_1042_);
v_val_1072_ = lean_ctor_get(v___x_1071_, 0);
v_isSharedCheck_1122_ = !lean_is_exclusive(v___x_1071_);
if (v_isSharedCheck_1122_ == 0)
{
v___x_1074_ = v___x_1071_;
v_isShared_1075_ = v_isSharedCheck_1122_;
goto v_resetjp_1073_;
}
else
{
lean_inc(v_val_1072_);
lean_dec(v___x_1071_);
v___x_1074_ = lean_box(0);
v_isShared_1075_ = v_isSharedCheck_1122_;
goto v_resetjp_1073_;
}
v_resetjp_1073_:
{
lean_object* v___x_1076_; lean_object* v_a_1077_; lean_object* v___x_1079_; uint8_t v_isShared_1080_; uint8_t v_isSharedCheck_1121_; 
v___x_1076_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v_a_1031_);
v_a_1077_ = lean_ctor_get(v___x_1076_, 0);
v_isSharedCheck_1121_ = !lean_is_exclusive(v___x_1076_);
if (v_isSharedCheck_1121_ == 0)
{
v___x_1079_ = v___x_1076_;
v_isShared_1080_ = v_isSharedCheck_1121_;
goto v_resetjp_1078_;
}
else
{
lean_inc(v_a_1077_);
lean_dec(v___x_1076_);
v___x_1079_ = lean_box(0);
v_isShared_1080_ = v_isSharedCheck_1121_;
goto v_resetjp_1078_;
}
v_resetjp_1078_:
{
lean_object* v_toEditableDocumentCore_1081_; lean_object* v_meta_1082_; lean_object* v___x_1084_; uint8_t v_isShared_1085_; uint8_t v_isSharedCheck_1117_; 
v_toEditableDocumentCore_1081_ = lean_ctor_get(v_a_1077_, 0);
lean_inc_ref(v_toEditableDocumentCore_1081_);
v_meta_1082_ = lean_ctor_get(v_toEditableDocumentCore_1081_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v_toEditableDocumentCore_1081_);
if (v_isSharedCheck_1117_ == 0)
{
lean_object* v_unused_1118_; lean_object* v_unused_1119_; lean_object* v_unused_1120_; 
v_unused_1118_ = lean_ctor_get(v_toEditableDocumentCore_1081_, 3);
lean_dec(v_unused_1118_);
v_unused_1119_ = lean_ctor_get(v_toEditableDocumentCore_1081_, 2);
lean_dec(v_unused_1119_);
v_unused_1120_ = lean_ctor_get(v_toEditableDocumentCore_1081_, 1);
lean_dec(v_unused_1120_);
v___x_1084_ = v_toEditableDocumentCore_1081_;
v_isShared_1085_ = v_isSharedCheck_1117_;
goto v_resetjp_1083_;
}
else
{
lean_inc(v_meta_1082_);
lean_dec(v_toEditableDocumentCore_1081_);
v___x_1084_ = lean_box(0);
v_isShared_1085_ = v_isSharedCheck_1117_;
goto v_resetjp_1083_;
}
v_resetjp_1083_:
{
lean_object* v_text_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1092_; 
v_text_1086_ = lean_ctor_get(v_meta_1082_, 3);
lean_inc_ref(v_text_1086_);
lean_dec_ref(v_meta_1082_);
v___x_1087_ = lean_box(0);
v___x_1088_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg___closed__0));
v___x_1089_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3));
v___x_1090_ = lean_box(v___x_1045_);
if (v_isShared_1075_ == 0)
{
lean_ctor_set(v___x_1074_, 0, v___x_1090_);
v___x_1092_ = v___x_1074_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v___x_1090_);
v___x_1092_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1091_;
}
v_reusejp_1091_:
{
lean_object* v___x_1093_; lean_object* v___x_1095_; 
v___x_1093_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_1077_);
if (v_isShared_1059_ == 0)
{
lean_ctor_set(v___x_1058_, 1, v_val_1062_);
lean_ctor_set(v___x_1058_, 0, v_val_1072_);
v___x_1095_ = v___x_1058_;
goto v_reusejp_1094_;
}
else
{
lean_object* v_reuseFailAlloc_1115_; 
v_reuseFailAlloc_1115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1115_, 0, v_val_1072_);
lean_ctor_set(v_reuseFailAlloc_1115_, 1, v_val_1062_);
v___x_1095_ = v_reuseFailAlloc_1115_;
goto v_reusejp_1094_;
}
v_reusejp_1094_:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1099_; 
v___x_1096_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_1086_, v___x_1095_);
v___x_1097_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10));
if (v_isShared_1085_ == 0)
{
lean_ctor_set(v___x_1084_, 3, v___x_1087_);
lean_ctor_set(v___x_1084_, 2, v___x_1087_);
lean_ctor_set(v___x_1084_, 1, v___x_1097_);
lean_ctor_set(v___x_1084_, 0, v___x_1096_);
v___x_1099_ = v___x_1084_;
goto v_reusejp_1098_;
}
else
{
lean_object* v_reuseFailAlloc_1114_; 
v_reuseFailAlloc_1114_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1114_, 0, v___x_1096_);
lean_ctor_set(v_reuseFailAlloc_1114_, 1, v___x_1097_);
lean_ctor_set(v_reuseFailAlloc_1114_, 2, v___x_1087_);
lean_ctor_set(v_reuseFailAlloc_1114_, 3, v___x_1087_);
v___x_1099_ = v_reuseFailAlloc_1114_;
goto v_reusejp_1098_;
}
v_reusejp_1098_:
{
lean_object* v___x_1100_; lean_object* v___x_1102_; 
v___x_1100_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___x_1093_, v___x_1099_);
if (v_isShared_1065_ == 0)
{
lean_ctor_set(v___x_1064_, 0, v___x_1100_);
v___x_1102_ = v___x_1064_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v___x_1100_);
v___x_1102_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
lean_object* v___x_1103_; lean_object* v___x_1105_; 
v___x_1103_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1103_, 0, v___x_1087_);
lean_ctor_set(v___x_1103_, 1, v___x_1087_);
lean_ctor_set(v___x_1103_, 2, v___x_1088_);
lean_ctor_set(v___x_1103_, 3, v___x_1089_);
lean_ctor_set(v___x_1103_, 4, v___x_1087_);
lean_ctor_set(v___x_1103_, 5, v___x_1092_);
lean_ctor_set(v___x_1103_, 6, v___x_1087_);
lean_ctor_set(v___x_1103_, 7, v___x_1102_);
lean_ctor_set(v___x_1103_, 8, v___x_1087_);
lean_ctor_set(v___x_1103_, 9, v___x_1087_);
if (v_isShared_1054_ == 0)
{
lean_ctor_set_tag(v___x_1053_, 0);
lean_ctor_set(v___x_1053_, 1, v___x_1087_);
lean_ctor_set(v___x_1053_, 0, v___x_1103_);
v___x_1105_ = v___x_1053_;
goto v_reusejp_1104_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v___x_1103_);
lean_ctor_set(v_reuseFailAlloc_1112_, 1, v___x_1087_);
v___x_1105_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1104_;
}
v_reusejp_1104_:
{
lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1110_; 
v___x_1106_ = lean_unsigned_to_nat(1u);
v___x_1107_ = lean_mk_empty_array_with_capacity(v___x_1106_);
v___x_1108_ = lean_array_push(v___x_1107_, v___x_1105_);
if (v_isShared_1080_ == 0)
{
lean_ctor_set(v___x_1079_, 0, v___x_1108_);
v___x_1110_ = v___x_1079_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v___x_1108_);
v___x_1110_ = v_reuseFailAlloc_1111_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
return v___x_1110_;
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
lean_object* v___x_1123_; lean_object* v___x_1125_; 
lean_dec(v___x_1071_);
lean_del_object(v___x_1064_);
lean_dec(v_val_1062_);
lean_del_object(v___x_1058_);
lean_del_object(v___x_1053_);
v___x_1123_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_1043_ == 0)
{
lean_ctor_set(v___x_1042_, 0, v___x_1123_);
v___x_1125_ = v___x_1042_;
goto v_reusejp_1124_;
}
else
{
lean_object* v_reuseFailAlloc_1126_; 
v_reuseFailAlloc_1126_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1126_, 0, v___x_1123_);
v___x_1125_ = v_reuseFailAlloc_1126_;
goto v_reusejp_1124_;
}
v_reusejp_1124_:
{
return v___x_1125_;
}
}
}
}
else
{
lean_object* v___x_1128_; lean_object* v___x_1130_; 
lean_dec(v___x_1061_);
lean_del_object(v___x_1058_);
lean_dec(v_snd_1056_);
lean_dec(v_fst_1055_);
lean_del_object(v___x_1053_);
v___x_1128_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_1043_ == 0)
{
lean_ctor_set(v___x_1042_, 0, v___x_1128_);
v___x_1130_ = v___x_1042_;
goto v_reusejp_1129_;
}
else
{
lean_object* v_reuseFailAlloc_1131_; 
v_reuseFailAlloc_1131_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1131_, 0, v___x_1128_);
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
}
else
{
lean_dec(v_tail_1050_);
lean_del_object(v___x_1042_);
goto v___jp_1036_;
}
}
else
{
lean_del_object(v___x_1042_);
lean_dec(v_stk_1029_);
goto v___jp_1036_;
}
}
}
}
else
{
lean_dec_ref(v_i_1039_);
lean_dec(v_stk_1029_);
goto v___jp_1033_;
}
}
else
{
lean_dec_ref(v_node_1030_);
lean_dec(v_stk_1029_);
goto v___jp_1033_;
}
v___jp_1033_:
{
lean_object* v___x_1034_; lean_object* v___x_1035_; 
v___x_1034_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_1035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1034_);
return v___x_1035_;
}
v___jp_1036_:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1037_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_1038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1037_);
return v___x_1038_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg___boxed(lean_object* v_stk_1136_, lean_object* v_node_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_){
_start:
{
lean_object* v_res_1140_; 
v_res_1140_ = lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg(v_stk_1136_, v_node_1137_, v_a_1138_);
lean_dec_ref(v_a_1138_);
return v_res_1140_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction(lean_object* v_x_1141_, lean_object* v_x_1142_, lean_object* v_x_1143_, lean_object* v_stk_1144_, lean_object* v_node_1145_, lean_object* v_a_1146_){
_start:
{
lean_object* v___x_1148_; 
v___x_1148_ = lp_batteries_Batteries_CodeAction_removeAfterDoneAction___redArg(v_stk_1144_, v_node_1145_, v_a_1146_);
return v___x_1148_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_removeAfterDoneAction___boxed(lean_object* v_x_1149_, lean_object* v_x_1150_, lean_object* v_x_1151_, lean_object* v_stk_1152_, lean_object* v_node_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_){
_start:
{
lean_object* v_res_1156_; 
v_res_1156_ = lp_batteries_Batteries_CodeAction_removeAfterDoneAction(v_x_1149_, v_x_1150_, v_x_1151_, v_stk_1152_, v_node_1153_, v_a_1154_);
lean_dec_ref(v_a_1154_);
lean_dec_ref(v_x_1151_);
lean_dec_ref(v_x_1150_);
lean_dec_ref(v_x_1149_);
return v_res_1156_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_CodeAction_getElimExprNames_spec__0(lean_object* v_msg_1157_){
_start:
{
lean_object* v___x_1158_; lean_object* v___x_1159_; 
v___x_1158_ = l_Lean_instInhabitedLocalDecl_default;
v___x_1159_ = lean_panic_fn_borrowed(v___x_1158_, v_msg_1157_);
return v___x_1159_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___lam__0(lean_object* v_k_1160_, lean_object* v_b_1161_, lean_object* v_c_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v___x_1168_; 
lean_inc(v___y_1166_);
lean_inc_ref(v___y_1165_);
lean_inc(v___y_1164_);
lean_inc_ref(v___y_1163_);
v___x_1168_ = lean_apply_7(v_k_1160_, v_b_1161_, v_c_1162_, v___y_1163_, v___y_1164_, v___y_1165_, v___y_1166_, lean_box(0));
return v___x_1168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___lam__0___boxed(lean_object* v_k_1169_, lean_object* v_b_1170_, lean_object* v_c_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_){
_start:
{
lean_object* v_res_1177_; 
v_res_1177_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___lam__0(v_k_1169_, v_b_1170_, v_c_1171_, v___y_1172_, v___y_1173_, v___y_1174_, v___y_1175_);
lean_dec(v___y_1175_);
lean_dec_ref(v___y_1174_);
lean_dec(v___y_1173_);
lean_dec_ref(v___y_1172_);
return v_res_1177_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg(lean_object* v_type_1178_, lean_object* v_k_1179_, uint8_t v_cleanupAnnotations_1180_, uint8_t v_whnfType_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_){
_start:
{
lean_object* v___f_1187_; lean_object* v___x_1188_; 
v___f_1187_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1187_, 0, v_k_1179_);
v___x_1188_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_1178_, v___f_1187_, v_cleanupAnnotations_1180_, v_whnfType_1181_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_);
if (lean_obj_tag(v___x_1188_) == 0)
{
lean_object* v_a_1189_; lean_object* v___x_1191_; uint8_t v_isShared_1192_; uint8_t v_isSharedCheck_1196_; 
v_a_1189_ = lean_ctor_get(v___x_1188_, 0);
v_isSharedCheck_1196_ = !lean_is_exclusive(v___x_1188_);
if (v_isSharedCheck_1196_ == 0)
{
v___x_1191_ = v___x_1188_;
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
else
{
lean_inc(v_a_1189_);
lean_dec(v___x_1188_);
v___x_1191_ = lean_box(0);
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
v_resetjp_1190_:
{
lean_object* v___x_1194_; 
if (v_isShared_1192_ == 0)
{
v___x_1194_ = v___x_1191_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_a_1189_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
else
{
lean_object* v_a_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1204_; 
v_a_1197_ = lean_ctor_get(v___x_1188_, 0);
v_isSharedCheck_1204_ = !lean_is_exclusive(v___x_1188_);
if (v_isSharedCheck_1204_ == 0)
{
v___x_1199_ = v___x_1188_;
v_isShared_1200_ = v_isSharedCheck_1204_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_a_1197_);
lean_dec(v___x_1188_);
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
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg___boxed(lean_object* v_type_1205_, lean_object* v_k_1206_, lean_object* v_cleanupAnnotations_1207_, lean_object* v_whnfType_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1214_; uint8_t v_whnfType_boxed_1215_; lean_object* v_res_1216_; 
v_cleanupAnnotations_boxed_1214_ = lean_unbox(v_cleanupAnnotations_1207_);
v_whnfType_boxed_1215_ = lean_unbox(v_whnfType_1208_);
v_res_1216_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg(v_type_1205_, v_k_1206_, v_cleanupAnnotations_boxed_1214_, v_whnfType_boxed_1215_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
lean_dec(v___y_1212_);
lean_dec_ref(v___y_1211_);
lean_dec(v___y_1210_);
lean_dec_ref(v___y_1209_);
return v_res_1216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3(lean_object* v_00_u03b1_1217_, lean_object* v_type_1218_, lean_object* v_k_1219_, uint8_t v_cleanupAnnotations_1220_, uint8_t v_whnfType_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_){
_start:
{
lean_object* v___x_1227_; 
v___x_1227_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg(v_type_1218_, v_k_1219_, v_cleanupAnnotations_1220_, v_whnfType_1221_, v___y_1222_, v___y_1223_, v___y_1224_, v___y_1225_);
return v___x_1227_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___boxed(lean_object* v_00_u03b1_1228_, lean_object* v_type_1229_, lean_object* v_k_1230_, lean_object* v_cleanupAnnotations_1231_, lean_object* v_whnfType_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1238_; uint8_t v_whnfType_boxed_1239_; lean_object* v_res_1240_; 
v_cleanupAnnotations_boxed_1238_ = lean_unbox(v_cleanupAnnotations_1231_);
v_whnfType_boxed_1239_ = lean_unbox(v_whnfType_1232_);
v_res_1240_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3(v_00_u03b1_1228_, v_type_1229_, v_k_1230_, v_cleanupAnnotations_boxed_1238_, v_whnfType_boxed_1239_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_);
lean_dec(v___y_1236_);
lean_dec_ref(v___y_1235_);
lean_dec(v___y_1234_);
lean_dec_ref(v___y_1233_);
return v_res_1240_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2_spec__3(lean_object* v_a_1241_, lean_object* v_as_1242_, size_t v_i_1243_, size_t v_stop_1244_){
_start:
{
uint8_t v___x_1245_; 
v___x_1245_ = lean_usize_dec_eq(v_i_1243_, v_stop_1244_);
if (v___x_1245_ == 0)
{
lean_object* v___x_1246_; uint8_t v___x_1247_; 
v___x_1246_ = lean_array_uget_borrowed(v_as_1242_, v_i_1243_);
v___x_1247_ = lean_expr_eqv(v_a_1241_, v___x_1246_);
if (v___x_1247_ == 0)
{
size_t v___x_1248_; size_t v___x_1249_; 
v___x_1248_ = ((size_t)1ULL);
v___x_1249_ = lean_usize_add(v_i_1243_, v___x_1248_);
v_i_1243_ = v___x_1249_;
goto _start;
}
else
{
return v___x_1247_;
}
}
else
{
uint8_t v___x_1251_; 
v___x_1251_ = 0;
return v___x_1251_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2_spec__3___boxed(lean_object* v_a_1252_, lean_object* v_as_1253_, lean_object* v_i_1254_, lean_object* v_stop_1255_){
_start:
{
size_t v_i_boxed_1256_; size_t v_stop_boxed_1257_; uint8_t v_res_1258_; lean_object* v_r_1259_; 
v_i_boxed_1256_ = lean_unbox_usize(v_i_1254_);
lean_dec(v_i_1254_);
v_stop_boxed_1257_ = lean_unbox_usize(v_stop_1255_);
lean_dec(v_stop_1255_);
v_res_1258_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2_spec__3(v_a_1252_, v_as_1253_, v_i_boxed_1256_, v_stop_boxed_1257_);
lean_dec_ref(v_as_1253_);
lean_dec_ref(v_a_1252_);
v_r_1259_ = lean_box(v_res_1258_);
return v_r_1259_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2(lean_object* v_as_1260_, lean_object* v_a_1261_){
_start:
{
lean_object* v___x_1262_; lean_object* v___x_1263_; uint8_t v___x_1264_; 
v___x_1262_ = lean_unsigned_to_nat(0u);
v___x_1263_ = lean_array_get_size(v_as_1260_);
v___x_1264_ = lean_nat_dec_lt(v___x_1262_, v___x_1263_);
if (v___x_1264_ == 0)
{
return v___x_1264_;
}
else
{
if (v___x_1264_ == 0)
{
return v___x_1264_;
}
else
{
size_t v___x_1265_; size_t v___x_1266_; uint8_t v___x_1267_; 
v___x_1265_ = ((size_t)0ULL);
v___x_1266_ = lean_usize_of_nat(v___x_1263_);
v___x_1267_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2_spec__3(v_a_1261_, v_as_1260_, v___x_1265_, v___x_1266_);
return v___x_1267_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2___boxed(lean_object* v_as_1268_, lean_object* v_a_1269_){
_start:
{
uint8_t v_res_1270_; lean_object* v_r_1271_; 
v_res_1270_ = lp_batteries_Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2(v_as_1268_, v_a_1269_);
lean_dec_ref(v_a_1269_);
lean_dec_ref(v_as_1268_);
v_r_1271_ = lean_box(v_res_1270_);
return v_r_1271_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1(lean_object* v___x_1272_, lean_object* v_as_1273_, size_t v_i_1274_, size_t v_stop_1275_, lean_object* v_b_1276_){
_start:
{
lean_object* v___y_1278_; lean_object* v___y_1283_; uint8_t v___x_1288_; 
v___x_1288_ = lean_usize_dec_eq(v_i_1274_, v_stop_1275_);
if (v___x_1288_ == 0)
{
lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; 
v___x_1289_ = lean_array_uget_borrowed(v_as_1273_, v_i_1274_);
v___x_1290_ = l_Lean_Expr_fvarId_x21(v___x_1289_);
lean_inc_ref(v___x_1272_);
v___x_1291_ = lean_local_ctx_find(v___x_1272_, v___x_1290_);
if (lean_obj_tag(v___x_1291_) == 0)
{
lean_object* v___x_1292_; lean_object* v___x_1293_; 
v___x_1292_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_1293_ = lp_batteries_panic___at___00Batteries_CodeAction_getElimExprNames_spec__0(v___x_1292_);
v___y_1283_ = v___x_1293_;
goto v___jp_1282_;
}
else
{
lean_object* v_val_1294_; 
v_val_1294_ = lean_ctor_get(v___x_1291_, 0);
lean_inc(v_val_1294_);
lean_dec_ref_known(v___x_1291_, 1);
v___y_1283_ = v_val_1294_;
goto v___jp_1282_;
}
}
else
{
lean_dec_ref(v___x_1272_);
return v_b_1276_;
}
v___jp_1277_:
{
size_t v___x_1279_; size_t v___x_1280_; 
v___x_1279_ = ((size_t)1ULL);
v___x_1280_ = lean_usize_add(v_i_1274_, v___x_1279_);
v_i_1274_ = v___x_1280_;
v_b_1276_ = v___y_1278_;
goto _start;
}
v___jp_1282_:
{
uint8_t v___x_1284_; uint8_t v___x_1285_; 
v___x_1284_ = l_Lean_LocalDecl_binderInfo(v___y_1283_);
v___x_1285_ = l_Lean_BinderInfo_isExplicit(v___x_1284_);
if (v___x_1285_ == 0)
{
lean_dec_ref(v___y_1283_);
v___y_1278_ = v_b_1276_;
goto v___jp_1277_;
}
else
{
lean_object* v___x_1286_; lean_object* v___x_1287_; 
v___x_1286_ = l_Lean_LocalDecl_userName(v___y_1283_);
lean_dec_ref(v___y_1283_);
v___x_1287_ = lean_array_push(v_b_1276_, v___x_1286_);
v___y_1278_ = v___x_1287_;
goto v___jp_1277_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1___boxed(lean_object* v___x_1295_, lean_object* v_as_1296_, lean_object* v_i_1297_, lean_object* v_stop_1298_, lean_object* v_b_1299_){
_start:
{
size_t v_i_boxed_1300_; size_t v_stop_boxed_1301_; lean_object* v_res_1302_; 
v_i_boxed_1300_ = lean_unbox_usize(v_i_1297_);
lean_dec(v_i_1297_);
v_stop_boxed_1301_ = lean_unbox_usize(v_stop_1298_);
lean_dec(v_stop_1298_);
v_res_1302_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1(v___x_1295_, v_as_1296_, v_i_boxed_1300_, v_stop_boxed_1301_, v_b_1299_);
lean_dec_ref(v_as_1296_);
return v_res_1302_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1(lean_object* v___x_1303_, lean_object* v_as_1304_, lean_object* v_start_1305_, lean_object* v_stop_1306_){
_start:
{
lean_object* v___x_1307_; uint8_t v___x_1308_; 
v___x_1307_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__5));
v___x_1308_ = lean_nat_dec_lt(v_start_1305_, v_stop_1306_);
if (v___x_1308_ == 0)
{
lean_dec_ref(v___x_1303_);
return v___x_1307_;
}
else
{
lean_object* v___x_1309_; uint8_t v___x_1310_; 
v___x_1309_ = lean_array_get_size(v_as_1304_);
v___x_1310_ = lean_nat_dec_le(v_stop_1306_, v___x_1309_);
if (v___x_1310_ == 0)
{
uint8_t v___x_1311_; 
v___x_1311_ = lean_nat_dec_lt(v_start_1305_, v___x_1309_);
if (v___x_1311_ == 0)
{
lean_dec_ref(v___x_1303_);
return v___x_1307_;
}
else
{
size_t v___x_1312_; size_t v___x_1313_; lean_object* v___x_1314_; 
v___x_1312_ = lean_usize_of_nat(v_start_1305_);
v___x_1313_ = lean_usize_of_nat(v___x_1309_);
v___x_1314_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1(v___x_1303_, v_as_1304_, v___x_1312_, v___x_1313_, v___x_1307_);
return v___x_1314_;
}
}
else
{
size_t v___x_1315_; size_t v___x_1316_; lean_object* v___x_1317_; 
v___x_1315_ = lean_usize_of_nat(v_start_1305_);
v___x_1316_ = lean_usize_of_nat(v_stop_1306_);
v___x_1317_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1_spec__1(v___x_1303_, v_as_1304_, v___x_1315_, v___x_1316_, v___x_1307_);
return v___x_1317_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1___boxed(lean_object* v___x_1318_, lean_object* v_as_1319_, lean_object* v_start_1320_, lean_object* v_stop_1321_){
_start:
{
lean_object* v_res_1322_; 
v_res_1322_ = lp_batteries_Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1(v___x_1318_, v_as_1319_, v_start_1320_, v_stop_1321_);
lean_dec(v_stop_1321_);
lean_dec(v_start_1320_);
lean_dec_ref(v_as_1319_);
return v_res_1322_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___lam__0(lean_object* v___x_1323_, lean_object* v_args_1324_, lean_object* v_x_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_){
_start:
{
lean_object* v_lctx_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; 
v_lctx_1331_ = lean_ctor_get(v___y_1326_, 2);
v___x_1332_ = lean_array_get_size(v_args_1324_);
lean_inc_ref(v_lctx_1331_);
v___x_1333_ = lp_batteries_Array_filterMapM___at___00Batteries_CodeAction_getElimExprNames_spec__1(v_lctx_1331_, v_args_1324_, v___x_1323_, v___x_1332_);
v___x_1334_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1334_, 0, v___x_1333_);
return v___x_1334_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___lam__0___boxed(lean_object* v___x_1335_, lean_object* v_args_1336_, lean_object* v_x_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_){
_start:
{
lean_object* v_res_1343_; 
v_res_1343_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___lam__0(v___x_1335_, v_args_1336_, v_x_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec(v___y_1339_);
lean_dec_ref(v___y_1338_);
lean_dec_ref(v_x_1337_);
lean_dec_ref(v_args_1336_);
lean_dec(v___x_1335_);
return v_res_1343_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg(lean_object* v_xs_1346_, lean_object* v_motive_1347_, lean_object* v_targets_1348_, lean_object* v_range_1349_, lean_object* v_b_1350_, lean_object* v_i_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_){
_start:
{
lean_object* v_stop_1357_; lean_object* v_step_1358_; lean_object* v_a_1360_; uint8_t v___x_1363_; 
v_stop_1357_ = lean_ctor_get(v_range_1349_, 1);
v_step_1358_ = lean_ctor_get(v_range_1349_, 2);
v___x_1363_ = lean_nat_dec_lt(v_i_1351_, v_stop_1357_);
if (v___x_1363_ == 0)
{
lean_object* v___x_1364_; 
lean_dec(v_i_1351_);
v___x_1364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1364_, 0, v_b_1350_);
return v___x_1364_;
}
else
{
lean_object* v___x_1365_; uint8_t v___x_1366_; 
v___x_1365_ = lean_array_fget_borrowed(v_xs_1346_, v_i_1351_);
v___x_1366_ = lean_expr_eqv(v___x_1365_, v_motive_1347_);
if (v___x_1366_ == 0)
{
uint8_t v___x_1367_; 
v___x_1367_ = lp_batteries_Array_contains___at___00Batteries_CodeAction_getElimExprNames_spec__2(v_targets_1348_, v___x_1365_);
if (v___x_1367_ == 0)
{
lean_object* v___x_1368_; lean_object* v___x_1369_; 
v___x_1368_ = l_Lean_Expr_fvarId_x21(v___x_1365_);
v___x_1369_ = l_Lean_FVarId_getDecl___redArg(v___x_1368_, v___y_1352_, v___y_1354_, v___y_1355_);
if (lean_obj_tag(v___x_1369_) == 0)
{
lean_object* v_a_1370_; uint8_t v___x_1371_; uint8_t v___x_1372_; 
v_a_1370_ = lean_ctor_get(v___x_1369_, 0);
lean_inc(v_a_1370_);
lean_dec_ref_known(v___x_1369_, 1);
v___x_1371_ = l_Lean_LocalDecl_binderInfo(v_a_1370_);
v___x_1372_ = l_Lean_BinderInfo_isExplicit(v___x_1371_);
if (v___x_1372_ == 0)
{
lean_dec(v_a_1370_);
v_a_1360_ = v_b_1350_;
goto v___jp_1359_;
}
else
{
lean_object* v___f_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; 
v___f_1373_ = ((lean_object*)(lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___closed__0));
v___x_1374_ = l_Lean_LocalDecl_type(v_a_1370_);
v___x_1375_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg(v___x_1374_, v___f_1373_, v___x_1367_, v___x_1367_, v___y_1352_, v___y_1353_, v___y_1354_, v___y_1355_);
if (lean_obj_tag(v___x_1375_) == 0)
{
lean_object* v_a_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; 
v_a_1376_ = lean_ctor_get(v___x_1375_, 0);
lean_inc(v_a_1376_);
lean_dec_ref_known(v___x_1375_, 1);
v___x_1377_ = l_Lean_LocalDecl_userName(v_a_1370_);
lean_dec(v_a_1370_);
v___x_1378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1377_);
lean_ctor_set(v___x_1378_, 1, v_a_1376_);
v___x_1379_ = lean_array_push(v_b_1350_, v___x_1378_);
v_a_1360_ = v___x_1379_;
goto v___jp_1359_;
}
else
{
lean_object* v_a_1380_; lean_object* v___x_1382_; uint8_t v_isShared_1383_; uint8_t v_isSharedCheck_1387_; 
lean_dec(v_a_1370_);
lean_dec(v_i_1351_);
lean_dec_ref(v_b_1350_);
v_a_1380_ = lean_ctor_get(v___x_1375_, 0);
v_isSharedCheck_1387_ = !lean_is_exclusive(v___x_1375_);
if (v_isSharedCheck_1387_ == 0)
{
v___x_1382_ = v___x_1375_;
v_isShared_1383_ = v_isSharedCheck_1387_;
goto v_resetjp_1381_;
}
else
{
lean_inc(v_a_1380_);
lean_dec(v___x_1375_);
v___x_1382_ = lean_box(0);
v_isShared_1383_ = v_isSharedCheck_1387_;
goto v_resetjp_1381_;
}
v_resetjp_1381_:
{
lean_object* v___x_1385_; 
if (v_isShared_1383_ == 0)
{
v___x_1385_ = v___x_1382_;
goto v_reusejp_1384_;
}
else
{
lean_object* v_reuseFailAlloc_1386_; 
v_reuseFailAlloc_1386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1386_, 0, v_a_1380_);
v___x_1385_ = v_reuseFailAlloc_1386_;
goto v_reusejp_1384_;
}
v_reusejp_1384_:
{
return v___x_1385_;
}
}
}
}
}
else
{
lean_object* v_a_1388_; lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1395_; 
lean_dec(v_i_1351_);
lean_dec_ref(v_b_1350_);
v_a_1388_ = lean_ctor_get(v___x_1369_, 0);
v_isSharedCheck_1395_ = !lean_is_exclusive(v___x_1369_);
if (v_isSharedCheck_1395_ == 0)
{
v___x_1390_ = v___x_1369_;
v_isShared_1391_ = v_isSharedCheck_1395_;
goto v_resetjp_1389_;
}
else
{
lean_inc(v_a_1388_);
lean_dec(v___x_1369_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1395_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
lean_object* v___x_1393_; 
if (v_isShared_1391_ == 0)
{
v___x_1393_ = v___x_1390_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v_a_1388_);
v___x_1393_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
return v___x_1393_;
}
}
}
}
else
{
v_a_1360_ = v_b_1350_;
goto v___jp_1359_;
}
}
else
{
v_a_1360_ = v_b_1350_;
goto v___jp_1359_;
}
}
v___jp_1359_:
{
lean_object* v___x_1361_; 
v___x_1361_ = lean_nat_add(v_i_1351_, v_step_1358_);
lean_dec(v_i_1351_);
v_b_1350_ = v_a_1360_;
v_i_1351_ = v___x_1361_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg___boxed(lean_object* v_xs_1396_, lean_object* v_motive_1397_, lean_object* v_targets_1398_, lean_object* v_range_1399_, lean_object* v_b_1400_, lean_object* v_i_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_){
_start:
{
lean_object* v_res_1407_; 
v_res_1407_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg(v_xs_1396_, v_motive_1397_, v_targets_1398_, v_range_1399_, v_b_1400_, v_i_1401_, v___y_1402_, v___y_1403_, v___y_1404_, v___y_1405_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec(v___y_1403_);
lean_dec_ref(v___y_1402_);
lean_dec_ref(v_range_1399_);
lean_dec_ref(v_targets_1398_);
lean_dec_ref(v_motive_1397_);
lean_dec_ref(v_xs_1396_);
return v_res_1407_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1408_; lean_object* v_dummy_1409_; 
v___x_1408_ = lean_box(0);
v_dummy_1409_ = l_Lean_Expr_sort___override(v___x_1408_);
return v_dummy_1409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0(lean_object* v_xs_1412_, lean_object* v_type_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_){
_start:
{
lean_object* v_motive_1419_; lean_object* v___x_1420_; 
v_motive_1419_ = l_Lean_Expr_getAppFn(v_type_1413_);
lean_inc(v___y_1417_);
lean_inc_ref(v___y_1416_);
lean_inc(v___y_1415_);
lean_inc_ref(v___y_1414_);
lean_inc_ref(v_motive_1419_);
v___x_1420_ = lean_infer_type(v_motive_1419_, v___y_1414_, v___y_1415_, v___y_1416_, v___y_1417_);
if (lean_obj_tag(v___x_1420_) == 0)
{
lean_object* v_nargs_1421_; lean_object* v_dummy_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v_targets_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; 
lean_dec_ref_known(v___x_1420_, 1);
v_nargs_1421_ = l_Lean_Expr_getAppNumArgs(v_type_1413_);
v_dummy_1422_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__0, &lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__0_once, _init_lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__0);
lean_inc(v_nargs_1421_);
v___x_1423_ = lean_mk_array(v_nargs_1421_, v_dummy_1422_);
v___x_1424_ = lean_unsigned_to_nat(1u);
v___x_1425_ = lean_nat_sub(v_nargs_1421_, v___x_1424_);
lean_dec(v_nargs_1421_);
v_targets_1426_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_type_1413_, v___x_1423_, v___x_1425_);
v___x_1427_ = lean_unsigned_to_nat(0u);
v___x_1428_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__1));
v___x_1429_ = lean_array_get_size(v_xs_1412_);
v___x_1430_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1427_);
lean_ctor_set(v___x_1430_, 1, v___x_1429_);
lean_ctor_set(v___x_1430_, 2, v___x_1424_);
v___x_1431_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg(v_xs_1412_, v_motive_1419_, v_targets_1426_, v___x_1430_, v___x_1428_, v___x_1427_, v___y_1414_, v___y_1415_, v___y_1416_, v___y_1417_);
lean_dec_ref_known(v___x_1430_, 3);
lean_dec_ref(v_targets_1426_);
lean_dec_ref(v_motive_1419_);
return v___x_1431_;
}
else
{
lean_object* v_a_1432_; lean_object* v___x_1434_; uint8_t v_isShared_1435_; uint8_t v_isSharedCheck_1439_; 
lean_dec_ref(v_motive_1419_);
lean_dec_ref(v_type_1413_);
v_a_1432_ = lean_ctor_get(v___x_1420_, 0);
v_isSharedCheck_1439_ = !lean_is_exclusive(v___x_1420_);
if (v_isSharedCheck_1439_ == 0)
{
v___x_1434_ = v___x_1420_;
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
else
{
lean_inc(v_a_1432_);
lean_dec(v___x_1420_);
v___x_1434_ = lean_box(0);
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
v_resetjp_1433_:
{
lean_object* v___x_1437_; 
if (v_isShared_1435_ == 0)
{
v___x_1437_ = v___x_1434_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v_a_1432_);
v___x_1437_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
return v___x_1437_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___boxed(lean_object* v_xs_1440_, lean_object* v_type_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_){
_start:
{
lean_object* v_res_1447_; 
v_res_1447_ = lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0(v_xs_1440_, v_type_1441_, v___y_1442_, v___y_1443_, v___y_1444_, v___y_1445_);
lean_dec(v___y_1445_);
lean_dec_ref(v___y_1444_);
lean_dec(v___y_1443_);
lean_dec_ref(v___y_1442_);
lean_dec_ref(v_xs_1440_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames(lean_object* v_elimType_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_){
_start:
{
lean_object* v___f_1455_; uint8_t v___x_1456_; lean_object* v___x_1457_; 
v___f_1455_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getElimExprNames___closed__0));
v___x_1456_ = 0;
v___x_1457_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_CodeAction_getElimExprNames_spec__3___redArg(v_elimType_1449_, v___f_1455_, v___x_1456_, v___x_1456_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_);
return v___x_1457_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_getElimExprNames___boxed(lean_object* v_elimType_1458_, lean_object* v_a_1459_, lean_object* v_a_1460_, lean_object* v_a_1461_, lean_object* v_a_1462_, lean_object* v_a_1463_){
_start:
{
lean_object* v_res_1464_; 
v_res_1464_ = lp_batteries_Batteries_CodeAction_getElimExprNames(v_elimType_1458_, v_a_1459_, v_a_1460_, v_a_1461_, v_a_1462_);
lean_dec(v_a_1462_);
lean_dec_ref(v_a_1461_);
lean_dec(v_a_1460_);
lean_dec_ref(v_a_1459_);
return v_res_1464_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4(lean_object* v_xs_1465_, lean_object* v_motive_1466_, lean_object* v_targets_1467_, lean_object* v_range_1468_, lean_object* v_b_1469_, lean_object* v_i_1470_, lean_object* v_hs_1471_, lean_object* v_hl_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_){
_start:
{
lean_object* v___x_1478_; 
v___x_1478_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___redArg(v_xs_1465_, v_motive_1466_, v_targets_1467_, v_range_1468_, v_b_1469_, v_i_1470_, v___y_1473_, v___y_1474_, v___y_1475_, v___y_1476_);
return v___x_1478_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4___boxed(lean_object* v_xs_1479_, lean_object* v_motive_1480_, lean_object* v_targets_1481_, lean_object* v_range_1482_, lean_object* v_b_1483_, lean_object* v_i_1484_, lean_object* v_hs_1485_, lean_object* v_hl_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_){
_start:
{
lean_object* v_res_1492_; 
v_res_1492_ = lp_batteries___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Batteries_CodeAction_getElimExprNames_spec__4(v_xs_1479_, v_motive_1480_, v_targets_1481_, v_range_1482_, v_b_1483_, v_i_1484_, v_hs_1485_, v_hl_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
lean_dec(v___y_1490_);
lean_dec_ref(v___y_1489_);
lean_dec(v___y_1488_);
lean_dec_ref(v___y_1487_);
lean_dec_ref(v_range_1482_);
lean_dec_ref(v_targets_1481_);
lean_dec_ref(v_motive_1480_);
lean_dec_ref(v_xs_1479_);
return v_res_1492_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_CodeAction_findTermInfo_x3f___lam__0(lean_object* v_stx_1493_, lean_object* v_x_1494_){
_start:
{
if (lean_obj_tag(v_x_1494_) == 1)
{
lean_object* v_i_1495_; lean_object* v_toElabInfo_1496_; lean_object* v_stx_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; uint8_t v___x_1500_; 
v_i_1495_ = lean_ctor_get(v_x_1494_, 0);
lean_inc_ref(v_i_1495_);
lean_dec_ref_known(v_x_1494_, 1);
v_toElabInfo_1496_ = lean_ctor_get(v_i_1495_, 0);
lean_inc_ref(v_toElabInfo_1496_);
lean_dec_ref(v_i_1495_);
v_stx_1497_ = lean_ctor_get(v_toElabInfo_1496_, 1);
lean_inc_n(v_stx_1497_, 2);
lean_dec_ref(v_toElabInfo_1496_);
v___x_1498_ = l_Lean_Syntax_getKind(v_stx_1497_);
lean_inc(v_stx_1493_);
v___x_1499_ = l_Lean_Syntax_getKind(v_stx_1493_);
v___x_1500_ = lean_name_eq(v___x_1498_, v___x_1499_);
lean_dec(v___x_1499_);
lean_dec(v___x_1498_);
if (v___x_1500_ == 0)
{
lean_dec(v_stx_1497_);
lean_dec(v_stx_1493_);
return v___x_1500_;
}
else
{
uint8_t v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; uint8_t v___x_1504_; 
v___x_1501_ = 0;
v___x_1502_ = l_Lean_Syntax_getRange_x3f(v_stx_1497_, v___x_1501_);
lean_dec(v_stx_1497_);
v___x_1503_ = l_Lean_Syntax_getRange_x3f(v_stx_1493_, v___x_1501_);
lean_dec(v_stx_1493_);
v___x_1504_ = lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0(v___x_1502_, v___x_1503_);
lean_dec(v___x_1503_);
lean_dec(v___x_1502_);
return v___x_1504_;
}
}
else
{
uint8_t v___x_1505_; 
lean_dec_ref(v_x_1494_);
lean_dec(v_stx_1493_);
v___x_1505_ = 0;
return v___x_1505_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findTermInfo_x3f___lam__0___boxed(lean_object* v_stx_1506_, lean_object* v_x_1507_){
_start:
{
uint8_t v_res_1508_; lean_object* v_r_1509_; 
v_res_1508_ = lp_batteries_Batteries_CodeAction_findTermInfo_x3f___lam__0(v_stx_1506_, v_x_1507_);
v_r_1509_ = lean_box(v_res_1508_);
return v_r_1509_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findTermInfo_x3f(lean_object* v_node_1510_, lean_object* v_stx_1511_){
_start:
{
lean_object* v___f_1512_; lean_object* v___x_1513_; 
v___f_1512_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_findTermInfo_x3f___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1512_, 0, v_stx_1511_);
v___x_1513_ = l_Lean_Elab_InfoTree_findInfo_x3f(v___f_1512_, v_node_1510_);
if (lean_obj_tag(v___x_1513_) == 1)
{
lean_object* v_val_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1523_; 
v_val_1514_ = lean_ctor_get(v___x_1513_, 0);
v_isSharedCheck_1523_ = !lean_is_exclusive(v___x_1513_);
if (v_isSharedCheck_1523_ == 0)
{
v___x_1516_ = v___x_1513_;
v_isShared_1517_ = v_isSharedCheck_1523_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_val_1514_);
lean_dec(v___x_1513_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1523_;
goto v_resetjp_1515_;
}
v_resetjp_1515_:
{
if (lean_obj_tag(v_val_1514_) == 1)
{
lean_object* v_i_1518_; lean_object* v___x_1520_; 
v_i_1518_ = lean_ctor_get(v_val_1514_, 0);
lean_inc_ref(v_i_1518_);
lean_dec_ref_known(v_val_1514_, 1);
if (v_isShared_1517_ == 0)
{
lean_ctor_set(v___x_1516_, 0, v_i_1518_);
v___x_1520_ = v___x_1516_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v_i_1518_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
else
{
lean_object* v___x_1522_; 
lean_del_object(v___x_1516_);
lean_dec(v_val_1514_);
v___x_1522_ = lean_box(0);
return v___x_1522_;
}
}
}
else
{
lean_object* v___x_1524_; 
lean_dec(v___x_1513_);
v___x_1524_ = lean_box(0);
return v___x_1524_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0_spec__1(lean_object* v_stx_1528_, lean_object* v_ctx_1529_, lean_object* v_as_1530_, size_t v_sz_1531_, size_t v_i_1532_, lean_object* v_b_1533_){
_start:
{
uint8_t v___x_1534_; 
v___x_1534_ = lean_usize_dec_lt(v_i_1532_, v_sz_1531_);
if (v___x_1534_ == 0)
{
lean_dec_ref(v_ctx_1529_);
lean_dec(v_stx_1528_);
lean_inc_ref(v_b_1533_);
return v_b_1533_;
}
else
{
lean_object* v___x_1535_; lean_object* v_a_1536_; lean_object* v___x_1537_; 
v___x_1535_ = lean_box(0);
v_a_1536_ = lean_array_uget_borrowed(v_as_1530_, v_i_1532_);
lean_inc_ref(v_ctx_1529_);
lean_inc(v_stx_1528_);
v___x_1537_ = lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0(v_stx_1528_, v_ctx_1529_, v_a_1536_);
if (lean_obj_tag(v___x_1537_) == 1)
{
lean_object* v___x_1538_; lean_object* v___x_1539_; 
lean_dec_ref(v_ctx_1529_);
lean_dec(v_stx_1528_);
v___x_1538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1538_, 0, v___x_1537_);
v___x_1539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1538_);
lean_ctor_set(v___x_1539_, 1, v___x_1535_);
return v___x_1539_;
}
else
{
lean_object* v___x_1540_; size_t v___x_1541_; size_t v___x_1542_; 
lean_dec(v___x_1537_);
v___x_1540_ = ((lean_object*)(lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0));
v___x_1541_ = ((size_t)1ULL);
v___x_1542_ = lean_usize_add(v_i_1532_, v___x_1541_);
v_i_1532_ = v___x_1542_;
v_b_1533_ = v___x_1540_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_findTermInfoWithCtx_x3f(lean_object* v_t_1544_, lean_object* v_stx_1545_, lean_object* v_ctx_1546_){
_start:
{
switch(lean_obj_tag(v_t_1544_))
{
case 0:
{
lean_object* v_i_1547_; lean_object* v_t_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; 
v_i_1547_ = lean_ctor_get(v_t_1544_, 0);
lean_inc_ref(v_i_1547_);
v_t_1548_ = lean_ctor_get(v_t_1544_, 1);
lean_inc_ref(v_t_1548_);
lean_dec_ref_known(v_t_1544_, 2);
lean_inc_ref(v_ctx_1546_);
v___x_1549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1549_, 0, v_ctx_1546_);
v___x_1550_ = l_Lean_Elab_PartialContextInfo_mergeIntoOuter_x3f(v_i_1547_, v___x_1549_);
if (lean_obj_tag(v___x_1550_) == 0)
{
v_t_1544_ = v_t_1548_;
goto _start;
}
else
{
lean_object* v_val_1552_; 
lean_dec_ref(v_ctx_1546_);
v_val_1552_ = lean_ctor_get(v___x_1550_, 0);
lean_inc(v_val_1552_);
lean_dec_ref_known(v___x_1550_, 1);
v_t_1544_ = v_t_1548_;
v_ctx_1546_ = v_val_1552_;
goto _start;
}
}
case 1:
{
lean_object* v_i_1554_; 
v_i_1554_ = lean_ctor_get(v_t_1544_, 0);
lean_inc_ref(v_i_1554_);
if (lean_obj_tag(v_i_1554_) == 1)
{
lean_object* v_children_1555_; lean_object* v___x_1557_; uint8_t v_isShared_1558_; uint8_t v_isSharedCheck_1582_; 
v_children_1555_ = lean_ctor_get(v_t_1544_, 1);
v_isSharedCheck_1582_ = !lean_is_exclusive(v_t_1544_);
if (v_isSharedCheck_1582_ == 0)
{
lean_object* v_unused_1583_; 
v_unused_1583_ = lean_ctor_get(v_t_1544_, 0);
lean_dec(v_unused_1583_);
v___x_1557_ = v_t_1544_;
v_isShared_1558_ = v_isSharedCheck_1582_;
goto v_resetjp_1556_;
}
else
{
lean_inc(v_children_1555_);
lean_dec(v_t_1544_);
v___x_1557_ = lean_box(0);
v_isShared_1558_ = v_isSharedCheck_1582_;
goto v_resetjp_1556_;
}
v_resetjp_1556_:
{
lean_object* v_i_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1581_; 
v_i_1559_ = lean_ctor_get(v_i_1554_, 0);
v_isSharedCheck_1581_ = !lean_is_exclusive(v_i_1554_);
if (v_isSharedCheck_1581_ == 0)
{
v___x_1561_ = v_i_1554_;
v_isShared_1562_ = v_isSharedCheck_1581_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_i_1559_);
lean_dec(v_i_1554_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1581_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
uint8_t v___y_1564_; lean_object* v_toElabInfo_1572_; lean_object* v_stx_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; uint8_t v___x_1576_; 
v_toElabInfo_1572_ = lean_ctor_get(v_i_1559_, 0);
v_stx_1573_ = lean_ctor_get(v_toElabInfo_1572_, 1);
lean_inc(v_stx_1573_);
v___x_1574_ = l_Lean_Syntax_getKind(v_stx_1573_);
lean_inc(v_stx_1545_);
v___x_1575_ = l_Lean_Syntax_getKind(v_stx_1545_);
v___x_1576_ = lean_name_eq(v___x_1574_, v___x_1575_);
lean_dec(v___x_1575_);
lean_dec(v___x_1574_);
if (v___x_1576_ == 0)
{
v___y_1564_ = v___x_1576_;
goto v___jp_1563_;
}
else
{
uint8_t v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; uint8_t v___x_1580_; 
v___x_1577_ = 0;
v___x_1578_ = l_Lean_Syntax_getRange_x3f(v_stx_1573_, v___x_1577_);
v___x_1579_ = l_Lean_Syntax_getRange_x3f(v_stx_1545_, v___x_1577_);
v___x_1580_ = lp_batteries_Option_instBEq_beq___at___00Batteries_CodeAction_findStack_x3f_spec__0(v___x_1578_, v___x_1579_);
lean_dec(v___x_1579_);
lean_dec(v___x_1578_);
v___y_1564_ = v___x_1580_;
goto v___jp_1563_;
}
v___jp_1563_:
{
if (v___y_1564_ == 0)
{
lean_object* v___x_1565_; 
lean_del_object(v___x_1561_);
lean_dec_ref(v_i_1559_);
lean_del_object(v___x_1557_);
v___x_1565_ = lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0(v_stx_1545_, v_ctx_1546_, v_children_1555_);
lean_dec_ref(v_children_1555_);
return v___x_1565_;
}
else
{
lean_object* v___x_1567_; 
lean_dec_ref(v_children_1555_);
lean_dec(v_stx_1545_);
if (v_isShared_1558_ == 0)
{
lean_ctor_set_tag(v___x_1557_, 0);
lean_ctor_set(v___x_1557_, 1, v_ctx_1546_);
lean_ctor_set(v___x_1557_, 0, v_i_1559_);
v___x_1567_ = v___x_1557_;
goto v_reusejp_1566_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v_i_1559_);
lean_ctor_set(v_reuseFailAlloc_1571_, 1, v_ctx_1546_);
v___x_1567_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1566_;
}
v_reusejp_1566_:
{
lean_object* v___x_1569_; 
if (v_isShared_1562_ == 0)
{
lean_ctor_set(v___x_1561_, 0, v___x_1567_);
v___x_1569_ = v___x_1561_;
goto v_reusejp_1568_;
}
else
{
lean_object* v_reuseFailAlloc_1570_; 
v_reuseFailAlloc_1570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1570_, 0, v___x_1567_);
v___x_1569_ = v_reuseFailAlloc_1570_;
goto v_reusejp_1568_;
}
v_reusejp_1568_:
{
return v___x_1569_;
}
}
}
}
}
}
}
else
{
lean_object* v_children_1584_; lean_object* v___x_1585_; 
lean_dec_ref(v_i_1554_);
v_children_1584_ = lean_ctor_get(v_t_1544_, 1);
lean_inc_ref(v_children_1584_);
lean_dec_ref_known(v_t_1544_, 2);
v___x_1585_ = lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0(v_stx_1545_, v_ctx_1546_, v_children_1584_);
lean_dec_ref(v_children_1584_);
return v___x_1585_;
}
}
default: 
{
lean_object* v___x_1586_; 
lean_dec_ref_known(v_t_1544_, 1);
lean_dec_ref(v_ctx_1546_);
lean_dec(v_stx_1545_);
v___x_1586_ = lean_box(0);
return v___x_1586_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1(lean_object* v_stx_1587_, lean_object* v_ctx_1588_, lean_object* v_as_1589_, size_t v_sz_1590_, size_t v_i_1591_, lean_object* v_b_1592_){
_start:
{
uint8_t v___x_1593_; 
v___x_1593_ = lean_usize_dec_lt(v_i_1591_, v_sz_1590_);
if (v___x_1593_ == 0)
{
lean_dec_ref(v_ctx_1588_);
lean_dec(v_stx_1587_);
lean_inc_ref(v_b_1592_);
return v_b_1592_;
}
else
{
lean_object* v___x_1594_; lean_object* v_a_1595_; lean_object* v___x_1596_; 
v___x_1594_ = lean_box(0);
v_a_1595_ = lean_array_uget_borrowed(v_as_1589_, v_i_1591_);
lean_inc_ref(v_ctx_1588_);
lean_inc(v_stx_1587_);
lean_inc(v_a_1595_);
v___x_1596_ = lp_batteries_Batteries_CodeAction_findTermInfoWithCtx_x3f(v_a_1595_, v_stx_1587_, v_ctx_1588_);
if (lean_obj_tag(v___x_1596_) == 1)
{
lean_object* v___x_1597_; lean_object* v___x_1598_; 
lean_dec_ref(v_ctx_1588_);
lean_dec(v_stx_1587_);
v___x_1597_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1597_, 0, v___x_1596_);
v___x_1598_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1598_, 0, v___x_1597_);
lean_ctor_set(v___x_1598_, 1, v___x_1594_);
return v___x_1598_;
}
else
{
lean_object* v___x_1599_; size_t v___x_1600_; size_t v___x_1601_; 
lean_dec(v___x_1596_);
v___x_1599_ = ((lean_object*)(lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0));
v___x_1600_ = ((size_t)1ULL);
v___x_1601_ = lean_usize_add(v_i_1591_, v___x_1600_);
v_i_1591_ = v___x_1601_;
v_b_1592_ = v___x_1599_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0(lean_object* v_stx_1603_, lean_object* v_ctx_1604_, lean_object* v_x_1605_){
_start:
{
if (lean_obj_tag(v_x_1605_) == 0)
{
lean_object* v_cs_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; size_t v_sz_1609_; size_t v___x_1610_; lean_object* v___x_1611_; lean_object* v_fst_1612_; 
v_cs_1606_ = lean_ctor_get(v_x_1605_, 0);
v___x_1607_ = lean_box(0);
v___x_1608_ = ((lean_object*)(lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0));
v_sz_1609_ = lean_array_size(v_cs_1606_);
v___x_1610_ = ((size_t)0ULL);
v___x_1611_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0_spec__1(v_stx_1603_, v_ctx_1604_, v_cs_1606_, v_sz_1609_, v___x_1610_, v___x_1608_);
v_fst_1612_ = lean_ctor_get(v___x_1611_, 0);
lean_inc(v_fst_1612_);
lean_dec_ref(v___x_1611_);
if (lean_obj_tag(v_fst_1612_) == 0)
{
return v___x_1607_;
}
else
{
lean_object* v_val_1613_; 
v_val_1613_ = lean_ctor_get(v_fst_1612_, 0);
lean_inc(v_val_1613_);
lean_dec_ref_known(v_fst_1612_, 1);
return v_val_1613_;
}
}
else
{
lean_object* v_vs_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; size_t v_sz_1617_; size_t v___x_1618_; lean_object* v___x_1619_; lean_object* v_fst_1620_; 
v_vs_1614_ = lean_ctor_get(v_x_1605_, 0);
v___x_1615_ = lean_box(0);
v___x_1616_ = ((lean_object*)(lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0));
v_sz_1617_ = lean_array_size(v_vs_1614_);
v___x_1618_ = ((size_t)0ULL);
v___x_1619_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1(v_stx_1603_, v_ctx_1604_, v_vs_1614_, v_sz_1617_, v___x_1618_, v___x_1616_);
v_fst_1620_ = lean_ctor_get(v___x_1619_, 0);
lean_inc(v_fst_1620_);
lean_dec_ref(v___x_1619_);
if (lean_obj_tag(v_fst_1620_) == 0)
{
return v___x_1615_;
}
else
{
lean_object* v_val_1621_; 
v_val_1621_ = lean_ctor_get(v_fst_1620_, 0);
lean_inc(v_val_1621_);
lean_dec_ref_known(v_fst_1620_, 1);
return v_val_1621_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0(lean_object* v_stx_1622_, lean_object* v_ctx_1623_, lean_object* v_t_1624_){
_start:
{
lean_object* v_root_1625_; lean_object* v_tail_1626_; lean_object* v___x_1627_; 
v_root_1625_ = lean_ctor_get(v_t_1624_, 0);
v_tail_1626_ = lean_ctor_get(v_t_1624_, 1);
lean_inc_ref(v_ctx_1623_);
lean_inc(v_stx_1622_);
v___x_1627_ = lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0(v_stx_1622_, v_ctx_1623_, v_root_1625_);
if (lean_obj_tag(v___x_1627_) == 0)
{
lean_object* v___x_1628_; size_t v_sz_1629_; size_t v___x_1630_; lean_object* v___x_1631_; lean_object* v_fst_1632_; 
v___x_1628_ = ((lean_object*)(lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___closed__0));
v_sz_1629_ = lean_array_size(v_tail_1626_);
v___x_1630_ = ((size_t)0ULL);
v___x_1631_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1(v_stx_1622_, v_ctx_1623_, v_tail_1626_, v_sz_1629_, v___x_1630_, v___x_1628_);
v_fst_1632_ = lean_ctor_get(v___x_1631_, 0);
lean_inc(v_fst_1632_);
lean_dec_ref(v___x_1631_);
if (lean_obj_tag(v_fst_1632_) == 0)
{
return v___x_1627_;
}
else
{
lean_object* v_val_1633_; 
v_val_1633_ = lean_ctor_get(v_fst_1632_, 0);
lean_inc(v_val_1633_);
lean_dec_ref_known(v_fst_1632_, 1);
return v_val_1633_;
}
}
else
{
lean_dec_ref(v_ctx_1623_);
lean_dec(v_stx_1622_);
return v___x_1627_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0___boxed(lean_object* v_stx_1634_, lean_object* v_ctx_1635_, lean_object* v_t_1636_){
_start:
{
lean_object* v_res_1637_; 
v_res_1637_ = lp_batteries_Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0(v_stx_1634_, v_ctx_1635_, v_t_1636_);
lean_dec_ref(v_t_1636_);
return v_res_1637_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1___boxed(lean_object* v_stx_1638_, lean_object* v_ctx_1639_, lean_object* v_as_1640_, lean_object* v_sz_1641_, lean_object* v_i_1642_, lean_object* v_b_1643_){
_start:
{
size_t v_sz_boxed_1644_; size_t v_i_boxed_1645_; lean_object* v_res_1646_; 
v_sz_boxed_1644_ = lean_unbox_usize(v_sz_1641_);
lean_dec(v_sz_1641_);
v_i_boxed_1645_ = lean_unbox_usize(v_i_1642_);
lean_dec(v_i_1642_);
v_res_1646_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__1(v_stx_1638_, v_ctx_1639_, v_as_1640_, v_sz_boxed_1644_, v_i_boxed_1645_, v_b_1643_);
lean_dec_ref(v_b_1643_);
lean_dec_ref(v_as_1640_);
return v_res_1646_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0_spec__1___boxed(lean_object* v_stx_1647_, lean_object* v_ctx_1648_, lean_object* v_as_1649_, lean_object* v_sz_1650_, lean_object* v_i_1651_, lean_object* v_b_1652_){
_start:
{
size_t v_sz_boxed_1653_; size_t v_i_boxed_1654_; lean_object* v_res_1655_; 
v_sz_boxed_1653_ = lean_unbox_usize(v_sz_1650_);
lean_dec(v_sz_1650_);
v_i_boxed_1654_ = lean_unbox_usize(v_i_1651_);
lean_dec(v_i_1651_);
v_res_1655_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0_spec__1(v_stx_1647_, v_ctx_1648_, v_as_1649_, v_sz_boxed_1653_, v_i_boxed_1654_, v_b_1652_);
lean_dec_ref(v_b_1652_);
lean_dec_ref(v_as_1649_);
return v_res_1655_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0___boxed(lean_object* v_stx_1656_, lean_object* v_ctx_1657_, lean_object* v_x_1658_){
_start:
{
lean_object* v_res_1659_; 
v_res_1659_ = lp_batteries_Lean_PersistentArray_findSomeMAux___at___00Lean_PersistentArray_findSomeM_x3f___at___00Batteries_CodeAction_findTermInfoWithCtx_x3f_spec__0_spec__0(v_stx_1656_, v_ctx_1657_, v_x_1658_);
lean_dec_ref(v_x_1658_);
return v_res_1659_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Batteries_CodeAction_casesExpand_spec__9(lean_object* v_opts_1660_, lean_object* v_opt_1661_){
_start:
{
lean_object* v_name_1662_; lean_object* v_defValue_1663_; lean_object* v_map_1664_; lean_object* v___x_1665_; 
v_name_1662_ = lean_ctor_get(v_opt_1661_, 0);
v_defValue_1663_ = lean_ctor_get(v_opt_1661_, 1);
v_map_1664_ = lean_ctor_get(v_opts_1660_, 0);
v___x_1665_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1664_, v_name_1662_);
if (lean_obj_tag(v___x_1665_) == 0)
{
uint8_t v___x_1666_; 
v___x_1666_ = lean_unbox(v_defValue_1663_);
return v___x_1666_;
}
else
{
lean_object* v_val_1667_; 
v_val_1667_ = lean_ctor_get(v___x_1665_, 0);
lean_inc(v_val_1667_);
lean_dec_ref_known(v___x_1665_, 1);
if (lean_obj_tag(v_val_1667_) == 1)
{
uint8_t v_v_1668_; 
v_v_1668_ = lean_ctor_get_uint8(v_val_1667_, 0);
lean_dec_ref_known(v_val_1667_, 0);
return v_v_1668_;
}
else
{
uint8_t v___x_1669_; 
lean_dec(v_val_1667_);
v___x_1669_ = lean_unbox(v_defValue_1663_);
return v___x_1669_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Batteries_CodeAction_casesExpand_spec__9___boxed(lean_object* v_opts_1670_, lean_object* v_opt_1671_){
_start:
{
uint8_t v_res_1672_; lean_object* v_r_1673_; 
v_res_1672_ = lp_batteries_Lean_Option_get___at___00Batteries_CodeAction_casesExpand_spec__9(v_opts_1670_, v_opt_1671_);
lean_dec_ref(v_opt_1671_);
lean_dec_ref(v_opts_1670_);
v_r_1673_ = lean_box(v_res_1672_);
return v_r_1673_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7_spec__7(lean_object* v_msgData_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_){
_start:
{
lean_object* v___x_1680_; lean_object* v_env_1681_; lean_object* v___x_1682_; lean_object* v_mctx_1683_; lean_object* v_lctx_1684_; lean_object* v_options_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; 
v___x_1680_ = lean_st_ref_get(v___y_1678_);
v_env_1681_ = lean_ctor_get(v___x_1680_, 0);
lean_inc_ref(v_env_1681_);
lean_dec(v___x_1680_);
v___x_1682_ = lean_st_ref_get(v___y_1676_);
v_mctx_1683_ = lean_ctor_get(v___x_1682_, 0);
lean_inc_ref(v_mctx_1683_);
lean_dec(v___x_1682_);
v_lctx_1684_ = lean_ctor_get(v___y_1675_, 2);
v_options_1685_ = lean_ctor_get(v___y_1677_, 2);
lean_inc_ref(v_options_1685_);
lean_inc_ref(v_lctx_1684_);
v___x_1686_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1686_, 0, v_env_1681_);
lean_ctor_set(v___x_1686_, 1, v_mctx_1683_);
lean_ctor_set(v___x_1686_, 2, v_lctx_1684_);
lean_ctor_set(v___x_1686_, 3, v_options_1685_);
v___x_1687_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1687_, 0, v___x_1686_);
lean_ctor_set(v___x_1687_, 1, v_msgData_1674_);
v___x_1688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1688_, 0, v___x_1687_);
return v___x_1688_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7_spec__7___boxed(lean_object* v_msgData_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_){
_start:
{
lean_object* v_res_1695_; 
v_res_1695_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7_spec__7(v_msgData_1689_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_);
lean_dec(v___y_1693_);
lean_dec_ref(v___y_1692_);
lean_dec(v___y_1691_);
lean_dec_ref(v___y_1690_);
return v_res_1695_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg(lean_object* v_msg_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_){
_start:
{
lean_object* v_ref_1702_; lean_object* v___x_1703_; lean_object* v_a_1704_; lean_object* v___x_1706_; uint8_t v_isShared_1707_; uint8_t v_isSharedCheck_1712_; 
v_ref_1702_ = lean_ctor_get(v___y_1699_, 5);
v___x_1703_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7_spec__7(v_msg_1696_, v___y_1697_, v___y_1698_, v___y_1699_, v___y_1700_);
v_a_1704_ = lean_ctor_get(v___x_1703_, 0);
v_isSharedCheck_1712_ = !lean_is_exclusive(v___x_1703_);
if (v_isSharedCheck_1712_ == 0)
{
v___x_1706_ = v___x_1703_;
v_isShared_1707_ = v_isSharedCheck_1712_;
goto v_resetjp_1705_;
}
else
{
lean_inc(v_a_1704_);
lean_dec(v___x_1703_);
v___x_1706_ = lean_box(0);
v_isShared_1707_ = v_isSharedCheck_1712_;
goto v_resetjp_1705_;
}
v_resetjp_1705_:
{
lean_object* v___x_1708_; lean_object* v___x_1710_; 
lean_inc(v_ref_1702_);
v___x_1708_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1708_, 0, v_ref_1702_);
lean_ctor_set(v___x_1708_, 1, v_a_1704_);
if (v_isShared_1707_ == 0)
{
lean_ctor_set_tag(v___x_1706_, 1);
lean_ctor_set(v___x_1706_, 0, v___x_1708_);
v___x_1710_ = v___x_1706_;
goto v_reusejp_1709_;
}
else
{
lean_object* v_reuseFailAlloc_1711_; 
v_reuseFailAlloc_1711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1711_, 0, v___x_1708_);
v___x_1710_ = v_reuseFailAlloc_1711_;
goto v_reusejp_1709_;
}
v_reusejp_1709_:
{
return v___x_1710_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg___boxed(lean_object* v_msg_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_){
_start:
{
lean_object* v_res_1719_; 
v_res_1719_ = lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg(v_msg_1713_, v___y_1714_, v___y_1715_, v___y_1716_, v___y_1717_);
lean_dec(v___y_1717_);
lean_dec_ref(v___y_1716_);
lean_dec(v___y_1715_);
lean_dec_ref(v___y_1714_);
return v_res_1719_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg(lean_object* v_ref_1720_, lean_object* v_msg_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_){
_start:
{
lean_object* v_fileName_1727_; lean_object* v_fileMap_1728_; lean_object* v_options_1729_; lean_object* v_currRecDepth_1730_; lean_object* v_maxRecDepth_1731_; lean_object* v_ref_1732_; lean_object* v_currNamespace_1733_; lean_object* v_openDecls_1734_; lean_object* v_initHeartbeats_1735_; lean_object* v_maxHeartbeats_1736_; lean_object* v_quotContext_1737_; lean_object* v_currMacroScope_1738_; uint8_t v_diag_1739_; lean_object* v_cancelTk_x3f_1740_; uint8_t v_suppressElabErrors_1741_; lean_object* v_inheritedTraceOptions_1742_; lean_object* v_ref_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; 
v_fileName_1727_ = lean_ctor_get(v___y_1724_, 0);
v_fileMap_1728_ = lean_ctor_get(v___y_1724_, 1);
v_options_1729_ = lean_ctor_get(v___y_1724_, 2);
v_currRecDepth_1730_ = lean_ctor_get(v___y_1724_, 3);
v_maxRecDepth_1731_ = lean_ctor_get(v___y_1724_, 4);
v_ref_1732_ = lean_ctor_get(v___y_1724_, 5);
v_currNamespace_1733_ = lean_ctor_get(v___y_1724_, 6);
v_openDecls_1734_ = lean_ctor_get(v___y_1724_, 7);
v_initHeartbeats_1735_ = lean_ctor_get(v___y_1724_, 8);
v_maxHeartbeats_1736_ = lean_ctor_get(v___y_1724_, 9);
v_quotContext_1737_ = lean_ctor_get(v___y_1724_, 10);
v_currMacroScope_1738_ = lean_ctor_get(v___y_1724_, 11);
v_diag_1739_ = lean_ctor_get_uint8(v___y_1724_, sizeof(void*)*14);
v_cancelTk_x3f_1740_ = lean_ctor_get(v___y_1724_, 12);
v_suppressElabErrors_1741_ = lean_ctor_get_uint8(v___y_1724_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1742_ = lean_ctor_get(v___y_1724_, 13);
v_ref_1743_ = l_Lean_replaceRef(v_ref_1720_, v_ref_1732_);
lean_inc_ref(v_inheritedTraceOptions_1742_);
lean_inc(v_cancelTk_x3f_1740_);
lean_inc(v_currMacroScope_1738_);
lean_inc(v_quotContext_1737_);
lean_inc(v_maxHeartbeats_1736_);
lean_inc(v_initHeartbeats_1735_);
lean_inc(v_openDecls_1734_);
lean_inc(v_currNamespace_1733_);
lean_inc(v_maxRecDepth_1731_);
lean_inc(v_currRecDepth_1730_);
lean_inc_ref(v_options_1729_);
lean_inc_ref(v_fileMap_1728_);
lean_inc_ref(v_fileName_1727_);
v___x_1744_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1744_, 0, v_fileName_1727_);
lean_ctor_set(v___x_1744_, 1, v_fileMap_1728_);
lean_ctor_set(v___x_1744_, 2, v_options_1729_);
lean_ctor_set(v___x_1744_, 3, v_currRecDepth_1730_);
lean_ctor_set(v___x_1744_, 4, v_maxRecDepth_1731_);
lean_ctor_set(v___x_1744_, 5, v_ref_1743_);
lean_ctor_set(v___x_1744_, 6, v_currNamespace_1733_);
lean_ctor_set(v___x_1744_, 7, v_openDecls_1734_);
lean_ctor_set(v___x_1744_, 8, v_initHeartbeats_1735_);
lean_ctor_set(v___x_1744_, 9, v_maxHeartbeats_1736_);
lean_ctor_set(v___x_1744_, 10, v_quotContext_1737_);
lean_ctor_set(v___x_1744_, 11, v_currMacroScope_1738_);
lean_ctor_set(v___x_1744_, 12, v_cancelTk_x3f_1740_);
lean_ctor_set(v___x_1744_, 13, v_inheritedTraceOptions_1742_);
lean_ctor_set_uint8(v___x_1744_, sizeof(void*)*14, v_diag_1739_);
lean_ctor_set_uint8(v___x_1744_, sizeof(void*)*14 + 1, v_suppressElabErrors_1741_);
v___x_1745_ = lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg(v_msg_1721_, v___y_1722_, v___y_1723_, v___x_1744_, v___y_1725_);
lean_dec_ref_known(v___x_1744_, 14);
return v___x_1745_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg___boxed(lean_object* v_ref_1746_, lean_object* v_msg_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_){
_start:
{
lean_object* v_res_1753_; 
v_res_1753_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg(v_ref_1746_, v_msg_1747_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_ref_1746_);
return v_res_1753_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__0(void){
_start:
{
lean_object* v___x_1754_; 
v___x_1754_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1754_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1(void){
_start:
{
lean_object* v___x_1755_; lean_object* v___x_1756_; 
v___x_1755_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__0, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__0_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__0);
v___x_1756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1756_, 0, v___x_1755_);
return v___x_1756_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__2(void){
_start:
{
lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; 
v___x_1757_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1);
v___x_1758_ = lean_unsigned_to_nat(0u);
v___x_1759_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1759_, 0, v___x_1758_);
lean_ctor_set(v___x_1759_, 1, v___x_1758_);
lean_ctor_set(v___x_1759_, 2, v___x_1758_);
lean_ctor_set(v___x_1759_, 3, v___x_1758_);
lean_ctor_set(v___x_1759_, 4, v___x_1757_);
lean_ctor_set(v___x_1759_, 5, v___x_1757_);
lean_ctor_set(v___x_1759_, 6, v___x_1757_);
lean_ctor_set(v___x_1759_, 7, v___x_1757_);
lean_ctor_set(v___x_1759_, 8, v___x_1757_);
lean_ctor_set(v___x_1759_, 9, v___x_1757_);
return v___x_1759_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__3(void){
_start:
{
lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; 
v___x_1760_ = lean_unsigned_to_nat(32u);
v___x_1761_ = lean_mk_empty_array_with_capacity(v___x_1760_);
v___x_1762_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1762_, 0, v___x_1761_);
return v___x_1762_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__4(void){
_start:
{
size_t v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; 
v___x_1763_ = ((size_t)5ULL);
v___x_1764_ = lean_unsigned_to_nat(0u);
v___x_1765_ = lean_unsigned_to_nat(32u);
v___x_1766_ = lean_mk_empty_array_with_capacity(v___x_1765_);
v___x_1767_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__3, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__3_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__3);
v___x_1768_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1768_, 0, v___x_1767_);
lean_ctor_set(v___x_1768_, 1, v___x_1766_);
lean_ctor_set(v___x_1768_, 2, v___x_1764_);
lean_ctor_set(v___x_1768_, 3, v___x_1764_);
lean_ctor_set_usize(v___x_1768_, 4, v___x_1763_);
return v___x_1768_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__5(void){
_start:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; 
v___x_1769_ = lean_box(1);
v___x_1770_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__4, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__4_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__4);
v___x_1771_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__1);
v___x_1772_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1772_, 0, v___x_1771_);
lean_ctor_set(v___x_1772_, 1, v___x_1770_);
lean_ctor_set(v___x_1772_, 2, v___x_1769_);
return v___x_1772_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7(void){
_start:
{
lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1774_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__6));
v___x_1775_ = l_Lean_stringToMessageData(v___x_1774_);
return v___x_1775_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__9(void){
_start:
{
lean_object* v___x_1777_; lean_object* v___x_1778_; 
v___x_1777_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__8));
v___x_1778_ = l_Lean_stringToMessageData(v___x_1777_);
return v___x_1778_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__11(void){
_start:
{
lean_object* v___x_1780_; lean_object* v___x_1781_; 
v___x_1780_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__10));
v___x_1781_ = l_Lean_stringToMessageData(v___x_1780_);
return v___x_1781_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__13(void){
_start:
{
lean_object* v___x_1783_; lean_object* v___x_1784_; 
v___x_1783_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__12));
v___x_1784_ = l_Lean_stringToMessageData(v___x_1783_);
return v___x_1784_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__15(void){
_start:
{
lean_object* v___x_1786_; lean_object* v___x_1787_; 
v___x_1786_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__14));
v___x_1787_ = l_Lean_stringToMessageData(v___x_1786_);
return v___x_1787_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__17(void){
_start:
{
lean_object* v___x_1789_; lean_object* v___x_1790_; 
v___x_1789_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__16));
v___x_1790_ = l_Lean_stringToMessageData(v___x_1789_);
return v___x_1790_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__19(void){
_start:
{
lean_object* v___x_1792_; lean_object* v___x_1793_; 
v___x_1792_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__18));
v___x_1793_ = l_Lean_stringToMessageData(v___x_1792_);
return v___x_1793_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg(lean_object* v_msg_1794_, lean_object* v_declHint_1795_, lean_object* v___y_1796_){
_start:
{
lean_object* v___x_1798_; lean_object* v_env_1799_; uint8_t v___x_1800_; 
v___x_1798_ = lean_st_ref_get(v___y_1796_);
v_env_1799_ = lean_ctor_get(v___x_1798_, 0);
lean_inc_ref(v_env_1799_);
lean_dec(v___x_1798_);
v___x_1800_ = l_Lean_Name_isAnonymous(v_declHint_1795_);
if (v___x_1800_ == 0)
{
uint8_t v_isExporting_1801_; 
v_isExporting_1801_ = lean_ctor_get_uint8(v_env_1799_, sizeof(void*)*8);
if (v_isExporting_1801_ == 0)
{
lean_object* v___x_1802_; 
lean_dec_ref(v_env_1799_);
lean_dec(v_declHint_1795_);
v___x_1802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1802_, 0, v_msg_1794_);
return v___x_1802_;
}
else
{
lean_object* v___x_1803_; uint8_t v___x_1804_; 
lean_inc_ref(v_env_1799_);
v___x_1803_ = l_Lean_Environment_setExporting(v_env_1799_, v___x_1800_);
lean_inc(v_declHint_1795_);
lean_inc_ref(v___x_1803_);
v___x_1804_ = l_Lean_Environment_contains(v___x_1803_, v_declHint_1795_, v_isExporting_1801_);
if (v___x_1804_ == 0)
{
lean_object* v___x_1805_; 
lean_dec_ref(v___x_1803_);
lean_dec_ref(v_env_1799_);
lean_dec(v_declHint_1795_);
v___x_1805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1805_, 0, v_msg_1794_);
return v___x_1805_;
}
else
{
lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v_c_1811_; lean_object* v___x_1812_; 
v___x_1806_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__2, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__2_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__2);
v___x_1807_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__5, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__5_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__5);
v___x_1808_ = l_Lean_Options_empty;
v___x_1809_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1809_, 0, v___x_1803_);
lean_ctor_set(v___x_1809_, 1, v___x_1806_);
lean_ctor_set(v___x_1809_, 2, v___x_1807_);
lean_ctor_set(v___x_1809_, 3, v___x_1808_);
lean_inc(v_declHint_1795_);
v___x_1810_ = l_Lean_MessageData_ofConstName(v_declHint_1795_, v___x_1800_);
v_c_1811_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_1811_, 0, v___x_1809_);
lean_ctor_set(v_c_1811_, 1, v___x_1810_);
v___x_1812_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1799_, v_declHint_1795_);
if (lean_obj_tag(v___x_1812_) == 0)
{
lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; 
lean_dec_ref(v_env_1799_);
lean_dec(v_declHint_1795_);
v___x_1813_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7);
v___x_1814_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1814_, 0, v___x_1813_);
lean_ctor_set(v___x_1814_, 1, v_c_1811_);
v___x_1815_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__9, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__9_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__9);
v___x_1816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1816_, 0, v___x_1814_);
lean_ctor_set(v___x_1816_, 1, v___x_1815_);
v___x_1817_ = l_Lean_MessageData_note(v___x_1816_);
v___x_1818_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1818_, 0, v_msg_1794_);
lean_ctor_set(v___x_1818_, 1, v___x_1817_);
v___x_1819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1819_, 0, v___x_1818_);
return v___x_1819_;
}
else
{
lean_object* v_val_1820_; lean_object* v___x_1822_; uint8_t v_isShared_1823_; uint8_t v_isSharedCheck_1855_; 
v_val_1820_ = lean_ctor_get(v___x_1812_, 0);
v_isSharedCheck_1855_ = !lean_is_exclusive(v___x_1812_);
if (v_isSharedCheck_1855_ == 0)
{
v___x_1822_ = v___x_1812_;
v_isShared_1823_ = v_isSharedCheck_1855_;
goto v_resetjp_1821_;
}
else
{
lean_inc(v_val_1820_);
lean_dec(v___x_1812_);
v___x_1822_ = lean_box(0);
v_isShared_1823_ = v_isSharedCheck_1855_;
goto v_resetjp_1821_;
}
v_resetjp_1821_:
{
lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v_mod_1827_; uint8_t v___x_1828_; 
v___x_1824_ = lean_box(0);
v___x_1825_ = l_Lean_Environment_header(v_env_1799_);
lean_dec_ref(v_env_1799_);
v___x_1826_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1825_);
v_mod_1827_ = lean_array_get(v___x_1824_, v___x_1826_, v_val_1820_);
lean_dec(v_val_1820_);
lean_dec_ref(v___x_1826_);
v___x_1828_ = l_Lean_isPrivateName(v_declHint_1795_);
lean_dec(v_declHint_1795_);
if (v___x_1828_ == 0)
{
lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1840_; 
v___x_1829_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__11, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__11_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__11);
v___x_1830_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1830_, 0, v___x_1829_);
lean_ctor_set(v___x_1830_, 1, v_c_1811_);
v___x_1831_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__13, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__13_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__13);
v___x_1832_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1832_, 0, v___x_1830_);
lean_ctor_set(v___x_1832_, 1, v___x_1831_);
v___x_1833_ = l_Lean_MessageData_ofName(v_mod_1827_);
v___x_1834_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1832_);
lean_ctor_set(v___x_1834_, 1, v___x_1833_);
v___x_1835_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__15, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__15_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__15);
v___x_1836_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1834_);
lean_ctor_set(v___x_1836_, 1, v___x_1835_);
v___x_1837_ = l_Lean_MessageData_note(v___x_1836_);
v___x_1838_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1838_, 0, v_msg_1794_);
lean_ctor_set(v___x_1838_, 1, v___x_1837_);
if (v_isShared_1823_ == 0)
{
lean_ctor_set_tag(v___x_1822_, 0);
lean_ctor_set(v___x_1822_, 0, v___x_1838_);
v___x_1840_ = v___x_1822_;
goto v_reusejp_1839_;
}
else
{
lean_object* v_reuseFailAlloc_1841_; 
v_reuseFailAlloc_1841_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1841_, 0, v___x_1838_);
v___x_1840_ = v_reuseFailAlloc_1841_;
goto v_reusejp_1839_;
}
v_reusejp_1839_:
{
return v___x_1840_;
}
}
else
{
lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1853_; 
v___x_1842_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__7);
v___x_1843_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1843_, 0, v___x_1842_);
lean_ctor_set(v___x_1843_, 1, v_c_1811_);
v___x_1844_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__17, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__17_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__17);
v___x_1845_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1845_, 0, v___x_1843_);
lean_ctor_set(v___x_1845_, 1, v___x_1844_);
v___x_1846_ = l_Lean_MessageData_ofName(v_mod_1827_);
v___x_1847_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1847_, 0, v___x_1845_);
lean_ctor_set(v___x_1847_, 1, v___x_1846_);
v___x_1848_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__19, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__19_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___closed__19);
v___x_1849_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1849_, 0, v___x_1847_);
lean_ctor_set(v___x_1849_, 1, v___x_1848_);
v___x_1850_ = l_Lean_MessageData_note(v___x_1849_);
v___x_1851_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1851_, 0, v_msg_1794_);
lean_ctor_set(v___x_1851_, 1, v___x_1850_);
if (v_isShared_1823_ == 0)
{
lean_ctor_set_tag(v___x_1822_, 0);
lean_ctor_set(v___x_1822_, 0, v___x_1851_);
v___x_1853_ = v___x_1822_;
goto v_reusejp_1852_;
}
else
{
lean_object* v_reuseFailAlloc_1854_; 
v_reuseFailAlloc_1854_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1854_, 0, v___x_1851_);
v___x_1853_ = v_reuseFailAlloc_1854_;
goto v_reusejp_1852_;
}
v_reusejp_1852_:
{
return v___x_1853_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1856_; 
lean_dec_ref(v_env_1799_);
lean_dec(v_declHint_1795_);
v___x_1856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1856_, 0, v_msg_1794_);
return v___x_1856_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg___boxed(lean_object* v_msg_1857_, lean_object* v_declHint_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_){
_start:
{
lean_object* v_res_1861_; 
v_res_1861_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg(v_msg_1857_, v_declHint_1858_, v___y_1859_);
lean_dec(v___y_1859_);
return v_res_1861_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18(lean_object* v_msg_1862_, lean_object* v_declHint_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_){
_start:
{
lean_object* v___x_1869_; lean_object* v_a_1870_; lean_object* v___x_1872_; uint8_t v_isShared_1873_; uint8_t v_isSharedCheck_1879_; 
v___x_1869_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg(v_msg_1862_, v_declHint_1863_, v___y_1867_);
v_a_1870_ = lean_ctor_get(v___x_1869_, 0);
v_isSharedCheck_1879_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1879_ == 0)
{
v___x_1872_ = v___x_1869_;
v_isShared_1873_ = v_isSharedCheck_1879_;
goto v_resetjp_1871_;
}
else
{
lean_inc(v_a_1870_);
lean_dec(v___x_1869_);
v___x_1872_ = lean_box(0);
v_isShared_1873_ = v_isSharedCheck_1879_;
goto v_resetjp_1871_;
}
v_resetjp_1871_:
{
lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1877_; 
v___x_1874_ = l_Lean_unknownIdentifierMessageTag;
v___x_1875_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1875_, 0, v___x_1874_);
lean_ctor_set(v___x_1875_, 1, v_a_1870_);
if (v_isShared_1873_ == 0)
{
lean_ctor_set(v___x_1872_, 0, v___x_1875_);
v___x_1877_ = v___x_1872_;
goto v_reusejp_1876_;
}
else
{
lean_object* v_reuseFailAlloc_1878_; 
v_reuseFailAlloc_1878_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1878_, 0, v___x_1875_);
v___x_1877_ = v_reuseFailAlloc_1878_;
goto v_reusejp_1876_;
}
v_reusejp_1876_:
{
return v___x_1877_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18___boxed(lean_object* v_msg_1880_, lean_object* v_declHint_1881_, lean_object* v___y_1882_, lean_object* v___y_1883_, lean_object* v___y_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_){
_start:
{
lean_object* v_res_1887_; 
v_res_1887_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18(v_msg_1880_, v_declHint_1881_, v___y_1882_, v___y_1883_, v___y_1884_, v___y_1885_);
lean_dec(v___y_1885_);
lean_dec_ref(v___y_1884_);
lean_dec(v___y_1883_);
lean_dec_ref(v___y_1882_);
return v_res_1887_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg(lean_object* v_ref_1888_, lean_object* v_msg_1889_, lean_object* v_declHint_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_){
_start:
{
lean_object* v___x_1896_; lean_object* v_a_1897_; lean_object* v___x_1898_; 
v___x_1896_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18(v_msg_1889_, v_declHint_1890_, v___y_1891_, v___y_1892_, v___y_1893_, v___y_1894_);
v_a_1897_ = lean_ctor_get(v___x_1896_, 0);
lean_inc(v_a_1897_);
lean_dec_ref(v___x_1896_);
v___x_1898_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg(v_ref_1888_, v_a_1897_, v___y_1891_, v___y_1892_, v___y_1893_, v___y_1894_);
return v___x_1898_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg___boxed(lean_object* v_ref_1899_, lean_object* v_msg_1900_, lean_object* v_declHint_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_){
_start:
{
lean_object* v_res_1907_; 
v_res_1907_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg(v_ref_1899_, v_msg_1900_, v_declHint_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v_ref_1899_);
return v_res_1907_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__1(void){
_start:
{
lean_object* v___x_1909_; lean_object* v___x_1910_; 
v___x_1909_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__0));
v___x_1910_ = l_Lean_stringToMessageData(v___x_1909_);
return v___x_1910_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_1912_; lean_object* v___x_1913_; 
v___x_1912_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__2));
v___x_1913_ = l_Lean_stringToMessageData(v___x_1912_);
return v___x_1913_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg(lean_object* v_ref_1914_, lean_object* v_constName_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_){
_start:
{
lean_object* v___x_1921_; uint8_t v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; 
v___x_1921_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__1, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__1_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__1);
v___x_1922_ = 0;
lean_inc(v_constName_1915_);
v___x_1923_ = l_Lean_MessageData_ofConstName(v_constName_1915_, v___x_1922_);
v___x_1924_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1924_, 0, v___x_1921_);
lean_ctor_set(v___x_1924_, 1, v___x_1923_);
v___x_1925_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__3, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__3_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___closed__3);
v___x_1926_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1926_, 0, v___x_1924_);
lean_ctor_set(v___x_1926_, 1, v___x_1925_);
v___x_1927_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg(v_ref_1914_, v___x_1926_, v_constName_1915_, v___y_1916_, v___y_1917_, v___y_1918_, v___y_1919_);
return v___x_1927_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg___boxed(lean_object* v_ref_1928_, lean_object* v_constName_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_){
_start:
{
lean_object* v_res_1935_; 
v_res_1935_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg(v_ref_1928_, v_constName_1929_, v___y_1930_, v___y_1931_, v___y_1932_, v___y_1933_);
lean_dec(v___y_1933_);
lean_dec_ref(v___y_1932_);
lean_dec(v___y_1931_);
lean_dec_ref(v___y_1930_);
lean_dec(v_ref_1928_);
return v_res_1935_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg(lean_object* v_constName_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_){
_start:
{
lean_object* v_ref_1942_; lean_object* v___x_1943_; 
v_ref_1942_ = lean_ctor_get(v___y_1939_, 5);
v___x_1943_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg(v_ref_1942_, v_constName_1936_, v___y_1937_, v___y_1938_, v___y_1939_, v___y_1940_);
return v___x_1943_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg___boxed(lean_object* v_constName_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
lean_object* v_res_1950_; 
v_res_1950_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg(v_constName_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
lean_dec(v___y_1948_);
lean_dec_ref(v___y_1947_);
lean_dec(v___y_1946_);
lean_dec_ref(v___y_1945_);
return v_res_1950_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8(lean_object* v_constName_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_){
_start:
{
lean_object* v___x_1957_; lean_object* v_env_1958_; uint8_t v___x_1959_; lean_object* v___x_1960_; 
v___x_1957_ = lean_st_ref_get(v___y_1955_);
v_env_1958_ = lean_ctor_get(v___x_1957_, 0);
lean_inc_ref(v_env_1958_);
lean_dec(v___x_1957_);
v___x_1959_ = 0;
lean_inc(v_constName_1951_);
v___x_1960_ = l_Lean_Environment_find_x3f(v_env_1958_, v_constName_1951_, v___x_1959_);
if (lean_obj_tag(v___x_1960_) == 0)
{
lean_object* v___x_1961_; 
v___x_1961_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg(v_constName_1951_, v___y_1952_, v___y_1953_, v___y_1954_, v___y_1955_);
return v___x_1961_;
}
else
{
lean_object* v_val_1962_; lean_object* v___x_1964_; uint8_t v_isShared_1965_; uint8_t v_isSharedCheck_1969_; 
lean_dec(v_constName_1951_);
v_val_1962_ = lean_ctor_get(v___x_1960_, 0);
v_isSharedCheck_1969_ = !lean_is_exclusive(v___x_1960_);
if (v_isSharedCheck_1969_ == 0)
{
v___x_1964_ = v___x_1960_;
v_isShared_1965_ = v_isSharedCheck_1969_;
goto v_resetjp_1963_;
}
else
{
lean_inc(v_val_1962_);
lean_dec(v___x_1960_);
v___x_1964_ = lean_box(0);
v_isShared_1965_ = v_isSharedCheck_1969_;
goto v_resetjp_1963_;
}
v_resetjp_1963_:
{
lean_object* v___x_1967_; 
if (v_isShared_1965_ == 0)
{
lean_ctor_set_tag(v___x_1964_, 0);
v___x_1967_ = v___x_1964_;
goto v_reusejp_1966_;
}
else
{
lean_object* v_reuseFailAlloc_1968_; 
v_reuseFailAlloc_1968_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1968_, 0, v_val_1962_);
v___x_1967_ = v_reuseFailAlloc_1968_;
goto v_reusejp_1966_;
}
v_reusejp_1966_:
{
return v___x_1967_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8___boxed(lean_object* v_constName_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_){
_start:
{
lean_object* v_res_1976_; 
v_res_1976_ = lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8(v_constName_1970_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_);
lean_dec(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec(v___y_1972_);
lean_dec_ref(v___y_1971_);
return v_res_1976_;
}
}
static lean_object* _init_lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1978_; lean_object* v___x_1979_; 
v___x_1978_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__0));
v___x_1979_ = l_Lean_stringToMessageData(v___x_1978_);
return v___x_1979_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0(lean_object* v_fst_1980_, lean_object* v___x_1981_, uint8_t v_fst_1982_, lean_object* v_targets_1983_, lean_object* v_node_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_){
_start:
{
lean_object* v___y_1991_; lean_object* v___y_1992_; lean_object* v___y_1993_; lean_object* v___y_1994_; lean_object* v___y_1998_; lean_object* v___y_1999_; lean_object* v___y_2000_; lean_object* v___y_2001_; lean_object* v___y_2002_; lean_object* v___y_2033_; lean_object* v___y_2034_; lean_object* v___y_2035_; lean_object* v___y_2036_; 
if (lean_obj_tag(v_fst_1980_) == 0)
{
lean_object* v_options_2073_; lean_object* v___x_2074_; uint8_t v___x_2075_; 
lean_dec_ref(v_node_1984_);
v_options_2073_ = lean_ctor_get(v___y_1987_, 2);
v___x_2074_ = l_Lean_tactic_customEliminators;
v___x_2075_ = lp_batteries_Lean_Option_get___at___00Batteries_CodeAction_casesExpand_spec__9(v_options_2073_, v___x_2074_);
if (v___x_2075_ == 0)
{
v___y_2033_ = v___y_1985_;
v___y_2034_ = v___y_1986_;
v___y_2035_ = v___y_1987_;
v___y_2036_ = v___y_1988_;
goto v___jp_2032_;
}
else
{
lean_object* v___x_2076_; 
v___x_2076_ = l_Lean_Meta_getCustomEliminator_x3f(v_targets_1983_, v_fst_1982_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
if (lean_obj_tag(v___x_2076_) == 0)
{
lean_object* v_a_2077_; 
v_a_2077_ = lean_ctor_get(v___x_2076_, 0);
lean_inc(v_a_2077_);
lean_dec_ref_known(v___x_2076_, 1);
if (lean_obj_tag(v_a_2077_) == 1)
{
lean_object* v_val_2078_; lean_object* v___x_2080_; uint8_t v_isShared_2081_; uint8_t v_isSharedCheck_2113_; 
lean_dec_ref(v___x_1981_);
v_val_2078_ = lean_ctor_get(v_a_2077_, 0);
v_isSharedCheck_2113_ = !lean_is_exclusive(v_a_2077_);
if (v_isSharedCheck_2113_ == 0)
{
v___x_2080_ = v_a_2077_;
v_isShared_2081_ = v_isSharedCheck_2113_;
goto v_resetjp_2079_;
}
else
{
lean_inc(v_val_2078_);
lean_dec(v_a_2077_);
v___x_2080_ = lean_box(0);
v_isShared_2081_ = v_isSharedCheck_2113_;
goto v_resetjp_2079_;
}
v_resetjp_2079_:
{
lean_object* v___x_2082_; 
v___x_2082_ = lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8(v_val_2078_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
if (lean_obj_tag(v___x_2082_) == 0)
{
lean_object* v_a_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; 
v_a_2083_ = lean_ctor_get(v___x_2082_, 0);
lean_inc(v_a_2083_);
lean_dec_ref_known(v___x_2082_, 1);
v___x_2084_ = l_Lean_ConstantInfo_type(v_a_2083_);
lean_dec(v_a_2083_);
v___x_2085_ = lp_batteries_Batteries_CodeAction_getElimExprNames(v___x_2084_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
if (lean_obj_tag(v___x_2085_) == 0)
{
lean_object* v_a_2086_; lean_object* v___x_2088_; uint8_t v_isShared_2089_; uint8_t v_isSharedCheck_2096_; 
v_a_2086_ = lean_ctor_get(v___x_2085_, 0);
v_isSharedCheck_2096_ = !lean_is_exclusive(v___x_2085_);
if (v_isSharedCheck_2096_ == 0)
{
v___x_2088_ = v___x_2085_;
v_isShared_2089_ = v_isSharedCheck_2096_;
goto v_resetjp_2087_;
}
else
{
lean_inc(v_a_2086_);
lean_dec(v___x_2085_);
v___x_2088_ = lean_box(0);
v_isShared_2089_ = v_isSharedCheck_2096_;
goto v_resetjp_2087_;
}
v_resetjp_2087_:
{
lean_object* v___x_2091_; 
if (v_isShared_2081_ == 0)
{
lean_ctor_set(v___x_2080_, 0, v_a_2086_);
v___x_2091_ = v___x_2080_;
goto v_reusejp_2090_;
}
else
{
lean_object* v_reuseFailAlloc_2095_; 
v_reuseFailAlloc_2095_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2095_, 0, v_a_2086_);
v___x_2091_ = v_reuseFailAlloc_2095_;
goto v_reusejp_2090_;
}
v_reusejp_2090_:
{
lean_object* v___x_2093_; 
if (v_isShared_2089_ == 0)
{
lean_ctor_set(v___x_2088_, 0, v___x_2091_);
v___x_2093_ = v___x_2088_;
goto v_reusejp_2092_;
}
else
{
lean_object* v_reuseFailAlloc_2094_; 
v_reuseFailAlloc_2094_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2094_, 0, v___x_2091_);
v___x_2093_ = v_reuseFailAlloc_2094_;
goto v_reusejp_2092_;
}
v_reusejp_2092_:
{
return v___x_2093_;
}
}
}
}
else
{
lean_object* v_a_2097_; lean_object* v___x_2099_; uint8_t v_isShared_2100_; uint8_t v_isSharedCheck_2104_; 
lean_del_object(v___x_2080_);
v_a_2097_ = lean_ctor_get(v___x_2085_, 0);
v_isSharedCheck_2104_ = !lean_is_exclusive(v___x_2085_);
if (v_isSharedCheck_2104_ == 0)
{
v___x_2099_ = v___x_2085_;
v_isShared_2100_ = v_isSharedCheck_2104_;
goto v_resetjp_2098_;
}
else
{
lean_inc(v_a_2097_);
lean_dec(v___x_2085_);
v___x_2099_ = lean_box(0);
v_isShared_2100_ = v_isSharedCheck_2104_;
goto v_resetjp_2098_;
}
v_resetjp_2098_:
{
lean_object* v___x_2102_; 
if (v_isShared_2100_ == 0)
{
v___x_2102_ = v___x_2099_;
goto v_reusejp_2101_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v_a_2097_);
v___x_2102_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2101_;
}
v_reusejp_2101_:
{
return v___x_2102_;
}
}
}
}
else
{
lean_object* v_a_2105_; lean_object* v___x_2107_; uint8_t v_isShared_2108_; uint8_t v_isSharedCheck_2112_; 
lean_del_object(v___x_2080_);
v_a_2105_ = lean_ctor_get(v___x_2082_, 0);
v_isSharedCheck_2112_ = !lean_is_exclusive(v___x_2082_);
if (v_isSharedCheck_2112_ == 0)
{
v___x_2107_ = v___x_2082_;
v_isShared_2108_ = v_isSharedCheck_2112_;
goto v_resetjp_2106_;
}
else
{
lean_inc(v_a_2105_);
lean_dec(v___x_2082_);
v___x_2107_ = lean_box(0);
v_isShared_2108_ = v_isSharedCheck_2112_;
goto v_resetjp_2106_;
}
v_resetjp_2106_:
{
lean_object* v___x_2110_; 
if (v_isShared_2108_ == 0)
{
v___x_2110_ = v___x_2107_;
goto v_reusejp_2109_;
}
else
{
lean_object* v_reuseFailAlloc_2111_; 
v_reuseFailAlloc_2111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2111_, 0, v_a_2105_);
v___x_2110_ = v_reuseFailAlloc_2111_;
goto v_reusejp_2109_;
}
v_reusejp_2109_:
{
return v___x_2110_;
}
}
}
}
}
else
{
lean_dec(v_a_2077_);
v___y_2033_ = v___y_1985_;
v___y_2034_ = v___y_1986_;
v___y_2035_ = v___y_1987_;
v___y_2036_ = v___y_1988_;
goto v___jp_2032_;
}
}
else
{
lean_object* v_a_2114_; lean_object* v___x_2116_; uint8_t v_isShared_2117_; uint8_t v_isSharedCheck_2121_; 
lean_dec_ref(v___x_1981_);
v_a_2114_ = lean_ctor_get(v___x_2076_, 0);
v_isSharedCheck_2121_ = !lean_is_exclusive(v___x_2076_);
if (v_isSharedCheck_2121_ == 0)
{
v___x_2116_ = v___x_2076_;
v_isShared_2117_ = v_isSharedCheck_2121_;
goto v_resetjp_2115_;
}
else
{
lean_inc(v_a_2114_);
lean_dec(v___x_2076_);
v___x_2116_ = lean_box(0);
v_isShared_2117_ = v_isSharedCheck_2121_;
goto v_resetjp_2115_;
}
v_resetjp_2115_:
{
lean_object* v___x_2119_; 
if (v_isShared_2117_ == 0)
{
v___x_2119_ = v___x_2116_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2120_; 
v_reuseFailAlloc_2120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2120_, 0, v_a_2114_);
v___x_2119_ = v_reuseFailAlloc_2120_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
return v___x_2119_;
}
}
}
}
}
else
{
lean_object* v_val_2122_; lean_object* v___x_2124_; uint8_t v_isShared_2125_; uint8_t v_isSharedCheck_2167_; 
lean_dec_ref(v___x_1981_);
v_val_2122_ = lean_ctor_get(v_fst_1980_, 0);
v_isSharedCheck_2167_ = !lean_is_exclusive(v_fst_1980_);
if (v_isSharedCheck_2167_ == 0)
{
v___x_2124_ = v_fst_1980_;
v_isShared_2125_ = v_isSharedCheck_2167_;
goto v_resetjp_2123_;
}
else
{
lean_inc(v_val_2122_);
lean_dec(v_fst_1980_);
v___x_2124_ = lean_box(0);
v_isShared_2125_ = v_isSharedCheck_2167_;
goto v_resetjp_2123_;
}
v_resetjp_2123_:
{
lean_object* v___x_2126_; 
v___x_2126_ = lp_batteries_Batteries_CodeAction_findTermInfo_x3f(v_node_1984_, v_val_2122_);
if (lean_obj_tag(v___x_2126_) == 1)
{
lean_object* v_val_2127_; lean_object* v___x_2129_; uint8_t v_isShared_2130_; uint8_t v_isSharedCheck_2162_; 
lean_del_object(v___x_2124_);
v_val_2127_ = lean_ctor_get(v___x_2126_, 0);
v_isSharedCheck_2162_ = !lean_is_exclusive(v___x_2126_);
if (v_isSharedCheck_2162_ == 0)
{
v___x_2129_ = v___x_2126_;
v_isShared_2130_ = v_isSharedCheck_2162_;
goto v_resetjp_2128_;
}
else
{
lean_inc(v_val_2127_);
lean_dec(v___x_2126_);
v___x_2129_ = lean_box(0);
v_isShared_2130_ = v_isSharedCheck_2162_;
goto v_resetjp_2128_;
}
v_resetjp_2128_:
{
lean_object* v_expr_2131_; lean_object* v___x_2132_; 
v_expr_2131_ = lean_ctor_get(v_val_2127_, 3);
lean_inc_ref(v_expr_2131_);
lean_dec(v_val_2127_);
lean_inc(v___y_1988_);
lean_inc_ref(v___y_1987_);
lean_inc(v___y_1986_);
lean_inc_ref(v___y_1985_);
v___x_2132_ = lean_infer_type(v_expr_2131_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
if (lean_obj_tag(v___x_2132_) == 0)
{
lean_object* v_a_2133_; lean_object* v___x_2134_; 
v_a_2133_ = lean_ctor_get(v___x_2132_, 0);
lean_inc(v_a_2133_);
lean_dec_ref_known(v___x_2132_, 1);
v___x_2134_ = lp_batteries_Batteries_CodeAction_getElimExprNames(v_a_2133_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
if (lean_obj_tag(v___x_2134_) == 0)
{
lean_object* v_a_2135_; lean_object* v___x_2137_; uint8_t v_isShared_2138_; uint8_t v_isSharedCheck_2145_; 
v_a_2135_ = lean_ctor_get(v___x_2134_, 0);
v_isSharedCheck_2145_ = !lean_is_exclusive(v___x_2134_);
if (v_isSharedCheck_2145_ == 0)
{
v___x_2137_ = v___x_2134_;
v_isShared_2138_ = v_isSharedCheck_2145_;
goto v_resetjp_2136_;
}
else
{
lean_inc(v_a_2135_);
lean_dec(v___x_2134_);
v___x_2137_ = lean_box(0);
v_isShared_2138_ = v_isSharedCheck_2145_;
goto v_resetjp_2136_;
}
v_resetjp_2136_:
{
lean_object* v___x_2140_; 
if (v_isShared_2130_ == 0)
{
lean_ctor_set(v___x_2129_, 0, v_a_2135_);
v___x_2140_ = v___x_2129_;
goto v_reusejp_2139_;
}
else
{
lean_object* v_reuseFailAlloc_2144_; 
v_reuseFailAlloc_2144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2144_, 0, v_a_2135_);
v___x_2140_ = v_reuseFailAlloc_2144_;
goto v_reusejp_2139_;
}
v_reusejp_2139_:
{
lean_object* v___x_2142_; 
if (v_isShared_2138_ == 0)
{
lean_ctor_set(v___x_2137_, 0, v___x_2140_);
v___x_2142_ = v___x_2137_;
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
}
else
{
lean_object* v_a_2146_; lean_object* v___x_2148_; uint8_t v_isShared_2149_; uint8_t v_isSharedCheck_2153_; 
lean_del_object(v___x_2129_);
v_a_2146_ = lean_ctor_get(v___x_2134_, 0);
v_isSharedCheck_2153_ = !lean_is_exclusive(v___x_2134_);
if (v_isSharedCheck_2153_ == 0)
{
v___x_2148_ = v___x_2134_;
v_isShared_2149_ = v_isSharedCheck_2153_;
goto v_resetjp_2147_;
}
else
{
lean_inc(v_a_2146_);
lean_dec(v___x_2134_);
v___x_2148_ = lean_box(0);
v_isShared_2149_ = v_isSharedCheck_2153_;
goto v_resetjp_2147_;
}
v_resetjp_2147_:
{
lean_object* v___x_2151_; 
if (v_isShared_2149_ == 0)
{
v___x_2151_ = v___x_2148_;
goto v_reusejp_2150_;
}
else
{
lean_object* v_reuseFailAlloc_2152_; 
v_reuseFailAlloc_2152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2152_, 0, v_a_2146_);
v___x_2151_ = v_reuseFailAlloc_2152_;
goto v_reusejp_2150_;
}
v_reusejp_2150_:
{
return v___x_2151_;
}
}
}
}
else
{
lean_object* v_a_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2161_; 
lean_del_object(v___x_2129_);
v_a_2154_ = lean_ctor_get(v___x_2132_, 0);
v_isSharedCheck_2161_ = !lean_is_exclusive(v___x_2132_);
if (v_isSharedCheck_2161_ == 0)
{
v___x_2156_ = v___x_2132_;
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_a_2154_);
lean_dec(v___x_2132_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
lean_object* v___x_2159_; 
if (v_isShared_2157_ == 0)
{
v___x_2159_ = v___x_2156_;
goto v_reusejp_2158_;
}
else
{
lean_object* v_reuseFailAlloc_2160_; 
v_reuseFailAlloc_2160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2160_, 0, v_a_2154_);
v___x_2159_ = v_reuseFailAlloc_2160_;
goto v_reusejp_2158_;
}
v_reusejp_2158_:
{
return v___x_2159_;
}
}
}
}
}
else
{
lean_object* v___x_2163_; lean_object* v___x_2165_; 
lean_dec(v___x_2126_);
v___x_2163_ = lean_box(0);
if (v_isShared_2125_ == 0)
{
lean_ctor_set_tag(v___x_2124_, 0);
lean_ctor_set(v___x_2124_, 0, v___x_2163_);
v___x_2165_ = v___x_2124_;
goto v_reusejp_2164_;
}
else
{
lean_object* v_reuseFailAlloc_2166_; 
v_reuseFailAlloc_2166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2166_, 0, v___x_2163_);
v___x_2165_ = v_reuseFailAlloc_2166_;
goto v_reusejp_2164_;
}
v_reusejp_2164_:
{
return v___x_2165_;
}
}
}
}
v___jp_1990_:
{
lean_object* v___x_1995_; lean_object* v___x_1996_; 
v___x_1995_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__1, &lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__1_once, _init_lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___closed__1);
v___x_1996_ = lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg(v___x_1995_, v___y_1991_, v___y_1992_, v___y_1993_, v___y_1994_);
return v___x_1996_;
}
v___jp_1997_:
{
lean_object* v___x_2003_; 
v___x_2003_ = lp_batteries_Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8(v___y_2002_, v___y_1999_, v___y_2001_, v___y_2000_, v___y_1998_);
if (lean_obj_tag(v___x_2003_) == 0)
{
lean_object* v_a_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; 
v_a_2004_ = lean_ctor_get(v___x_2003_, 0);
lean_inc(v_a_2004_);
lean_dec_ref_known(v___x_2003_, 1);
v___x_2005_ = l_Lean_ConstantInfo_type(v_a_2004_);
lean_dec(v_a_2004_);
v___x_2006_ = lp_batteries_Batteries_CodeAction_getElimExprNames(v___x_2005_, v___y_1999_, v___y_2001_, v___y_2000_, v___y_1998_);
if (lean_obj_tag(v___x_2006_) == 0)
{
lean_object* v_a_2007_; lean_object* v___x_2009_; uint8_t v_isShared_2010_; uint8_t v_isSharedCheck_2015_; 
v_a_2007_ = lean_ctor_get(v___x_2006_, 0);
v_isSharedCheck_2015_ = !lean_is_exclusive(v___x_2006_);
if (v_isSharedCheck_2015_ == 0)
{
v___x_2009_ = v___x_2006_;
v_isShared_2010_ = v_isSharedCheck_2015_;
goto v_resetjp_2008_;
}
else
{
lean_inc(v_a_2007_);
lean_dec(v___x_2006_);
v___x_2009_ = lean_box(0);
v_isShared_2010_ = v_isSharedCheck_2015_;
goto v_resetjp_2008_;
}
v_resetjp_2008_:
{
lean_object* v___x_2011_; lean_object* v___x_2013_; 
v___x_2011_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2011_, 0, v_a_2007_);
if (v_isShared_2010_ == 0)
{
lean_ctor_set(v___x_2009_, 0, v___x_2011_);
v___x_2013_ = v___x_2009_;
goto v_reusejp_2012_;
}
else
{
lean_object* v_reuseFailAlloc_2014_; 
v_reuseFailAlloc_2014_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2014_, 0, v___x_2011_);
v___x_2013_ = v_reuseFailAlloc_2014_;
goto v_reusejp_2012_;
}
v_reusejp_2012_:
{
return v___x_2013_;
}
}
}
else
{
lean_object* v_a_2016_; lean_object* v___x_2018_; uint8_t v_isShared_2019_; uint8_t v_isSharedCheck_2023_; 
v_a_2016_ = lean_ctor_get(v___x_2006_, 0);
v_isSharedCheck_2023_ = !lean_is_exclusive(v___x_2006_);
if (v_isSharedCheck_2023_ == 0)
{
v___x_2018_ = v___x_2006_;
v_isShared_2019_ = v_isSharedCheck_2023_;
goto v_resetjp_2017_;
}
else
{
lean_inc(v_a_2016_);
lean_dec(v___x_2006_);
v___x_2018_ = lean_box(0);
v_isShared_2019_ = v_isSharedCheck_2023_;
goto v_resetjp_2017_;
}
v_resetjp_2017_:
{
lean_object* v___x_2021_; 
if (v_isShared_2019_ == 0)
{
v___x_2021_ = v___x_2018_;
goto v_reusejp_2020_;
}
else
{
lean_object* v_reuseFailAlloc_2022_; 
v_reuseFailAlloc_2022_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2022_, 0, v_a_2016_);
v___x_2021_ = v_reuseFailAlloc_2022_;
goto v_reusejp_2020_;
}
v_reusejp_2020_:
{
return v___x_2021_;
}
}
}
}
else
{
lean_object* v_a_2024_; lean_object* v___x_2026_; uint8_t v_isShared_2027_; uint8_t v_isSharedCheck_2031_; 
v_a_2024_ = lean_ctor_get(v___x_2003_, 0);
v_isSharedCheck_2031_ = !lean_is_exclusive(v___x_2003_);
if (v_isSharedCheck_2031_ == 0)
{
v___x_2026_ = v___x_2003_;
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
else
{
lean_inc(v_a_2024_);
lean_dec(v___x_2003_);
v___x_2026_ = lean_box(0);
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
v_resetjp_2025_:
{
lean_object* v___x_2029_; 
if (v_isShared_2027_ == 0)
{
v___x_2029_ = v___x_2026_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2030_; 
v_reuseFailAlloc_2030_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2030_, 0, v_a_2024_);
v___x_2029_ = v_reuseFailAlloc_2030_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
return v___x_2029_;
}
}
}
}
v___jp_2032_:
{
lean_object* v_expr_2037_; lean_object* v___x_2038_; 
v_expr_2037_ = lean_ctor_get(v___x_1981_, 3);
lean_inc_ref(v_expr_2037_);
lean_dec_ref(v___x_1981_);
lean_inc(v___y_2036_);
lean_inc_ref(v___y_2035_);
lean_inc(v___y_2034_);
lean_inc_ref(v___y_2033_);
v___x_2038_ = lean_infer_type(v_expr_2037_, v___y_2033_, v___y_2034_, v___y_2035_, v___y_2036_);
if (lean_obj_tag(v___x_2038_) == 0)
{
lean_object* v_a_2039_; lean_object* v___x_2040_; 
v_a_2039_ = lean_ctor_get(v___x_2038_, 0);
lean_inc(v_a_2039_);
lean_dec_ref_known(v___x_2038_, 1);
lean_inc(v___y_2036_);
lean_inc_ref(v___y_2035_);
lean_inc(v___y_2034_);
lean_inc_ref(v___y_2033_);
v___x_2040_ = lean_whnf(v_a_2039_, v___y_2033_, v___y_2034_, v___y_2035_, v___y_2036_);
if (lean_obj_tag(v___x_2040_) == 0)
{
lean_object* v_a_2041_; lean_object* v___x_2042_; 
v_a_2041_ = lean_ctor_get(v___x_2040_, 0);
lean_inc(v_a_2041_);
lean_dec_ref_known(v___x_2040_, 1);
v___x_2042_ = l_Lean_Expr_getAppFn(v_a_2041_);
lean_dec(v_a_2041_);
if (lean_obj_tag(v___x_2042_) == 4)
{
lean_object* v_declName_2043_; lean_object* v___x_2044_; lean_object* v_env_2045_; uint8_t v___x_2046_; lean_object* v___x_2047_; 
v_declName_2043_ = lean_ctor_get(v___x_2042_, 0);
lean_inc(v_declName_2043_);
lean_dec_ref_known(v___x_2042_, 2);
v___x_2044_ = lean_st_ref_get(v___y_2036_);
v_env_2045_ = lean_ctor_get(v___x_2044_, 0);
lean_inc_ref(v_env_2045_);
lean_dec(v___x_2044_);
v___x_2046_ = 0;
v___x_2047_ = l_Lean_Environment_find_x3f(v_env_2045_, v_declName_2043_, v___x_2046_);
if (lean_obj_tag(v___x_2047_) == 0)
{
v___y_1991_ = v___y_2033_;
v___y_1992_ = v___y_2034_;
v___y_1993_ = v___y_2035_;
v___y_1994_ = v___y_2036_;
goto v___jp_1990_;
}
else
{
lean_object* v_val_2048_; 
v_val_2048_ = lean_ctor_get(v___x_2047_, 0);
lean_inc(v_val_2048_);
lean_dec_ref_known(v___x_2047_, 1);
if (lean_obj_tag(v_val_2048_) == 5)
{
if (v_fst_1982_ == 0)
{
lean_object* v_val_2049_; lean_object* v_toConstantVal_2050_; lean_object* v_name_2051_; lean_object* v___x_2052_; 
v_val_2049_ = lean_ctor_get(v_val_2048_, 0);
lean_inc_ref(v_val_2049_);
lean_dec_ref_known(v_val_2048_, 1);
v_toConstantVal_2050_ = lean_ctor_get(v_val_2049_, 0);
lean_inc_ref(v_toConstantVal_2050_);
lean_dec_ref(v_val_2049_);
v_name_2051_ = lean_ctor_get(v_toConstantVal_2050_, 0);
lean_inc(v_name_2051_);
lean_dec_ref(v_toConstantVal_2050_);
v___x_2052_ = l_Lean_mkCasesOnName(v_name_2051_);
v___y_1998_ = v___y_2036_;
v___y_1999_ = v___y_2033_;
v___y_2000_ = v___y_2035_;
v___y_2001_ = v___y_2034_;
v___y_2002_ = v___x_2052_;
goto v___jp_1997_;
}
else
{
lean_object* v_val_2053_; lean_object* v_toConstantVal_2054_; lean_object* v_name_2055_; lean_object* v___x_2056_; 
v_val_2053_ = lean_ctor_get(v_val_2048_, 0);
lean_inc_ref(v_val_2053_);
lean_dec_ref_known(v_val_2048_, 1);
v_toConstantVal_2054_ = lean_ctor_get(v_val_2053_, 0);
lean_inc_ref(v_toConstantVal_2054_);
lean_dec_ref(v_val_2053_);
v_name_2055_ = lean_ctor_get(v_toConstantVal_2054_, 0);
lean_inc(v_name_2055_);
lean_dec_ref(v_toConstantVal_2054_);
v___x_2056_ = l_Lean_mkRecName(v_name_2055_);
v___y_1998_ = v___y_2036_;
v___y_1999_ = v___y_2033_;
v___y_2000_ = v___y_2035_;
v___y_2001_ = v___y_2034_;
v___y_2002_ = v___x_2056_;
goto v___jp_1997_;
}
}
else
{
lean_dec(v_val_2048_);
v___y_1991_ = v___y_2033_;
v___y_1992_ = v___y_2034_;
v___y_1993_ = v___y_2035_;
v___y_1994_ = v___y_2036_;
goto v___jp_1990_;
}
}
}
else
{
lean_dec_ref(v___x_2042_);
v___y_1991_ = v___y_2033_;
v___y_1992_ = v___y_2034_;
v___y_1993_ = v___y_2035_;
v___y_1994_ = v___y_2036_;
goto v___jp_1990_;
}
}
else
{
lean_object* v_a_2057_; lean_object* v___x_2059_; uint8_t v_isShared_2060_; uint8_t v_isSharedCheck_2064_; 
v_a_2057_ = lean_ctor_get(v___x_2040_, 0);
v_isSharedCheck_2064_ = !lean_is_exclusive(v___x_2040_);
if (v_isSharedCheck_2064_ == 0)
{
v___x_2059_ = v___x_2040_;
v_isShared_2060_ = v_isSharedCheck_2064_;
goto v_resetjp_2058_;
}
else
{
lean_inc(v_a_2057_);
lean_dec(v___x_2040_);
v___x_2059_ = lean_box(0);
v_isShared_2060_ = v_isSharedCheck_2064_;
goto v_resetjp_2058_;
}
v_resetjp_2058_:
{
lean_object* v___x_2062_; 
if (v_isShared_2060_ == 0)
{
v___x_2062_ = v___x_2059_;
goto v_reusejp_2061_;
}
else
{
lean_object* v_reuseFailAlloc_2063_; 
v_reuseFailAlloc_2063_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2063_, 0, v_a_2057_);
v___x_2062_ = v_reuseFailAlloc_2063_;
goto v_reusejp_2061_;
}
v_reusejp_2061_:
{
return v___x_2062_;
}
}
}
}
else
{
lean_object* v_a_2065_; lean_object* v___x_2067_; uint8_t v_isShared_2068_; uint8_t v_isSharedCheck_2072_; 
v_a_2065_ = lean_ctor_get(v___x_2038_, 0);
v_isSharedCheck_2072_ = !lean_is_exclusive(v___x_2038_);
if (v_isSharedCheck_2072_ == 0)
{
v___x_2067_ = v___x_2038_;
v_isShared_2068_ = v_isSharedCheck_2072_;
goto v_resetjp_2066_;
}
else
{
lean_inc(v_a_2065_);
lean_dec(v___x_2038_);
v___x_2067_ = lean_box(0);
v_isShared_2068_ = v_isSharedCheck_2072_;
goto v_resetjp_2066_;
}
v_resetjp_2066_:
{
lean_object* v___x_2070_; 
if (v_isShared_2068_ == 0)
{
v___x_2070_ = v___x_2067_;
goto v_reusejp_2069_;
}
else
{
lean_object* v_reuseFailAlloc_2071_; 
v_reuseFailAlloc_2071_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2071_, 0, v_a_2065_);
v___x_2070_ = v_reuseFailAlloc_2071_;
goto v_reusejp_2069_;
}
v_reusejp_2069_:
{
return v___x_2070_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___boxed(lean_object* v_fst_2168_, lean_object* v___x_2169_, lean_object* v_fst_2170_, lean_object* v_targets_2171_, lean_object* v_node_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_){
_start:
{
uint8_t v_fst_38809__boxed_2178_; lean_object* v_res_2179_; 
v_fst_38809__boxed_2178_ = lean_unbox(v_fst_2170_);
v_res_2179_ = lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0(v_fst_2168_, v___x_2169_, v_fst_38809__boxed_2178_, v_targets_2171_, v_node_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_);
lean_dec(v___y_2176_);
lean_dec_ref(v___y_2175_);
lean_dec(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec_ref(v_targets_2171_);
return v_res_2179_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__0(lean_object* v_as_2180_, size_t v_sz_2181_, size_t v_i_2182_, lean_object* v_b_2183_){
_start:
{
lean_object* v___y_2185_; uint8_t v___x_2192_; 
v___x_2192_ = lean_usize_dec_lt(v_i_2182_, v_sz_2181_);
if (v___x_2192_ == 0)
{
return v_b_2183_;
}
else
{
lean_object* v_a_2193_; uint8_t v___x_2194_; 
v_a_2193_ = lean_array_uget_borrowed(v_as_2180_, v_i_2182_);
v___x_2194_ = l_Lean_Name_hasNum(v_a_2193_);
if (v___x_2194_ == 0)
{
uint8_t v___x_2195_; 
v___x_2195_ = l_Lean_Name_isInternal(v_a_2193_);
if (v___x_2195_ == 0)
{
lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; 
v___x_2196_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_instanceStub_spec__3___closed__2));
lean_inc(v_a_2193_);
v___x_2197_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_a_2193_, v___x_2192_);
v___x_2198_ = lean_string_append(v___x_2196_, v___x_2197_);
lean_dec_ref(v___x_2197_);
v___y_2185_ = v___x_2198_;
goto v___jp_2184_;
}
else
{
goto v___jp_2190_;
}
}
else
{
goto v___jp_2190_;
}
}
v___jp_2184_:
{
lean_object* v___x_2186_; size_t v___x_2187_; size_t v___x_2188_; 
v___x_2186_ = lean_string_append(v_b_2183_, v___y_2185_);
lean_dec_ref(v___y_2185_);
v___x_2187_ = ((size_t)1ULL);
v___x_2188_ = lean_usize_add(v_i_2182_, v___x_2187_);
v_i_2182_ = v___x_2188_;
v_b_2183_ = v___x_2186_;
goto _start;
}
v___jp_2190_:
{
lean_object* v___x_2191_; 
v___x_2191_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_eqnStub_spec__1___closed__0));
v___y_2185_ = v___x_2191_;
goto v___jp_2184_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__0___boxed(lean_object* v_as_2199_, lean_object* v_sz_2200_, lean_object* v_i_2201_, lean_object* v_b_2202_){
_start:
{
size_t v_sz_boxed_2203_; size_t v_i_boxed_2204_; lean_object* v_res_2205_; 
v_sz_boxed_2203_ = lean_unbox_usize(v_sz_2200_);
lean_dec(v_sz_2200_);
v_i_boxed_2204_ = lean_unbox_usize(v_i_2201_);
lean_dec(v_i_2201_);
v_res_2205_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__0(v_as_2199_, v_sz_boxed_2203_, v_i_boxed_2204_, v_b_2202_);
lean_dec_ref(v_as_2199_);
return v_res_2205_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3(void){
_start:
{
lean_object* v___x_2210_; lean_object* v___x_2211_; 
v___x_2210_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__2));
v___x_2211_ = lean_string_utf8_byte_size(v___x_2210_);
return v___x_2211_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1(lean_object* v___x_2212_, size_t v_sz_2213_, size_t v_i_2214_, lean_object* v_bs_2215_){
_start:
{
uint8_t v___x_2216_; 
v___x_2216_ = lean_usize_dec_lt(v_i_2214_, v_sz_2213_);
if (v___x_2216_ == 0)
{
return v_bs_2215_;
}
else
{
lean_object* v___x_2217_; uint8_t v___x_2218_; lean_object* v_v_2219_; lean_object* v_bs_x27_2220_; lean_object* v___y_2222_; uint8_t v___y_2228_; uint8_t v___x_2238_; 
v___x_2217_ = lean_unsigned_to_nat(0u);
v___x_2218_ = lean_nat_dec_eq(v___x_2212_, v___x_2217_);
v_v_2219_ = lean_array_uget(v_bs_2215_, v_i_2214_);
v_bs_x27_2220_ = lean_array_uset(v_bs_2215_, v_i_2214_, v___x_2217_);
v___x_2238_ = l_Lean_Name_hasMacroScopes(v_v_2219_);
if (v___x_2238_ == 0)
{
goto v___jp_2230_;
}
else
{
if (v___x_2218_ == 0)
{
v___y_2228_ = v___x_2218_;
goto v___jp_2227_;
}
else
{
goto v___jp_2230_;
}
}
v___jp_2221_:
{
size_t v___x_2223_; size_t v___x_2224_; lean_object* v___x_2225_; 
v___x_2223_ = ((size_t)1ULL);
v___x_2224_ = lean_usize_add(v_i_2214_, v___x_2223_);
v___x_2225_ = lean_array_uset(v_bs_x27_2220_, v_i_2214_, v___y_2222_);
v_i_2214_ = v___x_2224_;
v_bs_2215_ = v___x_2225_;
goto _start;
}
v___jp_2227_:
{
if (v___y_2228_ == 0)
{
v___y_2222_ = v_v_2219_;
goto v___jp_2221_;
}
else
{
lean_object* v___x_2229_; 
lean_dec(v_v_2219_);
v___x_2229_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__1));
v___y_2222_ = v___x_2229_;
goto v___jp_2221_;
}
}
v___jp_2230_:
{
lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; uint8_t v___x_2235_; 
v___x_2231_ = l_Lean_Name_getString_x21(v_v_2219_);
v___x_2232_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__2));
v___x_2233_ = lean_string_utf8_byte_size(v___x_2231_);
v___x_2234_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3);
v___x_2235_ = lean_nat_dec_le(v___x_2234_, v___x_2233_);
if (v___x_2235_ == 0)
{
lean_dec_ref(v___x_2231_);
v___y_2228_ = v___x_2218_;
goto v___jp_2227_;
}
else
{
lean_object* v___x_2236_; uint8_t v___x_2237_; 
v___x_2236_ = lean_nat_sub(v___x_2233_, v___x_2234_);
v___x_2237_ = lean_string_memcmp(v___x_2231_, v___x_2232_, v___x_2236_, v___x_2217_, v___x_2234_);
lean_dec(v___x_2236_);
lean_dec_ref(v___x_2231_);
v___y_2228_ = v___x_2237_;
goto v___jp_2227_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___boxed(lean_object* v___x_2239_, lean_object* v_sz_2240_, lean_object* v_i_2241_, lean_object* v_bs_2242_){
_start:
{
size_t v_sz_boxed_2243_; size_t v_i_boxed_2244_; lean_object* v_res_2245_; 
v_sz_boxed_2243_ = lean_unbox_usize(v_sz_2240_);
lean_dec(v_sz_2240_);
v_i_boxed_2244_ = lean_unbox_usize(v_i_2241_);
lean_dec(v_i_2241_);
v_res_2245_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1(v___x_2239_, v_sz_boxed_2243_, v_i_boxed_2244_, v_bs_2242_);
lean_dec(v___x_2239_);
return v_res_2245_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2(lean_object* v___x_2246_, lean_object* v_as_2247_, size_t v_i_2248_, size_t v_stop_2249_, lean_object* v_b_2250_){
_start:
{
lean_object* v___y_2252_; uint8_t v___y_2257_; uint8_t v___x_2260_; 
v___x_2260_ = lean_usize_dec_eq(v_i_2248_, v_stop_2249_);
if (v___x_2260_ == 0)
{
lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; uint8_t v___x_2268_; 
v___x_2261_ = lean_unsigned_to_nat(0u);
v___x_2262_ = lean_array_uget_borrowed(v_as_2247_, v_i_2248_);
v___x_2263_ = l_Lean_Name_eraseMacroScopes(v___x_2262_);
v___x_2264_ = l_Lean_Name_getString_x21(v___x_2263_);
lean_dec(v___x_2263_);
v___x_2265_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__2));
v___x_2266_ = lean_string_utf8_byte_size(v___x_2264_);
v___x_2267_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1___closed__3);
v___x_2268_ = lean_nat_dec_le(v___x_2267_, v___x_2266_);
if (v___x_2268_ == 0)
{
uint8_t v___x_2269_; 
lean_dec_ref(v___x_2264_);
v___x_2269_ = lean_nat_dec_eq(v___x_2246_, v___x_2261_);
v___y_2257_ = v___x_2269_;
goto v___jp_2256_;
}
else
{
lean_object* v___x_2270_; uint8_t v___x_2271_; 
v___x_2270_ = lean_nat_sub(v___x_2266_, v___x_2267_);
v___x_2271_ = lean_string_memcmp(v___x_2264_, v___x_2265_, v___x_2270_, v___x_2261_, v___x_2267_);
lean_dec(v___x_2270_);
lean_dec_ref(v___x_2264_);
v___y_2257_ = v___x_2271_;
goto v___jp_2256_;
}
}
else
{
return v_b_2250_;
}
v___jp_2251_:
{
size_t v___x_2253_; size_t v___x_2254_; 
v___x_2253_ = ((size_t)1ULL);
v___x_2254_ = lean_usize_add(v_i_2248_, v___x_2253_);
v_i_2248_ = v___x_2254_;
v_b_2250_ = v___y_2252_;
goto _start;
}
v___jp_2256_:
{
if (v___y_2257_ == 0)
{
v___y_2252_ = v_b_2250_;
goto v___jp_2251_;
}
else
{
lean_object* v___x_2258_; lean_object* v___x_2259_; 
v___x_2258_ = lean_unsigned_to_nat(1u);
v___x_2259_ = lean_nat_add(v_b_2250_, v___x_2258_);
lean_dec(v_b_2250_);
v___y_2252_ = v___x_2259_;
goto v___jp_2251_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2___boxed(lean_object* v___x_2272_, lean_object* v_as_2273_, lean_object* v_i_2274_, lean_object* v_stop_2275_, lean_object* v_b_2276_){
_start:
{
size_t v_i_boxed_2277_; size_t v_stop_boxed_2278_; lean_object* v_res_2279_; 
v_i_boxed_2277_ = lean_unbox_usize(v_i_2274_);
lean_dec(v_i_2274_);
v_stop_boxed_2278_ = lean_unbox_usize(v_stop_2275_);
lean_dec(v_stop_2275_);
v_res_2279_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2(v___x_2272_, v_as_2273_, v_i_boxed_2277_, v_stop_boxed_2278_, v_b_2276_);
lean_dec_ref(v_as_2273_);
lean_dec(v___x_2272_);
return v_res_2279_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__1(void){
_start:
{
uint32_t v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; 
v___x_2281_ = l_Lean_idBeginEscape;
v___x_2282_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10));
v___x_2283_ = lean_string_push(v___x_2282_, v___x_2281_);
return v___x_2283_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__2(void){
_start:
{
uint32_t v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; 
v___x_2284_ = l_Lean_idEndEscape;
v___x_2285_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10));
v___x_2286_ = lean_string_push(v___x_2285_, v___x_2284_);
return v___x_2286_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3(lean_object* v___x_2287_, lean_object* v___y_2288_, lean_object* v___x_2289_, uint8_t v_fst_2290_, lean_object* v_snap_2291_, lean_object* v_as_2292_, size_t v_sz_2293_, size_t v_i_2294_, lean_object* v_b_2295_){
_start:
{
lean_object* v___y_2297_; lean_object* v___y_2298_; uint8_t v___x_2308_; 
v___x_2308_ = lean_usize_dec_lt(v_i_2294_, v_sz_2293_);
if (v___x_2308_ == 0)
{
return v_b_2295_;
}
else
{
lean_object* v_a_2309_; lean_object* v_fst_2310_; lean_object* v_snd_2311_; lean_object* v___y_2313_; uint8_t v___y_2314_; lean_object* v___y_2319_; lean_object* v___y_2320_; lean_object* v___x_2323_; uint8_t v___x_2324_; lean_object* v_ctor_2326_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; 
v_a_2309_ = lean_array_uget_borrowed(v_as_2292_, v_i_2294_);
v_fst_2310_ = lean_ctor_get(v_a_2309_, 0);
v_snd_2311_ = lean_ctor_get(v_a_2309_, 1);
v___x_2323_ = lean_unsigned_to_nat(0u);
v___x_2324_ = lean_nat_dec_eq(v___x_2289_, v___x_2323_);
lean_inc(v_fst_2310_);
v___x_2340_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_2310_, v___x_2308_);
v___x_2341_ = l_Lean_Server_Snapshots_Snapshot_env(v_snap_2291_);
v___x_2342_ = l_Lean_Parser_getTokenTable(v___x_2341_);
v___x_2343_ = l_Lean_Data_Trie_find_x3f___redArg(v___x_2342_, v___x_2340_);
lean_dec_ref(v___x_2342_);
if (lean_obj_tag(v___x_2343_) == 1)
{
lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; 
lean_dec_ref_known(v___x_2343_, 1);
v___x_2344_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__1);
v___x_2345_ = lean_string_append(v___x_2344_, v___x_2340_);
lean_dec_ref(v___x_2340_);
v___x_2346_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__2, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__2_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__2);
v___x_2347_ = lean_string_append(v___x_2345_, v___x_2346_);
v_ctor_2326_ = v___x_2347_;
goto v___jp_2325_;
}
else
{
lean_dec(v___x_2343_);
v_ctor_2326_ = v___x_2340_;
goto v___jp_2325_;
}
v___jp_2312_:
{
if (v___y_2314_ == 0)
{
lean_inc(v_snd_2311_);
v___y_2297_ = v___y_2313_;
v___y_2298_ = v_snd_2311_;
goto v___jp_2296_;
}
else
{
size_t v_sz_2315_; size_t v___x_2316_; lean_object* v___x_2317_; 
v_sz_2315_ = lean_array_size(v_snd_2311_);
v___x_2316_ = ((size_t)0ULL);
lean_inc(v_snd_2311_);
v___x_2317_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__1(v___x_2289_, v_sz_2315_, v___x_2316_, v_snd_2311_);
v___y_2297_ = v___y_2313_;
v___y_2298_ = v___x_2317_;
goto v___jp_2296_;
}
}
v___jp_2318_:
{
lean_object* v___x_2321_; uint8_t v___x_2322_; 
v___x_2321_ = lean_unsigned_to_nat(1u);
v___x_2322_ = lean_nat_dec_eq(v___y_2320_, v___x_2321_);
lean_dec(v___y_2320_);
v___y_2313_ = v___y_2319_;
v___y_2314_ = v___x_2322_;
goto v___jp_2312_;
}
v___jp_2325_:
{
lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; 
v___x_2327_ = lean_string_append(v_b_2295_, v___x_2287_);
v___x_2328_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___closed__0));
v___x_2329_ = lean_string_append(v___x_2328_, v_ctor_2326_);
lean_dec_ref(v_ctor_2326_);
v___x_2330_ = lean_string_append(v___x_2327_, v___x_2329_);
lean_dec_ref(v___x_2329_);
if (v_fst_2290_ == 0)
{
v___y_2313_ = v___x_2330_;
v___y_2314_ = v___x_2324_;
goto v___jp_2312_;
}
else
{
lean_object* v___x_2331_; uint8_t v___x_2332_; 
v___x_2331_ = lean_array_get_size(v_snd_2311_);
v___x_2332_ = lean_nat_dec_lt(v___x_2323_, v___x_2331_);
if (v___x_2332_ == 0)
{
v___y_2319_ = v___x_2330_;
v___y_2320_ = v___x_2323_;
goto v___jp_2318_;
}
else
{
uint8_t v___x_2333_; 
v___x_2333_ = lean_nat_dec_le(v___x_2331_, v___x_2331_);
if (v___x_2333_ == 0)
{
if (v___x_2332_ == 0)
{
v___y_2319_ = v___x_2330_;
v___y_2320_ = v___x_2323_;
goto v___jp_2318_;
}
else
{
size_t v___x_2334_; size_t v___x_2335_; lean_object* v___x_2336_; 
v___x_2334_ = ((size_t)0ULL);
v___x_2335_ = lean_usize_of_nat(v___x_2331_);
v___x_2336_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2(v___x_2289_, v_snd_2311_, v___x_2334_, v___x_2335_, v___x_2323_);
v___y_2319_ = v___x_2330_;
v___y_2320_ = v___x_2336_;
goto v___jp_2318_;
}
}
else
{
size_t v___x_2337_; size_t v___x_2338_; lean_object* v___x_2339_; 
v___x_2337_ = ((size_t)0ULL);
v___x_2338_ = lean_usize_of_nat(v___x_2331_);
v___x_2339_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__2(v___x_2289_, v_snd_2311_, v___x_2337_, v___x_2338_, v___x_2323_);
v___y_2319_ = v___x_2330_;
v___y_2320_ = v___x_2339_;
goto v___jp_2318_;
}
}
}
}
}
v___jp_2296_:
{
size_t v_sz_2299_; size_t v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; size_t v___x_2305_; size_t v___x_2306_; 
v_sz_2299_ = lean_array_size(v___y_2298_);
v___x_2300_ = ((size_t)0ULL);
v___x_2301_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__0(v___y_2298_, v_sz_2299_, v___x_2300_, v___y_2297_);
lean_dec_ref(v___y_2298_);
v___x_2302_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_eqnStub_spec__2___redArg___closed__6));
v___x_2303_ = lean_string_append(v___x_2301_, v___x_2302_);
v___x_2304_ = lean_string_append(v___x_2303_, v___y_2288_);
v___x_2305_ = ((size_t)1ULL);
v___x_2306_ = lean_usize_add(v_i_2294_, v___x_2305_);
v_i_2294_ = v___x_2306_;
v_b_2295_ = v___x_2304_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3___boxed(lean_object* v___x_2348_, lean_object* v___y_2349_, lean_object* v___x_2350_, lean_object* v_fst_2351_, lean_object* v_snap_2352_, lean_object* v_as_2353_, lean_object* v_sz_2354_, lean_object* v_i_2355_, lean_object* v_b_2356_){
_start:
{
uint8_t v_fst_39335__boxed_2357_; size_t v_sz_boxed_2358_; size_t v_i_boxed_2359_; lean_object* v_res_2360_; 
v_fst_39335__boxed_2357_ = lean_unbox(v_fst_2351_);
v_sz_boxed_2358_ = lean_unbox_usize(v_sz_2354_);
lean_dec(v_sz_2354_);
v_i_boxed_2359_ = lean_unbox_usize(v_i_2355_);
lean_dec(v_i_2355_);
v_res_2360_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3(v___x_2348_, v___y_2349_, v___x_2350_, v_fst_39335__boxed_2357_, v_snap_2352_, v_as_2353_, v_sz_boxed_2358_, v_i_boxed_2359_, v_b_2356_);
lean_dec_ref(v_as_2353_);
lean_dec_ref(v_snap_2352_);
lean_dec(v___x_2350_);
lean_dec_ref(v___y_2349_);
lean_dec_ref(v___x_2348_);
return v_res_2360_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__1(lean_object* v___y_2361_, lean_object* v_ctors_2362_, lean_object* v___x_2363_, lean_object* v___x_2364_, uint8_t v_fst_2365_, lean_object* v_snap_2366_, lean_object* v_a_2367_, lean_object* v___x_2368_, lean_object* v___x_2369_, lean_object* v___x_2370_, lean_object* v___x_2371_, lean_object* v___x_2372_, lean_object* v___x_2373_, lean_object* v___x_2374_, lean_object* v___x_2375_, lean_object* v___x_2376_, lean_object* v_fallback_2377_, lean_object* v_source_2378_){
_start:
{
lean_object* v_fst_2380_; lean_object* v_snd_2381_; lean_object* v___x_2383_; uint8_t v_isShared_2384_; uint8_t v_isSharedCheck_2405_; 
v_fst_2380_ = lean_ctor_get(v___y_2361_, 0);
v_snd_2381_ = lean_ctor_get(v___y_2361_, 1);
v_isSharedCheck_2405_ = !lean_is_exclusive(v___y_2361_);
if (v_isSharedCheck_2405_ == 0)
{
v___x_2383_ = v___y_2361_;
v_isShared_2384_ = v_isSharedCheck_2405_;
goto v_resetjp_2382_;
}
else
{
lean_inc(v_snd_2381_);
lean_inc(v_fst_2380_);
lean_dec(v___y_2361_);
v___x_2383_ = lean_box(0);
v_isShared_2384_ = v_isSharedCheck_2405_;
goto v_resetjp_2382_;
}
v_resetjp_2382_:
{
lean_object* v___y_2386_; 
if (lean_obj_tag(v_fallback_2377_) == 1)
{
lean_object* v_val_2400_; lean_object* v_start_2401_; lean_object* v_stop_2402_; lean_object* v___x_2403_; 
v_val_2400_ = lean_ctor_get(v_fallback_2377_, 0);
v_start_2401_ = lean_ctor_get(v_val_2400_, 0);
v_stop_2402_ = lean_ctor_get(v_val_2400_, 1);
v___x_2403_ = lean_string_utf8_extract(v_source_2378_, v_start_2401_, v_stop_2402_);
v___y_2386_ = v___x_2403_;
goto v___jp_2385_;
}
else
{
lean_object* v___x_2404_; 
v___x_2404_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_holeKindToHoleString___closed__6));
v___y_2386_ = v___x_2404_;
goto v___jp_2385_;
}
v___jp_2385_:
{
size_t v_sz_2387_; size_t v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2392_; 
v_sz_2387_ = lean_array_size(v_ctors_2362_);
v___x_2388_ = ((size_t)0ULL);
v___x_2389_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__3(v___x_2363_, v___y_2386_, v___x_2364_, v_fst_2365_, v_snap_2366_, v_ctors_2362_, v_sz_2387_, v___x_2388_, v_snd_2381_);
lean_dec_ref(v___y_2386_);
v___x_2390_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_2367_);
if (v_isShared_2384_ == 0)
{
lean_ctor_set(v___x_2383_, 1, v___x_2368_);
v___x_2392_ = v___x_2383_;
goto v_reusejp_2391_;
}
else
{
lean_object* v_reuseFailAlloc_2399_; 
v_reuseFailAlloc_2399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2399_, 0, v_fst_2380_);
lean_ctor_set(v_reuseFailAlloc_2399_, 1, v___x_2368_);
v___x_2392_ = v_reuseFailAlloc_2399_;
goto v_reusejp_2391_;
}
v_reusejp_2391_:
{
lean_object* v___x_2393_; lean_object* v___x_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v___x_2398_; 
v___x_2393_ = lean_box(0);
lean_inc_n(v___x_2369_, 2);
v___x_2394_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2394_, 0, v___x_2392_);
lean_ctor_set(v___x_2394_, 1, v___x_2389_);
lean_ctor_set(v___x_2394_, 2, v___x_2393_);
lean_ctor_set(v___x_2394_, 3, v___x_2369_);
v___x_2395_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___x_2390_, v___x_2394_);
v___x_2396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2396_, 0, v___x_2395_);
v___x_2397_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2397_, 0, v___x_2369_);
lean_ctor_set(v___x_2397_, 1, v___x_2369_);
lean_ctor_set(v___x_2397_, 2, v___x_2370_);
lean_ctor_set(v___x_2397_, 3, v___x_2371_);
lean_ctor_set(v___x_2397_, 4, v___x_2372_);
lean_ctor_set(v___x_2397_, 5, v___x_2373_);
lean_ctor_set(v___x_2397_, 6, v___x_2374_);
lean_ctor_set(v___x_2397_, 7, v___x_2396_);
lean_ctor_set(v___x_2397_, 8, v___x_2375_);
lean_ctor_set(v___x_2397_, 9, v___x_2376_);
v___x_2398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2398_, 0, v___x_2397_);
return v___x_2398_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__1___boxed(lean_object** _args){
lean_object* v___y_2406_ = _args[0];
lean_object* v_ctors_2407_ = _args[1];
lean_object* v___x_2408_ = _args[2];
lean_object* v___x_2409_ = _args[3];
lean_object* v_fst_2410_ = _args[4];
lean_object* v_snap_2411_ = _args[5];
lean_object* v_a_2412_ = _args[6];
lean_object* v___x_2413_ = _args[7];
lean_object* v___x_2414_ = _args[8];
lean_object* v___x_2415_ = _args[9];
lean_object* v___x_2416_ = _args[10];
lean_object* v___x_2417_ = _args[11];
lean_object* v___x_2418_ = _args[12];
lean_object* v___x_2419_ = _args[13];
lean_object* v___x_2420_ = _args[14];
lean_object* v___x_2421_ = _args[15];
lean_object* v_fallback_2422_ = _args[16];
lean_object* v_source_2423_ = _args[17];
lean_object* v___y_2424_ = _args[18];
_start:
{
uint8_t v_fst_39452__boxed_2425_; lean_object* v_res_2426_; 
v_fst_39452__boxed_2425_ = lean_unbox(v_fst_2410_);
v_res_2426_ = lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__1(v___y_2406_, v_ctors_2407_, v___x_2408_, v___x_2409_, v_fst_39452__boxed_2425_, v_snap_2411_, v_a_2412_, v___x_2413_, v___x_2414_, v___x_2415_, v___x_2416_, v___x_2417_, v___x_2418_, v___x_2419_, v___x_2420_, v___x_2421_, v_fallback_2422_, v_source_2423_);
lean_dec_ref(v_source_2423_);
lean_dec(v_fallback_2422_);
lean_dec_ref(v_snap_2411_);
lean_dec(v___x_2409_);
lean_dec_ref(v___x_2408_);
lean_dec_ref(v_ctors_2407_);
return v_res_2426_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__6(size_t v_sz_2427_, size_t v_i_2428_, lean_object* v_bs_2429_){
_start:
{
uint8_t v___x_2430_; 
v___x_2430_ = lean_usize_dec_lt(v_i_2428_, v_sz_2427_);
if (v___x_2430_ == 0)
{
return v_bs_2429_;
}
else
{
lean_object* v_v_2431_; lean_object* v_expr_2432_; lean_object* v___x_2433_; lean_object* v_bs_x27_2434_; size_t v___x_2435_; size_t v___x_2436_; lean_object* v___x_2437_; 
v_v_2431_ = lean_array_uget_borrowed(v_bs_2429_, v_i_2428_);
v_expr_2432_ = lean_ctor_get(v_v_2431_, 3);
lean_inc_ref(v_expr_2432_);
v___x_2433_ = lean_unsigned_to_nat(0u);
v_bs_x27_2434_ = lean_array_uset(v_bs_2429_, v_i_2428_, v___x_2433_);
v___x_2435_ = ((size_t)1ULL);
v___x_2436_ = lean_usize_add(v_i_2428_, v___x_2435_);
v___x_2437_ = lean_array_uset(v_bs_x27_2434_, v_i_2428_, v_expr_2432_);
v_i_2428_ = v___x_2436_;
v_bs_2429_ = v___x_2437_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__6___boxed(lean_object* v_sz_2439_, lean_object* v_i_2440_, lean_object* v_bs_2441_){
_start:
{
size_t v_sz_boxed_2442_; size_t v_i_boxed_2443_; lean_object* v_res_2444_; 
v_sz_boxed_2442_ = lean_unbox_usize(v_sz_2439_);
lean_dec(v_sz_2439_);
v_i_boxed_2443_ = lean_unbox_usize(v_i_2440_);
lean_dec(v_i_2440_);
v_res_2444_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__6(v_sz_boxed_2442_, v_i_boxed_2443_, v_bs_2441_);
return v_res_2444_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14(uint8_t v___x_2445_, lean_object* v_as_2446_, size_t v_i_2447_, size_t v_stop_2448_, lean_object* v_b_2449_){
_start:
{
lean_object* v___y_2451_; uint8_t v___x_2455_; 
v___x_2455_ = lean_usize_dec_eq(v_i_2447_, v_stop_2448_);
if (v___x_2455_ == 0)
{
lean_object* v_fst_2456_; uint8_t v___x_2457_; 
v_fst_2456_ = lean_ctor_get(v_b_2449_, 0);
v___x_2457_ = lean_unbox(v_fst_2456_);
if (v___x_2457_ == 0)
{
lean_object* v_snd_2458_; lean_object* v___x_2460_; uint8_t v_isShared_2461_; uint8_t v_isSharedCheck_2466_; 
v_snd_2458_ = lean_ctor_get(v_b_2449_, 1);
v_isSharedCheck_2466_ = !lean_is_exclusive(v_b_2449_);
if (v_isSharedCheck_2466_ == 0)
{
lean_object* v_unused_2467_; 
v_unused_2467_ = lean_ctor_get(v_b_2449_, 0);
lean_dec(v_unused_2467_);
v___x_2460_ = v_b_2449_;
v_isShared_2461_ = v_isSharedCheck_2466_;
goto v_resetjp_2459_;
}
else
{
lean_inc(v_snd_2458_);
lean_dec(v_b_2449_);
v___x_2460_ = lean_box(0);
v_isShared_2461_ = v_isSharedCheck_2466_;
goto v_resetjp_2459_;
}
v_resetjp_2459_:
{
lean_object* v___x_2462_; lean_object* v___x_2464_; 
v___x_2462_ = lean_box(v___x_2445_);
if (v_isShared_2461_ == 0)
{
lean_ctor_set(v___x_2460_, 0, v___x_2462_);
v___x_2464_ = v___x_2460_;
goto v_reusejp_2463_;
}
else
{
lean_object* v_reuseFailAlloc_2465_; 
v_reuseFailAlloc_2465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2465_, 0, v___x_2462_);
lean_ctor_set(v_reuseFailAlloc_2465_, 1, v_snd_2458_);
v___x_2464_ = v_reuseFailAlloc_2465_;
goto v_reusejp_2463_;
}
v_reusejp_2463_:
{
v___y_2451_ = v___x_2464_;
goto v___jp_2450_;
}
}
}
else
{
lean_object* v_snd_2468_; lean_object* v___x_2470_; uint8_t v_isShared_2471_; uint8_t v_isSharedCheck_2478_; 
v_snd_2468_ = lean_ctor_get(v_b_2449_, 1);
v_isSharedCheck_2478_ = !lean_is_exclusive(v_b_2449_);
if (v_isSharedCheck_2478_ == 0)
{
lean_object* v_unused_2479_; 
v_unused_2479_ = lean_ctor_get(v_b_2449_, 0);
lean_dec(v_unused_2479_);
v___x_2470_ = v_b_2449_;
v_isShared_2471_ = v_isSharedCheck_2478_;
goto v_resetjp_2469_;
}
else
{
lean_inc(v_snd_2468_);
lean_dec(v_b_2449_);
v___x_2470_ = lean_box(0);
v_isShared_2471_ = v_isSharedCheck_2478_;
goto v_resetjp_2469_;
}
v_resetjp_2469_:
{
lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2476_; 
v___x_2472_ = lean_array_uget_borrowed(v_as_2446_, v_i_2447_);
lean_inc(v___x_2472_);
v___x_2473_ = lean_array_push(v_snd_2468_, v___x_2472_);
v___x_2474_ = lean_box(v___x_2455_);
if (v_isShared_2471_ == 0)
{
lean_ctor_set(v___x_2470_, 1, v___x_2473_);
lean_ctor_set(v___x_2470_, 0, v___x_2474_);
v___x_2476_ = v___x_2470_;
goto v_reusejp_2475_;
}
else
{
lean_object* v_reuseFailAlloc_2477_; 
v_reuseFailAlloc_2477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2477_, 0, v___x_2474_);
lean_ctor_set(v_reuseFailAlloc_2477_, 1, v___x_2473_);
v___x_2476_ = v_reuseFailAlloc_2477_;
goto v_reusejp_2475_;
}
v_reusejp_2475_:
{
v___y_2451_ = v___x_2476_;
goto v___jp_2450_;
}
}
}
}
else
{
return v_b_2449_;
}
v___jp_2450_:
{
size_t v___x_2452_; size_t v___x_2453_; 
v___x_2452_ = ((size_t)1ULL);
v___x_2453_ = lean_usize_add(v_i_2447_, v___x_2452_);
v_i_2447_ = v___x_2453_;
v_b_2449_ = v___y_2451_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14___boxed(lean_object* v___x_2480_, lean_object* v_as_2481_, lean_object* v_i_2482_, lean_object* v_stop_2483_, lean_object* v_b_2484_){
_start:
{
uint8_t v___x_39556__boxed_2485_; size_t v_i_boxed_2486_; size_t v_stop_boxed_2487_; lean_object* v_res_2488_; 
v___x_39556__boxed_2485_ = lean_unbox(v___x_2480_);
v_i_boxed_2486_ = lean_unbox_usize(v_i_2482_);
lean_dec(v_i_2482_);
v_stop_boxed_2487_ = lean_unbox_usize(v_stop_2483_);
lean_dec(v_stop_2483_);
v_res_2488_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14(v___x_39556__boxed_2485_, v_as_2481_, v_i_boxed_2486_, v_stop_boxed_2487_, v_b_2484_);
lean_dec_ref(v_as_2481_);
return v_res_2488_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__5(lean_object* v_node_2489_, size_t v_sz_2490_, size_t v_i_2491_, lean_object* v_bs_2492_){
_start:
{
uint8_t v___x_2493_; 
v___x_2493_ = lean_usize_dec_lt(v_i_2491_, v_sz_2490_);
if (v___x_2493_ == 0)
{
lean_object* v___x_2494_; 
lean_dec_ref(v_node_2489_);
v___x_2494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2494_, 0, v_bs_2492_);
return v___x_2494_;
}
else
{
lean_object* v_v_2495_; lean_object* v___x_2496_; 
v_v_2495_ = lean_array_uget_borrowed(v_bs_2492_, v_i_2491_);
lean_inc(v_v_2495_);
lean_inc_ref(v_node_2489_);
v___x_2496_ = lp_batteries_Batteries_CodeAction_findTermInfo_x3f(v_node_2489_, v_v_2495_);
if (lean_obj_tag(v___x_2496_) == 0)
{
lean_object* v___x_2497_; 
lean_dec_ref(v_bs_2492_);
lean_dec_ref(v_node_2489_);
v___x_2497_ = lean_box(0);
return v___x_2497_;
}
else
{
lean_object* v_val_2498_; lean_object* v___x_2499_; lean_object* v_bs_x27_2500_; size_t v___x_2501_; size_t v___x_2502_; lean_object* v___x_2503_; 
v_val_2498_ = lean_ctor_get(v___x_2496_, 0);
lean_inc(v_val_2498_);
lean_dec_ref_known(v___x_2496_, 1);
v___x_2499_ = lean_unsigned_to_nat(0u);
v_bs_x27_2500_ = lean_array_uset(v_bs_2492_, v_i_2491_, v___x_2499_);
v___x_2501_ = ((size_t)1ULL);
v___x_2502_ = lean_usize_add(v_i_2491_, v___x_2501_);
v___x_2503_ = lean_array_uset(v_bs_x27_2500_, v_i_2491_, v_val_2498_);
v_i_2491_ = v___x_2502_;
v_bs_2492_ = v___x_2503_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__5___boxed(lean_object* v_node_2505_, lean_object* v_sz_2506_, lean_object* v_i_2507_, lean_object* v_bs_2508_){
_start:
{
size_t v_sz_boxed_2509_; size_t v_i_boxed_2510_; lean_object* v_res_2511_; 
v_sz_boxed_2509_ = lean_unbox_usize(v_sz_2506_);
lean_dec(v_sz_2506_);
v_i_boxed_2510_ = lean_unbox_usize(v_i_2507_);
lean_dec(v_i_2507_);
v_res_2511_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__5(v_node_2505_, v_sz_boxed_2509_, v_i_boxed_2510_, v_bs_2508_);
return v_res_2511_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10(lean_object* v___x_2512_, uint8_t v___x_2513_, lean_object* v_as_2514_, size_t v_i_2515_, size_t v_stop_2516_, lean_object* v_b_2517_){
_start:
{
lean_object* v___y_2519_; uint8_t v___x_2523_; 
v___x_2523_ = lean_usize_dec_eq(v_i_2515_, v_stop_2516_);
if (v___x_2523_ == 0)
{
lean_object* v___x_2524_; lean_object* v_fst_2525_; lean_object* v___x_2526_; uint8_t v___x_2527_; 
v___x_2524_ = lean_array_uget_borrowed(v_as_2514_, v_i_2515_);
v_fst_2525_ = lean_ctor_get(v___x_2524_, 0);
v___x_2526_ = l_Lean_TSyntax_getId(v___x_2512_);
v___x_2527_ = lean_name_eq(v_fst_2525_, v___x_2526_);
lean_dec(v___x_2526_);
if (v___x_2527_ == 0)
{
if (v___x_2513_ == 0)
{
v___y_2519_ = v_b_2517_;
goto v___jp_2518_;
}
else
{
lean_object* v___x_2528_; 
lean_inc(v___x_2524_);
v___x_2528_ = lean_array_push(v_b_2517_, v___x_2524_);
v___y_2519_ = v___x_2528_;
goto v___jp_2518_;
}
}
else
{
v___y_2519_ = v_b_2517_;
goto v___jp_2518_;
}
}
else
{
return v_b_2517_;
}
v___jp_2518_:
{
size_t v___x_2520_; size_t v___x_2521_; 
v___x_2520_ = ((size_t)1ULL);
v___x_2521_ = lean_usize_add(v_i_2515_, v___x_2520_);
v_i_2515_ = v___x_2521_;
v_b_2517_ = v___y_2519_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10___boxed(lean_object* v___x_2529_, lean_object* v___x_2530_, lean_object* v_as_2531_, lean_object* v_i_2532_, lean_object* v_stop_2533_, lean_object* v_b_2534_){
_start:
{
uint8_t v___x_39640__boxed_2535_; size_t v_i_boxed_2536_; size_t v_stop_boxed_2537_; lean_object* v_res_2538_; 
v___x_39640__boxed_2535_ = lean_unbox(v___x_2530_);
v_i_boxed_2536_ = lean_unbox_usize(v_i_2532_);
lean_dec(v_i_2532_);
v_stop_boxed_2537_ = lean_unbox_usize(v_stop_2533_);
lean_dec(v_stop_2533_);
v_res_2538_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10(v___x_2529_, v___x_39640__boxed_2535_, v_as_2531_, v_i_boxed_2536_, v_stop_boxed_2537_, v_b_2534_);
lean_dec_ref(v_as_2531_);
lean_dec(v___x_2529_);
return v_res_2538_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg(lean_object* v_as_2564_, size_t v_sz_2565_, size_t v_i_2566_, lean_object* v_b_2567_){
_start:
{
lean_object* v_a_2570_; uint8_t v___x_2574_; 
v___x_2574_ = lean_usize_dec_lt(v_i_2566_, v_sz_2565_);
if (v___x_2574_ == 0)
{
lean_object* v___x_2575_; 
v___x_2575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2575_, 0, v_b_2567_);
return v___x_2575_;
}
else
{
lean_object* v_fst_2576_; lean_object* v_snd_2577_; lean_object* v___x_2579_; uint8_t v_isShared_2580_; uint8_t v_isSharedCheck_2633_; 
v_fst_2576_ = lean_ctor_get(v_b_2567_, 0);
v_snd_2577_ = lean_ctor_get(v_b_2567_, 1);
v_isSharedCheck_2633_ = !lean_is_exclusive(v_b_2567_);
if (v_isSharedCheck_2633_ == 0)
{
v___x_2579_ = v_b_2567_;
v_isShared_2580_ = v_isSharedCheck_2633_;
goto v_resetjp_2578_;
}
else
{
lean_inc(v_snd_2577_);
lean_inc(v_fst_2576_);
lean_dec(v_b_2567_);
v___x_2579_ = lean_box(0);
v_isShared_2580_ = v_isSharedCheck_2633_;
goto v_resetjp_2578_;
}
v_resetjp_2578_:
{
lean_object* v___y_2582_; lean_object* v___x_2586_; lean_object* v_a_2587_; uint8_t v___x_2588_; 
v___x_2586_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2));
v_a_2587_ = lean_array_uget_borrowed(v_as_2564_, v_i_2566_);
lean_inc(v_a_2587_);
v___x_2588_ = l_Lean_Syntax_isOfKind(v_a_2587_, v___x_2586_);
if (v___x_2588_ == 0)
{
lean_object* v___x_2589_; 
lean_del_object(v___x_2579_);
v___x_2589_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2589_, 0, v_fst_2576_);
lean_ctor_set(v___x_2589_, 1, v_snd_2577_);
v_a_2570_ = v___x_2589_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; uint8_t v___x_2593_; 
v___x_2590_ = lean_unsigned_to_nat(0u);
v___x_2591_ = lean_unsigned_to_nat(1u);
v___x_2592_ = l_Lean_Syntax_getArg(v_a_2587_, v___x_2590_);
lean_inc(v___x_2592_);
v___x_2593_ = l_Lean_Syntax_matchesNull(v___x_2592_, v___x_2591_);
if (v___x_2593_ == 0)
{
lean_object* v___x_2594_; 
lean_dec(v___x_2592_);
lean_del_object(v___x_2579_);
v___x_2594_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2594_, 0, v_fst_2576_);
lean_ctor_set(v___x_2594_, 1, v_snd_2577_);
v_a_2570_ = v___x_2594_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2595_; lean_object* v___x_2596_; uint8_t v___x_2597_; 
v___x_2595_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4));
v___x_2596_ = l_Lean_Syntax_getArg(v___x_2592_, v___x_2590_);
lean_dec(v___x_2592_);
lean_inc(v___x_2596_);
v___x_2597_ = l_Lean_Syntax_isOfKind(v___x_2596_, v___x_2595_);
if (v___x_2597_ == 0)
{
lean_object* v___x_2598_; 
lean_dec(v___x_2596_);
lean_del_object(v___x_2579_);
v___x_2598_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2598_, 0, v_fst_2576_);
lean_ctor_set(v___x_2598_, 1, v_snd_2577_);
v_a_2570_ = v___x_2598_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; uint8_t v___x_2602_; 
v___x_2599_ = lean_unsigned_to_nat(2u);
v___x_2600_ = l_Lean_Syntax_getArg(v___x_2596_, v___x_2591_);
lean_dec(v___x_2596_);
v___x_2601_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6));
lean_inc(v___x_2600_);
v___x_2602_ = l_Lean_Syntax_isOfKind(v___x_2600_, v___x_2601_);
if (v___x_2602_ == 0)
{
lean_object* v___x_2603_; uint8_t v___x_2604_; 
v___x_2603_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__8));
lean_inc(v___x_2600_);
v___x_2604_ = l_Lean_Syntax_isOfKind(v___x_2600_, v___x_2603_);
if (v___x_2604_ == 0)
{
lean_object* v___x_2605_; 
lean_dec(v___x_2600_);
lean_del_object(v___x_2579_);
v___x_2605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2605_, 0, v_fst_2576_);
lean_ctor_set(v___x_2605_, 1, v_snd_2577_);
v_a_2570_ = v___x_2605_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2606_; uint8_t v___x_2607_; 
v___x_2606_ = l_Lean_Syntax_getArg(v___x_2600_, v___x_2590_);
v___x_2607_ = l_Lean_Syntax_matchesNull(v___x_2606_, v___x_2590_);
if (v___x_2607_ == 0)
{
lean_object* v___x_2608_; 
lean_dec(v___x_2600_);
lean_del_object(v___x_2579_);
v___x_2608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2608_, 0, v_fst_2576_);
lean_ctor_set(v___x_2608_, 1, v_snd_2577_);
v_a_2570_ = v___x_2608_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2609_; lean_object* v___x_2610_; uint8_t v___x_2611_; 
v___x_2609_ = l_Lean_Syntax_getArg(v___x_2600_, v___x_2591_);
lean_dec(v___x_2600_);
v___x_2610_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__10));
lean_inc(v___x_2609_);
v___x_2611_ = l_Lean_Syntax_isOfKind(v___x_2609_, v___x_2610_);
if (v___x_2611_ == 0)
{
lean_object* v___x_2612_; 
lean_dec(v___x_2609_);
lean_del_object(v___x_2579_);
v___x_2612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2612_, 0, v_fst_2576_);
lean_ctor_set(v___x_2612_, 1, v_snd_2577_);
v_a_2570_ = v___x_2612_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2613_; uint8_t v___x_2614_; 
v___x_2613_ = l_Lean_Syntax_getArg(v_a_2587_, v___x_2591_);
v___x_2614_ = l_Lean_Syntax_matchesNull(v___x_2613_, v___x_2599_);
if (v___x_2614_ == 0)
{
lean_object* v___x_2615_; 
lean_dec(v___x_2609_);
lean_del_object(v___x_2579_);
v___x_2615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2615_, 0, v_fst_2576_);
lean_ctor_set(v___x_2615_, 1, v_snd_2577_);
v_a_2570_ = v___x_2615_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2616_; lean_object* v___x_2617_; uint8_t v___x_2618_; 
v___x_2616_ = lean_array_get_size(v_fst_2576_);
v___x_2617_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_getElimExprNames___lam__0___closed__1));
v___x_2618_ = lean_nat_dec_lt(v___x_2590_, v___x_2616_);
if (v___x_2618_ == 0)
{
lean_dec(v___x_2609_);
lean_dec(v_fst_2576_);
v___y_2582_ = v___x_2617_;
goto v___jp_2581_;
}
else
{
uint8_t v___x_2619_; 
v___x_2619_ = lean_nat_dec_le(v___x_2616_, v___x_2616_);
if (v___x_2619_ == 0)
{
if (v___x_2618_ == 0)
{
lean_dec(v___x_2609_);
lean_dec(v_fst_2576_);
v___y_2582_ = v___x_2617_;
goto v___jp_2581_;
}
else
{
size_t v___x_2620_; size_t v___x_2621_; lean_object* v___x_2622_; 
v___x_2620_ = ((size_t)0ULL);
v___x_2621_ = lean_usize_of_nat(v___x_2616_);
v___x_2622_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10(v___x_2609_, v___x_2611_, v_fst_2576_, v___x_2620_, v___x_2621_, v___x_2617_);
lean_dec(v_fst_2576_);
lean_dec(v___x_2609_);
v___y_2582_ = v___x_2622_;
goto v___jp_2581_;
}
}
else
{
size_t v___x_2623_; size_t v___x_2624_; lean_object* v___x_2625_; 
v___x_2623_ = ((size_t)0ULL);
v___x_2624_ = lean_usize_of_nat(v___x_2616_);
v___x_2625_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__10(v___x_2609_, v___x_2611_, v_fst_2576_, v___x_2623_, v___x_2624_, v___x_2617_);
lean_dec(v_fst_2576_);
lean_dec(v___x_2609_);
v___y_2582_ = v___x_2625_;
goto v___jp_2581_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2626_; uint8_t v___x_2627_; 
lean_dec(v___x_2600_);
lean_del_object(v___x_2579_);
v___x_2626_ = l_Lean_Syntax_getArg(v_a_2587_, v___x_2591_);
lean_inc(v___x_2626_);
v___x_2627_ = l_Lean_Syntax_matchesNull(v___x_2626_, v___x_2599_);
if (v___x_2627_ == 0)
{
lean_object* v___x_2628_; 
lean_dec(v___x_2626_);
v___x_2628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2628_, 0, v_fst_2576_);
lean_ctor_set(v___x_2628_, 1, v_snd_2577_);
v_a_2570_ = v___x_2628_;
goto v___jp_2569_;
}
else
{
lean_object* v___x_2629_; uint8_t v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; 
lean_dec(v_snd_2577_);
v___x_2629_ = l_Lean_Syntax_getArg(v___x_2626_, v___x_2591_);
lean_dec(v___x_2626_);
v___x_2630_ = 0;
v___x_2631_ = l_Lean_Syntax_getRange_x3f(v___x_2629_, v___x_2630_);
lean_dec(v___x_2629_);
v___x_2632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2632_, 0, v_fst_2576_);
lean_ctor_set(v___x_2632_, 1, v___x_2631_);
v_a_2570_ = v___x_2632_;
goto v___jp_2569_;
}
}
}
}
}
v___jp_2581_:
{
lean_object* v___x_2584_; 
if (v_isShared_2580_ == 0)
{
lean_ctor_set(v___x_2579_, 0, v___y_2582_);
v___x_2584_ = v___x_2579_;
goto v_reusejp_2583_;
}
else
{
lean_object* v_reuseFailAlloc_2585_; 
v_reuseFailAlloc_2585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2585_, 0, v___y_2582_);
lean_ctor_set(v_reuseFailAlloc_2585_, 1, v_snd_2577_);
v___x_2584_ = v_reuseFailAlloc_2585_;
goto v_reusejp_2583_;
}
v_reusejp_2583_:
{
v_a_2570_ = v___x_2584_;
goto v___jp_2569_;
}
}
}
}
v___jp_2569_:
{
size_t v___x_2571_; size_t v___x_2572_; 
v___x_2571_ = ((size_t)1ULL);
v___x_2572_ = lean_usize_add(v_i_2566_, v___x_2571_);
v_i_2566_ = v___x_2572_;
v_b_2567_ = v_a_2570_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___boxed(lean_object* v_as_2634_, lean_object* v_sz_2635_, lean_object* v_i_2636_, lean_object* v_b_2637_, lean_object* v___y_2638_){
_start:
{
size_t v_sz_boxed_2639_; size_t v_i_boxed_2640_; lean_object* v_res_2641_; 
v_sz_boxed_2639_ = lean_unbox_usize(v_sz_2635_);
lean_dec(v_sz_2635_);
v_i_boxed_2640_ = lean_unbox_usize(v_i_2636_);
lean_dec(v_i_2636_);
v_res_2641_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg(v_as_2634_, v_sz_boxed_2639_, v_i_boxed_2640_, v_b_2637_);
lean_dec_ref(v_as_2634_);
return v_res_2641_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12(size_t v_sz_2648_, size_t v_i_2649_, lean_object* v_bs_2650_){
_start:
{
uint8_t v___x_2651_; 
v___x_2651_ = lean_usize_dec_lt(v_i_2649_, v_sz_2648_);
if (v___x_2651_ == 0)
{
lean_object* v___x_2652_; 
v___x_2652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2652_, 0, v_bs_2650_);
return v___x_2652_;
}
else
{
lean_object* v_v_2653_; lean_object* v___x_2654_; uint8_t v___x_2655_; 
v_v_2653_ = lean_array_uget(v_bs_2650_, v_i_2649_);
v___x_2654_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___closed__1));
lean_inc(v_v_2653_);
v___x_2655_ = l_Lean_Syntax_isOfKind(v_v_2653_, v___x_2654_);
if (v___x_2655_ == 0)
{
lean_object* v___x_2656_; 
lean_dec(v_v_2653_);
lean_dec_ref(v_bs_2650_);
v___x_2656_ = lean_box(0);
return v___x_2656_;
}
else
{
lean_object* v___x_2657_; lean_object* v___x_2658_; lean_object* v_bs_x27_2659_; lean_object* v___x_2666_; uint8_t v___x_2667_; 
v___x_2657_ = lean_unsigned_to_nat(1u);
v___x_2658_ = lean_unsigned_to_nat(0u);
v_bs_x27_2659_ = lean_array_uset(v_bs_2650_, v_i_2649_, v___x_2658_);
v___x_2666_ = l_Lean_Syntax_getArg(v_v_2653_, v___x_2658_);
v___x_2667_ = l_Lean_Syntax_isNone(v___x_2666_);
if (v___x_2667_ == 0)
{
lean_object* v___x_2668_; uint8_t v___x_2669_; 
v___x_2668_ = lean_unsigned_to_nat(2u);
v___x_2669_ = l_Lean_Syntax_matchesNull(v___x_2666_, v___x_2668_);
if (v___x_2669_ == 0)
{
lean_object* v___x_2670_; 
lean_dec_ref(v_bs_x27_2659_);
lean_dec(v_v_2653_);
v___x_2670_ = lean_box(0);
return v___x_2670_;
}
else
{
goto v___jp_2660_;
}
}
else
{
lean_dec(v___x_2666_);
goto v___jp_2660_;
}
v___jp_2660_:
{
lean_object* v_targets_2661_; size_t v___x_2662_; size_t v___x_2663_; lean_object* v___x_2664_; 
v_targets_2661_ = l_Lean_Syntax_getArg(v_v_2653_, v___x_2657_);
lean_dec(v_v_2653_);
v___x_2662_ = ((size_t)1ULL);
v___x_2663_ = lean_usize_add(v_i_2649_, v___x_2662_);
v___x_2664_ = lean_array_uset(v_bs_x27_2659_, v_i_2649_, v_targets_2661_);
v_i_2649_ = v___x_2663_;
v_bs_2650_ = v___x_2664_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12___boxed(lean_object* v_sz_2671_, lean_object* v_i_2672_, lean_object* v_bs_2673_){
_start:
{
size_t v_sz_boxed_2674_; size_t v_i_boxed_2675_; lean_object* v_res_2676_; 
v_sz_boxed_2674_ = lean_unbox_usize(v_sz_2671_);
lean_dec(v_sz_2671_);
v_i_boxed_2675_ = lean_unbox_usize(v_i_2672_);
lean_dec(v_i_2672_);
v_res_2676_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12(v_sz_boxed_2674_, v_i_boxed_2675_, v_bs_2673_);
return v_res_2676_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13(uint8_t v___x_2677_, uint8_t v___x_2678_, lean_object* v_as_2679_, size_t v_i_2680_, size_t v_stop_2681_, lean_object* v_b_2682_){
_start:
{
lean_object* v___y_2684_; uint8_t v___x_2688_; 
v___x_2688_ = lean_usize_dec_eq(v_i_2680_, v_stop_2681_);
if (v___x_2688_ == 0)
{
lean_object* v_fst_2689_; uint8_t v___x_2690_; 
v_fst_2689_ = lean_ctor_get(v_b_2682_, 0);
v___x_2690_ = lean_unbox(v_fst_2689_);
if (v___x_2690_ == 0)
{
lean_object* v_snd_2691_; lean_object* v___x_2693_; uint8_t v_isShared_2694_; uint8_t v_isSharedCheck_2699_; 
v_snd_2691_ = lean_ctor_get(v_b_2682_, 1);
v_isSharedCheck_2699_ = !lean_is_exclusive(v_b_2682_);
if (v_isSharedCheck_2699_ == 0)
{
lean_object* v_unused_2700_; 
v_unused_2700_ = lean_ctor_get(v_b_2682_, 0);
lean_dec(v_unused_2700_);
v___x_2693_ = v_b_2682_;
v_isShared_2694_ = v_isSharedCheck_2699_;
goto v_resetjp_2692_;
}
else
{
lean_inc(v_snd_2691_);
lean_dec(v_b_2682_);
v___x_2693_ = lean_box(0);
v_isShared_2694_ = v_isSharedCheck_2699_;
goto v_resetjp_2692_;
}
v_resetjp_2692_:
{
lean_object* v___x_2695_; lean_object* v___x_2697_; 
v___x_2695_ = lean_box(v___x_2677_);
if (v_isShared_2694_ == 0)
{
lean_ctor_set(v___x_2693_, 0, v___x_2695_);
v___x_2697_ = v___x_2693_;
goto v_reusejp_2696_;
}
else
{
lean_object* v_reuseFailAlloc_2698_; 
v_reuseFailAlloc_2698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2698_, 0, v___x_2695_);
lean_ctor_set(v_reuseFailAlloc_2698_, 1, v_snd_2691_);
v___x_2697_ = v_reuseFailAlloc_2698_;
goto v_reusejp_2696_;
}
v_reusejp_2696_:
{
v___y_2684_ = v___x_2697_;
goto v___jp_2683_;
}
}
}
else
{
lean_object* v_snd_2701_; lean_object* v___x_2703_; uint8_t v_isShared_2704_; uint8_t v_isSharedCheck_2711_; 
v_snd_2701_ = lean_ctor_get(v_b_2682_, 1);
v_isSharedCheck_2711_ = !lean_is_exclusive(v_b_2682_);
if (v_isSharedCheck_2711_ == 0)
{
lean_object* v_unused_2712_; 
v_unused_2712_ = lean_ctor_get(v_b_2682_, 0);
lean_dec(v_unused_2712_);
v___x_2703_ = v_b_2682_;
v_isShared_2704_ = v_isSharedCheck_2711_;
goto v_resetjp_2702_;
}
else
{
lean_inc(v_snd_2701_);
lean_dec(v_b_2682_);
v___x_2703_ = lean_box(0);
v_isShared_2704_ = v_isSharedCheck_2711_;
goto v_resetjp_2702_;
}
v_resetjp_2702_:
{
lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2709_; 
v___x_2705_ = lean_array_uget_borrowed(v_as_2679_, v_i_2680_);
lean_inc(v___x_2705_);
v___x_2706_ = lean_array_push(v_snd_2701_, v___x_2705_);
v___x_2707_ = lean_box(v___x_2678_);
if (v_isShared_2704_ == 0)
{
lean_ctor_set(v___x_2703_, 1, v___x_2706_);
lean_ctor_set(v___x_2703_, 0, v___x_2707_);
v___x_2709_ = v___x_2703_;
goto v_reusejp_2708_;
}
else
{
lean_object* v_reuseFailAlloc_2710_; 
v_reuseFailAlloc_2710_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2710_, 0, v___x_2707_);
lean_ctor_set(v_reuseFailAlloc_2710_, 1, v___x_2706_);
v___x_2709_ = v_reuseFailAlloc_2710_;
goto v_reusejp_2708_;
}
v_reusejp_2708_:
{
v___y_2684_ = v___x_2709_;
goto v___jp_2683_;
}
}
}
}
else
{
return v_b_2682_;
}
v___jp_2683_:
{
size_t v___x_2685_; size_t v___x_2686_; 
v___x_2685_ = ((size_t)1ULL);
v___x_2686_ = lean_usize_add(v_i_2680_, v___x_2685_);
v_i_2680_ = v___x_2686_;
v_b_2682_ = v___y_2684_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13___boxed(lean_object* v___x_2713_, lean_object* v___x_2714_, lean_object* v_as_2715_, lean_object* v_i_2716_, lean_object* v_stop_2717_, lean_object* v_b_2718_){
_start:
{
uint8_t v___x_39927__boxed_2719_; uint8_t v___x_39928__boxed_2720_; size_t v_i_boxed_2721_; size_t v_stop_boxed_2722_; lean_object* v_res_2723_; 
v___x_39927__boxed_2719_ = lean_unbox(v___x_2713_);
v___x_39928__boxed_2720_ = lean_unbox(v___x_2714_);
v_i_boxed_2721_ = lean_unbox_usize(v_i_2716_);
lean_dec(v_i_2716_);
v_stop_boxed_2722_ = lean_unbox_usize(v_stop_2717_);
lean_dec(v_stop_2717_);
v_res_2723_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13(v___x_39927__boxed_2719_, v___x_39928__boxed_2720_, v_as_2715_, v_i_boxed_2721_, v_stop_boxed_2722_, v_b_2718_);
lean_dec_ref(v_as_2715_);
return v_res_2723_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4(lean_object* v_as_2724_, size_t v_i_2725_, size_t v_stop_2726_, lean_object* v_b_2727_){
_start:
{
lean_object* v___y_2729_; uint8_t v___x_2733_; 
v___x_2733_ = lean_usize_dec_eq(v_i_2725_, v_stop_2726_);
if (v___x_2733_ == 0)
{
lean_object* v___x_2734_; lean_object* v___x_2735_; uint8_t v___x_2736_; 
v___x_2734_ = lean_array_uget_borrowed(v_as_2724_, v_i_2725_);
v___x_2735_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2));
lean_inc(v___x_2734_);
v___x_2736_ = l_Lean_Syntax_isOfKind(v___x_2734_, v___x_2735_);
if (v___x_2736_ == 0)
{
lean_object* v___x_2737_; 
lean_inc(v___x_2734_);
v___x_2737_ = lean_array_push(v_b_2727_, v___x_2734_);
v___y_2729_ = v___x_2737_;
goto v___jp_2728_;
}
else
{
lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; uint8_t v___x_2741_; 
v___x_2738_ = lean_unsigned_to_nat(0u);
v___x_2739_ = l_Lean_Syntax_getArg(v___x_2734_, v___x_2738_);
v___x_2740_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_2739_);
v___x_2741_ = l_Lean_Syntax_matchesNull(v___x_2739_, v___x_2740_);
if (v___x_2741_ == 0)
{
lean_object* v___x_2742_; 
lean_dec(v___x_2739_);
lean_inc(v___x_2734_);
v___x_2742_ = lean_array_push(v_b_2727_, v___x_2734_);
v___y_2729_ = v___x_2742_;
goto v___jp_2728_;
}
else
{
lean_object* v___x_2743_; lean_object* v___x_2744_; uint8_t v___x_2745_; 
v___x_2743_ = l_Lean_Syntax_getArg(v___x_2739_, v___x_2738_);
lean_dec(v___x_2739_);
v___x_2744_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__4));
lean_inc(v___x_2743_);
v___x_2745_ = l_Lean_Syntax_isOfKind(v___x_2743_, v___x_2744_);
if (v___x_2745_ == 0)
{
lean_object* v___x_2746_; 
lean_dec(v___x_2743_);
lean_inc(v___x_2734_);
v___x_2746_ = lean_array_push(v_b_2727_, v___x_2734_);
v___y_2729_ = v___x_2746_;
goto v___jp_2728_;
}
else
{
lean_object* v___x_2747_; lean_object* v___x_2748_; uint8_t v___x_2749_; 
v___x_2747_ = l_Lean_Syntax_getArg(v___x_2743_, v___x_2740_);
lean_dec(v___x_2743_);
v___x_2748_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__6));
v___x_2749_ = l_Lean_Syntax_isOfKind(v___x_2747_, v___x_2748_);
if (v___x_2749_ == 0)
{
lean_object* v___x_2750_; 
lean_inc(v___x_2734_);
v___x_2750_ = lean_array_push(v_b_2727_, v___x_2734_);
v___y_2729_ = v___x_2750_;
goto v___jp_2728_;
}
else
{
lean_object* v___x_2751_; lean_object* v___x_2752_; uint8_t v___x_2753_; 
v___x_2751_ = lean_unsigned_to_nat(2u);
v___x_2752_ = l_Lean_Syntax_getArg(v___x_2734_, v___x_2740_);
v___x_2753_ = l_Lean_Syntax_matchesNull(v___x_2752_, v___x_2751_);
if (v___x_2753_ == 0)
{
lean_object* v___x_2754_; 
lean_inc(v___x_2734_);
v___x_2754_ = lean_array_push(v_b_2727_, v___x_2734_);
v___y_2729_ = v___x_2754_;
goto v___jp_2728_;
}
else
{
v___y_2729_ = v_b_2727_;
goto v___jp_2728_;
}
}
}
}
}
}
else
{
return v_b_2727_;
}
v___jp_2728_:
{
size_t v___x_2730_; size_t v___x_2731_; 
v___x_2730_ = ((size_t)1ULL);
v___x_2731_ = lean_usize_add(v_i_2725_, v___x_2730_);
v_i_2725_ = v___x_2731_;
v_b_2727_ = v___y_2729_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4___boxed(lean_object* v_as_2755_, lean_object* v_i_2756_, lean_object* v_stop_2757_, lean_object* v_b_2758_){
_start:
{
size_t v_i_boxed_2759_; size_t v_stop_boxed_2760_; lean_object* v_res_2761_; 
v_i_boxed_2759_ = lean_unbox_usize(v_i_2756_);
lean_dec(v_i_2756_);
v_stop_boxed_2760_ = lean_unbox_usize(v_stop_2757_);
lean_dec(v_stop_2757_);
v_res_2761_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4(v_as_2755_, v_i_boxed_2759_, v_stop_boxed_2760_, v_b_2758_);
lean_dec_ref(v_as_2755_);
return v_res_2761_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg(lean_object* v_snap_2785_, lean_object* v_ctx_2786_, lean_object* v_node_2787_, lean_object* v_a_2788_){
_start:
{
lean_object* v___y_2791_; lean_object* v___y_2792_; lean_object* v___y_2793_; lean_object* v___y_2794_; lean_object* v___y_2795_; lean_object* v___y_2796_; lean_object* v___y_2797_; lean_object* v___y_2798_; lean_object* v___y_2799_; lean_object* v___y_2800_; lean_object* v___y_2801_; lean_object* v___y_2802_; lean_object* v___y_2803_; uint8_t v___y_2804_; lean_object* v___y_2805_; lean_object* v___y_2806_; lean_object* v___y_2807_; lean_object* v___y_2808_; lean_object* v___y_2818_; lean_object* v___y_2819_; lean_object* v___y_2820_; lean_object* v___y_2821_; lean_object* v___y_2822_; lean_object* v___y_2823_; lean_object* v___y_2824_; lean_object* v___y_2825_; lean_object* v___y_2826_; lean_object* v___y_2827_; lean_object* v___y_2828_; lean_object* v___y_2829_; lean_object* v___y_2830_; uint8_t v___y_2831_; lean_object* v___y_2832_; lean_object* v___y_2833_; lean_object* v___y_2834_; lean_object* v___y_2835_; lean_object* v___y_2836_; lean_object* v___y_2841_; lean_object* v___y_2842_; lean_object* v___y_2843_; lean_object* v___y_2844_; lean_object* v___y_2845_; lean_object* v___y_2846_; lean_object* v___y_2847_; lean_object* v___y_2848_; lean_object* v___y_2849_; lean_object* v___y_2850_; lean_object* v___y_2851_; lean_object* v___y_2852_; lean_object* v___y_2853_; uint8_t v___y_2854_; lean_object* v___y_2855_; lean_object* v___y_2856_; uint8_t v___y_2857_; lean_object* v___y_2858_; lean_object* v___y_2859_; lean_object* v___y_2860_; lean_object* v___y_2866_; lean_object* v___y_2867_; lean_object* v___y_2868_; lean_object* v___y_2869_; lean_object* v___y_2870_; lean_object* v___y_2871_; lean_object* v___y_2872_; lean_object* v___y_2873_; lean_object* v___y_2874_; lean_object* v___y_2875_; lean_object* v___y_2876_; lean_object* v___y_2877_; lean_object* v___y_2878_; uint8_t v___y_2879_; lean_object* v___y_2880_; lean_object* v___y_2881_; uint8_t v___y_2882_; lean_object* v___y_2883_; lean_object* v___y_2884_; lean_object* v___y_2885_; lean_object* v___y_2886_; lean_object* v___y_2887_; lean_object* v___y_2888_; lean_object* v___y_2889_; lean_object* v___y_2893_; lean_object* v___y_2894_; lean_object* v___y_2895_; lean_object* v___y_2896_; lean_object* v___y_2897_; lean_object* v___y_2898_; lean_object* v___y_2899_; lean_object* v___y_2900_; lean_object* v___y_2901_; lean_object* v___y_2902_; lean_object* v___y_2903_; lean_object* v___y_2904_; lean_object* v___y_2905_; uint8_t v___y_2906_; lean_object* v___y_2907_; lean_object* v___y_2908_; lean_object* v___y_2909_; uint8_t v___y_2910_; lean_object* v___y_2911_; lean_object* v___y_2912_; lean_object* v___y_2913_; lean_object* v___y_2914_; lean_object* v___y_2915_; lean_object* v___y_2916_; lean_object* v___y_2917_; lean_object* v___y_2918_; lean_object* v___y_2919_; lean_object* v___y_2920_; lean_object* v___y_2924_; lean_object* v___y_2925_; lean_object* v___y_2926_; lean_object* v___y_2927_; lean_object* v___y_2928_; lean_object* v___y_2929_; lean_object* v___y_2930_; lean_object* v___y_2931_; lean_object* v___y_2932_; lean_object* v___y_2933_; lean_object* v___y_2934_; lean_object* v___y_2935_; lean_object* v___y_2936_; uint8_t v___y_2937_; lean_object* v___y_2938_; lean_object* v___y_2939_; lean_object* v___y_2940_; uint8_t v___y_2941_; lean_object* v___y_2942_; lean_object* v___y_2943_; lean_object* v___y_2944_; lean_object* v___y_2945_; lean_object* v___y_2946_; lean_object* v___y_2947_; lean_object* v___y_2948_; lean_object* v___y_2949_; lean_object* v___y_2950_; lean_object* v___y_2951_; lean_object* v___y_2952_; lean_object* v___y_2953_; lean_object* v___y_2954_; lean_object* v___y_2955_; lean_object* v___y_2959_; lean_object* v___y_2960_; lean_object* v___y_2961_; lean_object* v___y_2962_; lean_object* v___y_2963_; lean_object* v___y_2964_; lean_object* v___y_2965_; lean_object* v___y_2966_; lean_object* v___y_2967_; lean_object* v___y_2968_; lean_object* v___y_2969_; lean_object* v___y_2970_; lean_object* v___y_2971_; uint8_t v___y_2972_; lean_object* v___y_2973_; lean_object* v___y_2974_; lean_object* v___y_2975_; uint8_t v___y_2976_; lean_object* v___y_2977_; lean_object* v___y_2978_; lean_object* v___y_2979_; lean_object* v___y_2980_; lean_object* v___y_2981_; lean_object* v___y_2982_; lean_object* v___y_2983_; lean_object* v___y_2984_; lean_object* v___y_2985_; lean_object* v___y_2986_; lean_object* v___y_2987_; lean_object* v___y_2988_; lean_object* v___y_2989_; lean_object* v___y_2990_; lean_object* v___y_2991_; lean_object* v___y_2992_; lean_object* v___y_2995_; lean_object* v___y_2996_; lean_object* v___y_2997_; lean_object* v___y_2998_; lean_object* v___y_2999_; lean_object* v___y_3000_; lean_object* v___y_3001_; lean_object* v___y_3002_; lean_object* v___y_3003_; lean_object* v___y_3004_; lean_object* v___y_3005_; lean_object* v___y_3006_; lean_object* v___y_3007_; uint8_t v___y_3008_; lean_object* v___y_3009_; lean_object* v___y_3010_; uint8_t v___y_3011_; lean_object* v___y_3012_; lean_object* v___y_3013_; lean_object* v___y_3014_; lean_object* v___y_3015_; lean_object* v___y_3016_; lean_object* v___y_3054_; lean_object* v___y_3055_; lean_object* v___y_3056_; lean_object* v___y_3057_; lean_object* v___y_3058_; lean_object* v___y_3059_; lean_object* v___y_3060_; lean_object* v___y_3061_; lean_object* v___y_3062_; lean_object* v___y_3063_; lean_object* v___y_3064_; lean_object* v___y_3065_; lean_object* v___y_3066_; uint8_t v___y_3067_; lean_object* v___y_3068_; lean_object* v___y_3069_; uint8_t v___y_3070_; lean_object* v___y_3071_; lean_object* v___y_3072_; lean_object* v___y_3073_; uint8_t v___y_3074_; lean_object* v___y_3075_; lean_object* v___y_3079_; lean_object* v___y_3080_; lean_object* v___y_3081_; lean_object* v___y_3082_; lean_object* v___y_3083_; lean_object* v___y_3084_; lean_object* v___y_3085_; lean_object* v___y_3086_; lean_object* v___y_3087_; lean_object* v___y_3088_; lean_object* v___y_3089_; lean_object* v___y_3090_; lean_object* v___y_3091_; uint8_t v___y_3092_; lean_object* v___y_3093_; lean_object* v___y_3094_; lean_object* v___y_3095_; uint8_t v___y_3096_; lean_object* v___y_3097_; lean_object* v___y_3098_; uint8_t v___y_3099_; lean_object* v___y_3100_; lean_object* v___y_3101_; lean_object* v___y_3103_; lean_object* v___y_3104_; lean_object* v___y_3105_; lean_object* v___y_3106_; lean_object* v___y_3107_; lean_object* v___y_3108_; lean_object* v___y_3109_; lean_object* v___y_3110_; lean_object* v___y_3111_; lean_object* v___y_3112_; uint8_t v___y_3113_; lean_object* v___y_3114_; lean_object* v___y_3115_; lean_object* v___y_3116_; lean_object* v___y_3117_; lean_object* v___y_3118_; uint8_t v___y_3119_; lean_object* v___y_3120_; lean_object* v___y_3121_; uint8_t v___y_3122_; lean_object* v___y_3123_; lean_object* v___y_3124_; lean_object* v___y_3125_; 
if (lean_obj_tag(v_node_2787_) == 1)
{
lean_object* v_i_3144_; 
v_i_3144_ = lean_ctor_get(v_node_2787_, 0);
lean_inc_ref(v_i_3144_);
if (lean_obj_tag(v_i_3144_) == 0)
{
lean_object* v_i_3145_; lean_object* v___x_3147_; uint8_t v_isShared_3148_; uint8_t v_isSharedCheck_3433_; 
v_i_3145_ = lean_ctor_get(v_i_3144_, 0);
v_isSharedCheck_3433_ = !lean_is_exclusive(v_i_3144_);
if (v_isSharedCheck_3433_ == 0)
{
v___x_3147_ = v_i_3144_;
v_isShared_3148_ = v_isSharedCheck_3433_;
goto v_resetjp_3146_;
}
else
{
lean_inc(v_i_3145_);
lean_dec(v_i_3144_);
v___x_3147_ = lean_box(0);
v_isShared_3148_ = v_isSharedCheck_3433_;
goto v_resetjp_3146_;
}
v_resetjp_3146_:
{
lean_object* v_toElabInfo_3149_; lean_object* v_stx_3150_; lean_object* v___x_3152_; uint8_t v_isShared_3153_; uint8_t v_isSharedCheck_3431_; 
v_toElabInfo_3149_ = lean_ctor_get(v_i_3145_, 0);
lean_inc_ref(v_toElabInfo_3149_);
lean_dec_ref(v_i_3145_);
v_stx_3150_ = lean_ctor_get(v_toElabInfo_3149_, 1);
v_isSharedCheck_3431_ = !lean_is_exclusive(v_toElabInfo_3149_);
if (v_isSharedCheck_3431_ == 0)
{
lean_object* v_unused_3432_; 
v_unused_3432_ = lean_ctor_get(v_toElabInfo_3149_, 0);
lean_dec(v_unused_3432_);
v___x_3152_ = v_toElabInfo_3149_;
v_isShared_3153_ = v_isSharedCheck_3431_;
goto v_resetjp_3151_;
}
else
{
lean_inc(v_stx_3150_);
lean_dec(v_toElabInfo_3149_);
v___x_3152_ = lean_box(0);
v_isShared_3153_ = v_isSharedCheck_3431_;
goto v_resetjp_3151_;
}
v_resetjp_3151_:
{
lean_object* v___y_3155_; lean_object* v___y_3156_; lean_object* v___y_3157_; lean_object* v___y_3158_; lean_object* v___y_3159_; lean_object* v___y_3160_; lean_object* v___y_3161_; lean_object* v___y_3162_; lean_object* v___y_3163_; lean_object* v___y_3164_; uint8_t v___y_3165_; lean_object* v___y_3166_; lean_object* v___y_3167_; lean_object* v___y_3168_; lean_object* v___y_3169_; uint8_t v___y_3170_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v___y_3173_; uint8_t v___y_3174_; lean_object* v___y_3175_; uint8_t v___y_3184_; uint8_t v___y_3185_; lean_object* v___y_3186_; uint8_t v___y_3187_; lean_object* v_ctors_3188_; lean_object* v_fallback_3189_; lean_object* v___y_3190_; lean_object* v_fst_3216_; uint8_t v_fst_3217_; lean_object* v_fst_3218_; lean_object* v_snd_3219_; lean_object* v___y_3220_; lean_object* v___y_3295_; lean_object* v___y_3296_; lean_object* v___y_3297_; lean_object* v___y_3298_; lean_object* v___y_3301_; lean_object* v_u_3302_; lean_object* v___y_3303_; lean_object* v___x_3316_; uint8_t v___x_3317_; 
v___x_3316_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__6));
lean_inc(v_stx_3150_);
v___x_3317_ = l_Lean_Syntax_isOfKind(v_stx_3150_, v___x_3316_);
if (v___x_3317_ == 0)
{
lean_object* v___x_3318_; uint8_t v___x_3319_; lean_object* v___y_3321_; lean_object* v___y_3322_; lean_object* v___y_3323_; lean_object* v___y_3337_; lean_object* v___y_3338_; lean_object* v_u_3339_; lean_object* v___y_3340_; 
v___x_3318_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__8));
lean_inc(v_stx_3150_);
v___x_3319_ = l_Lean_Syntax_isOfKind(v_stx_3150_, v___x_3318_);
if (v___x_3319_ == 0)
{
lean_object* v___x_3347_; lean_object* v___x_3348_; 
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3347_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3348_, 0, v___x_3347_);
return v___x_3348_;
}
else
{
lean_object* v___x_3349_; lean_object* v___y_3351_; lean_object* v___x_3373_; lean_object* v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; lean_object* v___x_3377_; uint8_t v___x_3378_; 
v___x_3349_ = lean_unsigned_to_nat(1u);
v___x_3373_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3349_);
v___x_3374_ = l_Lean_Syntax_getArgs(v___x_3373_);
lean_dec(v___x_3373_);
v___x_3375_ = lean_unsigned_to_nat(0u);
v___x_3376_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__9));
v___x_3377_ = lean_array_get_size(v___x_3374_);
v___x_3378_ = lean_nat_dec_lt(v___x_3375_, v___x_3377_);
if (v___x_3378_ == 0)
{
lean_dec_ref(v___x_3374_);
v___y_3351_ = v___x_3376_;
goto v___jp_3350_;
}
else
{
lean_object* v___x_3379_; lean_object* v___x_3380_; uint8_t v___x_3381_; 
v___x_3379_ = lean_box(v___x_3319_);
v___x_3380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3380_, 0, v___x_3379_);
lean_ctor_set(v___x_3380_, 1, v___x_3376_);
v___x_3381_ = lean_nat_dec_le(v___x_3377_, v___x_3377_);
if (v___x_3381_ == 0)
{
if (v___x_3378_ == 0)
{
lean_dec_ref_known(v___x_3380_, 2);
lean_dec_ref(v___x_3374_);
v___y_3351_ = v___x_3376_;
goto v___jp_3350_;
}
else
{
size_t v___x_3382_; size_t v___x_3383_; lean_object* v___x_3384_; lean_object* v_snd_3385_; 
v___x_3382_ = ((size_t)0ULL);
v___x_3383_ = lean_usize_of_nat(v___x_3377_);
v___x_3384_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13(v___x_3319_, v___x_3317_, v___x_3374_, v___x_3382_, v___x_3383_, v___x_3380_);
lean_dec_ref(v___x_3374_);
v_snd_3385_ = lean_ctor_get(v___x_3384_, 1);
lean_inc(v_snd_3385_);
lean_dec_ref(v___x_3384_);
v___y_3351_ = v_snd_3385_;
goto v___jp_3350_;
}
}
else
{
size_t v___x_3386_; size_t v___x_3387_; lean_object* v___x_3388_; lean_object* v_snd_3389_; 
v___x_3386_ = ((size_t)0ULL);
v___x_3387_ = lean_usize_of_nat(v___x_3377_);
v___x_3388_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__13(v___x_3319_, v___x_3317_, v___x_3374_, v___x_3386_, v___x_3387_, v___x_3380_);
lean_dec_ref(v___x_3374_);
v_snd_3389_ = lean_ctor_get(v___x_3388_, 1);
lean_inc(v_snd_3389_);
lean_dec_ref(v___x_3388_);
v___y_3351_ = v_snd_3389_;
goto v___jp_3350_;
}
}
v___jp_3350_:
{
size_t v_sz_3352_; size_t v___x_3353_; lean_object* v___x_3354_; 
v_sz_3352_ = lean_array_size(v___y_3351_);
v___x_3353_ = ((size_t)0ULL);
v___x_3354_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12(v_sz_3352_, v___x_3353_, v___y_3351_);
if (lean_obj_tag(v___x_3354_) == 0)
{
lean_object* v___x_3355_; lean_object* v___x_3356_; 
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3355_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3356_, 0, v___x_3355_);
return v___x_3356_;
}
else
{
lean_object* v_val_3357_; lean_object* v___x_3359_; uint8_t v_isShared_3360_; uint8_t v_isSharedCheck_3372_; 
v_val_3357_ = lean_ctor_get(v___x_3354_, 0);
v_isSharedCheck_3372_ = !lean_is_exclusive(v___x_3354_);
if (v_isSharedCheck_3372_ == 0)
{
v___x_3359_ = v___x_3354_;
v_isShared_3360_ = v_isSharedCheck_3372_;
goto v_resetjp_3358_;
}
else
{
lean_inc(v_val_3357_);
lean_dec(v___x_3354_);
v___x_3359_ = lean_box(0);
v_isShared_3360_ = v_isSharedCheck_3372_;
goto v_resetjp_3358_;
}
v_resetjp_3358_:
{
lean_object* v___x_3361_; lean_object* v___x_3362_; uint8_t v___x_3363_; 
v___x_3361_ = lean_unsigned_to_nat(2u);
v___x_3362_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3361_);
v___x_3363_ = l_Lean_Syntax_isNone(v___x_3362_);
if (v___x_3363_ == 0)
{
uint8_t v___x_3364_; 
lean_inc(v___x_3362_);
v___x_3364_ = l_Lean_Syntax_matchesNull(v___x_3362_, v___x_3361_);
if (v___x_3364_ == 0)
{
lean_object* v___x_3365_; lean_object* v___x_3366_; 
lean_dec(v___x_3362_);
lean_del_object(v___x_3359_);
lean_dec(v_val_3357_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3365_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3366_, 0, v___x_3365_);
return v___x_3366_;
}
else
{
lean_object* v_u_3367_; lean_object* v___x_3369_; 
v_u_3367_ = l_Lean_Syntax_getArg(v___x_3362_, v___x_3349_);
lean_dec(v___x_3362_);
if (v_isShared_3360_ == 0)
{
lean_ctor_set(v___x_3359_, 0, v_u_3367_);
v___x_3369_ = v___x_3359_;
goto v_reusejp_3368_;
}
else
{
lean_object* v_reuseFailAlloc_3370_; 
v_reuseFailAlloc_3370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3370_, 0, v_u_3367_);
v___x_3369_ = v_reuseFailAlloc_3370_;
goto v_reusejp_3368_;
}
v_reusejp_3368_:
{
v___y_3337_ = v___x_3361_;
v___y_3338_ = v_val_3357_;
v_u_3339_ = v___x_3369_;
v___y_3340_ = v_a_2788_;
goto v___jp_3336_;
}
}
}
else
{
lean_object* v___x_3371_; 
lean_dec(v___x_3362_);
lean_del_object(v___x_3359_);
v___x_3371_ = lean_box(0);
v___y_3337_ = v___x_3361_;
v___y_3338_ = v_val_3357_;
v_u_3339_ = v___x_3371_;
v___y_3340_ = v_a_2788_;
goto v___jp_3336_;
}
}
}
}
}
v___jp_3320_:
{
lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; 
v___x_3324_ = lean_unsigned_to_nat(4u);
v___x_3325_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3324_);
v___x_3326_ = l_Lean_Syntax_getOptional_x3f(v___x_3325_);
lean_dec(v___x_3325_);
if (lean_obj_tag(v___x_3326_) == 0)
{
lean_object* v___x_3327_; 
v___x_3327_ = lean_box(0);
v_fst_3216_ = v___y_3322_;
v_fst_3217_ = v___x_3319_;
v_fst_3218_ = v___y_3321_;
v_snd_3219_ = v___x_3327_;
v___y_3220_ = v___y_3323_;
goto v___jp_3215_;
}
else
{
lean_object* v_val_3328_; lean_object* v___x_3330_; uint8_t v_isShared_3331_; uint8_t v_isSharedCheck_3335_; 
v_val_3328_ = lean_ctor_get(v___x_3326_, 0);
v_isSharedCheck_3335_ = !lean_is_exclusive(v___x_3326_);
if (v_isSharedCheck_3335_ == 0)
{
v___x_3330_ = v___x_3326_;
v_isShared_3331_ = v_isSharedCheck_3335_;
goto v_resetjp_3329_;
}
else
{
lean_inc(v_val_3328_);
lean_dec(v___x_3326_);
v___x_3330_ = lean_box(0);
v_isShared_3331_ = v_isSharedCheck_3335_;
goto v_resetjp_3329_;
}
v_resetjp_3329_:
{
lean_object* v___x_3333_; 
if (v_isShared_3331_ == 0)
{
v___x_3333_ = v___x_3330_;
goto v_reusejp_3332_;
}
else
{
lean_object* v_reuseFailAlloc_3334_; 
v_reuseFailAlloc_3334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3334_, 0, v_val_3328_);
v___x_3333_ = v_reuseFailAlloc_3334_;
goto v_reusejp_3332_;
}
v_reusejp_3332_:
{
v_fst_3216_ = v___y_3322_;
v_fst_3217_ = v___x_3319_;
v_fst_3218_ = v___y_3321_;
v_snd_3219_ = v___x_3333_;
v___y_3220_ = v___y_3323_;
goto v___jp_3215_;
}
}
}
}
v___jp_3336_:
{
lean_object* v___x_3341_; lean_object* v___x_3342_; uint8_t v___x_3343_; 
v___x_3341_ = lean_unsigned_to_nat(3u);
v___x_3342_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3341_);
v___x_3343_ = l_Lean_Syntax_isNone(v___x_3342_);
if (v___x_3343_ == 0)
{
uint8_t v___x_3344_; 
v___x_3344_ = l_Lean_Syntax_matchesNull(v___x_3342_, v___y_3337_);
if (v___x_3344_ == 0)
{
lean_object* v___x_3345_; lean_object* v___x_3346_; 
lean_dec(v_u_3339_);
lean_dec_ref(v___y_3338_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3345_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3346_, 0, v___x_3345_);
return v___x_3346_;
}
else
{
v___y_3321_ = v_u_3339_;
v___y_3322_ = v___y_3338_;
v___y_3323_ = v___y_3340_;
goto v___jp_3320_;
}
}
else
{
lean_dec(v___x_3342_);
v___y_3321_ = v_u_3339_;
v___y_3322_ = v___y_3338_;
v___y_3323_ = v___y_3340_;
goto v___jp_3320_;
}
}
}
else
{
lean_object* v___x_3390_; lean_object* v___y_3392_; lean_object* v___x_3414_; lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v___x_3417_; lean_object* v___x_3418_; uint8_t v___x_3419_; 
v___x_3390_ = lean_unsigned_to_nat(1u);
v___x_3414_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3390_);
v___x_3415_ = l_Lean_Syntax_getArgs(v___x_3414_);
lean_dec(v___x_3414_);
v___x_3416_ = lean_unsigned_to_nat(0u);
v___x_3417_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__9));
v___x_3418_ = lean_array_get_size(v___x_3415_);
v___x_3419_ = lean_nat_dec_lt(v___x_3416_, v___x_3418_);
if (v___x_3419_ == 0)
{
lean_dec_ref(v___x_3415_);
v___y_3392_ = v___x_3417_;
goto v___jp_3391_;
}
else
{
lean_object* v___x_3420_; lean_object* v___x_3421_; uint8_t v___x_3422_; 
v___x_3420_ = lean_box(v___x_3317_);
v___x_3421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3421_, 0, v___x_3420_);
lean_ctor_set(v___x_3421_, 1, v___x_3417_);
v___x_3422_ = lean_nat_dec_le(v___x_3418_, v___x_3418_);
if (v___x_3422_ == 0)
{
if (v___x_3419_ == 0)
{
lean_dec_ref_known(v___x_3421_, 2);
lean_dec_ref(v___x_3415_);
v___y_3392_ = v___x_3417_;
goto v___jp_3391_;
}
else
{
size_t v___x_3423_; size_t v___x_3424_; lean_object* v___x_3425_; lean_object* v_snd_3426_; 
v___x_3423_ = ((size_t)0ULL);
v___x_3424_ = lean_usize_of_nat(v___x_3418_);
v___x_3425_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14(v___x_3317_, v___x_3415_, v___x_3423_, v___x_3424_, v___x_3421_);
lean_dec_ref(v___x_3415_);
v_snd_3426_ = lean_ctor_get(v___x_3425_, 1);
lean_inc(v_snd_3426_);
lean_dec_ref(v___x_3425_);
v___y_3392_ = v_snd_3426_;
goto v___jp_3391_;
}
}
else
{
size_t v___x_3427_; size_t v___x_3428_; lean_object* v___x_3429_; lean_object* v_snd_3430_; 
v___x_3427_ = ((size_t)0ULL);
v___x_3428_ = lean_usize_of_nat(v___x_3418_);
v___x_3429_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__14(v___x_3317_, v___x_3415_, v___x_3427_, v___x_3428_, v___x_3421_);
lean_dec_ref(v___x_3415_);
v_snd_3430_ = lean_ctor_get(v___x_3429_, 1);
lean_inc(v_snd_3430_);
lean_dec_ref(v___x_3429_);
v___y_3392_ = v_snd_3430_;
goto v___jp_3391_;
}
}
v___jp_3391_:
{
size_t v_sz_3393_; size_t v___x_3394_; lean_object* v___x_3395_; 
v_sz_3393_ = lean_array_size(v___y_3392_);
v___x_3394_ = ((size_t)0ULL);
v___x_3395_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__12(v_sz_3393_, v___x_3394_, v___y_3392_);
if (lean_obj_tag(v___x_3395_) == 0)
{
lean_object* v___x_3396_; lean_object* v___x_3397_; 
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3396_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3397_, 0, v___x_3396_);
return v___x_3397_;
}
else
{
lean_object* v_val_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3413_; 
v_val_3398_ = lean_ctor_get(v___x_3395_, 0);
v_isSharedCheck_3413_ = !lean_is_exclusive(v___x_3395_);
if (v_isSharedCheck_3413_ == 0)
{
v___x_3400_ = v___x_3395_;
v_isShared_3401_ = v_isSharedCheck_3413_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_val_3398_);
lean_dec(v___x_3395_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3413_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3402_; lean_object* v___x_3403_; uint8_t v___x_3404_; 
v___x_3402_ = lean_unsigned_to_nat(2u);
v___x_3403_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3402_);
v___x_3404_ = l_Lean_Syntax_isNone(v___x_3403_);
if (v___x_3404_ == 0)
{
uint8_t v___x_3405_; 
lean_inc(v___x_3403_);
v___x_3405_ = l_Lean_Syntax_matchesNull(v___x_3403_, v___x_3402_);
if (v___x_3405_ == 0)
{
lean_object* v___x_3406_; lean_object* v___x_3407_; 
lean_dec(v___x_3403_);
lean_del_object(v___x_3400_);
lean_dec(v_val_3398_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3406_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3407_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3407_, 0, v___x_3406_);
return v___x_3407_;
}
else
{
lean_object* v_u_3408_; lean_object* v___x_3410_; 
v_u_3408_ = l_Lean_Syntax_getArg(v___x_3403_, v___x_3390_);
lean_dec(v___x_3403_);
if (v_isShared_3401_ == 0)
{
lean_ctor_set(v___x_3400_, 0, v_u_3408_);
v___x_3410_ = v___x_3400_;
goto v_reusejp_3409_;
}
else
{
lean_object* v_reuseFailAlloc_3411_; 
v_reuseFailAlloc_3411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3411_, 0, v_u_3408_);
v___x_3410_ = v_reuseFailAlloc_3411_;
goto v_reusejp_3409_;
}
v_reusejp_3409_:
{
v___y_3301_ = v_val_3398_;
v_u_3302_ = v___x_3410_;
v___y_3303_ = v_a_2788_;
goto v___jp_3300_;
}
}
}
else
{
lean_object* v___x_3412_; 
lean_dec(v___x_3403_);
lean_del_object(v___x_3400_);
v___x_3412_ = lean_box(0);
v___y_3301_ = v_val_3398_;
v_u_3302_ = v___x_3412_;
v___y_3303_ = v_a_2788_;
goto v___jp_3300_;
}
}
}
}
}
v___jp_3154_:
{
lean_object* v_toEditableDocumentCore_3176_; lean_object* v_meta_3177_; lean_object* v_text_3178_; lean_object* v___x_3179_; 
v_toEditableDocumentCore_3176_ = lean_ctor_get(v___y_3171_, 0);
lean_inc_ref(v_toEditableDocumentCore_3176_);
lean_dec_ref(v___y_3171_);
v_meta_3177_ = lean_ctor_get(v_toEditableDocumentCore_3176_, 0);
lean_inc_ref(v_meta_3177_);
lean_dec_ref(v_toEditableDocumentCore_3176_);
v_text_3178_ = lean_ctor_get(v_meta_3177_, 3);
lean_inc_ref(v_text_3178_);
lean_dec_ref(v_meta_3177_);
v___x_3179_ = l_Lean_Syntax_getTailPos_x3f(v_stx_3150_, v___y_3170_);
if (lean_obj_tag(v___x_3179_) == 0)
{
lean_object* v___x_3180_; lean_object* v___x_3181_; 
v___x_3180_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_3181_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_3180_);
v___y_3103_ = v___y_3155_;
v___y_3104_ = v___y_3156_;
v___y_3105_ = v___y_3157_;
v___y_3106_ = v___y_3158_;
v___y_3107_ = v___y_3159_;
v___y_3108_ = v___y_3160_;
v___y_3109_ = v___y_3161_;
v___y_3110_ = v___y_3162_;
v___y_3111_ = v___y_3163_;
v___y_3112_ = v___y_3164_;
v___y_3113_ = v___y_3165_;
v___y_3114_ = v___y_3166_;
v___y_3115_ = v___y_3167_;
v___y_3116_ = v___y_3169_;
v___y_3117_ = v___y_3168_;
v___y_3118_ = v___y_3175_;
v___y_3119_ = v___y_3170_;
v___y_3120_ = v_text_3178_;
v___y_3121_ = v___y_3172_;
v___y_3122_ = v___y_3174_;
v___y_3123_ = v___y_3173_;
v___y_3124_ = v_stx_3150_;
v___y_3125_ = v___x_3181_;
goto v___jp_3102_;
}
else
{
lean_object* v_val_3182_; 
v_val_3182_ = lean_ctor_get(v___x_3179_, 0);
lean_inc(v_val_3182_);
lean_dec_ref_known(v___x_3179_, 1);
v___y_3103_ = v___y_3155_;
v___y_3104_ = v___y_3156_;
v___y_3105_ = v___y_3157_;
v___y_3106_ = v___y_3158_;
v___y_3107_ = v___y_3159_;
v___y_3108_ = v___y_3160_;
v___y_3109_ = v___y_3161_;
v___y_3110_ = v___y_3162_;
v___y_3111_ = v___y_3163_;
v___y_3112_ = v___y_3164_;
v___y_3113_ = v___y_3165_;
v___y_3114_ = v___y_3166_;
v___y_3115_ = v___y_3167_;
v___y_3116_ = v___y_3169_;
v___y_3117_ = v___y_3168_;
v___y_3118_ = v___y_3175_;
v___y_3119_ = v___y_3170_;
v___y_3120_ = v_text_3178_;
v___y_3121_ = v___y_3172_;
v___y_3122_ = v___y_3174_;
v___y_3123_ = v___y_3173_;
v___y_3124_ = v_stx_3150_;
v___y_3125_ = v_val_3182_;
goto v___jp_3102_;
}
}
v___jp_3183_:
{
lean_object* v___x_3191_; lean_object* v___x_3192_; uint8_t v___x_3193_; 
v___x_3191_ = lean_array_get_size(v_ctors_3188_);
v___x_3192_ = lean_unsigned_to_nat(0u);
v___x_3193_ = lean_nat_dec_eq(v___x_3191_, v___x_3192_);
if (v___x_3193_ == 0)
{
lean_object* v___x_3194_; lean_object* v_a_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3201_; lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___x_3206_; lean_object* v___x_3207_; 
lean_del_object(v___x_3147_);
v___x_3194_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v___y_3190_);
v_a_3195_ = lean_ctor_get(v___x_3194_, 0);
lean_inc(v_a_3195_);
lean_dec_ref(v___x_3194_);
v___x_3196_ = lean_box(0);
lean_inc(v_stx_3150_);
v___x_3197_ = l_Lean_Syntax_getKind(v_stx_3150_);
v___x_3198_ = lean_box(0);
v___x_3199_ = l_Lean_Name_updatePrefix(v___x_3197_, v___x_3198_);
v___x_3200_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__1));
v___x_3201_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v___x_3199_, v___y_3184_);
v___x_3202_ = lean_string_append(v___x_3200_, v___x_3201_);
lean_dec_ref(v___x_3201_);
v___x_3203_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__2));
v___x_3204_ = lean_string_append(v___x_3202_, v___x_3203_);
v___x_3205_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3));
lean_inc_ref(v___x_3204_);
v___x_3206_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3206_, 0, v___x_3196_);
lean_ctor_set(v___x_3206_, 1, v___x_3196_);
lean_ctor_set(v___x_3206_, 2, v___x_3204_);
lean_ctor_set(v___x_3206_, 3, v___x_3205_);
lean_ctor_set(v___x_3206_, 4, v___x_3196_);
lean_ctor_set(v___x_3206_, 5, v___x_3196_);
lean_ctor_set(v___x_3206_, 6, v___x_3196_);
lean_ctor_set(v___x_3206_, 7, v___x_3196_);
lean_ctor_set(v___x_3206_, 8, v___x_3196_);
lean_ctor_set(v___x_3206_, 9, v___x_3196_);
v___x_3207_ = l_Lean_Syntax_getPos_x3f(v_stx_3150_, v___x_3193_);
if (lean_obj_tag(v___x_3207_) == 0)
{
lean_object* v___x_3208_; lean_object* v___x_3209_; 
v___x_3208_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_3209_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_3208_);
lean_inc(v_a_3195_);
lean_inc(v_fallback_3189_);
v___y_3155_ = v___x_3196_;
v___y_3156_ = v___x_3204_;
v___y_3157_ = v_ctors_3188_;
v___y_3158_ = v___x_3196_;
v___y_3159_ = v___x_3196_;
v___y_3160_ = v_fallback_3189_;
v___y_3161_ = v___x_3196_;
v___y_3162_ = v_a_3195_;
v___y_3163_ = v___x_3196_;
v___y_3164_ = v___x_3196_;
v___y_3165_ = v___y_3185_;
v___y_3166_ = v___x_3205_;
v___y_3167_ = v___x_3191_;
v___y_3168_ = v_fallback_3189_;
v___y_3169_ = v___y_3186_;
v___y_3170_ = v___x_3193_;
v___y_3171_ = v_a_3195_;
v___y_3172_ = v___x_3192_;
v___y_3173_ = v___x_3206_;
v___y_3174_ = v___y_3187_;
v___y_3175_ = v___x_3209_;
goto v___jp_3154_;
}
else
{
lean_object* v_val_3210_; 
v_val_3210_ = lean_ctor_get(v___x_3207_, 0);
lean_inc(v_val_3210_);
lean_dec_ref_known(v___x_3207_, 1);
lean_inc(v_a_3195_);
lean_inc(v_fallback_3189_);
v___y_3155_ = v___x_3196_;
v___y_3156_ = v___x_3204_;
v___y_3157_ = v_ctors_3188_;
v___y_3158_ = v___x_3196_;
v___y_3159_ = v___x_3196_;
v___y_3160_ = v_fallback_3189_;
v___y_3161_ = v___x_3196_;
v___y_3162_ = v_a_3195_;
v___y_3163_ = v___x_3196_;
v___y_3164_ = v___x_3196_;
v___y_3165_ = v___y_3185_;
v___y_3166_ = v___x_3205_;
v___y_3167_ = v___x_3191_;
v___y_3168_ = v_fallback_3189_;
v___y_3169_ = v___y_3186_;
v___y_3170_ = v___x_3193_;
v___y_3171_ = v_a_3195_;
v___y_3172_ = v___x_3192_;
v___y_3173_ = v___x_3206_;
v___y_3174_ = v___y_3187_;
v___y_3175_ = v_val_3210_;
goto v___jp_3154_;
}
}
else
{
lean_object* v___x_3211_; lean_object* v___x_3213_; 
lean_dec(v_fallback_3189_);
lean_dec_ref(v_ctors_3188_);
lean_dec(v___y_3186_);
lean_dec(v_stx_3150_);
lean_dec_ref(v_snap_2785_);
v___x_3211_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_3148_ == 0)
{
lean_ctor_set(v___x_3147_, 0, v___x_3211_);
v___x_3213_ = v___x_3147_;
goto v_reusejp_3212_;
}
else
{
lean_object* v_reuseFailAlloc_3214_; 
v_reuseFailAlloc_3214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3214_, 0, v___x_3211_);
v___x_3213_ = v_reuseFailAlloc_3214_;
goto v_reusejp_3212_;
}
v_reusejp_3212_:
{
return v___x_3213_;
}
}
}
v___jp_3215_:
{
size_t v_sz_3221_; size_t v___x_3222_; lean_object* v___x_3223_; 
v_sz_3221_ = lean_array_size(v_fst_3216_);
v___x_3222_ = ((size_t)0ULL);
lean_inc_ref(v_node_2787_);
v___x_3223_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__5(v_node_2787_, v_sz_3221_, v___x_3222_, v_fst_3216_);
if (lean_obj_tag(v___x_3223_) == 1)
{
lean_object* v_val_3224_; lean_object* v___x_3226_; uint8_t v_isShared_3227_; uint8_t v_isSharedCheck_3291_; 
v_val_3224_ = lean_ctor_get(v___x_3223_, 0);
v_isSharedCheck_3291_ = !lean_is_exclusive(v___x_3223_);
if (v_isSharedCheck_3291_ == 0)
{
v___x_3226_ = v___x_3223_;
v_isShared_3227_ = v_isSharedCheck_3291_;
goto v_resetjp_3225_;
}
else
{
lean_inc(v_val_3224_);
lean_dec(v___x_3223_);
v___x_3226_ = lean_box(0);
v_isShared_3227_ = v_isSharedCheck_3291_;
goto v_resetjp_3225_;
}
v_resetjp_3225_:
{
lean_object* v___x_3228_; lean_object* v___x_3229_; uint8_t v___x_3230_; 
v___x_3228_ = lean_unsigned_to_nat(0u);
v___x_3229_ = lean_array_get_size(v_val_3224_);
v___x_3230_ = lean_nat_dec_lt(v___x_3228_, v___x_3229_);
if (v___x_3230_ == 0)
{
lean_object* v___x_3231_; lean_object* v___x_3233_; 
lean_dec(v_val_3224_);
lean_dec(v_snd_3219_);
lean_dec(v_fst_3218_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3231_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_3227_ == 0)
{
lean_ctor_set_tag(v___x_3226_, 0);
lean_ctor_set(v___x_3226_, 0, v___x_3231_);
v___x_3233_ = v___x_3226_;
goto v_reusejp_3232_;
}
else
{
lean_object* v_reuseFailAlloc_3234_; 
v_reuseFailAlloc_3234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3234_, 0, v___x_3231_);
v___x_3233_ = v_reuseFailAlloc_3234_;
goto v_reusejp_3232_;
}
v_reusejp_3232_:
{
return v___x_3233_;
}
}
else
{
lean_object* v___x_3235_; size_t v_sz_3236_; lean_object* v_targets_3237_; lean_object* v___x_3238_; lean_object* v___y_3239_; lean_object* v___x_3240_; 
lean_del_object(v___x_3226_);
v___x_3235_ = lean_array_fget(v_val_3224_, v___x_3228_);
v_sz_3236_ = lean_array_size(v_val_3224_);
v_targets_3237_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_CodeAction_casesExpand_spec__6(v_sz_3236_, v___x_3222_, v_val_3224_);
v___x_3238_ = lean_box(v_fst_3217_);
lean_inc(v___x_3235_);
v___y_3239_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___y_3239_, 0, v_fst_3218_);
lean_closure_set(v___y_3239_, 1, v___x_3235_);
lean_closure_set(v___y_3239_, 2, v___x_3238_);
lean_closure_set(v___y_3239_, 3, v_targets_3237_);
lean_closure_set(v___y_3239_, 4, v_node_2787_);
v___x_3240_ = l_Lean_Elab_TermInfo_runMetaM___redArg(v___x_3235_, v_ctx_2786_, v___y_3239_);
if (lean_obj_tag(v___x_3240_) == 0)
{
lean_object* v_a_3241_; lean_object* v___x_3243_; uint8_t v_isShared_3244_; uint8_t v_isSharedCheck_3281_; 
v_a_3241_ = lean_ctor_get(v___x_3240_, 0);
v_isSharedCheck_3281_ = !lean_is_exclusive(v___x_3240_);
if (v_isSharedCheck_3281_ == 0)
{
v___x_3243_ = v___x_3240_;
v_isShared_3244_ = v_isSharedCheck_3281_;
goto v_resetjp_3242_;
}
else
{
lean_inc(v_a_3241_);
lean_dec(v___x_3240_);
v___x_3243_ = lean_box(0);
v_isShared_3244_ = v_isSharedCheck_3281_;
goto v_resetjp_3242_;
}
v_resetjp_3242_:
{
if (lean_obj_tag(v_a_3241_) == 1)
{
lean_object* v_val_3245_; lean_object* v___x_3246_; 
lean_del_object(v___x_3243_);
v_val_3245_ = lean_ctor_get(v_a_3241_, 0);
lean_inc(v_val_3245_);
lean_dec_ref_known(v_a_3241_, 1);
v___x_3246_ = lean_box(0);
if (lean_obj_tag(v_snd_3219_) == 1)
{
lean_object* v_val_3247_; lean_object* v___x_3248_; uint8_t v___x_3249_; 
v_val_3247_ = lean_ctor_get(v_snd_3219_, 0);
v___x_3248_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__4));
lean_inc(v_val_3247_);
v___x_3249_ = l_Lean_Syntax_isOfKind(v_val_3247_, v___x_3248_);
if (v___x_3249_ == 0)
{
lean_del_object(v___x_3152_);
v___y_3184_ = v___x_3230_;
v___y_3185_ = v_fst_3217_;
v___y_3186_ = v_snd_3219_;
v___y_3187_ = v_fst_3217_;
v_ctors_3188_ = v_val_3245_;
v_fallback_3189_ = v___x_3246_;
v___y_3190_ = v___y_3220_;
goto v___jp_3183_;
}
else
{
lean_object* v___x_3250_; lean_object* v___x_3251_; lean_object* v___x_3252_; uint8_t v___x_3253_; 
v___x_3250_ = lean_unsigned_to_nat(1u);
v___x_3251_ = lean_unsigned_to_nat(2u);
v___x_3252_ = l_Lean_Syntax_getArg(v_val_3247_, v___x_3251_);
lean_inc(v___x_3252_);
v___x_3253_ = l_Lean_Syntax_matchesNull(v___x_3252_, v___x_3250_);
if (v___x_3253_ == 0)
{
lean_dec(v___x_3252_);
lean_del_object(v___x_3152_);
v___y_3184_ = v___x_3230_;
v___y_3185_ = v_fst_3217_;
v___y_3186_ = v_snd_3219_;
v___y_3187_ = v_fst_3217_;
v_ctors_3188_ = v_val_3245_;
v_fallback_3189_ = v___x_3246_;
v___y_3190_ = v___y_3220_;
goto v___jp_3183_;
}
else
{
lean_object* v___x_3254_; lean_object* v___x_3255_; uint8_t v___x_3256_; 
v___x_3254_ = l_Lean_Syntax_getArg(v___x_3252_, v___x_3228_);
lean_dec(v___x_3252_);
v___x_3255_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg___closed__2));
lean_inc(v___x_3254_);
v___x_3256_ = l_Lean_Syntax_isOfKind(v___x_3254_, v___x_3255_);
if (v___x_3256_ == 0)
{
lean_dec(v___x_3254_);
lean_del_object(v___x_3152_);
v___y_3184_ = v___x_3230_;
v___y_3185_ = v_fst_3217_;
v___y_3186_ = v_snd_3219_;
v___y_3187_ = v_fst_3217_;
v_ctors_3188_ = v_val_3245_;
v_fallback_3189_ = v___x_3246_;
v___y_3190_ = v___y_3220_;
goto v___jp_3183_;
}
else
{
lean_object* v___x_3257_; uint8_t v___x_3258_; 
v___x_3257_ = l_Lean_Syntax_getArg(v___x_3254_, v___x_3250_);
v___x_3258_ = l_Lean_Syntax_matchesNull(v___x_3257_, v___x_3228_);
if (v___x_3258_ == 0)
{
lean_dec(v___x_3254_);
lean_del_object(v___x_3152_);
v___y_3184_ = v___x_3230_;
v___y_3185_ = v_fst_3217_;
v___y_3186_ = v_snd_3219_;
v___y_3187_ = v_fst_3217_;
v_ctors_3188_ = v_val_3245_;
v_fallback_3189_ = v___x_3246_;
v___y_3190_ = v___y_3220_;
goto v___jp_3183_;
}
else
{
lean_object* v___x_3259_; lean_object* v___x_3260_; lean_object* v___x_3262_; 
v___x_3259_ = l_Lean_Syntax_getArg(v___x_3254_, v___x_3228_);
lean_dec(v___x_3254_);
v___x_3260_ = l_Lean_Syntax_getArgs(v___x_3259_);
lean_dec(v___x_3259_);
if (v_isShared_3153_ == 0)
{
lean_ctor_set(v___x_3152_, 1, v___x_3246_);
lean_ctor_set(v___x_3152_, 0, v_val_3245_);
v___x_3262_ = v___x_3152_;
goto v_reusejp_3261_;
}
else
{
lean_object* v_reuseFailAlloc_3276_; 
v_reuseFailAlloc_3276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3276_, 0, v_val_3245_);
lean_ctor_set(v_reuseFailAlloc_3276_, 1, v___x_3246_);
v___x_3262_ = v_reuseFailAlloc_3276_;
goto v_reusejp_3261_;
}
v_reusejp_3261_:
{
size_t v_sz_3263_; lean_object* v___x_3264_; 
v_sz_3263_ = lean_array_size(v___x_3260_);
v___x_3264_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg(v___x_3260_, v_sz_3263_, v___x_3222_, v___x_3262_);
lean_dec_ref(v___x_3260_);
if (lean_obj_tag(v___x_3264_) == 0)
{
lean_object* v_a_3265_; lean_object* v_fst_3266_; lean_object* v_snd_3267_; 
v_a_3265_ = lean_ctor_get(v___x_3264_, 0);
lean_inc(v_a_3265_);
lean_dec_ref_known(v___x_3264_, 1);
v_fst_3266_ = lean_ctor_get(v_a_3265_, 0);
lean_inc(v_fst_3266_);
v_snd_3267_ = lean_ctor_get(v_a_3265_, 1);
lean_inc(v_snd_3267_);
lean_dec(v_a_3265_);
v___y_3184_ = v___x_3230_;
v___y_3185_ = v_fst_3217_;
v___y_3186_ = v_snd_3219_;
v___y_3187_ = v_fst_3217_;
v_ctors_3188_ = v_fst_3266_;
v_fallback_3189_ = v_snd_3267_;
v___y_3190_ = v___y_3220_;
goto v___jp_3183_;
}
else
{
lean_object* v_a_3268_; lean_object* v___x_3270_; uint8_t v_isShared_3271_; uint8_t v_isSharedCheck_3275_; 
lean_dec_ref_known(v_snd_3219_, 1);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref(v_snap_2785_);
v_a_3268_ = lean_ctor_get(v___x_3264_, 0);
v_isSharedCheck_3275_ = !lean_is_exclusive(v___x_3264_);
if (v_isSharedCheck_3275_ == 0)
{
v___x_3270_ = v___x_3264_;
v_isShared_3271_ = v_isSharedCheck_3275_;
goto v_resetjp_3269_;
}
else
{
lean_inc(v_a_3268_);
lean_dec(v___x_3264_);
v___x_3270_ = lean_box(0);
v_isShared_3271_ = v_isSharedCheck_3275_;
goto v_resetjp_3269_;
}
v_resetjp_3269_:
{
lean_object* v___x_3273_; 
if (v_isShared_3271_ == 0)
{
v___x_3273_ = v___x_3270_;
goto v_reusejp_3272_;
}
else
{
lean_object* v_reuseFailAlloc_3274_; 
v_reuseFailAlloc_3274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3274_, 0, v_a_3268_);
v___x_3273_ = v_reuseFailAlloc_3274_;
goto v_reusejp_3272_;
}
v_reusejp_3272_:
{
return v___x_3273_;
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
lean_del_object(v___x_3152_);
v___y_3184_ = v___x_3230_;
v___y_3185_ = v_fst_3217_;
v___y_3186_ = v_snd_3219_;
v___y_3187_ = v_fst_3217_;
v_ctors_3188_ = v_val_3245_;
v_fallback_3189_ = v___x_3246_;
v___y_3190_ = v___y_3220_;
goto v___jp_3183_;
}
}
else
{
lean_object* v___x_3277_; lean_object* v___x_3279_; 
lean_dec(v_a_3241_);
lean_dec(v_snd_3219_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref(v_snap_2785_);
v___x_3277_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_3244_ == 0)
{
lean_ctor_set(v___x_3243_, 0, v___x_3277_);
v___x_3279_ = v___x_3243_;
goto v_reusejp_3278_;
}
else
{
lean_object* v_reuseFailAlloc_3280_; 
v_reuseFailAlloc_3280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3280_, 0, v___x_3277_);
v___x_3279_ = v_reuseFailAlloc_3280_;
goto v_reusejp_3278_;
}
v_reusejp_3278_:
{
return v___x_3279_;
}
}
}
}
else
{
lean_object* v_a_3282_; lean_object* v___x_3284_; uint8_t v_isShared_3285_; uint8_t v_isSharedCheck_3290_; 
lean_dec(v_snd_3219_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref(v_snap_2785_);
v_a_3282_ = lean_ctor_get(v___x_3240_, 0);
v_isSharedCheck_3290_ = !lean_is_exclusive(v___x_3240_);
if (v_isSharedCheck_3290_ == 0)
{
v___x_3284_ = v___x_3240_;
v_isShared_3285_ = v_isSharedCheck_3290_;
goto v_resetjp_3283_;
}
else
{
lean_inc(v_a_3282_);
lean_dec(v___x_3240_);
v___x_3284_ = lean_box(0);
v_isShared_3285_ = v_isSharedCheck_3290_;
goto v_resetjp_3283_;
}
v_resetjp_3283_:
{
lean_object* v___x_3286_; lean_object* v___x_3288_; 
v___x_3286_ = l_Lean_Server_RequestError_ofIoError(v_a_3282_);
if (v_isShared_3285_ == 0)
{
lean_ctor_set(v___x_3284_, 0, v___x_3286_);
v___x_3288_ = v___x_3284_;
goto v_reusejp_3287_;
}
else
{
lean_object* v_reuseFailAlloc_3289_; 
v_reuseFailAlloc_3289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3289_, 0, v___x_3286_);
v___x_3288_ = v_reuseFailAlloc_3289_;
goto v_reusejp_3287_;
}
v_reusejp_3287_:
{
return v___x_3288_;
}
}
}
}
}
}
else
{
lean_object* v___x_3292_; lean_object* v___x_3293_; 
lean_dec(v___x_3223_);
lean_dec(v_snd_3219_);
lean_dec(v_fst_3218_);
lean_del_object(v___x_3152_);
lean_dec(v_stx_3150_);
lean_del_object(v___x_3147_);
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
v___x_3292_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3293_, 0, v___x_3292_);
return v___x_3293_;
}
}
v___jp_3294_:
{
uint8_t v___x_3299_; 
v___x_3299_ = 0;
v_fst_3216_ = v___y_3296_;
v_fst_3217_ = v___x_3299_;
v_fst_3218_ = v___y_3295_;
v_snd_3219_ = v___y_3298_;
v___y_3220_ = v___y_3297_;
goto v___jp_3215_;
}
v___jp_3300_:
{
lean_object* v___x_3304_; lean_object* v___x_3305_; lean_object* v___x_3306_; 
v___x_3304_ = lean_unsigned_to_nat(3u);
v___x_3305_ = l_Lean_Syntax_getArg(v_stx_3150_, v___x_3304_);
v___x_3306_ = l_Lean_Syntax_getOptional_x3f(v___x_3305_);
lean_dec(v___x_3305_);
if (lean_obj_tag(v___x_3306_) == 0)
{
lean_object* v___x_3307_; 
v___x_3307_ = lean_box(0);
v___y_3295_ = v_u_3302_;
v___y_3296_ = v___y_3301_;
v___y_3297_ = v___y_3303_;
v___y_3298_ = v___x_3307_;
goto v___jp_3294_;
}
else
{
lean_object* v_val_3308_; lean_object* v___x_3310_; uint8_t v_isShared_3311_; uint8_t v_isSharedCheck_3315_; 
v_val_3308_ = lean_ctor_get(v___x_3306_, 0);
v_isSharedCheck_3315_ = !lean_is_exclusive(v___x_3306_);
if (v_isSharedCheck_3315_ == 0)
{
v___x_3310_ = v___x_3306_;
v_isShared_3311_ = v_isSharedCheck_3315_;
goto v_resetjp_3309_;
}
else
{
lean_inc(v_val_3308_);
lean_dec(v___x_3306_);
v___x_3310_ = lean_box(0);
v_isShared_3311_ = v_isSharedCheck_3315_;
goto v_resetjp_3309_;
}
v_resetjp_3309_:
{
lean_object* v___x_3313_; 
if (v_isShared_3311_ == 0)
{
v___x_3313_ = v___x_3310_;
goto v_reusejp_3312_;
}
else
{
lean_object* v_reuseFailAlloc_3314_; 
v_reuseFailAlloc_3314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3314_, 0, v_val_3308_);
v___x_3313_ = v_reuseFailAlloc_3314_;
goto v_reusejp_3312_;
}
v_reusejp_3312_:
{
v___y_3295_ = v_u_3302_;
v___y_3296_ = v___y_3301_;
v___y_3297_ = v___y_3303_;
v___y_3298_ = v___x_3313_;
goto v___jp_3294_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_node_2787_, 2);
lean_dec_ref(v_i_3144_);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
goto v___jp_3141_;
}
}
else
{
lean_dec_ref(v_node_2787_);
lean_dec_ref(v_ctx_2786_);
lean_dec_ref(v_snap_2785_);
goto v___jp_3141_;
}
v___jp_2790_:
{
lean_object* v___x_2809_; lean_object* v___y_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; lean_object* v___x_2816_; 
v___x_2809_ = lean_box(v___y_2804_);
lean_inc(v___y_2794_);
lean_inc(v___y_2796_);
lean_inc(v___y_2802_);
lean_inc(v___y_2801_);
lean_inc(v___y_2798_);
lean_inc(v___y_2805_);
lean_inc(v___y_2791_);
v___y_2810_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___lam__1___boxed), 19, 18);
lean_closure_set(v___y_2810_, 0, v___y_2808_);
lean_closure_set(v___y_2810_, 1, v___y_2793_);
lean_closure_set(v___y_2810_, 2, v___y_2795_);
lean_closure_set(v___y_2810_, 3, v___y_2806_);
lean_closure_set(v___y_2810_, 4, v___x_2809_);
lean_closure_set(v___y_2810_, 5, v_snap_2785_);
lean_closure_set(v___y_2810_, 6, v___y_2799_);
lean_closure_set(v___y_2810_, 7, v___y_2800_);
lean_closure_set(v___y_2810_, 8, v___y_2791_);
lean_closure_set(v___y_2810_, 9, v___y_2792_);
lean_closure_set(v___y_2810_, 10, v___y_2805_);
lean_closure_set(v___y_2810_, 11, v___y_2798_);
lean_closure_set(v___y_2810_, 12, v___y_2801_);
lean_closure_set(v___y_2810_, 13, v___y_2802_);
lean_closure_set(v___y_2810_, 14, v___y_2796_);
lean_closure_set(v___y_2810_, 15, v___y_2794_);
lean_closure_set(v___y_2810_, 16, v___y_2797_);
lean_closure_set(v___y_2810_, 17, v___y_2803_);
v___x_2811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2811_, 0, v___y_2810_);
v___x_2812_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2812_, 0, v___y_2807_);
lean_ctor_set(v___x_2812_, 1, v___x_2811_);
v___x_2813_ = lean_unsigned_to_nat(1u);
v___x_2814_ = lean_mk_empty_array_with_capacity(v___x_2813_);
v___x_2815_ = lean_array_push(v___x_2814_, v___x_2812_);
v___x_2816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2816_, 0, v___x_2815_);
return v___x_2816_;
}
v___jp_2817_:
{
lean_object* v___x_2837_; lean_object* v___x_2838_; lean_object* v___x_2839_; 
v___x_2837_ = l_Lean_FileMap_utf8PosToLspPos(v___y_2834_, v___y_2836_);
lean_dec(v___y_2836_);
v___x_2838_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10));
v___x_2839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2839_, 0, v___x_2837_);
lean_ctor_set(v___x_2839_, 1, v___x_2838_);
v___y_2791_ = v___y_2818_;
v___y_2792_ = v___y_2819_;
v___y_2793_ = v___y_2820_;
v___y_2794_ = v___y_2821_;
v___y_2795_ = v___y_2822_;
v___y_2796_ = v___y_2823_;
v___y_2797_ = v___y_2824_;
v___y_2798_ = v___y_2825_;
v___y_2799_ = v___y_2826_;
v___y_2800_ = v___y_2827_;
v___y_2801_ = v___y_2828_;
v___y_2802_ = v___y_2829_;
v___y_2803_ = v___y_2832_;
v___y_2804_ = v___y_2831_;
v___y_2805_ = v___y_2830_;
v___y_2806_ = v___y_2833_;
v___y_2807_ = v___y_2835_;
v___y_2808_ = v___x_2839_;
goto v___jp_2790_;
}
v___jp_2840_:
{
lean_object* v___x_2861_; 
v___x_2861_ = l_Lean_Syntax_getTailPos_x3f(v___y_2860_, v___y_2857_);
lean_dec(v___y_2860_);
if (lean_obj_tag(v___x_2861_) == 0)
{
lean_object* v___x_2862_; lean_object* v___x_2863_; 
v___x_2862_ = lean_obj_once(&lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3, &lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3_once, _init_lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__3);
v___x_2863_ = lp_batteries_panic___at___00Batteries_CodeAction_instanceStub_spec__1(v___x_2862_);
v___y_2818_ = v___y_2841_;
v___y_2819_ = v___y_2842_;
v___y_2820_ = v___y_2843_;
v___y_2821_ = v___y_2844_;
v___y_2822_ = v___y_2845_;
v___y_2823_ = v___y_2846_;
v___y_2824_ = v___y_2847_;
v___y_2825_ = v___y_2848_;
v___y_2826_ = v___y_2849_;
v___y_2827_ = v___y_2850_;
v___y_2828_ = v___y_2851_;
v___y_2829_ = v___y_2852_;
v___y_2830_ = v___y_2855_;
v___y_2831_ = v___y_2854_;
v___y_2832_ = v___y_2853_;
v___y_2833_ = v___y_2856_;
v___y_2834_ = v___y_2858_;
v___y_2835_ = v___y_2859_;
v___y_2836_ = v___x_2863_;
goto v___jp_2817_;
}
else
{
lean_object* v_val_2864_; 
v_val_2864_ = lean_ctor_get(v___x_2861_, 0);
lean_inc(v_val_2864_);
lean_dec_ref_known(v___x_2861_, 1);
v___y_2818_ = v___y_2841_;
v___y_2819_ = v___y_2842_;
v___y_2820_ = v___y_2843_;
v___y_2821_ = v___y_2844_;
v___y_2822_ = v___y_2845_;
v___y_2823_ = v___y_2846_;
v___y_2824_ = v___y_2847_;
v___y_2825_ = v___y_2848_;
v___y_2826_ = v___y_2849_;
v___y_2827_ = v___y_2850_;
v___y_2828_ = v___y_2851_;
v___y_2829_ = v___y_2852_;
v___y_2830_ = v___y_2855_;
v___y_2831_ = v___y_2854_;
v___y_2832_ = v___y_2853_;
v___y_2833_ = v___y_2856_;
v___y_2834_ = v___y_2858_;
v___y_2835_ = v___y_2859_;
v___y_2836_ = v_val_2864_;
goto v___jp_2817_;
}
}
v___jp_2865_:
{
lean_object* v___x_2890_; lean_object* v___x_2891_; 
v___x_2890_ = lean_array_fset(v___y_2885_, v___y_2883_, v___y_2889_);
v___x_2891_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2891_, 0, v___y_2888_);
lean_ctor_set(v___x_2891_, 1, v___y_2886_);
lean_ctor_set(v___x_2891_, 2, v___x_2890_);
v___y_2841_ = v___y_2866_;
v___y_2842_ = v___y_2867_;
v___y_2843_ = v___y_2868_;
v___y_2844_ = v___y_2869_;
v___y_2845_ = v___y_2870_;
v___y_2846_ = v___y_2871_;
v___y_2847_ = v___y_2872_;
v___y_2848_ = v___y_2873_;
v___y_2849_ = v___y_2874_;
v___y_2850_ = v___y_2875_;
v___y_2851_ = v___y_2876_;
v___y_2852_ = v___y_2877_;
v___y_2853_ = v___y_2880_;
v___y_2854_ = v___y_2879_;
v___y_2855_ = v___y_2878_;
v___y_2856_ = v___y_2881_;
v___y_2857_ = v___y_2882_;
v___y_2858_ = v___y_2884_;
v___y_2859_ = v___y_2887_;
v___y_2860_ = v___x_2891_;
goto v___jp_2840_;
}
v___jp_2892_:
{
lean_object* v___x_2921_; lean_object* v___x_2922_; 
v___x_2921_ = lean_array_fset(v___y_2916_, v___y_2914_, v___y_2920_);
v___x_2922_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2922_, 0, v___y_2919_);
lean_ctor_set(v___x_2922_, 1, v___y_2918_);
lean_ctor_set(v___x_2922_, 2, v___x_2921_);
v___y_2866_ = v___y_2893_;
v___y_2867_ = v___y_2894_;
v___y_2868_ = v___y_2895_;
v___y_2869_ = v___y_2896_;
v___y_2870_ = v___y_2897_;
v___y_2871_ = v___y_2898_;
v___y_2872_ = v___y_2899_;
v___y_2873_ = v___y_2900_;
v___y_2874_ = v___y_2901_;
v___y_2875_ = v___y_2902_;
v___y_2876_ = v___y_2903_;
v___y_2877_ = v___y_2904_;
v___y_2878_ = v___y_2907_;
v___y_2879_ = v___y_2906_;
v___y_2880_ = v___y_2905_;
v___y_2881_ = v___y_2908_;
v___y_2882_ = v___y_2910_;
v___y_2883_ = v___y_2909_;
v___y_2884_ = v___y_2912_;
v___y_2885_ = v___y_2911_;
v___y_2886_ = v___y_2913_;
v___y_2887_ = v___y_2915_;
v___y_2888_ = v___y_2917_;
v___y_2889_ = v___x_2922_;
goto v___jp_2865_;
}
v___jp_2923_:
{
lean_object* v___x_2956_; lean_object* v___x_2957_; 
v___x_2956_ = lean_array_fset(v___y_2950_, v___y_2945_, v___y_2955_);
v___x_2957_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2957_, 0, v___y_2952_);
lean_ctor_set(v___x_2957_, 1, v___y_2942_);
lean_ctor_set(v___x_2957_, 2, v___x_2956_);
v___y_2893_ = v___y_2924_;
v___y_2894_ = v___y_2925_;
v___y_2895_ = v___y_2926_;
v___y_2896_ = v___y_2927_;
v___y_2897_ = v___y_2928_;
v___y_2898_ = v___y_2929_;
v___y_2899_ = v___y_2930_;
v___y_2900_ = v___y_2931_;
v___y_2901_ = v___y_2932_;
v___y_2902_ = v___y_2933_;
v___y_2903_ = v___y_2934_;
v___y_2904_ = v___y_2935_;
v___y_2905_ = v___y_2938_;
v___y_2906_ = v___y_2937_;
v___y_2907_ = v___y_2936_;
v___y_2908_ = v___y_2939_;
v___y_2909_ = v___y_2940_;
v___y_2910_ = v___y_2941_;
v___y_2911_ = v___y_2943_;
v___y_2912_ = v___y_2944_;
v___y_2913_ = v___y_2951_;
v___y_2914_ = v___y_2946_;
v___y_2915_ = v___y_2947_;
v___y_2916_ = v___y_2948_;
v___y_2917_ = v___y_2954_;
v___y_2918_ = v___y_2953_;
v___y_2919_ = v___y_2949_;
v___y_2920_ = v___x_2957_;
goto v___jp_2892_;
}
v___jp_2958_:
{
lean_object* v___x_2993_; 
v___x_2993_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2993_, 0, v___y_2986_);
lean_ctor_set(v___x_2993_, 1, v___y_2991_);
lean_ctor_set(v___x_2993_, 2, v___y_2992_);
v___y_2924_ = v___y_2959_;
v___y_2925_ = v___y_2960_;
v___y_2926_ = v___y_2961_;
v___y_2927_ = v___y_2962_;
v___y_2928_ = v___y_2963_;
v___y_2929_ = v___y_2964_;
v___y_2930_ = v___y_2965_;
v___y_2931_ = v___y_2966_;
v___y_2932_ = v___y_2967_;
v___y_2933_ = v___y_2968_;
v___y_2934_ = v___y_2969_;
v___y_2935_ = v___y_2970_;
v___y_2936_ = v___y_2973_;
v___y_2937_ = v___y_2972_;
v___y_2938_ = v___y_2971_;
v___y_2939_ = v___y_2974_;
v___y_2940_ = v___y_2975_;
v___y_2941_ = v___y_2976_;
v___y_2942_ = v___y_2977_;
v___y_2943_ = v___y_2978_;
v___y_2944_ = v___y_2979_;
v___y_2945_ = v___y_2980_;
v___y_2946_ = v___y_2981_;
v___y_2947_ = v___y_2982_;
v___y_2948_ = v___y_2983_;
v___y_2949_ = v___y_2984_;
v___y_2950_ = v___y_2985_;
v___y_2951_ = v___y_2987_;
v___y_2952_ = v___y_2988_;
v___y_2953_ = v___y_2990_;
v___y_2954_ = v___y_2989_;
v___y_2955_ = v___x_2993_;
goto v___jp_2923_;
}
v___jp_2994_:
{
if (lean_obj_tag(v___y_3015_) == 1)
{
lean_object* v_info_3017_; lean_object* v_kind_3018_; lean_object* v_args_3019_; lean_object* v___x_3020_; uint8_t v___x_3021_; 
v_info_3017_ = lean_ctor_get(v___y_3015_, 0);
v_kind_3018_ = lean_ctor_get(v___y_3015_, 1);
v_args_3019_ = lean_ctor_get(v___y_3015_, 2);
v___x_3020_ = lean_array_get_size(v_args_3019_);
v___x_3021_ = lean_nat_dec_lt(v___y_3016_, v___x_3020_);
if (v___x_3021_ == 0)
{
v___y_2841_ = v___y_2995_;
v___y_2842_ = v___y_2996_;
v___y_2843_ = v___y_2997_;
v___y_2844_ = v___y_2998_;
v___y_2845_ = v___y_2999_;
v___y_2846_ = v___y_3000_;
v___y_2847_ = v___y_3001_;
v___y_2848_ = v___y_3002_;
v___y_2849_ = v___y_3003_;
v___y_2850_ = v___y_3004_;
v___y_2851_ = v___y_3005_;
v___y_2852_ = v___y_3006_;
v___y_2853_ = v___y_3009_;
v___y_2854_ = v___y_3008_;
v___y_2855_ = v___y_3007_;
v___y_2856_ = v___y_3010_;
v___y_2857_ = v___y_3011_;
v___y_2858_ = v___y_3012_;
v___y_2859_ = v___y_3014_;
v___y_2860_ = v___y_3015_;
goto v___jp_2840_;
}
else
{
lean_object* v_v_3022_; lean_object* v___x_3023_; lean_object* v_xs_x27_3024_; 
lean_inc_ref(v_args_3019_);
lean_inc(v_kind_3018_);
lean_inc(v_info_3017_);
lean_dec_ref_known(v___y_3015_, 3);
v_v_3022_ = lean_array_fget(v_args_3019_, v___y_3016_);
v___x_3023_ = lean_box(0);
v_xs_x27_3024_ = lean_array_fset(v_args_3019_, v___y_3016_, v___x_3023_);
if (lean_obj_tag(v_v_3022_) == 1)
{
lean_object* v_info_3025_; lean_object* v_kind_3026_; lean_object* v_args_3027_; lean_object* v___x_3028_; uint8_t v___x_3029_; 
v_info_3025_ = lean_ctor_get(v_v_3022_, 0);
v_kind_3026_ = lean_ctor_get(v_v_3022_, 1);
v_args_3027_ = lean_ctor_get(v_v_3022_, 2);
v___x_3028_ = lean_array_get_size(v_args_3027_);
v___x_3029_ = lean_nat_dec_lt(v___y_3013_, v___x_3028_);
if (v___x_3029_ == 0)
{
v___y_2866_ = v___y_2995_;
v___y_2867_ = v___y_2996_;
v___y_2868_ = v___y_2997_;
v___y_2869_ = v___y_2998_;
v___y_2870_ = v___y_2999_;
v___y_2871_ = v___y_3000_;
v___y_2872_ = v___y_3001_;
v___y_2873_ = v___y_3002_;
v___y_2874_ = v___y_3003_;
v___y_2875_ = v___y_3004_;
v___y_2876_ = v___y_3005_;
v___y_2877_ = v___y_3006_;
v___y_2878_ = v___y_3007_;
v___y_2879_ = v___y_3008_;
v___y_2880_ = v___y_3009_;
v___y_2881_ = v___y_3010_;
v___y_2882_ = v___y_3011_;
v___y_2883_ = v___y_3016_;
v___y_2884_ = v___y_3012_;
v___y_2885_ = v_xs_x27_3024_;
v___y_2886_ = v_kind_3018_;
v___y_2887_ = v___y_3014_;
v___y_2888_ = v_info_3017_;
v___y_2889_ = v_v_3022_;
goto v___jp_2865_;
}
else
{
lean_object* v_v_3030_; lean_object* v_xs_x27_3031_; 
lean_inc_ref(v_args_3027_);
lean_inc(v_kind_3026_);
lean_inc(v_info_3025_);
lean_dec_ref_known(v_v_3022_, 3);
v_v_3030_ = lean_array_fget(v_args_3027_, v___y_3013_);
v_xs_x27_3031_ = lean_array_fset(v_args_3027_, v___y_3013_, v___x_3023_);
if (lean_obj_tag(v_v_3030_) == 1)
{
lean_object* v_info_3032_; lean_object* v_kind_3033_; lean_object* v_args_3034_; lean_object* v___x_3035_; lean_object* v___x_3036_; uint8_t v___x_3037_; 
v_info_3032_ = lean_ctor_get(v_v_3030_, 0);
v_kind_3033_ = lean_ctor_get(v_v_3030_, 1);
v_args_3034_ = lean_ctor_get(v_v_3030_, 2);
v___x_3035_ = lean_unsigned_to_nat(2u);
v___x_3036_ = lean_array_get_size(v_args_3034_);
v___x_3037_ = lean_nat_dec_lt(v___x_3035_, v___x_3036_);
if (v___x_3037_ == 0)
{
v___y_2893_ = v___y_2995_;
v___y_2894_ = v___y_2996_;
v___y_2895_ = v___y_2997_;
v___y_2896_ = v___y_2998_;
v___y_2897_ = v___y_2999_;
v___y_2898_ = v___y_3000_;
v___y_2899_ = v___y_3001_;
v___y_2900_ = v___y_3002_;
v___y_2901_ = v___y_3003_;
v___y_2902_ = v___y_3004_;
v___y_2903_ = v___y_3005_;
v___y_2904_ = v___y_3006_;
v___y_2905_ = v___y_3009_;
v___y_2906_ = v___y_3008_;
v___y_2907_ = v___y_3007_;
v___y_2908_ = v___y_3010_;
v___y_2909_ = v___y_3016_;
v___y_2910_ = v___y_3011_;
v___y_2911_ = v_xs_x27_3024_;
v___y_2912_ = v___y_3012_;
v___y_2913_ = v_kind_3018_;
v___y_2914_ = v___y_3013_;
v___y_2915_ = v___y_3014_;
v___y_2916_ = v_xs_x27_3031_;
v___y_2917_ = v_info_3017_;
v___y_2918_ = v_kind_3026_;
v___y_2919_ = v_info_3025_;
v___y_2920_ = v_v_3030_;
goto v___jp_2892_;
}
else
{
lean_object* v_v_3038_; lean_object* v_xs_x27_3039_; 
lean_inc_ref(v_args_3034_);
lean_inc(v_kind_3033_);
lean_inc(v_info_3032_);
lean_dec_ref_known(v_v_3030_, 3);
v_v_3038_ = lean_array_fget(v_args_3034_, v___x_3035_);
v_xs_x27_3039_ = lean_array_fset(v_args_3034_, v___x_3035_, v___x_3023_);
if (lean_obj_tag(v_v_3038_) == 1)
{
lean_object* v_info_3040_; lean_object* v_kind_3041_; lean_object* v_args_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; uint8_t v___x_3045_; 
v_info_3040_ = lean_ctor_get(v_v_3038_, 0);
lean_inc(v_info_3040_);
v_kind_3041_ = lean_ctor_get(v_v_3038_, 1);
lean_inc(v_kind_3041_);
v_args_3042_ = lean_ctor_get(v_v_3038_, 2);
lean_inc_ref(v_args_3042_);
lean_dec_ref_known(v_v_3038_, 3);
v___x_3043_ = lean_array_get_size(v_args_3042_);
v___x_3044_ = lean_mk_empty_array_with_capacity(v___y_3013_);
v___x_3045_ = lean_nat_dec_lt(v___y_3013_, v___x_3043_);
if (v___x_3045_ == 0)
{
lean_dec_ref(v_args_3042_);
v___y_2959_ = v___y_2995_;
v___y_2960_ = v___y_2996_;
v___y_2961_ = v___y_2997_;
v___y_2962_ = v___y_2998_;
v___y_2963_ = v___y_2999_;
v___y_2964_ = v___y_3000_;
v___y_2965_ = v___y_3001_;
v___y_2966_ = v___y_3002_;
v___y_2967_ = v___y_3003_;
v___y_2968_ = v___y_3004_;
v___y_2969_ = v___y_3005_;
v___y_2970_ = v___y_3006_;
v___y_2971_ = v___y_3009_;
v___y_2972_ = v___y_3008_;
v___y_2973_ = v___y_3007_;
v___y_2974_ = v___y_3010_;
v___y_2975_ = v___y_3016_;
v___y_2976_ = v___y_3011_;
v___y_2977_ = v_kind_3033_;
v___y_2978_ = v_xs_x27_3024_;
v___y_2979_ = v___y_3012_;
v___y_2980_ = v___x_3035_;
v___y_2981_ = v___y_3013_;
v___y_2982_ = v___y_3014_;
v___y_2983_ = v_xs_x27_3031_;
v___y_2984_ = v_info_3025_;
v___y_2985_ = v_xs_x27_3039_;
v___y_2986_ = v_info_3040_;
v___y_2987_ = v_kind_3018_;
v___y_2988_ = v_info_3032_;
v___y_2989_ = v_info_3017_;
v___y_2990_ = v_kind_3026_;
v___y_2991_ = v_kind_3041_;
v___y_2992_ = v___x_3044_;
goto v___jp_2958_;
}
else
{
uint8_t v___x_3046_; 
v___x_3046_ = lean_nat_dec_le(v___x_3043_, v___x_3043_);
if (v___x_3046_ == 0)
{
if (v___x_3045_ == 0)
{
lean_dec_ref(v_args_3042_);
v___y_2959_ = v___y_2995_;
v___y_2960_ = v___y_2996_;
v___y_2961_ = v___y_2997_;
v___y_2962_ = v___y_2998_;
v___y_2963_ = v___y_2999_;
v___y_2964_ = v___y_3000_;
v___y_2965_ = v___y_3001_;
v___y_2966_ = v___y_3002_;
v___y_2967_ = v___y_3003_;
v___y_2968_ = v___y_3004_;
v___y_2969_ = v___y_3005_;
v___y_2970_ = v___y_3006_;
v___y_2971_ = v___y_3009_;
v___y_2972_ = v___y_3008_;
v___y_2973_ = v___y_3007_;
v___y_2974_ = v___y_3010_;
v___y_2975_ = v___y_3016_;
v___y_2976_ = v___y_3011_;
v___y_2977_ = v_kind_3033_;
v___y_2978_ = v_xs_x27_3024_;
v___y_2979_ = v___y_3012_;
v___y_2980_ = v___x_3035_;
v___y_2981_ = v___y_3013_;
v___y_2982_ = v___y_3014_;
v___y_2983_ = v_xs_x27_3031_;
v___y_2984_ = v_info_3025_;
v___y_2985_ = v_xs_x27_3039_;
v___y_2986_ = v_info_3040_;
v___y_2987_ = v_kind_3018_;
v___y_2988_ = v_info_3032_;
v___y_2989_ = v_info_3017_;
v___y_2990_ = v_kind_3026_;
v___y_2991_ = v_kind_3041_;
v___y_2992_ = v___x_3044_;
goto v___jp_2958_;
}
else
{
size_t v___x_3047_; size_t v___x_3048_; lean_object* v___x_3049_; 
v___x_3047_ = ((size_t)0ULL);
v___x_3048_ = lean_usize_of_nat(v___x_3043_);
v___x_3049_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4(v_args_3042_, v___x_3047_, v___x_3048_, v___x_3044_);
lean_dec_ref(v_args_3042_);
v___y_2959_ = v___y_2995_;
v___y_2960_ = v___y_2996_;
v___y_2961_ = v___y_2997_;
v___y_2962_ = v___y_2998_;
v___y_2963_ = v___y_2999_;
v___y_2964_ = v___y_3000_;
v___y_2965_ = v___y_3001_;
v___y_2966_ = v___y_3002_;
v___y_2967_ = v___y_3003_;
v___y_2968_ = v___y_3004_;
v___y_2969_ = v___y_3005_;
v___y_2970_ = v___y_3006_;
v___y_2971_ = v___y_3009_;
v___y_2972_ = v___y_3008_;
v___y_2973_ = v___y_3007_;
v___y_2974_ = v___y_3010_;
v___y_2975_ = v___y_3016_;
v___y_2976_ = v___y_3011_;
v___y_2977_ = v_kind_3033_;
v___y_2978_ = v_xs_x27_3024_;
v___y_2979_ = v___y_3012_;
v___y_2980_ = v___x_3035_;
v___y_2981_ = v___y_3013_;
v___y_2982_ = v___y_3014_;
v___y_2983_ = v_xs_x27_3031_;
v___y_2984_ = v_info_3025_;
v___y_2985_ = v_xs_x27_3039_;
v___y_2986_ = v_info_3040_;
v___y_2987_ = v_kind_3018_;
v___y_2988_ = v_info_3032_;
v___y_2989_ = v_info_3017_;
v___y_2990_ = v_kind_3026_;
v___y_2991_ = v_kind_3041_;
v___y_2992_ = v___x_3049_;
goto v___jp_2958_;
}
}
else
{
size_t v___x_3050_; size_t v___x_3051_; lean_object* v___x_3052_; 
v___x_3050_ = ((size_t)0ULL);
v___x_3051_ = lean_usize_of_nat(v___x_3043_);
v___x_3052_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_CodeAction_casesExpand_spec__4(v_args_3042_, v___x_3050_, v___x_3051_, v___x_3044_);
lean_dec_ref(v_args_3042_);
v___y_2959_ = v___y_2995_;
v___y_2960_ = v___y_2996_;
v___y_2961_ = v___y_2997_;
v___y_2962_ = v___y_2998_;
v___y_2963_ = v___y_2999_;
v___y_2964_ = v___y_3000_;
v___y_2965_ = v___y_3001_;
v___y_2966_ = v___y_3002_;
v___y_2967_ = v___y_3003_;
v___y_2968_ = v___y_3004_;
v___y_2969_ = v___y_3005_;
v___y_2970_ = v___y_3006_;
v___y_2971_ = v___y_3009_;
v___y_2972_ = v___y_3008_;
v___y_2973_ = v___y_3007_;
v___y_2974_ = v___y_3010_;
v___y_2975_ = v___y_3016_;
v___y_2976_ = v___y_3011_;
v___y_2977_ = v_kind_3033_;
v___y_2978_ = v_xs_x27_3024_;
v___y_2979_ = v___y_3012_;
v___y_2980_ = v___x_3035_;
v___y_2981_ = v___y_3013_;
v___y_2982_ = v___y_3014_;
v___y_2983_ = v_xs_x27_3031_;
v___y_2984_ = v_info_3025_;
v___y_2985_ = v_xs_x27_3039_;
v___y_2986_ = v_info_3040_;
v___y_2987_ = v_kind_3018_;
v___y_2988_ = v_info_3032_;
v___y_2989_ = v_info_3017_;
v___y_2990_ = v_kind_3026_;
v___y_2991_ = v_kind_3041_;
v___y_2992_ = v___x_3052_;
goto v___jp_2958_;
}
}
}
else
{
v___y_2924_ = v___y_2995_;
v___y_2925_ = v___y_2996_;
v___y_2926_ = v___y_2997_;
v___y_2927_ = v___y_2998_;
v___y_2928_ = v___y_2999_;
v___y_2929_ = v___y_3000_;
v___y_2930_ = v___y_3001_;
v___y_2931_ = v___y_3002_;
v___y_2932_ = v___y_3003_;
v___y_2933_ = v___y_3004_;
v___y_2934_ = v___y_3005_;
v___y_2935_ = v___y_3006_;
v___y_2936_ = v___y_3007_;
v___y_2937_ = v___y_3008_;
v___y_2938_ = v___y_3009_;
v___y_2939_ = v___y_3010_;
v___y_2940_ = v___y_3016_;
v___y_2941_ = v___y_3011_;
v___y_2942_ = v_kind_3033_;
v___y_2943_ = v_xs_x27_3024_;
v___y_2944_ = v___y_3012_;
v___y_2945_ = v___x_3035_;
v___y_2946_ = v___y_3013_;
v___y_2947_ = v___y_3014_;
v___y_2948_ = v_xs_x27_3031_;
v___y_2949_ = v_info_3025_;
v___y_2950_ = v_xs_x27_3039_;
v___y_2951_ = v_kind_3018_;
v___y_2952_ = v_info_3032_;
v___y_2953_ = v_kind_3026_;
v___y_2954_ = v_info_3017_;
v___y_2955_ = v_v_3038_;
goto v___jp_2923_;
}
}
}
else
{
v___y_2893_ = v___y_2995_;
v___y_2894_ = v___y_2996_;
v___y_2895_ = v___y_2997_;
v___y_2896_ = v___y_2998_;
v___y_2897_ = v___y_2999_;
v___y_2898_ = v___y_3000_;
v___y_2899_ = v___y_3001_;
v___y_2900_ = v___y_3002_;
v___y_2901_ = v___y_3003_;
v___y_2902_ = v___y_3004_;
v___y_2903_ = v___y_3005_;
v___y_2904_ = v___y_3006_;
v___y_2905_ = v___y_3009_;
v___y_2906_ = v___y_3008_;
v___y_2907_ = v___y_3007_;
v___y_2908_ = v___y_3010_;
v___y_2909_ = v___y_3016_;
v___y_2910_ = v___y_3011_;
v___y_2911_ = v_xs_x27_3024_;
v___y_2912_ = v___y_3012_;
v___y_2913_ = v_kind_3018_;
v___y_2914_ = v___y_3013_;
v___y_2915_ = v___y_3014_;
v___y_2916_ = v_xs_x27_3031_;
v___y_2917_ = v_info_3017_;
v___y_2918_ = v_kind_3026_;
v___y_2919_ = v_info_3025_;
v___y_2920_ = v_v_3030_;
goto v___jp_2892_;
}
}
}
else
{
v___y_2866_ = v___y_2995_;
v___y_2867_ = v___y_2996_;
v___y_2868_ = v___y_2997_;
v___y_2869_ = v___y_2998_;
v___y_2870_ = v___y_2999_;
v___y_2871_ = v___y_3000_;
v___y_2872_ = v___y_3001_;
v___y_2873_ = v___y_3002_;
v___y_2874_ = v___y_3003_;
v___y_2875_ = v___y_3004_;
v___y_2876_ = v___y_3005_;
v___y_2877_ = v___y_3006_;
v___y_2878_ = v___y_3007_;
v___y_2879_ = v___y_3008_;
v___y_2880_ = v___y_3009_;
v___y_2881_ = v___y_3010_;
v___y_2882_ = v___y_3011_;
v___y_2883_ = v___y_3016_;
v___y_2884_ = v___y_3012_;
v___y_2885_ = v_xs_x27_3024_;
v___y_2886_ = v_kind_3018_;
v___y_2887_ = v___y_3014_;
v___y_2888_ = v_info_3017_;
v___y_2889_ = v_v_3022_;
goto v___jp_2865_;
}
}
}
else
{
v___y_2841_ = v___y_2995_;
v___y_2842_ = v___y_2996_;
v___y_2843_ = v___y_2997_;
v___y_2844_ = v___y_2998_;
v___y_2845_ = v___y_2999_;
v___y_2846_ = v___y_3000_;
v___y_2847_ = v___y_3001_;
v___y_2848_ = v___y_3002_;
v___y_2849_ = v___y_3003_;
v___y_2850_ = v___y_3004_;
v___y_2851_ = v___y_3005_;
v___y_2852_ = v___y_3006_;
v___y_2853_ = v___y_3009_;
v___y_2854_ = v___y_3008_;
v___y_2855_ = v___y_3007_;
v___y_2856_ = v___y_3010_;
v___y_2857_ = v___y_3011_;
v___y_2858_ = v___y_3012_;
v___y_2859_ = v___y_3014_;
v___y_2860_ = v___y_3015_;
goto v___jp_2840_;
}
}
v___jp_3053_:
{
if (v___y_3074_ == 0)
{
lean_object* v___x_3076_; 
v___x_3076_ = lean_unsigned_to_nat(3u);
v___y_2995_ = v___y_3054_;
v___y_2996_ = v___y_3055_;
v___y_2997_ = v___y_3056_;
v___y_2998_ = v___y_3057_;
v___y_2999_ = v___y_3058_;
v___y_3000_ = v___y_3059_;
v___y_3001_ = v___y_3060_;
v___y_3002_ = v___y_3061_;
v___y_3003_ = v___y_3062_;
v___y_3004_ = v___y_3063_;
v___y_3005_ = v___y_3064_;
v___y_3006_ = v___y_3065_;
v___y_3007_ = v___y_3068_;
v___y_3008_ = v___y_3067_;
v___y_3009_ = v___y_3066_;
v___y_3010_ = v___y_3069_;
v___y_3011_ = v___y_3070_;
v___y_3012_ = v___y_3071_;
v___y_3013_ = v___y_3072_;
v___y_3014_ = v___y_3073_;
v___y_3015_ = v___y_3075_;
v___y_3016_ = v___x_3076_;
goto v___jp_2994_;
}
else
{
lean_object* v___x_3077_; 
v___x_3077_ = lean_unsigned_to_nat(4u);
v___y_2995_ = v___y_3054_;
v___y_2996_ = v___y_3055_;
v___y_2997_ = v___y_3056_;
v___y_2998_ = v___y_3057_;
v___y_2999_ = v___y_3058_;
v___y_3000_ = v___y_3059_;
v___y_3001_ = v___y_3060_;
v___y_3002_ = v___y_3061_;
v___y_3003_ = v___y_3062_;
v___y_3004_ = v___y_3063_;
v___y_3005_ = v___y_3064_;
v___y_3006_ = v___y_3065_;
v___y_3007_ = v___y_3068_;
v___y_3008_ = v___y_3067_;
v___y_3009_ = v___y_3066_;
v___y_3010_ = v___y_3069_;
v___y_3011_ = v___y_3070_;
v___y_3012_ = v___y_3071_;
v___y_3013_ = v___y_3072_;
v___y_3014_ = v___y_3073_;
v___y_3015_ = v___y_3075_;
v___y_3016_ = v___x_3077_;
goto v___jp_2994_;
}
}
v___jp_3078_:
{
if (lean_obj_tag(v___y_3095_) == 0)
{
if (v___y_3096_ == 0)
{
v___y_2841_ = v___y_3079_;
v___y_2842_ = v___y_3080_;
v___y_2843_ = v___y_3081_;
v___y_2844_ = v___y_3082_;
v___y_2845_ = v___y_3083_;
v___y_2846_ = v___y_3084_;
v___y_2847_ = v___y_3085_;
v___y_2848_ = v___y_3086_;
v___y_2849_ = v___y_3087_;
v___y_2850_ = v___y_3088_;
v___y_2851_ = v___y_3089_;
v___y_2852_ = v___y_3090_;
v___y_2853_ = v___y_3093_;
v___y_2854_ = v___y_3092_;
v___y_2855_ = v___y_3091_;
v___y_2856_ = v___y_3094_;
v___y_2857_ = v___y_3096_;
v___y_2858_ = v___y_3097_;
v___y_2859_ = v___y_3100_;
v___y_2860_ = v___y_3101_;
goto v___jp_2840_;
}
else
{
v___y_3054_ = v___y_3079_;
v___y_3055_ = v___y_3080_;
v___y_3056_ = v___y_3081_;
v___y_3057_ = v___y_3082_;
v___y_3058_ = v___y_3083_;
v___y_3059_ = v___y_3084_;
v___y_3060_ = v___y_3085_;
v___y_3061_ = v___y_3086_;
v___y_3062_ = v___y_3087_;
v___y_3063_ = v___y_3088_;
v___y_3064_ = v___y_3089_;
v___y_3065_ = v___y_3090_;
v___y_3066_ = v___y_3093_;
v___y_3067_ = v___y_3092_;
v___y_3068_ = v___y_3091_;
v___y_3069_ = v___y_3094_;
v___y_3070_ = v___y_3096_;
v___y_3071_ = v___y_3097_;
v___y_3072_ = v___y_3098_;
v___y_3073_ = v___y_3100_;
v___y_3074_ = v___y_3099_;
v___y_3075_ = v___y_3101_;
goto v___jp_3053_;
}
}
else
{
lean_dec_ref_known(v___y_3095_, 1);
v___y_3054_ = v___y_3079_;
v___y_3055_ = v___y_3080_;
v___y_3056_ = v___y_3081_;
v___y_3057_ = v___y_3082_;
v___y_3058_ = v___y_3083_;
v___y_3059_ = v___y_3084_;
v___y_3060_ = v___y_3085_;
v___y_3061_ = v___y_3086_;
v___y_3062_ = v___y_3087_;
v___y_3063_ = v___y_3088_;
v___y_3064_ = v___y_3089_;
v___y_3065_ = v___y_3090_;
v___y_3066_ = v___y_3093_;
v___y_3067_ = v___y_3092_;
v___y_3068_ = v___y_3091_;
v___y_3069_ = v___y_3094_;
v___y_3070_ = v___y_3096_;
v___y_3071_ = v___y_3097_;
v___y_3072_ = v___y_3098_;
v___y_3073_ = v___y_3100_;
v___y_3074_ = v___y_3099_;
v___y_3075_ = v___y_3101_;
goto v___jp_3053_;
}
}
v___jp_3102_:
{
lean_object* v_source_3126_; lean_object* v___x_3127_; lean_object* v_fst_3128_; lean_object* v___x_3130_; uint8_t v_isShared_3131_; uint8_t v_isSharedCheck_3139_; 
v_source_3126_ = lean_ctor_get(v___y_3120_, 0);
lean_inc_ref_n(v_source_3126_, 2);
v___x_3127_ = lp_batteries_Lean_findIndentAndIsStart(v_source_3126_, v___y_3118_);
lean_dec(v___y_3118_);
v_fst_3128_ = lean_ctor_get(v___x_3127_, 0);
v_isSharedCheck_3139_ = !lean_is_exclusive(v___x_3127_);
if (v_isSharedCheck_3139_ == 0)
{
lean_object* v_unused_3140_; 
v_unused_3140_ = lean_ctor_get(v___x_3127_, 1);
lean_dec(v_unused_3140_);
v___x_3130_ = v___x_3127_;
v_isShared_3131_ = v_isSharedCheck_3139_;
goto v_resetjp_3129_;
}
else
{
lean_inc(v_fst_3128_);
lean_dec(v___x_3127_);
v___x_3130_ = lean_box(0);
v_isShared_3131_ = v_isSharedCheck_3139_;
goto v_resetjp_3129_;
}
v_resetjp_3129_:
{
lean_object* v___x_3132_; lean_object* v___x_3133_; lean_object* v___x_3134_; 
lean_inc_ref(v___y_3120_);
v___x_3132_ = l_Lean_FileMap_utf8PosToLspPos(v___y_3120_, v___y_3125_);
lean_dec(v___y_3125_);
v___x_3133_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4));
v___x_3134_ = lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(v_fst_3128_, v___x_3133_);
if (lean_obj_tag(v___y_3116_) == 0)
{
if (v___y_3119_ == 0)
{
lean_object* v___x_3135_; lean_object* v___x_3137_; 
lean_dec(v___y_3124_);
lean_dec_ref(v___y_3120_);
lean_dec(v___y_3117_);
v___x_3135_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_casesExpand___redArg___closed__0));
lean_inc_ref(v___x_3132_);
if (v_isShared_3131_ == 0)
{
lean_ctor_set(v___x_3130_, 1, v___x_3135_);
lean_ctor_set(v___x_3130_, 0, v___x_3132_);
v___x_3137_ = v___x_3130_;
goto v_reusejp_3136_;
}
else
{
lean_object* v_reuseFailAlloc_3138_; 
v_reuseFailAlloc_3138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3138_, 0, v___x_3132_);
lean_ctor_set(v_reuseFailAlloc_3138_, 1, v___x_3135_);
v___x_3137_ = v_reuseFailAlloc_3138_;
goto v_reusejp_3136_;
}
v_reusejp_3136_:
{
v___y_2791_ = v___y_3103_;
v___y_2792_ = v___y_3104_;
v___y_2793_ = v___y_3105_;
v___y_2794_ = v___y_3106_;
v___y_2795_ = v___x_3134_;
v___y_2796_ = v___y_3107_;
v___y_2797_ = v___y_3108_;
v___y_2798_ = v___y_3109_;
v___y_2799_ = v___y_3110_;
v___y_2800_ = v___x_3132_;
v___y_2801_ = v___y_3111_;
v___y_2802_ = v___y_3112_;
v___y_2803_ = v_source_3126_;
v___y_2804_ = v___y_3113_;
v___y_2805_ = v___y_3114_;
v___y_2806_ = v___y_3115_;
v___y_2807_ = v___y_3123_;
v___y_2808_ = v___x_3137_;
goto v___jp_2790_;
}
}
else
{
lean_del_object(v___x_3130_);
v___y_3079_ = v___y_3103_;
v___y_3080_ = v___y_3104_;
v___y_3081_ = v___y_3105_;
v___y_3082_ = v___y_3106_;
v___y_3083_ = v___x_3134_;
v___y_3084_ = v___y_3107_;
v___y_3085_ = v___y_3108_;
v___y_3086_ = v___y_3109_;
v___y_3087_ = v___y_3110_;
v___y_3088_ = v___x_3132_;
v___y_3089_ = v___y_3111_;
v___y_3090_ = v___y_3112_;
v___y_3091_ = v___y_3114_;
v___y_3092_ = v___y_3113_;
v___y_3093_ = v_source_3126_;
v___y_3094_ = v___y_3115_;
v___y_3095_ = v___y_3117_;
v___y_3096_ = v___y_3119_;
v___y_3097_ = v___y_3120_;
v___y_3098_ = v___y_3121_;
v___y_3099_ = v___y_3122_;
v___y_3100_ = v___y_3123_;
v___y_3101_ = v___y_3124_;
goto v___jp_3078_;
}
}
else
{
lean_dec_ref_known(v___y_3116_, 1);
lean_del_object(v___x_3130_);
v___y_3079_ = v___y_3103_;
v___y_3080_ = v___y_3104_;
v___y_3081_ = v___y_3105_;
v___y_3082_ = v___y_3106_;
v___y_3083_ = v___x_3134_;
v___y_3084_ = v___y_3107_;
v___y_3085_ = v___y_3108_;
v___y_3086_ = v___y_3109_;
v___y_3087_ = v___y_3110_;
v___y_3088_ = v___x_3132_;
v___y_3089_ = v___y_3111_;
v___y_3090_ = v___y_3112_;
v___y_3091_ = v___y_3114_;
v___y_3092_ = v___y_3113_;
v___y_3093_ = v_source_3126_;
v___y_3094_ = v___y_3115_;
v___y_3095_ = v___y_3117_;
v___y_3096_ = v___y_3119_;
v___y_3097_ = v___y_3120_;
v___y_3098_ = v___y_3121_;
v___y_3099_ = v___y_3122_;
v___y_3100_ = v___y_3123_;
v___y_3101_ = v___y_3124_;
goto v___jp_3078_;
}
}
}
v___jp_3141_:
{
lean_object* v___x_3142_; lean_object* v___x_3143_; 
v___x_3142_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3143_, 0, v___x_3142_);
return v___x_3143_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___redArg___boxed(lean_object* v_snap_3434_, lean_object* v_ctx_3435_, lean_object* v_node_3436_, lean_object* v_a_3437_, lean_object* v_a_3438_){
_start:
{
lean_object* v_res_3439_; 
v_res_3439_ = lp_batteries_Batteries_CodeAction_casesExpand___redArg(v_snap_3434_, v_ctx_3435_, v_node_3436_, v_a_3437_);
lean_dec_ref(v_a_3437_);
return v_res_3439_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand(lean_object* v_x_3440_, lean_object* v_snap_3441_, lean_object* v_ctx_3442_, lean_object* v_x_3443_, lean_object* v_node_3444_, lean_object* v_a_3445_){
_start:
{
lean_object* v___x_3447_; 
v___x_3447_ = lp_batteries_Batteries_CodeAction_casesExpand___redArg(v_snap_3441_, v_ctx_3442_, v_node_3444_, v_a_3445_);
return v___x_3447_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_casesExpand___boxed(lean_object* v_x_3448_, lean_object* v_snap_3449_, lean_object* v_ctx_3450_, lean_object* v_x_3451_, lean_object* v_node_3452_, lean_object* v_a_3453_, lean_object* v_a_3454_){
_start:
{
lean_object* v_res_3455_; 
v_res_3455_ = lp_batteries_Batteries_CodeAction_casesExpand(v_x_3448_, v_snap_3449_, v_ctx_3450_, v_x_3451_, v_node_3452_, v_a_3453_);
lean_dec_ref(v_a_3453_);
lean_dec(v_x_3451_);
lean_dec_ref(v_x_3448_);
return v_res_3455_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7(lean_object* v_00_u03b1_3456_, lean_object* v_msg_3457_, lean_object* v___y_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_, lean_object* v___y_3461_){
_start:
{
lean_object* v___x_3463_; 
v___x_3463_ = lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___redArg(v_msg_3457_, v___y_3458_, v___y_3459_, v___y_3460_, v___y_3461_);
return v___x_3463_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7___boxed(lean_object* v_00_u03b1_3464_, lean_object* v_msg_3465_, lean_object* v___y_3466_, lean_object* v___y_3467_, lean_object* v___y_3468_, lean_object* v___y_3469_, lean_object* v___y_3470_){
_start:
{
lean_object* v_res_3471_; 
v_res_3471_ = lp_batteries_Lean_throwError___at___00Batteries_CodeAction_casesExpand_spec__7(v_00_u03b1_3464_, v_msg_3465_, v___y_3466_, v___y_3467_, v___y_3468_, v___y_3469_);
lean_dec(v___y_3469_);
lean_dec_ref(v___y_3468_);
lean_dec(v___y_3467_);
lean_dec_ref(v___y_3466_);
return v_res_3471_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11(lean_object* v_as_3472_, size_t v_sz_3473_, size_t v_i_3474_, lean_object* v_b_3475_, lean_object* v___y_3476_){
_start:
{
lean_object* v___x_3478_; 
v___x_3478_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___redArg(v_as_3472_, v_sz_3473_, v_i_3474_, v_b_3475_);
return v___x_3478_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11___boxed(lean_object* v_as_3479_, lean_object* v_sz_3480_, lean_object* v_i_3481_, lean_object* v_b_3482_, lean_object* v___y_3483_, lean_object* v___y_3484_){
_start:
{
size_t v_sz_boxed_3485_; size_t v_i_boxed_3486_; lean_object* v_res_3487_; 
v_sz_boxed_3485_ = lean_unbox_usize(v_sz_3480_);
lean_dec(v_sz_3480_);
v_i_boxed_3486_ = lean_unbox_usize(v_i_3481_);
lean_dec(v_i_3481_);
v_res_3487_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_CodeAction_casesExpand_spec__11(v_as_3479_, v_sz_boxed_3485_, v_i_boxed_3486_, v_b_3482_, v___y_3483_);
lean_dec_ref(v___y_3483_);
lean_dec_ref(v_as_3479_);
return v_res_3487_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9(lean_object* v_00_u03b1_3488_, lean_object* v_constName_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_, lean_object* v___y_3493_){
_start:
{
lean_object* v___x_3495_; 
v___x_3495_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___redArg(v_constName_3489_, v___y_3490_, v___y_3491_, v___y_3492_, v___y_3493_);
return v___x_3495_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9___boxed(lean_object* v_00_u03b1_3496_, lean_object* v_constName_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_, lean_object* v___y_3502_){
_start:
{
lean_object* v_res_3503_; 
v_res_3503_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9(v_00_u03b1_3496_, v_constName_3497_, v___y_3498_, v___y_3499_, v___y_3500_, v___y_3501_);
lean_dec(v___y_3501_);
lean_dec_ref(v___y_3500_);
lean_dec(v___y_3499_);
lean_dec_ref(v___y_3498_);
return v_res_3503_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11(lean_object* v_00_u03b1_3504_, lean_object* v_ref_3505_, lean_object* v_constName_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_, lean_object* v___y_3510_){
_start:
{
lean_object* v___x_3512_; 
v___x_3512_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___redArg(v_ref_3505_, v_constName_3506_, v___y_3507_, v___y_3508_, v___y_3509_, v___y_3510_);
return v___x_3512_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11___boxed(lean_object* v_00_u03b1_3513_, lean_object* v_ref_3514_, lean_object* v_constName_3515_, lean_object* v___y_3516_, lean_object* v___y_3517_, lean_object* v___y_3518_, lean_object* v___y_3519_, lean_object* v___y_3520_){
_start:
{
lean_object* v_res_3521_; 
v_res_3521_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11(v_00_u03b1_3513_, v_ref_3514_, v_constName_3515_, v___y_3516_, v___y_3517_, v___y_3518_, v___y_3519_);
lean_dec(v___y_3519_);
lean_dec_ref(v___y_3518_);
lean_dec(v___y_3517_);
lean_dec_ref(v___y_3516_);
lean_dec(v_ref_3514_);
return v_res_3521_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17(lean_object* v_00_u03b1_3522_, lean_object* v_ref_3523_, lean_object* v_msg_3524_, lean_object* v_declHint_3525_, lean_object* v___y_3526_, lean_object* v___y_3527_, lean_object* v___y_3528_, lean_object* v___y_3529_){
_start:
{
lean_object* v___x_3531_; 
v___x_3531_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___redArg(v_ref_3523_, v_msg_3524_, v_declHint_3525_, v___y_3526_, v___y_3527_, v___y_3528_, v___y_3529_);
return v___x_3531_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17___boxed(lean_object* v_00_u03b1_3532_, lean_object* v_ref_3533_, lean_object* v_msg_3534_, lean_object* v_declHint_3535_, lean_object* v___y_3536_, lean_object* v___y_3537_, lean_object* v___y_3538_, lean_object* v___y_3539_, lean_object* v___y_3540_){
_start:
{
lean_object* v_res_3541_; 
v_res_3541_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17(v_00_u03b1_3532_, v_ref_3533_, v_msg_3534_, v_declHint_3535_, v___y_3536_, v___y_3537_, v___y_3538_, v___y_3539_);
lean_dec(v___y_3539_);
lean_dec_ref(v___y_3538_);
lean_dec(v___y_3537_);
lean_dec_ref(v___y_3536_);
lean_dec(v_ref_3533_);
return v_res_3541_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19(lean_object* v_msg_3542_, lean_object* v_declHint_3543_, lean_object* v___y_3544_, lean_object* v___y_3545_, lean_object* v___y_3546_, lean_object* v___y_3547_){
_start:
{
lean_object* v___x_3549_; 
v___x_3549_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___redArg(v_msg_3542_, v_declHint_3543_, v___y_3547_);
return v___x_3549_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19___boxed(lean_object* v_msg_3550_, lean_object* v_declHint_3551_, lean_object* v___y_3552_, lean_object* v___y_3553_, lean_object* v___y_3554_, lean_object* v___y_3555_, lean_object* v___y_3556_){
_start:
{
lean_object* v_res_3557_; 
v_res_3557_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__18_spec__19(v_msg_3550_, v_declHint_3551_, v___y_3552_, v___y_3553_, v___y_3554_, v___y_3555_);
lean_dec(v___y_3555_);
lean_dec_ref(v___y_3554_);
lean_dec(v___y_3553_);
lean_dec_ref(v___y_3552_);
return v_res_3557_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19(lean_object* v_00_u03b1_3558_, lean_object* v_ref_3559_, lean_object* v_msg_3560_, lean_object* v___y_3561_, lean_object* v___y_3562_, lean_object* v___y_3563_, lean_object* v___y_3564_){
_start:
{
lean_object* v___x_3566_; 
v___x_3566_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___redArg(v_ref_3559_, v_msg_3560_, v___y_3561_, v___y_3562_, v___y_3563_, v___y_3564_);
return v___x_3566_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19___boxed(lean_object* v_00_u03b1_3567_, lean_object* v_ref_3568_, lean_object* v_msg_3569_, lean_object* v___y_3570_, lean_object* v___y_3571_, lean_object* v___y_3572_, lean_object* v___y_3573_, lean_object* v___y_3574_){
_start:
{
lean_object* v_res_3575_; 
v_res_3575_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Batteries_CodeAction_casesExpand_spec__8_spec__9_spec__11_spec__17_spec__19(v_00_u03b1_3567_, v_ref_3568_, v_msg_3569_, v___y_3570_, v___y_3571_, v___y_3572_, v___y_3573_);
lean_dec(v___y_3573_);
lean_dec_ref(v___y_3572_);
lean_dec(v___y_3571_);
lean_dec_ref(v___y_3570_);
lean_dec(v_ref_3568_);
return v_res_3575_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg(lean_object* v___x_3577_, lean_object* v_as_x27_3578_, lean_object* v_b_3579_){
_start:
{
if (lean_obj_tag(v_as_x27_3578_) == 0)
{
lean_object* v___x_3581_; 
v___x_3581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3581_, 0, v_b_3579_);
return v___x_3581_;
}
else
{
lean_object* v_tail_3582_; lean_object* v___x_3583_; lean_object* v___x_3584_; lean_object* v___x_3585_; 
v_tail_3582_ = lean_ctor_get(v_as_x27_3578_, 1);
v___x_3583_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___closed__0));
v___x_3584_ = lean_string_append(v_b_3579_, v___x_3577_);
v___x_3585_ = lean_string_append(v___x_3584_, v___x_3583_);
v_as_x27_3578_ = v_tail_3582_;
v_b_3579_ = v___x_3585_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___boxed(lean_object* v___x_3587_, lean_object* v_as_x27_3588_, lean_object* v_b_3589_, lean_object* v___y_3590_){
_start:
{
lean_object* v_res_3591_; 
v_res_3591_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg(v___x_3587_, v_as_x27_3588_, v_b_3589_);
lean_dec(v_as_x27_3588_);
lean_dec_ref(v___x_3587_);
return v_res_3591_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___lam__0(lean_object* v_goals_3592_, lean_object* v___x_3593_, lean_object* v_a_3594_, lean_object* v_text_3595_, lean_object* v___x_3596_, lean_object* v___x_3597_, lean_object* v___x_3598_, lean_object* v___x_3599_, lean_object* v___x_3600_, lean_object* v___x_3601_, lean_object* v___x_3602_, lean_object* v___x_3603_, lean_object* v_fst_3604_, lean_object* v___x_3605_, lean_object* v_i_3606_, uint8_t v___x_3607_, lean_object* v_params_3608_, lean_object* v___x_3609_, lean_object* v___x_3610_, lean_object* v___x_3611_){
_start:
{
lean_object* v_range_3614_; lean_object* v_newText_3615_; lean_object* v___y_3636_; lean_object* v___y_3637_; lean_object* v___y_3638_; uint8_t v___y_3639_; lean_object* v___y_3642_; lean_object* v___x_3654_; lean_object* v___x_3655_; lean_object* v___x_3656_; uint8_t v___x_3657_; 
v___x_3654_ = l_Lean_Syntax_getArgs(v_fst_3604_);
v___x_3655_ = lean_nat_mul(v___x_3605_, v_i_3606_);
v___x_3656_ = lean_array_get_size(v___x_3654_);
v___x_3657_ = lean_nat_dec_lt(v___x_3655_, v___x_3656_);
if (v___x_3657_ == 0)
{
lean_dec_ref(v___x_3654_);
lean_dec_ref(v___x_3611_);
if (lean_obj_tag(v_fst_3604_) == 1)
{
lean_object* v_info_3658_; lean_object* v_kind_3659_; lean_object* v_args_3660_; lean_object* v___x_3662_; uint8_t v_isShared_3663_; uint8_t v_isSharedCheck_3669_; 
v_info_3658_ = lean_ctor_get(v_fst_3604_, 0);
v_kind_3659_ = lean_ctor_get(v_fst_3604_, 1);
v_args_3660_ = lean_ctor_get(v_fst_3604_, 2);
v_isSharedCheck_3669_ = !lean_is_exclusive(v_fst_3604_);
if (v_isSharedCheck_3669_ == 0)
{
v___x_3662_ = v_fst_3604_;
v_isShared_3663_ = v_isSharedCheck_3669_;
goto v_resetjp_3661_;
}
else
{
lean_inc(v_args_3660_);
lean_inc(v_kind_3659_);
lean_inc(v_info_3658_);
lean_dec(v_fst_3604_);
v___x_3662_ = lean_box(0);
v_isShared_3663_ = v_isSharedCheck_3669_;
goto v_resetjp_3661_;
}
v_resetjp_3661_:
{
lean_object* v___x_3664_; lean_object* v___x_3665_; lean_object* v___x_3667_; 
v___x_3664_ = l_Array_toSubarray___redArg(v_args_3660_, v___x_3610_, v___x_3655_);
v___x_3665_ = l_Subarray_copy___redArg(v___x_3664_);
if (v_isShared_3663_ == 0)
{
lean_ctor_set(v___x_3662_, 2, v___x_3665_);
v___x_3667_ = v___x_3662_;
goto v_reusejp_3666_;
}
else
{
lean_object* v_reuseFailAlloc_3668_; 
v_reuseFailAlloc_3668_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3668_, 0, v_info_3658_);
lean_ctor_set(v_reuseFailAlloc_3668_, 1, v_kind_3659_);
lean_ctor_set(v_reuseFailAlloc_3668_, 2, v___x_3665_);
v___x_3667_ = v_reuseFailAlloc_3668_;
goto v_reusejp_3666_;
}
v_reusejp_3666_:
{
v___y_3642_ = v___x_3667_;
goto v___jp_3641_;
}
}
}
else
{
lean_dec(v___x_3655_);
lean_dec(v___x_3610_);
v___y_3642_ = v_fst_3604_;
goto v___jp_3641_;
}
}
else
{
lean_object* v___x_3670_; lean_object* v___x_3671_; 
lean_dec(v___x_3610_);
lean_dec_ref(v_params_3608_);
lean_dec(v_fst_3604_);
v___x_3670_ = lean_array_fget(v___x_3654_, v___x_3655_);
lean_dec(v___x_3655_);
lean_dec_ref(v___x_3654_);
v___x_3671_ = l_Lean_Syntax_getRange_x3f(v___x_3670_, v___x_3607_);
lean_dec(v___x_3670_);
if (lean_obj_tag(v___x_3671_) == 1)
{
lean_object* v_val_3672_; 
lean_dec_ref(v___x_3609_);
v_val_3672_ = lean_ctor_get(v___x_3671_, 0);
lean_inc(v_val_3672_);
lean_dec_ref_known(v___x_3671_, 1);
v_range_3614_ = v_val_3672_;
v_newText_3615_ = v___x_3611_;
goto v___jp_3613_;
}
else
{
lean_object* v___x_3673_; 
lean_dec(v___x_3671_);
lean_dec_ref(v___x_3611_);
lean_dec(v___x_3603_);
lean_dec(v___x_3602_);
lean_dec(v___x_3601_);
lean_dec(v___x_3600_);
lean_dec(v___x_3599_);
lean_dec(v___x_3598_);
lean_dec_ref(v___x_3597_);
lean_dec(v___x_3596_);
lean_dec_ref(v_text_3595_);
lean_dec_ref(v_a_3594_);
lean_dec_ref(v___x_3593_);
v___x_3673_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3673_, 0, v___x_3609_);
return v___x_3673_;
}
}
v___jp_3613_:
{
lean_object* v___x_3616_; lean_object* v___x_3617_; lean_object* v___x_3618_; lean_object* v___x_3619_; lean_object* v_a_3620_; lean_object* v___x_3622_; uint8_t v_isShared_3623_; uint8_t v_isSharedCheck_3634_; 
v___x_3616_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg___closed__0));
v___x_3617_ = lean_string_append(v_newText_3615_, v___x_3616_);
v___x_3618_ = l_List_tail_x21___redArg(v_goals_3592_);
v___x_3619_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg(v___x_3593_, v___x_3618_, v___x_3617_);
lean_dec(v___x_3618_);
lean_dec_ref(v___x_3593_);
v_a_3620_ = lean_ctor_get(v___x_3619_, 0);
v_isSharedCheck_3634_ = !lean_is_exclusive(v___x_3619_);
if (v_isSharedCheck_3634_ == 0)
{
v___x_3622_ = v___x_3619_;
v_isShared_3623_ = v_isSharedCheck_3634_;
goto v_resetjp_3621_;
}
else
{
lean_inc(v_a_3620_);
lean_dec(v___x_3619_);
v___x_3622_ = lean_box(0);
v_isShared_3623_ = v_isSharedCheck_3634_;
goto v_resetjp_3621_;
}
v_resetjp_3621_:
{
lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___x_3626_; lean_object* v___x_3627_; lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v___x_3632_; 
v___x_3624_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_3594_);
v___x_3625_ = l_Lean_FileMap_utf8RangeToLspRange(v_text_3595_, v_range_3614_);
v___x_3626_ = lean_box(0);
lean_inc_n(v___x_3596_, 2);
v___x_3627_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3627_, 0, v___x_3625_);
lean_ctor_set(v___x_3627_, 1, v_a_3620_);
lean_ctor_set(v___x_3627_, 2, v___x_3626_);
lean_ctor_set(v___x_3627_, 3, v___x_3596_);
v___x_3628_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___x_3624_, v___x_3627_);
v___x_3629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3629_, 0, v___x_3628_);
v___x_3630_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3630_, 0, v___x_3596_);
lean_ctor_set(v___x_3630_, 1, v___x_3596_);
lean_ctor_set(v___x_3630_, 2, v___x_3597_);
lean_ctor_set(v___x_3630_, 3, v___x_3598_);
lean_ctor_set(v___x_3630_, 4, v___x_3599_);
lean_ctor_set(v___x_3630_, 5, v___x_3600_);
lean_ctor_set(v___x_3630_, 6, v___x_3601_);
lean_ctor_set(v___x_3630_, 7, v___x_3629_);
lean_ctor_set(v___x_3630_, 8, v___x_3602_);
lean_ctor_set(v___x_3630_, 9, v___x_3603_);
if (v_isShared_3623_ == 0)
{
lean_ctor_set(v___x_3622_, 0, v___x_3630_);
v___x_3632_ = v___x_3622_;
goto v_reusejp_3631_;
}
else
{
lean_object* v_reuseFailAlloc_3633_; 
v_reuseFailAlloc_3633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3633_, 0, v___x_3630_);
v___x_3632_ = v_reuseFailAlloc_3633_;
goto v_reusejp_3631_;
}
v_reusejp_3631_:
{
return v___x_3632_;
}
}
}
v___jp_3635_:
{
if (v___y_3639_ == 0)
{
lean_dec(v___y_3637_);
lean_dec(v___y_3636_);
lean_inc_ref(v___x_3593_);
v_range_3614_ = v___y_3638_;
v_newText_3615_ = v___x_3593_;
goto v___jp_3613_;
}
else
{
lean_object* v___x_3640_; 
lean_dec_ref(v___y_3638_);
v___x_3640_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3640_, 0, v___y_3637_);
lean_ctor_set(v___x_3640_, 1, v___y_3636_);
lean_inc_ref(v___x_3593_);
v_range_3614_ = v___x_3640_;
v_newText_3615_ = v___x_3593_;
goto v___jp_3613_;
}
}
v___jp_3641_:
{
lean_object* v___x_3643_; 
v___x_3643_ = l_Lean_Syntax_getTailPos_x3f(v___y_3642_, v___x_3607_);
if (lean_obj_tag(v___x_3643_) == 1)
{
lean_object* v_val_3644_; lean_object* v___x_3645_; lean_object* v_range_3646_; lean_object* v_end_3647_; lean_object* v___x_3648_; uint8_t v___x_3649_; 
lean_dec_ref(v___x_3609_);
v_val_3644_ = lean_ctor_get(v___x_3643_, 0);
lean_inc_n(v_val_3644_, 3);
lean_dec_ref_known(v___x_3643_, 1);
v___x_3645_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3645_, 0, v_val_3644_);
lean_ctor_set(v___x_3645_, 1, v_val_3644_);
v_range_3646_ = lean_ctor_get(v_params_3608_, 3);
lean_inc_ref(v_range_3646_);
lean_dec_ref(v_params_3608_);
v_end_3647_ = lean_ctor_get(v_range_3646_, 1);
lean_inc_ref(v_end_3647_);
lean_dec_ref(v_range_3646_);
v___x_3648_ = l_Lean_FileMap_lspPosToUtf8Pos(v_text_3595_, v_end_3647_);
v___x_3649_ = lean_nat_dec_le(v_val_3644_, v___x_3648_);
if (v___x_3649_ == 0)
{
lean_dec(v___y_3642_);
v___y_3636_ = v___x_3648_;
v___y_3637_ = v_val_3644_;
v___y_3638_ = v___x_3645_;
v___y_3639_ = v___x_3649_;
goto v___jp_3635_;
}
else
{
lean_object* v___x_3650_; lean_object* v___x_3651_; uint8_t v___x_3652_; 
v___x_3650_ = l_Lean_Syntax_getTrailingSize(v___y_3642_);
lean_dec(v___y_3642_);
v___x_3651_ = lean_nat_add(v_val_3644_, v___x_3650_);
lean_dec(v___x_3650_);
v___x_3652_ = lean_nat_dec_le(v___x_3648_, v___x_3651_);
lean_dec(v___x_3651_);
v___y_3636_ = v___x_3648_;
v___y_3637_ = v_val_3644_;
v___y_3638_ = v___x_3645_;
v___y_3639_ = v___x_3652_;
goto v___jp_3635_;
}
}
else
{
lean_object* v___x_3653_; 
lean_dec(v___x_3643_);
lean_dec(v___y_3642_);
lean_dec_ref(v_params_3608_);
lean_dec(v___x_3603_);
lean_dec(v___x_3602_);
lean_dec(v___x_3601_);
lean_dec(v___x_3600_);
lean_dec(v___x_3599_);
lean_dec(v___x_3598_);
lean_dec_ref(v___x_3597_);
lean_dec(v___x_3596_);
lean_dec_ref(v_text_3595_);
lean_dec_ref(v_a_3594_);
lean_dec_ref(v___x_3593_);
v___x_3653_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3653_, 0, v___x_3609_);
return v___x_3653_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___lam__0___boxed(lean_object** _args){
lean_object* v_goals_3674_ = _args[0];
lean_object* v___x_3675_ = _args[1];
lean_object* v_a_3676_ = _args[2];
lean_object* v_text_3677_ = _args[3];
lean_object* v___x_3678_ = _args[4];
lean_object* v___x_3679_ = _args[5];
lean_object* v___x_3680_ = _args[6];
lean_object* v___x_3681_ = _args[7];
lean_object* v___x_3682_ = _args[8];
lean_object* v___x_3683_ = _args[9];
lean_object* v___x_3684_ = _args[10];
lean_object* v___x_3685_ = _args[11];
lean_object* v_fst_3686_ = _args[12];
lean_object* v___x_3687_ = _args[13];
lean_object* v_i_3688_ = _args[14];
lean_object* v___x_3689_ = _args[15];
lean_object* v_params_3690_ = _args[16];
lean_object* v___x_3691_ = _args[17];
lean_object* v___x_3692_ = _args[18];
lean_object* v___x_3693_ = _args[19];
lean_object* v___y_3694_ = _args[20];
_start:
{
uint8_t v___x_2915__boxed_3695_; lean_object* v_res_3696_; 
v___x_2915__boxed_3695_ = lean_unbox(v___x_3689_);
v_res_3696_ = lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___lam__0(v_goals_3674_, v___x_3675_, v_a_3676_, v_text_3677_, v___x_3678_, v___x_3679_, v___x_3680_, v___x_3681_, v___x_3682_, v___x_3683_, v___x_3684_, v___x_3685_, v_fst_3686_, v___x_3687_, v_i_3688_, v___x_2915__boxed_3695_, v_params_3690_, v___x_3691_, v___x_3692_, v___x_3693_);
lean_dec(v_i_3688_);
lean_dec(v___x_3687_);
lean_dec(v_goals_3674_);
return v_res_3696_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_elem___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__1(lean_object* v_a_3697_, lean_object* v_x_3698_){
_start:
{
if (lean_obj_tag(v_x_3698_) == 0)
{
uint8_t v___x_3699_; 
v___x_3699_ = 0;
return v___x_3699_;
}
else
{
lean_object* v_head_3700_; lean_object* v_tail_3701_; uint8_t v___x_3702_; 
v_head_3700_ = lean_ctor_get(v_x_3698_, 0);
v_tail_3701_ = lean_ctor_get(v_x_3698_, 1);
v___x_3702_ = lean_name_eq(v_a_3697_, v_head_3700_);
if (v___x_3702_ == 0)
{
v_x_3698_ = v_tail_3701_;
goto _start;
}
else
{
return v___x_3702_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_elem___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__1___boxed(lean_object* v_a_3704_, lean_object* v_x_3705_){
_start:
{
uint8_t v_res_3706_; lean_object* v_r_3707_; 
v_res_3706_ = lp_batteries_List_elem___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__1(v_a_3704_, v_x_3705_);
lean_dec(v_x_3705_);
lean_dec(v_a_3704_);
v_r_3707_ = lean_box(v_res_3706_);
return v_r_3707_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore(lean_object* v_params_3742_, lean_object* v_i_3743_, lean_object* v_stk_3744_, lean_object* v_goals_3745_, lean_object* v_a_3746_){
_start:
{
lean_object* v___x_3748_; lean_object* v___x_3749_; uint8_t v___x_3750_; 
v___x_3748_ = lean_unsigned_to_nat(1u);
v___x_3749_ = l_List_lengthTR___redArg(v_goals_3745_);
v___x_3750_ = lean_nat_dec_lt(v___x_3748_, v___x_3749_);
lean_dec(v___x_3749_);
if (v___x_3750_ == 0)
{
lean_object* v___x_3751_; lean_object* v___x_3752_; 
lean_dec(v_goals_3745_);
lean_dec(v_i_3743_);
lean_dec_ref(v_params_3742_);
v___x_3751_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3752_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3752_, 0, v___x_3751_);
return v___x_3752_;
}
else
{
lean_object* v___x_3753_; lean_object* v___x_3754_; lean_object* v___x_3755_; lean_object* v_fst_3756_; lean_object* v___x_3757_; lean_object* v___x_3758_; lean_object* v___x_3759_; lean_object* v___y_3761_; uint8_t v___y_3808_; lean_object* v_nargs_3811_; uint8_t v___x_3812_; 
v___x_3753_ = lean_unsigned_to_nat(0u);
v___x_3754_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__0));
v___x_3755_ = l_List_head_x21___redArg(v___x_3754_, v_stk_3744_);
v_fst_3756_ = lean_ctor_get(v___x_3755_, 0);
lean_inc(v_fst_3756_);
lean_dec(v___x_3755_);
v___x_3757_ = l_Lean_Syntax_getNumArgs(v_fst_3756_);
v___x_3758_ = lean_nat_add(v___x_3757_, v___x_3748_);
lean_dec(v___x_3757_);
v___x_3759_ = lean_unsigned_to_nat(2u);
v_nargs_3811_ = lean_nat_shiftr(v___x_3758_, v___x_3748_);
lean_dec(v___x_3758_);
v___x_3812_ = lean_nat_dec_eq(v_i_3743_, v_nargs_3811_);
if (v___x_3812_ == 0)
{
lean_object* v___x_3813_; uint8_t v___x_3814_; 
v___x_3813_ = lean_nat_add(v_i_3743_, v___x_3748_);
v___x_3814_ = lean_nat_dec_eq(v___x_3813_, v_nargs_3811_);
lean_dec(v_nargs_3811_);
lean_dec(v___x_3813_);
if (v___x_3814_ == 0)
{
v___y_3808_ = v___x_3814_;
goto v___jp_3807_;
}
else
{
lean_object* v___x_3815_; lean_object* v___x_3816_; lean_object* v___x_3817_; lean_object* v___x_3818_; uint8_t v___x_3819_; 
v___x_3815_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__10));
v___x_3816_ = lean_nat_mul(v___x_3759_, v_i_3743_);
v___x_3817_ = l_Lean_Syntax_getArg(v_fst_3756_, v___x_3816_);
lean_dec(v___x_3816_);
v___x_3818_ = l_Lean_Syntax_getKind(v___x_3817_);
v___x_3819_ = lp_batteries_List_elem___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__1(v___x_3818_, v___x_3815_);
lean_dec(v___x_3818_);
v___y_3808_ = v___x_3819_;
goto v___jp_3807_;
}
}
else
{
lean_dec(v_nargs_3811_);
v___y_3761_ = v_a_3746_;
goto v___jp_3760_;
}
v___jp_3760_:
{
lean_object* v___x_3762_; lean_object* v___x_3763_; 
v___x_3762_ = l_Lean_Syntax_getArg(v_fst_3756_, v___x_3753_);
v___x_3763_ = l_Lean_Syntax_getPos_x3f(v___x_3762_, v___x_3750_);
lean_dec(v___x_3762_);
if (lean_obj_tag(v___x_3763_) == 1)
{
lean_object* v_val_3764_; lean_object* v___x_3766_; uint8_t v_isShared_3767_; uint8_t v_isSharedCheck_3804_; 
v_val_3764_ = lean_ctor_get(v___x_3763_, 0);
v_isSharedCheck_3804_ = !lean_is_exclusive(v___x_3763_);
if (v_isSharedCheck_3804_ == 0)
{
v___x_3766_ = v___x_3763_;
v_isShared_3767_ = v_isSharedCheck_3804_;
goto v_resetjp_3765_;
}
else
{
lean_inc(v_val_3764_);
lean_dec(v___x_3763_);
v___x_3766_ = lean_box(0);
v_isShared_3767_ = v_isSharedCheck_3804_;
goto v_resetjp_3765_;
}
v_resetjp_3765_:
{
lean_object* v___x_3768_; lean_object* v_a_3769_; lean_object* v___x_3771_; uint8_t v_isShared_3772_; uint8_t v_isSharedCheck_3803_; 
v___x_3768_ = lp_batteries_Lean_Server_RequestM_readDoc___at___00Batteries_CodeAction_instanceStub_spec__0(v___y_3761_);
v_a_3769_ = lean_ctor_get(v___x_3768_, 0);
v_isSharedCheck_3803_ = !lean_is_exclusive(v___x_3768_);
if (v_isSharedCheck_3803_ == 0)
{
v___x_3771_ = v___x_3768_;
v_isShared_3772_ = v_isSharedCheck_3803_;
goto v_resetjp_3770_;
}
else
{
lean_inc(v_a_3769_);
lean_dec(v___x_3768_);
v___x_3771_ = lean_box(0);
v_isShared_3772_ = v_isSharedCheck_3803_;
goto v_resetjp_3770_;
}
v_resetjp_3770_:
{
lean_object* v___x_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; lean_object* v___x_3776_; lean_object* v_toEditableDocumentCore_3777_; lean_object* v_meta_3778_; lean_object* v_text_3779_; lean_object* v___x_3780_; lean_object* v_column_3781_; lean_object* v___x_3783_; uint8_t v_isShared_3784_; uint8_t v_isSharedCheck_3801_; 
v___x_3773_ = lean_box(0);
v___x_3774_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__1));
v___x_3775_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__3));
v___x_3776_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___closed__2));
v_toEditableDocumentCore_3777_ = lean_ctor_get(v_a_3769_, 0);
v_meta_3778_ = lean_ctor_get(v_toEditableDocumentCore_3777_, 0);
v_text_3779_ = lean_ctor_get(v_meta_3778_, 3);
lean_inc_ref_n(v_text_3779_, 2);
v___x_3780_ = l_Lean_FileMap_toPosition(v_text_3779_, v_val_3764_);
lean_dec(v_val_3764_);
v_column_3781_ = lean_ctor_get(v___x_3780_, 1);
v_isSharedCheck_3801_ = !lean_is_exclusive(v___x_3780_);
if (v_isSharedCheck_3801_ == 0)
{
lean_object* v_unused_3802_; 
v_unused_3802_ = lean_ctor_get(v___x_3780_, 0);
lean_dec(v_unused_3802_);
v___x_3783_ = v___x_3780_;
v_isShared_3784_ = v_isSharedCheck_3801_;
goto v_resetjp_3782_;
}
else
{
lean_inc(v_column_3781_);
lean_dec(v___x_3780_);
v___x_3783_ = lean_box(0);
v_isShared_3784_ = v_isSharedCheck_3801_;
goto v_resetjp_3782_;
}
v_resetjp_3782_:
{
lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; lean_object* v___x_3788_; lean_object* v___y_3789_; lean_object* v___x_3791_; 
v___x_3785_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__0___closed__4));
v___x_3786_ = lp_batteries___private_Init_Data_Nat_Basic_0__Nat_repeatTR_loop___at___00Batteries_CodeAction_instanceStub_spec__2(v_column_3781_, v___x_3785_);
v___x_3787_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___lam__1___closed__10));
v___x_3788_ = lean_box(v___x_3750_);
v___y_3789_ = lean_alloc_closure((void*)(lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___lam__0___boxed), 21, 20);
lean_closure_set(v___y_3789_, 0, v_goals_3745_);
lean_closure_set(v___y_3789_, 1, v___x_3786_);
lean_closure_set(v___y_3789_, 2, v_a_3769_);
lean_closure_set(v___y_3789_, 3, v_text_3779_);
lean_closure_set(v___y_3789_, 4, v___x_3773_);
lean_closure_set(v___y_3789_, 5, v___x_3774_);
lean_closure_set(v___y_3789_, 6, v___x_3775_);
lean_closure_set(v___y_3789_, 7, v___x_3773_);
lean_closure_set(v___y_3789_, 8, v___x_3773_);
lean_closure_set(v___y_3789_, 9, v___x_3773_);
lean_closure_set(v___y_3789_, 10, v___x_3773_);
lean_closure_set(v___y_3789_, 11, v___x_3773_);
lean_closure_set(v___y_3789_, 12, v_fst_3756_);
lean_closure_set(v___y_3789_, 13, v___x_3759_);
lean_closure_set(v___y_3789_, 14, v_i_3743_);
lean_closure_set(v___y_3789_, 15, v___x_3788_);
lean_closure_set(v___y_3789_, 16, v_params_3742_);
lean_closure_set(v___y_3789_, 17, v___x_3776_);
lean_closure_set(v___y_3789_, 18, v___x_3753_);
lean_closure_set(v___y_3789_, 19, v___x_3787_);
if (v_isShared_3767_ == 0)
{
lean_ctor_set(v___x_3766_, 0, v___y_3789_);
v___x_3791_ = v___x_3766_;
goto v_reusejp_3790_;
}
else
{
lean_object* v_reuseFailAlloc_3800_; 
v_reuseFailAlloc_3800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3800_, 0, v___y_3789_);
v___x_3791_ = v_reuseFailAlloc_3800_;
goto v_reusejp_3790_;
}
v_reusejp_3790_:
{
lean_object* v___x_3793_; 
if (v_isShared_3784_ == 0)
{
lean_ctor_set(v___x_3783_, 1, v___x_3791_);
lean_ctor_set(v___x_3783_, 0, v___x_3776_);
v___x_3793_ = v___x_3783_;
goto v_reusejp_3792_;
}
else
{
lean_object* v_reuseFailAlloc_3799_; 
v_reuseFailAlloc_3799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3799_, 0, v___x_3776_);
lean_ctor_set(v_reuseFailAlloc_3799_, 1, v___x_3791_);
v___x_3793_ = v_reuseFailAlloc_3799_;
goto v_reusejp_3792_;
}
v_reusejp_3792_:
{
lean_object* v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3797_; 
v___x_3794_ = lean_mk_empty_array_with_capacity(v___x_3748_);
v___x_3795_ = lean_array_push(v___x_3794_, v___x_3793_);
if (v_isShared_3772_ == 0)
{
lean_ctor_set(v___x_3771_, 0, v___x_3795_);
v___x_3797_ = v___x_3771_;
goto v_reusejp_3796_;
}
else
{
lean_object* v_reuseFailAlloc_3798_; 
v_reuseFailAlloc_3798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3798_, 0, v___x_3795_);
v___x_3797_ = v_reuseFailAlloc_3798_;
goto v_reusejp_3796_;
}
v_reusejp_3796_:
{
return v___x_3797_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3805_; lean_object* v___x_3806_; 
lean_dec(v___x_3763_);
lean_dec(v_fst_3756_);
lean_dec(v_goals_3745_);
lean_dec(v_i_3743_);
lean_dec_ref(v_params_3742_);
v___x_3805_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3806_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3806_, 0, v___x_3805_);
return v___x_3806_;
}
}
v___jp_3807_:
{
if (v___y_3808_ == 0)
{
lean_object* v___x_3809_; lean_object* v___x_3810_; 
lean_dec(v_fst_3756_);
lean_dec(v_goals_3745_);
lean_dec(v_i_3743_);
lean_dec_ref(v_params_3742_);
v___x_3809_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3810_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3810_, 0, v___x_3809_);
return v___x_3810_;
}
else
{
v___y_3761_ = v_a_3746_;
goto v___jp_3760_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsActionCore___boxed(lean_object* v_params_3820_, lean_object* v_i_3821_, lean_object* v_stk_3822_, lean_object* v_goals_3823_, lean_object* v_a_3824_, lean_object* v_a_3825_){
_start:
{
lean_object* v_res_3826_; 
v_res_3826_ = lp_batteries_Batteries_CodeAction_addSubgoalsActionCore(v_params_3820_, v_i_3821_, v_stk_3822_, v_goals_3823_, v_a_3824_);
lean_dec_ref(v_a_3824_);
lean_dec(v_stk_3822_);
return v_res_3826_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0(lean_object* v___x_3827_, lean_object* v_as_3828_, lean_object* v_as_x27_3829_, lean_object* v_b_3830_, lean_object* v_a_3831_){
_start:
{
lean_object* v___x_3833_; 
v___x_3833_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___redArg(v___x_3827_, v_as_x27_3829_, v_b_3830_);
return v___x_3833_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0___boxed(lean_object* v___x_3834_, lean_object* v_as_3835_, lean_object* v_as_x27_3836_, lean_object* v_b_3837_, lean_object* v_a_3838_, lean_object* v___y_3839_){
_start:
{
lean_object* v_res_3840_; 
v_res_3840_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_CodeAction_addSubgoalsActionCore_spec__0(v___x_3834_, v_as_3835_, v_as_x27_3836_, v_b_3837_, v_a_3838_);
lean_dec(v_as_x27_3836_);
lean_dec(v_as_3835_);
lean_dec_ref(v___x_3834_);
return v_res_3840_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___redArg(lean_object* v_params_3841_, lean_object* v_i_3842_, lean_object* v_stk_3843_, lean_object* v_goals_3844_, lean_object* v_a_3845_){
_start:
{
lean_object* v___x_3847_; 
v___x_3847_ = lp_batteries_Batteries_CodeAction_addSubgoalsActionCore(v_params_3841_, v_i_3842_, v_stk_3843_, v_goals_3844_, v_a_3845_);
return v___x_3847_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___redArg___boxed(lean_object* v_params_3848_, lean_object* v_i_3849_, lean_object* v_stk_3850_, lean_object* v_goals_3851_, lean_object* v_a_3852_, lean_object* v_a_3853_){
_start:
{
lean_object* v_res_3854_; 
v_res_3854_ = lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___redArg(v_params_3848_, v_i_3849_, v_stk_3850_, v_goals_3851_, v_a_3852_);
lean_dec_ref(v_a_3852_);
lean_dec(v_stk_3850_);
return v_res_3854_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction(lean_object* v_params_3855_, lean_object* v_x_3856_, lean_object* v_x_3857_, lean_object* v_i_3858_, lean_object* v_stk_3859_, lean_object* v_goals_3860_, lean_object* v_a_3861_){
_start:
{
lean_object* v___x_3863_; 
v___x_3863_ = lp_batteries_Batteries_CodeAction_addSubgoalsActionCore(v_params_3855_, v_i_3858_, v_stk_3859_, v_goals_3860_, v_a_3861_);
return v___x_3863_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction___boxed(lean_object* v_params_3864_, lean_object* v_x_3865_, lean_object* v_x_3866_, lean_object* v_i_3867_, lean_object* v_stk_3868_, lean_object* v_goals_3869_, lean_object* v_a_3870_, lean_object* v_a_3871_){
_start:
{
lean_object* v_res_3872_; 
v_res_3872_ = lp_batteries_Batteries_CodeAction_addSubgoalsSeqAction(v_params_3864_, v_x_3865_, v_x_3866_, v_i_3867_, v_stk_3868_, v_goals_3869_, v_a_3870_);
lean_dec_ref(v_a_3870_);
lean_dec(v_stk_3868_);
lean_dec_ref(v_x_3866_);
lean_dec_ref(v_x_3865_);
return v_res_3872_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg(lean_object* v_params_3879_, lean_object* v_stk_3880_, lean_object* v_node_3881_, lean_object* v_a_3882_){
_start:
{
if (lean_obj_tag(v_stk_3880_) == 1)
{
lean_object* v_tail_3887_; 
v_tail_3887_ = lean_ctor_get(v_stk_3880_, 1);
lean_inc(v_tail_3887_);
lean_dec_ref_known(v_stk_3880_, 2);
if (lean_obj_tag(v_tail_3887_) == 1)
{
lean_object* v_head_3888_; lean_object* v_tail_3889_; 
v_head_3888_ = lean_ctor_get(v_tail_3887_, 0);
lean_inc(v_head_3888_);
v_tail_3889_ = lean_ctor_get(v_tail_3887_, 1);
lean_inc(v_tail_3889_);
lean_dec_ref_known(v_tail_3887_, 2);
if (lean_obj_tag(v_tail_3889_) == 1)
{
lean_object* v_tail_3890_; 
v_tail_3890_ = lean_ctor_get(v_tail_3889_, 1);
lean_inc(v_tail_3890_);
if (lean_obj_tag(v_tail_3890_) == 1)
{
if (lean_obj_tag(v_node_3881_) == 1)
{
lean_object* v_i_3891_; 
v_i_3891_ = lean_ctor_get(v_node_3881_, 0);
lean_inc_ref(v_i_3891_);
lean_dec_ref_known(v_node_3881_, 2);
if (lean_obj_tag(v_i_3891_) == 0)
{
lean_object* v_head_3892_; lean_object* v___x_3894_; uint8_t v_isShared_3895_; uint8_t v_isSharedCheck_3927_; 
v_head_3892_ = lean_ctor_get(v_tail_3890_, 0);
v_isSharedCheck_3927_ = !lean_is_exclusive(v_tail_3890_);
if (v_isSharedCheck_3927_ == 0)
{
lean_object* v_unused_3928_; 
v_unused_3928_ = lean_ctor_get(v_tail_3890_, 1);
lean_dec(v_unused_3928_);
v___x_3894_ = v_tail_3890_;
v_isShared_3895_ = v_isSharedCheck_3927_;
goto v_resetjp_3893_;
}
else
{
lean_inc(v_head_3892_);
lean_dec(v_tail_3890_);
v___x_3894_ = lean_box(0);
v_isShared_3895_ = v_isSharedCheck_3927_;
goto v_resetjp_3893_;
}
v_resetjp_3893_:
{
lean_object* v_fst_3896_; lean_object* v_snd_3897_; lean_object* v_i_3898_; lean_object* v___x_3900_; uint8_t v_isShared_3901_; uint8_t v_isSharedCheck_3926_; 
v_fst_3896_ = lean_ctor_get(v_head_3888_, 0);
lean_inc(v_fst_3896_);
v_snd_3897_ = lean_ctor_get(v_head_3888_, 1);
lean_inc(v_snd_3897_);
lean_dec(v_head_3888_);
v_i_3898_ = lean_ctor_get(v_i_3891_, 0);
v_isSharedCheck_3926_ = !lean_is_exclusive(v_i_3891_);
if (v_isSharedCheck_3926_ == 0)
{
v___x_3900_ = v_i_3891_;
v_isShared_3901_ = v_isSharedCheck_3926_;
goto v_resetjp_3899_;
}
else
{
lean_inc(v_i_3898_);
lean_dec(v_i_3891_);
v___x_3900_ = lean_box(0);
v_isShared_3901_ = v_isSharedCheck_3926_;
goto v_resetjp_3899_;
}
v_resetjp_3899_:
{
lean_object* v_fst_3902_; lean_object* v___x_3904_; uint8_t v_isShared_3905_; uint8_t v_isSharedCheck_3924_; 
v_fst_3902_ = lean_ctor_get(v_head_3892_, 0);
v_isSharedCheck_3924_ = !lean_is_exclusive(v_head_3892_);
if (v_isSharedCheck_3924_ == 0)
{
lean_object* v_unused_3925_; 
v_unused_3925_ = lean_ctor_get(v_head_3892_, 1);
lean_dec(v_unused_3925_);
v___x_3904_ = v_head_3892_;
v_isShared_3905_ = v_isSharedCheck_3924_;
goto v_resetjp_3903_;
}
else
{
lean_inc(v_fst_3902_);
lean_dec(v_head_3892_);
v___x_3904_ = lean_box(0);
v_isShared_3905_ = v_isSharedCheck_3924_;
goto v_resetjp_3903_;
}
v_resetjp_3903_:
{
lean_object* v___x_3906_; lean_object* v___x_3907_; uint8_t v___x_3908_; 
v___x_3906_ = l_Lean_Syntax_getKind(v_fst_3902_);
v___x_3907_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___closed__1));
v___x_3908_ = lean_name_eq(v___x_3906_, v___x_3907_);
lean_dec(v___x_3906_);
if (v___x_3908_ == 0)
{
lean_object* v___x_3909_; lean_object* v___x_3911_; 
lean_del_object(v___x_3904_);
lean_dec_ref(v_i_3898_);
lean_dec(v_snd_3897_);
lean_dec(v_fst_3896_);
lean_del_object(v___x_3894_);
lean_dec_ref_known(v_tail_3889_, 2);
lean_dec_ref(v_params_3879_);
v___x_3909_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
if (v_isShared_3901_ == 0)
{
lean_ctor_set(v___x_3900_, 0, v___x_3909_);
v___x_3911_ = v___x_3900_;
goto v_reusejp_3910_;
}
else
{
lean_object* v_reuseFailAlloc_3912_; 
v_reuseFailAlloc_3912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3912_, 0, v___x_3909_);
v___x_3911_ = v_reuseFailAlloc_3912_;
goto v_reusejp_3910_;
}
v_reusejp_3910_:
{
return v___x_3911_;
}
}
else
{
lean_object* v_goalsBefore_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; lean_object* v___x_3918_; 
lean_del_object(v___x_3900_);
v_goalsBefore_3913_ = lean_ctor_get(v_i_3898_, 2);
lean_inc(v_goalsBefore_3913_);
lean_dec_ref(v_i_3898_);
v___x_3914_ = lean_unsigned_to_nat(1u);
v___x_3915_ = lean_nat_shiftr(v_snd_3897_, v___x_3914_);
lean_dec(v_snd_3897_);
v___x_3916_ = lean_unsigned_to_nat(0u);
if (v_isShared_3905_ == 0)
{
lean_ctor_set(v___x_3904_, 1, v___x_3916_);
lean_ctor_set(v___x_3904_, 0, v_fst_3896_);
v___x_3918_ = v___x_3904_;
goto v_reusejp_3917_;
}
else
{
lean_object* v_reuseFailAlloc_3923_; 
v_reuseFailAlloc_3923_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3923_, 0, v_fst_3896_);
lean_ctor_set(v_reuseFailAlloc_3923_, 1, v___x_3916_);
v___x_3918_ = v_reuseFailAlloc_3923_;
goto v_reusejp_3917_;
}
v_reusejp_3917_:
{
lean_object* v___x_3920_; 
if (v_isShared_3895_ == 0)
{
lean_ctor_set(v___x_3894_, 1, v_tail_3889_);
lean_ctor_set(v___x_3894_, 0, v___x_3918_);
v___x_3920_ = v___x_3894_;
goto v_reusejp_3919_;
}
else
{
lean_object* v_reuseFailAlloc_3922_; 
v_reuseFailAlloc_3922_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3922_, 0, v___x_3918_);
lean_ctor_set(v_reuseFailAlloc_3922_, 1, v_tail_3889_);
v___x_3920_ = v_reuseFailAlloc_3922_;
goto v_reusejp_3919_;
}
v_reusejp_3919_:
{
lean_object* v___x_3921_; 
v___x_3921_ = lp_batteries_Batteries_CodeAction_addSubgoalsActionCore(v_params_3879_, v___x_3915_, v___x_3920_, v_goalsBefore_3913_, v_a_3882_);
lean_dec_ref(v___x_3920_);
return v___x_3921_;
}
}
}
}
}
}
}
else
{
lean_dec_ref(v_i_3891_);
lean_dec_ref_known(v_tail_3890_, 2);
lean_dec_ref_known(v_tail_3889_, 2);
lean_dec(v_head_3888_);
lean_dec_ref(v_params_3879_);
goto v___jp_3884_;
}
}
else
{
lean_dec_ref_known(v_tail_3890_, 2);
lean_dec_ref_known(v_tail_3889_, 2);
lean_dec(v_head_3888_);
lean_dec_ref(v_node_3881_);
lean_dec_ref(v_params_3879_);
goto v___jp_3884_;
}
}
else
{
lean_dec(v_tail_3890_);
lean_dec_ref_known(v_tail_3889_, 2);
lean_dec(v_head_3888_);
lean_dec_ref(v_node_3881_);
lean_dec_ref(v_params_3879_);
goto v___jp_3884_;
}
}
else
{
lean_dec(v_tail_3889_);
lean_dec(v_head_3888_);
lean_dec_ref(v_node_3881_);
lean_dec_ref(v_params_3879_);
goto v___jp_3884_;
}
}
else
{
lean_dec(v_tail_3887_);
lean_dec_ref(v_node_3881_);
lean_dec_ref(v_params_3879_);
goto v___jp_3884_;
}
}
else
{
lean_dec_ref(v_node_3881_);
lean_dec(v_stk_3880_);
lean_dec_ref(v_params_3879_);
goto v___jp_3884_;
}
v___jp_3884_:
{
lean_object* v___x_3885_; lean_object* v___x_3886_; 
v___x_3885_ = ((lean_object*)(lp_batteries_Batteries_CodeAction_instanceStub___redArg___closed__0));
v___x_3886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3886_, 0, v___x_3885_);
return v___x_3886_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg___boxed(lean_object* v_params_3929_, lean_object* v_stk_3930_, lean_object* v_node_3931_, lean_object* v_a_3932_, lean_object* v_a_3933_){
_start:
{
lean_object* v_res_3934_; 
v_res_3934_ = lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg(v_params_3929_, v_stk_3930_, v_node_3931_, v_a_3932_);
lean_dec_ref(v_a_3932_);
return v_res_3934_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction(lean_object* v_params_3935_, lean_object* v_x_3936_, lean_object* v_x_3937_, lean_object* v_stk_3938_, lean_object* v_node_3939_, lean_object* v_a_3940_){
_start:
{
lean_object* v___x_3942_; 
v___x_3942_ = lp_batteries_Batteries_CodeAction_addSubgoalsAction___redArg(v_params_3935_, v_stk_3938_, v_node_3939_, v_a_3940_);
return v___x_3942_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_CodeAction_addSubgoalsAction___boxed(lean_object* v_params_3943_, lean_object* v_x_3944_, lean_object* v_x_3945_, lean_object* v_stk_3946_, lean_object* v_node_3947_, lean_object* v_a_3948_, lean_object* v_a_3949_){
_start:
{
lean_object* v_res_3950_; 
v_res_3950_ = lp_batteries_Batteries_CodeAction_addSubgoalsAction(v_params_3943_, v_x_3944_, v_x_3945_, v_stk_3946_, v_node_3947_, v_a_3948_);
lean_dec_ref(v_a_3948_);
lean_dec_ref(v_x_3945_);
lean_dec_ref(v_x_3944_);
return v_res_3950_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_CodeAction_Misc(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Induction(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Position(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_CodeActions_Provider(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_CodeAction_Misc(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Position(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_CodeActions_Provider(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Induction(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Position(uint8_t builtin);
lean_object* initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
lean_object* initialize_Lean_Server_CodeActions_Provider(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_CodeAction_Misc(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Position(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_CodeActions_Provider(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_CodeAction_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_CodeAction_Misc(builtin);
}
#ifdef __cplusplus
}
#endif
