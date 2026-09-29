// Lean compiler output
// Module: Mathlib.Tactic.Widget.Calc
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Calc public meta import Lean.Meta.Tactic.TryThis public meta import Mathlib.Data.String.Defs public meta import Mathlib.Tactic.Widget.SelectPanelUtils public meta import Batteries.CodeAction.Attr public import Batteries.CodeAction.Attr public import Mathlib.Tactic.Widget.SelectPanelUtils public import ProofWidgets.Component.Basic public import ProofWidgets.Component.OfRpcMethod
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
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_getCalcRelation_x3f___redArg(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_FileMap_lspRangeOfStx_x3f(lean_object*, lean_object*, uint8_t);
uint64_t lean_string_hash(lean_object*);
lean_object* l_Lean_Lsp_instToJsonRange_toJson(lean_object*);
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_Widget_savePanelWidgetInfo(uint64_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_mkCalcStepViews(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Elab_Tactic_evalCalc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_insertMetaVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveGoal_dec_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_sanitizeNames(lean_object*, lean_object*);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink;
lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveGoal_enc_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_instToJsonGoalsLocation_toJson(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_FileMap_utf8PosToLspPos(lean_object*, lean_object*);
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Lsp_WorkspaceEdit_ofTextEdit(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_getGoalLocations(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lp_mathlib_String_renameMetaVar(lean_object*);
lean_object* lp_mathlib_String_replicate(lean_object*, uint32_t);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instToJsonPosition_toJson(lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instFromJsonPosition_fromJson(lean_object*);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_instFromJsonGoalsLocation_fromJson(lean_object*);
lean_object* l_Lean_Lsp_instFromJsonRange_fromJson(lean_object*);
lean_object* l_Lean_Json_getBool_x3f(lean_object*);
lean_object* l_Lean_Json_getNat_x3f(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00createCalc_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_createCalc___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "calc "};
static const lean_object* lp_mathlib_createCalc___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_createCalc___redArg___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_createCalc___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " := by sorry"};
static const lean_object* lp_mathlib_createCalc___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_createCalc___redArg___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_createCalc___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_createCalc___redArg___closed__0 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__0_value;
static const lean_string_object lp_mathlib_createCalc___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "Generate a calc block."};
static const lean_object* lp_mathlib_createCalc___redArg___closed__1 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__1_value;
static const lean_string_object lp_mathlib_createCalc___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "quickfix"};
static const lean_object* lp_mathlib_createCalc___redArg___closed__2 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_createCalc___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_createCalc___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_createCalc___redArg___closed__3 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_createCalc___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*10 + 0, .m_other = 10, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_createCalc___redArg___closed__1_value),((lean_object*)&lp_mathlib_createCalc___redArg___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_createCalc___redArg___closed__4 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_createCalc___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_createCalc___redArg___closed__5;
static lean_once_cell_t lp_mathlib_createCalc___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_createCalc___redArg___closed__6;
static lean_once_cell_t lp_mathlib_createCalc___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_createCalc___redArg___closed__7;
static lean_once_cell_t lp_mathlib_createCalc___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_createCalc___redArg___closed__8;
static lean_once_cell_t lp_mathlib_createCalc___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_createCalc___redArg___closed__9;
static const lean_string_object lp_mathlib_createCalc___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_createCalc___redArg___closed__10 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__10_value;
static const lean_string_object lp_mathlib_createCalc___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_createCalc___redArg___closed__11 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__11_value;
static const lean_string_object lp_mathlib_createCalc___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_createCalc___redArg___closed__12 = (const lean_object*)&lp_mathlib_createCalc___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_createCalc___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_createCalc___redArg___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_createCalc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_createCalc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__3___boxed(lean_object*);
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassCalcParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassCalcParams___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___closed__0 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__0_value;
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassCalcParams___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassCalcParams___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___closed__1 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__1_value;
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassCalcParams___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassCalcParams___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___closed__2 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__2_value;
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassCalcParams___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassCalcParams___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___closed__3 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__3_value;
static const lean_ctor_object lp_mathlib_instSelectInsertParamsClassCalcParams___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__0_value),((lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__1_value),((lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__2_value),((lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__3_value)}};
static const lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___closed__4 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassCalcParams___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pos"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goals"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "selectedLocations"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "replaceRange"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "isFirst"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "indent"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
LEAN_EXPORT lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_(lean_object*);
static const lean_closure_object lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
LEAN_EXPORT const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__spec__0(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_ = (const lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value;
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39____boxed(lean_object*);
static const lean_closure_object lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_ = (const lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value;
LEAN_EXPORT const lean_object* lp_mathlib_instToJsonRpcEncodablePacket_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_ = (const lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected JSON array, got '"};
static const lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instRpcEncodableCalcParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instRpcEncodableCalcParams___closed__0 = (const lean_object*)&lp_mathlib_instRpcEncodableCalcParams___closed__0_value;
static const lean_closure_object lp_mathlib_instRpcEncodableCalcParams___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instRpcEncodableCalcParams___closed__1 = (const lean_object*)&lp_mathlib_instRpcEncodableCalcParams___closed__1_value;
static const lean_ctor_object lp_mathlib_instRpcEncodableCalcParams___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instRpcEncodableCalcParams___closed__0_value),((lean_object*)&lp_mathlib_instRpcEncodableCalcParams___closed__1_value)}};
static const lean_object* lp_mathlib_instRpcEncodableCalcParams___closed__2 = (const lean_object*)&lp_mathlib_instRpcEncodableCalcParams___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_instRpcEncodableCalcParams = (const lean_object*)&lp_mathlib_instRpcEncodableCalcParams___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0___boxed(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 246}, .m_size = 2, .m_capacity = 2, .m_data = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00suggestSteps_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00suggestSteps_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_suggestSteps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_suggestSteps___closed__0 = (const lean_object*)&lp_mathlib_suggestSteps___closed__0_value;
static const lean_string_object lp_mathlib_suggestSteps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Create two new steps"};
static const lean_object* lp_mathlib_suggestSteps___closed__1 = (const lean_object*)&lp_mathlib_suggestSteps___closed__1_value;
static const lean_string_object lp_mathlib_suggestSteps___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Create a new step"};
static const lean_object* lp_mathlib_suggestSteps___closed__2 = (const lean_object*)&lp_mathlib_suggestSteps___closed__2_value;
static const lean_string_object lp_mathlib_suggestSteps___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "This should not happen"};
static const lean_object* lp_mathlib_suggestSteps___closed__3 = (const lean_object*)&lp_mathlib_suggestSteps___closed__3_value;
static const lean_string_object lp_mathlib_suggestSteps___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_ "};
static const lean_object* lp_mathlib_suggestSteps___closed__4 = (const lean_object*)&lp_mathlib_suggestSteps___closed__4_value;
static const lean_string_object lp_mathlib_suggestSteps___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = " := by sorry\n"};
static const lean_object* lp_mathlib_suggestSteps___closed__5 = (const lean_object*)&lp_mathlib_suggestSteps___closed__5_value;
static const lean_string_object lp_mathlib_suggestSteps___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "invalid 'calc' step, relation expected"};
static const lean_object* lp_mathlib_suggestSteps___closed__6 = (const lean_object*)&lp_mathlib_suggestSteps___closed__6_value;
static lean_once_cell_t lp_mathlib_suggestSteps___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_suggestSteps___closed__7;
static const lean_string_object lp_mathlib_suggestSteps___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "could not find relation symbol in "};
static const lean_object* lp_mathlib_suggestSteps___closed__8 = (const lean_object*)&lp_mathlib_suggestSteps___closed__8_value;
static lean_once_cell_t lp_mathlib_suggestSteps___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_suggestSteps___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_suggestSteps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_suggestSteps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__0 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__1 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__2 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__3 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__4 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__4_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__4_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__5 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__5_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__5_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__6 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__6_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__6_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__7 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__8 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ml1"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__9 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__9_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__9_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__10 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__10_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__11 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__11_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__11_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__12 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__12_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "There is no goal to solve!"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__13 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__13_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__13_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__14 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__14_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__14_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__15 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__15_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0_value),((lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__15_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__16 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__16_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " should be "};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__17 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__17_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "You should select only one sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__18 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__18_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__18_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__19 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__19_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__19_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__20 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__20_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0_value),((lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__20_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__21 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__21_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "in the main goal or its context."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__22 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__22_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "in the main goal."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__23 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__23_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "All selected sub-expressions"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__24 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__24_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "The selected sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__25 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__25_value;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_CalcPanel_rpc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_suggestSteps___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_CalcPanel_rpc___closed__0 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___closed__0_value;
static const lean_string_object lp_mathlib_CalcPanel_rpc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Please select subterms using Shift-click."};
static const lean_object* lp_mathlib_CalcPanel_rpc___closed__1 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___closed__1_value;
static const lean_string_object lp_mathlib_CalcPanel_rpc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 7, .m_data = "Calc 🔍️"};
static const lean_object* lp_mathlib_CalcPanel_rpc___closed__2 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_CalcPanel_rpc(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CalcPanel_rpc___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "CalcPanel"};
static const lean_object* lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__0 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__0_value;
static const lean_string_object lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rpc"};
static const lean_object* lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__1 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__1_value;
static const lean_ctor_object lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 245, 118, 247, 33, 213, 220, 162)}};
static const lean_ctor_object lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__2_value_aux_0),((lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(127, 81, 58, 177, 17, 223, 172, 243)}};
static const lean_object* lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__2 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__2_value;
static const lean_closure_object lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_CalcPanel_rpc___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__3 = (const lean_object*)&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__3_value;
static lean_once_cell_t lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_CalcPanel_rpc___rpc__wrapped;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_CalcPanel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3830, .m_capacity = 3830, .m_length = 3829, .m_data = "window;import{jsxs as e,jsx as t,Fragment as r}from\"react/jsx-runtime\";import*as n from\"react\";import{useRpcSession as o,EnvPosContext as a,useAsyncPersistent as i,mapRpcError as f,importWidgetModule as c}from\"@leanprover/infoview\";function u(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,\"default\")\?e.default:e}var s,l;var p=u(function(){if(l)return s;l=1;var e=\"undefined\"!=typeof Element,t=\"function\"==typeof Map,r=\"function\"==typeof Set,n=\"function\"==typeof ArrayBuffer&&!!ArrayBuffer.isView;function o(a,i){if(a===i)return!0;if(a&&i&&\"object\"==typeof a&&\"object\"==typeof i){if(a.constructor!==i.constructor)return!1;var f,c,u,s;if(Array.isArray(a)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(!o(a[c],i[c]))return!1;return!0}if(t&&a instanceof Map&&i instanceof Map){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;for(s=a.entries();!(c=s.next()).done;)if(!o(c.value[1],i.get(c.value[0])))return!1;return!0}if(r&&a instanceof Set&&i instanceof Set){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;return!0}if(n&&ArrayBuffer.isView(a)&&ArrayBuffer.isView(i)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(a[c]!==i[c])return!1;return!0}if(a.constructor===RegExp)return a.source===i.source&&a.flags===i.flags;if(a.valueOf!==Object.prototype.valueOf&&\"function\"==typeof a.valueOf&&\"function\"==typeof i.valueOf)return a.valueOf()===i.valueOf();if(a.toString!==Object.prototype.toString&&\"function\"==typeof a.toString&&\"function\"==typeof i.toString)return a.toString()===i.toString();if((f=(u=Object.keys(a)).length)!==Object.keys(i).length)return!1;for(c=f;0!==c--;)if(!Object.prototype.hasOwnProperty.call(i,u[c]))return!1;if(e&&a instanceof Element)return!1;for(c=f;0!==c--;)if((\"_owner\"!==u[c]&&\"__v\"!==u[c]&&\"__o\"!==u[c]||!a.$$typeof)&&!o(a[u[c]],i[u[c]]))return!1;return!0}return a!=a&&i!=i}return s=function(e,t){try{return o(e,t)}catch(e){if((e.message||\"\").match(/stack|recursion/i))return console.warn(\"react-fast-compare cannot handle circular refs\"),!1;throw e}}}());async function y(o,a,i){if(\"text\"in i)return t(r,{children:i.text});if(\"element\"in i){const[e,r,f]=i.element,c={};for(const[e,t]of r)c[e]=t;const u=await Promise.all(f.map(async e=>await y(o,a,e)));return\"hr\"===e\?t(\"hr\",{}):0===u.length\?n.createElement(e,c):n.createElement(e,c,u)}if(\"component\"in i){const[e,t,r,f]=i.component,u=await Promise.all(f.map(async e=>await y(o,a,e))),s={...r,pos:a},l=await c(o,a,e);if(!(t in l))throw new Error(`Module '${e}' does not export '${t}'`);return 0===u.length\?n.createElement(l[t],s):n.createElement(l[t],s,u)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(i)]})}function d({html:c}){const u=o(),s=n.useContext(a),l=i(()=>y(u,s,c),[u,s,c]);return\"resolved\"===l.state\?l.value:\"rejected\"===l.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",f(l.error).message]}):t(r,{})}const m=\"CalcPanel.rpc\",g='false';var w=n.memo(e=>{const a=o(),c=n.useRef({fn:()=>{}}),u=i(async()=>{if(c.current.fn(),\"true\"===g){const[t,r]=function(e,t,r){const n={fn:()=>{}};return[new Promise(async(o,a)=>{const i=await e.call(t,r),f=window.setInterval(async()=>{try{const t=await e.call(\"ProofWidgets.checkRequest\",i);if(\"running\"===t)return;window.clearInterval(f),o(t.done.result)}catch(e){window.clearInterval(f),a(e)}},100);n.fn=()=>{e.call(\"ProofWidgets.cancelRequest\",i)}}),n]}(a,m,e);return c.current=r,t}{const t=new AbortController,r=a.call(m,e,{abortSignal:t.signal});return c.current={fn:()=>t.abort()},r}},[a,e]);return n.useEffect(()=>()=>{c.current.fn()},[]),\"rejected\"===u.state\?t(\"p\",{style:{color:\"red\"},children:f(u.error).message}):\"loading\"===u.state\?t(r,{children:\"Loading..\"}):t(d,{html:u.value})},p);export{w as default};"};
static const lean_object* lp_mathlib_CalcPanel___closed__0 = (const lean_object*)&lp_mathlib_CalcPanel___closed__0_value;
static lean_once_cell_t lp_mathlib_CalcPanel___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib_CalcPanel___closed__1;
static lean_once_cell_t lp_mathlib_CalcPanel___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_CalcPanel___closed__2;
static const lean_string_object lp_mathlib_CalcPanel___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_mathlib_CalcPanel___closed__3 = (const lean_object*)&lp_mathlib_CalcPanel___closed__3_value;
static lean_once_cell_t lp_mathlib_CalcPanel___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_CalcPanel___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_CalcPanel;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticCalc\?"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(88, 16, 0, 123, 235, 26, 81, 50)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "calc\?"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__6_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "calcTactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "calc"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "calcSteps"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "calcFirstStep"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byTactic"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "by"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticSorry"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__15_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "sorry"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__17;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Create calc tactic:"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__18 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__18_value;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Cannot start a calculation here: the goal"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__19 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__19_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__20;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "\nis not a relation."};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__21 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__21_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__22;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg(lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(143, 188, 49, 237, 47, 139, 25, 127)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0(lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0(lean_object* v___y_1_){
_start:
{
lean_object* v_doc_3_; lean_object* v___x_4_; 
v_doc_3_ = lean_ctor_get(v___y_1_, 1);
lean_inc_ref(v_doc_3_);
v___x_4_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4_, 0, v_doc_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0___boxed(lean_object* v___y_5_, lean_object* v___y_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0(v___y_5_);
lean_dec_ref(v___y_5_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___redArg(lean_object* v_mvarId_8_, lean_object* v_x_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_8_, v_x_9_, v___y_10_, v___y_11_, v___y_12_, v___y_13_);
if (lean_obj_tag(v___x_15_) == 0)
{
lean_object* v_a_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_23_; 
v_a_16_ = lean_ctor_get(v___x_15_, 0);
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_23_ == 0)
{
v___x_18_ = v___x_15_;
v_isShared_19_ = v_isSharedCheck_23_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_a_16_);
lean_dec(v___x_15_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_23_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_21_; 
if (v_isShared_19_ == 0)
{
v___x_21_ = v___x_18_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v_a_16_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
else
{
lean_object* v_a_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_31_; 
v_a_24_ = lean_ctor_get(v___x_15_, 0);
v_isSharedCheck_31_ = !lean_is_exclusive(v___x_15_);
if (v_isSharedCheck_31_ == 0)
{
v___x_26_ = v___x_15_;
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_a_24_);
lean_dec(v___x_15_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_29_; 
if (v_isShared_27_ == 0)
{
v___x_29_ = v___x_26_;
goto v_reusejp_28_;
}
else
{
lean_object* v_reuseFailAlloc_30_; 
v_reuseFailAlloc_30_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_30_, 0, v_a_24_);
v___x_29_ = v_reuseFailAlloc_30_;
goto v_reusejp_28_;
}
v_reusejp_28_:
{
return v___x_29_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___redArg___boxed(lean_object* v_mvarId_32_, lean_object* v_x_33_, lean_object* v___y_34_, lean_object* v___y_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___redArg(v_mvarId_32_, v_x_33_, v___y_34_, v___y_35_, v___y_36_, v___y_37_);
lean_dec(v___y_37_);
lean_dec_ref(v___y_36_);
lean_dec(v___y_35_);
lean_dec_ref(v___y_34_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1(lean_object* v_00_u03b1_40_, lean_object* v_mvarId_41_, lean_object* v_x_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___redArg(v_mvarId_41_, v_x_42_, v___y_43_, v___y_44_, v___y_45_, v___y_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___boxed(lean_object* v_00_u03b1_49_, lean_object* v_mvarId_50_, lean_object* v_x_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1(v_00_u03b1_49_, v_mvarId_50_, v_x_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_);
lean_dec(v___y_55_);
lean_dec_ref(v___y_54_);
lean_dec(v___y_53_);
lean_dec_ref(v___y_52_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00createCalc_spec__2(lean_object* v_msg_58_){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_unsigned_to_nat(0u);
v___x_60_ = lean_panic_fn_borrowed(v___x_59_, v_msg_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__0(lean_object* v___x_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = l_Lean_MVarId_getType(v___x_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
if (lean_obj_tag(v___x_67_) == 0)
{
lean_object* v_a_68_; lean_object* v___x_69_; 
v_a_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc(v_a_68_);
lean_dec_ref_known(v___x_67_, 1);
v___x_69_ = l_Lean_Meta_ppExpr(v_a_68_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
return v___x_69_;
}
else
{
lean_object* v_a_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_77_; 
v_a_70_ = lean_ctor_get(v___x_67_, 0);
v_isSharedCheck_77_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_77_ == 0)
{
v___x_72_ = v___x_67_;
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_a_70_);
lean_dec(v___x_67_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_75_; 
if (v_isShared_73_ == 0)
{
v___x_75_ = v___x_72_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v_a_70_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__0___boxed(lean_object* v___x_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib_createCalc___redArg___lam__0(v___x_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__1(lean_object* v_ctx_87_, lean_object* v___x_88_, lean_object* v___x_89_, lean_object* v_a_90_, lean_object* v___x_91_, lean_object* v___x_92_, lean_object* v___x_93_, lean_object* v___x_94_, lean_object* v___x_95_, lean_object* v___x_96_, lean_object* v___x_97_, lean_object* v___x_98_, lean_object* v___x_99_, lean_object* v___x_100_, lean_object* v___x_101_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_ctx_87_, v___x_88_, v___x_89_);
if (lean_obj_tag(v___x_103_) == 0)
{
lean_object* v_a_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_124_; 
v_a_104_ = lean_ctor_get(v___x_103_, 0);
v_isSharedCheck_124_ = !lean_is_exclusive(v___x_103_);
if (v_isSharedCheck_124_ == 0)
{
v___x_106_ = v___x_103_;
v_isShared_107_ = v_isSharedCheck_124_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_a_104_);
lean_dec(v___x_103_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_124_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_122_; 
v___x_108_ = l_Lean_Server_FileWorker_EditableDocument_versionedIdentifier(v_a_90_);
v___x_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_91_);
lean_ctor_set(v___x_109_, 1, v___x_92_);
v___x_110_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__0));
v___x_111_ = l_Std_Format_defWidth;
lean_inc(v___x_93_);
v___x_112_ = l_Std_Format_pretty(v_a_104_, v___x_111_, v___x_93_, v___x_93_);
v___x_113_ = lean_string_append(v___x_110_, v___x_112_);
lean_dec_ref(v___x_112_);
v___x_114_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_115_ = lean_string_append(v___x_113_, v___x_114_);
v___x_116_ = lean_box(0);
lean_inc_n(v___x_94_, 2);
v___x_117_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_117_, 0, v___x_109_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
lean_ctor_set(v___x_117_, 2, v___x_116_);
lean_ctor_set(v___x_117_, 3, v___x_94_);
v___x_118_ = l_Lean_Lsp_WorkspaceEdit_ofTextEdit(v___x_108_, v___x_117_);
v___x_119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
v___x_120_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_120_, 0, v___x_94_);
lean_ctor_set(v___x_120_, 1, v___x_94_);
lean_ctor_set(v___x_120_, 2, v___x_95_);
lean_ctor_set(v___x_120_, 3, v___x_96_);
lean_ctor_set(v___x_120_, 4, v___x_97_);
lean_ctor_set(v___x_120_, 5, v___x_98_);
lean_ctor_set(v___x_120_, 6, v___x_99_);
lean_ctor_set(v___x_120_, 7, v___x_119_);
lean_ctor_set(v___x_120_, 8, v___x_100_);
lean_ctor_set(v___x_120_, 9, v___x_101_);
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 0, v___x_120_);
v___x_122_ = v___x_106_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v___x_120_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
}
else
{
lean_object* v_a_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_132_; 
lean_dec(v___x_101_);
lean_dec(v___x_100_);
lean_dec(v___x_99_);
lean_dec(v___x_98_);
lean_dec(v___x_97_);
lean_dec(v___x_96_);
lean_dec_ref(v___x_95_);
lean_dec(v___x_94_);
lean_dec(v___x_93_);
lean_dec_ref(v___x_92_);
lean_dec_ref(v___x_91_);
lean_dec_ref(v_a_90_);
v_a_125_ = lean_ctor_get(v___x_103_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_103_);
if (v_isSharedCheck_132_ == 0)
{
v___x_127_ = v___x_103_;
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_a_125_);
lean_dec(v___x_103_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_a_125_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
return v___x_130_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___lam__1___boxed(lean_object* v_ctx_133_, lean_object* v___x_134_, lean_object* v___x_135_, lean_object* v_a_136_, lean_object* v___x_137_, lean_object* v___x_138_, lean_object* v___x_139_, lean_object* v___x_140_, lean_object* v___x_141_, lean_object* v___x_142_, lean_object* v___x_143_, lean_object* v___x_144_, lean_object* v___x_145_, lean_object* v___x_146_, lean_object* v___x_147_, lean_object* v___y_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_createCalc___redArg___lam__1(v_ctx_133_, v___x_134_, v___x_135_, v_a_136_, v___x_137_, v___x_138_, v___x_139_, v___x_140_, v___x_141_, v___x_142_, v___x_143_, v___x_144_, v___x_145_, v___x_146_, v___x_147_);
return v_res_149_;
}
}
static lean_object* _init_lp_mathlib_createCalc___redArg___closed__5(void){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_160_;
}
}
static lean_object* _init_lp_mathlib_createCalc___redArg___closed__6(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__5, &lp_mathlib_createCalc___redArg___closed__5_once, _init_lp_mathlib_createCalc___redArg___closed__5);
v___x_162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_162_, 0, v___x_161_);
return v___x_162_;
}
}
static lean_object* _init_lp_mathlib_createCalc___redArg___closed__7(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_163_ = lean_unsigned_to_nat(32u);
v___x_164_ = lean_mk_empty_array_with_capacity(v___x_163_);
v___x_165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib_createCalc___redArg___closed__8(void){
_start:
{
size_t v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_166_ = ((size_t)5ULL);
v___x_167_ = lean_unsigned_to_nat(0u);
v___x_168_ = lean_unsigned_to_nat(32u);
v___x_169_ = lean_mk_empty_array_with_capacity(v___x_168_);
v___x_170_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__7, &lp_mathlib_createCalc___redArg___closed__7_once, _init_lp_mathlib_createCalc___redArg___closed__7);
v___x_171_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___x_169_);
lean_ctor_set(v___x_171_, 2, v___x_167_);
lean_ctor_set(v___x_171_, 3, v___x_167_);
lean_ctor_set_usize(v___x_171_, 4, v___x_166_);
return v___x_171_;
}
}
static lean_object* _init_lp_mathlib_createCalc___redArg___closed__9(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_172_ = lean_box(1);
v___x_173_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__8, &lp_mathlib_createCalc___redArg___closed__8_once, _init_lp_mathlib_createCalc___redArg___closed__8);
v___x_174_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__6, &lp_mathlib_createCalc___redArg___closed__6_once, _init_lp_mathlib_createCalc___redArg___closed__6);
v___x_175_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set(v___x_175_, 1, v___x_173_);
lean_ctor_set(v___x_175_, 2, v___x_172_);
return v___x_175_;
}
}
static lean_object* _init_lp_mathlib_createCalc___redArg___closed__13(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_179_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__12));
v___x_180_ = lean_unsigned_to_nat(14u);
v___x_181_ = lean_unsigned_to_nat(22u);
v___x_182_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__11));
v___x_183_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__10));
v___x_184_ = l_mkPanicMessageWithDecl(v___x_183_, v___x_182_, v___x_181_, v___x_180_, v___x_179_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg(lean_object* v_ctx_185_, lean_object* v_node_186_, lean_object* v_a_187_){
_start:
{
if (lean_obj_tag(v_node_186_) == 1)
{
lean_object* v_i_192_; lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_256_; 
v_i_192_ = lean_ctor_get(v_node_186_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v_node_186_);
if (v_isSharedCheck_256_ == 0)
{
lean_object* v_unused_257_; 
v_unused_257_ = lean_ctor_get(v_node_186_, 1);
lean_dec(v_unused_257_);
v___x_194_ = v_node_186_;
v_isShared_195_ = v_isSharedCheck_256_;
goto v_resetjp_193_;
}
else
{
lean_inc(v_i_192_);
lean_dec(v_node_186_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_256_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
if (lean_obj_tag(v_i_192_) == 0)
{
lean_object* v_i_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_255_; 
v_i_196_ = lean_ctor_get(v_i_192_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v_i_192_);
if (v_isSharedCheck_255_ == 0)
{
v___x_198_ = v_i_192_;
v_isShared_199_ = v_isSharedCheck_255_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_i_196_);
lean_dec(v_i_192_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_255_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v_toElabInfo_200_; lean_object* v_goalsBefore_201_; uint8_t v___x_202_; 
v_toElabInfo_200_ = lean_ctor_get(v_i_196_, 0);
lean_inc_ref(v_toElabInfo_200_);
v_goalsBefore_201_ = lean_ctor_get(v_i_196_, 2);
lean_inc(v_goalsBefore_201_);
lean_dec_ref(v_i_196_);
v___x_202_ = l_List_isEmpty___redArg(v_goalsBefore_201_);
if (v___x_202_ == 0)
{
lean_object* v___x_203_; lean_object* v_a_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_250_; 
v___x_203_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0(v_a_187_);
v_a_204_ = lean_ctor_get(v___x_203_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_250_ == 0)
{
v___x_206_ = v___x_203_;
v_isShared_207_ = v_isSharedCheck_250_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_a_204_);
lean_dec(v___x_203_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_250_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v_eager_211_; lean_object* v_toEditableDocumentCore_212_; lean_object* v_meta_213_; lean_object* v_text_214_; lean_object* v___y_216_; lean_object* v___y_217_; lean_object* v_stx_238_; lean_object* v___y_240_; lean_object* v___x_246_; 
v___x_208_ = lean_box(0);
v___x_209_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__1));
v___x_210_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__3));
v_eager_211_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__4));
v_toEditableDocumentCore_212_ = lean_ctor_get(v_a_204_, 0);
v_meta_213_ = lean_ctor_get(v_toEditableDocumentCore_212_, 0);
v_text_214_ = lean_ctor_get(v_meta_213_, 3);
v_stx_238_ = lean_ctor_get(v_toElabInfo_200_, 1);
lean_inc(v_stx_238_);
lean_dec_ref(v_toElabInfo_200_);
v___x_246_ = l_Lean_Syntax_getPos_x3f(v_stx_238_, v___x_202_);
if (lean_obj_tag(v___x_246_) == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_247_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__13, &lp_mathlib_createCalc___redArg___closed__13_once, _init_lp_mathlib_createCalc___redArg___closed__13);
v___x_248_ = lp_mathlib_panic___at___00createCalc_spec__2(v___x_247_);
v___y_240_ = v___x_248_;
goto v___jp_239_;
}
else
{
lean_object* v_val_249_; 
v_val_249_ = lean_ctor_get(v___x_246_, 0);
lean_inc(v_val_249_);
lean_dec_ref_known(v___x_246_, 1);
v___y_240_ = v_val_249_;
goto v___jp_239_;
}
v___jp_215_:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___f_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___f_225_; lean_object* v___x_227_; 
lean_inc_ref(v_text_214_);
v___x_218_ = l_Lean_FileMap_utf8PosToLspPos(v_text_214_, v___y_217_);
lean_dec(v___y_217_);
v___x_219_ = lean_box(0);
v___x_220_ = lean_unsigned_to_nat(0u);
v___x_221_ = l_List_get_x21Internal___redArg(v___x_219_, v_goalsBefore_201_, v___x_220_);
lean_dec(v_goalsBefore_201_);
lean_inc(v___x_221_);
v___f_222_ = lean_alloc_closure((void*)(lp_mathlib_createCalc___redArg___lam__0___boxed), 6, 1);
lean_closure_set(v___f_222_, 0, v___x_221_);
v___x_223_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__9, &lp_mathlib_createCalc___redArg___closed__9_once, _init_lp_mathlib_createCalc___redArg___closed__9);
v___x_224_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00createCalc_spec__1___boxed), 8, 3);
lean_closure_set(v___x_224_, 0, lean_box(0));
lean_closure_set(v___x_224_, 1, v___x_221_);
lean_closure_set(v___x_224_, 2, v___f_222_);
v___f_225_ = lean_alloc_closure((void*)(lp_mathlib_createCalc___redArg___lam__1___boxed), 16, 15);
lean_closure_set(v___f_225_, 0, v_ctx_185_);
lean_closure_set(v___f_225_, 1, v___x_223_);
lean_closure_set(v___f_225_, 2, v___x_224_);
lean_closure_set(v___f_225_, 3, v_a_204_);
lean_closure_set(v___f_225_, 4, v___y_216_);
lean_closure_set(v___f_225_, 5, v___x_218_);
lean_closure_set(v___f_225_, 6, v___x_220_);
lean_closure_set(v___f_225_, 7, v___x_208_);
lean_closure_set(v___f_225_, 8, v___x_209_);
lean_closure_set(v___f_225_, 9, v___x_210_);
lean_closure_set(v___f_225_, 10, v___x_208_);
lean_closure_set(v___f_225_, 11, v___x_208_);
lean_closure_set(v___f_225_, 12, v___x_208_);
lean_closure_set(v___f_225_, 13, v___x_208_);
lean_closure_set(v___f_225_, 14, v___x_208_);
if (v_isShared_199_ == 0)
{
lean_ctor_set_tag(v___x_198_, 1);
lean_ctor_set(v___x_198_, 0, v___f_225_);
v___x_227_ = v___x_198_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v___f_225_);
v___x_227_ = v_reuseFailAlloc_237_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
lean_object* v___x_229_; 
if (v_isShared_195_ == 0)
{
lean_ctor_set_tag(v___x_194_, 0);
lean_ctor_set(v___x_194_, 1, v___x_227_);
lean_ctor_set(v___x_194_, 0, v_eager_211_);
v___x_229_ = v___x_194_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v_eager_211_);
lean_ctor_set(v_reuseFailAlloc_236_, 1, v___x_227_);
v___x_229_ = v_reuseFailAlloc_236_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_234_; 
v___x_230_ = lean_unsigned_to_nat(1u);
v___x_231_ = lean_mk_empty_array_with_capacity(v___x_230_);
v___x_232_ = lean_array_push(v___x_231_, v___x_229_);
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 0, v___x_232_);
v___x_234_ = v___x_206_;
goto v_reusejp_233_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_232_);
v___x_234_ = v_reuseFailAlloc_235_;
goto v_reusejp_233_;
}
v_reusejp_233_:
{
return v___x_234_;
}
}
}
}
v___jp_239_:
{
lean_object* v___x_241_; lean_object* v___x_242_; 
lean_inc_ref(v_text_214_);
v___x_241_ = l_Lean_FileMap_utf8PosToLspPos(v_text_214_, v___y_240_);
lean_dec(v___y_240_);
v___x_242_ = l_Lean_Syntax_getTailPos_x3f(v_stx_238_, v___x_202_);
lean_dec(v_stx_238_);
if (lean_obj_tag(v___x_242_) == 0)
{
lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_243_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__13, &lp_mathlib_createCalc___redArg___closed__13_once, _init_lp_mathlib_createCalc___redArg___closed__13);
v___x_244_ = lp_mathlib_panic___at___00createCalc_spec__2(v___x_243_);
v___y_216_ = v___x_241_;
v___y_217_ = v___x_244_;
goto v___jp_215_;
}
else
{
lean_object* v_val_245_; 
v_val_245_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_val_245_);
lean_dec_ref_known(v___x_242_, 1);
v___y_216_ = v___x_241_;
v___y_217_ = v_val_245_;
goto v___jp_215_;
}
}
}
}
else
{
lean_object* v___x_251_; lean_object* v___x_253_; 
lean_dec(v_goalsBefore_201_);
lean_dec_ref(v_toElabInfo_200_);
lean_del_object(v___x_194_);
lean_dec_ref(v_ctx_185_);
v___x_251_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__0));
if (v_isShared_199_ == 0)
{
lean_ctor_set(v___x_198_, 0, v___x_251_);
v___x_253_ = v___x_198_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v___x_251_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
}
else
{
lean_del_object(v___x_194_);
lean_dec_ref(v_i_192_);
lean_dec_ref(v_ctx_185_);
goto v___jp_189_;
}
}
}
else
{
lean_dec_ref(v_node_186_);
lean_dec_ref(v_ctx_185_);
goto v___jp_189_;
}
v___jp_189_:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib_createCalc___redArg___closed__0));
v___x_191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
return v___x_191_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___redArg___boxed(lean_object* v_ctx_258_, lean_object* v_node_259_, lean_object* v_a_260_, lean_object* v_a_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_createCalc___redArg(v_ctx_258_, v_node_259_, v_a_260_);
lean_dec_ref(v_a_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc(lean_object* v___params_263_, lean_object* v___snap_264_, lean_object* v_ctx_265_, lean_object* v___stack_266_, lean_object* v_node_267_, lean_object* v_a_268_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lp_mathlib_createCalc___redArg(v_ctx_265_, v_node_267_, v_a_268_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_createCalc___boxed(lean_object* v___params_271_, lean_object* v___snap_272_, lean_object* v_ctx_273_, lean_object* v___stack_274_, lean_object* v_node_275_, lean_object* v_a_276_, lean_object* v_a_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_createCalc(v___params_271_, v___snap_272_, v_ctx_273_, v___stack_274_, v_node_275_, v_a_276_);
lean_dec_ref(v_a_276_);
lean_dec(v___stack_274_);
lean_dec_ref(v___snap_272_);
lean_dec_ref(v___params_271_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__0(lean_object* v_prop_279_){
_start:
{
lean_object* v_toSelectInsertParams_280_; lean_object* v_pos_281_; 
v_toSelectInsertParams_280_ = lean_ctor_get(v_prop_279_, 0);
v_pos_281_ = lean_ctor_get(v_toSelectInsertParams_280_, 0);
lean_inc_ref(v_pos_281_);
return v_pos_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__0___boxed(lean_object* v_prop_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_instSelectInsertParamsClassCalcParams___lam__0(v_prop_282_);
lean_dec_ref(v_prop_282_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__1(lean_object* v_prop_284_){
_start:
{
lean_object* v_toSelectInsertParams_285_; lean_object* v_goals_286_; 
v_toSelectInsertParams_285_ = lean_ctor_get(v_prop_284_, 0);
v_goals_286_ = lean_ctor_get(v_toSelectInsertParams_285_, 1);
lean_inc_ref(v_goals_286_);
return v_goals_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__1___boxed(lean_object* v_prop_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_instSelectInsertParamsClassCalcParams___lam__1(v_prop_287_);
lean_dec_ref(v_prop_287_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__2(lean_object* v_prop_289_){
_start:
{
lean_object* v_toSelectInsertParams_290_; lean_object* v_selectedLocations_291_; 
v_toSelectInsertParams_290_ = lean_ctor_get(v_prop_289_, 0);
v_selectedLocations_291_ = lean_ctor_get(v_toSelectInsertParams_290_, 2);
lean_inc_ref(v_selectedLocations_291_);
return v_selectedLocations_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__2___boxed(lean_object* v_prop_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_mathlib_instSelectInsertParamsClassCalcParams___lam__2(v_prop_292_);
lean_dec_ref(v_prop_292_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__3(lean_object* v_prop_294_){
_start:
{
lean_object* v_toSelectInsertParams_295_; lean_object* v_replaceRange_296_; 
v_toSelectInsertParams_295_ = lean_ctor_get(v_prop_294_, 0);
v_replaceRange_296_ = lean_ctor_get(v_toSelectInsertParams_295_, 3);
lean_inc_ref(v_replaceRange_296_);
return v_replaceRange_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassCalcParams___lam__3___boxed(lean_object* v_prop_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_instSelectInsertParamsClassCalcParams___lam__3(v_prop_297_);
lean_dec_ref(v_prop_297_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(lean_object* v_j_309_, lean_object* v_k_310_){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_311_ = l_Lean_Json_getObjValD(v_j_309_, v_k_310_);
v___x_312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0___boxed(lean_object* v_j_313_, lean_object* v_k_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_j_313_, v_k_314_);
lean_dec_ref(v_k_314_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_(lean_object* v_json_322_){
_start:
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v_a_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v_a_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v_a_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v_a_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v_a_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v_a_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_348_; 
v___x_323_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc_n(v_json_322_, 5);
v___x_324_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_json_322_, v___x_323_);
v_a_325_ = lean_ctor_get(v___x_324_, 0);
lean_inc(v_a_325_);
lean_dec_ref(v___x_324_);
v___x_326_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_327_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_json_322_, v___x_326_);
v_a_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_a_328_);
lean_dec_ref(v___x_327_);
v___x_329_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_330_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_json_322_, v___x_329_);
v_a_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc(v_a_331_);
lean_dec_ref(v___x_330_);
v___x_332_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_333_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_json_322_, v___x_332_);
v_a_334_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_a_334_);
lean_dec_ref(v___x_333_);
v___x_335_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_336_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_json_322_, v___x_335_);
v_a_337_ = lean_ctor_get(v___x_336_, 0);
lean_inc(v_a_337_);
lean_dec_ref(v___x_336_);
v___x_338_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_339_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20__spec__0(v_json_322_, v___x_338_);
v_a_340_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_348_ == 0)
{
v___x_342_ = v___x_339_;
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_a_340_);
lean_dec(v___x_339_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_348_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
lean_object* v___x_344_; lean_object* v___x_346_; 
v___x_344_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_344_, 0, v_a_325_);
lean_ctor_set(v___x_344_, 1, v_a_328_);
lean_ctor_set(v___x_344_, 2, v_a_331_);
lean_ctor_set(v___x_344_, 3, v_a_334_);
lean_ctor_set(v___x_344_, 4, v_a_337_);
lean_ctor_set(v___x_344_, 5, v_a_340_);
if (v_isShared_343_ == 0)
{
lean_ctor_set(v___x_342_, 0, v___x_344_);
v___x_346_ = v___x_342_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v___x_344_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__spec__0(lean_object* v_a_351_, lean_object* v_a_352_){
_start:
{
if (lean_obj_tag(v_a_351_) == 0)
{
lean_object* v___x_353_; 
v___x_353_ = lean_array_to_list(v_a_352_);
return v___x_353_;
}
else
{
lean_object* v_head_354_; lean_object* v_tail_355_; lean_object* v___x_356_; 
v_head_354_ = lean_ctor_get(v_a_351_, 0);
lean_inc(v_head_354_);
v_tail_355_ = lean_ctor_get(v_a_351_, 1);
lean_inc(v_tail_355_);
lean_dec_ref_known(v_a_351_, 2);
v___x_356_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_352_, v_head_354_);
v_a_351_ = v_tail_355_;
v_a_352_ = v___x_356_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_(lean_object* v_x_360_){
_start:
{
lean_object* v_pos_361_; lean_object* v_goals_362_; lean_object* v_selectedLocations_363_; lean_object* v_replaceRange_364_; lean_object* v_isFirst_365_; lean_object* v_indent_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v_pos_361_ = lean_ctor_get(v_x_360_, 0);
v_goals_362_ = lean_ctor_get(v_x_360_, 1);
v_selectedLocations_363_ = lean_ctor_get(v_x_360_, 2);
v_replaceRange_364_ = lean_ctor_get(v_x_360_, 3);
v_isFirst_365_ = lean_ctor_get(v_x_360_, 4);
v_indent_366_ = lean_ctor_get(v_x_360_, 5);
v___x_367_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc(v_pos_361_);
v___x_368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_368_, 0, v___x_367_);
lean_ctor_set(v___x_368_, 1, v_pos_361_);
v___x_369_ = lean_box(0);
v___x_370_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_370_, 0, v___x_368_);
lean_ctor_set(v___x_370_, 1, v___x_369_);
v___x_371_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc(v_goals_362_);
v___x_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v_goals_362_);
v___x_373_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_373_, 0, v___x_372_);
lean_ctor_set(v___x_373_, 1, v___x_369_);
v___x_374_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc(v_selectedLocations_363_);
v___x_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v_selectedLocations_363_);
v___x_376_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
lean_ctor_set(v___x_376_, 1, v___x_369_);
v___x_377_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc(v_replaceRange_364_);
v___x_378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set(v___x_378_, 1, v_replaceRange_364_);
v___x_379_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
lean_ctor_set(v___x_379_, 1, v___x_369_);
v___x_380_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc(v_isFirst_365_);
v___x_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v_isFirst_365_);
v___x_382_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_382_, 0, v___x_381_);
lean_ctor_set(v___x_382_, 1, v___x_369_);
v___x_383_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
lean_inc(v_indent_366_);
v___x_384_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
lean_ctor_set(v___x_384_, 1, v_indent_366_);
v___x_385_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_385_, 0, v___x_384_);
lean_ctor_set(v___x_385_, 1, v___x_369_);
v___x_386_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
lean_ctor_set(v___x_386_, 1, v___x_369_);
v___x_387_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_387_, 0, v___x_382_);
lean_ctor_set(v___x_387_, 1, v___x_386_);
v___x_388_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_379_);
lean_ctor_set(v___x_388_, 1, v___x_387_);
v___x_389_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_376_);
lean_ctor_set(v___x_389_, 1, v___x_388_);
v___x_390_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_373_);
lean_ctor_set(v___x_390_, 1, v___x_389_);
v___x_391_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_370_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = ((lean_object*)(lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_));
v___x_393_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39__spec__0(v___x_391_, v___x_392_);
v___x_394_ = l_Lean_Json_mkObj(v___x_393_);
lean_dec(v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39____boxed(lean_object* v_x_395_){
_start:
{
lean_object* v_res_396_; 
v_res_396_ = lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_(v_x_395_);
lean_dec_ref(v_x_395_);
return v_res_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(size_t v_sz_399_, size_t v_i_400_, lean_object* v_bs_401_, lean_object* v___y_402_){
_start:
{
uint8_t v___x_403_; 
v___x_403_ = lean_usize_dec_lt(v_i_400_, v_sz_399_);
if (v___x_403_ == 0)
{
lean_object* v___x_404_; 
v___x_404_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_404_, 0, v_bs_401_);
lean_ctor_set(v___x_404_, 1, v___y_402_);
return v___x_404_;
}
else
{
lean_object* v_v_405_; lean_object* v___x_406_; lean_object* v_fst_407_; lean_object* v_snd_408_; lean_object* v___x_409_; lean_object* v_bs_x27_410_; size_t v___x_411_; size_t v___x_412_; lean_object* v___x_413_; 
v_v_405_ = lean_array_uget_borrowed(v_bs_401_, v_i_400_);
lean_inc(v_v_405_);
v___x_406_ = l_Lean_Widget_instRpcEncodableInteractiveGoal_enc_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(v_v_405_, v___y_402_);
v_fst_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc(v_fst_407_);
v_snd_408_ = lean_ctor_get(v___x_406_, 1);
lean_inc(v_snd_408_);
lean_dec_ref(v___x_406_);
v___x_409_ = lean_unsigned_to_nat(0u);
v_bs_x27_410_ = lean_array_uset(v_bs_401_, v_i_400_, v___x_409_);
v___x_411_ = ((size_t)1ULL);
v___x_412_ = lean_usize_add(v_i_400_, v___x_411_);
v___x_413_ = lean_array_uset(v_bs_x27_410_, v_i_400_, v_fst_407_);
v_i_400_ = v___x_412_;
v_bs_401_ = v___x_413_;
v___y_402_ = v_snd_408_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___boxed(lean_object* v_sz_415_, lean_object* v_i_416_, lean_object* v_bs_417_, lean_object* v___y_418_){
_start:
{
size_t v_sz_boxed_419_; size_t v_i_boxed_420_; lean_object* v_res_421_; 
v_sz_boxed_419_ = lean_unbox_usize(v_sz_415_);
lean_dec(v_sz_415_);
v_i_boxed_420_ = lean_unbox_usize(v_i_416_);
lean_dec(v_i_416_);
v_res_421_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(v_sz_boxed_419_, v_i_boxed_420_, v_bs_417_, v___y_418_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2_spec__2(size_t v_sz_422_, size_t v_i_423_, lean_object* v_bs_424_){
_start:
{
uint8_t v___x_425_; 
v___x_425_ = lean_usize_dec_lt(v_i_423_, v_sz_422_);
if (v___x_425_ == 0)
{
return v_bs_424_;
}
else
{
lean_object* v_v_426_; lean_object* v___x_427_; lean_object* v_bs_x27_428_; size_t v___x_429_; size_t v___x_430_; lean_object* v___x_431_; 
v_v_426_ = lean_array_uget(v_bs_424_, v_i_423_);
v___x_427_ = lean_unsigned_to_nat(0u);
v_bs_x27_428_ = lean_array_uset(v_bs_424_, v_i_423_, v___x_427_);
v___x_429_ = ((size_t)1ULL);
v___x_430_ = lean_usize_add(v_i_423_, v___x_429_);
v___x_431_ = lean_array_uset(v_bs_x27_428_, v_i_423_, v_v_426_);
v_i_423_ = v___x_430_;
v_bs_424_ = v___x_431_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2_spec__2___boxed(lean_object* v_sz_433_, lean_object* v_i_434_, lean_object* v_bs_435_){
_start:
{
size_t v_sz_boxed_436_; size_t v_i_boxed_437_; lean_object* v_res_438_; 
v_sz_boxed_436_ = lean_unbox_usize(v_sz_433_);
lean_dec(v_sz_433_);
v_i_boxed_437_ = lean_unbox_usize(v_i_434_);
lean_dec(v_i_434_);
v_res_438_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2_spec__2(v_sz_boxed_436_, v_i_boxed_437_, v_bs_435_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(lean_object* v_a_439_){
_start:
{
size_t v_sz_440_; size_t v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v_sz_440_ = lean_array_size(v_a_439_);
v___x_441_ = ((size_t)0ULL);
v___x_442_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2_spec__2(v_sz_440_, v___x_441_, v_a_439_);
v___x_443_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(size_t v_sz_444_, size_t v_i_445_, lean_object* v_bs_446_, lean_object* v___y_447_){
_start:
{
uint8_t v___x_448_; 
v___x_448_ = lean_usize_dec_lt(v_i_445_, v_sz_444_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; 
v___x_449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_449_, 0, v_bs_446_);
lean_ctor_set(v___x_449_, 1, v___y_447_);
return v___x_449_;
}
else
{
lean_object* v_v_450_; lean_object* v___x_451_; lean_object* v_bs_x27_452_; lean_object* v___x_453_; size_t v___x_454_; size_t v___x_455_; lean_object* v___x_456_; 
v_v_450_ = lean_array_uget(v_bs_446_, v_i_445_);
v___x_451_ = lean_unsigned_to_nat(0u);
v_bs_x27_452_ = lean_array_uset(v_bs_446_, v_i_445_, v___x_451_);
v___x_453_ = l_Lean_SubExpr_instToJsonGoalsLocation_toJson(v_v_450_);
v___x_454_ = ((size_t)1ULL);
v___x_455_ = lean_usize_add(v_i_445_, v___x_454_);
v___x_456_ = lean_array_uset(v_bs_x27_452_, v_i_445_, v___x_453_);
v_i_445_ = v___x_455_;
v_bs_446_ = v___x_456_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___boxed(lean_object* v_sz_458_, lean_object* v_i_459_, lean_object* v_bs_460_, lean_object* v___y_461_){
_start:
{
size_t v_sz_boxed_462_; size_t v_i_boxed_463_; lean_object* v_res_464_; 
v_sz_boxed_462_ = lean_unbox_usize(v_sz_458_);
lean_dec(v_sz_458_);
v_i_boxed_463_ = lean_unbox_usize(v_i_459_);
lean_dec(v_i_459_);
v_res_464_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(v_sz_boxed_462_, v_i_boxed_463_, v_bs_460_, v___y_461_);
return v_res_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_(lean_object* v_a_465_, lean_object* v_a_466_){
_start:
{
lean_object* v_toSelectInsertParams_467_; uint8_t v_isFirst_468_; lean_object* v_indent_469_; lean_object* v_pos_470_; lean_object* v_goals_471_; lean_object* v_selectedLocations_472_; lean_object* v_replaceRange_473_; size_t v_sz_474_; size_t v___x_475_; lean_object* v___x_476_; lean_object* v_fst_477_; lean_object* v_snd_478_; size_t v_sz_479_; lean_object* v___x_480_; lean_object* v_fst_481_; lean_object* v_snd_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_498_; 
v_toSelectInsertParams_467_ = lean_ctor_get(v_a_465_, 0);
lean_inc_ref(v_toSelectInsertParams_467_);
v_isFirst_468_ = lean_ctor_get_uint8(v_a_465_, sizeof(void*)*2);
v_indent_469_ = lean_ctor_get(v_a_465_, 1);
lean_inc(v_indent_469_);
lean_dec_ref(v_a_465_);
v_pos_470_ = lean_ctor_get(v_toSelectInsertParams_467_, 0);
lean_inc_ref(v_pos_470_);
v_goals_471_ = lean_ctor_get(v_toSelectInsertParams_467_, 1);
lean_inc_ref(v_goals_471_);
v_selectedLocations_472_ = lean_ctor_get(v_toSelectInsertParams_467_, 2);
lean_inc_ref(v_selectedLocations_472_);
v_replaceRange_473_ = lean_ctor_get(v_toSelectInsertParams_467_, 3);
lean_inc_ref(v_replaceRange_473_);
lean_dec_ref(v_toSelectInsertParams_467_);
v_sz_474_ = lean_array_size(v_goals_471_);
v___x_475_ = ((size_t)0ULL);
v___x_476_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(v_sz_474_, v___x_475_, v_goals_471_, v_a_466_);
v_fst_477_ = lean_ctor_get(v___x_476_, 0);
lean_inc(v_fst_477_);
v_snd_478_ = lean_ctor_get(v___x_476_, 1);
lean_inc(v_snd_478_);
lean_dec_ref(v___x_476_);
v_sz_479_ = lean_array_size(v_selectedLocations_472_);
v___x_480_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(v_sz_479_, v___x_475_, v_selectedLocations_472_, v_snd_478_);
v_fst_481_ = lean_ctor_get(v___x_480_, 0);
v_snd_482_ = lean_ctor_get(v___x_480_, 1);
v_isSharedCheck_498_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_498_ == 0)
{
v___x_484_ = v___x_480_;
v_isShared_485_ = v_isSharedCheck_498_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_snd_482_);
lean_inc(v_fst_481_);
lean_dec(v___x_480_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_498_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_496_; 
v___x_486_ = l_Lean_Lsp_instToJsonPosition_toJson(v_pos_470_);
v___x_487_ = lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(v_fst_477_);
v___x_488_ = lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableCalcParams_enc_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(v_fst_481_);
v___x_489_ = l_Lean_Lsp_instToJsonRange_toJson(v_replaceRange_473_);
v___x_490_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_490_, 0, v_isFirst_468_);
v___x_491_ = l_Lean_JsonNumber_fromNat(v_indent_469_);
v___x_492_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
v___x_493_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_493_, 0, v___x_486_);
lean_ctor_set(v___x_493_, 1, v___x_487_);
lean_ctor_set(v___x_493_, 2, v___x_488_);
lean_ctor_set(v___x_493_, 3, v___x_489_);
lean_ctor_set(v___x_493_, 4, v___x_490_);
lean_ctor_set(v___x_493_, 5, v___x_492_);
v___x_494_ = lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_(v___x_493_);
lean_dec_ref_known(v___x_493_, 6);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 0, v___x_494_);
v___x_496_ = v___x_484_;
goto v_reusejp_495_;
}
else
{
lean_object* v_reuseFailAlloc_497_; 
v_reuseFailAlloc_497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_497_, 0, v___x_494_);
lean_ctor_set(v_reuseFailAlloc_497_, 1, v_snd_482_);
v___x_496_ = v_reuseFailAlloc_497_;
goto v_reusejp_495_;
}
v_reusejp_495_:
{
return v___x_496_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___redArg(lean_object* v_x_499_){
_start:
{
lean_inc_ref(v_x_499_);
return v_x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object* v_x_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___redArg(v_x_500_);
lean_dec_ref(v_x_500_);
return v_res_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(lean_object* v_00_u03b1_502_, lean_object* v_x_503_, lean_object* v___y_504_){
_start:
{
lean_inc_ref(v_x_503_);
return v_x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0___boxed(lean_object* v_00_u03b1_505_, lean_object* v_x_506_, lean_object* v___y_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__0(v_00_u03b1_505_, v_x_506_, v___y_507_);
lean_dec_ref(v___y_507_);
lean_dec_ref(v_x_506_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1_spec__1(size_t v_sz_509_, size_t v_i_510_, lean_object* v_bs_511_){
_start:
{
uint8_t v___x_512_; 
v___x_512_ = lean_usize_dec_lt(v_i_510_, v_sz_509_);
if (v___x_512_ == 0)
{
lean_object* v___x_513_; 
v___x_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_513_, 0, v_bs_511_);
return v___x_513_;
}
else
{
lean_object* v_v_514_; lean_object* v___x_515_; lean_object* v_bs_x27_516_; size_t v___x_517_; size_t v___x_518_; lean_object* v___x_519_; 
v_v_514_ = lean_array_uget(v_bs_511_, v_i_510_);
v___x_515_ = lean_unsigned_to_nat(0u);
v_bs_x27_516_ = lean_array_uset(v_bs_511_, v_i_510_, v___x_515_);
v___x_517_ = ((size_t)1ULL);
v___x_518_ = lean_usize_add(v_i_510_, v___x_517_);
v___x_519_ = lean_array_uset(v_bs_x27_516_, v_i_510_, v_v_514_);
v_i_510_ = v___x_518_;
v_bs_511_ = v___x_519_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object* v_sz_521_, lean_object* v_i_522_, lean_object* v_bs_523_){
_start:
{
size_t v_sz_boxed_524_; size_t v_i_boxed_525_; lean_object* v_res_526_; 
v_sz_boxed_524_ = lean_unbox_usize(v_sz_521_);
lean_dec(v_sz_521_);
v_i_boxed_525_ = lean_unbox_usize(v_i_522_);
lean_dec(v_i_522_);
v_res_526_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1_spec__1(v_sz_boxed_524_, v_i_boxed_525_, v_bs_523_);
return v_res_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(lean_object* v_x_529_){
_start:
{
if (lean_obj_tag(v_x_529_) == 4)
{
lean_object* v_elems_530_; size_t v_sz_531_; size_t v___x_532_; lean_object* v___x_533_; 
v_elems_530_ = lean_ctor_get(v_x_529_, 0);
lean_inc_ref(v_elems_530_);
lean_dec_ref_known(v_x_529_, 1);
v_sz_531_ = lean_array_size(v_elems_530_);
v___x_532_ = ((size_t)0ULL);
v___x_533_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1_spec__1(v_sz_531_, v___x_532_, v_elems_530_);
return v___x_533_;
}
else
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_534_ = ((lean_object*)(lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__0));
v___x_535_ = lean_unsigned_to_nat(80u);
v___x_536_ = l_Lean_Json_pretty(v_x_529_, v___x_535_);
v___x_537_ = lean_string_append(v___x_534_, v___x_536_);
lean_dec_ref(v___x_536_);
v___x_538_ = ((lean_object*)(lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1___closed__1));
v___x_539_ = lean_string_append(v___x_537_, v___x_538_);
v___x_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
return v___x_540_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(size_t v_sz_541_, size_t v_i_542_, lean_object* v_bs_543_, lean_object* v___y_544_){
_start:
{
uint8_t v___x_545_; 
v___x_545_ = lean_usize_dec_lt(v_i_542_, v_sz_541_);
if (v___x_545_ == 0)
{
lean_object* v___x_546_; 
v___x_546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_546_, 0, v_bs_543_);
return v___x_546_;
}
else
{
lean_object* v_v_547_; lean_object* v___x_548_; 
v_v_547_ = lean_array_uget_borrowed(v_bs_543_, v_i_542_);
lean_inc(v_v_547_);
v___x_548_ = l_Lean_Widget_instRpcEncodableInteractiveGoal_dec_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(v_v_547_, v___y_544_);
if (lean_obj_tag(v___x_548_) == 0)
{
lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec_ref(v_bs_543_);
v_a_549_ = lean_ctor_get(v___x_548_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_548_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___x_548_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_dec(v___x_548_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
if (v_isShared_552_ == 0)
{
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_a_549_);
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
lean_object* v_a_557_; lean_object* v___x_558_; lean_object* v_bs_x27_559_; size_t v___x_560_; size_t v___x_561_; lean_object* v___x_562_; 
v_a_557_ = lean_ctor_get(v___x_548_, 0);
lean_inc(v_a_557_);
lean_dec_ref_known(v___x_548_, 1);
v___x_558_ = lean_unsigned_to_nat(0u);
v_bs_x27_559_ = lean_array_uset(v_bs_543_, v_i_542_, v___x_558_);
v___x_560_ = ((size_t)1ULL);
v___x_561_ = lean_usize_add(v_i_542_, v___x_560_);
v___x_562_ = lean_array_uset(v_bs_x27_559_, v_i_542_, v_a_557_);
v_i_542_ = v___x_561_;
v_bs_543_ = v___x_562_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2___boxed(lean_object* v_sz_564_, lean_object* v_i_565_, lean_object* v_bs_566_, lean_object* v___y_567_){
_start:
{
size_t v_sz_boxed_568_; size_t v_i_boxed_569_; lean_object* v_res_570_; 
v_sz_boxed_568_ = lean_unbox_usize(v_sz_564_);
lean_dec(v_sz_564_);
v_i_boxed_569_ = lean_unbox_usize(v_i_565_);
lean_dec(v_i_565_);
v_res_570_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(v_sz_boxed_568_, v_i_boxed_569_, v_bs_566_, v___y_567_);
lean_dec_ref(v___y_567_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg(size_t v_sz_571_, size_t v_i_572_, lean_object* v_bs_573_){
_start:
{
uint8_t v___x_574_; 
v___x_574_ = lean_usize_dec_lt(v_i_572_, v_sz_571_);
if (v___x_574_ == 0)
{
lean_object* v___x_575_; 
v___x_575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_575_, 0, v_bs_573_);
return v___x_575_;
}
else
{
lean_object* v_v_576_; lean_object* v___x_577_; 
v_v_576_ = lean_array_uget_borrowed(v_bs_573_, v_i_572_);
lean_inc(v_v_576_);
v___x_577_ = l_Lean_SubExpr_instFromJsonGoalsLocation_fromJson(v_v_576_);
if (lean_obj_tag(v___x_577_) == 0)
{
lean_object* v_a_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_585_; 
lean_dec_ref(v_bs_573_);
v_a_578_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_585_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_585_ == 0)
{
v___x_580_ = v___x_577_;
v_isShared_581_ = v_isSharedCheck_585_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_a_578_);
lean_dec(v___x_577_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_585_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v___x_583_; 
if (v_isShared_581_ == 0)
{
v___x_583_ = v___x_580_;
goto v_reusejp_582_;
}
else
{
lean_object* v_reuseFailAlloc_584_; 
v_reuseFailAlloc_584_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_584_, 0, v_a_578_);
v___x_583_ = v_reuseFailAlloc_584_;
goto v_reusejp_582_;
}
v_reusejp_582_:
{
return v___x_583_;
}
}
}
else
{
lean_object* v_a_586_; lean_object* v___x_587_; lean_object* v_bs_x27_588_; size_t v___x_589_; size_t v___x_590_; lean_object* v___x_591_; 
v_a_586_ = lean_ctor_get(v___x_577_, 0);
lean_inc(v_a_586_);
lean_dec_ref_known(v___x_577_, 1);
v___x_587_ = lean_unsigned_to_nat(0u);
v_bs_x27_588_ = lean_array_uset(v_bs_573_, v_i_572_, v___x_587_);
v___x_589_ = ((size_t)1ULL);
v___x_590_ = lean_usize_add(v_i_572_, v___x_589_);
v___x_591_ = lean_array_uset(v_bs_x27_588_, v_i_572_, v_a_586_);
v_i_572_ = v___x_590_;
v_bs_573_ = v___x_591_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg___boxed(lean_object* v_sz_593_, lean_object* v_i_594_, lean_object* v_bs_595_){
_start:
{
size_t v_sz_boxed_596_; size_t v_i_boxed_597_; lean_object* v_res_598_; 
v_sz_boxed_596_ = lean_unbox_usize(v_sz_593_);
lean_dec(v_sz_593_);
v_i_boxed_597_ = lean_unbox_usize(v_i_594_);
lean_dec(v_i_594_);
v_res_598_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg(v_sz_boxed_596_, v_i_boxed_597_, v_bs_595_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_(lean_object* v_j_599_, lean_object* v_a_600_){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_(v_j_599_);
if (lean_obj_tag(v___x_601_) == 0)
{
lean_object* v_a_602_; lean_object* v___x_604_; uint8_t v_isShared_605_; uint8_t v_isSharedCheck_609_; 
v_a_602_ = lean_ctor_get(v___x_601_, 0);
v_isSharedCheck_609_ = !lean_is_exclusive(v___x_601_);
if (v_isSharedCheck_609_ == 0)
{
v___x_604_ = v___x_601_;
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
else
{
lean_inc(v_a_602_);
lean_dec(v___x_601_);
v___x_604_ = lean_box(0);
v_isShared_605_ = v_isSharedCheck_609_;
goto v_resetjp_603_;
}
v_resetjp_603_:
{
lean_object* v___x_607_; 
if (v_isShared_605_ == 0)
{
v___x_607_ = v___x_604_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_a_602_);
v___x_607_ = v_reuseFailAlloc_608_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
return v___x_607_;
}
}
}
else
{
lean_object* v_a_610_; lean_object* v_pos_611_; lean_object* v_goals_612_; lean_object* v_selectedLocations_613_; lean_object* v_replaceRange_614_; lean_object* v_isFirst_615_; lean_object* v_indent_616_; lean_object* v___x_617_; 
v_a_610_ = lean_ctor_get(v___x_601_, 0);
lean_inc(v_a_610_);
lean_dec_ref_known(v___x_601_, 1);
v_pos_611_ = lean_ctor_get(v_a_610_, 0);
lean_inc(v_pos_611_);
v_goals_612_ = lean_ctor_get(v_a_610_, 1);
lean_inc(v_goals_612_);
v_selectedLocations_613_ = lean_ctor_get(v_a_610_, 2);
lean_inc(v_selectedLocations_613_);
v_replaceRange_614_ = lean_ctor_get(v_a_610_, 3);
lean_inc(v_replaceRange_614_);
v_isFirst_615_ = lean_ctor_get(v_a_610_, 4);
lean_inc(v_isFirst_615_);
v_indent_616_ = lean_ctor_get(v_a_610_, 5);
lean_inc(v_indent_616_);
lean_dec(v_a_610_);
v___x_617_ = l_Lean_Lsp_instFromJsonPosition_fromJson(v_pos_611_);
if (lean_obj_tag(v___x_617_) == 0)
{
lean_object* v_a_618_; lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_625_; 
lean_dec(v_indent_616_);
lean_dec(v_isFirst_615_);
lean_dec(v_replaceRange_614_);
lean_dec(v_selectedLocations_613_);
lean_dec(v_goals_612_);
v_a_618_ = lean_ctor_get(v___x_617_, 0);
v_isSharedCheck_625_ = !lean_is_exclusive(v___x_617_);
if (v_isSharedCheck_625_ == 0)
{
v___x_620_ = v___x_617_;
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
else
{
lean_inc(v_a_618_);
lean_dec(v___x_617_);
v___x_620_ = lean_box(0);
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
v_resetjp_619_:
{
lean_object* v___x_623_; 
if (v_isShared_621_ == 0)
{
v___x_623_ = v___x_620_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_a_618_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
}
else
{
lean_object* v_a_626_; lean_object* v___x_627_; 
v_a_626_ = lean_ctor_get(v___x_617_, 0);
lean_inc(v_a_626_);
lean_dec_ref_known(v___x_617_, 1);
v___x_627_ = lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(v_goals_612_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v___x_630_; uint8_t v_isShared_631_; uint8_t v_isSharedCheck_635_; 
lean_dec(v_a_626_);
lean_dec(v_indent_616_);
lean_dec(v_isFirst_615_);
lean_dec(v_replaceRange_614_);
lean_dec(v_selectedLocations_613_);
v_a_628_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_635_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_635_ == 0)
{
v___x_630_ = v___x_627_;
v_isShared_631_ = v_isSharedCheck_635_;
goto v_resetjp_629_;
}
else
{
lean_inc(v_a_628_);
lean_dec(v___x_627_);
v___x_630_ = lean_box(0);
v_isShared_631_ = v_isSharedCheck_635_;
goto v_resetjp_629_;
}
v_resetjp_629_:
{
lean_object* v___x_633_; 
if (v_isShared_631_ == 0)
{
v___x_633_ = v___x_630_;
goto v_reusejp_632_;
}
else
{
lean_object* v_reuseFailAlloc_634_; 
v_reuseFailAlloc_634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_634_, 0, v_a_628_);
v___x_633_ = v_reuseFailAlloc_634_;
goto v_reusejp_632_;
}
v_reusejp_632_:
{
return v___x_633_;
}
}
}
else
{
lean_object* v_a_636_; size_t v_sz_637_; size_t v___x_638_; lean_object* v___x_639_; 
v_a_636_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_a_636_);
lean_dec_ref_known(v___x_627_, 1);
v_sz_637_ = lean_array_size(v_a_636_);
v___x_638_ = ((size_t)0ULL);
v___x_639_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__2(v_sz_637_, v___x_638_, v_a_636_, v_a_600_);
if (lean_obj_tag(v___x_639_) == 0)
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_dec(v_a_626_);
lean_dec(v_indent_616_);
lean_dec(v_isFirst_615_);
lean_dec(v_replaceRange_614_);
lean_dec(v_selectedLocations_613_);
v_a_640_ = lean_ctor_get(v___x_639_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_639_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_639_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
else
{
lean_object* v_a_648_; lean_object* v___x_649_; 
v_a_648_ = lean_ctor_get(v___x_639_, 0);
lean_inc(v_a_648_);
lean_dec_ref_known(v___x_639_, 1);
v___x_649_ = lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__1(v_selectedLocations_613_);
if (lean_obj_tag(v___x_649_) == 0)
{
lean_object* v_a_650_; lean_object* v___x_652_; uint8_t v_isShared_653_; uint8_t v_isSharedCheck_657_; 
lean_dec(v_a_648_);
lean_dec(v_a_626_);
lean_dec(v_indent_616_);
lean_dec(v_isFirst_615_);
lean_dec(v_replaceRange_614_);
v_a_650_ = lean_ctor_get(v___x_649_, 0);
v_isSharedCheck_657_ = !lean_is_exclusive(v___x_649_);
if (v_isSharedCheck_657_ == 0)
{
v___x_652_ = v___x_649_;
v_isShared_653_ = v_isSharedCheck_657_;
goto v_resetjp_651_;
}
else
{
lean_inc(v_a_650_);
lean_dec(v___x_649_);
v___x_652_ = lean_box(0);
v_isShared_653_ = v_isSharedCheck_657_;
goto v_resetjp_651_;
}
v_resetjp_651_:
{
lean_object* v___x_655_; 
if (v_isShared_653_ == 0)
{
v___x_655_ = v___x_652_;
goto v_reusejp_654_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v_a_650_);
v___x_655_ = v_reuseFailAlloc_656_;
goto v_reusejp_654_;
}
v_reusejp_654_:
{
return v___x_655_;
}
}
}
else
{
lean_object* v_a_658_; size_t v_sz_659_; lean_object* v___x_660_; 
v_a_658_ = lean_ctor_get(v___x_649_, 0);
lean_inc(v_a_658_);
lean_dec_ref_known(v___x_649_, 1);
v_sz_659_ = lean_array_size(v_a_658_);
v___x_660_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg(v_sz_659_, v___x_638_, v_a_658_);
if (lean_obj_tag(v___x_660_) == 0)
{
lean_object* v_a_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_668_; 
lean_dec(v_a_648_);
lean_dec(v_a_626_);
lean_dec(v_indent_616_);
lean_dec(v_isFirst_615_);
lean_dec(v_replaceRange_614_);
v_a_661_ = lean_ctor_get(v___x_660_, 0);
v_isSharedCheck_668_ = !lean_is_exclusive(v___x_660_);
if (v_isSharedCheck_668_ == 0)
{
v___x_663_ = v___x_660_;
v_isShared_664_ = v_isSharedCheck_668_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_a_661_);
lean_dec(v___x_660_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_668_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v___x_666_; 
if (v_isShared_664_ == 0)
{
v___x_666_ = v___x_663_;
goto v_reusejp_665_;
}
else
{
lean_object* v_reuseFailAlloc_667_; 
v_reuseFailAlloc_667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_667_, 0, v_a_661_);
v___x_666_ = v_reuseFailAlloc_667_;
goto v_reusejp_665_;
}
v_reusejp_665_:
{
return v___x_666_;
}
}
}
else
{
lean_object* v_a_669_; lean_object* v___x_670_; 
v_a_669_ = lean_ctor_get(v___x_660_, 0);
lean_inc(v_a_669_);
lean_dec_ref_known(v___x_660_, 1);
v___x_670_ = l_Lean_Lsp_instFromJsonRange_fromJson(v_replaceRange_614_);
if (lean_obj_tag(v___x_670_) == 0)
{
lean_object* v_a_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_678_; 
lean_dec(v_a_669_);
lean_dec(v_a_648_);
lean_dec(v_a_626_);
lean_dec(v_indent_616_);
lean_dec(v_isFirst_615_);
v_a_671_ = lean_ctor_get(v___x_670_, 0);
v_isSharedCheck_678_ = !lean_is_exclusive(v___x_670_);
if (v_isSharedCheck_678_ == 0)
{
v___x_673_ = v___x_670_;
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_a_671_);
lean_dec(v___x_670_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
lean_object* v___x_676_; 
if (v_isShared_674_ == 0)
{
v___x_676_ = v___x_673_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_671_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
else
{
lean_object* v_a_679_; lean_object* v___x_680_; 
v_a_679_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_a_679_);
lean_dec_ref_known(v___x_670_, 1);
v___x_680_ = l_Lean_Json_getBool_x3f(v_isFirst_615_);
lean_dec(v_isFirst_615_);
if (lean_obj_tag(v___x_680_) == 0)
{
lean_object* v_a_681_; lean_object* v___x_683_; uint8_t v_isShared_684_; uint8_t v_isSharedCheck_688_; 
lean_dec(v_a_679_);
lean_dec(v_a_669_);
lean_dec(v_a_648_);
lean_dec(v_a_626_);
lean_dec(v_indent_616_);
v_a_681_ = lean_ctor_get(v___x_680_, 0);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_680_);
if (v_isSharedCheck_688_ == 0)
{
v___x_683_ = v___x_680_;
v_isShared_684_ = v_isSharedCheck_688_;
goto v_resetjp_682_;
}
else
{
lean_inc(v_a_681_);
lean_dec(v___x_680_);
v___x_683_ = lean_box(0);
v_isShared_684_ = v_isSharedCheck_688_;
goto v_resetjp_682_;
}
v_resetjp_682_:
{
lean_object* v___x_686_; 
if (v_isShared_684_ == 0)
{
v___x_686_ = v___x_683_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v_a_681_);
v___x_686_ = v_reuseFailAlloc_687_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
return v___x_686_;
}
}
}
else
{
lean_object* v_a_689_; lean_object* v___x_690_; 
v_a_689_ = lean_ctor_get(v___x_680_, 0);
lean_inc(v_a_689_);
lean_dec_ref_known(v___x_680_, 1);
v___x_690_ = l_Lean_Json_getNat_x3f(v_indent_616_);
if (lean_obj_tag(v___x_690_) == 0)
{
lean_object* v_a_691_; lean_object* v___x_693_; uint8_t v_isShared_694_; uint8_t v_isSharedCheck_698_; 
lean_dec(v_a_689_);
lean_dec(v_a_679_);
lean_dec(v_a_669_);
lean_dec(v_a_648_);
lean_dec(v_a_626_);
v_a_691_ = lean_ctor_get(v___x_690_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_690_);
if (v_isSharedCheck_698_ == 0)
{
v___x_693_ = v___x_690_;
v_isShared_694_ = v_isSharedCheck_698_;
goto v_resetjp_692_;
}
else
{
lean_inc(v_a_691_);
lean_dec(v___x_690_);
v___x_693_ = lean_box(0);
v_isShared_694_ = v_isSharedCheck_698_;
goto v_resetjp_692_;
}
v_resetjp_692_:
{
lean_object* v___x_696_; 
if (v_isShared_694_ == 0)
{
v___x_696_ = v___x_693_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v_a_691_);
v___x_696_ = v_reuseFailAlloc_697_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
return v___x_696_;
}
}
}
else
{
lean_object* v_a_699_; lean_object* v___x_701_; uint8_t v_isShared_702_; uint8_t v_isSharedCheck_709_; 
v_a_699_ = lean_ctor_get(v___x_690_, 0);
v_isSharedCheck_709_ = !lean_is_exclusive(v___x_690_);
if (v_isSharedCheck_709_ == 0)
{
v___x_701_ = v___x_690_;
v_isShared_702_ = v_isSharedCheck_709_;
goto v_resetjp_700_;
}
else
{
lean_inc(v_a_699_);
lean_dec(v___x_690_);
v___x_701_ = lean_box(0);
v_isShared_702_ = v_isSharedCheck_709_;
goto v_resetjp_700_;
}
v_resetjp_700_:
{
lean_object* v___x_703_; lean_object* v___x_704_; uint8_t v___x_705_; lean_object* v___x_707_; 
v___x_703_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_703_, 0, v_a_626_);
lean_ctor_set(v___x_703_, 1, v_a_648_);
lean_ctor_set(v___x_703_, 2, v_a_669_);
lean_ctor_set(v___x_703_, 3, v_a_679_);
v___x_704_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_704_, 0, v___x_703_);
lean_ctor_set(v___x_704_, 1, v_a_699_);
v___x_705_ = lean_unbox(v_a_689_);
lean_dec(v_a_689_);
lean_ctor_set_uint8(v___x_704_, sizeof(void*)*2, v___x_705_);
if (v_isShared_702_ == 0)
{
lean_ctor_set(v___x_701_, 0, v___x_704_);
v___x_707_ = v___x_701_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v___x_704_);
v___x_707_ = v_reuseFailAlloc_708_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
return v___x_707_;
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1____boxed(lean_object* v_j_710_, lean_object* v_a_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_(v_j_710_, v_a_711_);
lean_dec_ref(v_a_711_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3(size_t v_sz_713_, size_t v_i_714_, lean_object* v_bs_715_, lean_object* v___y_716_){
_start:
{
lean_object* v___x_717_; 
v___x_717_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___redArg(v_sz_713_, v_i_714_, v_bs_715_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3___boxed(lean_object* v_sz_718_, lean_object* v_i_719_, lean_object* v_bs_720_, lean_object* v___y_721_){
_start:
{
size_t v_sz_boxed_722_; size_t v_i_boxed_723_; lean_object* v_res_724_; 
v_sz_boxed_722_ = lean_unbox_usize(v_sz_718_);
lean_dec(v_sz_718_);
v_i_boxed_723_ = lean_unbox_usize(v_i_719_);
lean_dec(v_i_719_);
v_res_724_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1__spec__3(v_sz_boxed_722_, v_i_boxed_723_, v_bs_720_, v___y_721_);
lean_dec_ref(v___y_721_);
return v_res_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4(lean_object* v_msgData_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_){
_start:
{
lean_object* v___x_737_; lean_object* v_env_738_; lean_object* v___x_739_; lean_object* v_mctx_740_; lean_object* v_lctx_741_; lean_object* v_options_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v___x_737_ = lean_st_ref_get(v___y_735_);
v_env_738_ = lean_ctor_get(v___x_737_, 0);
lean_inc_ref(v_env_738_);
lean_dec(v___x_737_);
v___x_739_ = lean_st_ref_get(v___y_733_);
v_mctx_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc_ref(v_mctx_740_);
lean_dec(v___x_739_);
v_lctx_741_ = lean_ctor_get(v___y_732_, 2);
v_options_742_ = lean_ctor_get(v___y_734_, 2);
lean_inc_ref(v_options_742_);
lean_inc_ref(v_lctx_741_);
v___x_743_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_743_, 0, v_env_738_);
lean_ctor_set(v___x_743_, 1, v_mctx_740_);
lean_ctor_set(v___x_743_, 2, v_lctx_741_);
lean_ctor_set(v___x_743_, 3, v_options_742_);
v___x_744_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_744_, 0, v___x_743_);
lean_ctor_set(v___x_744_, 1, v_msgData_731_);
v___x_745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4___boxed(lean_object* v_msgData_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_){
_start:
{
lean_object* v_res_752_; 
v_res_752_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4(v_msgData_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_748_);
lean_dec_ref(v___y_747_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(lean_object* v_msg_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_){
_start:
{
lean_object* v_ref_759_; lean_object* v___x_760_; lean_object* v_a_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_769_; 
v_ref_759_ = lean_ctor_get(v___y_756_, 5);
v___x_760_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4(v_msg_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_);
v_a_761_ = lean_ctor_get(v___x_760_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v___x_760_);
if (v_isSharedCheck_769_ == 0)
{
v___x_763_ = v___x_760_;
v_isShared_764_ = v_isSharedCheck_769_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_a_761_);
lean_dec(v___x_760_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_769_;
goto v_resetjp_762_;
}
v_resetjp_762_:
{
lean_object* v___x_765_; lean_object* v___x_767_; 
lean_inc(v_ref_759_);
v___x_765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_765_, 0, v_ref_759_);
lean_ctor_set(v___x_765_, 1, v_a_761_);
if (v_isShared_764_ == 0)
{
lean_ctor_set_tag(v___x_763_, 1);
lean_ctor_set(v___x_763_, 0, v___x_765_);
v___x_767_ = v___x_763_;
goto v_reusejp_766_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v___x_765_);
v___x_767_ = v_reuseFailAlloc_768_;
goto v_reusejp_766_;
}
v_reusejp_766_:
{
return v___x_767_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg___boxed(lean_object* v_msg_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_){
_start:
{
lean_object* v_res_776_; 
v_res_776_ = lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(v_msg_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_);
lean_dec(v___y_774_);
lean_dec_ref(v___y_773_);
lean_dec(v___y_772_);
lean_dec_ref(v___y_771_);
return v_res_776_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg(lean_object* v_as_777_, lean_object* v_bs_778_, lean_object* v_i_779_){
_start:
{
lean_object* v___x_780_; uint8_t v___x_781_; 
v___x_780_ = lean_array_get_size(v_as_777_);
v___x_781_ = lean_nat_dec_lt(v_i_779_, v___x_780_);
if (v___x_781_ == 0)
{
uint8_t v___x_782_; 
lean_dec(v_i_779_);
v___x_782_ = 1;
return v___x_782_;
}
else
{
lean_object* v_a_783_; lean_object* v_b_784_; uint8_t v___x_785_; 
v_a_783_ = lean_array_fget_borrowed(v_as_777_, v_i_779_);
v_b_784_ = lean_array_fget_borrowed(v_bs_778_, v_i_779_);
v___x_785_ = lean_nat_dec_eq(v_a_783_, v_b_784_);
if (v___x_785_ == 0)
{
lean_dec(v_i_779_);
return v___x_785_;
}
else
{
lean_object* v___x_786_; lean_object* v___x_787_; 
v___x_786_ = lean_unsigned_to_nat(1u);
v___x_787_ = lean_nat_add(v_i_779_, v___x_786_);
lean_dec(v_i_779_);
v_i_779_ = v___x_787_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg___boxed(lean_object* v_as_789_, lean_object* v_bs_790_, lean_object* v_i_791_){
_start:
{
uint8_t v_res_792_; lean_object* v_r_793_; 
v_res_792_ = lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg(v_as_789_, v_bs_790_, v_i_791_);
lean_dec_ref(v_bs_790_);
lean_dec_ref(v_as_789_);
v_r_793_ = lean_box(v_res_792_);
return v_r_793_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0(lean_object* v_as_794_, lean_object* v_bs_795_){
_start:
{
lean_object* v___x_796_; lean_object* v___x_797_; uint8_t v___x_798_; 
v___x_796_ = lean_array_get_size(v_as_794_);
v___x_797_ = lean_array_get_size(v_bs_795_);
v___x_798_ = lean_nat_dec_le(v___x_796_, v___x_797_);
if (v___x_798_ == 0)
{
return v___x_798_;
}
else
{
lean_object* v___x_799_; uint8_t v___x_800_; 
v___x_799_ = lean_unsigned_to_nat(0u);
v___x_800_ = lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg(v_as_794_, v_bs_795_, v___x_799_);
return v___x_800_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0___boxed(lean_object* v_as_801_, lean_object* v_bs_802_){
_start:
{
uint8_t v_res_803_; lean_object* v_r_804_; 
v_res_803_ = lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0(v_as_801_, v_bs_802_);
lean_dec_ref(v_bs_802_);
lean_dec_ref(v_as_801_);
v_r_804_ = lean_box(v_res_803_);
return v_r_804_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4(lean_object* v_as_808_, size_t v_i_809_, size_t v_stop_810_){
_start:
{
uint8_t v___x_811_; 
v___x_811_ = lean_usize_dec_eq(v_i_809_, v_stop_810_);
if (v___x_811_ == 0)
{
lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; uint8_t v___x_815_; 
v___x_812_ = lean_array_uget_borrowed(v_as_808_, v_i_809_);
v___x_813_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4___closed__0));
v___x_814_ = l_Lean_SubExpr_Pos_toArray(v___x_812_);
v___x_815_ = lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0(v___x_813_, v___x_814_);
lean_dec_ref(v___x_814_);
if (v___x_815_ == 0)
{
size_t v___x_816_; size_t v___x_817_; 
v___x_816_ = ((size_t)1ULL);
v___x_817_ = lean_usize_add(v_i_809_, v___x_816_);
v_i_809_ = v___x_817_;
goto _start;
}
else
{
return v___x_815_;
}
}
else
{
uint8_t v___x_819_; 
v___x_819_ = 0;
return v___x_819_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4___boxed(lean_object* v_as_820_, lean_object* v_i_821_, lean_object* v_stop_822_){
_start:
{
size_t v_i_boxed_823_; size_t v_stop_boxed_824_; uint8_t v_res_825_; lean_object* v_r_826_; 
v_i_boxed_823_ = lean_unbox_usize(v_i_821_);
lean_dec(v_i_821_);
v_stop_boxed_824_ = lean_unbox_usize(v_stop_822_);
lean_dec(v_stop_822_);
v_res_825_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4(v_as_820_, v_i_boxed_823_, v_stop_boxed_824_);
lean_dec_ref(v_as_820_);
v_r_826_ = lean_box(v_res_825_);
return v_r_826_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5(lean_object* v_as_833_, size_t v_i_834_, size_t v_stop_835_){
_start:
{
uint8_t v___x_836_; 
v___x_836_ = lean_usize_dec_eq(v_i_834_, v_stop_835_);
if (v___x_836_ == 0)
{
lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; uint8_t v___x_840_; 
v___x_837_ = lean_array_uget_borrowed(v_as_833_, v_i_834_);
v___x_838_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5___closed__0));
v___x_839_ = l_Lean_SubExpr_Pos_toArray(v___x_837_);
v___x_840_ = lp_mathlib_Array_isPrefixOf___at___00suggestSteps_spec__0(v___x_838_, v___x_839_);
lean_dec_ref(v___x_839_);
if (v___x_840_ == 0)
{
size_t v___x_841_; size_t v___x_842_; 
v___x_841_ = ((size_t)1ULL);
v___x_842_ = lean_usize_add(v_i_834_, v___x_841_);
v_i_834_ = v___x_842_;
goto _start;
}
else
{
return v___x_840_;
}
}
else
{
uint8_t v___x_844_; 
v___x_844_ = 0;
return v___x_844_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5___boxed(lean_object* v_as_845_, lean_object* v_i_846_, lean_object* v_stop_847_){
_start:
{
size_t v_i_boxed_848_; size_t v_stop_boxed_849_; uint8_t v_res_850_; lean_object* v_r_851_; 
v_i_boxed_848_ = lean_unbox_usize(v_i_846_);
lean_dec(v_i_846_);
v_stop_boxed_849_ = lean_unbox_usize(v_stop_847_);
lean_dec(v_stop_847_);
v_res_850_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5(v_as_845_, v_i_boxed_848_, v_stop_boxed_849_);
lean_dec_ref(v_as_845_);
v_r_851_ = lean_box(v_res_850_);
return v_r_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00suggestSteps_spec__1(lean_object* v_as_852_, size_t v_sz_853_, size_t v_i_854_, lean_object* v_b_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_){
_start:
{
uint8_t v___x_861_; 
v___x_861_ = lean_usize_dec_lt(v_i_854_, v_sz_853_);
if (v___x_861_ == 0)
{
lean_object* v___x_862_; 
v___x_862_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_862_, 0, v_b_855_);
return v___x_862_;
}
else
{
lean_object* v_a_863_; lean_object* v___x_864_; 
v_a_863_ = lean_array_uget_borrowed(v_as_852_, v_i_854_);
v___x_864_ = lp_mathlib_insertMetaVar(v_b_855_, v_a_863_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
if (lean_obj_tag(v___x_864_) == 0)
{
lean_object* v_a_865_; size_t v___x_866_; size_t v___x_867_; 
v_a_865_ = lean_ctor_get(v___x_864_, 0);
lean_inc(v_a_865_);
lean_dec_ref_known(v___x_864_, 1);
v___x_866_ = ((size_t)1ULL);
v___x_867_ = lean_usize_add(v_i_854_, v___x_866_);
v_i_854_ = v___x_867_;
v_b_855_ = v_a_865_;
goto _start;
}
else
{
return v___x_864_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00suggestSteps_spec__1___boxed(lean_object* v_as_869_, lean_object* v_sz_870_, lean_object* v_i_871_, lean_object* v_b_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_){
_start:
{
size_t v_sz_boxed_878_; size_t v_i_boxed_879_; lean_object* v_res_880_; 
v_sz_boxed_878_ = lean_unbox_usize(v_sz_870_);
lean_dec(v_sz_870_);
v_i_boxed_879_ = lean_unbox_usize(v_i_871_);
lean_dec(v_i_871_);
v_res_880_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00suggestSteps_spec__1(v_as_869_, v_sz_boxed_878_, v_i_boxed_879_, v_b_872_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
lean_dec(v___y_876_);
lean_dec_ref(v___y_875_);
lean_dec(v___y_874_);
lean_dec_ref(v___y_873_);
lean_dec_ref(v_as_869_);
return v_res_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg(lean_object* v___x_881_, lean_object* v___y_882_, lean_object* v_a_883_, lean_object* v_b_884_){
_start:
{
lean_object* v_startInclusive_885_; lean_object* v_endExclusive_886_; lean_object* v___x_887_; uint8_t v___x_888_; 
v_startInclusive_885_ = lean_ctor_get(v___x_881_, 1);
v_endExclusive_886_ = lean_ctor_get(v___x_881_, 2);
v___x_887_ = lean_nat_sub(v_endExclusive_886_, v_startInclusive_885_);
v___x_888_ = lean_nat_dec_eq(v_a_883_, v___x_887_);
lean_dec(v___x_887_);
if (v___x_888_ == 0)
{
uint32_t v___x_889_; uint32_t v___x_890_; uint8_t v___x_891_; 
v___x_889_ = lean_string_utf8_get_fast(v___y_882_, v_a_883_);
v___x_890_ = 63;
v___x_891_ = lean_uint32_dec_eq(v___x_889_, v___x_890_);
if (v___x_891_ == 0)
{
lean_object* v___x_892_; lean_object* v___x_893_; 
v___x_892_ = lean_box(0);
v___x_893_ = lean_string_utf8_next_fast(v___y_882_, v_a_883_);
lean_dec(v_a_883_);
v_a_883_ = v___x_893_;
v_b_884_ = v___x_892_;
goto _start;
}
else
{
lean_object* v___x_895_; 
v___x_895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_895_, 0, v_a_883_);
return v___x_895_;
}
}
else
{
lean_dec(v_a_883_);
lean_inc(v_b_884_);
return v_b_884_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg___boxed(lean_object* v___x_896_, lean_object* v___y_897_, lean_object* v_a_898_, lean_object* v_b_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg(v___x_896_, v___y_897_, v_a_898_, v_b_899_);
lean_dec(v_b_899_);
lean_dec_ref(v___y_897_);
lean_dec_ref(v___x_896_);
return v_res_900_;
}
}
static lean_object* _init_lp_mathlib_suggestSteps___closed__7(void){
_start:
{
lean_object* v___x_908_; lean_object* v___x_909_; 
v___x_908_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__6));
v___x_909_ = l_Lean_stringToMessageData(v___x_908_);
return v___x_909_;
}
}
static lean_object* _init_lp_mathlib_suggestSteps___closed__9(void){
_start:
{
lean_object* v___x_911_; lean_object* v___x_912_; 
v___x_911_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__8));
v___x_912_ = l_Lean_stringToMessageData(v___x_911_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_suggestSteps(lean_object* v_pos_913_, lean_object* v_goalType_914_, lean_object* v_params_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_a_918_, lean_object* v_a_919_){
_start:
{
lean_object* v___y_922_; lean_object* v___y_923_; lean_object* v___y_924_; lean_object* v___x_932_; 
v___x_932_ = l_Lean_Elab_Term_getCalcRelation_x3f___redArg(v_goalType_914_);
if (lean_obj_tag(v___x_932_) == 0)
{
lean_object* v_a_933_; 
v_a_933_ = lean_ctor_get(v___x_932_, 0);
lean_inc(v_a_933_);
lean_dec_ref_known(v___x_932_, 1);
if (lean_obj_tag(v_a_933_) == 1)
{
lean_object* v_val_934_; lean_object* v_snd_935_; lean_object* v_fst_936_; lean_object* v_fst_937_; lean_object* v_snd_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_1200_; 
v_val_934_ = lean_ctor_get(v_a_933_, 0);
lean_inc(v_val_934_);
lean_dec_ref_known(v_a_933_, 1);
v_snd_935_ = lean_ctor_get(v_val_934_, 1);
lean_inc(v_snd_935_);
v_fst_936_ = lean_ctor_get(v_val_934_, 0);
lean_inc(v_fst_936_);
lean_dec(v_val_934_);
v_fst_937_ = lean_ctor_get(v_snd_935_, 0);
v_snd_938_ = lean_ctor_get(v_snd_935_, 1);
v_isSharedCheck_1200_ = !lean_is_exclusive(v_snd_935_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_940_ = v_snd_935_;
v_isShared_941_ = v_isSharedCheck_1200_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_snd_938_);
lean_inc(v_fst_937_);
lean_dec(v_snd_935_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_1200_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v___x_942_; uint8_t v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_942_ = lean_box(0);
v___x_943_ = 0;
v___x_944_ = lean_box(0);
v___x_945_ = l_Lean_Meta_mkFreshExprMVar(v___x_942_, v___x_943_, v___x_944_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_945_) == 0)
{
lean_object* v_a_946_; lean_object* v___x_947_; 
v_a_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc(v_a_946_);
lean_dec_ref_known(v___x_945_, 1);
v___x_947_ = l_Lean_Meta_mkFreshExprMVar(v___x_942_, v___x_943_, v___x_944_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_947_) == 0)
{
lean_object* v_a_948_; lean_object* v___x_949_; lean_object* v___x_950_; 
v_a_948_ = lean_ctor_get(v___x_947_, 0);
lean_inc(v_a_948_);
lean_dec_ref_known(v___x_947_, 1);
v___x_949_ = l_Lean_mkAppB(v_fst_936_, v_a_946_, v_a_948_);
lean_inc_ref(v___x_949_);
v___x_950_ = l_Lean_Meta_ppExpr(v___x_949_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_950_) == 0)
{
lean_object* v_a_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; 
v_a_951_ = lean_ctor_get(v___x_950_, 0);
lean_inc(v_a_951_);
lean_dec_ref_known(v___x_950_, 1);
v___x_952_ = l_Std_Format_defWidth;
v___x_953_ = lean_unsigned_to_nat(0u);
v___x_954_ = l_Std_Format_pretty(v_a_951_, v___x_952_, v___x_953_, v___x_953_);
v___x_955_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__0));
v___x_956_ = lean_box(0);
v___x_957_ = l_String_splitOnAux(v___x_954_, v___x_955_, v___x_953_, v___x_953_, v___x_953_, v___x_956_);
lean_dec_ref(v___x_954_);
v___x_958_ = lean_unsigned_to_nat(1u);
v___x_959_ = l_List_get_x3fInternal___redArg(v___x_957_, v___x_958_);
lean_dec(v___x_957_);
if (lean_obj_tag(v___x_959_) == 1)
{
lean_object* v_val_960_; lean_object* v_subexprPos_961_; lean_object* v___y_963_; lean_object* v___y_964_; lean_object* v___y_970_; lean_object* v___y_973_; lean_object* v___y_976_; uint8_t v___y_979_; uint8_t v___y_980_; lean_object* v___x_1159_; uint8_t v___y_1161_; uint8_t v___x_1166_; 
lean_dec_ref(v___x_949_);
v_val_960_ = lean_ctor_get(v___x_959_, 0);
lean_inc(v_val_960_);
lean_dec_ref_known(v___x_959_, 1);
v_subexprPos_961_ = lp_mathlib_getGoalLocations(v_pos_913_);
v___x_1159_ = lean_array_get_size(v_subexprPos_961_);
v___x_1166_ = lean_nat_dec_lt(v___x_953_, v___x_1159_);
if (v___x_1166_ == 0)
{
v___y_1161_ = v___x_1166_;
goto v___jp_1160_;
}
else
{
if (v___x_1166_ == 0)
{
v___y_1161_ = v___x_1166_;
goto v___jp_1160_;
}
else
{
size_t v___x_1167_; size_t v___x_1168_; uint8_t v___x_1169_; 
v___x_1167_ = ((size_t)0ULL);
v___x_1168_ = lean_usize_of_nat(v___x_1159_);
v___x_1169_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__5(v_subexprPos_961_, v___x_1167_, v___x_1168_);
v___y_1161_ = v___x_1169_;
goto v___jp_1160_;
}
}
v___jp_962_:
{
lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; 
v___x_965_ = lean_string_utf8_byte_size(v___y_963_);
lean_inc_ref(v___y_963_);
v___x_966_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_966_, 0, v___y_963_);
lean_ctor_set(v___x_966_, 1, v___x_953_);
lean_ctor_set(v___x_966_, 2, v___x_965_);
v___x_967_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg(v___x_966_, v___y_963_, v___x_953_, v___x_942_);
lean_dec_ref_known(v___x_966_, 3);
if (lean_obj_tag(v___x_967_) == 0)
{
v___y_922_ = v___y_964_;
v___y_923_ = v___y_963_;
v___y_924_ = v___x_965_;
goto v___jp_921_;
}
else
{
lean_object* v_val_968_; 
v_val_968_ = lean_ctor_get(v___x_967_, 0);
lean_inc(v_val_968_);
lean_dec_ref_known(v___x_967_, 1);
v___y_922_ = v___y_964_;
v___y_923_ = v___y_963_;
v___y_924_ = v_val_968_;
goto v___jp_921_;
}
}
v___jp_969_:
{
lean_object* v___x_971_; 
v___x_971_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__1));
v___y_963_ = v___y_970_;
v___y_964_ = v___x_971_;
goto v___jp_962_;
}
v___jp_972_:
{
lean_object* v___x_974_; 
v___x_974_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__2));
v___y_963_ = v___y_973_;
v___y_964_ = v___x_974_;
goto v___jp_962_;
}
v___jp_975_:
{
lean_object* v___x_977_; 
v___x_977_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__2));
v___y_963_ = v___y_976_;
v___y_964_ = v___x_977_;
goto v___jp_962_;
}
v___jp_978_:
{
size_t v_sz_981_; size_t v___x_982_; lean_object* v___x_983_; 
v_sz_981_ = lean_array_size(v_subexprPos_961_);
v___x_982_ = ((size_t)0ULL);
v___x_983_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00suggestSteps_spec__1(v_subexprPos_961_, v_sz_981_, v___x_982_, v_goalType_914_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
lean_dec_ref(v_subexprPos_961_);
if (lean_obj_tag(v___x_983_) == 0)
{
lean_object* v_a_984_; lean_object* v___x_985_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
lean_inc(v_a_984_);
lean_dec_ref_known(v___x_983_, 1);
v___x_985_ = l_Lean_Elab_Term_getCalcRelation_x3f___redArg(v_a_984_);
if (lean_obj_tag(v___x_985_) == 0)
{
lean_object* v_a_986_; 
v_a_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref_known(v___x_985_, 1);
if (lean_obj_tag(v_a_986_) == 1)
{
lean_object* v_val_987_; lean_object* v_snd_988_; lean_object* v_fst_989_; lean_object* v_snd_990_; lean_object* v___x_991_; 
lean_dec(v_a_984_);
lean_del_object(v___x_940_);
v_val_987_ = lean_ctor_get(v_a_986_, 0);
lean_inc(v_val_987_);
lean_dec_ref_known(v_a_986_, 1);
v_snd_988_ = lean_ctor_get(v_val_987_, 1);
lean_inc(v_snd_988_);
lean_dec(v_val_987_);
v_fst_989_ = lean_ctor_get(v_snd_988_, 0);
lean_inc(v_fst_989_);
v_snd_990_ = lean_ctor_get(v_snd_988_, 1);
lean_inc(v_snd_990_);
lean_dec(v_snd_988_);
v___x_991_ = l_Lean_Meta_ppExpr(v_fst_937_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_991_) == 0)
{
lean_object* v_a_992_; lean_object* v___x_993_; lean_object* v___x_994_; 
v_a_992_ = lean_ctor_get(v___x_991_, 0);
lean_inc(v_a_992_);
lean_dec_ref_known(v___x_991_, 1);
v___x_993_ = l_Std_Format_pretty(v_a_992_, v___x_952_, v___x_953_, v___x_953_);
v___x_994_ = l_Lean_Meta_ppExpr(v_fst_989_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_994_) == 0)
{
lean_object* v_a_995_; lean_object* v___x_996_; lean_object* v___x_997_; 
v_a_995_ = lean_ctor_get(v___x_994_, 0);
lean_inc(v_a_995_);
lean_dec_ref_known(v___x_994_, 1);
v___x_996_ = l_Std_Format_pretty(v_a_995_, v___x_952_, v___x_953_, v___x_953_);
v___x_997_ = l_Lean_Meta_ppExpr(v_snd_938_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_997_) == 0)
{
lean_object* v_a_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
v_a_998_ = lean_ctor_get(v___x_997_, 0);
lean_inc(v_a_998_);
lean_dec_ref_known(v___x_997_, 1);
v___x_999_ = l_Std_Format_pretty(v_a_998_, v___x_952_, v___x_953_, v___x_953_);
v___x_1000_ = l_Lean_Meta_ppExpr(v_snd_990_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_1000_) == 0)
{
lean_object* v_a_1001_; lean_object* v___x_1002_; uint8_t v_isFirst_1003_; lean_object* v_indent_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; uint32_t v___x_1008_; lean_object* v___x_1009_; 
v_a_1001_ = lean_ctor_get(v___x_1000_, 0);
lean_inc(v_a_1001_);
lean_dec_ref_known(v___x_1000_, 1);
v___x_1002_ = l_Std_Format_pretty(v_a_1001_, v___x_952_, v___x_953_, v___x_953_);
v_isFirst_1003_ = lean_ctor_get_uint8(v_params_915_, sizeof(void*)*2);
v_indent_1004_ = lean_ctor_get(v_params_915_, 1);
lean_inc(v_indent_1004_);
lean_dec_ref(v_params_915_);
v___x_1005_ = lp_mathlib_String_renameMetaVar(v___x_993_);
lean_dec_ref(v___x_993_);
v___x_1006_ = lp_mathlib_String_renameMetaVar(v___x_999_);
lean_dec_ref(v___x_999_);
v___x_1007_ = lp_mathlib_String_renameMetaVar(v___x_1002_);
lean_dec_ref(v___x_1002_);
v___x_1008_ = 32;
v___x_1009_ = lp_mathlib_String_replicate(v_indent_1004_, v___x_1008_);
if (v___y_979_ == 0)
{
lean_dec_ref(v___x_996_);
if (v___y_980_ == 0)
{
lean_object* v___x_1010_; 
lean_dec_ref(v___x_1009_);
lean_dec_ref(v___x_1007_);
lean_dec_ref(v___x_1006_);
lean_dec_ref(v___x_1005_);
lean_dec(v_val_960_);
v___x_1010_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__3));
v___y_963_ = v___x_1010_;
v___y_964_ = v___x_1010_;
goto v___jp_962_;
}
else
{
if (v_isFirst_1003_ == 0)
{
lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; 
lean_dec_ref(v___x_1005_);
v___x_1011_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__4));
v___x_1012_ = lean_string_append(v___x_1011_, v_val_960_);
v___x_1013_ = lean_string_append(v___x_1012_, v___x_955_);
v___x_1014_ = lean_string_append(v___x_1013_, v___x_1007_);
lean_dec_ref(v___x_1007_);
v___x_1015_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__5));
v___x_1016_ = lean_string_append(v___x_1014_, v___x_1015_);
v___x_1017_ = lean_string_append(v___x_1016_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1018_ = lean_string_append(v___x_1017_, v___x_1011_);
v___x_1019_ = lean_string_append(v___x_1018_, v_val_960_);
lean_dec(v_val_960_);
v___x_1020_ = lean_string_append(v___x_1019_, v___x_955_);
v___x_1021_ = lean_string_append(v___x_1020_, v___x_1006_);
lean_dec_ref(v___x_1006_);
v___x_1022_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_1023_ = lean_string_append(v___x_1021_, v___x_1022_);
v___y_976_ = v___x_1023_;
goto v___jp_975_;
}
else
{
lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; 
v___x_1024_ = lean_string_append(v___x_1005_, v___x_955_);
v___x_1025_ = lean_string_append(v___x_1024_, v_val_960_);
v___x_1026_ = lean_string_append(v___x_1025_, v___x_955_);
v___x_1027_ = lean_string_append(v___x_1026_, v___x_1007_);
lean_dec_ref(v___x_1007_);
v___x_1028_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__5));
v___x_1029_ = lean_string_append(v___x_1027_, v___x_1028_);
v___x_1030_ = lean_string_append(v___x_1029_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1031_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__4));
v___x_1032_ = lean_string_append(v___x_1030_, v___x_1031_);
v___x_1033_ = lean_string_append(v___x_1032_, v_val_960_);
lean_dec(v_val_960_);
v___x_1034_ = lean_string_append(v___x_1033_, v___x_955_);
v___x_1035_ = lean_string_append(v___x_1034_, v___x_1006_);
lean_dec_ref(v___x_1006_);
v___x_1036_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_1037_ = lean_string_append(v___x_1035_, v___x_1036_);
v___y_976_ = v___x_1037_;
goto v___jp_975_;
}
}
}
else
{
lean_object* v___x_1038_; 
v___x_1038_ = lp_mathlib_String_renameMetaVar(v___x_996_);
lean_dec_ref(v___x_996_);
if (v___y_980_ == 0)
{
lean_dec_ref(v___x_1007_);
if (v_isFirst_1003_ == 0)
{
lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; 
lean_dec_ref(v___x_1005_);
v___x_1039_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__4));
v___x_1040_ = lean_string_append(v___x_1039_, v_val_960_);
v___x_1041_ = lean_string_append(v___x_1040_, v___x_955_);
v___x_1042_ = lean_string_append(v___x_1041_, v___x_1038_);
lean_dec_ref(v___x_1038_);
v___x_1043_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__5));
v___x_1044_ = lean_string_append(v___x_1042_, v___x_1043_);
v___x_1045_ = lean_string_append(v___x_1044_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1046_ = lean_string_append(v___x_1045_, v___x_1039_);
v___x_1047_ = lean_string_append(v___x_1046_, v_val_960_);
lean_dec(v_val_960_);
v___x_1048_ = lean_string_append(v___x_1047_, v___x_955_);
v___x_1049_ = lean_string_append(v___x_1048_, v___x_1006_);
lean_dec_ref(v___x_1006_);
v___x_1050_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_1051_ = lean_string_append(v___x_1049_, v___x_1050_);
v___y_973_ = v___x_1051_;
goto v___jp_972_;
}
else
{
lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; 
v___x_1052_ = lean_string_append(v___x_1005_, v___x_955_);
v___x_1053_ = lean_string_append(v___x_1052_, v_val_960_);
v___x_1054_ = lean_string_append(v___x_1053_, v___x_955_);
v___x_1055_ = lean_string_append(v___x_1054_, v___x_1038_);
lean_dec_ref(v___x_1038_);
v___x_1056_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__5));
v___x_1057_ = lean_string_append(v___x_1055_, v___x_1056_);
v___x_1058_ = lean_string_append(v___x_1057_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1059_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__4));
v___x_1060_ = lean_string_append(v___x_1058_, v___x_1059_);
v___x_1061_ = lean_string_append(v___x_1060_, v_val_960_);
lean_dec(v_val_960_);
v___x_1062_ = lean_string_append(v___x_1061_, v___x_955_);
v___x_1063_ = lean_string_append(v___x_1062_, v___x_1006_);
lean_dec_ref(v___x_1006_);
v___x_1064_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_1065_ = lean_string_append(v___x_1063_, v___x_1064_);
v___y_973_ = v___x_1065_;
goto v___jp_972_;
}
}
else
{
if (v_isFirst_1003_ == 0)
{
lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; 
lean_dec_ref(v___x_1005_);
v___x_1066_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__4));
v___x_1067_ = lean_string_append(v___x_1066_, v_val_960_);
v___x_1068_ = lean_string_append(v___x_1067_, v___x_955_);
v___x_1069_ = lean_string_append(v___x_1068_, v___x_1038_);
lean_dec_ref(v___x_1038_);
v___x_1070_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__5));
v___x_1071_ = lean_string_append(v___x_1069_, v___x_1070_);
v___x_1072_ = lean_string_append(v___x_1071_, v___x_1009_);
v___x_1073_ = lean_string_append(v___x_1072_, v___x_1066_);
v___x_1074_ = lean_string_append(v___x_1073_, v_val_960_);
v___x_1075_ = lean_string_append(v___x_1074_, v___x_955_);
v___x_1076_ = lean_string_append(v___x_1075_, v___x_1007_);
lean_dec_ref(v___x_1007_);
v___x_1077_ = lean_string_append(v___x_1076_, v___x_1070_);
v___x_1078_ = lean_string_append(v___x_1077_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1079_ = lean_string_append(v___x_1078_, v___x_1066_);
v___x_1080_ = lean_string_append(v___x_1079_, v_val_960_);
lean_dec(v_val_960_);
v___x_1081_ = lean_string_append(v___x_1080_, v___x_955_);
v___x_1082_ = lean_string_append(v___x_1081_, v___x_1006_);
lean_dec_ref(v___x_1006_);
v___x_1083_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_1084_ = lean_string_append(v___x_1082_, v___x_1083_);
v___y_970_ = v___x_1084_;
goto v___jp_969_;
}
else
{
lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; 
v___x_1085_ = lean_string_append(v___x_1005_, v___x_955_);
v___x_1086_ = lean_string_append(v___x_1085_, v_val_960_);
v___x_1087_ = lean_string_append(v___x_1086_, v___x_955_);
v___x_1088_ = lean_string_append(v___x_1087_, v___x_1038_);
lean_dec_ref(v___x_1038_);
v___x_1089_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__5));
v___x_1090_ = lean_string_append(v___x_1088_, v___x_1089_);
v___x_1091_ = lean_string_append(v___x_1090_, v___x_1009_);
v___x_1092_ = ((lean_object*)(lp_mathlib_suggestSteps___closed__4));
v___x_1093_ = lean_string_append(v___x_1091_, v___x_1092_);
v___x_1094_ = lean_string_append(v___x_1093_, v_val_960_);
v___x_1095_ = lean_string_append(v___x_1094_, v___x_955_);
v___x_1096_ = lean_string_append(v___x_1095_, v___x_1007_);
lean_dec_ref(v___x_1007_);
v___x_1097_ = lean_string_append(v___x_1096_, v___x_1089_);
v___x_1098_ = lean_string_append(v___x_1097_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1099_ = lean_string_append(v___x_1098_, v___x_1092_);
v___x_1100_ = lean_string_append(v___x_1099_, v_val_960_);
lean_dec(v_val_960_);
v___x_1101_ = lean_string_append(v___x_1100_, v___x_955_);
v___x_1102_ = lean_string_append(v___x_1101_, v___x_1006_);
lean_dec_ref(v___x_1006_);
v___x_1103_ = ((lean_object*)(lp_mathlib_createCalc___redArg___lam__1___closed__1));
v___x_1104_ = lean_string_append(v___x_1102_, v___x_1103_);
v___y_970_ = v___x_1104_;
goto v___jp_969_;
}
}
}
}
else
{
lean_object* v_a_1105_; lean_object* v___x_1107_; uint8_t v_isShared_1108_; uint8_t v_isSharedCheck_1112_; 
lean_dec_ref(v___x_999_);
lean_dec_ref(v___x_996_);
lean_dec_ref(v___x_993_);
lean_dec(v_val_960_);
lean_dec_ref(v_params_915_);
v_a_1105_ = lean_ctor_get(v___x_1000_, 0);
v_isSharedCheck_1112_ = !lean_is_exclusive(v___x_1000_);
if (v_isSharedCheck_1112_ == 0)
{
v___x_1107_ = v___x_1000_;
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
else
{
lean_inc(v_a_1105_);
lean_dec(v___x_1000_);
v___x_1107_ = lean_box(0);
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
v_resetjp_1106_:
{
lean_object* v___x_1110_; 
if (v_isShared_1108_ == 0)
{
v___x_1110_ = v___x_1107_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v_a_1105_);
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
else
{
lean_object* v_a_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1120_; 
lean_dec_ref(v___x_996_);
lean_dec_ref(v___x_993_);
lean_dec(v_snd_990_);
lean_dec(v_val_960_);
lean_dec_ref(v_params_915_);
v_a_1113_ = lean_ctor_get(v___x_997_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_997_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1115_ = v___x_997_;
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_a_1113_);
lean_dec(v___x_997_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1118_; 
if (v_isShared_1116_ == 0)
{
v___x_1118_ = v___x_1115_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v_a_1113_);
v___x_1118_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
return v___x_1118_;
}
}
}
}
else
{
lean_object* v_a_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1128_; 
lean_dec_ref(v___x_993_);
lean_dec(v_snd_990_);
lean_dec(v_val_960_);
lean_dec(v_snd_938_);
lean_dec_ref(v_params_915_);
v_a_1121_ = lean_ctor_get(v___x_994_, 0);
v_isSharedCheck_1128_ = !lean_is_exclusive(v___x_994_);
if (v_isSharedCheck_1128_ == 0)
{
v___x_1123_ = v___x_994_;
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_a_1121_);
lean_dec(v___x_994_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1126_; 
if (v_isShared_1124_ == 0)
{
v___x_1126_ = v___x_1123_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v_a_1121_);
v___x_1126_ = v_reuseFailAlloc_1127_;
goto v_reusejp_1125_;
}
v_reusejp_1125_:
{
return v___x_1126_;
}
}
}
}
else
{
lean_object* v_a_1129_; lean_object* v___x_1131_; uint8_t v_isShared_1132_; uint8_t v_isSharedCheck_1136_; 
lean_dec(v_snd_990_);
lean_dec(v_fst_989_);
lean_dec(v_val_960_);
lean_dec(v_snd_938_);
lean_dec_ref(v_params_915_);
v_a_1129_ = lean_ctor_get(v___x_991_, 0);
v_isSharedCheck_1136_ = !lean_is_exclusive(v___x_991_);
if (v_isSharedCheck_1136_ == 0)
{
v___x_1131_ = v___x_991_;
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
else
{
lean_inc(v_a_1129_);
lean_dec(v___x_991_);
v___x_1131_ = lean_box(0);
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
v_resetjp_1130_:
{
lean_object* v___x_1134_; 
if (v_isShared_1132_ == 0)
{
v___x_1134_ = v___x_1131_;
goto v_reusejp_1133_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v_a_1129_);
v___x_1134_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1133_;
}
v_reusejp_1133_:
{
return v___x_1134_;
}
}
}
}
else
{
lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1140_; 
lean_dec(v_a_986_);
lean_dec(v_val_960_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec_ref(v_params_915_);
v___x_1137_ = lean_obj_once(&lp_mathlib_suggestSteps___closed__7, &lp_mathlib_suggestSteps___closed__7_once, _init_lp_mathlib_suggestSteps___closed__7);
v___x_1138_ = l_Lean_indentExpr(v_a_984_);
if (v_isShared_941_ == 0)
{
lean_ctor_set_tag(v___x_940_, 7);
lean_ctor_set(v___x_940_, 1, v___x_1138_);
lean_ctor_set(v___x_940_, 0, v___x_1137_);
v___x_1140_ = v___x_940_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1142_; 
v_reuseFailAlloc_1142_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1142_, 0, v___x_1137_);
lean_ctor_set(v_reuseFailAlloc_1142_, 1, v___x_1138_);
v___x_1140_ = v_reuseFailAlloc_1142_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
lean_object* v___x_1141_; 
v___x_1141_ = lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(v___x_1140_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
return v___x_1141_;
}
}
}
else
{
lean_object* v_a_1143_; lean_object* v___x_1145_; uint8_t v_isShared_1146_; uint8_t v_isSharedCheck_1150_; 
lean_dec(v_a_984_);
lean_dec(v_val_960_);
lean_del_object(v___x_940_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec_ref(v_params_915_);
v_a_1143_ = lean_ctor_get(v___x_985_, 0);
v_isSharedCheck_1150_ = !lean_is_exclusive(v___x_985_);
if (v_isSharedCheck_1150_ == 0)
{
v___x_1145_ = v___x_985_;
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
else
{
lean_inc(v_a_1143_);
lean_dec(v___x_985_);
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
else
{
lean_object* v_a_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1158_; 
lean_dec(v_val_960_);
lean_del_object(v___x_940_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec_ref(v_params_915_);
v_a_1151_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1158_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1158_ == 0)
{
v___x_1153_ = v___x_983_;
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_a_1151_);
lean_dec(v___x_983_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1156_; 
if (v_isShared_1154_ == 0)
{
v___x_1156_ = v___x_1153_;
goto v_reusejp_1155_;
}
else
{
lean_object* v_reuseFailAlloc_1157_; 
v_reuseFailAlloc_1157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1157_, 0, v_a_1151_);
v___x_1156_ = v_reuseFailAlloc_1157_;
goto v_reusejp_1155_;
}
v_reusejp_1155_:
{
return v___x_1156_;
}
}
}
}
v___jp_1160_:
{
uint8_t v___x_1162_; 
v___x_1162_ = lean_nat_dec_lt(v___x_953_, v___x_1159_);
if (v___x_1162_ == 0)
{
v___y_979_ = v___y_1161_;
v___y_980_ = v___x_1162_;
goto v___jp_978_;
}
else
{
if (v___x_1162_ == 0)
{
v___y_979_ = v___y_1161_;
v___y_980_ = v___x_1162_;
goto v___jp_978_;
}
else
{
size_t v___x_1163_; size_t v___x_1164_; uint8_t v___x_1165_; 
v___x_1163_ = ((size_t)0ULL);
v___x_1164_ = lean_usize_of_nat(v___x_1159_);
v___x_1165_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00suggestSteps_spec__4(v_subexprPos_961_, v___x_1163_, v___x_1164_);
v___y_979_ = v___y_1161_;
v___y_980_ = v___x_1165_;
goto v___jp_978_;
}
}
}
}
else
{
lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1173_; 
lean_dec(v___x_959_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec_ref(v_params_915_);
lean_dec_ref(v_goalType_914_);
v___x_1170_ = lean_obj_once(&lp_mathlib_suggestSteps___closed__9, &lp_mathlib_suggestSteps___closed__9_once, _init_lp_mathlib_suggestSteps___closed__9);
v___x_1171_ = l_Lean_MessageData_ofExpr(v___x_949_);
if (v_isShared_941_ == 0)
{
lean_ctor_set_tag(v___x_940_, 7);
lean_ctor_set(v___x_940_, 1, v___x_1171_);
lean_ctor_set(v___x_940_, 0, v___x_1170_);
v___x_1173_ = v___x_940_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v___x_1170_);
lean_ctor_set(v_reuseFailAlloc_1175_, 1, v___x_1171_);
v___x_1173_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
lean_object* v___x_1174_; 
v___x_1174_ = lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(v___x_1173_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
return v___x_1174_;
}
}
}
else
{
lean_object* v_a_1176_; lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1183_; 
lean_dec_ref(v___x_949_);
lean_del_object(v___x_940_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec_ref(v_params_915_);
lean_dec_ref(v_goalType_914_);
v_a_1176_ = lean_ctor_get(v___x_950_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_950_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1178_ = v___x_950_;
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
else
{
lean_inc(v_a_1176_);
lean_dec(v___x_950_);
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
else
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1191_; 
lean_dec(v_a_946_);
lean_del_object(v___x_940_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec(v_fst_936_);
lean_dec_ref(v_params_915_);
lean_dec_ref(v_goalType_914_);
v_a_1184_ = lean_ctor_get(v___x_947_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v___x_947_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1186_ = v___x_947_;
v_isShared_1187_ = v_isSharedCheck_1191_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_947_);
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
else
{
lean_object* v_a_1192_; lean_object* v___x_1194_; uint8_t v_isShared_1195_; uint8_t v_isSharedCheck_1199_; 
lean_del_object(v___x_940_);
lean_dec(v_snd_938_);
lean_dec(v_fst_937_);
lean_dec(v_fst_936_);
lean_dec_ref(v_params_915_);
lean_dec_ref(v_goalType_914_);
v_a_1192_ = lean_ctor_get(v___x_945_, 0);
v_isSharedCheck_1199_ = !lean_is_exclusive(v___x_945_);
if (v_isSharedCheck_1199_ == 0)
{
v___x_1194_ = v___x_945_;
v_isShared_1195_ = v_isSharedCheck_1199_;
goto v_resetjp_1193_;
}
else
{
lean_inc(v_a_1192_);
lean_dec(v___x_945_);
v___x_1194_ = lean_box(0);
v_isShared_1195_ = v_isSharedCheck_1199_;
goto v_resetjp_1193_;
}
v_resetjp_1193_:
{
lean_object* v___x_1197_; 
if (v_isShared_1195_ == 0)
{
v___x_1197_ = v___x_1194_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v_a_1192_);
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
else
{
lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; 
lean_dec(v_a_933_);
lean_dec_ref(v_params_915_);
v___x_1201_ = lean_obj_once(&lp_mathlib_suggestSteps___closed__7, &lp_mathlib_suggestSteps___closed__7_once, _init_lp_mathlib_suggestSteps___closed__7);
v___x_1202_ = l_Lean_indentExpr(v_goalType_914_);
v___x_1203_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1201_);
lean_ctor_set(v___x_1203_, 1, v___x_1202_);
v___x_1204_ = lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(v___x_1203_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
return v___x_1204_;
}
}
else
{
lean_object* v_a_1205_; lean_object* v___x_1207_; uint8_t v_isShared_1208_; uint8_t v_isSharedCheck_1212_; 
lean_dec_ref(v_params_915_);
lean_dec_ref(v_goalType_914_);
v_a_1205_ = lean_ctor_get(v___x_932_, 0);
v_isSharedCheck_1212_ = !lean_is_exclusive(v___x_932_);
if (v_isSharedCheck_1212_ == 0)
{
v___x_1207_ = v___x_932_;
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
else
{
lean_inc(v_a_1205_);
lean_dec(v___x_932_);
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
v___jp_921_:
{
lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; 
v___x_925_ = lean_unsigned_to_nat(2u);
v___x_926_ = lean_nat_add(v___y_924_, v___x_925_);
v___x_927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_927_, 0, v___y_924_);
lean_ctor_set(v___x_927_, 1, v___x_926_);
v___x_928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_928_, 0, v___x_927_);
v___x_929_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_929_, 0, v___y_923_);
lean_ctor_set(v___x_929_, 1, v___x_928_);
v___x_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_930_, 0, v___y_922_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
v___x_931_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_931_, 0, v___x_930_);
return v___x_931_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_suggestSteps___boxed(lean_object* v_pos_1213_, lean_object* v_goalType_1214_, lean_object* v_params_1215_, lean_object* v_a_1216_, lean_object* v_a_1217_, lean_object* v_a_1218_, lean_object* v_a_1219_, lean_object* v_a_1220_){
_start:
{
lean_object* v_res_1221_; 
v_res_1221_ = lp_mathlib_suggestSteps(v_pos_1213_, v_goalType_1214_, v_params_1215_, v_a_1216_, v_a_1217_, v_a_1218_, v_a_1219_);
lean_dec(v_a_1219_);
lean_dec_ref(v_a_1218_);
lean_dec(v_a_1217_);
lean_dec_ref(v_a_1216_);
lean_dec_ref(v_pos_1213_);
return v_res_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2(lean_object* v___x_1222_, lean_object* v___y_1223_, lean_object* v_inst_1224_, lean_object* v_R_1225_, lean_object* v_a_1226_, lean_object* v_b_1227_, lean_object* v_c_1228_){
_start:
{
lean_object* v___x_1229_; 
v___x_1229_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___redArg(v___x_1222_, v___y_1223_, v_a_1226_, v_b_1227_);
return v___x_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2___boxed(lean_object* v___x_1230_, lean_object* v___y_1231_, lean_object* v_inst_1232_, lean_object* v_R_1233_, lean_object* v_a_1234_, lean_object* v_b_1235_, lean_object* v_c_1236_){
_start:
{
lean_object* v_res_1237_; 
v_res_1237_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00suggestSteps_spec__2(v___x_1230_, v___y_1231_, v_inst_1232_, v_R_1233_, v_a_1234_, v_b_1235_, v_c_1236_);
lean_dec(v_b_1235_);
lean_dec_ref(v___y_1231_);
lean_dec_ref(v___x_1230_);
return v_res_1237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3(lean_object* v_00_u03b1_1238_, lean_object* v_msg_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_){
_start:
{
lean_object* v___x_1245_; 
v___x_1245_ = lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___redArg(v_msg_1239_, v___y_1240_, v___y_1241_, v___y_1242_, v___y_1243_);
return v___x_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3___boxed(lean_object* v_00_u03b1_1246_, lean_object* v_msg_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_){
_start:
{
lean_object* v_res_1253_; 
v_res_1253_ = lp_mathlib_Lean_throwError___at___00suggestSteps_spec__3(v_00_u03b1_1246_, v_msg_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_);
lean_dec(v___y_1251_);
lean_dec_ref(v___y_1250_);
lean_dec(v___y_1249_);
lean_dec_ref(v___y_1248_);
return v_res_1253_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0(lean_object* v_as_1254_, lean_object* v_bs_1255_, lean_object* v_hle_1256_, lean_object* v_i_1257_){
_start:
{
uint8_t v___x_1258_; 
v___x_1258_ = lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___redArg(v_as_1254_, v_bs_1255_, v_i_1257_);
return v___x_1258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0___boxed(lean_object* v_as_1259_, lean_object* v_bs_1260_, lean_object* v_hle_1261_, lean_object* v_i_1262_){
_start:
{
uint8_t v_res_1263_; lean_object* v_r_1264_; 
v_res_1263_ = lp_mathlib_Array_isPrefixOfAux___at___00Array_isPrefixOf___at___00suggestSteps_spec__0_spec__0(v_as_1259_, v_bs_1260_, v_hle_1261_, v_i_1262_);
lean_dec_ref(v_bs_1260_);
lean_dec_ref(v_as_1259_);
v_r_1264_ = lean_box(v_res_1263_);
return v_r_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg(lean_object* v_lctx_1265_, lean_object* v_localInsts_1266_, lean_object* v_x_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_){
_start:
{
lean_object* v___x_1273_; 
v___x_1273_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_1265_, v_localInsts_1266_, v_x_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_);
if (lean_obj_tag(v___x_1273_) == 0)
{
lean_object* v_a_1274_; lean_object* v___x_1276_; uint8_t v_isShared_1277_; uint8_t v_isSharedCheck_1281_; 
v_a_1274_ = lean_ctor_get(v___x_1273_, 0);
v_isSharedCheck_1281_ = !lean_is_exclusive(v___x_1273_);
if (v_isSharedCheck_1281_ == 0)
{
v___x_1276_ = v___x_1273_;
v_isShared_1277_ = v_isSharedCheck_1281_;
goto v_resetjp_1275_;
}
else
{
lean_inc(v_a_1274_);
lean_dec(v___x_1273_);
v___x_1276_ = lean_box(0);
v_isShared_1277_ = v_isSharedCheck_1281_;
goto v_resetjp_1275_;
}
v_resetjp_1275_:
{
lean_object* v___x_1279_; 
if (v_isShared_1277_ == 0)
{
v___x_1279_ = v___x_1276_;
goto v_reusejp_1278_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v_a_1274_);
v___x_1279_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1278_;
}
v_reusejp_1278_:
{
return v___x_1279_;
}
}
}
else
{
lean_object* v_a_1282_; lean_object* v___x_1284_; uint8_t v_isShared_1285_; uint8_t v_isSharedCheck_1289_; 
v_a_1282_ = lean_ctor_get(v___x_1273_, 0);
v_isSharedCheck_1289_ = !lean_is_exclusive(v___x_1273_);
if (v_isSharedCheck_1289_ == 0)
{
v___x_1284_ = v___x_1273_;
v_isShared_1285_ = v_isSharedCheck_1289_;
goto v_resetjp_1283_;
}
else
{
lean_inc(v_a_1282_);
lean_dec(v___x_1273_);
v___x_1284_ = lean_box(0);
v_isShared_1285_ = v_isSharedCheck_1289_;
goto v_resetjp_1283_;
}
v_resetjp_1283_:
{
lean_object* v___x_1287_; 
if (v_isShared_1285_ == 0)
{
v___x_1287_ = v___x_1284_;
goto v_reusejp_1286_;
}
else
{
lean_object* v_reuseFailAlloc_1288_; 
v_reuseFailAlloc_1288_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1288_, 0, v_a_1282_);
v___x_1287_ = v_reuseFailAlloc_1288_;
goto v_reusejp_1286_;
}
v_reusejp_1286_:
{
return v___x_1287_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg___boxed(lean_object* v_lctx_1290_, lean_object* v_localInsts_1291_, lean_object* v_x_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_){
_start:
{
lean_object* v_res_1298_; 
v_res_1298_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg(v_lctx_1290_, v_localInsts_1291_, v_x_1292_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_);
lean_dec(v___y_1296_);
lean_dec_ref(v___y_1295_);
lean_dec(v___y_1294_);
lean_dec_ref(v___y_1293_);
return v_res_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1___lam__0(lean_object* v_props_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v___x_1301_; lean_object* v___x_1302_; 
v___x_1301_ = lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(v_props_1299_);
v___x_1302_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1302_, 0, v___x_1301_);
lean_ctor_set(v___x_1302_, 1, v___y_1300_);
return v___x_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1(lean_object* v_c_1303_, lean_object* v_props_1304_, lean_object* v_children_1305_){
_start:
{
lean_object* v_toModule_1306_; lean_object* v_export_1307_; lean_object* v_javascript_1308_; lean_object* v___f_1309_; uint64_t v___x_1310_; lean_object* v___x_1311_; 
v_toModule_1306_ = lean_ctor_get(v_c_1303_, 0);
v_export_1307_ = lean_ctor_get(v_c_1303_, 1);
v_javascript_1308_ = lean_ctor_get(v_toModule_1306_, 0);
v___f_1309_ = lean_alloc_closure((void*)(lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1___lam__0), 2, 1);
lean_closure_set(v___f_1309_, 0, v_props_1304_);
v___x_1310_ = lean_string_hash(v_javascript_1308_);
lean_inc_ref(v_export_1307_);
v___x_1311_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_1311_, 0, v_export_1307_);
lean_ctor_set(v___x_1311_, 1, v___f_1309_);
lean_ctor_set(v___x_1311_, 2, v_children_1305_);
lean_ctor_set_uint64(v___x_1311_, sizeof(void*)*3, v___x_1310_);
return v___x_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1___boxed(lean_object* v_c_1312_, lean_object* v_props_1313_, lean_object* v_children_1314_){
_start:
{
lean_object* v_res_1315_; 
v_res_1315_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1(v_c_1312_, v_props_1313_, v_children_1314_);
lean_dec_ref(v_c_1312_);
return v_res_1315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__0(lean_object* v_mkCmdStr_1316_, lean_object* v_selectedLocations_1317_, lean_object* v___x_1318_, lean_object* v_params_1319_, lean_object* v_a_1320_, lean_object* v_replaceRange_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_){
_start:
{
lean_object* v___x_1327_; 
v___x_1327_ = lean_apply_8(v_mkCmdStr_1316_, v_selectedLocations_1317_, v___x_1318_, v_params_1319_, v___y_1322_, v___y_1323_, v___y_1324_, v___y_1325_, lean_box(0));
if (lean_obj_tag(v___x_1327_) == 0)
{
lean_object* v_a_1328_; lean_object* v___x_1330_; uint8_t v_isShared_1331_; uint8_t v_isSharedCheck_1348_; 
v_a_1328_ = lean_ctor_get(v___x_1327_, 0);
v_isSharedCheck_1348_ = !lean_is_exclusive(v___x_1327_);
if (v_isSharedCheck_1348_ == 0)
{
v___x_1330_ = v___x_1327_;
v_isShared_1331_ = v_isSharedCheck_1348_;
goto v_resetjp_1329_;
}
else
{
lean_inc(v_a_1328_);
lean_dec(v___x_1327_);
v___x_1330_ = lean_box(0);
v_isShared_1331_ = v_isSharedCheck_1348_;
goto v_resetjp_1329_;
}
v_resetjp_1329_:
{
lean_object* v_snd_1332_; lean_object* v_toEditableDocumentCore_1333_; lean_object* v_fst_1334_; lean_object* v_fst_1335_; lean_object* v_snd_1336_; lean_object* v_meta_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1346_; 
v_snd_1332_ = lean_ctor_get(v_a_1328_, 1);
lean_inc(v_snd_1332_);
v_toEditableDocumentCore_1333_ = lean_ctor_get(v_a_1320_, 0);
v_fst_1334_ = lean_ctor_get(v_a_1328_, 0);
lean_inc(v_fst_1334_);
lean_dec(v_a_1328_);
v_fst_1335_ = lean_ctor_get(v_snd_1332_, 0);
lean_inc(v_fst_1335_);
v_snd_1336_ = lean_ctor_get(v_snd_1332_, 1);
lean_inc(v_snd_1336_);
lean_dec(v_snd_1332_);
v_meta_1337_ = lean_ctor_get(v_toEditableDocumentCore_1333_, 0);
v___x_1338_ = lp_proofwidgets_ProofWidgets_MakeEditLink;
v___x_1339_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(v_meta_1337_, v_replaceRange_1321_, v_fst_1335_, v_snd_1336_);
v___x_1340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1340_, 0, v_fst_1334_);
v___x_1341_ = lean_unsigned_to_nat(1u);
v___x_1342_ = lean_mk_empty_array_with_capacity(v___x_1341_);
v___x_1343_ = lean_array_push(v___x_1342_, v___x_1340_);
v___x_1344_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__1(v___x_1338_, v___x_1339_, v___x_1343_);
if (v_isShared_1331_ == 0)
{
lean_ctor_set(v___x_1330_, 0, v___x_1344_);
v___x_1346_ = v___x_1330_;
goto v_reusejp_1345_;
}
else
{
lean_object* v_reuseFailAlloc_1347_; 
v_reuseFailAlloc_1347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1347_, 0, v___x_1344_);
v___x_1346_ = v_reuseFailAlloc_1347_;
goto v_reusejp_1345_;
}
v_reusejp_1345_:
{
return v___x_1346_;
}
}
}
else
{
lean_object* v_a_1349_; lean_object* v___x_1351_; uint8_t v_isShared_1352_; uint8_t v_isSharedCheck_1356_; 
lean_dec_ref(v_replaceRange_1321_);
v_a_1349_ = lean_ctor_get(v___x_1327_, 0);
v_isSharedCheck_1356_ = !lean_is_exclusive(v___x_1327_);
if (v_isSharedCheck_1356_ == 0)
{
v___x_1351_ = v___x_1327_;
v_isShared_1352_ = v_isSharedCheck_1356_;
goto v_resetjp_1350_;
}
else
{
lean_inc(v_a_1349_);
lean_dec(v___x_1327_);
v___x_1351_ = lean_box(0);
v_isShared_1352_ = v_isSharedCheck_1356_;
goto v_resetjp_1350_;
}
v_resetjp_1350_:
{
lean_object* v___x_1354_; 
if (v_isShared_1352_ == 0)
{
v___x_1354_ = v___x_1351_;
goto v_reusejp_1353_;
}
else
{
lean_object* v_reuseFailAlloc_1355_; 
v_reuseFailAlloc_1355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1355_, 0, v_a_1349_);
v___x_1354_ = v_reuseFailAlloc_1355_;
goto v_reusejp_1353_;
}
v_reusejp_1353_:
{
return v___x_1354_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__0___boxed(lean_object* v_mkCmdStr_1357_, lean_object* v_selectedLocations_1358_, lean_object* v___x_1359_, lean_object* v_params_1360_, lean_object* v_a_1361_, lean_object* v_replaceRange_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_){
_start:
{
lean_object* v_res_1368_; 
v_res_1368_ = lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__0(v_mkCmdStr_1357_, v_selectedLocations_1358_, v___x_1359_, v_params_1360_, v_a_1361_, v_replaceRange_1362_, v___y_1363_, v___y_1364_, v___y_1365_, v___y_1366_);
lean_dec_ref(v_a_1361_);
return v_res_1368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__1(lean_object* v_mvarId_1369_, lean_object* v_mkCmdStr_1370_, lean_object* v_selectedLocations_1371_, lean_object* v_params_1372_, lean_object* v_a_1373_, lean_object* v_replaceRange_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_){
_start:
{
lean_object* v___x_1380_; 
v___x_1380_ = l_Lean_MVarId_getDecl(v_mvarId_1369_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_);
if (lean_obj_tag(v___x_1380_) == 0)
{
lean_object* v_a_1381_; lean_object* v_options_1382_; lean_object* v_lctx_1383_; lean_object* v_type_1384_; lean_object* v_localInstances_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v_fst_1389_; lean_object* v___x_1390_; lean_object* v___f_1391_; lean_object* v___x_1392_; 
v_a_1381_ = lean_ctor_get(v___x_1380_, 0);
lean_inc(v_a_1381_);
lean_dec_ref_known(v___x_1380_, 1);
v_options_1382_ = lean_ctor_get(v___y_1377_, 2);
v_lctx_1383_ = lean_ctor_get(v_a_1381_, 1);
lean_inc_ref(v_lctx_1383_);
v_type_1384_ = lean_ctor_get(v_a_1381_, 2);
lean_inc_ref(v_type_1384_);
v_localInstances_1385_ = lean_ctor_get(v_a_1381_, 4);
lean_inc_ref(v_localInstances_1385_);
lean_dec(v_a_1381_);
v___x_1386_ = lean_box(1);
lean_inc_ref(v_options_1382_);
v___x_1387_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1387_, 0, v_options_1382_);
lean_ctor_set(v___x_1387_, 1, v___x_1386_);
lean_ctor_set(v___x_1387_, 2, v___x_1386_);
v___x_1388_ = l_Lean_LocalContext_sanitizeNames(v_lctx_1383_, v___x_1387_);
v_fst_1389_ = lean_ctor_get(v___x_1388_, 0);
lean_inc(v_fst_1389_);
lean_dec_ref(v___x_1388_);
v___x_1390_ = l_Lean_Expr_consumeMData(v_type_1384_);
lean_dec_ref(v_type_1384_);
v___f_1391_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__0___boxed), 11, 6);
lean_closure_set(v___f_1391_, 0, v_mkCmdStr_1370_);
lean_closure_set(v___f_1391_, 1, v_selectedLocations_1371_);
lean_closure_set(v___f_1391_, 2, v___x_1390_);
lean_closure_set(v___f_1391_, 3, v_params_1372_);
lean_closure_set(v___f_1391_, 4, v_a_1373_);
lean_closure_set(v___f_1391_, 5, v_replaceRange_1374_);
v___x_1392_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg(v_fst_1389_, v_localInstances_1385_, v___f_1391_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_);
return v___x_1392_;
}
else
{
lean_object* v_a_1393_; lean_object* v___x_1395_; uint8_t v_isShared_1396_; uint8_t v_isSharedCheck_1400_; 
lean_dec_ref(v_replaceRange_1374_);
lean_dec_ref(v_a_1373_);
lean_dec_ref(v_params_1372_);
lean_dec_ref(v_selectedLocations_1371_);
lean_dec_ref(v_mkCmdStr_1370_);
v_a_1393_ = lean_ctor_get(v___x_1380_, 0);
v_isSharedCheck_1400_ = !lean_is_exclusive(v___x_1380_);
if (v_isSharedCheck_1400_ == 0)
{
v___x_1395_ = v___x_1380_;
v_isShared_1396_ = v_isSharedCheck_1400_;
goto v_resetjp_1394_;
}
else
{
lean_inc(v_a_1393_);
lean_dec(v___x_1380_);
v___x_1395_ = lean_box(0);
v_isShared_1396_ = v_isSharedCheck_1400_;
goto v_resetjp_1394_;
}
v_resetjp_1394_:
{
lean_object* v___x_1398_; 
if (v_isShared_1396_ == 0)
{
v___x_1398_ = v___x_1395_;
goto v_reusejp_1397_;
}
else
{
lean_object* v_reuseFailAlloc_1399_; 
v_reuseFailAlloc_1399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1399_, 0, v_a_1393_);
v___x_1398_ = v_reuseFailAlloc_1399_;
goto v_reusejp_1397_;
}
v_reusejp_1397_:
{
return v___x_1398_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__1___boxed(lean_object* v_mvarId_1401_, lean_object* v_mkCmdStr_1402_, lean_object* v_selectedLocations_1403_, lean_object* v_params_1404_, lean_object* v_a_1405_, lean_object* v_replaceRange_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_){
_start:
{
lean_object* v_res_1412_; 
v_res_1412_ = lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__1(v_mvarId_1401_, v_mkCmdStr_1402_, v_selectedLocations_1403_, v_params_1404_, v_a_1405_, v_replaceRange_1406_, v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
lean_dec(v___y_1410_);
lean_dec_ref(v___y_1409_);
lean_dec(v___y_1408_);
lean_dec_ref(v___y_1407_);
return v_res_1412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg(lean_object* v_mainGoalName_1417_, lean_object* v_errorMsg_1418_, uint8_t v___y_1419_, lean_object* v_as_1420_, size_t v_sz_1421_, size_t v_i_1422_, lean_object* v_b_1423_){
_start:
{
lean_object* v_a_1426_; uint8_t v___x_1430_; 
v___x_1430_ = lean_usize_dec_lt(v_i_1422_, v_sz_1421_);
if (v___x_1430_ == 0)
{
lean_object* v___x_1431_; 
lean_dec_ref(v_errorMsg_1418_);
v___x_1431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1431_, 0, v_b_1423_);
return v___x_1431_;
}
else
{
lean_object* v_a_1432_; lean_object* v_mvarId_1433_; lean_object* v_loc_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1465_; 
lean_dec_ref(v_b_1423_);
v_a_1432_ = lean_array_uget(v_as_1420_, v_i_1422_);
v_mvarId_1433_ = lean_ctor_get(v_a_1432_, 0);
v_loc_1434_ = lean_ctor_get(v_a_1432_, 1);
v_isSharedCheck_1465_ = !lean_is_exclusive(v_a_1432_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1436_ = v_a_1432_;
v_isShared_1437_ = v_isSharedCheck_1465_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_loc_1434_);
lean_inc(v_mvarId_1433_);
lean_dec(v_a_1432_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1465_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v___x_1438_; uint8_t v___x_1439_; 
v___x_1438_ = lean_box(0);
v___x_1439_ = lean_name_eq(v_mvarId_1433_, v_mainGoalName_1417_);
lean_dec(v_mvarId_1433_);
if (v___x_1439_ == 0)
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1449_; 
lean_dec_ref(v_loc_1434_);
v___x_1440_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0));
v___x_1441_ = ((lean_object*)(lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_));
v___x_1442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1442_, 0, v_errorMsg_1418_);
v___x_1443_ = lean_unsigned_to_nat(1u);
v___x_1444_ = lean_mk_empty_array_with_capacity(v___x_1443_);
v___x_1445_ = lean_array_push(v___x_1444_, v___x_1442_);
v___x_1446_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1446_, 0, v___x_1440_);
lean_ctor_set(v___x_1446_, 1, v___x_1441_);
lean_ctor_set(v___x_1446_, 2, v___x_1445_);
v___x_1447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1447_, 0, v___x_1446_);
if (v_isShared_1437_ == 0)
{
lean_ctor_set(v___x_1436_, 1, v___x_1438_);
lean_ctor_set(v___x_1436_, 0, v___x_1447_);
v___x_1449_ = v___x_1436_;
goto v_reusejp_1448_;
}
else
{
lean_object* v_reuseFailAlloc_1451_; 
v_reuseFailAlloc_1451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1451_, 0, v___x_1447_);
lean_ctor_set(v_reuseFailAlloc_1451_, 1, v___x_1438_);
v___x_1449_ = v_reuseFailAlloc_1451_;
goto v_reusejp_1448_;
}
v_reusejp_1448_:
{
lean_object* v___x_1450_; 
v___x_1450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1450_, 0, v___x_1449_);
return v___x_1450_;
}
}
else
{
lean_object* v___x_1452_; 
v___x_1452_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__1));
if (v___y_1419_ == 0)
{
lean_del_object(v___x_1436_);
lean_dec_ref(v_loc_1434_);
v_a_1426_ = v___x_1452_;
goto v___jp_1425_;
}
else
{
if (lean_obj_tag(v_loc_1434_) == 3)
{
lean_dec_ref_known(v_loc_1434_, 1);
lean_del_object(v___x_1436_);
v_a_1426_ = v___x_1452_;
goto v___jp_1425_;
}
else
{
lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1462_; 
lean_dec_ref(v_loc_1434_);
v___x_1453_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0));
v___x_1454_ = ((lean_object*)(lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_));
v___x_1455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1455_, 0, v_errorMsg_1418_);
v___x_1456_ = lean_unsigned_to_nat(1u);
v___x_1457_ = lean_mk_empty_array_with_capacity(v___x_1456_);
v___x_1458_ = lean_array_push(v___x_1457_, v___x_1455_);
v___x_1459_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1459_, 0, v___x_1453_);
lean_ctor_set(v___x_1459_, 1, v___x_1454_);
lean_ctor_set(v___x_1459_, 2, v___x_1458_);
v___x_1460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1460_, 0, v___x_1459_);
if (v_isShared_1437_ == 0)
{
lean_ctor_set(v___x_1436_, 1, v___x_1438_);
lean_ctor_set(v___x_1436_, 0, v___x_1460_);
v___x_1462_ = v___x_1436_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1464_; 
v_reuseFailAlloc_1464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1464_, 0, v___x_1460_);
lean_ctor_set(v_reuseFailAlloc_1464_, 1, v___x_1438_);
v___x_1462_ = v_reuseFailAlloc_1464_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
lean_object* v___x_1463_; 
v___x_1463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1463_, 0, v___x_1462_);
return v___x_1463_;
}
}
}
}
}
}
v___jp_1425_:
{
size_t v___x_1427_; size_t v___x_1428_; 
v___x_1427_ = ((size_t)1ULL);
v___x_1428_ = lean_usize_add(v_i_1422_, v___x_1427_);
lean_inc_ref(v_a_1426_);
v_i_1422_ = v___x_1428_;
v_b_1423_ = v_a_1426_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___boxed(lean_object* v_mainGoalName_1466_, lean_object* v_errorMsg_1467_, lean_object* v___y_1468_, lean_object* v_as_1469_, lean_object* v_sz_1470_, lean_object* v_i_1471_, lean_object* v_b_1472_, lean_object* v___y_1473_){
_start:
{
uint8_t v___y_1997__boxed_1474_; size_t v_sz_boxed_1475_; size_t v_i_boxed_1476_; lean_object* v_res_1477_; 
v___y_1997__boxed_1474_ = lean_unbox(v___y_1468_);
v_sz_boxed_1475_ = lean_unbox_usize(v_sz_1470_);
lean_dec(v_sz_1470_);
v_i_boxed_1476_ = lean_unbox_usize(v_i_1471_);
lean_dec(v_i_1471_);
v_res_1477_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg(v_mainGoalName_1466_, v_errorMsg_1467_, v___y_1997__boxed_1474_, v_as_1469_, v_sz_boxed_1475_, v_i_boxed_1476_, v_b_1472_);
lean_dec_ref(v_as_1469_);
lean_dec(v_mainGoalName_1466_);
return v_res_1477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2(lean_object* v_params_1530_, lean_object* v_title_1531_, lean_object* v_mkCmdStr_1532_, uint8_t v_onlyGoal_1533_, lean_object* v_helpMsg_1534_, uint8_t v_onlyOne_1535_, lean_object* v___y_1536_){
_start:
{
lean_object* v___x_1538_; lean_object* v_toSelectInsertParams_1539_; lean_object* v_a_1540_; lean_object* v___x_1542_; uint8_t v_isShared_1543_; uint8_t v_isSharedCheck_1638_; 
v___x_1538_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00createCalc_spec__0(v___y_1536_);
v_toSelectInsertParams_1539_ = lean_ctor_get(v_params_1530_, 0);
v_a_1540_ = lean_ctor_get(v___x_1538_, 0);
v_isSharedCheck_1638_ = !lean_is_exclusive(v___x_1538_);
if (v_isSharedCheck_1638_ == 0)
{
v___x_1542_ = v___x_1538_;
v_isShared_1543_ = v_isSharedCheck_1638_;
goto v_resetjp_1541_;
}
else
{
lean_inc(v_a_1540_);
lean_dec(v___x_1538_);
v___x_1542_ = lean_box(0);
v_isShared_1543_ = v_isSharedCheck_1638_;
goto v_resetjp_1541_;
}
v_resetjp_1541_:
{
lean_object* v_goals_1544_; lean_object* v_selectedLocations_1545_; lean_object* v_replaceRange_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; uint8_t v___x_1549_; lean_object* v_a_1551_; 
v_goals_1544_ = lean_ctor_get(v_toSelectInsertParams_1539_, 1);
v_selectedLocations_1545_ = lean_ctor_get(v_toSelectInsertParams_1539_, 2);
lean_inc_ref(v_selectedLocations_1545_);
v_replaceRange_1546_ = lean_ctor_get(v_toSelectInsertParams_1539_, 3);
lean_inc_ref(v_replaceRange_1546_);
v___x_1547_ = lean_unsigned_to_nat(0u);
v___x_1548_ = lean_array_get_size(v_goals_1544_);
v___x_1549_ = lean_nat_dec_lt(v___x_1547_, v___x_1548_);
if (v___x_1549_ == 0)
{
lean_object* v___x_1576_; lean_object* v___x_1577_; 
lean_dec_ref(v_replaceRange_1546_);
lean_dec_ref(v_selectedLocations_1545_);
lean_del_object(v___x_1542_);
lean_dec(v_a_1540_);
lean_dec_ref(v_helpMsg_1534_);
lean_dec_ref(v_mkCmdStr_1532_);
lean_dec_ref(v_title_1531_);
lean_dec_ref(v_params_1530_);
v___x_1576_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__16));
v___x_1577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1577_, 0, v___x_1576_);
return v___x_1577_;
}
else
{
lean_object* v_mainGoal_1578_; lean_object* v_toInteractiveGoalCore_1579_; lean_object* v_mvarId_1580_; lean_object* v___f_1581_; lean_object* v___y_1583_; lean_object* v___y_1623_; lean_object* v___y_1624_; lean_object* v___y_1633_; 
v_mainGoal_1578_ = lean_array_fget_borrowed(v_goals_1544_, v___x_1547_);
v_toInteractiveGoalCore_1579_ = lean_ctor_get(v_mainGoal_1578_, 0);
lean_inc_ref(v_toInteractiveGoalCore_1579_);
v_mvarId_1580_ = lean_ctor_get(v_mainGoal_1578_, 3);
lean_inc_n(v_mvarId_1580_, 2);
lean_inc_ref(v_selectedLocations_1545_);
v___f_1581_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__1___boxed), 11, 6);
lean_closure_set(v___f_1581_, 0, v_mvarId_1580_);
lean_closure_set(v___f_1581_, 1, v_mkCmdStr_1532_);
lean_closure_set(v___f_1581_, 2, v_selectedLocations_1545_);
lean_closure_set(v___f_1581_, 3, v_params_1530_);
lean_closure_set(v___f_1581_, 4, v_a_1540_);
lean_closure_set(v___f_1581_, 5, v_replaceRange_1546_);
if (v_onlyOne_1535_ == 0)
{
lean_object* v___x_1636_; 
v___x_1636_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__24));
v___y_1633_ = v___x_1636_;
goto v___jp_1632_;
}
else
{
lean_object* v___x_1637_; 
v___x_1637_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__25));
v___y_1633_ = v___x_1637_;
goto v___jp_1632_;
}
v___jp_1582_:
{
lean_object* v___x_1584_; size_t v_sz_1585_; size_t v___x_1586_; lean_object* v___x_1587_; 
v___x_1584_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__1));
v_sz_1585_ = lean_array_size(v_selectedLocations_1545_);
v___x_1586_ = ((size_t)0ULL);
v___x_1587_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg(v_mvarId_1580_, v___y_1583_, v_onlyGoal_1533_, v_selectedLocations_1545_, v_sz_1585_, v___x_1586_, v___x_1584_);
lean_dec(v_mvarId_1580_);
if (lean_obj_tag(v___x_1587_) == 0)
{
lean_object* v_a_1588_; lean_object* v_fst_1589_; 
v_a_1588_ = lean_ctor_get(v___x_1587_, 0);
lean_inc(v_a_1588_);
lean_dec_ref_known(v___x_1587_, 1);
v_fst_1589_ = lean_ctor_get(v_a_1588_, 0);
lean_inc(v_fst_1589_);
lean_dec(v_a_1588_);
if (lean_obj_tag(v_fst_1589_) == 0)
{
lean_object* v___x_1590_; uint8_t v___x_1591_; 
v___x_1590_ = lean_array_get_size(v_selectedLocations_1545_);
lean_dec_ref(v_selectedLocations_1545_);
v___x_1591_ = lean_nat_dec_eq(v___x_1590_, v___x_1547_);
if (v___x_1591_ == 0)
{
lean_object* v_ctx_1592_; lean_object* v_val_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; 
lean_dec_ref(v_helpMsg_1534_);
v_ctx_1592_ = lean_ctor_get(v_toInteractiveGoalCore_1579_, 2);
lean_inc_ref(v_ctx_1592_);
lean_dec_ref(v_toInteractiveGoalCore_1579_);
v_val_1593_ = lean_ctor_get(v_ctx_1592_, 0);
lean_inc(v_val_1593_);
lean_dec_ref(v_ctx_1592_);
v___x_1594_ = lean_obj_once(&lp_mathlib_createCalc___redArg___closed__9, &lp_mathlib_createCalc___redArg___closed__9_once, _init_lp_mathlib_createCalc___redArg___closed__9);
v___x_1595_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_val_1593_, v___x_1594_, v___f_1581_);
if (lean_obj_tag(v___x_1595_) == 0)
{
lean_object* v_a_1596_; 
v_a_1596_ = lean_ctor_get(v___x_1595_, 0);
lean_inc(v_a_1596_);
lean_dec_ref_known(v___x_1595_, 1);
v_a_1551_ = v_a_1596_;
goto v___jp_1550_;
}
else
{
lean_object* v_a_1597_; lean_object* v___x_1599_; uint8_t v_isShared_1600_; uint8_t v_isSharedCheck_1605_; 
lean_del_object(v___x_1542_);
lean_dec_ref(v_title_1531_);
v_a_1597_ = lean_ctor_get(v___x_1595_, 0);
v_isSharedCheck_1605_ = !lean_is_exclusive(v___x_1595_);
if (v_isSharedCheck_1605_ == 0)
{
v___x_1599_ = v___x_1595_;
v_isShared_1600_ = v_isSharedCheck_1605_;
goto v_resetjp_1598_;
}
else
{
lean_inc(v_a_1597_);
lean_dec(v___x_1595_);
v___x_1599_ = lean_box(0);
v_isShared_1600_ = v_isSharedCheck_1605_;
goto v_resetjp_1598_;
}
v_resetjp_1598_:
{
lean_object* v___x_1601_; lean_object* v___x_1603_; 
v___x_1601_ = l_Lean_Server_RequestError_ofIoError(v_a_1597_);
if (v_isShared_1600_ == 0)
{
lean_ctor_set(v___x_1599_, 0, v___x_1601_);
v___x_1603_ = v___x_1599_;
goto v_reusejp_1602_;
}
else
{
lean_object* v_reuseFailAlloc_1604_; 
v_reuseFailAlloc_1604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1604_, 0, v___x_1601_);
v___x_1603_ = v_reuseFailAlloc_1604_;
goto v_reusejp_1602_;
}
v_reusejp_1602_:
{
return v___x_1603_;
}
}
}
}
else
{
lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; 
lean_dec_ref(v___f_1581_);
lean_dec_ref(v_toInteractiveGoalCore_1579_);
v___x_1606_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg___closed__0));
v___x_1607_ = ((lean_object*)(lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_39_));
v___x_1608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1608_, 0, v_helpMsg_1534_);
v___x_1609_ = lean_unsigned_to_nat(1u);
v___x_1610_ = lean_mk_empty_array_with_capacity(v___x_1609_);
v___x_1611_ = lean_array_push(v___x_1610_, v___x_1608_);
v___x_1612_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1606_);
lean_ctor_set(v___x_1612_, 1, v___x_1607_);
lean_ctor_set(v___x_1612_, 2, v___x_1611_);
v_a_1551_ = v___x_1612_;
goto v___jp_1550_;
}
}
else
{
lean_object* v_val_1613_; 
lean_dec_ref(v___f_1581_);
lean_dec_ref(v_toInteractiveGoalCore_1579_);
lean_dec_ref(v_selectedLocations_1545_);
lean_dec_ref(v_helpMsg_1534_);
v_val_1613_ = lean_ctor_get(v_fst_1589_, 0);
lean_inc(v_val_1613_);
lean_dec_ref_known(v_fst_1589_, 1);
v_a_1551_ = v_val_1613_;
goto v___jp_1550_;
}
}
else
{
lean_object* v_a_1614_; lean_object* v___x_1616_; uint8_t v_isShared_1617_; uint8_t v_isSharedCheck_1621_; 
lean_dec_ref(v___f_1581_);
lean_dec_ref(v_toInteractiveGoalCore_1579_);
lean_dec_ref(v_selectedLocations_1545_);
lean_del_object(v___x_1542_);
lean_dec_ref(v_helpMsg_1534_);
lean_dec_ref(v_title_1531_);
v_a_1614_ = lean_ctor_get(v___x_1587_, 0);
v_isSharedCheck_1621_ = !lean_is_exclusive(v___x_1587_);
if (v_isSharedCheck_1621_ == 0)
{
v___x_1616_ = v___x_1587_;
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
else
{
lean_inc(v_a_1614_);
lean_dec(v___x_1587_);
v___x_1616_ = lean_box(0);
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
v_resetjp_1615_:
{
lean_object* v___x_1619_; 
if (v_isShared_1617_ == 0)
{
v___x_1619_ = v___x_1616_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v_a_1614_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
return v___x_1619_;
}
}
}
}
v___jp_1622_:
{
lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v_errorMsg_1627_; 
v___x_1625_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__17));
lean_inc_ref(v___y_1623_);
v___x_1626_ = lean_string_append(v___y_1623_, v___x_1625_);
v_errorMsg_1627_ = lean_string_append(v___x_1626_, v___y_1624_);
if (v_onlyOne_1535_ == 0)
{
v___y_1583_ = v_errorMsg_1627_;
goto v___jp_1582_;
}
else
{
lean_object* v___x_1628_; lean_object* v___x_1629_; uint8_t v___x_1630_; 
v___x_1628_ = lean_unsigned_to_nat(1u);
v___x_1629_ = lean_array_get_size(v_selectedLocations_1545_);
v___x_1630_ = lean_nat_dec_lt(v___x_1628_, v___x_1629_);
if (v___x_1630_ == 0)
{
v___y_1583_ = v_errorMsg_1627_;
goto v___jp_1582_;
}
else
{
lean_object* v___x_1631_; 
lean_dec_ref(v_errorMsg_1627_);
lean_dec_ref(v___f_1581_);
lean_dec(v_mvarId_1580_);
lean_dec_ref(v_toInteractiveGoalCore_1579_);
lean_dec_ref(v_selectedLocations_1545_);
lean_dec_ref(v_helpMsg_1534_);
v___x_1631_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__21));
v_a_1551_ = v___x_1631_;
goto v___jp_1550_;
}
}
}
v___jp_1632_:
{
if (v_onlyGoal_1533_ == 0)
{
lean_object* v___x_1634_; 
v___x_1634_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__22));
v___y_1623_ = v___y_1633_;
v___y_1624_ = v___x_1634_;
goto v___jp_1622_;
}
else
{
lean_object* v___x_1635_; 
v___x_1635_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__23));
v___y_1623_ = v___y_1633_;
v___y_1624_ = v___x_1635_;
goto v___jp_1622_;
}
}
}
v___jp_1550_:
{
lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1574_; 
v___x_1552_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__0));
v___x_1553_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__1));
v___x_1554_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_1554_, 0, v___x_1549_);
v___x_1555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1555_, 0, v___x_1553_);
lean_ctor_set(v___x_1555_, 1, v___x_1554_);
v___x_1556_ = lean_unsigned_to_nat(1u);
v___x_1557_ = lean_mk_empty_array_with_capacity(v___x_1556_);
lean_inc_ref_n(v___x_1557_, 2);
v___x_1558_ = lean_array_push(v___x_1557_, v___x_1555_);
v___x_1559_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__2));
v___x_1560_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__7));
v___x_1561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1561_, 0, v_title_1531_);
v___x_1562_ = lean_array_push(v___x_1557_, v___x_1561_);
v___x_1563_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1563_, 0, v___x_1559_);
lean_ctor_set(v___x_1563_, 1, v___x_1560_);
lean_ctor_set(v___x_1563_, 2, v___x_1562_);
v___x_1564_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__8));
v___x_1565_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___closed__12));
v___x_1566_ = lean_array_push(v___x_1557_, v_a_1551_);
v___x_1567_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1567_, 0, v___x_1564_);
lean_ctor_set(v___x_1567_, 1, v___x_1565_);
lean_ctor_set(v___x_1567_, 2, v___x_1566_);
v___x_1568_ = lean_unsigned_to_nat(2u);
v___x_1569_ = lean_mk_empty_array_with_capacity(v___x_1568_);
v___x_1570_ = lean_array_push(v___x_1569_, v___x_1563_);
v___x_1571_ = lean_array_push(v___x_1570_, v___x_1567_);
v___x_1572_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1572_, 0, v___x_1552_);
lean_ctor_set(v___x_1572_, 1, v___x_1558_);
lean_ctor_set(v___x_1572_, 2, v___x_1571_);
if (v_isShared_1543_ == 0)
{
lean_ctor_set(v___x_1542_, 0, v___x_1572_);
v___x_1574_ = v___x_1542_;
goto v_reusejp_1573_;
}
else
{
lean_object* v_reuseFailAlloc_1575_; 
v_reuseFailAlloc_1575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1575_, 0, v___x_1572_);
v___x_1574_ = v_reuseFailAlloc_1575_;
goto v_reusejp_1573_;
}
v_reusejp_1573_:
{
return v___x_1574_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___boxed(lean_object* v_params_1639_, lean_object* v_title_1640_, lean_object* v_mkCmdStr_1641_, lean_object* v_onlyGoal_1642_, lean_object* v_helpMsg_1643_, lean_object* v_onlyOne_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_){
_start:
{
uint8_t v_onlyGoal_boxed_1647_; uint8_t v_onlyOne_boxed_1648_; lean_object* v_res_1649_; 
v_onlyGoal_boxed_1647_ = lean_unbox(v_onlyGoal_1642_);
v_onlyOne_boxed_1648_ = lean_unbox(v_onlyOne_1644_);
v_res_1649_ = lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2(v_params_1639_, v_title_1640_, v_mkCmdStr_1641_, v_onlyGoal_boxed_1647_, v_helpMsg_1643_, v_onlyOne_boxed_1648_, v___y_1645_);
lean_dec_ref(v___y_1645_);
return v_res_1649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0(lean_object* v_mkCmdStr_1650_, lean_object* v_helpMsg_1651_, lean_object* v_title_1652_, uint8_t v_onlyGoal_1653_, uint8_t v_onlyOne_1654_, lean_object* v_params_1655_, lean_object* v_a_1656_){
_start:
{
lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___f_1660_; lean_object* v___x_1661_; 
v___x_1658_ = lean_box(v_onlyGoal_1653_);
v___x_1659_ = lean_box(v_onlyOne_1654_);
v___f_1660_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___lam__2___boxed), 8, 6);
lean_closure_set(v___f_1660_, 0, v_params_1655_);
lean_closure_set(v___f_1660_, 1, v_title_1652_);
lean_closure_set(v___f_1660_, 2, v_mkCmdStr_1650_);
lean_closure_set(v___f_1660_, 3, v___x_1658_);
lean_closure_set(v___f_1660_, 4, v_helpMsg_1651_);
lean_closure_set(v___f_1660_, 5, v___x_1659_);
v___x_1661_ = l_Lean_Server_RequestM_asTask___redArg(v___f_1660_, v_a_1656_);
return v___x_1661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0___boxed(lean_object* v_mkCmdStr_1662_, lean_object* v_helpMsg_1663_, lean_object* v_title_1664_, lean_object* v_onlyGoal_1665_, lean_object* v_onlyOne_1666_, lean_object* v_params_1667_, lean_object* v_a_1668_, lean_object* v_a_1669_){
_start:
{
uint8_t v_onlyGoal_boxed_1670_; uint8_t v_onlyOne_boxed_1671_; lean_object* v_res_1672_; 
v_onlyGoal_boxed_1670_ = lean_unbox(v_onlyGoal_1665_);
v_onlyOne_boxed_1671_ = lean_unbox(v_onlyOne_1666_);
v_res_1672_ = lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0(v_mkCmdStr_1662_, v_helpMsg_1663_, v_title_1664_, v_onlyGoal_boxed_1670_, v_onlyOne_boxed_1671_, v_params_1667_, v_a_1668_);
lean_dec_ref(v_a_1668_);
return v_res_1672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CalcPanel_rpc(lean_object* v_params_1676_, lean_object* v_a_1677_){
_start:
{
lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; uint8_t v___x_1682_; uint8_t v___x_1683_; lean_object* v___x_1684_; 
v___x_1679_ = ((lean_object*)(lp_mathlib_CalcPanel_rpc___closed__0));
v___x_1680_ = ((lean_object*)(lp_mathlib_CalcPanel_rpc___closed__1));
v___x_1681_ = ((lean_object*)(lp_mathlib_CalcPanel_rpc___closed__2));
v___x_1682_ = 1;
v___x_1683_ = 0;
v___x_1684_ = lp_mathlib_mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0(v___x_1679_, v___x_1680_, v___x_1681_, v___x_1682_, v___x_1683_, v_params_1676_, v_a_1677_);
return v___x_1684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CalcPanel_rpc___boxed(lean_object* v_params_1685_, lean_object* v_a_1686_, lean_object* v_a_1687_){
_start:
{
lean_object* v_res_1688_; 
v_res_1688_ = lp_mathlib_CalcPanel_rpc(v_params_1685_, v_a_1686_);
lean_dec_ref(v_a_1686_);
return v_res_1688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2(lean_object* v_00_u03b1_1689_, lean_object* v_lctx_1690_, lean_object* v_localInsts_1691_, lean_object* v_x_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_){
_start:
{
lean_object* v___x_1698_; 
v___x_1698_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___redArg(v_lctx_1690_, v_localInsts_1691_, v_x_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_);
return v___x_1698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2___boxed(lean_object* v_00_u03b1_1699_, lean_object* v_lctx_1700_, lean_object* v_localInsts_1701_, lean_object* v_x_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_){
_start:
{
lean_object* v_res_1708_; 
v_res_1708_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__2(v_00_u03b1_1699_, v_lctx_1700_, v_localInsts_1701_, v_x_1702_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
lean_dec(v___y_1706_);
lean_dec_ref(v___y_1705_);
lean_dec(v___y_1704_);
lean_dec_ref(v___y_1703_);
return v_res_1708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0(lean_object* v_mainGoalName_1709_, lean_object* v_errorMsg_1710_, uint8_t v___y_1711_, lean_object* v_as_1712_, size_t v_sz_1713_, size_t v_i_1714_, lean_object* v_b_1715_, lean_object* v___y_1716_){
_start:
{
lean_object* v___x_1718_; 
v___x_1718_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___redArg(v_mainGoalName_1709_, v_errorMsg_1710_, v___y_1711_, v_as_1712_, v_sz_1713_, v_i_1714_, v_b_1715_);
return v___x_1718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0___boxed(lean_object* v_mainGoalName_1719_, lean_object* v_errorMsg_1720_, lean_object* v___y_1721_, lean_object* v_as_1722_, lean_object* v_sz_1723_, lean_object* v_i_1724_, lean_object* v_b_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_){
_start:
{
uint8_t v___y_2506__boxed_1728_; size_t v_sz_boxed_1729_; size_t v_i_boxed_1730_; lean_object* v_res_1731_; 
v___y_2506__boxed_1728_ = lean_unbox(v___y_1721_);
v_sz_boxed_1729_ = lean_unbox_usize(v_sz_1723_);
lean_dec(v_sz_1723_);
v_i_boxed_1730_ = lean_unbox_usize(v_i_1724_);
lean_dec(v_i_1724_);
v_res_1731_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CalcPanel_rpc_spec__0_spec__0(v_mainGoalName_1719_, v_errorMsg_1720_, v___y_2506__boxed_1728_, v_as_1722_, v_sz_boxed_1729_, v_i_boxed_1730_, v_b_1725_, v___y_1726_);
lean_dec_ref(v___y_1726_);
lean_dec_ref(v_as_1722_);
lean_dec(v_mainGoalName_1719_);
return v_res_1731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_1732_, lean_object* v_x_1733_){
_start:
{
lean_object* v___x_1734_; 
v___x_1734_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1734_, 0, v_x_1733_);
lean_ctor_set(v___x_1734_, 1, v_expireTime_1732_);
return v___x_1734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__2(lean_object* v_val_1735_, lean_object* v___f_1736_, lean_object* v_x_1737_, lean_object* v___y_1738_){
_start:
{
if (lean_obj_tag(v_x_1737_) == 0)
{
lean_object* v_a_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1747_; 
lean_dec_ref(v___f_1736_);
v_a_1740_ = lean_ctor_get(v_x_1737_, 0);
v_isSharedCheck_1747_ = !lean_is_exclusive(v_x_1737_);
if (v_isSharedCheck_1747_ == 0)
{
v___x_1742_ = v_x_1737_;
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_a_1740_);
lean_dec(v_x_1737_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1745_; 
if (v_isShared_1743_ == 0)
{
lean_ctor_set_tag(v___x_1742_, 1);
v___x_1745_ = v___x_1742_;
goto v_reusejp_1744_;
}
else
{
lean_object* v_reuseFailAlloc_1746_; 
v_reuseFailAlloc_1746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1746_, 0, v_a_1740_);
v___x_1745_ = v_reuseFailAlloc_1746_;
goto v_reusejp_1744_;
}
v_reusejp_1744_:
{
return v___x_1745_;
}
}
}
else
{
lean_object* v_a_1748_; lean_object* v___x_1750_; uint8_t v_isShared_1751_; uint8_t v_isSharedCheck_1764_; 
v_a_1748_ = lean_ctor_get(v_x_1737_, 0);
v_isSharedCheck_1764_ = !lean_is_exclusive(v_x_1737_);
if (v_isSharedCheck_1764_ == 0)
{
v___x_1750_ = v_x_1737_;
v_isShared_1751_ = v_isSharedCheck_1764_;
goto v_resetjp_1749_;
}
else
{
lean_inc(v_a_1748_);
lean_dec(v_x_1737_);
v___x_1750_ = lean_box(0);
v_isShared_1751_ = v_isSharedCheck_1764_;
goto v_resetjp_1749_;
}
v_resetjp_1749_:
{
lean_object* v___x_1752_; lean_object* v_objects_1753_; lean_object* v_expireTime_1754_; lean_object* v___f_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v_fst_1758_; lean_object* v_snd_1759_; lean_object* v___x_1760_; lean_object* v___x_1762_; 
v___x_1752_ = lean_st_ref_take(v_val_1735_);
v_objects_1753_ = lean_ctor_get(v___x_1752_, 0);
lean_inc_ref(v_objects_1753_);
v_expireTime_1754_ = lean_ctor_get(v___x_1752_, 1);
lean_inc(v_expireTime_1754_);
lean_dec(v___x_1752_);
v___f_1755_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_1755_, 0, v_expireTime_1754_);
v___x_1756_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_a_1748_, v_objects_1753_);
v___x_1757_ = l_Prod_map___redArg(v___f_1736_, v___f_1755_, v___x_1756_);
v_fst_1758_ = lean_ctor_get(v___x_1757_, 0);
lean_inc(v_fst_1758_);
v_snd_1759_ = lean_ctor_get(v___x_1757_, 1);
lean_inc(v_snd_1759_);
lean_dec_ref(v___x_1757_);
v___x_1760_ = lean_st_ref_set(v_val_1735_, v_snd_1759_);
if (v_isShared_1751_ == 0)
{
lean_ctor_set_tag(v___x_1750_, 0);
lean_ctor_set(v___x_1750_, 0, v_fst_1758_);
v___x_1762_ = v___x_1750_;
goto v_reusejp_1761_;
}
else
{
lean_object* v_reuseFailAlloc_1763_; 
v_reuseFailAlloc_1763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1763_, 0, v_fst_1758_);
v___x_1762_ = v_reuseFailAlloc_1763_;
goto v_reusejp_1761_;
}
v_reusejp_1761_:
{
return v___x_1762_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_1765_, lean_object* v___f_1766_, lean_object* v_x_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_){
_start:
{
lean_object* v_res_1770_; 
v_res_1770_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__2(v_val_1765_, v___f_1766_, v_x_1767_, v___y_1768_);
lean_dec_ref(v___y_1768_);
lean_dec(v_val_1765_);
return v_res_1770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_1771_, uint64_t v_k_1772_){
_start:
{
if (lean_obj_tag(v_t_1771_) == 0)
{
lean_object* v_k_1773_; lean_object* v_v_1774_; lean_object* v_l_1775_; lean_object* v_r_1776_; uint64_t v___x_1777_; uint8_t v___x_1778_; 
v_k_1773_ = lean_ctor_get(v_t_1771_, 1);
v_v_1774_ = lean_ctor_get(v_t_1771_, 2);
v_l_1775_ = lean_ctor_get(v_t_1771_, 3);
v_r_1776_ = lean_ctor_get(v_t_1771_, 4);
v___x_1777_ = lean_unbox_uint64(v_k_1773_);
v___x_1778_ = lean_uint64_dec_lt(v_k_1772_, v___x_1777_);
if (v___x_1778_ == 0)
{
uint64_t v___x_1779_; uint8_t v___x_1780_; 
v___x_1779_ = lean_unbox_uint64(v_k_1773_);
v___x_1780_ = lean_uint64_dec_eq(v_k_1772_, v___x_1779_);
if (v___x_1780_ == 0)
{
v_t_1771_ = v_r_1776_;
goto _start;
}
else
{
lean_object* v___x_1782_; 
lean_inc(v_v_1774_);
v___x_1782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1782_, 0, v_v_1774_);
return v___x_1782_;
}
}
else
{
v_t_1771_ = v_l_1775_;
goto _start;
}
}
else
{
lean_object* v___x_1784_; 
v___x_1784_ = lean_box(0);
return v___x_1784_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_1785_, lean_object* v_k_1786_){
_start:
{
uint64_t v_k_boxed_1787_; lean_object* v_res_1788_; 
v_k_boxed_1787_ = lean_unbox_uint64(v_k_1786_);
lean_dec_ref(v_k_1786_);
v_res_1788_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_1785_, v_k_boxed_1787_);
lean_dec(v_t_1785_);
return v_res_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3(lean_object* v_method_1796_, lean_object* v_handler_1797_, lean_object* v___f_1798_, uint64_t v_seshId_1799_, lean_object* v_j_1800_, lean_object* v___y_1801_){
_start:
{
lean_object* v_rpcSessions_1803_; lean_object* v___x_1804_; 
v_rpcSessions_1803_ = lean_ctor_get(v___y_1801_, 0);
v___x_1804_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_1803_, v_seshId_1799_);
if (lean_obj_tag(v___x_1804_) == 1)
{
lean_object* v_val_1805_; lean_object* v___x_1806_; lean_object* v_objects_1807_; lean_object* v___x_1808_; 
v_val_1805_ = lean_ctor_get(v___x_1804_, 0);
lean_inc(v_val_1805_);
lean_dec_ref_known(v___x_1804_, 1);
v___x_1806_ = lean_st_ref_get(v_val_1805_);
v_objects_1807_ = lean_ctor_get(v___x_1806_, 0);
lean_inc_ref(v_objects_1807_);
lean_dec(v___x_1806_);
lean_inc(v_j_1800_);
v___x_1808_ = lp_mathlib_instRpcEncodableCalcParams_dec_00___x40_Mathlib_Tactic_Widget_Calc_1713132172____hygCtx___hyg_1_(v_j_1800_, v_objects_1807_);
lean_dec_ref(v_objects_1807_);
if (lean_obj_tag(v___x_1808_) == 0)
{
lean_object* v_a_1809_; lean_object* v___x_1811_; uint8_t v_isShared_1812_; uint8_t v_isSharedCheck_1829_; 
lean_dec(v_val_1805_);
lean_dec_ref(v___f_1798_);
lean_dec_ref(v_handler_1797_);
v_a_1809_ = lean_ctor_get(v___x_1808_, 0);
v_isSharedCheck_1829_ = !lean_is_exclusive(v___x_1808_);
if (v_isSharedCheck_1829_ == 0)
{
v___x_1811_ = v___x_1808_;
v_isShared_1812_ = v_isSharedCheck_1829_;
goto v_resetjp_1810_;
}
else
{
lean_inc(v_a_1809_);
lean_dec(v___x_1808_);
v___x_1811_ = lean_box(0);
v_isShared_1812_ = v_isSharedCheck_1829_;
goto v_resetjp_1810_;
}
v_resetjp_1810_:
{
uint8_t v___x_1813_; lean_object* v___x_1814_; uint8_t v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1827_; 
v___x_1813_ = 3;
v___x_1814_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_1815_ = 1;
v___x_1816_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_1796_, v___x_1815_);
v___x_1817_ = lean_string_append(v___x_1814_, v___x_1816_);
lean_dec_ref(v___x_1816_);
v___x_1818_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_1819_ = lean_string_append(v___x_1817_, v___x_1818_);
v___x_1820_ = l_Lean_Json_compress(v_j_1800_);
v___x_1821_ = lean_string_append(v___x_1819_, v___x_1820_);
lean_dec_ref(v___x_1820_);
v___x_1822_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_1823_ = lean_string_append(v___x_1821_, v___x_1822_);
v___x_1824_ = lean_string_append(v___x_1823_, v_a_1809_);
lean_dec(v_a_1809_);
v___x_1825_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1825_, 0, v___x_1824_);
lean_ctor_set_uint8(v___x_1825_, sizeof(void*)*1, v___x_1813_);
if (v_isShared_1812_ == 0)
{
lean_ctor_set_tag(v___x_1811_, 1);
lean_ctor_set(v___x_1811_, 0, v___x_1825_);
v___x_1827_ = v___x_1811_;
goto v_reusejp_1826_;
}
else
{
lean_object* v_reuseFailAlloc_1828_; 
v_reuseFailAlloc_1828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1828_, 0, v___x_1825_);
v___x_1827_ = v_reuseFailAlloc_1828_;
goto v_reusejp_1826_;
}
v_reusejp_1826_:
{
return v___x_1827_;
}
}
}
else
{
lean_object* v_a_1830_; lean_object* v___x_1831_; 
lean_dec(v_j_1800_);
lean_dec(v_method_1796_);
v_a_1830_ = lean_ctor_get(v___x_1808_, 0);
lean_inc(v_a_1830_);
lean_dec_ref_known(v___x_1808_, 1);
lean_inc_ref(v___y_1801_);
v___x_1831_ = lean_apply_3(v_handler_1797_, v_a_1830_, v___y_1801_, lean_box(0));
if (lean_obj_tag(v___x_1831_) == 0)
{
lean_object* v_a_1832_; lean_object* v___f_1833_; lean_object* v___x_1834_; 
v_a_1832_ = lean_ctor_get(v___x_1831_, 0);
lean_inc(v_a_1832_);
lean_dec_ref_known(v___x_1831_, 1);
v___f_1833_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_1833_, 0, v_val_1805_);
lean_closure_set(v___f_1833_, 1, v___f_1798_);
v___x_1834_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_1832_, v___f_1833_, v___y_1801_);
return v___x_1834_;
}
else
{
lean_object* v_a_1835_; lean_object* v___x_1837_; uint8_t v_isShared_1838_; uint8_t v_isSharedCheck_1842_; 
lean_dec(v_val_1805_);
lean_dec_ref(v___f_1798_);
v_a_1835_ = lean_ctor_get(v___x_1831_, 0);
v_isSharedCheck_1842_ = !lean_is_exclusive(v___x_1831_);
if (v_isSharedCheck_1842_ == 0)
{
v___x_1837_ = v___x_1831_;
v_isShared_1838_ = v_isSharedCheck_1842_;
goto v_resetjp_1836_;
}
else
{
lean_inc(v_a_1835_);
lean_dec(v___x_1831_);
v___x_1837_ = lean_box(0);
v_isShared_1838_ = v_isSharedCheck_1842_;
goto v_resetjp_1836_;
}
v_resetjp_1836_:
{
lean_object* v___x_1840_; 
if (v_isShared_1838_ == 0)
{
v___x_1840_ = v___x_1837_;
goto v_reusejp_1839_;
}
else
{
lean_object* v_reuseFailAlloc_1841_; 
v_reuseFailAlloc_1841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1841_, 0, v_a_1835_);
v___x_1840_ = v_reuseFailAlloc_1841_;
goto v_reusejp_1839_;
}
v_reusejp_1839_:
{
return v___x_1840_;
}
}
}
}
}
else
{
lean_object* v___x_1843_; lean_object* v___x_1844_; 
lean_dec(v___x_1804_);
lean_dec(v_j_1800_);
lean_dec_ref(v___f_1798_);
lean_dec_ref(v_handler_1797_);
lean_dec(v_method_1796_);
v___x_1843_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_1844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1844_, 0, v___x_1843_);
return v___x_1844_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_1845_, lean_object* v_handler_1846_, lean_object* v___f_1847_, lean_object* v_seshId_1848_, lean_object* v_j_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_){
_start:
{
uint64_t v_seshId_boxed_1852_; lean_object* v_res_1853_; 
v_seshId_boxed_1852_ = lean_unbox_uint64(v_seshId_1848_);
lean_dec_ref(v_seshId_1848_);
v_res_1853_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3(v_method_1845_, v_handler_1846_, v___f_1847_, v_seshId_boxed_1852_, v_j_1849_, v___y_1850_);
lean_dec_ref(v___y_1850_);
return v_res_1853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__0(lean_object* v___y_1854_){
_start:
{
lean_inc(v___y_1854_);
return v___y_1854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_1855_){
_start:
{
lean_object* v_res_1856_; 
v_res_1856_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__0(v___y_1855_);
lean_dec(v___y_1855_);
return v_res_1856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0(lean_object* v_method_1858_, lean_object* v_handler_1859_){
_start:
{
lean_object* v___f_1860_; lean_object* v___f_1861_; 
v___f_1860_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___closed__0));
v___f_1861_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_1861_, 0, v_method_1858_);
lean_closure_set(v___f_1861_, 1, v_handler_1859_);
lean_closure_set(v___f_1861_, 2, v___f_1860_);
return v___f_1861_;
}
}
static lean_object* _init_lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__4(void){
_start:
{
lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; 
v___x_1868_ = ((lean_object*)(lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__3));
v___x_1869_ = ((lean_object*)(lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__2));
v___x_1870_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0(v___x_1869_, v___x_1868_);
return v___x_1870_;
}
}
static lean_object* _init_lp_mathlib_CalcPanel_rpc___rpc__wrapped(void){
_start:
{
lean_object* v___x_1871_; 
v___x_1871_ = lean_obj_once(&lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__4, &lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__4_once, _init_lp_mathlib_CalcPanel_rpc___rpc__wrapped___closed__4);
return v___x_1871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_1872_, lean_object* v_t_1873_, uint64_t v_k_1874_){
_start:
{
lean_object* v___x_1875_; 
v___x_1875_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_1873_, v_k_1874_);
return v___x_1875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_1876_, lean_object* v_t_1877_, lean_object* v_k_1878_){
_start:
{
uint64_t v_k_boxed_1879_; lean_object* v_res_1880_; 
v_k_boxed_1879_ = lean_unbox_uint64(v_k_1878_);
lean_dec_ref(v_k_1878_);
v_res_1880_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CalcPanel_rpc___rpc__wrapped_spec__0_spec__0(v_00_u03b4_1876_, v_t_1877_, v_k_boxed_1879_);
lean_dec(v_t_1877_);
return v_res_1880_;
}
}
static uint64_t _init_lp_mathlib_CalcPanel___closed__1(void){
_start:
{
lean_object* v___x_1882_; uint64_t v___x_1883_; 
v___x_1882_ = ((lean_object*)(lp_mathlib_CalcPanel___closed__0));
v___x_1883_ = lean_string_hash(v___x_1882_);
return v___x_1883_;
}
}
static lean_object* _init_lp_mathlib_CalcPanel___closed__2(void){
_start:
{
uint64_t v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; 
v___x_1884_ = lean_uint64_once(&lp_mathlib_CalcPanel___closed__1, &lp_mathlib_CalcPanel___closed__1_once, _init_lp_mathlib_CalcPanel___closed__1);
v___x_1885_ = ((lean_object*)(lp_mathlib_CalcPanel___closed__0));
v___x_1886_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1886_, 0, v___x_1885_);
lean_ctor_set_uint64(v___x_1886_, sizeof(void*)*1, v___x_1884_);
return v___x_1886_;
}
}
static lean_object* _init_lp_mathlib_CalcPanel___closed__4(void){
_start:
{
lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; 
v___x_1888_ = ((lean_object*)(lp_mathlib_CalcPanel___closed__3));
v___x_1889_ = lean_obj_once(&lp_mathlib_CalcPanel___closed__2, &lp_mathlib_CalcPanel___closed__2_once, _init_lp_mathlib_CalcPanel___closed__2);
v___x_1890_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1890_, 0, v___x_1889_);
lean_ctor_set(v___x_1890_, 1, v___x_1888_);
return v___x_1890_;
}
}
static lean_object* _init_lp_mathlib_CalcPanel(void){
_start:
{
lean_object* v___x_1891_; 
v___x_1891_ = lean_obj_once(&lp_mathlib_CalcPanel___closed__4, &lp_mathlib_CalcPanel___closed__4_once, _init_lp_mathlib_CalcPanel___closed__4);
return v___x_1891_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; 
v___x_1910_ = lean_box(0);
v___x_1911_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1912_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1912_, 0, v___x_1911_);
lean_ctor_set(v___x_1912_, 1, v___x_1910_);
return v___x_1912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; 
v___x_1914_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___closed__0);
v___x_1915_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1915_, 0, v___x_1914_);
return v___x_1915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg___boxed(lean_object* v___y_1916_){
_start:
{
lean_object* v_res_1917_; 
v_res_1917_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg();
return v_res_1917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0(lean_object* v_00_u03b1_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_){
_start:
{
lean_object* v___x_1928_; 
v___x_1928_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg();
return v___x_1928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___boxed(lean_object* v_00_u03b1_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_){
_start:
{
lean_object* v_res_1939_; 
v_res_1939_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0(v_00_u03b1_1929_, v___y_1930_, v___y_1931_, v___y_1932_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_, v___y_1937_);
lean_dec(v___y_1937_);
lean_dec_ref(v___y_1936_);
lean_dec(v___y_1935_);
lean_dec_ref(v___y_1934_);
lean_dec(v___y_1933_);
lean_dec_ref(v___y_1932_);
lean_dec(v___y_1931_);
lean_dec_ref(v___y_1930_);
return v_res_1939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg(lean_object* v_msg_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_){
_start:
{
lean_object* v_ref_1946_; lean_object* v___x_1947_; lean_object* v_a_1948_; lean_object* v___x_1950_; uint8_t v_isShared_1951_; uint8_t v_isSharedCheck_1956_; 
v_ref_1946_ = lean_ctor_get(v___y_1943_, 5);
v___x_1947_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00suggestSteps_spec__3_spec__4(v_msg_1940_, v___y_1941_, v___y_1942_, v___y_1943_, v___y_1944_);
v_a_1948_ = lean_ctor_get(v___x_1947_, 0);
v_isSharedCheck_1956_ = !lean_is_exclusive(v___x_1947_);
if (v_isSharedCheck_1956_ == 0)
{
v___x_1950_ = v___x_1947_;
v_isShared_1951_ = v_isSharedCheck_1956_;
goto v_resetjp_1949_;
}
else
{
lean_inc(v_a_1948_);
lean_dec(v___x_1947_);
v___x_1950_ = lean_box(0);
v_isShared_1951_ = v_isSharedCheck_1956_;
goto v_resetjp_1949_;
}
v_resetjp_1949_:
{
lean_object* v___x_1952_; lean_object* v___x_1954_; 
lean_inc(v_ref_1946_);
v___x_1952_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1952_, 0, v_ref_1946_);
lean_ctor_set(v___x_1952_, 1, v_a_1948_);
if (v_isShared_1951_ == 0)
{
lean_ctor_set_tag(v___x_1950_, 1);
lean_ctor_set(v___x_1950_, 0, v___x_1952_);
v___x_1954_ = v___x_1950_;
goto v_reusejp_1953_;
}
else
{
lean_object* v_reuseFailAlloc_1955_; 
v_reuseFailAlloc_1955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1955_, 0, v___x_1952_);
v___x_1954_ = v_reuseFailAlloc_1955_;
goto v_reusejp_1953_;
}
v_reusejp_1953_:
{
return v___x_1954_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg___boxed(lean_object* v_msg_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_){
_start:
{
lean_object* v_res_1963_; 
v_res_1963_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg(v_msg_1957_, v___y_1958_, v___y_1959_, v___y_1960_, v___y_1961_);
lean_dec(v___y_1961_);
lean_dec_ref(v___y_1960_);
lean_dec(v___y_1959_);
lean_dec_ref(v___y_1958_);
return v_res_1963_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__17(void){
_start:
{
lean_object* v___x_1983_; 
v___x_1983_ = l_Array_mkArray0(lean_box(0));
return v___x_1983_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__20(void){
_start:
{
lean_object* v___x_1986_; lean_object* v___x_1987_; 
v___x_1986_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__19));
v___x_1987_ = l_Lean_stringToMessageData(v___x_1986_);
return v___x_1987_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__22(void){
_start:
{
lean_object* v___x_1989_; lean_object* v___x_1990_; 
v___x_1989_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__21));
v___x_1990_ = l_Lean_stringToMessageData(v___x_1989_);
return v___x_1990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0(lean_object* v___x_1991_, lean_object* v___x_1992_, lean_object* v_stx_1993_, uint8_t v___x_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_){
_start:
{
lean_object* v___y_2005_; lean_object* v___y_2006_; lean_object* v___y_2007_; lean_object* v___y_2008_; lean_object* v___y_2009_; lean_object* v___y_2010_; lean_object* v___y_2011_; lean_object* v___y_2012_; lean_object* v___x_2085_; 
v___x_2085_ = l_Lean_Elab_Tactic_getMainTarget(v___y_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_);
if (lean_obj_tag(v___x_2085_) == 0)
{
lean_object* v_a_2086_; lean_object* v___x_2087_; 
v_a_2086_ = lean_ctor_get(v___x_2085_, 0);
lean_inc(v_a_2086_);
lean_dec_ref_known(v___x_2085_, 1);
v___x_2087_ = l_Lean_Meta_whnfR(v_a_2086_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_);
if (lean_obj_tag(v___x_2087_) == 0)
{
lean_object* v_a_2088_; lean_object* v___x_2089_; 
v_a_2088_ = lean_ctor_get(v___x_2087_, 0);
lean_inc(v_a_2088_);
lean_dec_ref_known(v___x_2087_, 1);
v___x_2089_ = l_Lean_Elab_Term_getCalcRelation_x3f___redArg(v_a_2088_);
if (lean_obj_tag(v___x_2089_) == 0)
{
lean_object* v_a_2090_; 
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
lean_inc(v_a_2090_);
lean_dec_ref_known(v___x_2089_, 1);
if (lean_obj_tag(v_a_2090_) == 0)
{
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
goto v___jp_2091_;
}
else
{
lean_dec_ref_known(v_a_2090_, 1);
if (v___x_1994_ == 0)
{
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
goto v___jp_2091_;
}
else
{
lean_dec(v_a_2088_);
v___y_2005_ = v___y_1995_;
v___y_2006_ = v___y_1996_;
v___y_2007_ = v___y_1997_;
v___y_2008_ = v___y_1998_;
v___y_2009_ = v___y_1999_;
v___y_2010_ = v___y_2000_;
v___y_2011_ = v___y_2001_;
v___y_2012_ = v___y_2002_;
goto v___jp_2004_;
}
}
v___jp_2091_:
{
lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; 
v___x_2092_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__20, &lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__20_once, _init_lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__20);
v___x_2093_ = l_Lean_indentExpr(v_a_2088_);
v___x_2094_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2094_, 0, v___x_2092_);
lean_ctor_set(v___x_2094_, 1, v___x_2093_);
v___x_2095_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__22, &lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__22_once, _init_lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__22);
v___x_2096_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2096_, 0, v___x_2094_);
lean_ctor_set(v___x_2096_, 1, v___x_2095_);
v___x_2097_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg(v___x_2096_, v___y_1999_, v___y_2000_, v___y_2001_, v___y_2002_);
return v___x_2097_;
}
}
else
{
lean_object* v_a_2098_; lean_object* v___x_2100_; uint8_t v_isShared_2101_; uint8_t v_isSharedCheck_2105_; 
lean_dec(v_a_2088_);
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
v_a_2098_ = lean_ctor_get(v___x_2089_, 0);
v_isSharedCheck_2105_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2105_ == 0)
{
v___x_2100_ = v___x_2089_;
v_isShared_2101_ = v_isSharedCheck_2105_;
goto v_resetjp_2099_;
}
else
{
lean_inc(v_a_2098_);
lean_dec(v___x_2089_);
v___x_2100_ = lean_box(0);
v_isShared_2101_ = v_isSharedCheck_2105_;
goto v_resetjp_2099_;
}
v_resetjp_2099_:
{
lean_object* v___x_2103_; 
if (v_isShared_2101_ == 0)
{
v___x_2103_ = v___x_2100_;
goto v_reusejp_2102_;
}
else
{
lean_object* v_reuseFailAlloc_2104_; 
v_reuseFailAlloc_2104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2104_, 0, v_a_2098_);
v___x_2103_ = v_reuseFailAlloc_2104_;
goto v_reusejp_2102_;
}
v_reusejp_2102_:
{
return v___x_2103_;
}
}
}
}
else
{
lean_object* v_a_2106_; lean_object* v___x_2108_; uint8_t v_isShared_2109_; uint8_t v_isSharedCheck_2113_; 
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
v_a_2106_ = lean_ctor_get(v___x_2087_, 0);
v_isSharedCheck_2113_ = !lean_is_exclusive(v___x_2087_);
if (v_isSharedCheck_2113_ == 0)
{
v___x_2108_ = v___x_2087_;
v_isShared_2109_ = v_isSharedCheck_2113_;
goto v_resetjp_2107_;
}
else
{
lean_inc(v_a_2106_);
lean_dec(v___x_2087_);
v___x_2108_ = lean_box(0);
v_isShared_2109_ = v_isSharedCheck_2113_;
goto v_resetjp_2107_;
}
v_resetjp_2107_:
{
lean_object* v___x_2111_; 
if (v_isShared_2109_ == 0)
{
v___x_2111_ = v___x_2108_;
goto v_reusejp_2110_;
}
else
{
lean_object* v_reuseFailAlloc_2112_; 
v_reuseFailAlloc_2112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2112_, 0, v_a_2106_);
v___x_2111_ = v_reuseFailAlloc_2112_;
goto v_reusejp_2110_;
}
v_reusejp_2110_:
{
return v___x_2111_;
}
}
}
}
else
{
lean_object* v_a_2114_; lean_object* v___x_2116_; uint8_t v_isShared_2117_; uint8_t v_isSharedCheck_2121_; 
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
v_a_2114_ = lean_ctor_get(v___x_2085_, 0);
v_isSharedCheck_2121_ = !lean_is_exclusive(v___x_2085_);
if (v_isSharedCheck_2121_ == 0)
{
v___x_2116_ = v___x_2085_;
v_isShared_2117_ = v_isSharedCheck_2121_;
goto v_resetjp_2115_;
}
else
{
lean_inc(v_a_2114_);
lean_dec(v___x_2085_);
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
v___jp_2004_:
{
lean_object* v___x_2013_; 
v___x_2013_ = l_Lean_Elab_Tactic_getMainTarget(v___y_2005_, v___y_2006_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_);
if (lean_obj_tag(v___x_2013_) == 0)
{
lean_object* v_a_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; 
v_a_2014_ = lean_ctor_get(v___x_2013_, 0);
lean_inc(v_a_2014_);
lean_dec_ref_known(v___x_2013_, 1);
v___x_2015_ = lean_box(1);
v___x_2016_ = l_Lean_PrettyPrinter_delab(v_a_2014_, v___x_2015_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_);
if (lean_obj_tag(v___x_2016_) == 0)
{
lean_object* v_a_2017_; lean_object* v_ref_2018_; uint8_t v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; uint8_t v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; 
v_a_2017_ = lean_ctor_get(v___x_2016_, 0);
lean_inc(v_a_2017_);
lean_dec_ref_known(v___x_2016_, 1);
v_ref_2018_ = lean_ctor_get(v___y_2011_, 5);
v___x_2019_ = 0;
v___x_2020_ = l_Lean_SourceInfo_fromRef(v_ref_2018_, v___x_2019_);
v___x_2021_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__1));
v___x_2022_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__2));
lean_inc_ref_n(v___x_1991_, 6);
v___x_2023_ = l_Lean_Name_mkStr2(v___x_1991_, v___x_2022_);
v___x_2024_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__3));
lean_inc_n(v___x_2020_, 13);
v___x_2025_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2025_, 0, v___x_2020_);
lean_ctor_set(v___x_2025_, 1, v___x_2024_);
v___x_2026_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__4));
v___x_2027_ = l_Lean_Name_mkStr2(v___x_1991_, v___x_2026_);
v___x_2028_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__5));
v___x_2029_ = l_Lean_Name_mkStr2(v___x_1991_, v___x_2028_);
v___x_2030_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__7));
v___x_2031_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__8));
v___x_2032_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2032_, 0, v___x_2020_);
lean_ctor_set(v___x_2032_, 1, v___x_2031_);
v___x_2033_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__9));
v___x_2034_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__10));
v___x_2035_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__11));
v___x_2036_ = l_Lean_Name_mkStr4(v___x_1991_, v___x_2033_, v___x_2034_, v___x_2035_);
v___x_2037_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__12));
v___x_2038_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2038_, 0, v___x_2020_);
lean_ctor_set(v___x_2038_, 1, v___x_2037_);
v___x_2039_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__13));
lean_inc_ref_n(v___x_1992_, 2);
v___x_2040_ = l_Lean_Name_mkStr4(v___x_1991_, v___x_2033_, v___x_1992_, v___x_2039_);
v___x_2041_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__14));
v___x_2042_ = l_Lean_Name_mkStr4(v___x_1991_, v___x_2033_, v___x_1992_, v___x_2041_);
v___x_2043_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__15));
v___x_2044_ = l_Lean_Name_mkStr4(v___x_1991_, v___x_2033_, v___x_1992_, v___x_2043_);
v___x_2045_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__16));
v___x_2046_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2046_, 0, v___x_2020_);
lean_ctor_set(v___x_2046_, 1, v___x_2045_);
v___x_2047_ = l_Lean_Syntax_node1(v___x_2020_, v___x_2044_, v___x_2046_);
lean_inc(v___x_2047_);
v___x_2048_ = l_Lean_Syntax_node1(v___x_2020_, v___x_2030_, v___x_2047_);
v___x_2049_ = l_Lean_Syntax_node1(v___x_2020_, v___x_2042_, v___x_2048_);
v___x_2050_ = l_Lean_Syntax_node1(v___x_2020_, v___x_2040_, v___x_2049_);
v___x_2051_ = l_Lean_Syntax_node2(v___x_2020_, v___x_2036_, v___x_2038_, v___x_2050_);
v___x_2052_ = l_Lean_Syntax_node2(v___x_2020_, v___x_2030_, v___x_2032_, v___x_2051_);
v___x_2053_ = l_Lean_Syntax_node2(v___x_2020_, v___x_2029_, v_a_2017_, v___x_2052_);
v___x_2054_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__17, &lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__17_once, _init_lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__17);
v___x_2055_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2055_, 0, v___x_2020_);
lean_ctor_set(v___x_2055_, 1, v___x_2030_);
lean_ctor_set(v___x_2055_, 2, v___x_2054_);
v___x_2056_ = l_Lean_Syntax_node2(v___x_2020_, v___x_2027_, v___x_2053_, v___x_2055_);
v___x_2057_ = l_Lean_Syntax_node2(v___x_2020_, v___x_2023_, v___x_2025_, v___x_2056_);
v___x_2058_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2058_, 0, v___x_2021_);
lean_ctor_set(v___x_2058_, 1, v___x_2057_);
v___x_2059_ = lean_box(0);
v___x_2060_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2060_, 0, v___x_2058_);
lean_ctor_set(v___x_2060_, 1, v___x_2059_);
lean_ctor_set(v___x_2060_, 2, v___x_2059_);
lean_ctor_set(v___x_2060_, 3, v___x_2059_);
lean_ctor_set(v___x_2060_, 4, v___x_2059_);
lean_ctor_set(v___x_2060_, 5, v___x_2059_);
v___x_2061_ = lean_unsigned_to_nat(1u);
v___x_2062_ = lean_mk_empty_array_with_capacity(v___x_2061_);
v___x_2063_ = lean_array_push(v___x_2062_, v___x_2060_);
v___x_2064_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__18));
v___x_2065_ = 4;
v___x_2066_ = l_Lean_MessageData_nil;
v___x_2067_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_1993_, v___x_2063_, v___x_2059_, v___x_2064_, v___x_2059_, v___x_2065_, v___x_2066_, v___y_2011_, v___y_2012_);
if (lean_obj_tag(v___x_2067_) == 0)
{
lean_object* v___x_2068_; 
lean_dec_ref_known(v___x_2067_, 1);
v___x_2068_ = l_Lean_Elab_Tactic_evalTactic(v___x_2047_, v___y_2005_, v___y_2006_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_, v___y_2012_);
return v___x_2068_;
}
else
{
lean_dec(v___x_2047_);
return v___x_2067_;
}
}
else
{
lean_object* v_a_2069_; lean_object* v___x_2071_; uint8_t v_isShared_2072_; uint8_t v_isSharedCheck_2076_; 
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
v_a_2069_ = lean_ctor_get(v___x_2016_, 0);
v_isSharedCheck_2076_ = !lean_is_exclusive(v___x_2016_);
if (v_isSharedCheck_2076_ == 0)
{
v___x_2071_ = v___x_2016_;
v_isShared_2072_ = v_isSharedCheck_2076_;
goto v_resetjp_2070_;
}
else
{
lean_inc(v_a_2069_);
lean_dec(v___x_2016_);
v___x_2071_ = lean_box(0);
v_isShared_2072_ = v_isSharedCheck_2076_;
goto v_resetjp_2070_;
}
v_resetjp_2070_:
{
lean_object* v___x_2074_; 
if (v_isShared_2072_ == 0)
{
v___x_2074_ = v___x_2071_;
goto v_reusejp_2073_;
}
else
{
lean_object* v_reuseFailAlloc_2075_; 
v_reuseFailAlloc_2075_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2075_, 0, v_a_2069_);
v___x_2074_ = v_reuseFailAlloc_2075_;
goto v_reusejp_2073_;
}
v_reusejp_2073_:
{
return v___x_2074_;
}
}
}
}
else
{
lean_object* v_a_2077_; lean_object* v___x_2079_; uint8_t v_isShared_2080_; uint8_t v_isSharedCheck_2084_; 
lean_dec(v_stx_1993_);
lean_dec_ref(v___x_1992_);
lean_dec_ref(v___x_1991_);
v_a_2077_ = lean_ctor_get(v___x_2013_, 0);
v_isSharedCheck_2084_ = !lean_is_exclusive(v___x_2013_);
if (v_isSharedCheck_2084_ == 0)
{
v___x_2079_ = v___x_2013_;
v_isShared_2080_ = v_isSharedCheck_2084_;
goto v_resetjp_2078_;
}
else
{
lean_inc(v_a_2077_);
lean_dec(v___x_2013_);
v___x_2079_ = lean_box(0);
v_isShared_2080_ = v_isSharedCheck_2084_;
goto v_resetjp_2078_;
}
v_resetjp_2078_:
{
lean_object* v___x_2082_; 
if (v_isShared_2080_ == 0)
{
v___x_2082_ = v___x_2079_;
goto v_reusejp_2081_;
}
else
{
lean_object* v_reuseFailAlloc_2083_; 
v_reuseFailAlloc_2083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2083_, 0, v_a_2077_);
v___x_2082_ = v_reuseFailAlloc_2083_;
goto v_reusejp_2081_;
}
v_reusejp_2081_:
{
return v___x_2082_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___boxed(lean_object* v___x_2122_, lean_object* v___x_2123_, lean_object* v_stx_2124_, lean_object* v___x_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_){
_start:
{
uint8_t v___x_8642__boxed_2135_; lean_object* v_res_2136_; 
v___x_8642__boxed_2135_ = lean_unbox(v___x_2125_);
v_res_2136_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0(v___x_2122_, v___x_2123_, v_stx_2124_, v___x_8642__boxed_2135_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_, v___y_2131_, v___y_2132_, v___y_2133_);
lean_dec(v___y_2133_);
lean_dec_ref(v___y_2132_);
lean_dec(v___y_2131_);
lean_dec_ref(v___y_2130_);
lean_dec(v___y_2129_);
lean_dec_ref(v___y_2128_);
lean_dec(v___y_2127_);
lean_dec_ref(v___y_2126_);
return v_res_2136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1(lean_object* v_x_2137_, lean_object* v_a_2138_, lean_object* v_a_2139_, lean_object* v_a_2140_, lean_object* v_a_2141_, lean_object* v_a_2142_, lean_object* v_a_2143_, lean_object* v_a_2144_, lean_object* v_a_2145_){
_start:
{
lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; uint8_t v___x_2150_; 
v___x_2147_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__0));
v___x_2148_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__2));
v___x_2149_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_tacticCalc_x3f___closed__4));
lean_inc(v_x_2137_);
v___x_2150_ = l_Lean_Syntax_isOfKind(v_x_2137_, v___x_2149_);
if (v___x_2150_ == 0)
{
lean_object* v___x_2151_; 
lean_dec(v_x_2137_);
v___x_2151_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg();
return v___x_2151_;
}
else
{
lean_object* v___x_2152_; lean_object* v_stx_2153_; lean_object* v___x_2154_; lean_object* v___f_2155_; lean_object* v___x_2156_; 
v___x_2152_ = lean_unsigned_to_nat(0u);
v_stx_2153_ = l_Lean_Syntax_getArg(v_x_2137_, v___x_2152_);
lean_dec(v_x_2137_);
v___x_2154_ = lean_box(v___x_2150_);
v___f_2155_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_2155_, 0, v___x_2147_);
lean_closure_set(v___f_2155_, 1, v___x_2148_);
lean_closure_set(v___f_2155_, 2, v_stx_2153_);
lean_closure_set(v___f_2155_, 3, v___x_2154_);
v___x_2156_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2155_, v_a_2138_, v_a_2139_, v_a_2140_, v_a_2141_, v_a_2142_, v_a_2143_, v_a_2144_, v_a_2145_);
return v___x_2156_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___boxed(lean_object* v_x_2157_, lean_object* v_a_2158_, lean_object* v_a_2159_, lean_object* v_a_2160_, lean_object* v_a_2161_, lean_object* v_a_2162_, lean_object* v_a_2163_, lean_object* v_a_2164_, lean_object* v_a_2165_, lean_object* v_a_2166_){
_start:
{
lean_object* v_res_2167_; 
v_res_2167_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1(v_x_2157_, v_a_2158_, v_a_2159_, v_a_2160_, v_a_2161_, v_a_2162_, v_a_2163_, v_a_2164_, v_a_2165_);
lean_dec(v_a_2165_);
lean_dec_ref(v_a_2164_);
lean_dec(v_a_2163_);
lean_dec_ref(v_a_2162_);
lean_dec(v_a_2161_);
lean_dec_ref(v_a_2160_);
lean_dec(v_a_2159_);
lean_dec_ref(v_a_2158_);
return v_res_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1(lean_object* v_00_u03b1_2168_, lean_object* v_msg_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_){
_start:
{
lean_object* v___x_2179_; 
v___x_2179_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___redArg(v_msg_2169_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
return v___x_2179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1___boxed(lean_object* v_00_u03b1_2180_, lean_object* v_msg_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_, lean_object* v___y_2189_, lean_object* v___y_2190_){
_start:
{
lean_object* v_res_2191_; 
v_res_2191_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__1(v_00_u03b1_2180_, v_msg_2181_, v___y_2182_, v___y_2183_, v___y_2184_, v___y_2185_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
lean_dec(v___y_2189_);
lean_dec_ref(v___y_2188_);
lean_dec(v___y_2187_);
lean_dec_ref(v___y_2186_);
lean_dec(v___y_2185_);
lean_dec_ref(v___y_2184_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
return v_res_2191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg___lam__0(lean_object* v___x_2192_, lean_object* v___y_2193_){
_start:
{
lean_object* v___x_2194_; 
v___x_2194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2194_, 0, v___x_2192_);
lean_ctor_set(v___x_2194_, 1, v___y_2193_);
return v___x_2194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg(lean_object* v_as_2195_, size_t v_sz_2196_, size_t v_i_2197_, uint8_t v_b_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_){
_start:
{
uint8_t v_a_2203_; uint8_t v___x_2207_; 
v___x_2207_ = lean_usize_dec_lt(v_i_2197_, v_sz_2196_);
if (v___x_2207_ == 0)
{
lean_object* v___x_2208_; lean_object* v___x_2209_; 
v___x_2208_ = lean_box(v_b_2198_);
v___x_2209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2209_, 0, v___x_2208_);
return v___x_2209_;
}
else
{
lean_object* v_fileMap_2210_; lean_object* v_a_2211_; lean_object* v_ref_2212_; lean_object* v_proof_2213_; uint8_t v___x_2214_; lean_object* v___x_2215_; 
v_fileMap_2210_ = lean_ctor_get(v___y_2199_, 1);
v_a_2211_ = lean_array_uget_borrowed(v_as_2195_, v_i_2197_);
v_ref_2212_ = lean_ctor_get(v_a_2211_, 0);
v_proof_2213_ = lean_ctor_get(v_a_2211_, 2);
v___x_2214_ = 0;
lean_inc_ref(v_fileMap_2210_);
v___x_2215_ = l_Lean_FileMap_lspRangeOfStx_x3f(v_fileMap_2210_, v_ref_2212_, v___x_2214_);
if (lean_obj_tag(v___x_2215_) == 1)
{
lean_object* v_val_2216_; lean_object* v___x_2218_; uint8_t v_isShared_2219_; uint8_t v_isSharedCheck_2259_; 
v_val_2216_ = lean_ctor_get(v___x_2215_, 0);
v_isSharedCheck_2259_ = !lean_is_exclusive(v___x_2215_);
if (v_isSharedCheck_2259_ == 0)
{
v___x_2218_ = v___x_2215_;
v_isShared_2219_ = v_isSharedCheck_2259_;
goto v_resetjp_2217_;
}
else
{
lean_inc(v_val_2216_);
lean_dec(v___x_2215_);
v___x_2218_ = lean_box(0);
v_isShared_2219_ = v_isSharedCheck_2259_;
goto v_resetjp_2217_;
}
v_resetjp_2217_:
{
lean_object* v_start_2220_; lean_object* v_character_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2257_; 
v_start_2220_ = lean_ctor_get(v_val_2216_, 0);
lean_inc_ref(v_start_2220_);
v_character_2221_ = lean_ctor_get(v_start_2220_, 1);
v_isSharedCheck_2257_ = !lean_is_exclusive(v_start_2220_);
if (v_isSharedCheck_2257_ == 0)
{
lean_object* v_unused_2258_; 
v_unused_2258_ = lean_ctor_get(v_start_2220_, 0);
lean_dec(v_unused_2258_);
v___x_2223_ = v_start_2220_;
v_isShared_2224_ = v_isSharedCheck_2257_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_character_2221_);
lean_dec(v_start_2220_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2257_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v___x_2225_; lean_object* v_toModule_2226_; uint64_t v_javascriptHash_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2231_; 
v___x_2225_ = lp_mathlib_CalcPanel;
v_toModule_2226_ = lean_ctor_get(v___x_2225_, 0);
v_javascriptHash_2227_ = lean_ctor_get_uint64(v_toModule_2226_, sizeof(void*)*1);
v___x_2228_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_2229_ = l_Lean_Lsp_instToJsonRange_toJson(v_val_2216_);
if (v_isShared_2224_ == 0)
{
lean_ctor_set(v___x_2223_, 1, v___x_2229_);
lean_ctor_set(v___x_2223_, 0, v___x_2228_);
v___x_2231_ = v___x_2223_;
goto v_reusejp_2230_;
}
else
{
lean_object* v_reuseFailAlloc_2256_; 
v_reuseFailAlloc_2256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2256_, 0, v___x_2228_);
lean_ctor_set(v_reuseFailAlloc_2256_, 1, v___x_2229_);
v___x_2231_ = v_reuseFailAlloc_2256_;
goto v_reusejp_2230_;
}
v_reusejp_2230_:
{
lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2238_; 
v___x_2232_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_2233_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_2233_, 0, v_b_2198_);
v___x_2234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2234_, 0, v___x_2232_);
lean_ctor_set(v___x_2234_, 1, v___x_2233_);
v___x_2235_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_Mathlib_Tactic_Widget_Calc_3238076937____hygCtx___hyg_20_));
v___x_2236_ = l_Lean_JsonNumber_fromNat(v_character_2221_);
if (v_isShared_2219_ == 0)
{
lean_ctor_set_tag(v___x_2218_, 2);
lean_ctor_set(v___x_2218_, 0, v___x_2236_);
v___x_2238_ = v___x_2218_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2255_; 
v_reuseFailAlloc_2255_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2255_, 0, v___x_2236_);
v___x_2238_ = v_reuseFailAlloc_2255_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___f_2245_; lean_object* v___x_2246_; 
v___x_2239_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2239_, 0, v___x_2235_);
lean_ctor_set(v___x_2239_, 1, v___x_2238_);
v___x_2240_ = lean_box(0);
v___x_2241_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2241_, 0, v___x_2239_);
lean_ctor_set(v___x_2241_, 1, v___x_2240_);
v___x_2242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2242_, 0, v___x_2234_);
lean_ctor_set(v___x_2242_, 1, v___x_2241_);
v___x_2243_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2231_);
lean_ctor_set(v___x_2243_, 1, v___x_2242_);
v___x_2244_ = l_Lean_Json_mkObj(v___x_2243_);
lean_dec_ref_known(v___x_2243_, 2);
v___f_2245_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2245_, 0, v___x_2244_);
lean_inc(v_proof_2213_);
v___x_2246_ = l_Lean_Widget_savePanelWidgetInfo(v_javascriptHash_2227_, v___f_2245_, v_proof_2213_, v___y_2199_, v___y_2200_);
if (lean_obj_tag(v___x_2246_) == 0)
{
lean_dec_ref_known(v___x_2246_, 1);
v_a_2203_ = v___x_2214_;
goto v___jp_2202_;
}
else
{
lean_object* v_a_2247_; lean_object* v___x_2249_; uint8_t v_isShared_2250_; uint8_t v_isSharedCheck_2254_; 
v_a_2247_ = lean_ctor_get(v___x_2246_, 0);
v_isSharedCheck_2254_ = !lean_is_exclusive(v___x_2246_);
if (v_isSharedCheck_2254_ == 0)
{
v___x_2249_ = v___x_2246_;
v_isShared_2250_ = v_isSharedCheck_2254_;
goto v_resetjp_2248_;
}
else
{
lean_inc(v_a_2247_);
lean_dec(v___x_2246_);
v___x_2249_ = lean_box(0);
v_isShared_2250_ = v_isSharedCheck_2254_;
goto v_resetjp_2248_;
}
v_resetjp_2248_:
{
lean_object* v___x_2252_; 
if (v_isShared_2250_ == 0)
{
v___x_2252_ = v___x_2249_;
goto v_reusejp_2251_;
}
else
{
lean_object* v_reuseFailAlloc_2253_; 
v_reuseFailAlloc_2253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2253_, 0, v_a_2247_);
v___x_2252_ = v_reuseFailAlloc_2253_;
goto v_reusejp_2251_;
}
v_reusejp_2251_:
{
return v___x_2252_;
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
lean_dec(v___x_2215_);
v_a_2203_ = v_b_2198_;
goto v___jp_2202_;
}
}
v___jp_2202_:
{
size_t v___x_2204_; size_t v___x_2205_; 
v___x_2204_ = ((size_t)1ULL);
v___x_2205_ = lean_usize_add(v_i_2197_, v___x_2204_);
v_i_2197_ = v___x_2205_;
v_b_2198_ = v_a_2203_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg___boxed(lean_object* v_as_2260_, lean_object* v_sz_2261_, lean_object* v_i_2262_, lean_object* v_b_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_){
_start:
{
size_t v_sz_boxed_2267_; size_t v_i_boxed_2268_; uint8_t v_b_boxed_2269_; lean_object* v_res_2270_; 
v_sz_boxed_2267_ = lean_unbox_usize(v_sz_2261_);
lean_dec(v_sz_2261_);
v_i_boxed_2268_ = lean_unbox_usize(v_i_2262_);
lean_dec(v_i_2262_);
v_b_boxed_2269_ = lean_unbox(v_b_2263_);
v_res_2270_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg(v_as_2260_, v_sz_boxed_2267_, v_i_boxed_2268_, v_b_boxed_2269_, v___y_2264_, v___y_2265_);
lean_dec(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec_ref(v_as_2260_);
return v_res_2270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1(lean_object* v_x_2274_, lean_object* v_a_2275_, lean_object* v_a_2276_, lean_object* v_a_2277_, lean_object* v_a_2278_, lean_object* v_a_2279_, lean_object* v_a_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_){
_start:
{
lean_object* v___x_2284_; uint8_t v_isFirst_2285_; 
v___x_2284_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___closed__0));
lean_inc(v_x_2274_);
v_isFirst_2285_ = l_Lean_Syntax_isOfKind(v_x_2274_, v___x_2284_);
if (v_isFirst_2285_ == 0)
{
lean_object* v___x_2286_; 
lean_dec(v_x_2274_);
v___x_2286_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1_spec__0___redArg();
return v___x_2286_;
}
else
{
lean_object* v___x_2287_; lean_object* v_steps_2288_; lean_object* v___x_2289_; 
v___x_2287_ = lean_unsigned_to_nat(1u);
v_steps_2288_ = l_Lean_Syntax_getArg(v_x_2274_, v___x_2287_);
lean_inc(v_steps_2288_);
v___x_2289_ = l_Lean_Elab_Term_mkCalcStepViews(v_steps_2288_, v_a_2277_, v_a_2278_, v_a_2279_, v_a_2280_, v_a_2281_, v_a_2282_);
if (lean_obj_tag(v___x_2289_) == 0)
{
lean_object* v_a_2290_; size_t v_sz_2291_; size_t v___x_2292_; lean_object* v___x_2293_; 
v_a_2290_ = lean_ctor_get(v___x_2289_, 0);
lean_inc(v_a_2290_);
lean_dec_ref_known(v___x_2289_, 1);
v_sz_2291_ = lean_array_size(v_a_2290_);
v___x_2292_ = ((size_t)0ULL);
v___x_2293_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg(v_a_2290_, v_sz_2291_, v___x_2292_, v_isFirst_2285_, v_a_2281_, v_a_2282_);
lean_dec(v_a_2290_);
if (lean_obj_tag(v___x_2293_) == 0)
{
lean_object* v_ref_2294_; lean_object* v___x_2295_; lean_object* v_calcstx_2296_; uint8_t v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; 
lean_dec_ref_known(v___x_2293_, 1);
v_ref_2294_ = lean_ctor_get(v_a_2281_, 5);
v___x_2295_ = lean_unsigned_to_nat(0u);
v_calcstx_2296_ = l_Lean_Syntax_getArg(v_x_2274_, v___x_2295_);
lean_dec(v_x_2274_);
v___x_2297_ = 0;
v___x_2298_ = l_Lean_SourceInfo_fromRef(v_ref_2294_, v___x_2297_);
v___x_2299_ = l_Lean_SourceInfo_fromRef(v_calcstx_2296_, v_isFirst_2285_);
lean_dec(v_calcstx_2296_);
v___x_2300_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__Elab__Tactic__tacticCalc_x3f__1___lam__0___closed__3));
v___x_2301_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2301_, 0, v___x_2299_);
lean_ctor_set(v___x_2301_, 1, v___x_2300_);
v___x_2302_ = l_Lean_Syntax_node2(v___x_2298_, v___x_2284_, v___x_2301_, v_steps_2288_);
v___x_2303_ = l_Lean_Elab_Tactic_evalCalc(v___x_2302_, v_a_2275_, v_a_2276_, v_a_2277_, v_a_2278_, v_a_2279_, v_a_2280_, v_a_2281_, v_a_2282_);
return v___x_2303_;
}
else
{
lean_object* v_a_2304_; lean_object* v___x_2306_; uint8_t v_isShared_2307_; uint8_t v_isSharedCheck_2311_; 
lean_dec(v_steps_2288_);
lean_dec(v_x_2274_);
v_a_2304_ = lean_ctor_get(v___x_2293_, 0);
v_isSharedCheck_2311_ = !lean_is_exclusive(v___x_2293_);
if (v_isSharedCheck_2311_ == 0)
{
v___x_2306_ = v___x_2293_;
v_isShared_2307_ = v_isSharedCheck_2311_;
goto v_resetjp_2305_;
}
else
{
lean_inc(v_a_2304_);
lean_dec(v___x_2293_);
v___x_2306_ = lean_box(0);
v_isShared_2307_ = v_isSharedCheck_2311_;
goto v_resetjp_2305_;
}
v_resetjp_2305_:
{
lean_object* v___x_2309_; 
if (v_isShared_2307_ == 0)
{
v___x_2309_ = v___x_2306_;
goto v_reusejp_2308_;
}
else
{
lean_object* v_reuseFailAlloc_2310_; 
v_reuseFailAlloc_2310_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2310_, 0, v_a_2304_);
v___x_2309_ = v_reuseFailAlloc_2310_;
goto v_reusejp_2308_;
}
v_reusejp_2308_:
{
return v___x_2309_;
}
}
}
}
else
{
lean_object* v_a_2312_; lean_object* v___x_2314_; uint8_t v_isShared_2315_; uint8_t v_isSharedCheck_2319_; 
lean_dec(v_steps_2288_);
lean_dec(v_x_2274_);
v_a_2312_ = lean_ctor_get(v___x_2289_, 0);
v_isSharedCheck_2319_ = !lean_is_exclusive(v___x_2289_);
if (v_isSharedCheck_2319_ == 0)
{
v___x_2314_ = v___x_2289_;
v_isShared_2315_ = v_isSharedCheck_2319_;
goto v_resetjp_2313_;
}
else
{
lean_inc(v_a_2312_);
lean_dec(v___x_2289_);
v___x_2314_ = lean_box(0);
v_isShared_2315_ = v_isSharedCheck_2319_;
goto v_resetjp_2313_;
}
v_resetjp_2313_:
{
lean_object* v___x_2317_; 
if (v_isShared_2315_ == 0)
{
v___x_2317_ = v___x_2314_;
goto v_reusejp_2316_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v_a_2312_);
v___x_2317_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2316_;
}
v_reusejp_2316_:
{
return v___x_2317_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1___boxed(lean_object* v_x_2320_, lean_object* v_a_2321_, lean_object* v_a_2322_, lean_object* v_a_2323_, lean_object* v_a_2324_, lean_object* v_a_2325_, lean_object* v_a_2326_, lean_object* v_a_2327_, lean_object* v_a_2328_, lean_object* v_a_2329_){
_start:
{
lean_object* v_res_2330_; 
v_res_2330_ = lp_mathlib_Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1(v_x_2320_, v_a_2321_, v_a_2322_, v_a_2323_, v_a_2324_, v_a_2325_, v_a_2326_, v_a_2327_, v_a_2328_);
lean_dec(v_a_2328_);
lean_dec_ref(v_a_2327_);
lean_dec(v_a_2326_);
lean_dec_ref(v_a_2325_);
lean_dec(v_a_2324_);
lean_dec_ref(v_a_2323_);
lean_dec(v_a_2322_);
lean_dec_ref(v_a_2321_);
return v_res_2330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0(lean_object* v_as_2331_, size_t v_sz_2332_, size_t v_i_2333_, uint8_t v_b_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_){
_start:
{
lean_object* v___x_2344_; 
v___x_2344_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___redArg(v_as_2331_, v_sz_2332_, v_i_2333_, v_b_2334_, v___y_2341_, v___y_2342_);
return v___x_2344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0___boxed(lean_object* v_as_2345_, lean_object* v_sz_2346_, lean_object* v_i_2347_, lean_object* v_b_2348_, lean_object* v___y_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_){
_start:
{
size_t v_sz_boxed_2358_; size_t v_i_boxed_2359_; uint8_t v_b_boxed_2360_; lean_object* v_res_2361_; 
v_sz_boxed_2358_ = lean_unbox_usize(v_sz_2346_);
lean_dec(v_sz_2346_);
v_i_boxed_2359_ = lean_unbox_usize(v_i_2347_);
lean_dec(v_i_2347_);
v_b_boxed_2360_ = lean_unbox(v_b_2348_);
v_res_2361_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_Tactic___aux__Mathlib__Tactic__Widget__Calc______elabRules__Lean__calcTactic__1_spec__0(v_as_2345_, v_sz_boxed_2358_, v_i_boxed_2359_, v_b_boxed_2360_, v___y_2349_, v___y_2350_, v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_, v___y_2356_);
lean_dec(v___y_2356_);
lean_dec_ref(v___y_2355_);
lean_dec(v___y_2354_);
lean_dec_ref(v___y_2353_);
lean_dec(v___y_2352_);
lean_dec_ref(v___y_2351_);
lean_dec(v___y_2350_);
lean_dec_ref(v___y_2349_);
lean_dec_ref(v_as_2345_);
return v_res_2361_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_Calc(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Calc(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Widget_Calc(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_String_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CalcPanel_rpc___rpc__wrapped = _init_lp_mathlib_CalcPanel_rpc___rpc__wrapped();
lean_mark_persistent(lp_mathlib_CalcPanel_rpc___rpc__wrapped);
lp_mathlib_CalcPanel = _init_lp_mathlib_CalcPanel();
lean_mark_persistent(lp_mathlib_CalcPanel);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Calc(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
lean_object* initialize_batteries_Batteries_CodeAction_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Widget_Calc(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_String_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_CodeAction_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Widget_Calc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Widget_Calc(builtin);
}
#ifdef __cplusplus
}
#endif
