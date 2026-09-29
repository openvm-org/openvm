// Lean compiler output
// Module: Mathlib.Tactic.Widget.Conv
// Imports: public import Init public meta import Init public import Mathlib.Lean.Name public import Mathlib.Tactic.Widget.SelectPanelUtils public import ProofWidgets.Component.OfRpcMethod public import ProofWidgets.Component.Basic public meta import Lean.PrettyPrinter.Delaborator.Builtins public meta import ProofWidgets.Component.Basic
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
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
extern lean_object* l_Lean_SubExpr_Pos_typeCoord;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Fin_succ___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_getParamKinds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLocalDeclNoLocalInstanceUpdate___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isArrow(lean_object*);
lean_object* l_Lean_Meta_isTypeCorrect(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_TSepArray_push___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_mkNumLit(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdentFrom(lean_object*, lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
uint8_t lp_mathlib_Lean_Name_willRoundTrip(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_mkIdentFromRef___redArg(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint64_t lean_string_hash(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink;
lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_sanitizeNames(lean_object*, lean_object*);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_lspRangeOfStx_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Lsp_instToJsonRange_toJson(lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_Widget_savePanelWidgetInfo(uint64_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_arg_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_arg_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_fun_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_fun_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_type_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_type_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_body_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_body_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__4(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " position "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Pos"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "SubExpr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__1_value),LEAN_SCALAR_PTR_LITERAL(170, 131, 175, 90, 105, 49, 153, 209)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__2_value),LEAN_SCALAR_PTR_LITERAL(235, 172, 159, 248, 236, 182, 173, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " is invalid for"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "unexpected bound variable #"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "conv mode does not support rewriting the binder type of a lambda"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 174, .m_capacity = 174, .m_length = 173, .m_data = "conv mode only supports rewriting forall binder types when the binder type is a proposition or when the body of the forall does not depend on the value of the bound variable"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 183, .m_capacity = 183, .m_length = 182, .m_data = "conv mode does not support entering let expressions for which the type-correctness of the body depends on the let value \nfailed to abstract let-expression, result is not type correct"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "conv mode does not yet support entering let values"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "conv mode does not yet support entering let types"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "conv mode does not yet support entering projections"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "conv mode does not support entering types of expressions"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__13;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ofSubExprPos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ofSubExprPos___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3_value),LEAN_SCALAR_PTR_LITERAL(192, 39, 103, 162, 58, 5, 181, 114)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__10_value),LEAN_SCALAR_PTR_LITERAL(202, 81, 30, 13, 252, 23, 29, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "enterArg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 39, 81, 184, 62, 123, 191, 109)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "argArg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__2_value),LEAN_SCALAR_PTR_LITERAL(59, 211, 157, 2, 56, 142, 56, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__4_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__3_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__32(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "enter"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(55, 212, 211, 21, 88, 173, 115, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__36(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__36___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__35(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__34(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__23(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__24(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "convSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 35, 202, 76, 198, 168, 114, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__30(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__30___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__28(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__25(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__33(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__33___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(177, 22, 157, 83, 164, 254, 43, 206)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__31(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__7(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__2(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "skip"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 180, 41, 36, 18, 201, 24, 192)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "You must select something."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "You must select something in the goal or in the type of a local hypothesis."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_Conv_insertEnter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_Conv_insertEnter_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Generate conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__0 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__1 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__2 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__3 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__4 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__4_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__4_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__5 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__5_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__5_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__6 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__6_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__6_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__7 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__8 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ml1"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__9 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__9_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__9_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__10 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__10_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__11 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__11_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__11_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__12 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__12_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "There is no goal to solve!"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__13 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__13_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__13_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__14 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__14_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__14_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__15 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__15_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__15_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__16 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__16_value;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__17;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__18;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__19;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__20;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__21;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " should be "};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__22 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__22_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "You should select only one sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__23 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__23_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__23_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__24 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__24_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__24_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__25 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__25_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__25_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__26 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__26_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "in the main goal or its context."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__27 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__27_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "in the main goal."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__28 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__28_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "All selected sub-expressions"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__29 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__29_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "The selected sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__30 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__30_value;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Conv_insertEnter___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 103, .m_capacity = 103, .m_length = 102, .m_data = "Use shift-click to select one sub-expression in the goal or local context that you want to zoom in on."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 7, .m_data = "Conv 🔍️"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SelectionPanel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rpc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(241, 229, 111, 7, 181, 47, 166, 117)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__2_value),LEAN_SCALAR_PTR_LITERAL(229, 151, 233, 64, 85, 185, 112, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3855, .m_capacity = 3855, .m_length = 3854, .m_data = "window;import{jsxs as e,jsx as t,Fragment as r}from\"react/jsx-runtime\";import*as n from\"react\";import{useRpcSession as o,EnvPosContext as a,useAsyncPersistent as i,mapRpcError as f,importWidgetModule as c}from\"@leanprover/infoview\";function u(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,\"default\")\?e.default:e}var s,l;var p=u(function(){if(l)return s;l=1;var e=\"undefined\"!=typeof Element,t=\"function\"==typeof Map,r=\"function\"==typeof Set,n=\"function\"==typeof ArrayBuffer&&!!ArrayBuffer.isView;function o(a,i){if(a===i)return!0;if(a&&i&&\"object\"==typeof a&&\"object\"==typeof i){if(a.constructor!==i.constructor)return!1;var f,c,u,s;if(Array.isArray(a)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(!o(a[c],i[c]))return!1;return!0}if(t&&a instanceof Map&&i instanceof Map){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;for(s=a.entries();!(c=s.next()).done;)if(!o(c.value[1],i.get(c.value[0])))return!1;return!0}if(r&&a instanceof Set&&i instanceof Set){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;return!0}if(n&&ArrayBuffer.isView(a)&&ArrayBuffer.isView(i)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(a[c]!==i[c])return!1;return!0}if(a.constructor===RegExp)return a.source===i.source&&a.flags===i.flags;if(a.valueOf!==Object.prototype.valueOf&&\"function\"==typeof a.valueOf&&\"function\"==typeof i.valueOf)return a.valueOf()===i.valueOf();if(a.toString!==Object.prototype.toString&&\"function\"==typeof a.toString&&\"function\"==typeof i.toString)return a.toString()===i.toString();if((f=(u=Object.keys(a)).length)!==Object.keys(i).length)return!1;for(c=f;0!==c--;)if(!Object.prototype.hasOwnProperty.call(i,u[c]))return!1;if(e&&a instanceof Element)return!1;for(c=f;0!==c--;)if((\"_owner\"!==u[c]&&\"__v\"!==u[c]&&\"__o\"!==u[c]||!a.$$typeof)&&!o(a[u[c]],i[u[c]]))return!1;return!0}return a!=a&&i!=i}return s=function(e,t){try{return o(e,t)}catch(e){if((e.message||\"\").match(/stack|recursion/i))return console.warn(\"react-fast-compare cannot handle circular refs\"),!1;throw e}}}());async function y(o,a,i){if(\"text\"in i)return t(r,{children:i.text});if(\"element\"in i){const[e,r,f]=i.element,c={};for(const[e,t]of r)c[e]=t;const u=await Promise.all(f.map(async e=>await y(o,a,e)));return\"hr\"===e\?t(\"hr\",{}):0===u.length\?n.createElement(e,c):n.createElement(e,c,u)}if(\"component\"in i){const[e,t,r,f]=i.component,u=await Promise.all(f.map(async e=>await y(o,a,e))),s={...r,pos:a},l=await c(o,a,e);if(!(t in l))throw new Error(`Module '${e}' does not export '${t}'`);return 0===u.length\?n.createElement(l[t],s):n.createElement(l[t],s,u)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(i)]})}function d({html:c}){const u=o(),s=n.useContext(a),l=i(()=>y(u,s,c),[u,s,c]);return\"resolved\"===l.state\?l.value:\"rejected\"===l.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",f(l.error).message]}):t(r,{})}const m=\"Mathlib.Tactic.Conv.SelectionPanel.rpc\",g='false';var w=n.memo(e=>{const a=o(),c=n.useRef({fn:()=>{}}),u=i(async()=>{if(c.current.fn(),\"true\"===g){const[t,r]=function(e,t,r){const n={fn:()=>{}};return[new Promise(async(o,a)=>{const i=await e.call(t,r),f=window.setInterval(async()=>{try{const t=await e.call(\"ProofWidgets.checkRequest\",i);if(\"running\"===t)return;window.clearInterval(f),o(t.done.result)}catch(e){window.clearInterval(f),a(e)}},100);n.fn=()=>{e.call(\"ProofWidgets.cancelRequest\",i)}}),n]}(a,m,e);return c.current=r,t}{const t=new AbortController,r=a.call(m,e,{abortSignal:t.signal});return c.current={fn:()=>t.abort()},r}},[a,e]);return n.useEffect(()=>()=>{c.current.fn()},[]),\"rejected\"===u.state\?t(\"p\",{style:{color:\"red\"},children:f(u.error).message}):\"loading\"===u.state\?t(r,{children:\"Loading..\"}):t(d,{html:u.value})},p);export{w as default};"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tacticConv\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(222, 39, 209, 71, 236, 134, 138, 73)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "conv\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___lam__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "replaceRange"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
default: 
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorIdx___boxed(lean_object* v_x_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorIdx(v_x_6_);
lean_dec_ref(v_x_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(lean_object* v_t_8_, lean_object* v_k_9_){
_start:
{
switch(lean_obj_tag(v_t_8_))
{
case 0:
{
lean_object* v_arg_10_; uint8_t v_all_11_; lean_object* v_next_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v_arg_10_ = lean_ctor_get(v_t_8_, 0);
lean_inc(v_arg_10_);
v_all_11_ = lean_ctor_get_uint8(v_t_8_, sizeof(void*)*2);
v_next_12_ = lean_ctor_get(v_t_8_, 1);
lean_inc_ref(v_next_12_);
lean_dec_ref_known(v_t_8_, 2);
v___x_13_ = lean_box(v_all_11_);
v___x_14_ = lean_apply_3(v_k_9_, v_arg_10_, v___x_13_, v_next_12_);
return v___x_14_;
}
case 1:
{
lean_object* v_depth_15_; lean_object* v___x_16_; 
v_depth_15_ = lean_ctor_get(v_t_8_, 0);
lean_inc(v_depth_15_);
lean_dec_ref_known(v_t_8_, 1);
v___x_16_ = lean_apply_1(v_k_9_, v_depth_15_);
return v___x_16_;
}
case 2:
{
lean_object* v_next_17_; lean_object* v___x_18_; 
v_next_17_ = lean_ctor_get(v_t_8_, 0);
lean_inc_ref(v_next_17_);
lean_dec_ref_known(v_t_8_, 1);
v___x_18_ = lean_apply_1(v_k_9_, v_next_17_);
return v___x_18_;
}
default: 
{
lean_object* v_name_19_; lean_object* v_next_20_; lean_object* v___x_21_; 
v_name_19_ = lean_ctor_get(v_t_8_, 0);
lean_inc(v_name_19_);
v_next_20_ = lean_ctor_get(v_t_8_, 1);
lean_inc_ref(v_next_20_);
lean_dec_ref_known(v_t_8_, 2);
v___x_21_ = lean_apply_2(v_k_9_, v_name_19_, v_next_20_);
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim(lean_object* v_motive_22_, lean_object* v_ctorIdx_23_, lean_object* v_t_24_, lean_object* v_h_25_, lean_object* v_k_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_24_, v_k_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___boxed(lean_object* v_motive_28_, lean_object* v_ctorIdx_29_, lean_object* v_t_30_, lean_object* v_h_31_, lean_object* v_k_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim(v_motive_28_, v_ctorIdx_29_, v_t_30_, v_h_31_, v_k_32_);
lean_dec(v_ctorIdx_29_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_arg_elim___redArg(lean_object* v_t_34_, lean_object* v_arg_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_34_, v_arg_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_arg_elim(lean_object* v_motive_37_, lean_object* v_t_38_, lean_object* v_h_39_, lean_object* v_arg_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_38_, v_arg_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_fun_elim___redArg(lean_object* v_t_42_, lean_object* v_fun_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_42_, v_fun_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_fun_elim(lean_object* v_motive_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_fun_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_46_, v_fun_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_type_elim___redArg(lean_object* v_t_50_, lean_object* v_type_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_50_, v_type_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_type_elim(lean_object* v_motive_53_, lean_object* v_t_54_, lean_object* v_h_55_, lean_object* v_type_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_54_, v_type_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_body_elim___redArg(lean_object* v_t_58_, lean_object* v_body_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_58_, v_body_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_body_elim(lean_object* v_motive_61_, lean_object* v_t_62_, lean_object* v_h_63_, lean_object* v_body_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ctorElim___redArg(v_t_62_, v_body_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0_spec__0(lean_object* v_msgData_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
lean_object* v___x_72_; lean_object* v_env_73_; lean_object* v___x_74_; lean_object* v_mctx_75_; lean_object* v_lctx_76_; lean_object* v_options_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_72_ = lean_st_ref_get(v___y_70_);
v_env_73_ = lean_ctor_get(v___x_72_, 0);
lean_inc_ref(v_env_73_);
lean_dec(v___x_72_);
v___x_74_ = lean_st_ref_get(v___y_68_);
v_mctx_75_ = lean_ctor_get(v___x_74_, 0);
lean_inc_ref(v_mctx_75_);
lean_dec(v___x_74_);
v_lctx_76_ = lean_ctor_get(v___y_67_, 2);
v_options_77_ = lean_ctor_get(v___y_69_, 2);
lean_inc_ref(v_options_77_);
lean_inc_ref(v_lctx_76_);
v___x_78_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_78_, 0, v_env_73_);
lean_ctor_set(v___x_78_, 1, v_mctx_75_);
lean_ctor_set(v___x_78_, 2, v_lctx_76_);
lean_ctor_set(v___x_78_, 3, v_options_77_);
v___x_79_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_msgData_66_);
v___x_80_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0_spec__0___boxed(lean_object* v_msgData_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0_spec__0(v_msgData_81_, v___y_82_, v___y_83_, v___y_84_, v___y_85_);
lean_dec(v___y_85_);
lean_dec_ref(v___y_84_);
lean_dec(v___y_83_);
lean_dec_ref(v___y_82_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(lean_object* v_msg_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_ref_94_; lean_object* v___x_95_; lean_object* v_a_96_; lean_object* v___x_98_; uint8_t v_isShared_99_; uint8_t v_isSharedCheck_104_; 
v_ref_94_ = lean_ctor_get(v___y_91_, 5);
v___x_95_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0_spec__0(v_msg_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
v_a_96_ = lean_ctor_get(v___x_95_, 0);
v_isSharedCheck_104_ = !lean_is_exclusive(v___x_95_);
if (v_isSharedCheck_104_ == 0)
{
v___x_98_ = v___x_95_;
v_isShared_99_ = v_isSharedCheck_104_;
goto v_resetjp_97_;
}
else
{
lean_inc(v_a_96_);
lean_dec(v___x_95_);
v___x_98_ = lean_box(0);
v_isShared_99_ = v_isSharedCheck_104_;
goto v_resetjp_97_;
}
v_resetjp_97_:
{
lean_object* v___x_100_; lean_object* v___x_102_; 
lean_inc(v_ref_94_);
v___x_100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_100_, 0, v_ref_94_);
lean_ctor_set(v___x_100_, 1, v_a_96_);
if (v_isShared_99_ == 0)
{
lean_ctor_set_tag(v___x_98_, 1);
lean_ctor_set(v___x_98_, 0, v___x_100_);
v___x_102_ = v___x_98_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v___x_100_);
v___x_102_ = v_reuseFailAlloc_103_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
return v___x_102_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg___boxed(lean_object* v_msg_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v_msg_105_, v___y_106_, v___y_107_, v___y_108_, v___y_109_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
lean_dec(v___y_107_);
lean_dec_ref(v___y_106_);
return v_res_111_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__3(lean_object* v_x_112_, lean_object* v_x_113_){
_start:
{
if (lean_obj_tag(v_x_112_) == 0)
{
if (lean_obj_tag(v_x_113_) == 0)
{
uint8_t v___x_114_; 
v___x_114_ = 1;
return v___x_114_;
}
else
{
uint8_t v___x_115_; 
v___x_115_ = 0;
return v___x_115_;
}
}
else
{
if (lean_obj_tag(v_x_113_) == 0)
{
uint8_t v___x_116_; 
v___x_116_ = 0;
return v___x_116_;
}
else
{
lean_object* v_val_117_; lean_object* v_val_118_; uint8_t v___x_119_; uint8_t v___x_120_; uint8_t v___x_121_; 
v_val_117_ = lean_ctor_get(v_x_112_, 0);
v_val_118_ = lean_ctor_get(v_x_113_, 0);
v___x_119_ = lean_unbox(v_val_117_);
v___x_120_ = lean_unbox(v_val_118_);
v___x_121_ = l_Lean_instBEqBinderInfo_beq(v___x_119_, v___x_120_);
return v___x_121_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__3___boxed(lean_object* v_x_122_, lean_object* v_x_123_){
_start:
{
uint8_t v_res_124_; lean_object* v_r_125_; 
v_res_124_ = lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__3(v_x_122_, v_x_123_);
lean_dec(v_x_123_);
lean_dec(v_x_122_);
v_r_125_ = lean_box(v_res_124_);
return v_r_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__4(lean_object* v_as_126_, size_t v_i_127_, size_t v_stop_128_, lean_object* v_b_129_){
_start:
{
uint8_t v___x_130_; 
v___x_130_ = lean_usize_dec_eq(v_i_127_, v_stop_128_);
if (v___x_130_ == 0)
{
uint8_t v___x_131_; size_t v___x_132_; size_t v___x_133_; lean_object* v___x_134_; uint8_t v___x_135_; uint8_t v___x_136_; 
v___x_131_ = 0;
v___x_132_ = ((size_t)1ULL);
v___x_133_ = lean_usize_sub(v_i_127_, v___x_132_);
v___x_134_ = lean_array_uget_borrowed(v_as_126_, v___x_133_);
v___x_135_ = lean_unbox(v___x_134_);
v___x_136_ = l_Lean_instBEqBinderInfo_beq(v___x_135_, v___x_131_);
if (v___x_136_ == 0)
{
v_i_127_ = v___x_133_;
goto _start;
}
else
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = lean_unsigned_to_nat(1u);
v___x_139_ = lean_nat_add(v_b_129_, v___x_138_);
lean_dec(v_b_129_);
v_i_127_ = v___x_133_;
v_b_129_ = v___x_139_;
goto _start;
}
}
else
{
return v_b_129_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__4___boxed(lean_object* v_as_141_, lean_object* v_i_142_, lean_object* v_stop_143_, lean_object* v_b_144_){
_start:
{
size_t v_i_boxed_145_; size_t v_stop_boxed_146_; lean_object* v_res_147_; 
v_i_boxed_145_ = lean_unbox_usize(v_i_142_);
lean_dec(v_i_142_);
v_stop_boxed_146_ = lean_unbox_usize(v_stop_143_);
lean_dec(v_stop_143_);
v_res_147_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__4(v_as_141_, v_i_boxed_145_, v_stop_boxed_146_, v_b_144_);
lean_dec_ref(v_as_141_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__2(size_t v_sz_148_, size_t v_i_149_, lean_object* v_bs_150_){
_start:
{
uint8_t v___x_151_; 
v___x_151_ = lean_usize_dec_lt(v_i_149_, v_sz_148_);
if (v___x_151_ == 0)
{
return v_bs_150_;
}
else
{
lean_object* v_v_152_; uint8_t v_bInfo_153_; lean_object* v___x_154_; lean_object* v_bs_x27_155_; size_t v___x_156_; size_t v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v_v_152_ = lean_array_uget_borrowed(v_bs_150_, v_i_149_);
v_bInfo_153_ = lean_ctor_get_uint8(v_v_152_, sizeof(void*)*2);
v___x_154_ = lean_unsigned_to_nat(0u);
v_bs_x27_155_ = lean_array_uset(v_bs_150_, v_i_149_, v___x_154_);
v___x_156_ = ((size_t)1ULL);
v___x_157_ = lean_usize_add(v_i_149_, v___x_156_);
v___x_158_ = lean_box(v_bInfo_153_);
v___x_159_ = lean_array_uset(v_bs_x27_155_, v_i_149_, v___x_158_);
v_i_149_ = v___x_157_;
v_bs_150_ = v___x_159_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__2___boxed(lean_object* v_sz_161_, lean_object* v_i_162_, lean_object* v_bs_163_){
_start:
{
size_t v_sz_boxed_164_; size_t v_i_boxed_165_; lean_object* v_res_166_; 
v_sz_boxed_164_ = lean_unbox_usize(v_sz_161_);
lean_dec(v_sz_161_);
v_i_boxed_165_ = lean_unbox_usize(v_i_162_);
lean_dec(v_i_162_);
v_res_166_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__2(v_sz_boxed_164_, v_i_boxed_165_, v_bs_163_);
return v_res_166_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__6(void){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__5));
v___x_169_ = l_Lean_stringToMessageData(v___x_168_);
return v___x_169_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__4(void){
_start:
{
uint8_t v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_177_ = 1;
v___x_178_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__3));
v___x_179_ = l_Lean_MessageData_ofConstName(v___x_178_, v___x_177_);
return v___x_179_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_180_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__6, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__6);
v___x_181_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__4, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__4);
v___x_182_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v___x_180_);
return v___x_182_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9(void){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__8));
v___x_185_ = l_Lean_stringToMessageData(v___x_184_);
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__1(void){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_187_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__0));
v___x_188_ = l_Lean_stringToMessageData(v___x_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT(lean_object* v_pos_194_, lean_object* v_expr_195_, lean_object* v_i_196_, lean_object* v_acc_197_, lean_object* v_n_198_, lean_object* v_a_199_, lean_object* v_a_200_, lean_object* v_a_201_, lean_object* v_a_202_){
_start:
{
lean_object* v___y_205_; lean_object* v___y_206_; 
if (lean_obj_tag(v_expr_195_) == 5)
{
if (lean_obj_tag(v_n_198_) == 1)
{
lean_object* v_fn_212_; lean_object* v_arg_213_; lean_object* v_val_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_224_; 
v_fn_212_ = lean_ctor_get(v_expr_195_, 0);
lean_inc_ref(v_fn_212_);
v_arg_213_ = lean_ctor_get(v_expr_195_, 1);
lean_inc_ref(v_arg_213_);
lean_dec_ref_known(v_expr_195_, 2);
v_val_214_ = lean_ctor_get(v_n_198_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v_n_198_);
if (v_isSharedCheck_224_ == 0)
{
v___x_216_ = v_n_198_;
v_isShared_217_ = v_isSharedCheck_224_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_val_214_);
lean_dec(v_n_198_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_224_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_221_; 
v___x_218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_218_, 0, v_arg_213_);
lean_ctor_set(v___x_218_, 1, v_acc_197_);
v___x_219_ = l_Fin_succ___redArg(v_val_214_);
lean_dec(v_val_214_);
if (v_isShared_217_ == 0)
{
lean_ctor_set(v___x_216_, 0, v___x_219_);
v___x_221_ = v___x_216_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v___x_219_);
v___x_221_ = v_reuseFailAlloc_223_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
v_expr_195_ = v_fn_212_;
v_acc_197_ = v___x_218_;
v_n_198_ = v___x_221_;
goto _start;
}
}
}
else
{
lean_object* v_fn_225_; lean_object* v_arg_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
lean_dec(v_n_198_);
v_fn_225_ = lean_ctor_get(v_expr_195_, 0);
v_arg_226_ = lean_ctor_get(v_expr_195_, 1);
v___x_227_ = lean_array_get_size(v_pos_194_);
v___x_228_ = lean_nat_dec_eq(v_i_196_, v___x_227_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; lean_object* v___x_230_; uint8_t v___x_231_; 
v___x_229_ = lean_array_fget_borrowed(v_pos_194_, v_i_196_);
v___x_230_ = lean_unsigned_to_nat(0u);
v___x_231_ = lean_nat_dec_eq(v___x_229_, v___x_230_);
if (v___x_231_ == 0)
{
lean_object* v___x_232_; uint8_t v___x_233_; 
v___x_232_ = lean_unsigned_to_nat(1u);
v___x_233_ = lean_nat_dec_eq(v___x_229_, v___x_232_);
if (v___x_233_ == 0)
{
lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; 
lean_inc(v___x_229_);
lean_dec(v_acc_197_);
lean_dec(v_i_196_);
lean_dec_ref(v_pos_194_);
v___x_234_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7);
v___x_235_ = l_Nat_reprFast(v___x_229_);
v___x_236_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
v___x_237_ = l_Lean_MessageData_ofFormat(v___x_236_);
v___x_238_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_234_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
v___x_239_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9);
v___x_240_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_238_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = l_Lean_indentExpr(v_expr_195_);
v___x_242_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_242_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
return v___x_243_;
}
else
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
lean_inc_ref(v_arg_226_);
lean_inc_ref(v_fn_225_);
lean_dec_ref_known(v_expr_195_, 2);
v___x_244_ = l_Fin_succ___redArg(v_i_196_);
lean_dec(v_i_196_);
v___x_245_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_245_, 0, v_arg_226_);
lean_ctor_set(v___x_245_, 1, v_acc_197_);
v___x_246_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__10));
v_expr_195_ = v_fn_225_;
v_i_196_ = v___x_244_;
v_acc_197_ = v___x_245_;
v_n_198_ = v___x_246_;
goto _start;
}
}
else
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
lean_inc_ref(v_arg_226_);
lean_inc_ref(v_fn_225_);
lean_dec_ref_known(v_expr_195_, 2);
v___x_248_ = l_Fin_succ___redArg(v_i_196_);
lean_dec(v_i_196_);
v___x_249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_249_, 0, v_arg_226_);
lean_ctor_set(v___x_249_, 1, v_acc_197_);
v___x_250_ = lean_box(0);
v_expr_195_ = v_fn_225_;
v_i_196_ = v___x_248_;
v_acc_197_ = v___x_249_;
v_n_198_ = v___x_250_;
goto _start;
}
}
else
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; 
lean_dec_ref_known(v_expr_195_, 2);
lean_dec(v_i_196_);
lean_dec_ref(v_pos_194_);
v___x_252_ = l_List_lengthTR___redArg(v_acc_197_);
lean_dec(v_acc_197_);
v___x_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
v___x_254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
return v___x_254_;
}
}
}
else
{
if (lean_obj_tag(v_n_198_) == 1)
{
lean_object* v_val_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_307_; 
v_val_255_ = lean_ctor_get(v_n_198_, 0);
v_isSharedCheck_307_ = !lean_is_exclusive(v_n_198_);
if (v_isSharedCheck_307_ == 0)
{
v___x_257_ = v_n_198_;
v_isShared_258_ = v_isSharedCheck_307_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_val_255_);
lean_dec(v_n_198_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_307_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_259_; lean_object* v___x_260_; 
lean_inc(v_acc_197_);
v___x_259_ = lean_array_mk(v_acc_197_);
v___x_260_ = l_Lean_PrettyPrinter_Delaborator_getParamKinds(v_expr_195_, v___x_259_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
lean_dec_ref(v___x_259_);
if (lean_obj_tag(v___x_260_) == 0)
{
lean_object* v_a_261_; size_t v_sz_262_; size_t v___x_263_; lean_object* v___x_264_; lean_object* v___y_266_; lean_object* v___x_292_; uint8_t v___x_293_; 
v_a_261_ = lean_ctor_get(v___x_260_, 0);
lean_inc(v_a_261_);
lean_dec_ref_known(v___x_260_, 1);
v_sz_262_ = lean_array_size(v_a_261_);
v___x_263_ = ((size_t)0ULL);
v___x_264_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__2(v_sz_262_, v___x_263_, v_a_261_);
v___x_292_ = lean_array_get_size(v___x_264_);
v___x_293_ = lean_nat_dec_lt(v_val_255_, v___x_292_);
if (v___x_293_ == 0)
{
lean_object* v___x_294_; 
lean_del_object(v___x_257_);
v___x_294_ = lean_box(0);
v___y_266_ = v___x_294_;
goto v___jp_265_;
}
else
{
lean_object* v___x_295_; lean_object* v___x_297_; 
v___x_295_ = lean_array_fget(v___x_264_, v_val_255_);
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 0, v___x_295_);
v___x_297_ = v___x_257_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v___x_295_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
v___y_266_ = v___x_297_;
goto v___jp_265_;
}
}
v___jp_265_:
{
lean_object* v___x_267_; uint8_t v___x_268_; 
v___x_267_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__11));
v___x_268_ = lp_mathlib_Option_instBEq_beq___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__3(v___y_266_, v___x_267_);
lean_dec(v___y_266_);
if (v___x_268_ == 0)
{
lean_object* v___x_269_; lean_object* v___x_270_; 
lean_dec_ref(v___x_264_);
lean_inc(v_val_255_);
v___x_269_ = l_List_get___redArg(v_acc_197_, v_val_255_);
lean_dec(v_acc_197_);
v___x_270_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_194_, v___x_269_, v_i_196_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_282_; 
v_a_271_ = lean_ctor_get(v___x_270_, 0);
v_isSharedCheck_282_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_282_ == 0)
{
v___x_273_ = v___x_270_;
v_isShared_274_ = v_isSharedCheck_282_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_270_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_282_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
uint8_t v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_280_; 
v___x_275_ = 1;
v___x_276_ = lean_unsigned_to_nat(1u);
v___x_277_ = lean_nat_add(v_val_255_, v___x_276_);
lean_dec(v_val_255_);
v___x_278_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_278_, 0, v___x_277_);
lean_ctor_set(v___x_278_, 1, v_a_271_);
lean_ctor_set_uint8(v___x_278_, sizeof(void*)*2, v___x_275_);
if (v_isShared_274_ == 0)
{
lean_ctor_set(v___x_273_, 0, v___x_278_);
v___x_280_ = v___x_273_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v___x_278_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
return v___x_280_;
}
}
}
else
{
lean_dec(v_val_255_);
return v___x_270_;
}
}
else
{
lean_object* v___x_283_; lean_object* v___x_284_; 
lean_inc(v_val_255_);
v___x_283_ = l_List_get___redArg(v_acc_197_, v_val_255_);
lean_dec(v_acc_197_);
v___x_284_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_194_, v___x_283_, v_i_196_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; uint8_t v___x_289_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_284_, 1);
v___x_286_ = lean_unsigned_to_nat(0u);
v___x_287_ = l_Array_extract___redArg(v___x_264_, v___x_286_, v_val_255_);
lean_dec_ref(v___x_264_);
v___x_288_ = lean_array_get_size(v___x_287_);
v___x_289_ = lean_nat_dec_lt(v___x_286_, v___x_288_);
if (v___x_289_ == 0)
{
lean_dec_ref(v___x_287_);
v___y_205_ = v_a_285_;
v___y_206_ = v___x_286_;
goto v___jp_204_;
}
else
{
size_t v___x_290_; lean_object* v___x_291_; 
v___x_290_ = lean_usize_of_nat(v___x_288_);
v___x_291_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT_spec__4(v___x_287_, v___x_290_, v___x_263_, v___x_286_);
lean_dec_ref(v___x_287_);
v___y_205_ = v_a_285_;
v___y_206_ = v___x_291_;
goto v___jp_204_;
}
}
else
{
lean_dec_ref(v___x_264_);
lean_dec(v_val_255_);
return v___x_284_;
}
}
}
}
else
{
lean_object* v_a_299_; lean_object* v___x_301_; uint8_t v_isShared_302_; uint8_t v_isSharedCheck_306_; 
lean_del_object(v___x_257_);
lean_dec(v_val_255_);
lean_dec(v_acc_197_);
lean_dec(v_i_196_);
lean_dec_ref(v_pos_194_);
v_a_299_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_306_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_306_ == 0)
{
v___x_301_ = v___x_260_;
v_isShared_302_ = v_isSharedCheck_306_;
goto v_resetjp_300_;
}
else
{
lean_inc(v_a_299_);
lean_dec(v___x_260_);
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
else
{
lean_object* v___x_308_; 
lean_dec(v_n_198_);
lean_dec(v_acc_197_);
v___x_308_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_194_, v_expr_195_, v_i_196_, v_a_199_, v_a_200_, v_a_201_, v_a_202_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v_a_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_319_; 
v_a_309_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_319_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_319_ == 0)
{
v___x_311_ = v___x_308_;
v_isShared_312_ = v_isSharedCheck_319_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_a_309_);
lean_dec(v___x_308_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_319_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_313_; uint8_t v___x_314_; lean_object* v___x_315_; lean_object* v___x_317_; 
v___x_313_ = lean_unsigned_to_nat(0u);
v___x_314_ = 0;
v___x_315_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_315_, 0, v___x_313_);
lean_ctor_set(v___x_315_, 1, v_a_309_);
lean_ctor_set_uint8(v___x_315_, sizeof(void*)*2, v___x_314_);
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 0, v___x_315_);
v___x_317_ = v___x_311_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v___x_315_);
v___x_317_ = v_reuseFailAlloc_318_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
return v___x_317_;
}
}
}
else
{
return v___x_308_;
}
}
}
v___jp_204_:
{
lean_object* v___x_207_; lean_object* v___x_208_; uint8_t v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_207_ = lean_unsigned_to_nat(1u);
v___x_208_ = lean_nat_add(v___y_206_, v___x_207_);
lean_dec(v___y_206_);
v___x_209_ = 0;
v___x_210_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_210_, 0, v___x_208_);
lean_ctor_set(v___x_210_, 1, v___y_205_);
lean_ctor_set_uint8(v___x_210_, sizeof(void*)*2, v___x_209_);
v___x_211_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
return v___x_211_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0___boxed(lean_object* v_body_320_, lean_object* v_i_321_, lean_object* v_pos_322_, lean_object* v_binderName_323_, lean_object* v_fvar_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0(v_body_320_, v_i_321_, v_pos_322_, v_binderName_323_, v_fvar_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec_ref(v_fvar_324_);
lean_dec(v_i_321_);
lean_dec_ref(v_body_320_);
return v_res_330_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__3(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__2));
v___x_333_ = l_Lean_stringToMessageData(v___x_332_);
return v___x_333_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__5(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__4));
v___x_336_ = l_Lean_stringToMessageData(v___x_335_);
return v___x_336_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__1(void){
_start:
{
lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_338_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__0));
v___x_339_ = l_Lean_stringToMessageData(v___x_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2(lean_object* v_body_340_, lean_object* v_i_341_, lean_object* v_pos_342_, lean_object* v_declName_343_, lean_object* v___x_344_, lean_object* v_fvar_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v_e_351_; lean_object* v___y_353_; lean_object* v___y_354_; lean_object* v___y_355_; lean_object* v___y_356_; lean_object* v___x_368_; 
v_e_351_ = lean_expr_instantiate1(v_body_340_, v_fvar_345_);
lean_inc_ref(v_e_351_);
v___x_368_ = l_Lean_Meta_isTypeCorrect(v_e_351_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_368_) == 0)
{
lean_object* v_a_369_; uint8_t v___x_370_; 
v_a_369_ = lean_ctor_get(v___x_368_, 0);
lean_inc(v_a_369_);
lean_dec_ref_known(v___x_368_, 1);
v___x_370_ = lean_unbox(v_a_369_);
lean_dec(v_a_369_);
if (v___x_370_ == 0)
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_371_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___closed__1);
v___x_372_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_344_);
v___x_373_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_372_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_373_) == 0)
{
lean_dec_ref_known(v___x_373_, 1);
v___y_353_ = v___y_346_;
v___y_354_ = v___y_347_;
v___y_355_ = v___y_348_;
v___y_356_ = v___y_349_;
goto v___jp_352_;
}
else
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_381_; 
lean_dec_ref(v_e_351_);
lean_dec(v_declName_343_);
lean_dec_ref(v_pos_342_);
v_a_374_ = lean_ctor_get(v___x_373_, 0);
v_isSharedCheck_381_ = !lean_is_exclusive(v___x_373_);
if (v_isSharedCheck_381_ == 0)
{
v___x_376_ = v___x_373_;
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_373_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_381_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_379_; 
if (v_isShared_377_ == 0)
{
v___x_379_ = v___x_376_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v_a_374_);
v___x_379_ = v_reuseFailAlloc_380_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
return v___x_379_;
}
}
}
}
else
{
lean_dec_ref(v___x_344_);
v___y_353_ = v___y_346_;
v___y_354_ = v___y_347_;
v___y_355_ = v___y_348_;
v___y_356_ = v___y_349_;
goto v___jp_352_;
}
}
else
{
lean_object* v_a_382_; lean_object* v___x_384_; uint8_t v_isShared_385_; uint8_t v_isSharedCheck_389_; 
lean_dec_ref(v_e_351_);
lean_dec_ref(v___x_344_);
lean_dec(v_declName_343_);
lean_dec_ref(v_pos_342_);
v_a_382_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_389_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_389_ == 0)
{
v___x_384_ = v___x_368_;
v_isShared_385_ = v_isSharedCheck_389_;
goto v_resetjp_383_;
}
else
{
lean_inc(v_a_382_);
lean_dec(v___x_368_);
v___x_384_ = lean_box(0);
v_isShared_385_ = v_isSharedCheck_389_;
goto v_resetjp_383_;
}
v_resetjp_383_:
{
lean_object* v___x_387_; 
if (v_isShared_385_ == 0)
{
v___x_387_ = v___x_384_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v_a_382_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
v___jp_352_:
{
lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_357_ = l_Fin_succ___redArg(v_i_341_);
v___x_358_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_342_, v_e_351_, v___x_357_, v___y_353_, v___y_354_, v___y_355_, v___y_356_);
if (lean_obj_tag(v___x_358_) == 0)
{
lean_object* v_a_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_367_; 
v_a_359_ = lean_ctor_get(v___x_358_, 0);
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_358_);
if (v_isSharedCheck_367_ == 0)
{
v___x_361_ = v___x_358_;
v_isShared_362_ = v_isSharedCheck_367_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_a_359_);
lean_dec(v___x_358_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_367_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v___x_363_; lean_object* v___x_365_; 
v___x_363_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_363_, 0, v_declName_343_);
lean_ctor_set(v___x_363_, 1, v_a_359_);
if (v_isShared_362_ == 0)
{
lean_ctor_set(v___x_361_, 0, v___x_363_);
v___x_365_ = v___x_361_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v___x_363_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
}
else
{
lean_dec(v_declName_343_);
return v___x_358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___boxed(lean_object* v_body_390_, lean_object* v_i_391_, lean_object* v_pos_392_, lean_object* v_declName_393_, lean_object* v___x_394_, lean_object* v_fvar_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2(v_body_390_, v_i_391_, v_pos_392_, v_declName_393_, v___x_394_, v_fvar_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
lean_dec(v___y_397_);
lean_dec_ref(v___y_396_);
lean_dec_ref(v_fvar_395_);
lean_dec(v_i_391_);
lean_dec_ref(v_body_390_);
return v_res_401_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__7(void){
_start:
{
lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_403_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__6));
v___x_404_ = l_Lean_stringToMessageData(v___x_403_);
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__9(void){
_start:
{
lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_406_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__8));
v___x_407_ = l_Lean_stringToMessageData(v___x_406_);
return v___x_407_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__11(void){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_409_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__10));
v___x_410_ = l_Lean_stringToMessageData(v___x_409_);
return v___x_410_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__13(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__12));
v___x_413_ = l_Lean_stringToMessageData(v___x_412_);
return v___x_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(lean_object* v_pos_416_, lean_object* v_expr_417_, lean_object* v_i_418_, lean_object* v_a_419_, lean_object* v_a_420_, lean_object* v_a_421_, lean_object* v_a_422_){
_start:
{
lean_object* v___x_424_; uint8_t v___x_425_; 
v___x_424_ = lean_array_get_size(v_pos_416_);
v___x_425_ = lean_nat_dec_eq(v_i_418_, v___x_424_);
if (v___x_425_ == 0)
{
lean_object* v___x_426_; lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_426_ = lean_array_fget_borrowed(v_pos_416_, v_i_418_);
v___x_427_ = l_Lean_SubExpr_Pos_typeCoord;
v___x_428_ = lean_nat_dec_eq(v___x_426_, v___x_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v_err_438_; 
v___x_429_ = lean_unsigned_to_nat(1u);
v___x_430_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__7);
lean_inc(v___x_426_);
v___x_431_ = l_Nat_reprFast(v___x_426_);
v___x_432_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
v___x_433_ = l_Lean_MessageData_ofFormat(v___x_432_);
v___x_434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_434_, 0, v___x_430_);
lean_ctor_set(v___x_434_, 1, v___x_433_);
v___x_435_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__9);
v___x_436_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_434_);
lean_ctor_set(v___x_436_, 1, v___x_435_);
lean_inc_ref(v_expr_417_);
v___x_437_ = l_Lean_indentExpr(v_expr_417_);
lean_inc_ref(v___x_437_);
v_err_438_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_err_438_, 0, v___x_436_);
lean_ctor_set(v_err_438_, 1, v___x_437_);
switch(lean_obj_tag(v_expr_417_))
{
case 0:
{
lean_object* v_deBruijnIndex_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; 
lean_dec_ref_known(v_err_438_, 2);
lean_dec_ref(v___x_437_);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v_deBruijnIndex_439_ = lean_ctor_get(v_expr_417_, 0);
lean_inc(v_deBruijnIndex_439_);
lean_dec_ref_known(v_expr_417_, 1);
v___x_440_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__1, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__1);
v___x_441_ = l_Nat_reprFast(v_deBruijnIndex_439_);
v___x_442_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
v___x_443_ = l_Lean_MessageData_ofFormat(v___x_442_);
v___x_444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_440_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
v___x_445_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_444_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_445_;
}
case 5:
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
lean_dec_ref_known(v_err_438_, 2);
lean_dec_ref(v___x_437_);
v___x_446_ = lean_box(0);
v___x_447_ = lean_box(0);
v___x_448_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT(v_pos_416_, v_expr_417_, v_i_418_, v___x_446_, v___x_447_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_448_;
}
case 6:
{
lean_object* v_binderName_449_; lean_object* v_binderType_450_; lean_object* v_body_451_; uint8_t v_binderInfo_452_; lean_object* v___x_453_; uint8_t v___x_454_; 
v_binderName_449_ = lean_ctor_get(v_expr_417_, 0);
lean_inc(v_binderName_449_);
v_binderType_450_ = lean_ctor_get(v_expr_417_, 1);
lean_inc_ref(v_binderType_450_);
v_body_451_ = lean_ctor_get(v_expr_417_, 2);
lean_inc_ref(v_body_451_);
v_binderInfo_452_ = lean_ctor_get_uint8(v_expr_417_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_expr_417_, 3);
v___x_453_ = lean_unsigned_to_nat(0u);
v___x_454_ = lean_nat_dec_eq(v___x_426_, v___x_453_);
if (v___x_454_ == 0)
{
uint8_t v___x_455_; 
lean_dec_ref(v___x_437_);
v___x_455_ = lean_nat_dec_eq(v___x_426_, v___x_429_);
if (v___x_455_ == 0)
{
lean_object* v___x_456_; 
lean_dec_ref(v_body_451_);
lean_dec_ref(v_binderType_450_);
lean_dec(v_binderName_449_);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_456_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v_err_438_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_456_;
}
else
{
lean_object* v___f_457_; lean_object* v___x_458_; 
lean_dec_ref_known(v_err_438_, 2);
lean_inc(v_binderName_449_);
v___f_457_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0___boxed), 10, 4);
lean_closure_set(v___f_457_, 0, v_body_451_);
lean_closure_set(v___f_457_, 1, v_i_418_);
lean_closure_set(v___f_457_, 2, v_pos_416_);
lean_closure_set(v___f_457_, 3, v_binderName_449_);
v___x_458_ = l_Lean_Meta_withLocalDeclNoLocalInstanceUpdate___redArg(v_binderName_449_, v_binderInfo_452_, v_binderType_450_, v___f_457_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_458_;
}
}
else
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
lean_dec_ref(v_body_451_);
lean_dec_ref(v_binderType_450_);
lean_dec(v_binderName_449_);
lean_dec_ref_known(v_err_438_, 2);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_459_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__3, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__3);
v___x_460_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
lean_ctor_set(v___x_460_, 1, v___x_437_);
v___x_461_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_460_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_461_;
}
}
case 7:
{
lean_object* v_binderName_462_; lean_object* v_binderType_463_; lean_object* v_body_464_; uint8_t v_binderInfo_465_; lean_object* v___y_467_; lean_object* v___y_468_; lean_object* v___y_469_; lean_object* v___y_470_; lean_object* v___x_482_; uint8_t v___x_483_; 
v_binderName_462_ = lean_ctor_get(v_expr_417_, 0);
v_binderType_463_ = lean_ctor_get(v_expr_417_, 1);
lean_inc_ref(v_binderType_463_);
v_body_464_ = lean_ctor_get(v_expr_417_, 2);
v_binderInfo_465_ = lean_ctor_get_uint8(v_expr_417_, sizeof(void*)*3 + 8);
v___x_482_ = lean_unsigned_to_nat(0u);
v___x_483_ = lean_nat_dec_eq(v___x_426_, v___x_482_);
if (v___x_483_ == 0)
{
uint8_t v___x_484_; 
lean_inc_ref(v_body_464_);
lean_inc(v_binderName_462_);
lean_dec_ref_known(v_expr_417_, 3);
lean_dec_ref(v___x_437_);
v___x_484_ = lean_nat_dec_eq(v___x_426_, v___x_429_);
if (v___x_484_ == 0)
{
lean_object* v___x_485_; 
lean_dec_ref(v_body_464_);
lean_dec_ref(v_binderType_463_);
lean_dec(v_binderName_462_);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_485_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v_err_438_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_485_;
}
else
{
lean_object* v___f_486_; lean_object* v___x_487_; 
lean_dec_ref_known(v_err_438_, 2);
lean_inc(v_binderName_462_);
v___f_486_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0___boxed), 10, 4);
lean_closure_set(v___f_486_, 0, v_body_464_);
lean_closure_set(v___f_486_, 1, v_i_418_);
lean_closure_set(v___f_486_, 2, v_pos_416_);
lean_closure_set(v___f_486_, 3, v_binderName_462_);
v___x_487_ = l_Lean_Meta_withLocalDeclNoLocalInstanceUpdate___redArg(v_binderName_462_, v_binderInfo_465_, v_binderType_463_, v___f_486_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_487_;
}
}
else
{
lean_object* v___x_488_; 
lean_dec_ref_known(v_err_438_, 2);
lean_inc_ref(v_binderType_463_);
v___x_488_ = l_Lean_Meta_isProp(v_binderType_463_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; uint8_t v___x_490_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
lean_inc(v_a_489_);
lean_dec_ref_known(v___x_488_, 1);
v___x_490_ = lean_unbox(v_a_489_);
lean_dec(v_a_489_);
if (v___x_490_ == 0)
{
uint8_t v___x_491_; 
v___x_491_ = l_Lean_Expr_isArrow(v_expr_417_);
lean_dec_ref_known(v_expr_417_, 3);
if (v___x_491_ == 0)
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; 
v___x_492_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__5, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__5);
v___x_493_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
lean_ctor_set(v___x_493_, 1, v___x_437_);
v___x_494_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_493_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_dec_ref_known(v___x_494_, 1);
v___y_467_ = v_a_419_;
v___y_468_ = v_a_420_;
v___y_469_ = v_a_421_;
v___y_470_ = v_a_422_;
goto v___jp_466_;
}
else
{
lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_502_; 
lean_dec_ref(v_binderType_463_);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v_a_495_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_502_ == 0)
{
v___x_497_ = v___x_494_;
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_494_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_500_; 
if (v_isShared_498_ == 0)
{
v___x_500_ = v___x_497_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_a_495_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
else
{
lean_dec_ref(v___x_437_);
v___y_467_ = v_a_419_;
v___y_468_ = v_a_420_;
v___y_469_ = v_a_421_;
v___y_470_ = v_a_422_;
goto v___jp_466_;
}
}
else
{
lean_dec_ref_known(v_expr_417_, 3);
lean_dec_ref(v___x_437_);
v___y_467_ = v_a_419_;
v___y_468_ = v_a_420_;
v___y_469_ = v_a_421_;
v___y_470_ = v_a_422_;
goto v___jp_466_;
}
}
else
{
lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_510_; 
lean_dec_ref(v_binderType_463_);
lean_dec_ref_known(v_expr_417_, 3);
lean_dec_ref(v___x_437_);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v_a_503_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_510_ == 0)
{
v___x_505_ = v___x_488_;
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_dec(v___x_488_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_510_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_508_; 
if (v_isShared_506_ == 0)
{
v___x_508_ = v___x_505_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v_a_503_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
}
}
v___jp_466_:
{
lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_471_ = l_Fin_succ___redArg(v_i_418_);
lean_dec(v_i_418_);
v___x_472_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_416_, v_binderType_463_, v___x_471_, v___y_467_, v___y_468_, v___y_469_, v___y_470_);
if (lean_obj_tag(v___x_472_) == 0)
{
lean_object* v_a_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_481_; 
v_a_473_ = lean_ctor_get(v___x_472_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v___x_472_);
if (v_isSharedCheck_481_ == 0)
{
v___x_475_ = v___x_472_;
v_isShared_476_ = v_isSharedCheck_481_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_a_473_);
lean_dec(v___x_472_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_481_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___x_477_; lean_object* v___x_479_; 
v___x_477_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_477_, 0, v_a_473_);
if (v_isShared_476_ == 0)
{
lean_ctor_set(v___x_475_, 0, v___x_477_);
v___x_479_ = v___x_475_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v___x_477_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
else
{
return v___x_472_;
}
}
}
case 8:
{
lean_object* v_declName_511_; lean_object* v_type_512_; lean_object* v_body_513_; lean_object* v___x_514_; uint8_t v___x_515_; 
v_declName_511_ = lean_ctor_get(v_expr_417_, 0);
lean_inc(v_declName_511_);
v_type_512_ = lean_ctor_get(v_expr_417_, 1);
lean_inc_ref(v_type_512_);
v_body_513_ = lean_ctor_get(v_expr_417_, 3);
lean_inc_ref(v_body_513_);
lean_dec_ref_known(v_expr_417_, 4);
v___x_514_ = lean_unsigned_to_nat(0u);
v___x_515_ = lean_nat_dec_eq(v___x_426_, v___x_514_);
if (v___x_515_ == 0)
{
uint8_t v___x_516_; 
v___x_516_ = lean_nat_dec_eq(v___x_426_, v___x_429_);
if (v___x_516_ == 0)
{
lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_517_ = lean_unsigned_to_nat(2u);
v___x_518_ = lean_nat_dec_eq(v___x_426_, v___x_517_);
if (v___x_518_ == 0)
{
lean_object* v___x_519_; 
lean_dec_ref(v_body_513_);
lean_dec_ref(v_type_512_);
lean_dec(v_declName_511_);
lean_dec_ref(v___x_437_);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_519_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v_err_438_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_519_;
}
else
{
lean_object* v___f_520_; uint8_t v___x_521_; lean_object* v___x_522_; 
lean_dec_ref_known(v_err_438_, 2);
lean_inc(v_declName_511_);
v___f_520_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__2___boxed), 11, 5);
lean_closure_set(v___f_520_, 0, v_body_513_);
lean_closure_set(v___f_520_, 1, v_i_418_);
lean_closure_set(v___f_520_, 2, v_pos_416_);
lean_closure_set(v___f_520_, 3, v_declName_511_);
lean_closure_set(v___f_520_, 4, v___x_437_);
v___x_521_ = 0;
v___x_522_ = l_Lean_Meta_withLocalDeclNoLocalInstanceUpdate___redArg(v_declName_511_, v___x_521_, v_type_512_, v___f_520_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_522_;
}
}
else
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
lean_dec_ref(v_body_513_);
lean_dec_ref(v_type_512_);
lean_dec(v_declName_511_);
lean_dec_ref_known(v_err_438_, 2);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_523_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__7, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__7);
v___x_524_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_524_, 0, v___x_523_);
lean_ctor_set(v___x_524_, 1, v___x_437_);
v___x_525_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_524_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_525_;
}
}
else
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
lean_dec_ref(v_body_513_);
lean_dec_ref(v_type_512_);
lean_dec(v_declName_511_);
lean_dec_ref_known(v_err_438_, 2);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_526_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__9, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__9);
v___x_527_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_527_, 0, v___x_526_);
lean_ctor_set(v___x_527_, 1, v___x_437_);
v___x_528_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_527_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_528_;
}
}
case 10:
{
lean_object* v_expr_529_; 
lean_dec_ref_known(v_err_438_, 2);
lean_dec_ref(v___x_437_);
v_expr_529_ = lean_ctor_get(v_expr_417_, 1);
lean_inc_ref(v_expr_529_);
lean_dec_ref_known(v_expr_417_, 2);
v_expr_417_ = v_expr_529_;
goto _start;
}
case 11:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; 
lean_dec_ref_known(v_expr_417_, 3);
lean_dec_ref_known(v_err_438_, 2);
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_531_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__11, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__11);
v___x_532_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_532_, 0, v___x_531_);
lean_ctor_set(v___x_532_, 1, v___x_437_);
v___x_533_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_532_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_533_;
}
default: 
{
lean_object* v___x_534_; 
lean_dec_ref(v___x_437_);
lean_dec(v_i_418_);
lean_dec_ref(v_expr_417_);
lean_dec_ref(v_pos_416_);
v___x_534_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v_err_438_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_534_;
}
}
}
else
{
lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; 
lean_dec(v_i_418_);
lean_dec_ref(v_pos_416_);
v___x_535_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__13, &lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__13);
v___x_536_ = l_Lean_indentExpr(v_expr_417_);
v___x_537_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_535_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_537_, v_a_419_, v_a_420_, v_a_421_, v_a_422_);
return v___x_538_;
}
}
else
{
lean_object* v___x_539_; lean_object* v___x_540_; 
lean_dec(v_i_418_);
lean_dec_ref(v_expr_417_);
lean_dec_ref(v_pos_416_);
v___x_539_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___closed__14));
v___x_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
return v___x_540_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___lam__0(lean_object* v_body_541_, lean_object* v_i_542_, lean_object* v_pos_543_, lean_object* v_binderName_544_, lean_object* v_fvar_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_){
_start:
{
lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_551_ = lean_expr_instantiate1(v_body_541_, v_fvar_545_);
v___x_552_ = l_Fin_succ___redArg(v_i_542_);
v___x_553_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_543_, v___x_551_, v___x_552_, v___y_546_, v___y_547_, v___y_548_, v___y_549_);
if (lean_obj_tag(v___x_553_) == 0)
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_562_; 
v_a_554_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_562_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_562_ == 0)
{
v___x_556_ = v___x_553_;
v_isShared_557_ = v_isSharedCheck_562_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_553_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_562_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___x_558_; lean_object* v___x_560_; 
v___x_558_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_558_, 0, v_binderName_544_);
lean_ctor_set(v___x_558_, 1, v_a_554_);
if (v_isShared_557_ == 0)
{
lean_ctor_set(v___x_556_, 0, v___x_558_);
v___x_560_ = v___x_556_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v___x_558_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
else
{
lean_dec(v_binderName_544_);
return v___x_553_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___boxed(lean_object* v_pos_563_, lean_object* v_expr_564_, lean_object* v_i_565_, lean_object* v_acc_566_, lean_object* v_n_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_){
_start:
{
lean_object* v_res_573_; 
v_res_573_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT(v_pos_563_, v_expr_564_, v_i_565_, v_acc_566_, v_n_567_, v_a_568_, v_a_569_, v_a_570_, v_a_571_);
lean_dec(v_a_571_);
lean_dec_ref(v_a_570_);
lean_dec(v_a_569_);
lean_dec_ref(v_a_568_);
return v_res_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go___boxed(lean_object* v_pos_574_, lean_object* v_expr_575_, lean_object* v_i_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_574_, v_expr_575_, v_i_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
lean_dec(v_a_578_);
lean_dec_ref(v_a_577_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0(lean_object* v_00_u03b1_583_, lean_object* v_msg_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
lean_object* v___x_590_; 
v___x_590_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v_msg_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___boxed(lean_object* v_00_u03b1_591_, lean_object* v_msg_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0(v_00_u03b1_591_, v_msg_592_, v___y_593_, v___y_594_, v___y_595_, v___y_596_);
lean_dec(v___y_596_);
lean_dec_ref(v___y_595_);
lean_dec(v___y_594_);
lean_dec_ref(v___y_593_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray(lean_object* v_expr_599_, lean_object* v_pos_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_606_ = lean_array_get_size(v_pos_600_);
v___x_607_ = lean_unsigned_to_nat(1u);
v___x_608_ = lean_nat_add(v___x_606_, v___x_607_);
v___x_609_ = lean_unsigned_to_nat(0u);
v___x_610_ = lean_nat_mod(v___x_609_, v___x_608_);
lean_dec(v___x_608_);
v___x_611_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go(v_pos_600_, v_expr_599_, v___x_610_, v_a_601_, v_a_602_, v_a_603_, v_a_604_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray___boxed(lean_object* v_expr_612_, lean_object* v_pos_613_, lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray(v_expr_612_, v_pos_613_, v_a_614_, v_a_615_, v_a_616_, v_a_617_);
lean_dec(v_a_617_);
lean_dec_ref(v_a_616_);
lean_dec(v_a_615_);
lean_dec_ref(v_a_614_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ofSubExprPos(lean_object* v_expr_620_, lean_object* v_pos_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_, lean_object* v_a_625_){
_start:
{
lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_627_ = l_Lean_SubExpr_Pos_toArray(v_pos_621_);
v___x_628_ = lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray(v_expr_620_, v___x_627_, v_a_622_, v_a_623_, v_a_624_, v_a_625_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_Path_ofSubExprPos___boxed(lean_object* v_expr_629_, lean_object* v_pos_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ofSubExprPos(v_expr_629_, v_pos_630_, v_a_631_, v_a_632_, v_a_633_, v_a_634_);
lean_dec(v_a_634_);
lean_dec_ref(v_a_633_);
lean_dec(v_a_632_);
lean_dec_ref(v_a_631_);
lean_dec(v_pos_630_);
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0(lean_object* v_toPure_637_, lean_object* v_____do__lift_638_){
_start:
{
uint8_t v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_639_ = 0;
v___x_640_ = l_Lean_SourceInfo_fromRef(v_____do__lift_638_, v___x_639_);
v___x_641_ = lean_apply_2(v_toPure_637_, lean_box(0), v___x_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0___boxed(lean_object* v_toPure_642_, lean_object* v_____do__lift_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0(v_toPure_642_, v_____do__lift_643_);
lean_dec(v_____do__lift_643_);
return v_res_644_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8(void){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = l_Array_mkArray0(lean_box(0));
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14(lean_object* v_info_668_, lean_object* v_____do__lift_669_, lean_object* v_seq_670_, lean_object* v_toPure_671_, lean_object* v_quotCtx_672_){
_start:
{
lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3));
v___x_674_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4));
lean_inc_n(v_info_668_, 6);
v___x_675_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_675_, 0, v_info_668_);
lean_ctor_set(v___x_675_, 1, v___x_673_);
v___x_676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_677_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__7));
v___x_678_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_678_, 0, v_info_668_);
lean_ctor_set(v___x_678_, 1, v___x_677_);
v___x_679_ = l_Lean_Syntax_node2(v_info_668_, v___x_676_, v___x_678_, v_____do__lift_669_);
v___x_680_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_681_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_681_, 0, v_info_668_);
lean_ctor_set(v___x_681_, 1, v___x_676_);
lean_ctor_set(v___x_681_, 2, v___x_680_);
v___x_682_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9));
v___x_683_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_683_, 0, v_info_668_);
lean_ctor_set(v___x_683_, 1, v___x_682_);
v___x_684_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11));
v___x_685_ = l_Lean_Syntax_node1(v_info_668_, v___x_684_, v_seq_670_);
v___x_686_ = l_Lean_Syntax_node5(v_info_668_, v___x_674_, v___x_675_, v___x_679_, v___x_681_, v___x_683_, v___x_685_);
v___x_687_ = lean_apply_2(v_toPure_671_, lean_box(0), v___x_686_);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___boxed(lean_object* v_info_688_, lean_object* v_____do__lift_689_, lean_object* v_seq_690_, lean_object* v_toPure_691_, lean_object* v_quotCtx_692_){
_start:
{
lean_object* v_res_693_; 
v_res_693_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14(v_info_688_, v_____do__lift_689_, v_seq_690_, v_toPure_691_, v_quotCtx_692_);
lean_dec(v_quotCtx_692_);
return v_res_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3(lean_object* v_toBind_694_, lean_object* v_getContext_695_, lean_object* v___f_696_, lean_object* v_scp_697_){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lean_apply_4(v_toBind_694_, lean_box(0), lean_box(0), v_getContext_695_, v___f_696_);
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed(lean_object* v_toBind_699_, lean_object* v_getContext_700_, lean_object* v___f_701_, lean_object* v_scp_702_){
_start:
{
lean_object* v_res_703_; 
v_res_703_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3(v_toBind_699_, v_getContext_700_, v___f_701_, v_scp_702_);
lean_dec(v_scp_702_);
return v_res_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__17(lean_object* v_____do__lift_704_, lean_object* v_seq_705_, lean_object* v_toPure_706_, lean_object* v_toBind_707_, lean_object* v_getContext_708_, lean_object* v_getCurrMacroScope_709_, lean_object* v_info_710_){
_start:
{
lean_object* v___f_711_; lean_object* v___f_712_; lean_object* v___x_713_; 
v___f_711_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___boxed), 5, 4);
lean_closure_set(v___f_711_, 0, v_info_710_);
lean_closure_set(v___f_711_, 1, v_____do__lift_704_);
lean_closure_set(v___f_711_, 2, v_seq_705_);
lean_closure_set(v___f_711_, 3, v_toPure_706_);
lean_inc(v_toBind_707_);
v___f_712_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_712_, 0, v_toBind_707_);
lean_closure_set(v___f_712_, 1, v_getContext_708_);
lean_closure_set(v___f_712_, 2, v___f_711_);
v___x_713_ = lean_apply_4(v_toBind_707_, lean_box(0), lean_box(0), v_getCurrMacroScope_709_, v___f_712_);
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__16(lean_object* v_seq_714_, lean_object* v_toPure_715_, lean_object* v_toBind_716_, lean_object* v_getContext_717_, lean_object* v_getCurrMacroScope_718_, lean_object* v___x_719_, lean_object* v_____do__lift_720_){
_start:
{
lean_object* v___f_721_; lean_object* v___x_722_; 
lean_inc(v_toBind_716_);
v___f_721_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__17), 7, 6);
lean_closure_set(v___f_721_, 0, v_____do__lift_720_);
lean_closure_set(v___f_721_, 1, v_seq_714_);
lean_closure_set(v___f_721_, 2, v_toPure_715_);
lean_closure_set(v___f_721_, 3, v_toBind_716_);
lean_closure_set(v___f_721_, 4, v_getContext_717_);
lean_closure_set(v___f_721_, 5, v_getCurrMacroScope_718_);
v___x_722_ = lean_apply_4(v_toBind_716_, lean_box(0), lean_box(0), v___x_719_, v___f_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__12(lean_object* v_info_723_, lean_object* v_seq_724_, lean_object* v_toPure_725_, lean_object* v_quotCtx_726_){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; 
v___x_727_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3));
v___x_728_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__4));
lean_inc_n(v_info_723_, 4);
v___x_729_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_729_, 0, v_info_723_);
lean_ctor_set(v___x_729_, 1, v___x_727_);
v___x_730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_731_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_732_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_732_, 0, v_info_723_);
lean_ctor_set(v___x_732_, 1, v___x_730_);
lean_ctor_set(v___x_732_, 2, v___x_731_);
v___x_733_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9));
v___x_734_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_734_, 0, v_info_723_);
lean_ctor_set(v___x_734_, 1, v___x_733_);
v___x_735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__11));
v___x_736_ = l_Lean_Syntax_node1(v_info_723_, v___x_735_, v_seq_724_);
lean_inc_ref(v___x_732_);
v___x_737_ = l_Lean_Syntax_node5(v_info_723_, v___x_728_, v___x_729_, v___x_732_, v___x_732_, v___x_734_, v___x_736_);
v___x_738_ = lean_apply_2(v_toPure_725_, lean_box(0), v___x_737_);
return v___x_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__12___boxed(lean_object* v_info_739_, lean_object* v_seq_740_, lean_object* v_toPure_741_, lean_object* v_quotCtx_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__12(v_info_739_, v_seq_740_, v_toPure_741_, v_quotCtx_742_);
lean_dec(v_quotCtx_742_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__15(lean_object* v_seq_744_, lean_object* v_toPure_745_, lean_object* v_toBind_746_, lean_object* v_getContext_747_, lean_object* v_getCurrMacroScope_748_, lean_object* v_info_749_){
_start:
{
lean_object* v___f_750_; lean_object* v___f_751_; lean_object* v___x_752_; 
v___f_750_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__12___boxed), 4, 3);
lean_closure_set(v___f_750_, 0, v_info_749_);
lean_closure_set(v___f_750_, 1, v_seq_744_);
lean_closure_set(v___f_750_, 2, v_toPure_745_);
lean_inc(v_toBind_746_);
v___f_751_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_751_, 0, v_toBind_746_);
lean_closure_set(v___f_751_, 1, v_getContext_747_);
lean_closure_set(v___f_751_, 2, v___f_750_);
v___x_752_ = lean_apply_4(v_toBind_746_, lean_box(0), lean_box(0), v_getCurrMacroScope_748_, v___f_751_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__18(lean_object* v_loc_753_, lean_object* v_toPure_754_, lean_object* v_toBind_755_, lean_object* v_getContext_756_, lean_object* v_getCurrMacroScope_757_, lean_object* v___x_758_, lean_object* v_inst_759_, lean_object* v_toMonadRef_760_, lean_object* v_seq_761_){
_start:
{
if (lean_obj_tag(v_loc_753_) == 0)
{
lean_object* v___f_762_; lean_object* v___x_763_; 
lean_dec_ref(v_toMonadRef_760_);
lean_dec_ref(v_inst_759_);
lean_inc(v_toBind_755_);
v___f_762_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__15), 6, 5);
lean_closure_set(v___f_762_, 0, v_seq_761_);
lean_closure_set(v___f_762_, 1, v_toPure_754_);
lean_closure_set(v___f_762_, 2, v_toBind_755_);
lean_closure_set(v___f_762_, 3, v_getContext_756_);
lean_closure_set(v___f_762_, 4, v_getCurrMacroScope_757_);
v___x_763_ = lean_apply_4(v_toBind_755_, lean_box(0), lean_box(0), v___x_758_, v___f_762_);
return v___x_763_;
}
else
{
lean_object* v_val_764_; lean_object* v___f_765_; uint8_t v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v_val_764_ = lean_ctor_get(v_loc_753_, 0);
lean_inc(v_val_764_);
lean_dec_ref_known(v_loc_753_, 1);
lean_inc(v_toBind_755_);
v___f_765_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__16), 7, 6);
lean_closure_set(v___f_765_, 0, v_seq_761_);
lean_closure_set(v___f_765_, 1, v_toPure_754_);
lean_closure_set(v___f_765_, 2, v_toBind_755_);
lean_closure_set(v___f_765_, 3, v_getContext_756_);
lean_closure_set(v___f_765_, 4, v_getCurrMacroScope_757_);
lean_closure_set(v___f_765_, 5, v___x_758_);
v___x_766_ = 0;
v___x_767_ = l_Lean_mkIdentFromRef___redArg(v_inst_759_, v_toMonadRef_760_, v_val_764_, v___x_766_);
v___x_768_ = lean_apply_4(v_toBind_755_, lean_box(0), lean_box(0), v___x_767_, v___f_765_);
return v___x_768_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26(lean_object* v_info_787_, lean_object* v_toPure_788_, lean_object* v_quotCtx_789_){
_start:
{
lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; 
v___x_790_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_791_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3));
v___x_792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_793_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
lean_inc_n(v_info_787_, 4);
v___x_794_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_794_, 0, v_info_787_);
lean_ctor_set(v___x_794_, 1, v___x_792_);
lean_ctor_set(v___x_794_, 2, v___x_793_);
v___x_795_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__5));
v___x_796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__6));
v___x_797_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_797_, 0, v_info_787_);
lean_ctor_set(v___x_797_, 1, v___x_796_);
v___x_798_ = l_Lean_Syntax_node1(v_info_787_, v___x_795_, v___x_797_);
lean_inc_ref(v___x_794_);
v___x_799_ = l_Lean_Syntax_node3(v_info_787_, v___x_791_, v___x_794_, v___x_794_, v___x_798_);
v___x_800_ = l_Lean_Syntax_node1(v_info_787_, v___x_790_, v___x_799_);
v___x_801_ = lean_apply_2(v_toPure_788_, lean_box(0), v___x_800_);
return v___x_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___boxed(lean_object* v_info_802_, lean_object* v_toPure_803_, lean_object* v_quotCtx_804_){
_start:
{
lean_object* v_res_805_; 
v_res_805_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26(v_info_802_, v_toPure_803_, v_quotCtx_804_);
lean_dec(v_quotCtx_804_);
return v_res_805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__27(lean_object* v_toPure_806_, lean_object* v_toBind_807_, lean_object* v_getContext_808_, lean_object* v_getCurrMacroScope_809_, lean_object* v_info_810_){
_start:
{
lean_object* v___f_811_; lean_object* v___f_812_; lean_object* v___x_813_; 
v___f_811_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___boxed), 3, 2);
lean_closure_set(v___f_811_, 0, v_info_810_);
lean_closure_set(v___f_811_, 1, v_toPure_806_);
lean_inc(v_toBind_807_);
v___f_812_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_812_, 0, v_toBind_807_);
lean_closure_set(v___f_812_, 1, v_getContext_808_);
lean_closure_set(v___f_812_, 2, v___f_811_);
v___x_813_ = lean_apply_4(v_toBind_807_, lean_box(0), lean_box(0), v_getCurrMacroScope_809_, v___f_812_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29(lean_object* v_info_826_, lean_object* v_toPure_827_, lean_object* v_quotCtx_828_){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_829_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1));
v___x_830_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4));
v___x_831_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__5));
lean_inc_n(v_info_826_, 2);
v___x_832_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_832_, 0, v_info_826_);
lean_ctor_set(v___x_832_, 1, v___x_831_);
v___x_833_ = l_Lean_Syntax_node1(v_info_826_, v___x_830_, v___x_832_);
v___x_834_ = l_Lean_Syntax_node1(v_info_826_, v___x_829_, v___x_833_);
v___x_835_ = lean_apply_2(v_toPure_827_, lean_box(0), v___x_834_);
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___boxed(lean_object* v_info_836_, lean_object* v_toPure_837_, lean_object* v_quotCtx_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29(v_info_836_, v_toPure_837_, v_quotCtx_838_);
lean_dec(v_quotCtx_838_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__32(lean_object* v_toPure_840_, lean_object* v_toBind_841_, lean_object* v_getContext_842_, lean_object* v_getCurrMacroScope_843_, lean_object* v_info_844_){
_start:
{
lean_object* v___f_845_; lean_object* v___f_846_; lean_object* v___x_847_; 
v___f_845_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___boxed), 3, 2);
lean_closure_set(v___f_845_, 0, v_info_844_);
lean_closure_set(v___f_845_, 1, v_toPure_840_);
lean_inc(v_toBind_841_);
v___f_846_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_846_, 0, v_toBind_841_);
lean_closure_set(v___f_846_, 1, v_getContext_842_);
lean_closure_set(v___f_846_, 2, v___f_845_);
v___x_847_ = lean_apply_4(v_toBind_841_, lean_box(0), lean_box(0), v_getCurrMacroScope_843_, v___f_846_);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10(lean_object* v_info_857_, lean_object* v_xs_858_, lean_object* v_toPure_859_, lean_object* v_quotCtx_860_){
_start:
{
lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
v___x_861_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__0));
v___x_862_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1));
lean_inc_n(v_info_857_, 4);
v___x_863_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_863_, 0, v_info_857_);
lean_ctor_set(v___x_863_, 1, v___x_861_);
v___x_864_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__2));
v___x_865_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_865_, 0, v_info_857_);
lean_ctor_set(v___x_865_, 1, v___x_864_);
v___x_866_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_867_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_868_ = l_Array_append___redArg(v___x_867_, v_xs_858_);
v___x_869_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_869_, 0, v_info_857_);
lean_ctor_set(v___x_869_, 1, v___x_866_);
lean_ctor_set(v___x_869_, 2, v___x_868_);
v___x_870_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__3));
v___x_871_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_871_, 0, v_info_857_);
lean_ctor_set(v___x_871_, 1, v___x_870_);
v___x_872_ = l_Lean_Syntax_node4(v_info_857_, v___x_862_, v___x_863_, v___x_865_, v___x_869_, v___x_871_);
v___x_873_ = lean_apply_2(v_toPure_859_, lean_box(0), v___x_872_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___boxed(lean_object* v_info_874_, lean_object* v_xs_875_, lean_object* v_toPure_876_, lean_object* v_quotCtx_877_){
_start:
{
lean_object* v_res_878_; 
v_res_878_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10(v_info_874_, v_xs_875_, v_toPure_876_, v_quotCtx_877_);
lean_dec(v_quotCtx_877_);
lean_dec_ref(v_xs_875_);
return v_res_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__11(lean_object* v_xs_879_, lean_object* v_toPure_880_, lean_object* v_toBind_881_, lean_object* v_getContext_882_, lean_object* v_getCurrMacroScope_883_, lean_object* v_info_884_){
_start:
{
lean_object* v___f_885_; lean_object* v___f_886_; lean_object* v___x_887_; 
v___f_885_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___boxed), 4, 3);
lean_closure_set(v___f_885_, 0, v_info_884_);
lean_closure_set(v___f_885_, 1, v_xs_879_);
lean_closure_set(v___f_885_, 2, v_toPure_880_);
lean_inc(v_toBind_881_);
v___f_886_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_886_, 0, v_toBind_881_);
lean_closure_set(v___f_886_, 1, v_getContext_882_);
lean_closure_set(v___f_886_, 2, v___f_885_);
v___x_887_ = lean_apply_4(v_toBind_881_, lean_box(0), lean_box(0), v_getCurrMacroScope_883_, v___f_886_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__36(lean_object* v_info_888_, lean_object* v_____do__lift_889_, lean_object* v_toPure_890_, lean_object* v_quotCtx_891_){
_start:
{
lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_892_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1));
v___x_893_ = l_Lean_Syntax_node1(v_info_888_, v___x_892_, v_____do__lift_889_);
v___x_894_ = lean_apply_2(v_toPure_890_, lean_box(0), v___x_893_);
return v___x_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__36___boxed(lean_object* v_info_895_, lean_object* v_____do__lift_896_, lean_object* v_toPure_897_, lean_object* v_quotCtx_898_){
_start:
{
lean_object* v_res_899_; 
v_res_899_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__36(v_info_895_, v_____do__lift_896_, v_toPure_897_, v_quotCtx_898_);
lean_dec(v_quotCtx_898_);
return v_res_899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__35(lean_object* v_____do__lift_900_, lean_object* v_toPure_901_, lean_object* v_toBind_902_, lean_object* v_getContext_903_, lean_object* v_getCurrMacroScope_904_, lean_object* v_info_905_){
_start:
{
lean_object* v___f_906_; lean_object* v___f_907_; lean_object* v___x_908_; 
v___f_906_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__36___boxed), 4, 3);
lean_closure_set(v___f_906_, 0, v_info_905_);
lean_closure_set(v___f_906_, 1, v_____do__lift_900_);
lean_closure_set(v___f_906_, 2, v_toPure_901_);
lean_inc(v_toBind_902_);
v___f_907_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_907_, 0, v_toBind_902_);
lean_closure_set(v___f_907_, 1, v_getContext_903_);
lean_closure_set(v___f_907_, 2, v___f_906_);
v___x_908_ = lean_apply_4(v_toBind_902_, lean_box(0), lean_box(0), v_getCurrMacroScope_904_, v___f_907_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__34(lean_object* v_toMonadRef_909_, lean_object* v_toPure_910_, lean_object* v_toBind_911_, lean_object* v_getContext_912_, lean_object* v_getCurrMacroScope_913_, lean_object* v___f_914_, lean_object* v___f_915_, lean_object* v_____do__lift_916_){
_start:
{
lean_object* v_getRef_917_; lean_object* v___f_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; 
v_getRef_917_ = lean_ctor_get(v_toMonadRef_909_, 0);
lean_inc(v_getRef_917_);
lean_dec_ref(v_toMonadRef_909_);
lean_inc_n(v_toBind_911_, 3);
v___f_918_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__35), 6, 5);
lean_closure_set(v___f_918_, 0, v_____do__lift_916_);
lean_closure_set(v___f_918_, 1, v_toPure_910_);
lean_closure_set(v___f_918_, 2, v_toBind_911_);
lean_closure_set(v___f_918_, 3, v_getContext_912_);
lean_closure_set(v___f_918_, 4, v_getCurrMacroScope_913_);
v___x_919_ = lean_apply_4(v_toBind_911_, lean_box(0), lean_box(0), v_getRef_917_, v___f_914_);
v___x_920_ = lean_apply_4(v_toBind_911_, lean_box(0), lean_box(0), v___x_919_, v___f_918_);
v___x_921_ = lean_apply_4(v_toBind_911_, lean_box(0), lean_box(0), v___x_920_, v___f_915_);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__22(lean_object* v_funStx_922_, lean_object* v_toPure_923_, lean_object* v_a_924_, lean_object* v_x_925_, lean_object* v___y_926_){
_start:
{
lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; 
v___x_927_ = lean_array_push(v___y_926_, v_funStx_922_);
v___x_928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_928_, 0, v___x_927_);
v___x_929_ = lean_apply_2(v_toPure_923_, lean_box(0), v___x_928_);
return v___x_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__22___boxed(lean_object* v_funStx_930_, lean_object* v_toPure_931_, lean_object* v_a_932_, lean_object* v_x_933_, lean_object* v___y_934_){
_start:
{
lean_object* v_res_935_; 
v_res_935_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__22(v_funStx_930_, v_toPure_931_, v_a_932_, v_x_933_, v___y_934_);
lean_dec(v_a_932_);
return v_res_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__23(lean_object* v_toPure_936_, lean_object* v___x_937_, lean_object* v_depth_938_, lean_object* v_inst_939_, lean_object* v_toBind_940_, lean_object* v___f_941_, uint8_t v___x_942_, lean_object* v_arr_943_, lean_object* v_enterStx_944_, lean_object* v_funStx_945_){
_start:
{
lean_object* v___f_946_; lean_object* v_arr_948_; 
v___f_946_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__22___boxed), 5, 2);
lean_closure_set(v___f_946_, 0, v_funStx_945_);
lean_closure_set(v___f_946_, 1, v_toPure_936_);
if (v___x_942_ == 0)
{
lean_object* v_arr_953_; 
v_arr_953_ = lean_array_push(v_arr_943_, v_enterStx_944_);
v_arr_948_ = v_arr_953_;
goto v___jp_947_;
}
else
{
lean_dec(v_enterStx_944_);
v_arr_948_ = v_arr_943_;
goto v___jp_947_;
}
v___jp_947_:
{
lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; 
v___x_949_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_937_);
v___x_950_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_950_, 0, v___x_937_);
lean_ctor_set(v___x_950_, 1, v_depth_938_);
lean_ctor_set(v___x_950_, 2, v___x_949_);
v___x_951_ = l___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop(lean_box(0), lean_box(0), v_inst_939_, v___x_950_, v___f_946_, v_arr_948_, v___x_937_, lean_box(0), lean_box(0));
v___x_952_ = lean_apply_4(v_toBind_940_, lean_box(0), lean_box(0), v___x_951_, v___f_941_);
return v___x_952_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__23___boxed(lean_object* v_toPure_954_, lean_object* v___x_955_, lean_object* v_depth_956_, lean_object* v_inst_957_, lean_object* v_toBind_958_, lean_object* v___f_959_, lean_object* v___x_960_, lean_object* v_arr_961_, lean_object* v_enterStx_962_, lean_object* v_funStx_963_){
_start:
{
uint8_t v___x_2691__boxed_964_; lean_object* v_res_965_; 
v___x_2691__boxed_964_ = lean_unbox(v___x_960_);
v_res_965_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__23(v_toPure_954_, v___x_955_, v_depth_956_, v_inst_957_, v_toBind_958_, v___f_959_, v___x_2691__boxed_964_, v_arr_961_, v_enterStx_962_, v_funStx_963_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__24(lean_object* v_toPure_966_, lean_object* v___x_967_, lean_object* v_depth_968_, lean_object* v_inst_969_, lean_object* v_toBind_970_, lean_object* v___f_971_, uint8_t v___x_972_, lean_object* v_arr_973_, lean_object* v___x_974_, lean_object* v___f_975_, lean_object* v_enterStx_976_){
_start:
{
lean_object* v___x_977_; lean_object* v___f_978_; lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_977_ = lean_box(v___x_972_);
lean_inc_n(v_toBind_970_, 2);
v___f_978_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__23___boxed), 10, 9);
lean_closure_set(v___f_978_, 0, v_toPure_966_);
lean_closure_set(v___f_978_, 1, v___x_967_);
lean_closure_set(v___f_978_, 2, v_depth_968_);
lean_closure_set(v___f_978_, 3, v_inst_969_);
lean_closure_set(v___f_978_, 4, v_toBind_970_);
lean_closure_set(v___f_978_, 5, v___f_971_);
lean_closure_set(v___f_978_, 6, v___x_977_);
lean_closure_set(v___f_978_, 7, v_arr_973_);
lean_closure_set(v___f_978_, 8, v_enterStx_976_);
v___x_979_ = lean_apply_4(v_toBind_970_, lean_box(0), lean_box(0), v___x_974_, v___f_975_);
v___x_980_ = lean_apply_4(v_toBind_970_, lean_box(0), lean_box(0), v___x_979_, v___f_978_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__24___boxed(lean_object* v_toPure_981_, lean_object* v___x_982_, lean_object* v_depth_983_, lean_object* v_inst_984_, lean_object* v_toBind_985_, lean_object* v___f_986_, lean_object* v___x_987_, lean_object* v_arr_988_, lean_object* v___x_989_, lean_object* v___f_990_, lean_object* v_enterStx_991_){
_start:
{
uint8_t v___x_2718__boxed_992_; lean_object* v_res_993_; 
v___x_2718__boxed_992_ = lean_unbox(v___x_987_);
v_res_993_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__24(v_toPure_981_, v___x_982_, v_depth_983_, v_inst_984_, v_toBind_985_, v___f_986_, v___x_2718__boxed_992_, v_arr_988_, v___x_989_, v___f_990_, v_enterStx_991_);
return v_res_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19(lean_object* v_arr_1002_, lean_object* v_info_1003_, lean_object* v_toPure_1004_, lean_object* v_quotCtx_1005_){
_start:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1006_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__1));
v___x_1007_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1008_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_1009_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__2));
v___x_1010_ = l_Lean_Syntax_SepArray_ofElems(v___x_1009_, v_arr_1002_);
v___x_1011_ = l_Array_append___redArg(v___x_1008_, v___x_1010_);
lean_dec_ref(v___x_1010_);
lean_inc(v_info_1003_);
v___x_1012_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1012_, 0, v_info_1003_);
lean_ctor_set(v___x_1012_, 1, v___x_1007_);
lean_ctor_set(v___x_1012_, 2, v___x_1011_);
v___x_1013_ = l_Lean_Syntax_node1(v_info_1003_, v___x_1006_, v___x_1012_);
v___x_1014_ = lean_apply_2(v_toPure_1004_, lean_box(0), v___x_1013_);
return v___x_1014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___boxed(lean_object* v_arr_1015_, lean_object* v_info_1016_, lean_object* v_toPure_1017_, lean_object* v_quotCtx_1018_){
_start:
{
lean_object* v_res_1019_; 
v_res_1019_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19(v_arr_1015_, v_info_1016_, v_toPure_1017_, v_quotCtx_1018_);
lean_dec(v_quotCtx_1018_);
lean_dec_ref(v_arr_1015_);
return v_res_1019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__21(lean_object* v_arr_1020_, lean_object* v_toPure_1021_, lean_object* v_toBind_1022_, lean_object* v_getContext_1023_, lean_object* v_getCurrMacroScope_1024_, lean_object* v_info_1025_){
_start:
{
lean_object* v___f_1026_; lean_object* v___f_1027_; lean_object* v___x_1028_; 
v___f_1026_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___boxed), 4, 3);
lean_closure_set(v___f_1026_, 0, v_arr_1020_);
lean_closure_set(v___f_1026_, 1, v_info_1025_);
lean_closure_set(v___f_1026_, 2, v_toPure_1021_);
lean_inc(v_toBind_1022_);
v___f_1027_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_1027_, 0, v_toBind_1022_);
lean_closure_set(v___f_1027_, 1, v_getContext_1023_);
lean_closure_set(v___f_1027_, 2, v___f_1026_);
v___x_1028_ = lean_apply_4(v_toBind_1022_, lean_box(0), lean_box(0), v_getCurrMacroScope_1024_, v___f_1027_);
return v___x_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__20(lean_object* v_convStx_1029_, lean_object* v_toPure_1030_, lean_object* v_toBind_1031_, lean_object* v_getContext_1032_, lean_object* v_getCurrMacroScope_1033_, lean_object* v___x_1034_, lean_object* v___f_1035_, lean_object* v_____s_1036_){
_start:
{
lean_object* v_arr_1037_; lean_object* v___f_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v_arr_1037_ = lean_array_push(v_____s_1036_, v_convStx_1029_);
lean_inc_n(v_toBind_1031_, 2);
v___f_1038_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__21), 6, 5);
lean_closure_set(v___f_1038_, 0, v_arr_1037_);
lean_closure_set(v___f_1038_, 1, v_toPure_1030_);
lean_closure_set(v___f_1038_, 2, v_toBind_1031_);
lean_closure_set(v___f_1038_, 3, v_getContext_1032_);
lean_closure_set(v___f_1038_, 4, v_getCurrMacroScope_1033_);
v___x_1039_ = lean_apply_4(v_toBind_1031_, lean_box(0), lean_box(0), v___x_1034_, v___f_1038_);
v___x_1040_ = lean_apply_4(v_toBind_1031_, lean_box(0), lean_box(0), v___x_1039_, v___f_1035_);
return v___x_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__30(lean_object* v_info_1041_, lean_object* v_bi_1042_, lean_object* v_toPure_1043_, lean_object* v_quotCtx_1044_){
_start:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; 
v___x_1045_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1046_ = l_Lean_Syntax_node1(v_info_1041_, v___x_1045_, v_bi_1042_);
v___x_1047_ = lean_apply_2(v_toPure_1043_, lean_box(0), v___x_1046_);
return v___x_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__30___boxed(lean_object* v_info_1048_, lean_object* v_bi_1049_, lean_object* v_toPure_1050_, lean_object* v_quotCtx_1051_){
_start:
{
lean_object* v_res_1052_; 
v_res_1052_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__30(v_info_1048_, v_bi_1049_, v_toPure_1050_, v_quotCtx_1051_);
lean_dec(v_quotCtx_1051_);
return v_res_1052_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__28(lean_object* v_bi_1053_, lean_object* v_toPure_1054_, lean_object* v_toBind_1055_, lean_object* v_getContext_1056_, lean_object* v_getCurrMacroScope_1057_, lean_object* v_info_1058_){
_start:
{
lean_object* v___f_1059_; lean_object* v___f_1060_; lean_object* v___x_1061_; 
v___f_1059_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__30___boxed), 4, 3);
lean_closure_set(v___f_1059_, 0, v_info_1058_);
lean_closure_set(v___f_1059_, 1, v_bi_1053_);
lean_closure_set(v___f_1059_, 2, v_toPure_1054_);
lean_inc(v_toBind_1055_);
v___f_1060_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_1060_, 0, v_toBind_1055_);
lean_closure_set(v___f_1060_, 1, v_getContext_1056_);
lean_closure_set(v___f_1060_, 2, v___f_1059_);
v___x_1061_ = lean_apply_4(v_toBind_1055_, lean_box(0), lean_box(0), v_getCurrMacroScope_1057_, v___f_1060_);
return v___x_1061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__25(lean_object* v_toMonadRef_1062_, lean_object* v_toPure_1063_, lean_object* v_toBind_1064_, lean_object* v_getContext_1065_, lean_object* v_getCurrMacroScope_1066_, lean_object* v___f_1067_, lean_object* v___f_1068_, lean_object* v_bi_1069_){
_start:
{
lean_object* v_getRef_1070_; lean_object* v___f_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
v_getRef_1070_ = lean_ctor_get(v_toMonadRef_1062_, 0);
lean_inc(v_getRef_1070_);
lean_dec_ref(v_toMonadRef_1062_);
lean_inc_n(v_toBind_1064_, 3);
v___f_1071_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__28), 6, 5);
lean_closure_set(v___f_1071_, 0, v_bi_1069_);
lean_closure_set(v___f_1071_, 1, v_toPure_1063_);
lean_closure_set(v___f_1071_, 2, v_toBind_1064_);
lean_closure_set(v___f_1071_, 3, v_getContext_1065_);
lean_closure_set(v___f_1071_, 4, v_getCurrMacroScope_1066_);
v___x_1072_ = lean_apply_4(v_toBind_1064_, lean_box(0), lean_box(0), v_getRef_1070_, v___f_1067_);
v___x_1073_ = lean_apply_4(v_toBind_1064_, lean_box(0), lean_box(0), v___x_1072_, v___f_1071_);
v___x_1074_ = lean_apply_4(v_toBind_1064_, lean_box(0), lean_box(0), v___x_1073_, v___f_1068_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__33(uint8_t v___x_1075_, lean_object* v_toPure_1076_, lean_object* v_____do__lift_1077_){
_start:
{
lean_object* v___x_1078_; lean_object* v___x_1079_; 
v___x_1078_ = l_Lean_SourceInfo_fromRef(v_____do__lift_1077_, v___x_1075_);
v___x_1079_ = lean_apply_2(v_toPure_1076_, lean_box(0), v___x_1078_);
return v___x_1079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__33___boxed(lean_object* v___x_1080_, lean_object* v_toPure_1081_, lean_object* v_____do__lift_1082_){
_start:
{
uint8_t v___x_2867__boxed_1083_; lean_object* v_res_1084_; 
v___x_2867__boxed_1083_ = lean_unbox(v___x_1080_);
v_res_1084_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__33(v___x_2867__boxed_1083_, v_toPure_1081_, v_____do__lift_1082_);
lean_dec(v_____do__lift_1082_);
return v_res_1084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9(lean_object* v_info_1092_, lean_object* v_toPure_1093_, lean_object* v_quotCtx_1094_){
_start:
{
lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; 
v___x_1095_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__0));
v___x_1096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1));
lean_inc(v_info_1092_);
v___x_1097_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1097_, 0, v_info_1092_);
lean_ctor_set(v___x_1097_, 1, v___x_1095_);
v___x_1098_ = l_Lean_Syntax_node1(v_info_1092_, v___x_1096_, v___x_1097_);
v___x_1099_ = lean_apply_2(v_toPure_1093_, lean_box(0), v___x_1098_);
return v___x_1099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___boxed(lean_object* v_info_1100_, lean_object* v_toPure_1101_, lean_object* v_quotCtx_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9(v_info_1100_, v_toPure_1101_, v_quotCtx_1102_);
lean_dec(v_quotCtx_1102_);
return v_res_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__13(lean_object* v_toPure_1104_, lean_object* v_toBind_1105_, lean_object* v_getContext_1106_, lean_object* v_getCurrMacroScope_1107_, lean_object* v_info_1108_){
_start:
{
lean_object* v___f_1109_; lean_object* v___f_1110_; lean_object* v___x_1111_; 
v___f_1109_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___boxed), 3, 2);
lean_closure_set(v___f_1109_, 0, v_info_1108_);
lean_closure_set(v___f_1109_, 1, v_toPure_1104_);
lean_inc(v_toBind_1105_);
v___f_1110_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_1110_, 0, v_toBind_1105_);
lean_closure_set(v___f_1110_, 1, v_getContext_1106_);
lean_closure_set(v___f_1110_, 2, v___f_1109_);
v___x_1111_ = lean_apply_4(v_toBind_1105_, lean_box(0), lean_box(0), v_getCurrMacroScope_1107_, v___f_1110_);
return v___x_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__31(lean_object* v___f_1112_, lean_object* v_bi_1113_){
_start:
{
lean_object* v___x_1114_; 
v___x_1114_ = lean_apply_1(v___f_1112_, v_bi_1113_);
return v___x_1114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__4(lean_object* v_info_1115_, lean_object* v_num_1116_, lean_object* v_toPure_1117_, lean_object* v_quotCtx_1118_){
_start:
{
lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; 
v___x_1119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3));
v___x_1121_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1122_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
lean_inc_n(v_info_1115_, 2);
v___x_1123_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1123_, 0, v_info_1115_);
lean_ctor_set(v___x_1123_, 1, v___x_1121_);
lean_ctor_set(v___x_1123_, 2, v___x_1122_);
lean_inc_ref(v___x_1123_);
v___x_1124_ = l_Lean_Syntax_node3(v_info_1115_, v___x_1120_, v___x_1123_, v___x_1123_, v_num_1116_);
v___x_1125_ = l_Lean_Syntax_node1(v_info_1115_, v___x_1119_, v___x_1124_);
v___x_1126_ = lean_apply_2(v_toPure_1117_, lean_box(0), v___x_1125_);
return v___x_1126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__4___boxed(lean_object* v_info_1127_, lean_object* v_num_1128_, lean_object* v_toPure_1129_, lean_object* v_quotCtx_1130_){
_start:
{
lean_object* v_res_1131_; 
v_res_1131_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__4(v_info_1127_, v_num_1128_, v_toPure_1129_, v_quotCtx_1130_);
lean_dec(v_quotCtx_1130_);
return v_res_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__5(lean_object* v_num_1132_, lean_object* v_toPure_1133_, lean_object* v_toBind_1134_, lean_object* v_getContext_1135_, lean_object* v_getCurrMacroScope_1136_, lean_object* v_info_1137_){
_start:
{
lean_object* v___f_1138_; lean_object* v___f_1139_; lean_object* v___x_1140_; 
v___f_1138_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__4___boxed), 4, 3);
lean_closure_set(v___f_1138_, 0, v_info_1137_);
lean_closure_set(v___f_1138_, 1, v_num_1132_);
lean_closure_set(v___f_1138_, 2, v_toPure_1133_);
lean_inc(v_toBind_1134_);
v___f_1139_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_1139_, 0, v_toBind_1134_);
lean_closure_set(v___f_1139_, 1, v_getContext_1135_);
lean_closure_set(v___f_1139_, 2, v___f_1138_);
v___x_1140_ = lean_apply_4(v_toBind_1134_, lean_box(0), lean_box(0), v_getCurrMacroScope_1136_, v___f_1139_);
return v___x_1140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6(lean_object* v_info_1142_, lean_object* v_num_1143_, lean_object* v_toPure_1144_, lean_object* v_quotCtx_1145_){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; 
v___x_1146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3));
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1149_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___closed__0));
lean_inc_n(v_info_1142_, 4);
v___x_1150_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1150_, 0, v_info_1142_);
lean_ctor_set(v___x_1150_, 1, v___x_1149_);
v___x_1151_ = l_Lean_Syntax_node1(v_info_1142_, v___x_1148_, v___x_1150_);
v___x_1152_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_1153_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1153_, 0, v_info_1142_);
lean_ctor_set(v___x_1153_, 1, v___x_1148_);
lean_ctor_set(v___x_1153_, 2, v___x_1152_);
v___x_1154_ = l_Lean_Syntax_node3(v_info_1142_, v___x_1147_, v___x_1151_, v___x_1153_, v_num_1143_);
v___x_1155_ = l_Lean_Syntax_node1(v_info_1142_, v___x_1146_, v___x_1154_);
v___x_1156_ = lean_apply_2(v_toPure_1144_, lean_box(0), v___x_1155_);
return v___x_1156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___boxed(lean_object* v_info_1157_, lean_object* v_num_1158_, lean_object* v_toPure_1159_, lean_object* v_quotCtx_1160_){
_start:
{
lean_object* v_res_1161_; 
v_res_1161_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6(v_info_1157_, v_num_1158_, v_toPure_1159_, v_quotCtx_1160_);
lean_dec(v_quotCtx_1160_);
return v_res_1161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__8(lean_object* v_num_1162_, lean_object* v_toPure_1163_, lean_object* v_toBind_1164_, lean_object* v_getContext_1165_, lean_object* v_getCurrMacroScope_1166_, lean_object* v_info_1167_){
_start:
{
lean_object* v___f_1168_; lean_object* v___f_1169_; lean_object* v___x_1170_; 
v___f_1168_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___boxed), 4, 3);
lean_closure_set(v___f_1168_, 0, v_info_1167_);
lean_closure_set(v___f_1168_, 1, v_num_1162_);
lean_closure_set(v___f_1168_, 2, v_toPure_1163_);
lean_inc(v_toBind_1164_);
v___f_1169_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_1169_, 0, v_toBind_1164_);
lean_closure_set(v___f_1169_, 1, v_getContext_1165_);
lean_closure_set(v___f_1169_, 2, v___f_1168_);
v___x_1170_ = lean_apply_4(v_toBind_1164_, lean_box(0), lean_box(0), v_getCurrMacroScope_1166_, v___f_1169_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__7(lean_object* v_arg_1171_, uint8_t v_all_1172_, lean_object* v_toPure_1173_, lean_object* v_toBind_1174_, lean_object* v_getContext_1175_, lean_object* v_getCurrMacroScope_1176_, lean_object* v___x_1177_, lean_object* v___f_1178_, lean_object* v___f_1179_, lean_object* v_____do__lift_1180_){
_start:
{
lean_object* v___x_1181_; lean_object* v_num_1182_; 
v___x_1181_ = l_Nat_reprFast(v_arg_1171_);
v_num_1182_ = l_Lean_Syntax_mkNumLit(v___x_1181_, v_____do__lift_1180_);
if (v_all_1172_ == 0)
{
lean_object* v___f_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; 
lean_dec(v___f_1179_);
lean_inc_n(v_toBind_1174_, 2);
v___f_1183_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__5), 6, 5);
lean_closure_set(v___f_1183_, 0, v_num_1182_);
lean_closure_set(v___f_1183_, 1, v_toPure_1173_);
lean_closure_set(v___f_1183_, 2, v_toBind_1174_);
lean_closure_set(v___f_1183_, 3, v_getContext_1175_);
lean_closure_set(v___f_1183_, 4, v_getCurrMacroScope_1176_);
v___x_1184_ = lean_apply_4(v_toBind_1174_, lean_box(0), lean_box(0), v___x_1177_, v___f_1183_);
v___x_1185_ = lean_apply_4(v_toBind_1174_, lean_box(0), lean_box(0), v___x_1184_, v___f_1178_);
return v___x_1185_;
}
else
{
lean_object* v___f_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; 
lean_dec(v___f_1178_);
lean_inc_n(v_toBind_1174_, 2);
v___f_1186_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__8), 6, 5);
lean_closure_set(v___f_1186_, 0, v_num_1182_);
lean_closure_set(v___f_1186_, 1, v_toPure_1173_);
lean_closure_set(v___f_1186_, 2, v_toBind_1174_);
lean_closure_set(v___f_1186_, 3, v_getContext_1175_);
lean_closure_set(v___f_1186_, 4, v_getCurrMacroScope_1176_);
v___x_1187_ = lean_apply_4(v_toBind_1174_, lean_box(0), lean_box(0), v___x_1177_, v___f_1186_);
v___x_1188_ = lean_apply_4(v_toBind_1174_, lean_box(0), lean_box(0), v___x_1187_, v___f_1179_);
return v___x_1188_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__7___boxed(lean_object* v_arg_1189_, lean_object* v_all_1190_, lean_object* v_toPure_1191_, lean_object* v_toBind_1192_, lean_object* v_getContext_1193_, lean_object* v_getCurrMacroScope_1194_, lean_object* v___x_1195_, lean_object* v___f_1196_, lean_object* v___f_1197_, lean_object* v_____do__lift_1198_){
_start:
{
uint8_t v_all_3051__boxed_1199_; lean_object* v_res_1200_; 
v_all_3051__boxed_1199_ = lean_unbox(v_all_1190_);
v_res_1200_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__7(v_arg_1189_, v_all_3051__boxed_1199_, v_toPure_1191_, v_toBind_1192_, v_getContext_1193_, v_getCurrMacroScope_1194_, v___x_1195_, v___f_1196_, v___f_1197_, v_____do__lift_1198_);
return v_res_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__2(lean_object* v___f_1201_, lean_object* v_arg_1202_){
_start:
{
lean_object* v___x_1203_; 
v___x_1203_ = lean_apply_1(v___f_1201_, v_arg_1202_);
return v___x_1203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg(lean_object* v_inst_1205_, lean_object* v_inst_1206_, lean_object* v_convStx_1207_, lean_object* v_path_1208_, lean_object* v_loc_1209_, lean_object* v_xs_1210_){
_start:
{
switch(lean_obj_tag(v_path_1208_))
{
case 0:
{
lean_object* v_toApplicative_1211_; lean_object* v_toMonadRef_1212_; lean_object* v_toBind_1213_; lean_object* v_getCurrMacroScope_1214_; lean_object* v_getContext_1215_; lean_object* v_toPure_1216_; lean_object* v_arg_1217_; uint8_t v_all_1218_; lean_object* v_next_1219_; lean_object* v_getRef_1220_; lean_object* v___f_1221_; lean_object* v___f_1222_; lean_object* v___f_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___f_1226_; lean_object* v___x_1227_; 
v_toApplicative_1211_ = lean_ctor_get(v_inst_1205_, 0);
v_toMonadRef_1212_ = lean_ctor_get(v_inst_1206_, 0);
v_toBind_1213_ = lean_ctor_get(v_inst_1205_, 1);
lean_inc_n(v_toBind_1213_, 3);
v_getCurrMacroScope_1214_ = lean_ctor_get(v_inst_1206_, 1);
lean_inc(v_getCurrMacroScope_1214_);
v_getContext_1215_ = lean_ctor_get(v_inst_1206_, 2);
lean_inc(v_getContext_1215_);
v_toPure_1216_ = lean_ctor_get(v_toApplicative_1211_, 1);
lean_inc_n(v_toPure_1216_, 2);
v_arg_1217_ = lean_ctor_get(v_path_1208_, 0);
lean_inc(v_arg_1217_);
v_all_1218_ = lean_ctor_get_uint8(v_path_1208_, sizeof(void*)*2);
v_next_1219_ = lean_ctor_get(v_path_1208_, 1);
lean_inc_ref(v_next_1219_);
lean_dec_ref_known(v_path_1208_, 2);
v_getRef_1220_ = lean_ctor_get(v_toMonadRef_1212_, 0);
lean_inc(v_getRef_1220_);
v___f_1221_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1221_, 0, v_toPure_1216_);
v___f_1222_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1), 7, 6);
lean_closure_set(v___f_1222_, 0, v_xs_1210_);
lean_closure_set(v___f_1222_, 1, v_inst_1205_);
lean_closure_set(v___f_1222_, 2, v_inst_1206_);
lean_closure_set(v___f_1222_, 3, v_convStx_1207_);
lean_closure_set(v___f_1222_, 4, v_next_1219_);
lean_closure_set(v___f_1222_, 5, v_loc_1209_);
v___f_1223_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1223_, 0, v___f_1222_);
v___x_1224_ = lean_apply_4(v_toBind_1213_, lean_box(0), lean_box(0), v_getRef_1220_, v___f_1221_);
v___x_1225_ = lean_box(v_all_1218_);
lean_inc_ref(v___f_1223_);
lean_inc(v___x_1224_);
v___f_1226_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__7___boxed), 10, 9);
lean_closure_set(v___f_1226_, 0, v_arg_1217_);
lean_closure_set(v___f_1226_, 1, v___x_1225_);
lean_closure_set(v___f_1226_, 2, v_toPure_1216_);
lean_closure_set(v___f_1226_, 3, v_toBind_1213_);
lean_closure_set(v___f_1226_, 4, v_getContext_1215_);
lean_closure_set(v___f_1226_, 5, v_getCurrMacroScope_1214_);
lean_closure_set(v___f_1226_, 6, v___x_1224_);
lean_closure_set(v___f_1226_, 7, v___f_1223_);
lean_closure_set(v___f_1226_, 8, v___f_1223_);
v___x_1227_ = lean_apply_4(v_toBind_1213_, lean_box(0), lean_box(0), v___x_1224_, v___f_1226_);
return v___x_1227_;
}
case 1:
{
lean_object* v_toApplicative_1228_; lean_object* v_toBind_1229_; lean_object* v_toMonadRef_1230_; lean_object* v_getCurrMacroScope_1231_; lean_object* v_getContext_1232_; lean_object* v_toPure_1233_; lean_object* v_depth_1234_; lean_object* v___f_1235_; lean_object* v___f_1236_; lean_object* v___f_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; uint8_t v___x_1240_; lean_object* v___y_1242_; 
v_toApplicative_1228_ = lean_ctor_get(v_inst_1205_, 0);
v_toBind_1229_ = lean_ctor_get(v_inst_1205_, 1);
lean_inc_n(v_toBind_1229_, 3);
v_toMonadRef_1230_ = lean_ctor_get(v_inst_1206_, 0);
lean_inc_ref(v_toMonadRef_1230_);
v_getCurrMacroScope_1231_ = lean_ctor_get(v_inst_1206_, 1);
lean_inc_n(v_getCurrMacroScope_1231_, 3);
v_getContext_1232_ = lean_ctor_get(v_inst_1206_, 2);
lean_inc_n(v_getContext_1232_, 3);
lean_dec_ref(v_inst_1206_);
v_toPure_1233_ = lean_ctor_get(v_toApplicative_1228_, 1);
lean_inc_n(v_toPure_1233_, 4);
v_depth_1234_ = lean_ctor_get(v_path_1208_, 0);
lean_inc(v_depth_1234_);
lean_dec_ref_known(v_path_1208_, 1);
v___f_1235_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1235_, 0, v_toPure_1233_);
lean_inc_ref(v_xs_1210_);
v___f_1236_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__11), 6, 5);
lean_closure_set(v___f_1236_, 0, v_xs_1210_);
lean_closure_set(v___f_1236_, 1, v_toPure_1233_);
lean_closure_set(v___f_1236_, 2, v_toBind_1229_);
lean_closure_set(v___f_1236_, 3, v_getContext_1232_);
lean_closure_set(v___f_1236_, 4, v_getCurrMacroScope_1231_);
v___f_1237_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__13), 5, 4);
lean_closure_set(v___f_1237_, 0, v_toPure_1233_);
lean_closure_set(v___f_1237_, 1, v_toBind_1229_);
lean_closure_set(v___f_1237_, 2, v_getContext_1232_);
lean_closure_set(v___f_1237_, 3, v_getCurrMacroScope_1231_);
v___x_1238_ = lean_array_get_size(v_xs_1210_);
lean_dec_ref(v_xs_1210_);
v___x_1239_ = lean_unsigned_to_nat(0u);
v___x_1240_ = lean_nat_dec_eq(v___x_1238_, v___x_1239_);
if (v___x_1240_ == 0)
{
lean_object* v___x_1252_; lean_object* v___x_1253_; 
v___x_1252_ = lean_unsigned_to_nat(2u);
v___x_1253_ = lean_nat_add(v_depth_1234_, v___x_1252_);
v___y_1242_ = v___x_1253_;
goto v___jp_1241_;
}
else
{
lean_object* v___x_1254_; lean_object* v___x_1255_; 
v___x_1254_ = lean_unsigned_to_nat(1u);
v___x_1255_ = lean_nat_add(v_depth_1234_, v___x_1254_);
v___y_1242_ = v___x_1255_;
goto v___jp_1241_;
}
v___jp_1241_:
{
lean_object* v_getRef_1243_; lean_object* v_arr_1244_; lean_object* v___x_1245_; lean_object* v___f_1246_; lean_object* v___f_1247_; lean_object* v___x_1248_; lean_object* v___f_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; 
v_getRef_1243_ = lean_ctor_get(v_toMonadRef_1230_, 0);
v_arr_1244_ = lean_mk_empty_array_with_capacity(v___y_1242_);
lean_dec(v___y_1242_);
lean_inc_n(v_toBind_1229_, 5);
lean_inc(v_getRef_1243_);
v___x_1245_ = lean_apply_4(v_toBind_1229_, lean_box(0), lean_box(0), v_getRef_1243_, v___f_1235_);
lean_inc_ref(v_inst_1205_);
lean_inc_n(v___x_1245_, 3);
lean_inc(v_getCurrMacroScope_1231_);
lean_inc(v_getContext_1232_);
lean_inc_n(v_toPure_1233_, 2);
v___f_1246_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__18), 9, 8);
lean_closure_set(v___f_1246_, 0, v_loc_1209_);
lean_closure_set(v___f_1246_, 1, v_toPure_1233_);
lean_closure_set(v___f_1246_, 2, v_toBind_1229_);
lean_closure_set(v___f_1246_, 3, v_getContext_1232_);
lean_closure_set(v___f_1246_, 4, v_getCurrMacroScope_1231_);
lean_closure_set(v___f_1246_, 5, v___x_1245_);
lean_closure_set(v___f_1246_, 6, v_inst_1205_);
lean_closure_set(v___f_1246_, 7, v_toMonadRef_1230_);
v___f_1247_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__20), 8, 7);
lean_closure_set(v___f_1247_, 0, v_convStx_1207_);
lean_closure_set(v___f_1247_, 1, v_toPure_1233_);
lean_closure_set(v___f_1247_, 2, v_toBind_1229_);
lean_closure_set(v___f_1247_, 3, v_getContext_1232_);
lean_closure_set(v___f_1247_, 4, v_getCurrMacroScope_1231_);
lean_closure_set(v___f_1247_, 5, v___x_1245_);
lean_closure_set(v___f_1247_, 6, v___f_1246_);
v___x_1248_ = lean_box(v___x_1240_);
v___f_1249_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__24___boxed), 11, 10);
lean_closure_set(v___f_1249_, 0, v_toPure_1233_);
lean_closure_set(v___f_1249_, 1, v___x_1239_);
lean_closure_set(v___f_1249_, 2, v_depth_1234_);
lean_closure_set(v___f_1249_, 3, v_inst_1205_);
lean_closure_set(v___f_1249_, 4, v_toBind_1229_);
lean_closure_set(v___f_1249_, 5, v___f_1247_);
lean_closure_set(v___f_1249_, 6, v___x_1248_);
lean_closure_set(v___f_1249_, 7, v_arr_1244_);
lean_closure_set(v___f_1249_, 8, v___x_1245_);
lean_closure_set(v___f_1249_, 9, v___f_1237_);
v___x_1250_ = lean_apply_4(v_toBind_1229_, lean_box(0), lean_box(0), v___x_1245_, v___f_1236_);
v___x_1251_ = lean_apply_4(v_toBind_1229_, lean_box(0), lean_box(0), v___x_1250_, v___f_1249_);
return v___x_1251_;
}
}
case 2:
{
lean_object* v_toApplicative_1256_; lean_object* v_toMonadRef_1257_; lean_object* v_toBind_1258_; lean_object* v_getCurrMacroScope_1259_; lean_object* v_getContext_1260_; lean_object* v_toPure_1261_; lean_object* v_next_1262_; lean_object* v_getRef_1263_; lean_object* v___f_1264_; lean_object* v___f_1265_; lean_object* v___f_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; 
v_toApplicative_1256_ = lean_ctor_get(v_inst_1205_, 0);
v_toMonadRef_1257_ = lean_ctor_get(v_inst_1206_, 0);
v_toBind_1258_ = lean_ctor_get(v_inst_1205_, 1);
lean_inc_n(v_toBind_1258_, 4);
v_getCurrMacroScope_1259_ = lean_ctor_get(v_inst_1206_, 1);
v_getContext_1260_ = lean_ctor_get(v_inst_1206_, 2);
v_toPure_1261_ = lean_ctor_get(v_toApplicative_1256_, 1);
v_next_1262_ = lean_ctor_get(v_path_1208_, 0);
lean_inc_ref(v_next_1262_);
lean_dec_ref_known(v_path_1208_, 1);
v_getRef_1263_ = lean_ctor_get(v_toMonadRef_1257_, 0);
lean_inc(v_getRef_1263_);
lean_inc_n(v_toPure_1261_, 2);
v___f_1264_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1264_, 0, v_toPure_1261_);
lean_inc(v_getCurrMacroScope_1259_);
lean_inc(v_getContext_1260_);
v___f_1265_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__27), 5, 4);
lean_closure_set(v___f_1265_, 0, v_toPure_1261_);
lean_closure_set(v___f_1265_, 1, v_toBind_1258_);
lean_closure_set(v___f_1265_, 2, v_getContext_1260_);
lean_closure_set(v___f_1265_, 3, v_getCurrMacroScope_1259_);
v___f_1266_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1), 7, 6);
lean_closure_set(v___f_1266_, 0, v_xs_1210_);
lean_closure_set(v___f_1266_, 1, v_inst_1205_);
lean_closure_set(v___f_1266_, 2, v_inst_1206_);
lean_closure_set(v___f_1266_, 3, v_convStx_1207_);
lean_closure_set(v___f_1266_, 4, v_next_1262_);
lean_closure_set(v___f_1266_, 5, v_loc_1209_);
v___x_1267_ = lean_apply_4(v_toBind_1258_, lean_box(0), lean_box(0), v_getRef_1263_, v___f_1264_);
v___x_1268_ = lean_apply_4(v_toBind_1258_, lean_box(0), lean_box(0), v___x_1267_, v___f_1265_);
v___x_1269_ = lean_apply_4(v_toBind_1258_, lean_box(0), lean_box(0), v___x_1268_, v___f_1266_);
return v___x_1269_;
}
default: 
{
lean_object* v_toApplicative_1270_; lean_object* v_toBind_1271_; lean_object* v_toMonadRef_1272_; lean_object* v_getCurrMacroScope_1273_; lean_object* v_getContext_1274_; lean_object* v_toPure_1275_; lean_object* v_name_1276_; lean_object* v_next_1277_; lean_object* v___f_1278_; lean_object* v___f_1279_; lean_object* v___f_1280_; lean_object* v___x_1281_; uint8_t v___x_1282_; 
v_toApplicative_1270_ = lean_ctor_get(v_inst_1205_, 0);
v_toBind_1271_ = lean_ctor_get(v_inst_1205_, 1);
lean_inc_n(v_toBind_1271_, 2);
v_toMonadRef_1272_ = lean_ctor_get(v_inst_1206_, 0);
lean_inc_ref_n(v_toMonadRef_1272_, 2);
v_getCurrMacroScope_1273_ = lean_ctor_get(v_inst_1206_, 1);
lean_inc_n(v_getCurrMacroScope_1273_, 2);
v_getContext_1274_ = lean_ctor_get(v_inst_1206_, 2);
lean_inc_n(v_getContext_1274_, 2);
v_toPure_1275_ = lean_ctor_get(v_toApplicative_1270_, 1);
v_name_1276_ = lean_ctor_get(v_path_1208_, 0);
lean_inc(v_name_1276_);
v_next_1277_ = lean_ctor_get(v_path_1208_, 1);
lean_inc_ref(v_next_1277_);
lean_dec_ref_known(v_path_1208_, 2);
lean_inc_n(v_toPure_1275_, 2);
v___f_1278_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1278_, 0, v_toPure_1275_);
lean_inc_ref(v_inst_1205_);
v___f_1279_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1), 7, 6);
lean_closure_set(v___f_1279_, 0, v_xs_1210_);
lean_closure_set(v___f_1279_, 1, v_inst_1205_);
lean_closure_set(v___f_1279_, 2, v_inst_1206_);
lean_closure_set(v___f_1279_, 3, v_convStx_1207_);
lean_closure_set(v___f_1279_, 4, v_next_1277_);
lean_closure_set(v___f_1279_, 5, v_loc_1209_);
lean_inc_ref(v___f_1278_);
v___f_1280_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__25), 8, 7);
lean_closure_set(v___f_1280_, 0, v_toMonadRef_1272_);
lean_closure_set(v___f_1280_, 1, v_toPure_1275_);
lean_closure_set(v___f_1280_, 2, v_toBind_1271_);
lean_closure_set(v___f_1280_, 3, v_getContext_1274_);
lean_closure_set(v___f_1280_, 4, v_getCurrMacroScope_1273_);
lean_closure_set(v___f_1280_, 5, v___f_1278_);
lean_closure_set(v___f_1280_, 6, v___f_1279_);
v___x_1281_ = l_Lean_Name_eraseMacroScopes(v_name_1276_);
lean_dec(v_name_1276_);
lean_inc(v___x_1281_);
v___x_1282_ = lp_mathlib_Lean_Name_willRoundTrip(v___x_1281_);
if (v___x_1282_ == 0)
{
lean_object* v_getRef_1283_; lean_object* v___f_1284_; lean_object* v___f_1285_; lean_object* v___x_1286_; lean_object* v___f_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; 
lean_inc_n(v_toPure_1275_, 2);
lean_dec(v___x_1281_);
lean_dec_ref(v___f_1278_);
lean_dec_ref(v_inst_1205_);
v_getRef_1283_ = lean_ctor_get(v_toMonadRef_1272_, 0);
lean_inc(v_getRef_1283_);
lean_dec_ref(v_toMonadRef_1272_);
lean_inc_n(v_toBind_1271_, 3);
v___f_1284_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__32), 5, 4);
lean_closure_set(v___f_1284_, 0, v_toPure_1275_);
lean_closure_set(v___f_1284_, 1, v_toBind_1271_);
lean_closure_set(v___f_1284_, 2, v_getContext_1274_);
lean_closure_set(v___f_1284_, 3, v_getCurrMacroScope_1273_);
v___f_1285_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__31), 2, 1);
lean_closure_set(v___f_1285_, 0, v___f_1280_);
v___x_1286_ = lean_box(v___x_1282_);
v___f_1287_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__33___boxed), 3, 2);
lean_closure_set(v___f_1287_, 0, v___x_1286_);
lean_closure_set(v___f_1287_, 1, v_toPure_1275_);
v___x_1288_ = lean_apply_4(v_toBind_1271_, lean_box(0), lean_box(0), v_getRef_1283_, v___f_1287_);
v___x_1289_ = lean_apply_4(v_toBind_1271_, lean_box(0), lean_box(0), v___x_1288_, v___f_1284_);
v___x_1290_ = lean_apply_4(v_toBind_1271_, lean_box(0), lean_box(0), v___x_1289_, v___f_1285_);
return v___x_1290_;
}
else
{
lean_object* v___f_1291_; lean_object* v___f_1292_; uint8_t v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v___f_1291_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__31), 2, 1);
lean_closure_set(v___f_1291_, 0, v___f_1280_);
lean_inc(v_toBind_1271_);
lean_inc(v_toPure_1275_);
lean_inc_ref(v_toMonadRef_1272_);
v___f_1292_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__34), 8, 7);
lean_closure_set(v___f_1292_, 0, v_toMonadRef_1272_);
lean_closure_set(v___f_1292_, 1, v_toPure_1275_);
lean_closure_set(v___f_1292_, 2, v_toBind_1271_);
lean_closure_set(v___f_1292_, 3, v_getContext_1274_);
lean_closure_set(v___f_1292_, 4, v_getCurrMacroScope_1273_);
lean_closure_set(v___f_1292_, 5, v___f_1278_);
lean_closure_set(v___f_1292_, 6, v___f_1291_);
v___x_1293_ = 0;
v___x_1294_ = l_Lean_mkIdentFromRef___redArg(v_inst_1205_, v_toMonadRef_1272_, v___x_1281_, v___x_1293_);
v___x_1295_ = lean_apply_4(v_toBind_1271_, lean_box(0), lean_box(0), v___x_1294_, v___f_1292_);
return v___x_1295_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1(lean_object* v_xs_1296_, lean_object* v_inst_1297_, lean_object* v_inst_1298_, lean_object* v_convStx_1299_, lean_object* v_next_1300_, lean_object* v_loc_1301_, lean_object* v_arg_1302_){
_start:
{
lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; 
v___x_1303_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0));
v___x_1304_ = l_Lean_Syntax_TSepArray_push___redArg(v___x_1303_, v_xs_1296_, v_arg_1302_);
v___x_1305_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg(v_inst_1297_, v_inst_1298_, v_convStx_1299_, v_next_1300_, v_loc_1301_, v___x_1304_);
return v___x_1305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx(lean_object* v_m_1306_, lean_object* v_inst_1307_, lean_object* v_inst_1308_, lean_object* v_convStx_1309_, lean_object* v_path_1310_, lean_object* v_loc_1311_, lean_object* v_xs_1312_){
_start:
{
lean_object* v___x_1313_; 
v___x_1313_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg(v_inst_1307_, v_inst_1308_, v_convStx_1309_, v_path_1310_, v_loc_1311_, v_xs_1312_);
return v___x_1313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg(lean_object* v_e_1314_, lean_object* v___y_1315_){
_start:
{
uint8_t v___x_1317_; 
v___x_1317_ = l_Lean_Expr_hasMVar(v_e_1314_);
if (v___x_1317_ == 0)
{
lean_object* v___x_1318_; 
v___x_1318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1318_, 0, v_e_1314_);
return v___x_1318_;
}
else
{
lean_object* v___x_1319_; lean_object* v_mctx_1320_; lean_object* v___x_1321_; lean_object* v_fst_1322_; lean_object* v_snd_1323_; lean_object* v___x_1324_; lean_object* v_cache_1325_; lean_object* v_zetaDeltaFVarIds_1326_; lean_object* v_postponed_1327_; lean_object* v_diag_1328_; lean_object* v___x_1330_; uint8_t v_isShared_1331_; uint8_t v_isSharedCheck_1337_; 
v___x_1319_ = lean_st_ref_get(v___y_1315_);
v_mctx_1320_ = lean_ctor_get(v___x_1319_, 0);
lean_inc_ref(v_mctx_1320_);
lean_dec(v___x_1319_);
v___x_1321_ = l_Lean_instantiateMVarsCore(v_mctx_1320_, v_e_1314_);
v_fst_1322_ = lean_ctor_get(v___x_1321_, 0);
lean_inc(v_fst_1322_);
v_snd_1323_ = lean_ctor_get(v___x_1321_, 1);
lean_inc(v_snd_1323_);
lean_dec_ref(v___x_1321_);
v___x_1324_ = lean_st_ref_take(v___y_1315_);
v_cache_1325_ = lean_ctor_get(v___x_1324_, 1);
v_zetaDeltaFVarIds_1326_ = lean_ctor_get(v___x_1324_, 2);
v_postponed_1327_ = lean_ctor_get(v___x_1324_, 3);
v_diag_1328_ = lean_ctor_get(v___x_1324_, 4);
v_isSharedCheck_1337_ = !lean_is_exclusive(v___x_1324_);
if (v_isSharedCheck_1337_ == 0)
{
lean_object* v_unused_1338_; 
v_unused_1338_ = lean_ctor_get(v___x_1324_, 0);
lean_dec(v_unused_1338_);
v___x_1330_ = v___x_1324_;
v_isShared_1331_ = v_isSharedCheck_1337_;
goto v_resetjp_1329_;
}
else
{
lean_inc(v_diag_1328_);
lean_inc(v_postponed_1327_);
lean_inc(v_zetaDeltaFVarIds_1326_);
lean_inc(v_cache_1325_);
lean_dec(v___x_1324_);
v___x_1330_ = lean_box(0);
v_isShared_1331_ = v_isSharedCheck_1337_;
goto v_resetjp_1329_;
}
v_resetjp_1329_:
{
lean_object* v___x_1333_; 
if (v_isShared_1331_ == 0)
{
lean_ctor_set(v___x_1330_, 0, v_snd_1323_);
v___x_1333_ = v___x_1330_;
goto v_reusejp_1332_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v_snd_1323_);
lean_ctor_set(v_reuseFailAlloc_1336_, 1, v_cache_1325_);
lean_ctor_set(v_reuseFailAlloc_1336_, 2, v_zetaDeltaFVarIds_1326_);
lean_ctor_set(v_reuseFailAlloc_1336_, 3, v_postponed_1327_);
lean_ctor_set(v_reuseFailAlloc_1336_, 4, v_diag_1328_);
v___x_1333_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1332_;
}
v_reusejp_1332_:
{
lean_object* v___x_1334_; lean_object* v___x_1335_; 
v___x_1334_ = lean_st_ref_set(v___y_1315_, v___x_1333_);
v___x_1335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1335_, 0, v_fst_1322_);
return v___x_1335_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg___boxed(lean_object* v_e_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v_res_1342_; 
v_res_1342_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg(v_e_1339_, v___y_1340_);
lean_dec(v___y_1340_);
return v_res_1342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0(lean_object* v_e_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_){
_start:
{
lean_object* v___x_1349_; 
v___x_1349_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg(v_e_1343_, v___y_1345_);
return v___x_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___boxed(lean_object* v_e_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_){
_start:
{
lean_object* v_res_1356_; 
v_res_1356_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0(v_e_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
lean_dec(v___y_1354_);
lean_dec_ref(v___y_1353_);
lean_dec(v___y_1352_);
lean_dec_ref(v___y_1351_);
return v_res_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg(lean_object* v___x_1357_, lean_object* v_range_1358_, lean_object* v_b_1359_, lean_object* v_i_1360_){
_start:
{
lean_object* v_stop_1362_; lean_object* v_step_1363_; uint8_t v___x_1364_; 
v_stop_1362_ = lean_ctor_get(v_range_1358_, 1);
v_step_1363_ = lean_ctor_get(v_range_1358_, 2);
v___x_1364_ = lean_nat_dec_lt(v_i_1360_, v_stop_1362_);
if (v___x_1364_ == 0)
{
lean_object* v___x_1365_; 
lean_dec(v_i_1360_);
lean_dec(v___x_1357_);
v___x_1365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1365_, 0, v_b_1359_);
return v___x_1365_;
}
else
{
lean_object* v___x_1366_; lean_object* v___x_1367_; 
lean_inc(v___x_1357_);
v___x_1366_ = lean_array_push(v_b_1359_, v___x_1357_);
v___x_1367_ = lean_nat_add(v_i_1360_, v_step_1363_);
lean_dec(v_i_1360_);
v_b_1359_ = v___x_1366_;
v_i_1360_ = v___x_1367_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg___boxed(lean_object* v___x_1369_, lean_object* v_range_1370_, lean_object* v_b_1371_, lean_object* v_i_1372_, lean_object* v___y_1373_){
_start:
{
lean_object* v_res_1374_; 
v_res_1374_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg(v___x_1369_, v_range_1370_, v_b_1371_, v_i_1372_);
lean_dec_ref(v_range_1370_);
return v_res_1374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_){
_start:
{
lean_object* v_ref_1380_; uint8_t v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; 
v_ref_1380_ = lean_ctor_get(v___y_1377_, 5);
v___x_1381_ = 0;
v___x_1382_ = l_Lean_SourceInfo_fromRef(v_ref_1380_, v___x_1381_);
v___x_1383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1383_, 0, v___x_1382_);
return v___x_1383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0___boxed(lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_){
_start:
{
lean_object* v_res_1389_; 
v_res_1389_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1384_, v___y_1385_, v___y_1386_, v___y_1387_);
lean_dec(v___y_1387_);
lean_dec_ref(v___y_1386_);
lean_dec(v___y_1385_);
lean_dec_ref(v___y_1384_);
return v_res_1389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg(lean_object* v_val_1390_, uint8_t v_canonical_1391_, lean_object* v___y_1392_){
_start:
{
lean_object* v_ref_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; 
v_ref_1394_ = lean_ctor_get(v___y_1392_, 5);
v___x_1395_ = l_Lean_mkIdentFrom(v_ref_1394_, v_val_1390_, v_canonical_1391_);
v___x_1396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1396_, 0, v___x_1395_);
return v___x_1396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg___boxed(lean_object* v_val_1397_, lean_object* v_canonical_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_){
_start:
{
uint8_t v_canonical_boxed_1401_; lean_object* v_res_1402_; 
v_canonical_boxed_1401_ = lean_unbox(v_canonical_1398_);
v_res_1402_ = lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg(v_val_1397_, v_canonical_boxed_1401_, v___y_1399_);
lean_dec_ref(v___y_1399_);
return v_res_1402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1(lean_object* v_convStx_1403_, lean_object* v_path_1404_, lean_object* v_loc_1405_, lean_object* v_xs_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_){
_start:
{
switch(lean_obj_tag(v_path_1404_))
{
case 0:
{
lean_object* v_arg_1412_; uint8_t v_all_1413_; lean_object* v_next_1414_; lean_object* v_arg_1416_; lean_object* v___y_1417_; lean_object* v___y_1418_; lean_object* v___y_1419_; lean_object* v___y_1420_; lean_object* v___x_1424_; lean_object* v_a_1425_; lean_object* v___x_1426_; lean_object* v_num_1427_; 
v_arg_1412_ = lean_ctor_get(v_path_1404_, 0);
lean_inc(v_arg_1412_);
v_all_1413_ = lean_ctor_get_uint8(v_path_1404_, sizeof(void*)*2);
v_next_1414_ = lean_ctor_get(v_path_1404_, 1);
lean_inc_ref(v_next_1414_);
lean_dec_ref_known(v_path_1404_, 2);
v___x_1424_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
v_a_1425_ = lean_ctor_get(v___x_1424_, 0);
lean_inc(v_a_1425_);
lean_dec_ref(v___x_1424_);
v___x_1426_ = l_Nat_reprFast(v_arg_1412_);
v_num_1427_ = l_Lean_Syntax_mkNumLit(v___x_1426_, v_a_1425_);
if (v_all_1413_ == 0)
{
lean_object* v_ref_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; 
v_ref_1428_ = lean_ctor_get(v___y_1409_, 5);
v___x_1429_ = l_Lean_SourceInfo_fromRef(v_ref_1428_, v_all_1413_);
v___x_1430_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1431_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3));
v___x_1432_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1433_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
lean_inc_n(v___x_1429_, 2);
v___x_1434_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1434_, 0, v___x_1429_);
lean_ctor_set(v___x_1434_, 1, v___x_1432_);
lean_ctor_set(v___x_1434_, 2, v___x_1433_);
lean_inc_ref(v___x_1434_);
v___x_1435_ = l_Lean_Syntax_node3(v___x_1429_, v___x_1431_, v___x_1434_, v___x_1434_, v_num_1427_);
v___x_1436_ = l_Lean_Syntax_node1(v___x_1429_, v___x_1430_, v___x_1435_);
v_arg_1416_ = v___x_1436_;
v___y_1417_ = v___y_1407_;
v___y_1418_ = v___y_1408_;
v___y_1419_ = v___y_1409_;
v___y_1420_ = v___y_1410_;
goto v___jp_1415_;
}
else
{
lean_object* v___x_1437_; lean_object* v_a_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; 
v___x_1437_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
v_a_1438_ = lean_ctor_get(v___x_1437_, 0);
lean_inc_n(v_a_1438_, 5);
lean_dec_ref(v___x_1437_);
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1440_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3));
v___x_1441_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__6___closed__0));
v___x_1443_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1443_, 0, v_a_1438_);
lean_ctor_set(v___x_1443_, 1, v___x_1442_);
v___x_1444_ = l_Lean_Syntax_node1(v_a_1438_, v___x_1441_, v___x_1443_);
v___x_1445_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_1446_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1446_, 0, v_a_1438_);
lean_ctor_set(v___x_1446_, 1, v___x_1441_);
lean_ctor_set(v___x_1446_, 2, v___x_1445_);
v___x_1447_ = l_Lean_Syntax_node3(v_a_1438_, v___x_1440_, v___x_1444_, v___x_1446_, v_num_1427_);
v___x_1448_ = l_Lean_Syntax_node1(v_a_1438_, v___x_1439_, v___x_1447_);
v_arg_1416_ = v___x_1448_;
v___y_1417_ = v___y_1407_;
v___y_1418_ = v___y_1408_;
v___y_1419_ = v___y_1409_;
v___y_1420_ = v___y_1410_;
goto v___jp_1415_;
}
v___jp_1415_:
{
lean_object* v___x_1421_; lean_object* v___x_1422_; 
v___x_1421_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0));
v___x_1422_ = l_Lean_Syntax_TSepArray_push___redArg(v___x_1421_, v_xs_1406_, v_arg_1416_);
v_path_1404_ = v_next_1414_;
v_xs_1406_ = v___x_1422_;
v___y_1407_ = v___y_1417_;
v___y_1408_ = v___y_1418_;
v___y_1409_ = v___y_1419_;
v___y_1410_ = v___y_1420_;
goto _start;
}
}
case 1:
{
lean_object* v_depth_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___y_1453_; lean_object* v___y_1454_; lean_object* v___y_1455_; lean_object* v___y_1456_; lean_object* v___y_1457_; lean_object* v___y_1458_; lean_object* v___y_1459_; lean_object* v___y_1460_; lean_object* v_arr_1461_; uint8_t v___x_1519_; lean_object* v___y_1521_; 
v_depth_1449_ = lean_ctor_get(v_path_1404_, 0);
lean_inc(v_depth_1449_);
lean_dec_ref_known(v_path_1404_, 1);
v___x_1450_ = lean_array_get_size(v_xs_1406_);
v___x_1451_ = lean_unsigned_to_nat(0u);
v___x_1519_ = lean_nat_dec_eq(v___x_1450_, v___x_1451_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1549_; lean_object* v___x_1550_; 
v___x_1549_ = lean_unsigned_to_nat(2u);
v___x_1550_ = lean_nat_add(v_depth_1449_, v___x_1549_);
v___y_1521_ = v___x_1550_;
goto v___jp_1520_;
}
else
{
lean_object* v___x_1551_; lean_object* v___x_1552_; 
v___x_1551_ = lean_unsigned_to_nat(1u);
v___x_1552_ = lean_nat_add(v_depth_1449_, v___x_1551_);
v___y_1521_ = v___x_1552_;
goto v___jp_1520_;
}
v___jp_1452_:
{
lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v_a_1465_; lean_object* v___x_1466_; lean_object* v_a_1467_; lean_object* v_arr_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; 
v___x_1462_ = lean_unsigned_to_nat(1u);
v___x_1463_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1463_, 0, v___x_1451_);
lean_ctor_set(v___x_1463_, 1, v_depth_1449_);
lean_ctor_set(v___x_1463_, 2, v___x_1462_);
v___x_1464_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg(v___y_1460_, v___x_1463_, v_arr_1461_, v___x_1451_);
lean_dec_ref_known(v___x_1463_, 3);
v_a_1465_ = lean_ctor_get(v___x_1464_, 0);
lean_inc(v_a_1465_);
lean_dec_ref(v___x_1464_);
v___x_1466_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
v_a_1467_ = lean_ctor_get(v___x_1466_, 0);
lean_inc_n(v_a_1467_, 2);
lean_dec_ref(v___x_1466_);
v_arr_1468_ = lean_array_push(v_a_1465_, v_convStx_1403_);
v___x_1469_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__0));
lean_inc_ref(v___y_1459_);
lean_inc_ref(v___y_1458_);
lean_inc_ref(v___y_1454_);
lean_inc_ref(v___y_1453_);
v___x_1470_ = l_Lean_Name_mkStr5(v___y_1453_, v___y_1454_, v___y_1458_, v___y_1459_, v___x_1469_);
v___x_1471_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__19___closed__2));
v___x_1472_ = l_Lean_Syntax_SepArray_ofElems(v___x_1471_, v_arr_1468_);
lean_dec_ref(v_arr_1468_);
lean_inc_ref(v___y_1457_);
v___x_1473_ = l_Array_append___redArg(v___y_1457_, v___x_1472_);
lean_dec_ref(v___x_1472_);
lean_inc(v___y_1456_);
v___x_1474_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1474_, 0, v_a_1467_);
lean_ctor_set(v___x_1474_, 1, v___y_1456_);
lean_ctor_set(v___x_1474_, 2, v___x_1473_);
v___x_1475_ = l_Lean_Syntax_node1(v_a_1467_, v___x_1470_, v___x_1474_);
if (lean_obj_tag(v_loc_1405_) == 0)
{
lean_object* v___x_1476_; lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1493_; 
v___x_1476_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
v_a_1477_ = lean_ctor_get(v___x_1476_, 0);
v_isSharedCheck_1493_ = !lean_is_exclusive(v___x_1476_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1479_ = v___x_1476_;
v_isShared_1480_ = v_isSharedCheck_1493_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1476_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1493_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1491_; 
lean_inc_ref_n(v___y_1455_, 2);
lean_inc_ref_n(v___y_1459_, 2);
lean_inc_ref_n(v___y_1458_, 2);
lean_inc_ref_n(v___y_1454_, 2);
lean_inc_ref_n(v___y_1453_, 2);
v___x_1481_ = l_Lean_Name_mkStr5(v___y_1453_, v___y_1454_, v___y_1458_, v___y_1459_, v___y_1455_);
lean_inc_n(v_a_1477_, 4);
v___x_1482_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1482_, 0, v_a_1477_);
lean_ctor_set(v___x_1482_, 1, v___y_1455_);
lean_inc(v___y_1456_);
v___x_1483_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1483_, 0, v_a_1477_);
lean_ctor_set(v___x_1483_, 1, v___y_1456_);
lean_ctor_set(v___x_1483_, 2, v___y_1457_);
v___x_1484_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9));
v___x_1485_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1485_, 0, v_a_1477_);
lean_ctor_set(v___x_1485_, 1, v___x_1484_);
v___x_1486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__10));
v___x_1487_ = l_Lean_Name_mkStr5(v___y_1453_, v___y_1454_, v___y_1458_, v___y_1459_, v___x_1486_);
v___x_1488_ = l_Lean_Syntax_node1(v_a_1477_, v___x_1487_, v___x_1475_);
lean_inc_ref(v___x_1483_);
v___x_1489_ = l_Lean_Syntax_node5(v_a_1477_, v___x_1481_, v___x_1482_, v___x_1483_, v___x_1483_, v___x_1485_, v___x_1488_);
if (v_isShared_1480_ == 0)
{
lean_ctor_set(v___x_1479_, 0, v___x_1489_);
v___x_1491_ = v___x_1479_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v___x_1489_);
v___x_1491_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
return v___x_1491_;
}
}
}
else
{
lean_object* v_val_1494_; uint8_t v___x_1495_; lean_object* v___x_1496_; lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1518_; 
v_val_1494_ = lean_ctor_get(v_loc_1405_, 0);
lean_inc(v_val_1494_);
lean_dec_ref_known(v_loc_1405_, 1);
v___x_1495_ = 0;
v___x_1496_ = lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg(v_val_1494_, v___x_1495_, v___y_1409_);
v_a_1497_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1518_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1518_ == 0)
{
v___x_1499_ = v___x_1496_;
v_isShared_1500_ = v_isSharedCheck_1518_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1496_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1518_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v_ref_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1516_; 
v_ref_1501_ = lean_ctor_get(v___y_1409_, 5);
v___x_1502_ = l_Lean_SourceInfo_fromRef(v_ref_1501_, v___x_1495_);
lean_inc_ref_n(v___y_1455_, 2);
lean_inc_ref_n(v___y_1459_, 2);
lean_inc_ref_n(v___y_1458_, 2);
lean_inc_ref_n(v___y_1454_, 2);
lean_inc_ref_n(v___y_1453_, 2);
v___x_1503_ = l_Lean_Name_mkStr5(v___y_1453_, v___y_1454_, v___y_1458_, v___y_1459_, v___y_1455_);
lean_inc_n(v___x_1502_, 6);
v___x_1504_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1504_, 0, v___x_1502_);
lean_ctor_set(v___x_1504_, 1, v___y_1455_);
v___x_1505_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__7));
v___x_1506_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1506_, 0, v___x_1502_);
lean_ctor_set(v___x_1506_, 1, v___x_1505_);
lean_inc_n(v___y_1456_, 2);
v___x_1507_ = l_Lean_Syntax_node2(v___x_1502_, v___y_1456_, v___x_1506_, v_a_1497_);
v___x_1508_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1508_, 0, v___x_1502_);
lean_ctor_set(v___x_1508_, 1, v___y_1456_);
lean_ctor_set(v___x_1508_, 2, v___y_1457_);
v___x_1509_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__9));
v___x_1510_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1510_, 0, v___x_1502_);
lean_ctor_set(v___x_1510_, 1, v___x_1509_);
v___x_1511_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__10));
v___x_1512_ = l_Lean_Name_mkStr5(v___y_1453_, v___y_1454_, v___y_1458_, v___y_1459_, v___x_1511_);
v___x_1513_ = l_Lean_Syntax_node1(v___x_1502_, v___x_1512_, v___x_1475_);
v___x_1514_ = l_Lean_Syntax_node5(v___x_1502_, v___x_1503_, v___x_1504_, v___x_1507_, v___x_1508_, v___x_1510_, v___x_1513_);
if (v_isShared_1500_ == 0)
{
lean_ctor_set(v___x_1499_, 0, v___x_1514_);
v___x_1516_ = v___x_1499_;
goto v_reusejp_1515_;
}
else
{
lean_object* v_reuseFailAlloc_1517_; 
v_reuseFailAlloc_1517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1517_, 0, v___x_1514_);
v___x_1516_ = v_reuseFailAlloc_1517_;
goto v_reusejp_1515_;
}
v_reusejp_1515_:
{
return v___x_1516_;
}
}
}
}
v___jp_1520_:
{
lean_object* v___x_1522_; lean_object* v_a_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v_a_1538_; lean_object* v_arr_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; 
v___x_1522_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
v_a_1523_ = lean_ctor_get(v___x_1522_, 0);
lean_inc_n(v_a_1523_, 4);
lean_dec_ref(v___x_1522_);
v___x_1524_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_appT___closed__0));
v___x_1525_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__0));
v___x_1526_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__1));
v___x_1527_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__2));
v___x_1528_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__0));
v___x_1529_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__1));
v___x_1530_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1530_, 0, v_a_1523_);
lean_ctor_set(v___x_1530_, 1, v___x_1528_);
v___x_1531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__2));
v___x_1532_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1532_, 0, v_a_1523_);
lean_ctor_set(v___x_1532_, 1, v___x_1531_);
v___x_1533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1534_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
v___x_1535_ = l_Array_append___redArg(v___x_1534_, v_xs_1406_);
lean_dec_ref(v_xs_1406_);
v___x_1536_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1536_, 0, v_a_1523_);
lean_ctor_set(v___x_1536_, 1, v___x_1533_);
lean_ctor_set(v___x_1536_, 2, v___x_1535_);
v___x_1537_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___lam__0(v___y_1407_, v___y_1408_, v___y_1409_, v___y_1410_);
v_a_1538_ = lean_ctor_get(v___x_1537_, 0);
lean_inc_n(v_a_1538_, 2);
lean_dec_ref(v___x_1537_);
v_arr_1539_ = lean_mk_empty_array_with_capacity(v___y_1521_);
lean_dec(v___y_1521_);
v___x_1540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__3));
v___x_1541_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__0));
v___x_1542_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__9___closed__1));
v___x_1543_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1543_, 0, v_a_1538_);
lean_ctor_set(v___x_1543_, 1, v___x_1541_);
v___x_1544_ = l_Lean_Syntax_node1(v_a_1538_, v___x_1542_, v___x_1543_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v_arr_1548_; 
v___x_1545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__10___closed__3));
lean_inc(v_a_1523_);
v___x_1546_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1546_, 0, v_a_1523_);
lean_ctor_set(v___x_1546_, 1, v___x_1545_);
v___x_1547_ = l_Lean_Syntax_node4(v_a_1523_, v___x_1529_, v___x_1530_, v___x_1532_, v___x_1536_, v___x_1546_);
v_arr_1548_ = lean_array_push(v_arr_1539_, v___x_1547_);
v___y_1453_ = v___x_1524_;
v___y_1454_ = v___x_1525_;
v___y_1455_ = v___x_1540_;
v___y_1456_ = v___x_1533_;
v___y_1457_ = v___x_1534_;
v___y_1458_ = v___x_1526_;
v___y_1459_ = v___x_1527_;
v___y_1460_ = v___x_1544_;
v_arr_1461_ = v_arr_1548_;
goto v___jp_1452_;
}
else
{
lean_dec_ref_known(v___x_1536_, 3);
lean_dec_ref_known(v___x_1532_, 2);
lean_dec_ref_known(v___x_1530_, 2);
lean_dec(v_a_1523_);
v___y_1453_ = v___x_1524_;
v___y_1454_ = v___x_1525_;
v___y_1455_ = v___x_1540_;
v___y_1456_ = v___x_1533_;
v___y_1457_ = v___x_1534_;
v___y_1458_ = v___x_1526_;
v___y_1459_ = v___x_1527_;
v___y_1460_ = v___x_1544_;
v_arr_1461_ = v_arr_1539_;
goto v___jp_1452_;
}
}
}
case 2:
{
lean_object* v_next_1553_; lean_object* v_ref_1554_; uint8_t v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; 
v_next_1553_ = lean_ctor_get(v_path_1404_, 0);
lean_inc_ref(v_next_1553_);
lean_dec_ref_known(v_path_1404_, 1);
v_ref_1554_ = lean_ctor_get(v___y_1409_, 5);
v___x_1555_ = 0;
v___x_1556_ = l_Lean_SourceInfo_fromRef(v_ref_1554_, v___x_1555_);
v___x_1557_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1558_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__3));
v___x_1559_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__6));
v___x_1560_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__14___closed__8);
lean_inc_n(v___x_1556_, 4);
v___x_1561_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1561_, 0, v___x_1556_);
lean_ctor_set(v___x_1561_, 1, v___x_1559_);
lean_ctor_set(v___x_1561_, 2, v___x_1560_);
v___x_1562_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__5));
v___x_1563_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__6));
v___x_1564_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1564_, 0, v___x_1556_);
lean_ctor_set(v___x_1564_, 1, v___x_1563_);
v___x_1565_ = l_Lean_Syntax_node1(v___x_1556_, v___x_1562_, v___x_1564_);
lean_inc_ref(v___x_1561_);
v___x_1566_ = l_Lean_Syntax_node3(v___x_1556_, v___x_1558_, v___x_1561_, v___x_1561_, v___x_1565_);
v___x_1567_ = l_Lean_Syntax_node1(v___x_1556_, v___x_1557_, v___x_1566_);
v___x_1568_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0));
v___x_1569_ = l_Lean_Syntax_TSepArray_push___redArg(v___x_1568_, v_xs_1406_, v___x_1567_);
v_path_1404_ = v_next_1553_;
v_xs_1406_ = v___x_1569_;
goto _start;
}
default: 
{
lean_object* v_name_1571_; lean_object* v_next_1572_; lean_object* v___x_1574_; uint8_t v_isShared_1575_; uint8_t v_isSharedCheck_1609_; 
v_name_1571_ = lean_ctor_get(v_path_1404_, 0);
v_next_1572_ = lean_ctor_get(v_path_1404_, 1);
v_isSharedCheck_1609_ = !lean_is_exclusive(v_path_1404_);
if (v_isSharedCheck_1609_ == 0)
{
v___x_1574_ = v_path_1404_;
v_isShared_1575_ = v_isSharedCheck_1609_;
goto v_resetjp_1573_;
}
else
{
lean_inc(v_next_1572_);
lean_inc(v_name_1571_);
lean_dec(v_path_1404_);
v___x_1574_ = lean_box(0);
v_isShared_1575_ = v_isSharedCheck_1609_;
goto v_resetjp_1573_;
}
v_resetjp_1573_:
{
lean_object* v_bi_1577_; lean_object* v___y_1578_; lean_object* v___y_1579_; lean_object* v___y_1580_; lean_object* v_ref_1581_; lean_object* v___y_1582_; lean_object* v___x_1590_; uint8_t v___x_1591_; 
v___x_1590_ = l_Lean_Name_eraseMacroScopes(v_name_1571_);
lean_dec(v_name_1571_);
lean_inc(v___x_1590_);
v___x_1591_ = lp_mathlib_Lean_Name_willRoundTrip(v___x_1590_);
if (v___x_1591_ == 0)
{
lean_object* v_ref_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1598_; 
lean_dec(v___x_1590_);
v_ref_1592_ = lean_ctor_get(v___y_1409_, 5);
v___x_1593_ = l_Lean_SourceInfo_fromRef(v_ref_1592_, v___x_1591_);
v___x_1594_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1));
v___x_1595_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__4));
v___x_1596_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__5));
lean_inc(v___x_1593_);
if (v_isShared_1575_ == 0)
{
lean_ctor_set_tag(v___x_1574_, 2);
lean_ctor_set(v___x_1574_, 1, v___x_1596_);
lean_ctor_set(v___x_1574_, 0, v___x_1593_);
v___x_1598_ = v___x_1574_;
goto v_reusejp_1597_;
}
else
{
lean_object* v_reuseFailAlloc_1601_; 
v_reuseFailAlloc_1601_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1601_, 0, v___x_1593_);
lean_ctor_set(v_reuseFailAlloc_1601_, 1, v___x_1596_);
v___x_1598_ = v_reuseFailAlloc_1601_;
goto v_reusejp_1597_;
}
v_reusejp_1597_:
{
lean_object* v___x_1599_; lean_object* v___x_1600_; 
lean_inc(v___x_1593_);
v___x_1599_ = l_Lean_Syntax_node1(v___x_1593_, v___x_1595_, v___x_1598_);
v___x_1600_ = l_Lean_Syntax_node1(v___x_1593_, v___x_1594_, v___x_1599_);
v_bi_1577_ = v___x_1600_;
v___y_1578_ = v___y_1407_;
v___y_1579_ = v___y_1408_;
v___y_1580_ = v___y_1409_;
v_ref_1581_ = v_ref_1592_;
v___y_1582_ = v___y_1410_;
goto v___jp_1576_;
}
}
else
{
uint8_t v___x_1602_; lean_object* v___x_1603_; lean_object* v_a_1604_; lean_object* v_ref_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; 
lean_del_object(v___x_1574_);
v___x_1602_ = 0;
v___x_1603_ = lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg(v___x_1590_, v___x_1602_, v___y_1409_);
v_a_1604_ = lean_ctor_get(v___x_1603_, 0);
lean_inc(v_a_1604_);
lean_dec_ref(v___x_1603_);
v_ref_1605_ = lean_ctor_get(v___y_1409_, 5);
v___x_1606_ = l_Lean_SourceInfo_fromRef(v_ref_1605_, v___x_1602_);
v___x_1607_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__29___closed__1));
v___x_1608_ = l_Lean_Syntax_node1(v___x_1606_, v___x_1607_, v_a_1604_);
v_bi_1577_ = v___x_1608_;
v___y_1578_ = v___y_1407_;
v___y_1579_ = v___y_1408_;
v___y_1580_ = v___y_1409_;
v_ref_1581_ = v_ref_1605_;
v___y_1582_ = v___y_1410_;
goto v___jp_1576_;
}
v___jp_1576_:
{
uint8_t v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; 
v___x_1583_ = 0;
v___x_1584_ = l_Lean_SourceInfo_fromRef(v_ref_1581_, v___x_1583_);
v___x_1585_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__26___closed__1));
v___x_1586_ = l_Lean_Syntax_node1(v___x_1584_, v___x_1585_, v_bi_1577_);
v___x_1587_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_pathToStx___redArg___lam__1___closed__0));
v___x_1588_ = l_Lean_Syntax_TSepArray_push___redArg(v___x_1587_, v_xs_1406_, v___x_1586_);
v_path_1404_ = v_next_1572_;
v_xs_1406_ = v___x_1588_;
v___y_1407_ = v___y_1578_;
v___y_1408_ = v___y_1579_;
v___y_1409_ = v___y_1580_;
v___y_1410_ = v___y_1582_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1___boxed(lean_object* v_convStx_1610_, lean_object* v_path_1611_, lean_object* v_loc_1612_, lean_object* v_xs_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_, lean_object* v___y_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_){
_start:
{
lean_object* v_res_1619_; 
v_res_1619_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1(v_convStx_1610_, v_path_1611_, v_loc_1612_, v_xs_1613_, v___y_1614_, v___y_1615_, v___y_1616_, v___y_1617_);
lean_dec(v___y_1617_);
lean_dec_ref(v___y_1616_);
lean_dec(v___y_1615_);
lean_dec_ref(v___y_1614_);
return v_res_1619_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__4(void){
_start:
{
lean_object* v___x_1630_; lean_object* v___x_1631_; 
v___x_1630_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__3));
v___x_1631_ = l_Lean_stringToMessageData(v___x_1630_);
return v___x_1631_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__6(void){
_start:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1633_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__5));
v___x_1634_ = l_Lean_stringToMessageData(v___x_1633_);
return v___x_1634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax(lean_object* v_locations_1635_, lean_object* v_goalType_1636_, lean_object* v_a_1637_, lean_object* v_a_1638_, lean_object* v_a_1639_, lean_object* v_a_1640_){
_start:
{
lean_object* v_fst_1643_; lean_object* v_fst_1644_; lean_object* v_snd_1645_; lean_object* v___y_1646_; lean_object* v___y_1647_; lean_object* v___y_1648_; lean_object* v___y_1649_; lean_object* v___x_1687_; lean_object* v___x_1688_; uint8_t v___x_1689_; 
v___x_1687_ = lean_unsigned_to_nat(0u);
v___x_1688_ = lean_array_get_size(v_locations_1635_);
v___x_1689_ = lean_nat_dec_lt(v___x_1687_, v___x_1688_);
if (v___x_1689_ == 0)
{
lean_object* v___x_1690_; lean_object* v___x_1691_; 
lean_dec_ref(v_goalType_1636_);
v___x_1690_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__4, &lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__4);
v___x_1691_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_1690_, v_a_1637_, v_a_1638_, v_a_1639_, v_a_1640_);
return v___x_1691_;
}
else
{
lean_object* v___x_1692_; lean_object* v_loc_1693_; 
v___x_1692_ = lean_array_fget_borrowed(v_locations_1635_, v___x_1687_);
v_loc_1693_ = lean_ctor_get(v___x_1692_, 1);
switch(lean_obj_tag(v_loc_1693_))
{
case 3:
{
lean_object* v_a_1694_; lean_object* v___x_1695_; 
v_a_1694_ = lean_ctor_get(v_loc_1693_, 0);
v___x_1695_ = lean_box(0);
lean_inc(v_a_1694_);
v_fst_1643_ = v_goalType_1636_;
v_fst_1644_ = v_a_1694_;
v_snd_1645_ = v___x_1695_;
v___y_1646_ = v_a_1637_;
v___y_1647_ = v_a_1638_;
v___y_1648_ = v_a_1639_;
v___y_1649_ = v_a_1640_;
goto v___jp_1642_;
}
case 1:
{
lean_object* v_a_1696_; lean_object* v_a_1697_; lean_object* v___x_1698_; 
lean_dec_ref(v_goalType_1636_);
v_a_1696_ = lean_ctor_get(v_loc_1693_, 0);
v_a_1697_ = lean_ctor_get(v_loc_1693_, 1);
lean_inc(v_a_1696_);
v___x_1698_ = l_Lean_FVarId_getType___redArg(v_a_1696_, v_a_1637_, v_a_1639_, v_a_1640_);
if (lean_obj_tag(v___x_1698_) == 0)
{
lean_object* v_a_1699_; lean_object* v___x_1700_; 
v_a_1699_ = lean_ctor_get(v___x_1698_, 0);
lean_inc(v_a_1699_);
lean_dec_ref_known(v___x_1698_, 1);
lean_inc(v_a_1696_);
v___x_1700_ = l_Lean_FVarId_getUserName___redArg(v_a_1696_, v_a_1637_, v_a_1639_, v_a_1640_);
if (lean_obj_tag(v___x_1700_) == 0)
{
lean_object* v_a_1701_; lean_object* v___x_1702_; 
v_a_1701_ = lean_ctor_get(v___x_1700_, 0);
lean_inc(v_a_1701_);
lean_dec_ref_known(v___x_1700_, 1);
v___x_1702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1702_, 0, v_a_1701_);
lean_inc(v_a_1697_);
v_fst_1643_ = v_a_1699_;
v_fst_1644_ = v_a_1697_;
v_snd_1645_ = v___x_1702_;
v___y_1646_ = v_a_1637_;
v___y_1647_ = v_a_1638_;
v___y_1648_ = v_a_1639_;
v___y_1649_ = v_a_1640_;
goto v___jp_1642_;
}
else
{
lean_object* v_a_1703_; lean_object* v___x_1705_; uint8_t v_isShared_1706_; uint8_t v_isSharedCheck_1710_; 
lean_dec(v_a_1699_);
v_a_1703_ = lean_ctor_get(v___x_1700_, 0);
v_isSharedCheck_1710_ = !lean_is_exclusive(v___x_1700_);
if (v_isSharedCheck_1710_ == 0)
{
v___x_1705_ = v___x_1700_;
v_isShared_1706_ = v_isSharedCheck_1710_;
goto v_resetjp_1704_;
}
else
{
lean_inc(v_a_1703_);
lean_dec(v___x_1700_);
v___x_1705_ = lean_box(0);
v_isShared_1706_ = v_isSharedCheck_1710_;
goto v_resetjp_1704_;
}
v_resetjp_1704_:
{
lean_object* v___x_1708_; 
if (v_isShared_1706_ == 0)
{
v___x_1708_ = v___x_1705_;
goto v_reusejp_1707_;
}
else
{
lean_object* v_reuseFailAlloc_1709_; 
v_reuseFailAlloc_1709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1709_, 0, v_a_1703_);
v___x_1708_ = v_reuseFailAlloc_1709_;
goto v_reusejp_1707_;
}
v_reusejp_1707_:
{
return v___x_1708_;
}
}
}
}
else
{
lean_object* v_a_1711_; lean_object* v___x_1713_; uint8_t v_isShared_1714_; uint8_t v_isSharedCheck_1718_; 
v_a_1711_ = lean_ctor_get(v___x_1698_, 0);
v_isSharedCheck_1718_ = !lean_is_exclusive(v___x_1698_);
if (v_isSharedCheck_1718_ == 0)
{
v___x_1713_ = v___x_1698_;
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
else
{
lean_inc(v_a_1711_);
lean_dec(v___x_1698_);
v___x_1713_ = lean_box(0);
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
v_resetjp_1712_:
{
lean_object* v___x_1716_; 
if (v_isShared_1714_ == 0)
{
v___x_1716_ = v___x_1713_;
goto v_reusejp_1715_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v_a_1711_);
v___x_1716_ = v_reuseFailAlloc_1717_;
goto v_reusejp_1715_;
}
v_reusejp_1715_:
{
return v___x_1716_;
}
}
}
}
default: 
{
lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v_a_1721_; lean_object* v___x_1723_; uint8_t v_isShared_1724_; uint8_t v_isSharedCheck_1728_; 
lean_dec_ref(v_goalType_1636_);
v___x_1719_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__6, &lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__6);
v___x_1720_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Widget_Conv_0__Mathlib_Tactic_Conv_Path_ofSubExprPosArray_go_spec__0___redArg(v___x_1719_, v_a_1637_, v_a_1638_, v_a_1639_, v_a_1640_);
v_a_1721_ = lean_ctor_get(v___x_1720_, 0);
v_isSharedCheck_1728_ = !lean_is_exclusive(v___x_1720_);
if (v_isSharedCheck_1728_ == 0)
{
v___x_1723_ = v___x_1720_;
v_isShared_1724_ = v_isSharedCheck_1728_;
goto v_resetjp_1722_;
}
else
{
lean_inc(v_a_1721_);
lean_dec(v___x_1720_);
v___x_1723_ = lean_box(0);
v_isShared_1724_ = v_isSharedCheck_1728_;
goto v_resetjp_1722_;
}
v_resetjp_1722_:
{
lean_object* v___x_1726_; 
if (v_isShared_1724_ == 0)
{
v___x_1726_ = v___x_1723_;
goto v_reusejp_1725_;
}
else
{
lean_object* v_reuseFailAlloc_1727_; 
v_reuseFailAlloc_1727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1727_, 0, v_a_1721_);
v___x_1726_ = v_reuseFailAlloc_1727_;
goto v_reusejp_1725_;
}
v_reusejp_1725_:
{
return v___x_1726_;
}
}
}
}
}
v___jp_1642_:
{
lean_object* v___x_1650_; lean_object* v_a_1651_; lean_object* v___x_1652_; 
v___x_1650_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__0___redArg(v_fst_1643_, v___y_1647_);
v_a_1651_ = lean_ctor_get(v___x_1650_, 0);
lean_inc(v_a_1651_);
lean_dec_ref(v___x_1650_);
v___x_1652_ = lp_mathlib_Mathlib_Tactic_Conv_Path_ofSubExprPos(v_a_1651_, v_fst_1644_, v___y_1646_, v___y_1647_, v___y_1648_, v___y_1649_);
lean_dec(v_fst_1644_);
if (lean_obj_tag(v___x_1652_) == 0)
{
lean_object* v_a_1653_; lean_object* v_ref_1654_; uint8_t v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; 
v_a_1653_ = lean_ctor_get(v___x_1652_, 0);
lean_inc(v_a_1653_);
lean_dec_ref_known(v___x_1652_, 1);
v_ref_1654_ = lean_ctor_get(v___y_1648_, 5);
v___x_1655_ = 0;
v___x_1656_ = l_Lean_SourceInfo_fromRef(v_ref_1654_, v___x_1655_);
v___x_1657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0));
v___x_1658_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__1));
lean_inc(v___x_1656_);
v___x_1659_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1659_, 0, v___x_1656_);
lean_ctor_set(v___x_1659_, 1, v___x_1657_);
v___x_1660_ = l_Lean_Syntax_node1(v___x_1656_, v___x_1658_, v___x_1659_);
v___x_1661_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__2));
v___x_1662_ = lp_mathlib_Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1(v___x_1660_, v_a_1653_, v_snd_1645_, v___x_1661_, v___y_1646_, v___y_1647_, v___y_1648_, v___y_1649_);
if (lean_obj_tag(v___x_1662_) == 0)
{
lean_object* v_a_1663_; lean_object* v___x_1665_; uint8_t v_isShared_1666_; uint8_t v_isSharedCheck_1670_; 
v_a_1663_ = lean_ctor_get(v___x_1662_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1662_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1665_ = v___x_1662_;
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
else
{
lean_inc(v_a_1663_);
lean_dec(v___x_1662_);
v___x_1665_ = lean_box(0);
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
v_resetjp_1664_:
{
lean_object* v___x_1668_; 
if (v_isShared_1666_ == 0)
{
v___x_1668_ = v___x_1665_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v_a_1663_);
v___x_1668_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
return v___x_1668_;
}
}
}
else
{
lean_object* v_a_1671_; lean_object* v___x_1673_; uint8_t v_isShared_1674_; uint8_t v_isSharedCheck_1678_; 
v_a_1671_ = lean_ctor_get(v___x_1662_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1662_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1673_ = v___x_1662_;
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___x_1662_);
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
else
{
lean_object* v_a_1679_; lean_object* v___x_1681_; uint8_t v_isShared_1682_; uint8_t v_isSharedCheck_1686_; 
lean_dec(v_snd_1645_);
v_a_1679_ = lean_ctor_get(v___x_1652_, 0);
v_isSharedCheck_1686_ = !lean_is_exclusive(v___x_1652_);
if (v_isSharedCheck_1686_ == 0)
{
v___x_1681_ = v___x_1652_;
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
else
{
lean_inc(v_a_1679_);
lean_dec(v___x_1652_);
v___x_1681_ = lean_box(0);
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
v_resetjp_1680_:
{
lean_object* v___x_1684_; 
if (v_isShared_1682_ == 0)
{
v___x_1684_ = v___x_1681_;
goto v_reusejp_1683_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v_a_1679_);
v___x_1684_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1683_;
}
v_reusejp_1683_:
{
return v___x_1684_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___boxed(lean_object* v_locations_1729_, lean_object* v_goalType_1730_, lean_object* v_a_1731_, lean_object* v_a_1732_, lean_object* v_a_1733_, lean_object* v_a_1734_, lean_object* v_a_1735_){
_start:
{
lean_object* v_res_1736_; 
v_res_1736_ = lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax(v_locations_1729_, v_goalType_1730_, v_a_1731_, v_a_1732_, v_a_1733_, v_a_1734_);
lean_dec(v_a_1734_);
lean_dec_ref(v_a_1733_);
lean_dec(v_a_1732_);
lean_dec_ref(v_a_1731_);
lean_dec_ref(v_locations_1729_);
return v_res_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2(lean_object* v_val_1737_, uint8_t v_canonical_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_){
_start:
{
lean_object* v___x_1744_; 
v___x_1744_ = lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___redArg(v_val_1737_, v_canonical_1738_, v___y_1741_);
return v___x_1744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2___boxed(lean_object* v_val_1745_, lean_object* v_canonical_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_){
_start:
{
uint8_t v_canonical_boxed_1752_; lean_object* v_res_1753_; 
v_canonical_boxed_1752_ = lean_unbox(v_canonical_1746_);
v_res_1753_ = lp_mathlib_Lean_mkIdentFromRef___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__2(v_val_1745_, v_canonical_boxed_1752_, v___y_1747_, v___y_1748_, v___y_1749_, v___y_1750_);
lean_dec(v___y_1750_);
lean_dec_ref(v___y_1749_);
lean_dec(v___y_1748_);
lean_dec_ref(v___y_1747_);
return v_res_1753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1(lean_object* v___x_1754_, lean_object* v_range_1755_, lean_object* v_b_1756_, lean_object* v_i_1757_, lean_object* v_hs_1758_, lean_object* v_hl_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
lean_object* v___x_1765_; 
v___x_1765_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___redArg(v___x_1754_, v_range_1755_, v_b_1756_, v_i_1757_);
return v___x_1765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1___boxed(lean_object* v___x_1766_, lean_object* v_range_1767_, lean_object* v_b_1768_, lean_object* v_i_1769_, lean_object* v_hs_1770_, lean_object* v_hl_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_){
_start:
{
lean_object* v_res_1777_; 
v_res_1777_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Conv_pathToStx___at___00Mathlib_Tactic_Conv_insertEnterSyntax_spec__1_spec__1(v___x_1766_, v_range_1767_, v_b_1768_, v_i_1769_, v_hs_1770_, v_hl_1771_, v___y_1772_, v___y_1773_, v___y_1774_, v___y_1775_);
lean_dec(v___y_1775_);
lean_dec_ref(v___y_1774_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec_ref(v_range_1767_);
return v_res_1777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_Conv_insertEnter_spec__0(lean_object* v_s_1778_, lean_object* v_pos_1779_){
_start:
{
lean_object* v_str_1780_; lean_object* v_startInclusive_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; uint8_t v___x_1785_; 
v_str_1780_ = lean_ctor_get(v_s_1778_, 0);
v_startInclusive_1781_ = lean_ctor_get(v_s_1778_, 1);
v___x_1782_ = lean_nat_add(v_startInclusive_1781_, v_pos_1779_);
v___x_1783_ = lean_nat_sub(v___x_1782_, v_startInclusive_1781_);
v___x_1784_ = lean_unsigned_to_nat(0u);
v___x_1785_ = lean_nat_dec_eq(v___x_1783_, v___x_1784_);
if (v___x_1785_ == 0)
{
lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; uint8_t v___y_1794_; lean_object* v___x_1795_; uint32_t v___x_1796_; uint8_t v___y_1798_; uint32_t v___x_1803_; uint8_t v___x_1804_; 
lean_inc(v_startInclusive_1781_);
lean_inc_ref(v_str_1780_);
v___x_1786_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1786_, 0, v_str_1780_);
lean_ctor_set(v___x_1786_, 1, v_startInclusive_1781_);
lean_ctor_set(v___x_1786_, 2, v___x_1782_);
v___x_1787_ = lean_unsigned_to_nat(1u);
v___x_1788_ = lean_nat_sub(v___x_1783_, v___x_1787_);
lean_dec(v___x_1783_);
v___x_1789_ = l_String_Slice_posLE(v___x_1786_, v___x_1788_);
lean_dec_ref_known(v___x_1786_, 3);
v___x_1795_ = lean_nat_add(v_startInclusive_1781_, v___x_1789_);
v___x_1796_ = lean_string_utf8_get_fast(v_str_1780_, v___x_1795_);
lean_dec(v___x_1795_);
v___x_1803_ = 32;
v___x_1804_ = lean_uint32_dec_eq(v___x_1796_, v___x_1803_);
if (v___x_1804_ == 0)
{
uint32_t v___x_1805_; uint8_t v___x_1806_; 
v___x_1805_ = 9;
v___x_1806_ = lean_uint32_dec_eq(v___x_1796_, v___x_1805_);
v___y_1798_ = v___x_1806_;
goto v___jp_1797_;
}
else
{
v___y_1798_ = v___x_1804_;
goto v___jp_1797_;
}
v___jp_1790_:
{
uint8_t v___x_1791_; 
v___x_1791_ = lean_nat_dec_lt(v___x_1789_, v_pos_1779_);
if (v___x_1791_ == 0)
{
lean_dec(v___x_1789_);
return v_pos_1779_;
}
else
{
lean_dec(v_pos_1779_);
v_pos_1779_ = v___x_1789_;
goto _start;
}
}
v___jp_1793_:
{
if (v___y_1794_ == 0)
{
lean_dec(v___x_1789_);
return v_pos_1779_;
}
else
{
goto v___jp_1790_;
}
}
v___jp_1797_:
{
if (v___y_1798_ == 0)
{
uint32_t v___x_1799_; uint8_t v___x_1800_; 
v___x_1799_ = 13;
v___x_1800_ = lean_uint32_dec_eq(v___x_1796_, v___x_1799_);
if (v___x_1800_ == 0)
{
uint32_t v___x_1801_; uint8_t v___x_1802_; 
v___x_1801_ = 10;
v___x_1802_ = lean_uint32_dec_eq(v___x_1796_, v___x_1801_);
v___y_1794_ = v___x_1802_;
goto v___jp_1793_;
}
else
{
v___y_1794_ = v___x_1800_;
goto v___jp_1793_;
}
}
else
{
goto v___jp_1790_;
}
}
}
else
{
lean_dec(v___x_1783_);
lean_dec(v___x_1782_);
return v_pos_1779_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_Conv_insertEnter_spec__0___boxed(lean_object* v_s_1807_, lean_object* v_pos_1808_){
_start:
{
lean_object* v_res_1809_; 
v_res_1809_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_Conv_insertEnter_spec__0(v_s_1807_, v_pos_1808_);
lean_dec_ref(v_s_1807_);
return v_res_1809_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__3(void){
_start:
{
lean_object* v___x_1814_; lean_object* v___x_1815_; 
v___x_1814_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0));
v___x_1815_ = lean_string_utf8_byte_size(v___x_1814_);
return v___x_1815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter(lean_object* v_locations_1816_, lean_object* v_goalType_1817_, lean_object* v_params_1818_, lean_object* v_a_1819_, lean_object* v_a_1820_, lean_object* v_a_1821_, lean_object* v_a_1822_){
_start:
{
lean_object* v___x_1824_; 
v___x_1824_ = lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax(v_locations_1816_, v_goalType_1817_, v_a_1819_, v_a_1820_, v_a_1821_, v_a_1822_);
if (lean_obj_tag(v___x_1824_) == 0)
{
lean_object* v_a_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; 
v_a_1825_ = lean_ctor_get(v___x_1824_, 0);
lean_inc(v_a_1825_);
lean_dec_ref_known(v___x_1824_, 1);
v___x_1826_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__1));
v___x_1827_ = l_Lean_PrettyPrinter_ppCategory(v___x_1826_, v_a_1825_, v_a_1821_, v_a_1822_);
if (lean_obj_tag(v___x_1827_) == 0)
{
lean_object* v_replaceRange_1828_; lean_object* v_start_1829_; lean_object* v___x_1831_; uint8_t v_isShared_1832_; uint8_t v_isSharedCheck_1873_; 
v_replaceRange_1828_ = lean_ctor_get(v_params_1818_, 3);
lean_inc_ref(v_replaceRange_1828_);
lean_dec_ref(v_params_1818_);
v_start_1829_ = lean_ctor_get(v_replaceRange_1828_, 0);
v_isSharedCheck_1873_ = !lean_is_exclusive(v_replaceRange_1828_);
if (v_isSharedCheck_1873_ == 0)
{
lean_object* v_unused_1874_; 
v_unused_1874_ = lean_ctor_get(v_replaceRange_1828_, 1);
lean_dec(v_unused_1874_);
v___x_1831_ = v_replaceRange_1828_;
v_isShared_1832_ = v_isSharedCheck_1873_;
goto v_resetjp_1830_;
}
else
{
lean_inc(v_start_1829_);
lean_dec(v_replaceRange_1828_);
v___x_1831_ = lean_box(0);
v_isShared_1832_ = v_isSharedCheck_1873_;
goto v_resetjp_1830_;
}
v_resetjp_1830_:
{
lean_object* v_a_1833_; lean_object* v___x_1835_; uint8_t v_isShared_1836_; uint8_t v_isSharedCheck_1872_; 
v_a_1833_ = lean_ctor_get(v___x_1827_, 0);
v_isSharedCheck_1872_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1872_ == 0)
{
v___x_1835_ = v___x_1827_;
v_isShared_1836_ = v_isSharedCheck_1872_;
goto v_resetjp_1834_;
}
else
{
lean_inc(v_a_1833_);
lean_dec(v___x_1827_);
v___x_1835_ = lean_box(0);
v_isShared_1836_ = v_isSharedCheck_1872_;
goto v_resetjp_1834_;
}
v_resetjp_1834_:
{
lean_object* v_character_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1870_; 
v_character_1837_ = lean_ctor_get(v_start_1829_, 1);
v_isSharedCheck_1870_ = !lean_is_exclusive(v_start_1829_);
if (v_isSharedCheck_1870_ == 0)
{
lean_object* v_unused_1871_; 
v_unused_1871_ = lean_ctor_get(v_start_1829_, 0);
lean_dec(v_unused_1871_);
v___x_1839_ = v_start_1829_;
v_isShared_1840_ = v_isSharedCheck_1870_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_character_1837_);
lean_dec(v_start_1829_);
v___x_1839_ = lean_box(0);
v_isShared_1840_ = v_isSharedCheck_1870_;
goto v_resetjp_1838_;
}
v_resetjp_1838_:
{
lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___y_1844_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; uint8_t v___x_1863_; 
v___x_1841_ = lean_unsigned_to_nat(100u);
lean_inc(v_character_1837_);
v___x_1842_ = l_Std_Format_pretty(v_a_1833_, v___x_1841_, v_character_1837_, v_character_1837_);
v___x_1857_ = lean_unsigned_to_nat(0u);
v___x_1858_ = lean_string_utf8_byte_size(v___x_1842_);
lean_inc_ref(v___x_1842_);
v___x_1859_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1859_, 0, v___x_1842_);
lean_ctor_set(v___x_1859_, 1, v___x_1857_);
lean_ctor_set(v___x_1859_, 2, v___x_1858_);
v___x_1860_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_Conv_insertEnter_spec__0(v___x_1859_, v___x_1858_);
lean_dec_ref_known(v___x_1859_, 3);
v___x_1861_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnterSyntax___closed__0));
v___x_1862_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__3, &lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__3);
v___x_1863_ = lean_nat_dec_le(v___x_1862_, v___x_1860_);
if (v___x_1863_ == 0)
{
lean_dec(v___x_1860_);
goto v___jp_1855_;
}
else
{
lean_object* v___x_1864_; uint8_t v___x_1865_; 
v___x_1864_ = lean_nat_sub(v___x_1860_, v___x_1862_);
v___x_1865_ = lean_string_memcmp(v___x_1842_, v___x_1861_, v___x_1864_, v___x_1857_, v___x_1862_);
if (v___x_1865_ == 0)
{
lean_dec(v___x_1864_);
lean_dec(v___x_1860_);
goto v___jp_1855_;
}
else
{
lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; 
lean_inc(v___x_1860_);
lean_inc_ref(v___x_1842_);
v___x_1866_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1866_, 0, v___x_1842_);
lean_ctor_set(v___x_1866_, 1, v___x_1857_);
lean_ctor_set(v___x_1866_, 2, v___x_1860_);
v___x_1867_ = l_String_Slice_pos_x21(v___x_1866_, v___x_1864_);
lean_dec(v___x_1864_);
lean_dec_ref_known(v___x_1866_, 3);
v___x_1868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1868_, 0, v___x_1867_);
lean_ctor_set(v___x_1868_, 1, v___x_1860_);
v___x_1869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1869_, 0, v___x_1868_);
v___y_1844_ = v___x_1869_;
goto v___jp_1843_;
}
}
v___jp_1843_:
{
lean_object* v___x_1845_; lean_object* v___x_1847_; 
v___x_1845_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_insertEnter___closed__2));
if (v_isShared_1840_ == 0)
{
lean_ctor_set(v___x_1839_, 1, v___y_1844_);
lean_ctor_set(v___x_1839_, 0, v___x_1842_);
v___x_1847_ = v___x_1839_;
goto v_reusejp_1846_;
}
else
{
lean_object* v_reuseFailAlloc_1854_; 
v_reuseFailAlloc_1854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1854_, 0, v___x_1842_);
lean_ctor_set(v_reuseFailAlloc_1854_, 1, v___y_1844_);
v___x_1847_ = v_reuseFailAlloc_1854_;
goto v_reusejp_1846_;
}
v_reusejp_1846_:
{
lean_object* v___x_1849_; 
if (v_isShared_1832_ == 0)
{
lean_ctor_set(v___x_1831_, 1, v___x_1847_);
lean_ctor_set(v___x_1831_, 0, v___x_1845_);
v___x_1849_ = v___x_1831_;
goto v_reusejp_1848_;
}
else
{
lean_object* v_reuseFailAlloc_1853_; 
v_reuseFailAlloc_1853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1853_, 0, v___x_1845_);
lean_ctor_set(v_reuseFailAlloc_1853_, 1, v___x_1847_);
v___x_1849_ = v_reuseFailAlloc_1853_;
goto v_reusejp_1848_;
}
v_reusejp_1848_:
{
lean_object* v___x_1851_; 
if (v_isShared_1836_ == 0)
{
lean_ctor_set(v___x_1835_, 0, v___x_1849_);
v___x_1851_ = v___x_1835_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v___x_1849_);
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
v___jp_1855_:
{
lean_object* v___x_1856_; 
v___x_1856_ = lean_box(0);
v___y_1844_ = v___x_1856_;
goto v___jp_1843_;
}
}
}
}
}
else
{
lean_object* v_a_1875_; lean_object* v___x_1877_; uint8_t v_isShared_1878_; uint8_t v_isSharedCheck_1882_; 
lean_dec_ref(v_params_1818_);
v_a_1875_ = lean_ctor_get(v___x_1827_, 0);
v_isSharedCheck_1882_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1882_ == 0)
{
v___x_1877_ = v___x_1827_;
v_isShared_1878_ = v_isSharedCheck_1882_;
goto v_resetjp_1876_;
}
else
{
lean_inc(v_a_1875_);
lean_dec(v___x_1827_);
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
lean_ctor_set(v_reuseFailAlloc_1881_, 0, v_a_1875_);
v___x_1880_ = v_reuseFailAlloc_1881_;
goto v_reusejp_1879_;
}
v_reusejp_1879_:
{
return v___x_1880_;
}
}
}
}
else
{
lean_object* v_a_1883_; lean_object* v___x_1885_; uint8_t v_isShared_1886_; uint8_t v_isSharedCheck_1890_; 
lean_dec_ref(v_params_1818_);
v_a_1883_ = lean_ctor_get(v___x_1824_, 0);
v_isSharedCheck_1890_ = !lean_is_exclusive(v___x_1824_);
if (v_isSharedCheck_1890_ == 0)
{
v___x_1885_ = v___x_1824_;
v_isShared_1886_ = v_isSharedCheck_1890_;
goto v_resetjp_1884_;
}
else
{
lean_inc(v_a_1883_);
lean_dec(v___x_1824_);
v___x_1885_ = lean_box(0);
v_isShared_1886_ = v_isSharedCheck_1890_;
goto v_resetjp_1884_;
}
v_resetjp_1884_:
{
lean_object* v___x_1888_; 
if (v_isShared_1886_ == 0)
{
v___x_1888_ = v___x_1885_;
goto v_reusejp_1887_;
}
else
{
lean_object* v_reuseFailAlloc_1889_; 
v_reuseFailAlloc_1889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1889_, 0, v_a_1883_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_insertEnter___boxed(lean_object* v_locations_1891_, lean_object* v_goalType_1892_, lean_object* v_params_1893_, lean_object* v_a_1894_, lean_object* v_a_1895_, lean_object* v_a_1896_, lean_object* v_a_1897_, lean_object* v_a_1898_){
_start:
{
lean_object* v_res_1899_; 
v_res_1899_ = lp_mathlib_Mathlib_Tactic_Conv_insertEnter(v_locations_1891_, v_goalType_1892_, v_params_1893_, v_a_1894_, v_a_1895_, v_a_1896_, v_a_1897_);
lean_dec(v_a_1897_);
lean_dec_ref(v_a_1896_);
lean_dec(v_a_1895_);
lean_dec_ref(v_a_1894_);
lean_dec_ref(v_locations_1891_);
return v_res_1899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg(lean_object* v_mainGoalName_1906_, lean_object* v_errorMsg_1907_, uint8_t v___y_1908_, lean_object* v_as_1909_, size_t v_sz_1910_, size_t v_i_1911_, lean_object* v_b_1912_){
_start:
{
lean_object* v_a_1915_; uint8_t v___x_1919_; 
v___x_1919_ = lean_usize_dec_lt(v_i_1911_, v_sz_1910_);
if (v___x_1919_ == 0)
{
lean_object* v___x_1920_; 
lean_dec_ref(v_errorMsg_1907_);
v___x_1920_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1920_, 0, v_b_1912_);
return v___x_1920_;
}
else
{
lean_object* v_a_1921_; lean_object* v_mvarId_1922_; lean_object* v_loc_1923_; lean_object* v___x_1925_; uint8_t v_isShared_1926_; uint8_t v_isSharedCheck_1954_; 
lean_dec_ref(v_b_1912_);
v_a_1921_ = lean_array_uget(v_as_1909_, v_i_1911_);
v_mvarId_1922_ = lean_ctor_get(v_a_1921_, 0);
v_loc_1923_ = lean_ctor_get(v_a_1921_, 1);
v_isSharedCheck_1954_ = !lean_is_exclusive(v_a_1921_);
if (v_isSharedCheck_1954_ == 0)
{
v___x_1925_ = v_a_1921_;
v_isShared_1926_ = v_isSharedCheck_1954_;
goto v_resetjp_1924_;
}
else
{
lean_inc(v_loc_1923_);
lean_inc(v_mvarId_1922_);
lean_dec(v_a_1921_);
v___x_1925_ = lean_box(0);
v_isShared_1926_ = v_isSharedCheck_1954_;
goto v_resetjp_1924_;
}
v_resetjp_1924_:
{
lean_object* v___x_1927_; uint8_t v___x_1928_; 
v___x_1927_ = lean_box(0);
v___x_1928_ = lean_name_eq(v_mvarId_1922_, v_mainGoalName_1906_);
lean_dec(v_mvarId_1922_);
if (v___x_1928_ == 0)
{
lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1938_; 
lean_dec_ref(v_loc_1923_);
v___x_1929_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0));
v___x_1930_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1));
v___x_1931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1931_, 0, v_errorMsg_1907_);
v___x_1932_ = lean_unsigned_to_nat(1u);
v___x_1933_ = lean_mk_empty_array_with_capacity(v___x_1932_);
v___x_1934_ = lean_array_push(v___x_1933_, v___x_1931_);
v___x_1935_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1935_, 0, v___x_1929_);
lean_ctor_set(v___x_1935_, 1, v___x_1930_);
lean_ctor_set(v___x_1935_, 2, v___x_1934_);
v___x_1936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1936_, 0, v___x_1935_);
if (v_isShared_1926_ == 0)
{
lean_ctor_set(v___x_1925_, 1, v___x_1927_);
lean_ctor_set(v___x_1925_, 0, v___x_1936_);
v___x_1938_ = v___x_1925_;
goto v_reusejp_1937_;
}
else
{
lean_object* v_reuseFailAlloc_1940_; 
v_reuseFailAlloc_1940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1940_, 0, v___x_1936_);
lean_ctor_set(v_reuseFailAlloc_1940_, 1, v___x_1927_);
v___x_1938_ = v_reuseFailAlloc_1940_;
goto v_reusejp_1937_;
}
v_reusejp_1937_:
{
lean_object* v___x_1939_; 
v___x_1939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1939_, 0, v___x_1938_);
return v___x_1939_;
}
}
else
{
lean_object* v___x_1941_; 
v___x_1941_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__2));
if (v___y_1908_ == 0)
{
lean_del_object(v___x_1925_);
lean_dec_ref(v_loc_1923_);
v_a_1915_ = v___x_1941_;
goto v___jp_1914_;
}
else
{
if (lean_obj_tag(v_loc_1923_) == 3)
{
lean_dec_ref_known(v_loc_1923_, 1);
lean_del_object(v___x_1925_);
v_a_1915_ = v___x_1941_;
goto v___jp_1914_;
}
else
{
lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1951_; 
lean_dec_ref(v_loc_1923_);
v___x_1942_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0));
v___x_1943_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1));
v___x_1944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1944_, 0, v_errorMsg_1907_);
v___x_1945_ = lean_unsigned_to_nat(1u);
v___x_1946_ = lean_mk_empty_array_with_capacity(v___x_1945_);
v___x_1947_ = lean_array_push(v___x_1946_, v___x_1944_);
v___x_1948_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1948_, 0, v___x_1942_);
lean_ctor_set(v___x_1948_, 1, v___x_1943_);
lean_ctor_set(v___x_1948_, 2, v___x_1947_);
v___x_1949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1949_, 0, v___x_1948_);
if (v_isShared_1926_ == 0)
{
lean_ctor_set(v___x_1925_, 1, v___x_1927_);
lean_ctor_set(v___x_1925_, 0, v___x_1949_);
v___x_1951_ = v___x_1925_;
goto v_reusejp_1950_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v___x_1949_);
lean_ctor_set(v_reuseFailAlloc_1953_, 1, v___x_1927_);
v___x_1951_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1950_;
}
v_reusejp_1950_:
{
lean_object* v___x_1952_; 
v___x_1952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1952_, 0, v___x_1951_);
return v___x_1952_;
}
}
}
}
}
}
v___jp_1914_:
{
size_t v___x_1916_; size_t v___x_1917_; 
v___x_1916_ = ((size_t)1ULL);
v___x_1917_ = lean_usize_add(v_i_1911_, v___x_1916_);
lean_inc_ref(v_a_1915_);
v_i_1911_ = v___x_1917_;
v_b_1912_ = v_a_1915_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___boxed(lean_object* v_mainGoalName_1955_, lean_object* v_errorMsg_1956_, lean_object* v___y_1957_, lean_object* v_as_1958_, lean_object* v_sz_1959_, lean_object* v_i_1960_, lean_object* v_b_1961_, lean_object* v___y_1962_){
_start:
{
uint8_t v___y_1955__boxed_1963_; size_t v_sz_boxed_1964_; size_t v_i_boxed_1965_; lean_object* v_res_1966_; 
v___y_1955__boxed_1963_ = lean_unbox(v___y_1957_);
v_sz_boxed_1964_ = lean_unbox_usize(v_sz_1959_);
lean_dec(v_sz_1959_);
v_i_boxed_1965_ = lean_unbox_usize(v_i_1960_);
lean_dec(v_i_1960_);
v_res_1966_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg(v_mainGoalName_1955_, v_errorMsg_1956_, v___y_1955__boxed_1963_, v_as_1958_, v_sz_boxed_1964_, v_i_boxed_1965_, v_b_1961_);
lean_dec_ref(v_as_1958_);
lean_dec(v_mainGoalName_1955_);
return v_res_1966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__0(lean_object* v___y_1967_){
_start:
{
lean_object* v_doc_1969_; lean_object* v___x_1970_; 
v_doc_1969_ = lean_ctor_get(v___y_1967_, 1);
lean_inc_ref(v_doc_1969_);
v___x_1970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1970_, 0, v_doc_1969_);
return v___x_1970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__0___boxed(lean_object* v___y_1971_, lean_object* v___y_1972_){
_start:
{
lean_object* v_res_1973_; 
v_res_1973_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__0(v___y_1971_);
lean_dec_ref(v___y_1971_);
return v_res_1973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2___lam__0(lean_object* v_props_1974_, lean_object* v___y_1975_){
_start:
{
lean_object* v___x_1976_; lean_object* v___x_1977_; 
v___x_1976_ = lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(v_props_1974_);
v___x_1977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1977_, 0, v___x_1976_);
lean_ctor_set(v___x_1977_, 1, v___y_1975_);
return v___x_1977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2(lean_object* v_c_1978_, lean_object* v_props_1979_, lean_object* v_children_1980_){
_start:
{
lean_object* v_toModule_1981_; lean_object* v_export_1982_; lean_object* v_javascript_1983_; lean_object* v___f_1984_; uint64_t v___x_1985_; lean_object* v___x_1986_; 
v_toModule_1981_ = lean_ctor_get(v_c_1978_, 0);
v_export_1982_ = lean_ctor_get(v_c_1978_, 1);
v_javascript_1983_ = lean_ctor_get(v_toModule_1981_, 0);
v___f_1984_ = lean_alloc_closure((void*)(lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2___lam__0), 2, 1);
lean_closure_set(v___f_1984_, 0, v_props_1979_);
v___x_1985_ = lean_string_hash(v_javascript_1983_);
lean_inc_ref(v_export_1982_);
v___x_1986_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_1986_, 0, v_export_1982_);
lean_ctor_set(v___x_1986_, 1, v___f_1984_);
lean_ctor_set(v___x_1986_, 2, v_children_1980_);
lean_ctor_set_uint64(v___x_1986_, sizeof(void*)*3, v___x_1985_);
return v___x_1986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2___boxed(lean_object* v_c_1987_, lean_object* v_props_1988_, lean_object* v_children_1989_){
_start:
{
lean_object* v_res_1990_; 
v_res_1990_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2(v_c_1987_, v_props_1988_, v_children_1989_);
lean_dec_ref(v_c_1987_);
return v_res_1990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__0(lean_object* v_mkCmdStr_1991_, lean_object* v_selectedLocations_1992_, lean_object* v___x_1993_, lean_object* v_params_1994_, lean_object* v_a_1995_, lean_object* v_replaceRange_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_){
_start:
{
lean_object* v___x_2002_; 
v___x_2002_ = lean_apply_8(v_mkCmdStr_1991_, v_selectedLocations_1992_, v___x_1993_, v_params_1994_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_, lean_box(0));
if (lean_obj_tag(v___x_2002_) == 0)
{
lean_object* v_a_2003_; lean_object* v___x_2005_; uint8_t v_isShared_2006_; uint8_t v_isSharedCheck_2023_; 
v_a_2003_ = lean_ctor_get(v___x_2002_, 0);
v_isSharedCheck_2023_ = !lean_is_exclusive(v___x_2002_);
if (v_isSharedCheck_2023_ == 0)
{
v___x_2005_ = v___x_2002_;
v_isShared_2006_ = v_isSharedCheck_2023_;
goto v_resetjp_2004_;
}
else
{
lean_inc(v_a_2003_);
lean_dec(v___x_2002_);
v___x_2005_ = lean_box(0);
v_isShared_2006_ = v_isSharedCheck_2023_;
goto v_resetjp_2004_;
}
v_resetjp_2004_:
{
lean_object* v_snd_2007_; lean_object* v_toEditableDocumentCore_2008_; lean_object* v_fst_2009_; lean_object* v_fst_2010_; lean_object* v_snd_2011_; lean_object* v_meta_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2021_; 
v_snd_2007_ = lean_ctor_get(v_a_2003_, 1);
lean_inc(v_snd_2007_);
v_toEditableDocumentCore_2008_ = lean_ctor_get(v_a_1995_, 0);
v_fst_2009_ = lean_ctor_get(v_a_2003_, 0);
lean_inc(v_fst_2009_);
lean_dec(v_a_2003_);
v_fst_2010_ = lean_ctor_get(v_snd_2007_, 0);
lean_inc(v_fst_2010_);
v_snd_2011_ = lean_ctor_get(v_snd_2007_, 1);
lean_inc(v_snd_2011_);
lean_dec(v_snd_2007_);
v_meta_2012_ = lean_ctor_get(v_toEditableDocumentCore_2008_, 0);
v___x_2013_ = lp_proofwidgets_ProofWidgets_MakeEditLink;
v___x_2014_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(v_meta_2012_, v_replaceRange_1996_, v_fst_2010_, v_snd_2011_);
v___x_2015_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2015_, 0, v_fst_2009_);
v___x_2016_ = lean_unsigned_to_nat(1u);
v___x_2017_ = lean_mk_empty_array_with_capacity(v___x_2016_);
v___x_2018_ = lean_array_push(v___x_2017_, v___x_2015_);
v___x_2019_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__2(v___x_2013_, v___x_2014_, v___x_2018_);
if (v_isShared_2006_ == 0)
{
lean_ctor_set(v___x_2005_, 0, v___x_2019_);
v___x_2021_ = v___x_2005_;
goto v_reusejp_2020_;
}
else
{
lean_object* v_reuseFailAlloc_2022_; 
v_reuseFailAlloc_2022_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2022_, 0, v___x_2019_);
v___x_2021_ = v_reuseFailAlloc_2022_;
goto v_reusejp_2020_;
}
v_reusejp_2020_:
{
return v___x_2021_;
}
}
}
else
{
lean_object* v_a_2024_; lean_object* v___x_2026_; uint8_t v_isShared_2027_; uint8_t v_isSharedCheck_2031_; 
lean_dec_ref(v_replaceRange_1996_);
v_a_2024_ = lean_ctor_get(v___x_2002_, 0);
v_isSharedCheck_2031_ = !lean_is_exclusive(v___x_2002_);
if (v_isSharedCheck_2031_ == 0)
{
v___x_2026_ = v___x_2002_;
v_isShared_2027_ = v_isSharedCheck_2031_;
goto v_resetjp_2025_;
}
else
{
lean_inc(v_a_2024_);
lean_dec(v___x_2002_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__0___boxed(lean_object* v_mkCmdStr_2032_, lean_object* v_selectedLocations_2033_, lean_object* v___x_2034_, lean_object* v_params_2035_, lean_object* v_a_2036_, lean_object* v_replaceRange_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_){
_start:
{
lean_object* v_res_2043_; 
v_res_2043_ = lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__0(v_mkCmdStr_2032_, v_selectedLocations_2033_, v___x_2034_, v_params_2035_, v_a_2036_, v_replaceRange_2037_, v___y_2038_, v___y_2039_, v___y_2040_, v___y_2041_);
lean_dec_ref(v_a_2036_);
return v_res_2043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg(lean_object* v_lctx_2044_, lean_object* v_localInsts_2045_, lean_object* v_x_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_){
_start:
{
lean_object* v___x_2052_; 
v___x_2052_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_2044_, v_localInsts_2045_, v_x_2046_, v___y_2047_, v___y_2048_, v___y_2049_, v___y_2050_);
if (lean_obj_tag(v___x_2052_) == 0)
{
lean_object* v_a_2053_; lean_object* v___x_2055_; uint8_t v_isShared_2056_; uint8_t v_isSharedCheck_2060_; 
v_a_2053_ = lean_ctor_get(v___x_2052_, 0);
v_isSharedCheck_2060_ = !lean_is_exclusive(v___x_2052_);
if (v_isSharedCheck_2060_ == 0)
{
v___x_2055_ = v___x_2052_;
v_isShared_2056_ = v_isSharedCheck_2060_;
goto v_resetjp_2054_;
}
else
{
lean_inc(v_a_2053_);
lean_dec(v___x_2052_);
v___x_2055_ = lean_box(0);
v_isShared_2056_ = v_isSharedCheck_2060_;
goto v_resetjp_2054_;
}
v_resetjp_2054_:
{
lean_object* v___x_2058_; 
if (v_isShared_2056_ == 0)
{
v___x_2058_ = v___x_2055_;
goto v_reusejp_2057_;
}
else
{
lean_object* v_reuseFailAlloc_2059_; 
v_reuseFailAlloc_2059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2059_, 0, v_a_2053_);
v___x_2058_ = v_reuseFailAlloc_2059_;
goto v_reusejp_2057_;
}
v_reusejp_2057_:
{
return v___x_2058_;
}
}
}
else
{
lean_object* v_a_2061_; lean_object* v___x_2063_; uint8_t v_isShared_2064_; uint8_t v_isSharedCheck_2068_; 
v_a_2061_ = lean_ctor_get(v___x_2052_, 0);
v_isSharedCheck_2068_ = !lean_is_exclusive(v___x_2052_);
if (v_isSharedCheck_2068_ == 0)
{
v___x_2063_ = v___x_2052_;
v_isShared_2064_ = v_isSharedCheck_2068_;
goto v_resetjp_2062_;
}
else
{
lean_inc(v_a_2061_);
lean_dec(v___x_2052_);
v___x_2063_ = lean_box(0);
v_isShared_2064_ = v_isSharedCheck_2068_;
goto v_resetjp_2062_;
}
v_resetjp_2062_:
{
lean_object* v___x_2066_; 
if (v_isShared_2064_ == 0)
{
v___x_2066_ = v___x_2063_;
goto v_reusejp_2065_;
}
else
{
lean_object* v_reuseFailAlloc_2067_; 
v_reuseFailAlloc_2067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2067_, 0, v_a_2061_);
v___x_2066_ = v_reuseFailAlloc_2067_;
goto v_reusejp_2065_;
}
v_reusejp_2065_:
{
return v___x_2066_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg___boxed(lean_object* v_lctx_2069_, lean_object* v_localInsts_2070_, lean_object* v_x_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_){
_start:
{
lean_object* v_res_2077_; 
v_res_2077_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg(v_lctx_2069_, v_localInsts_2070_, v_x_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
lean_dec(v___y_2075_);
lean_dec_ref(v___y_2074_);
lean_dec(v___y_2073_);
lean_dec_ref(v___y_2072_);
return v_res_2077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__1(lean_object* v_mvarId_2078_, lean_object* v_mkCmdStr_2079_, lean_object* v_selectedLocations_2080_, lean_object* v_params_2081_, lean_object* v_a_2082_, lean_object* v_replaceRange_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_){
_start:
{
lean_object* v___x_2089_; 
v___x_2089_ = l_Lean_MVarId_getDecl(v_mvarId_2078_, v___y_2084_, v___y_2085_, v___y_2086_, v___y_2087_);
if (lean_obj_tag(v___x_2089_) == 0)
{
lean_object* v_a_2090_; lean_object* v_options_2091_; lean_object* v_lctx_2092_; lean_object* v_type_2093_; lean_object* v_localInstances_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v_fst_2098_; lean_object* v___x_2099_; lean_object* v___f_2100_; lean_object* v___x_2101_; 
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
lean_inc(v_a_2090_);
lean_dec_ref_known(v___x_2089_, 1);
v_options_2091_ = lean_ctor_get(v___y_2086_, 2);
v_lctx_2092_ = lean_ctor_get(v_a_2090_, 1);
lean_inc_ref(v_lctx_2092_);
v_type_2093_ = lean_ctor_get(v_a_2090_, 2);
lean_inc_ref(v_type_2093_);
v_localInstances_2094_ = lean_ctor_get(v_a_2090_, 4);
lean_inc_ref(v_localInstances_2094_);
lean_dec(v_a_2090_);
v___x_2095_ = lean_box(1);
lean_inc_ref(v_options_2091_);
v___x_2096_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2096_, 0, v_options_2091_);
lean_ctor_set(v___x_2096_, 1, v___x_2095_);
lean_ctor_set(v___x_2096_, 2, v___x_2095_);
v___x_2097_ = l_Lean_LocalContext_sanitizeNames(v_lctx_2092_, v___x_2096_);
v_fst_2098_ = lean_ctor_get(v___x_2097_, 0);
lean_inc(v_fst_2098_);
lean_dec_ref(v___x_2097_);
v___x_2099_ = l_Lean_Expr_consumeMData(v_type_2093_);
lean_dec_ref(v_type_2093_);
v___f_2100_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__0___boxed), 11, 6);
lean_closure_set(v___f_2100_, 0, v_mkCmdStr_2079_);
lean_closure_set(v___f_2100_, 1, v_selectedLocations_2080_);
lean_closure_set(v___f_2100_, 2, v___x_2099_);
lean_closure_set(v___f_2100_, 3, v_params_2081_);
lean_closure_set(v___f_2100_, 4, v_a_2082_);
lean_closure_set(v___f_2100_, 5, v_replaceRange_2083_);
v___x_2101_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg(v_fst_2098_, v_localInstances_2094_, v___f_2100_, v___y_2084_, v___y_2085_, v___y_2086_, v___y_2087_);
return v___x_2101_;
}
else
{
lean_object* v_a_2102_; lean_object* v___x_2104_; uint8_t v_isShared_2105_; uint8_t v_isSharedCheck_2109_; 
lean_dec_ref(v_replaceRange_2083_);
lean_dec_ref(v_a_2082_);
lean_dec_ref(v_params_2081_);
lean_dec_ref(v_selectedLocations_2080_);
lean_dec_ref(v_mkCmdStr_2079_);
v_a_2102_ = lean_ctor_get(v___x_2089_, 0);
v_isSharedCheck_2109_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2109_ == 0)
{
v___x_2104_ = v___x_2089_;
v_isShared_2105_ = v_isSharedCheck_2109_;
goto v_resetjp_2103_;
}
else
{
lean_inc(v_a_2102_);
lean_dec(v___x_2089_);
v___x_2104_ = lean_box(0);
v_isShared_2105_ = v_isSharedCheck_2109_;
goto v_resetjp_2103_;
}
v_resetjp_2103_:
{
lean_object* v___x_2107_; 
if (v_isShared_2105_ == 0)
{
v___x_2107_ = v___x_2104_;
goto v_reusejp_2106_;
}
else
{
lean_object* v_reuseFailAlloc_2108_; 
v_reuseFailAlloc_2108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2108_, 0, v_a_2102_);
v___x_2107_ = v_reuseFailAlloc_2108_;
goto v_reusejp_2106_;
}
v_reusejp_2106_:
{
return v___x_2107_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__1___boxed(lean_object* v_mvarId_2110_, lean_object* v_mkCmdStr_2111_, lean_object* v_selectedLocations_2112_, lean_object* v_params_2113_, lean_object* v_a_2114_, lean_object* v_replaceRange_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_){
_start:
{
lean_object* v_res_2121_; 
v_res_2121_ = lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__1(v_mvarId_2110_, v_mkCmdStr_2111_, v_selectedLocations_2112_, v_params_2113_, v_a_2114_, v_replaceRange_2115_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_);
lean_dec(v___y_2119_);
lean_dec_ref(v___y_2118_);
lean_dec(v___y_2117_);
lean_dec_ref(v___y_2116_);
return v_res_2121_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__17(void){
_start:
{
lean_object* v___x_2158_; 
v___x_2158_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2158_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__18(void){
_start:
{
lean_object* v___x_2159_; lean_object* v___x_2160_; 
v___x_2159_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__17, &lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__17_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__17);
v___x_2160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2160_, 0, v___x_2159_);
return v___x_2160_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__19(void){
_start:
{
lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; 
v___x_2161_ = lean_unsigned_to_nat(32u);
v___x_2162_ = lean_mk_empty_array_with_capacity(v___x_2161_);
v___x_2163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2163_, 0, v___x_2162_);
return v___x_2163_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__20(void){
_start:
{
size_t v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; lean_object* v___x_2169_; 
v___x_2164_ = ((size_t)5ULL);
v___x_2165_ = lean_unsigned_to_nat(0u);
v___x_2166_ = lean_unsigned_to_nat(32u);
v___x_2167_ = lean_mk_empty_array_with_capacity(v___x_2166_);
v___x_2168_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__19, &lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__19_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__19);
v___x_2169_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2169_, 0, v___x_2168_);
lean_ctor_set(v___x_2169_, 1, v___x_2167_);
lean_ctor_set(v___x_2169_, 2, v___x_2165_);
lean_ctor_set(v___x_2169_, 3, v___x_2165_);
lean_ctor_set_usize(v___x_2169_, 4, v___x_2164_);
return v___x_2169_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__21(void){
_start:
{
lean_object* v___x_2170_; lean_object* v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; 
v___x_2170_ = lean_box(1);
v___x_2171_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__20, &lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__20_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__20);
v___x_2172_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__18, &lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__18_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__18);
v___x_2173_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2173_, 0, v___x_2172_);
lean_ctor_set(v___x_2173_, 1, v___x_2171_);
lean_ctor_set(v___x_2173_, 2, v___x_2170_);
return v___x_2173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2(lean_object* v_params_2190_, lean_object* v_title_2191_, lean_object* v_mkCmdStr_2192_, uint8_t v_onlyGoal_2193_, lean_object* v_helpMsg_2194_, uint8_t v_onlyOne_2195_, lean_object* v___y_2196_){
_start:
{
lean_object* v___x_2198_; lean_object* v_a_2199_; lean_object* v___x_2201_; uint8_t v_isShared_2202_; uint8_t v_isSharedCheck_2297_; 
v___x_2198_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__0(v___y_2196_);
v_a_2199_ = lean_ctor_get(v___x_2198_, 0);
v_isSharedCheck_2297_ = !lean_is_exclusive(v___x_2198_);
if (v_isSharedCheck_2297_ == 0)
{
v___x_2201_ = v___x_2198_;
v_isShared_2202_ = v_isSharedCheck_2297_;
goto v_resetjp_2200_;
}
else
{
lean_inc(v_a_2199_);
lean_dec(v___x_2198_);
v___x_2201_ = lean_box(0);
v_isShared_2202_ = v_isSharedCheck_2297_;
goto v_resetjp_2200_;
}
v_resetjp_2200_:
{
lean_object* v_goals_2203_; lean_object* v_selectedLocations_2204_; lean_object* v_replaceRange_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; uint8_t v___x_2208_; lean_object* v_a_2210_; 
v_goals_2203_ = lean_ctor_get(v_params_2190_, 1);
v_selectedLocations_2204_ = lean_ctor_get(v_params_2190_, 2);
lean_inc_ref(v_selectedLocations_2204_);
v_replaceRange_2205_ = lean_ctor_get(v_params_2190_, 3);
lean_inc_ref(v_replaceRange_2205_);
v___x_2206_ = lean_unsigned_to_nat(0u);
v___x_2207_ = lean_array_get_size(v_goals_2203_);
v___x_2208_ = lean_nat_dec_lt(v___x_2206_, v___x_2207_);
if (v___x_2208_ == 0)
{
lean_object* v___x_2235_; lean_object* v___x_2236_; 
lean_dec_ref(v_replaceRange_2205_);
lean_dec_ref(v_selectedLocations_2204_);
lean_del_object(v___x_2201_);
lean_dec(v_a_2199_);
lean_dec_ref(v_helpMsg_2194_);
lean_dec_ref(v_mkCmdStr_2192_);
lean_dec_ref(v_title_2191_);
lean_dec_ref(v_params_2190_);
v___x_2235_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__16));
v___x_2236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2236_, 0, v___x_2235_);
return v___x_2236_;
}
else
{
lean_object* v_mainGoal_2237_; lean_object* v_toInteractiveGoalCore_2238_; lean_object* v_mvarId_2239_; lean_object* v___f_2240_; lean_object* v___y_2242_; lean_object* v___y_2282_; lean_object* v___y_2283_; lean_object* v___y_2292_; 
v_mainGoal_2237_ = lean_array_fget_borrowed(v_goals_2203_, v___x_2206_);
v_toInteractiveGoalCore_2238_ = lean_ctor_get(v_mainGoal_2237_, 0);
lean_inc_ref(v_toInteractiveGoalCore_2238_);
v_mvarId_2239_ = lean_ctor_get(v_mainGoal_2237_, 3);
lean_inc_n(v_mvarId_2239_, 2);
lean_inc_ref(v_selectedLocations_2204_);
v___f_2240_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__1___boxed), 11, 6);
lean_closure_set(v___f_2240_, 0, v_mvarId_2239_);
lean_closure_set(v___f_2240_, 1, v_mkCmdStr_2192_);
lean_closure_set(v___f_2240_, 2, v_selectedLocations_2204_);
lean_closure_set(v___f_2240_, 3, v_params_2190_);
lean_closure_set(v___f_2240_, 4, v_a_2199_);
lean_closure_set(v___f_2240_, 5, v_replaceRange_2205_);
if (v_onlyOne_2195_ == 0)
{
lean_object* v___x_2295_; 
v___x_2295_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__29));
v___y_2292_ = v___x_2295_;
goto v___jp_2291_;
}
else
{
lean_object* v___x_2296_; 
v___x_2296_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__30));
v___y_2292_ = v___x_2296_;
goto v___jp_2291_;
}
v___jp_2241_:
{
lean_object* v___x_2243_; size_t v_sz_2244_; size_t v___x_2245_; lean_object* v___x_2246_; 
v___x_2243_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__2));
v_sz_2244_ = lean_array_size(v_selectedLocations_2204_);
v___x_2245_ = ((size_t)0ULL);
v___x_2246_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg(v_mvarId_2239_, v___y_2242_, v_onlyGoal_2193_, v_selectedLocations_2204_, v_sz_2244_, v___x_2245_, v___x_2243_);
lean_dec(v_mvarId_2239_);
if (lean_obj_tag(v___x_2246_) == 0)
{
lean_object* v_a_2247_; lean_object* v_fst_2248_; 
v_a_2247_ = lean_ctor_get(v___x_2246_, 0);
lean_inc(v_a_2247_);
lean_dec_ref_known(v___x_2246_, 1);
v_fst_2248_ = lean_ctor_get(v_a_2247_, 0);
lean_inc(v_fst_2248_);
lean_dec(v_a_2247_);
if (lean_obj_tag(v_fst_2248_) == 0)
{
lean_object* v___x_2249_; uint8_t v___x_2250_; 
v___x_2249_ = lean_array_get_size(v_selectedLocations_2204_);
lean_dec_ref(v_selectedLocations_2204_);
v___x_2250_ = lean_nat_dec_eq(v___x_2249_, v___x_2206_);
if (v___x_2250_ == 0)
{
lean_object* v_ctx_2251_; lean_object* v_val_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; 
lean_dec_ref(v_helpMsg_2194_);
v_ctx_2251_ = lean_ctor_get(v_toInteractiveGoalCore_2238_, 2);
lean_inc_ref(v_ctx_2251_);
lean_dec_ref(v_toInteractiveGoalCore_2238_);
v_val_2252_ = lean_ctor_get(v_ctx_2251_, 0);
lean_inc(v_val_2252_);
lean_dec_ref(v_ctx_2251_);
v___x_2253_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__21, &lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__21_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__21);
v___x_2254_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_val_2252_, v___x_2253_, v___f_2240_);
if (lean_obj_tag(v___x_2254_) == 0)
{
lean_object* v_a_2255_; 
v_a_2255_ = lean_ctor_get(v___x_2254_, 0);
lean_inc(v_a_2255_);
lean_dec_ref_known(v___x_2254_, 1);
v_a_2210_ = v_a_2255_;
goto v___jp_2209_;
}
else
{
lean_object* v_a_2256_; lean_object* v___x_2258_; uint8_t v_isShared_2259_; uint8_t v_isSharedCheck_2264_; 
lean_del_object(v___x_2201_);
lean_dec_ref(v_title_2191_);
v_a_2256_ = lean_ctor_get(v___x_2254_, 0);
v_isSharedCheck_2264_ = !lean_is_exclusive(v___x_2254_);
if (v_isSharedCheck_2264_ == 0)
{
v___x_2258_ = v___x_2254_;
v_isShared_2259_ = v_isSharedCheck_2264_;
goto v_resetjp_2257_;
}
else
{
lean_inc(v_a_2256_);
lean_dec(v___x_2254_);
v___x_2258_ = lean_box(0);
v_isShared_2259_ = v_isSharedCheck_2264_;
goto v_resetjp_2257_;
}
v_resetjp_2257_:
{
lean_object* v___x_2260_; lean_object* v___x_2262_; 
v___x_2260_ = l_Lean_Server_RequestError_ofIoError(v_a_2256_);
if (v_isShared_2259_ == 0)
{
lean_ctor_set(v___x_2258_, 0, v___x_2260_);
v___x_2262_ = v___x_2258_;
goto v_reusejp_2261_;
}
else
{
lean_object* v_reuseFailAlloc_2263_; 
v_reuseFailAlloc_2263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2263_, 0, v___x_2260_);
v___x_2262_ = v_reuseFailAlloc_2263_;
goto v_reusejp_2261_;
}
v_reusejp_2261_:
{
return v___x_2262_;
}
}
}
}
else
{
lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; 
lean_dec_ref(v___f_2240_);
lean_dec_ref(v_toInteractiveGoalCore_2238_);
v___x_2265_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__0));
v___x_2266_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg___closed__1));
v___x_2267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2267_, 0, v_helpMsg_2194_);
v___x_2268_ = lean_unsigned_to_nat(1u);
v___x_2269_ = lean_mk_empty_array_with_capacity(v___x_2268_);
v___x_2270_ = lean_array_push(v___x_2269_, v___x_2267_);
v___x_2271_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2271_, 0, v___x_2265_);
lean_ctor_set(v___x_2271_, 1, v___x_2266_);
lean_ctor_set(v___x_2271_, 2, v___x_2270_);
v_a_2210_ = v___x_2271_;
goto v___jp_2209_;
}
}
else
{
lean_object* v_val_2272_; 
lean_dec_ref(v___f_2240_);
lean_dec_ref(v_toInteractiveGoalCore_2238_);
lean_dec_ref(v_selectedLocations_2204_);
lean_dec_ref(v_helpMsg_2194_);
v_val_2272_ = lean_ctor_get(v_fst_2248_, 0);
lean_inc(v_val_2272_);
lean_dec_ref_known(v_fst_2248_, 1);
v_a_2210_ = v_val_2272_;
goto v___jp_2209_;
}
}
else
{
lean_object* v_a_2273_; lean_object* v___x_2275_; uint8_t v_isShared_2276_; uint8_t v_isSharedCheck_2280_; 
lean_dec_ref(v___f_2240_);
lean_dec_ref(v_toInteractiveGoalCore_2238_);
lean_dec_ref(v_selectedLocations_2204_);
lean_del_object(v___x_2201_);
lean_dec_ref(v_helpMsg_2194_);
lean_dec_ref(v_title_2191_);
v_a_2273_ = lean_ctor_get(v___x_2246_, 0);
v_isSharedCheck_2280_ = !lean_is_exclusive(v___x_2246_);
if (v_isSharedCheck_2280_ == 0)
{
v___x_2275_ = v___x_2246_;
v_isShared_2276_ = v_isSharedCheck_2280_;
goto v_resetjp_2274_;
}
else
{
lean_inc(v_a_2273_);
lean_dec(v___x_2246_);
v___x_2275_ = lean_box(0);
v_isShared_2276_ = v_isSharedCheck_2280_;
goto v_resetjp_2274_;
}
v_resetjp_2274_:
{
lean_object* v___x_2278_; 
if (v_isShared_2276_ == 0)
{
v___x_2278_ = v___x_2275_;
goto v_reusejp_2277_;
}
else
{
lean_object* v_reuseFailAlloc_2279_; 
v_reuseFailAlloc_2279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2279_, 0, v_a_2273_);
v___x_2278_ = v_reuseFailAlloc_2279_;
goto v_reusejp_2277_;
}
v_reusejp_2277_:
{
return v___x_2278_;
}
}
}
}
v___jp_2281_:
{
lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v_errorMsg_2286_; 
v___x_2284_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__22));
lean_inc_ref(v___y_2282_);
v___x_2285_ = lean_string_append(v___y_2282_, v___x_2284_);
v_errorMsg_2286_ = lean_string_append(v___x_2285_, v___y_2283_);
if (v_onlyOne_2195_ == 0)
{
v___y_2242_ = v_errorMsg_2286_;
goto v___jp_2241_;
}
else
{
lean_object* v___x_2287_; lean_object* v___x_2288_; uint8_t v___x_2289_; 
v___x_2287_ = lean_unsigned_to_nat(1u);
v___x_2288_ = lean_array_get_size(v_selectedLocations_2204_);
v___x_2289_ = lean_nat_dec_lt(v___x_2287_, v___x_2288_);
if (v___x_2289_ == 0)
{
v___y_2242_ = v_errorMsg_2286_;
goto v___jp_2241_;
}
else
{
lean_object* v___x_2290_; 
lean_dec_ref(v_errorMsg_2286_);
lean_dec_ref(v___f_2240_);
lean_dec(v_mvarId_2239_);
lean_dec_ref(v_toInteractiveGoalCore_2238_);
lean_dec_ref(v_selectedLocations_2204_);
lean_dec_ref(v_helpMsg_2194_);
v___x_2290_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__26));
v_a_2210_ = v___x_2290_;
goto v___jp_2209_;
}
}
}
v___jp_2291_:
{
if (v_onlyGoal_2193_ == 0)
{
lean_object* v___x_2293_; 
v___x_2293_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__27));
v___y_2282_ = v___y_2292_;
v___y_2283_ = v___x_2293_;
goto v___jp_2281_;
}
else
{
lean_object* v___x_2294_; 
v___x_2294_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__28));
v___y_2282_ = v___y_2292_;
v___y_2283_ = v___x_2294_;
goto v___jp_2281_;
}
}
}
v___jp_2209_:
{
lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2233_; 
v___x_2211_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__0));
v___x_2212_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__1));
v___x_2213_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_2213_, 0, v___x_2208_);
v___x_2214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2214_, 0, v___x_2212_);
lean_ctor_set(v___x_2214_, 1, v___x_2213_);
v___x_2215_ = lean_unsigned_to_nat(1u);
v___x_2216_ = lean_mk_empty_array_with_capacity(v___x_2215_);
lean_inc_ref_n(v___x_2216_, 2);
v___x_2217_ = lean_array_push(v___x_2216_, v___x_2214_);
v___x_2218_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__2));
v___x_2219_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__7));
v___x_2220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2220_, 0, v_title_2191_);
v___x_2221_ = lean_array_push(v___x_2216_, v___x_2220_);
v___x_2222_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2222_, 0, v___x_2218_);
lean_ctor_set(v___x_2222_, 1, v___x_2219_);
lean_ctor_set(v___x_2222_, 2, v___x_2221_);
v___x_2223_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__8));
v___x_2224_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___closed__12));
v___x_2225_ = lean_array_push(v___x_2216_, v_a_2210_);
v___x_2226_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2226_, 0, v___x_2223_);
lean_ctor_set(v___x_2226_, 1, v___x_2224_);
lean_ctor_set(v___x_2226_, 2, v___x_2225_);
v___x_2227_ = lean_unsigned_to_nat(2u);
v___x_2228_ = lean_mk_empty_array_with_capacity(v___x_2227_);
v___x_2229_ = lean_array_push(v___x_2228_, v___x_2222_);
v___x_2230_ = lean_array_push(v___x_2229_, v___x_2226_);
v___x_2231_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2231_, 0, v___x_2211_);
lean_ctor_set(v___x_2231_, 1, v___x_2217_);
lean_ctor_set(v___x_2231_, 2, v___x_2230_);
if (v_isShared_2202_ == 0)
{
lean_ctor_set(v___x_2201_, 0, v___x_2231_);
v___x_2233_ = v___x_2201_;
goto v_reusejp_2232_;
}
else
{
lean_object* v_reuseFailAlloc_2234_; 
v_reuseFailAlloc_2234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2234_, 0, v___x_2231_);
v___x_2233_ = v_reuseFailAlloc_2234_;
goto v_reusejp_2232_;
}
v_reusejp_2232_:
{
return v___x_2233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___boxed(lean_object* v_params_2298_, lean_object* v_title_2299_, lean_object* v_mkCmdStr_2300_, lean_object* v_onlyGoal_2301_, lean_object* v_helpMsg_2302_, lean_object* v_onlyOne_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_){
_start:
{
uint8_t v_onlyGoal_boxed_2306_; uint8_t v_onlyOne_boxed_2307_; lean_object* v_res_2308_; 
v_onlyGoal_boxed_2306_ = lean_unbox(v_onlyGoal_2301_);
v_onlyOne_boxed_2307_ = lean_unbox(v_onlyOne_2303_);
v_res_2308_ = lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2(v_params_2298_, v_title_2299_, v_mkCmdStr_2300_, v_onlyGoal_boxed_2306_, v_helpMsg_2302_, v_onlyOne_boxed_2307_, v___y_2304_);
lean_dec_ref(v___y_2304_);
return v_res_2308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0(lean_object* v_mkCmdStr_2309_, lean_object* v_helpMsg_2310_, lean_object* v_title_2311_, uint8_t v_onlyGoal_2312_, uint8_t v_onlyOne_2313_, lean_object* v_params_2314_, lean_object* v_a_2315_){
_start:
{
lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___f_2319_; lean_object* v___x_2320_; 
v___x_2317_ = lean_box(v_onlyGoal_2312_);
v___x_2318_ = lean_box(v_onlyOne_2313_);
v___f_2319_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___lam__2___boxed), 8, 6);
lean_closure_set(v___f_2319_, 0, v_params_2314_);
lean_closure_set(v___f_2319_, 1, v_title_2311_);
lean_closure_set(v___f_2319_, 2, v_mkCmdStr_2309_);
lean_closure_set(v___f_2319_, 3, v___x_2317_);
lean_closure_set(v___f_2319_, 4, v_helpMsg_2310_);
lean_closure_set(v___f_2319_, 5, v___x_2318_);
v___x_2320_ = l_Lean_Server_RequestM_asTask___redArg(v___f_2319_, v_a_2315_);
return v___x_2320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0___boxed(lean_object* v_mkCmdStr_2321_, lean_object* v_helpMsg_2322_, lean_object* v_title_2323_, lean_object* v_onlyGoal_2324_, lean_object* v_onlyOne_2325_, lean_object* v_params_2326_, lean_object* v_a_2327_, lean_object* v_a_2328_){
_start:
{
uint8_t v_onlyGoal_boxed_2329_; uint8_t v_onlyOne_boxed_2330_; lean_object* v_res_2331_; 
v_onlyGoal_boxed_2329_ = lean_unbox(v_onlyGoal_2324_);
v_onlyOne_boxed_2330_ = lean_unbox(v_onlyOne_2325_);
v_res_2331_ = lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0(v_mkCmdStr_2321_, v_helpMsg_2322_, v_title_2323_, v_onlyGoal_boxed_2329_, v_onlyOne_boxed_2330_, v_params_2326_, v_a_2327_);
lean_dec_ref(v_a_2327_);
return v_res_2331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc(lean_object* v_params_2335_, lean_object* v_a_2336_){
_start:
{
lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; uint8_t v___x_2341_; uint8_t v___x_2342_; lean_object* v___x_2343_; 
v___x_2338_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__0));
v___x_2339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__1));
v___x_2340_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___closed__2));
v___x_2341_ = 0;
v___x_2342_ = 1;
v___x_2343_ = lp_mathlib_mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0(v___x_2338_, v___x_2339_, v___x_2340_, v___x_2341_, v___x_2342_, v_params_2335_, v_a_2336_);
return v___x_2343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___boxed(lean_object* v_params_2344_, lean_object* v_a_2345_, lean_object* v_a_2346_){
_start:
{
lean_object* v_res_2347_; 
v_res_2347_ = lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc(v_params_2344_, v_a_2345_);
lean_dec_ref(v_a_2345_);
return v_res_2347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3(lean_object* v_00_u03b1_2348_, lean_object* v_lctx_2349_, lean_object* v_localInsts_2350_, lean_object* v_x_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_){
_start:
{
lean_object* v___x_2357_; 
v___x_2357_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___redArg(v_lctx_2349_, v_localInsts_2350_, v_x_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_);
return v___x_2357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3___boxed(lean_object* v_00_u03b1_2358_, lean_object* v_lctx_2359_, lean_object* v_localInsts_2360_, lean_object* v_x_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_){
_start:
{
lean_object* v_res_2367_; 
v_res_2367_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__3(v_00_u03b1_2358_, v_lctx_2359_, v_localInsts_2360_, v_x_2361_, v___y_2362_, v___y_2363_, v___y_2364_, v___y_2365_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2362_);
return v_res_2367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1(lean_object* v_mainGoalName_2368_, lean_object* v_errorMsg_2369_, uint8_t v___y_2370_, lean_object* v_as_2371_, size_t v_sz_2372_, size_t v_i_2373_, lean_object* v_b_2374_, lean_object* v___y_2375_){
_start:
{
lean_object* v___x_2377_; 
v___x_2377_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___redArg(v_mainGoalName_2368_, v_errorMsg_2369_, v___y_2370_, v_as_2371_, v_sz_2372_, v_i_2373_, v_b_2374_);
return v___x_2377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1___boxed(lean_object* v_mainGoalName_2378_, lean_object* v_errorMsg_2379_, lean_object* v___y_2380_, lean_object* v_as_2381_, lean_object* v_sz_2382_, lean_object* v_i_2383_, lean_object* v_b_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_){
_start:
{
uint8_t v___y_2695__boxed_2387_; size_t v_sz_boxed_2388_; size_t v_i_boxed_2389_; lean_object* v_res_2390_; 
v___y_2695__boxed_2387_ = lean_unbox(v___y_2380_);
v_sz_boxed_2388_ = lean_unbox_usize(v_sz_2382_);
lean_dec(v_sz_2382_);
v_i_boxed_2389_ = lean_unbox_usize(v_i_2383_);
lean_dec(v_i_2383_);
v_res_2390_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc_spec__0_spec__1(v_mainGoalName_2378_, v_errorMsg_2379_, v___y_2695__boxed_2387_, v_as_2381_, v_sz_boxed_2388_, v_i_boxed_2389_, v_b_2384_, v___y_2385_);
lean_dec_ref(v___y_2385_);
lean_dec_ref(v_as_2381_);
lean_dec(v_mainGoalName_2378_);
return v_res_2390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_2391_, lean_object* v_x_2392_){
_start:
{
lean_object* v___x_2393_; 
v___x_2393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2393_, 0, v_x_2392_);
lean_ctor_set(v___x_2393_, 1, v_expireTime_2391_);
return v___x_2393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__2(lean_object* v_val_2394_, lean_object* v___f_2395_, lean_object* v_x_2396_, lean_object* v___y_2397_){
_start:
{
if (lean_obj_tag(v_x_2396_) == 0)
{
lean_object* v_a_2399_; lean_object* v___x_2401_; uint8_t v_isShared_2402_; uint8_t v_isSharedCheck_2406_; 
lean_dec_ref(v___f_2395_);
v_a_2399_ = lean_ctor_get(v_x_2396_, 0);
v_isSharedCheck_2406_ = !lean_is_exclusive(v_x_2396_);
if (v_isSharedCheck_2406_ == 0)
{
v___x_2401_ = v_x_2396_;
v_isShared_2402_ = v_isSharedCheck_2406_;
goto v_resetjp_2400_;
}
else
{
lean_inc(v_a_2399_);
lean_dec(v_x_2396_);
v___x_2401_ = lean_box(0);
v_isShared_2402_ = v_isSharedCheck_2406_;
goto v_resetjp_2400_;
}
v_resetjp_2400_:
{
lean_object* v___x_2404_; 
if (v_isShared_2402_ == 0)
{
lean_ctor_set_tag(v___x_2401_, 1);
v___x_2404_ = v___x_2401_;
goto v_reusejp_2403_;
}
else
{
lean_object* v_reuseFailAlloc_2405_; 
v_reuseFailAlloc_2405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2405_, 0, v_a_2399_);
v___x_2404_ = v_reuseFailAlloc_2405_;
goto v_reusejp_2403_;
}
v_reusejp_2403_:
{
return v___x_2404_;
}
}
}
else
{
lean_object* v_a_2407_; lean_object* v___x_2409_; uint8_t v_isShared_2410_; uint8_t v_isSharedCheck_2423_; 
v_a_2407_ = lean_ctor_get(v_x_2396_, 0);
v_isSharedCheck_2423_ = !lean_is_exclusive(v_x_2396_);
if (v_isSharedCheck_2423_ == 0)
{
v___x_2409_ = v_x_2396_;
v_isShared_2410_ = v_isSharedCheck_2423_;
goto v_resetjp_2408_;
}
else
{
lean_inc(v_a_2407_);
lean_dec(v_x_2396_);
v___x_2409_ = lean_box(0);
v_isShared_2410_ = v_isSharedCheck_2423_;
goto v_resetjp_2408_;
}
v_resetjp_2408_:
{
lean_object* v___x_2411_; lean_object* v_objects_2412_; lean_object* v_expireTime_2413_; lean_object* v___f_2414_; lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v_fst_2417_; lean_object* v_snd_2418_; lean_object* v___x_2419_; lean_object* v___x_2421_; 
v___x_2411_ = lean_st_ref_take(v_val_2394_);
v_objects_2412_ = lean_ctor_get(v___x_2411_, 0);
lean_inc_ref(v_objects_2412_);
v_expireTime_2413_ = lean_ctor_get(v___x_2411_, 1);
lean_inc(v_expireTime_2413_);
lean_dec(v___x_2411_);
v___f_2414_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_2414_, 0, v_expireTime_2413_);
v___x_2415_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_a_2407_, v_objects_2412_);
v___x_2416_ = l_Prod_map___redArg(v___f_2395_, v___f_2414_, v___x_2415_);
v_fst_2417_ = lean_ctor_get(v___x_2416_, 0);
lean_inc(v_fst_2417_);
v_snd_2418_ = lean_ctor_get(v___x_2416_, 1);
lean_inc(v_snd_2418_);
lean_dec_ref(v___x_2416_);
v___x_2419_ = lean_st_ref_set(v_val_2394_, v_snd_2418_);
if (v_isShared_2410_ == 0)
{
lean_ctor_set_tag(v___x_2409_, 0);
lean_ctor_set(v___x_2409_, 0, v_fst_2417_);
v___x_2421_ = v___x_2409_;
goto v_reusejp_2420_;
}
else
{
lean_object* v_reuseFailAlloc_2422_; 
v_reuseFailAlloc_2422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2422_, 0, v_fst_2417_);
v___x_2421_ = v_reuseFailAlloc_2422_;
goto v_reusejp_2420_;
}
v_reusejp_2420_:
{
return v___x_2421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_2424_, lean_object* v___f_2425_, lean_object* v_x_2426_, lean_object* v___y_2427_, lean_object* v___y_2428_){
_start:
{
lean_object* v_res_2429_; 
v_res_2429_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__2(v_val_2424_, v___f_2425_, v_x_2426_, v___y_2427_);
lean_dec_ref(v___y_2427_);
lean_dec(v_val_2424_);
return v_res_2429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_2430_, uint64_t v_k_2431_){
_start:
{
if (lean_obj_tag(v_t_2430_) == 0)
{
lean_object* v_k_2432_; lean_object* v_v_2433_; lean_object* v_l_2434_; lean_object* v_r_2435_; uint64_t v___x_2436_; uint8_t v___x_2437_; 
v_k_2432_ = lean_ctor_get(v_t_2430_, 1);
v_v_2433_ = lean_ctor_get(v_t_2430_, 2);
v_l_2434_ = lean_ctor_get(v_t_2430_, 3);
v_r_2435_ = lean_ctor_get(v_t_2430_, 4);
v___x_2436_ = lean_unbox_uint64(v_k_2432_);
v___x_2437_ = lean_uint64_dec_lt(v_k_2431_, v___x_2436_);
if (v___x_2437_ == 0)
{
uint64_t v___x_2438_; uint8_t v___x_2439_; 
v___x_2438_ = lean_unbox_uint64(v_k_2432_);
v___x_2439_ = lean_uint64_dec_eq(v_k_2431_, v___x_2438_);
if (v___x_2439_ == 0)
{
v_t_2430_ = v_r_2435_;
goto _start;
}
else
{
lean_object* v___x_2441_; 
lean_inc(v_v_2433_);
v___x_2441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2441_, 0, v_v_2433_);
return v___x_2441_;
}
}
else
{
v_t_2430_ = v_l_2434_;
goto _start;
}
}
else
{
lean_object* v___x_2443_; 
v___x_2443_ = lean_box(0);
return v___x_2443_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_2444_, lean_object* v_k_2445_){
_start:
{
uint64_t v_k_boxed_2446_; lean_object* v_res_2447_; 
v_k_boxed_2446_ = lean_unbox_uint64(v_k_2445_);
lean_dec_ref(v_k_2445_);
v_res_2447_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_2444_, v_k_boxed_2446_);
lean_dec(v_t_2444_);
return v_res_2447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3(lean_object* v_method_2455_, lean_object* v_handler_2456_, lean_object* v___f_2457_, uint64_t v_seshId_2458_, lean_object* v_j_2459_, lean_object* v___y_2460_){
_start:
{
lean_object* v_rpcSessions_2462_; lean_object* v___x_2463_; 
v_rpcSessions_2462_ = lean_ctor_get(v___y_2460_, 0);
v___x_2463_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_2462_, v_seshId_2458_);
if (lean_obj_tag(v___x_2463_) == 1)
{
lean_object* v_val_2464_; lean_object* v___x_2465_; lean_object* v_objects_2466_; lean_object* v___x_2467_; 
v_val_2464_ = lean_ctor_get(v___x_2463_, 0);
lean_inc(v_val_2464_);
lean_dec_ref_known(v___x_2463_, 1);
v___x_2465_ = lean_st_ref_get(v_val_2464_);
v_objects_2466_ = lean_ctor_get(v___x_2465_, 0);
lean_inc_ref(v_objects_2466_);
lean_dec(v___x_2465_);
lean_inc(v_j_2459_);
v___x_2467_ = lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(v_j_2459_, v_objects_2466_);
lean_dec_ref(v_objects_2466_);
if (lean_obj_tag(v___x_2467_) == 0)
{
lean_object* v_a_2468_; lean_object* v___x_2470_; uint8_t v_isShared_2471_; uint8_t v_isSharedCheck_2488_; 
lean_dec(v_val_2464_);
lean_dec_ref(v___f_2457_);
lean_dec_ref(v_handler_2456_);
v_a_2468_ = lean_ctor_get(v___x_2467_, 0);
v_isSharedCheck_2488_ = !lean_is_exclusive(v___x_2467_);
if (v_isSharedCheck_2488_ == 0)
{
v___x_2470_ = v___x_2467_;
v_isShared_2471_ = v_isSharedCheck_2488_;
goto v_resetjp_2469_;
}
else
{
lean_inc(v_a_2468_);
lean_dec(v___x_2467_);
v___x_2470_ = lean_box(0);
v_isShared_2471_ = v_isSharedCheck_2488_;
goto v_resetjp_2469_;
}
v_resetjp_2469_:
{
uint8_t v___x_2472_; lean_object* v___x_2473_; uint8_t v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2486_; 
v___x_2472_ = 3;
v___x_2473_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_2474_ = 1;
v___x_2475_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_2455_, v___x_2474_);
v___x_2476_ = lean_string_append(v___x_2473_, v___x_2475_);
lean_dec_ref(v___x_2475_);
v___x_2477_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_2478_ = lean_string_append(v___x_2476_, v___x_2477_);
v___x_2479_ = l_Lean_Json_compress(v_j_2459_);
v___x_2480_ = lean_string_append(v___x_2478_, v___x_2479_);
lean_dec_ref(v___x_2479_);
v___x_2481_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_2482_ = lean_string_append(v___x_2480_, v___x_2481_);
v___x_2483_ = lean_string_append(v___x_2482_, v_a_2468_);
lean_dec(v_a_2468_);
v___x_2484_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_2484_, 0, v___x_2483_);
lean_ctor_set_uint8(v___x_2484_, sizeof(void*)*1, v___x_2472_);
if (v_isShared_2471_ == 0)
{
lean_ctor_set_tag(v___x_2470_, 1);
lean_ctor_set(v___x_2470_, 0, v___x_2484_);
v___x_2486_ = v___x_2470_;
goto v_reusejp_2485_;
}
else
{
lean_object* v_reuseFailAlloc_2487_; 
v_reuseFailAlloc_2487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2487_, 0, v___x_2484_);
v___x_2486_ = v_reuseFailAlloc_2487_;
goto v_reusejp_2485_;
}
v_reusejp_2485_:
{
return v___x_2486_;
}
}
}
else
{
lean_object* v_a_2489_; lean_object* v___x_2490_; 
lean_dec(v_j_2459_);
lean_dec(v_method_2455_);
v_a_2489_ = lean_ctor_get(v___x_2467_, 0);
lean_inc(v_a_2489_);
lean_dec_ref_known(v___x_2467_, 1);
lean_inc_ref(v___y_2460_);
v___x_2490_ = lean_apply_3(v_handler_2456_, v_a_2489_, v___y_2460_, lean_box(0));
if (lean_obj_tag(v___x_2490_) == 0)
{
lean_object* v_a_2491_; lean_object* v___f_2492_; lean_object* v___x_2493_; 
v_a_2491_ = lean_ctor_get(v___x_2490_, 0);
lean_inc(v_a_2491_);
lean_dec_ref_known(v___x_2490_, 1);
v___f_2492_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_2492_, 0, v_val_2464_);
lean_closure_set(v___f_2492_, 1, v___f_2457_);
v___x_2493_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_2491_, v___f_2492_, v___y_2460_);
return v___x_2493_;
}
else
{
lean_object* v_a_2494_; lean_object* v___x_2496_; uint8_t v_isShared_2497_; uint8_t v_isSharedCheck_2501_; 
lean_dec(v_val_2464_);
lean_dec_ref(v___f_2457_);
v_a_2494_ = lean_ctor_get(v___x_2490_, 0);
v_isSharedCheck_2501_ = !lean_is_exclusive(v___x_2490_);
if (v_isSharedCheck_2501_ == 0)
{
v___x_2496_ = v___x_2490_;
v_isShared_2497_ = v_isSharedCheck_2501_;
goto v_resetjp_2495_;
}
else
{
lean_inc(v_a_2494_);
lean_dec(v___x_2490_);
v___x_2496_ = lean_box(0);
v_isShared_2497_ = v_isSharedCheck_2501_;
goto v_resetjp_2495_;
}
v_resetjp_2495_:
{
lean_object* v___x_2499_; 
if (v_isShared_2497_ == 0)
{
v___x_2499_ = v___x_2496_;
goto v_reusejp_2498_;
}
else
{
lean_object* v_reuseFailAlloc_2500_; 
v_reuseFailAlloc_2500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2500_, 0, v_a_2494_);
v___x_2499_ = v_reuseFailAlloc_2500_;
goto v_reusejp_2498_;
}
v_reusejp_2498_:
{
return v___x_2499_;
}
}
}
}
}
else
{
lean_object* v___x_2502_; lean_object* v___x_2503_; 
lean_dec(v___x_2463_);
lean_dec(v_j_2459_);
lean_dec_ref(v___f_2457_);
lean_dec_ref(v_handler_2456_);
lean_dec(v_method_2455_);
v___x_2502_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_2503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2503_, 0, v___x_2502_);
return v___x_2503_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_2504_, lean_object* v_handler_2505_, lean_object* v___f_2506_, lean_object* v_seshId_2507_, lean_object* v_j_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_){
_start:
{
uint64_t v_seshId_boxed_2511_; lean_object* v_res_2512_; 
v_seshId_boxed_2511_ = lean_unbox_uint64(v_seshId_2507_);
lean_dec_ref(v_seshId_2507_);
v_res_2512_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3(v_method_2504_, v_handler_2505_, v___f_2506_, v_seshId_boxed_2511_, v_j_2508_, v___y_2509_);
lean_dec_ref(v___y_2509_);
return v_res_2512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__0(lean_object* v___y_2513_){
_start:
{
lean_inc(v___y_2513_);
return v___y_2513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_2514_){
_start:
{
lean_object* v_res_2515_; 
v_res_2515_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__0(v___y_2514_);
lean_dec(v___y_2514_);
return v_res_2515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0(lean_object* v_method_2517_, lean_object* v_handler_2518_){
_start:
{
lean_object* v___f_2519_; lean_object* v___f_2520_; 
v___f_2519_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___closed__0));
v___f_2520_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_2520_, 0, v_method_2517_);
lean_closure_set(v___f_2520_, 1, v_handler_2518_);
lean_closure_set(v___f_2520_, 2, v___f_2519_);
return v___f_2520_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__5(void){
_start:
{
lean_object* v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; 
v___x_2531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__4));
v___x_2532_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__3));
v___x_2533_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0(v___x_2532_, v___x_2531_);
return v___x_2533_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped(void){
_start:
{
lean_object* v___x_2534_; 
v___x_2534_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__5, &lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped___closed__5);
return v___x_2534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_2535_, lean_object* v_t_2536_, uint64_t v_k_2537_){
_start:
{
lean_object* v___x_2538_; 
v___x_2538_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_2536_, v_k_2537_);
return v___x_2538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_2539_, lean_object* v_t_2540_, lean_object* v_k_2541_){
_start:
{
uint64_t v_k_boxed_2542_; lean_object* v_res_2543_; 
v_k_boxed_2542_ = lean_unbox_uint64(v_k_2541_);
lean_dec_ref(v_k_2541_);
v_res_2543_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped_spec__0_spec__0(v_00_u03b4_2539_, v_t_2540_, v_k_boxed_2542_);
lean_dec(v_t_2540_);
return v_res_2543_;
}
}
static uint64_t _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__1(void){
_start:
{
lean_object* v___x_2545_; uint64_t v___x_2546_; 
v___x_2545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__0));
v___x_2546_ = lean_string_hash(v___x_2545_);
return v___x_2546_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__2(void){
_start:
{
uint64_t v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; 
v___x_2547_ = lean_uint64_once(&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__1, &lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__1);
v___x_2548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__0));
v___x_2549_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_2549_, 0, v___x_2548_);
lean_ctor_set_uint64(v___x_2549_, sizeof(void*)*1, v___x_2547_);
return v___x_2549_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__4(void){
_start:
{
lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; 
v___x_2551_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__3));
v___x_2552_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__2, &lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__2);
v___x_2553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2553_, 0, v___x_2552_);
lean_ctor_set(v___x_2553_, 1, v___x_2551_);
return v___x_2553_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel(void){
_start:
{
lean_object* v___x_2554_; 
v___x_2554_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__4, &lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel___closed__4);
return v___x_2554_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; 
v___x_2570_ = lean_box(0);
v___x_2571_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2572_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2572_, 0, v___x_2571_);
lean_ctor_set(v___x_2572_, 1, v___x_2570_);
return v___x_2572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2574_; lean_object* v___x_2575_; 
v___x_2574_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___closed__0);
v___x_2575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2575_, 0, v___x_2574_);
return v___x_2575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg___boxed(lean_object* v___y_2576_){
_start:
{
lean_object* v_res_2577_; 
v_res_2577_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg();
return v_res_2577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0(lean_object* v_00_u03b1_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_){
_start:
{
lean_object* v___x_2588_; 
v___x_2588_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg();
return v___x_2588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___boxed(lean_object* v_00_u03b1_2589_, lean_object* v___y_2590_, lean_object* v___y_2591_, lean_object* v___y_2592_, lean_object* v___y_2593_, lean_object* v___y_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_, lean_object* v___y_2598_){
_start:
{
lean_object* v_res_2599_; 
v_res_2599_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0(v_00_u03b1_2589_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_, v___y_2594_, v___y_2595_, v___y_2596_, v___y_2597_);
lean_dec(v___y_2597_);
lean_dec_ref(v___y_2596_);
lean_dec(v___y_2595_);
lean_dec_ref(v___y_2594_);
lean_dec(v___y_2593_);
lean_dec_ref(v___y_2592_);
lean_dec(v___y_2591_);
lean_dec_ref(v___y_2590_);
return v_res_2599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___lam__0(lean_object* v___x_2600_, lean_object* v___y_2601_){
_start:
{
lean_object* v___x_2602_; 
v___x_2602_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2602_, 0, v___x_2600_);
lean_ctor_set(v___x_2602_, 1, v___y_2601_);
return v___x_2602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1(lean_object* v_x_2604_, lean_object* v_a_2605_, lean_object* v_a_2606_, lean_object* v_a_2607_, lean_object* v_a_2608_, lean_object* v_a_2609_, lean_object* v_a_2610_, lean_object* v_a_2611_, lean_object* v_a_2612_){
_start:
{
lean_object* v___x_2614_; uint8_t v___x_2615_; 
v___x_2614_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_tacticConv_x3f___closed__1));
lean_inc(v_x_2604_);
v___x_2615_ = l_Lean_Syntax_isOfKind(v_x_2604_, v___x_2614_);
if (v___x_2615_ == 0)
{
lean_object* v___x_2616_; 
lean_dec(v_x_2604_);
v___x_2616_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1_spec__0___redArg();
return v___x_2616_;
}
else
{
lean_object* v_fileMap_2617_; lean_object* v___x_2618_; lean_object* v_stx_2619_; uint8_t v___x_2620_; lean_object* v___x_2621_; 
v_fileMap_2617_ = lean_ctor_get(v_a_2611_, 1);
v___x_2618_ = lean_unsigned_to_nat(0u);
v_stx_2619_ = l_Lean_Syntax_getArg(v_x_2604_, v___x_2618_);
lean_dec(v_x_2604_);
v___x_2620_ = 0;
lean_inc_ref(v_fileMap_2617_);
v___x_2621_ = l_Lean_FileMap_lspRangeOfStx_x3f(v_fileMap_2617_, v_stx_2619_, v___x_2620_);
if (lean_obj_tag(v___x_2621_) == 1)
{
lean_object* v_val_2622_; lean_object* v___x_2623_; lean_object* v_toModule_2624_; uint64_t v_javascriptHash_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___f_2632_; lean_object* v___x_2633_; 
v_val_2622_ = lean_ctor_get(v___x_2621_, 0);
lean_inc(v_val_2622_);
lean_dec_ref_known(v___x_2621_, 1);
v___x_2623_ = lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel;
v_toModule_2624_ = lean_ctor_get(v___x_2623_, 0);
v_javascriptHash_2625_ = lean_ctor_get_uint64(v_toModule_2624_, sizeof(void*)*1);
v___x_2626_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___closed__0));
v___x_2627_ = l_Lean_Lsp_instToJsonRange_toJson(v_val_2622_);
v___x_2628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2628_, 0, v___x_2626_);
lean_ctor_set(v___x_2628_, 1, v___x_2627_);
v___x_2629_ = lean_box(0);
v___x_2630_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2630_, 0, v___x_2628_);
lean_ctor_set(v___x_2630_, 1, v___x_2629_);
v___x_2631_ = l_Lean_Json_mkObj(v___x_2630_);
lean_dec_ref_known(v___x_2630_, 2);
v___f_2632_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___lam__0), 2, 1);
lean_closure_set(v___f_2632_, 0, v___x_2631_);
v___x_2633_ = l_Lean_Widget_savePanelWidgetInfo(v_javascriptHash_2625_, v___f_2632_, v_stx_2619_, v_a_2611_, v_a_2612_);
return v___x_2633_;
}
else
{
lean_object* v___x_2634_; lean_object* v___x_2635_; 
lean_dec(v___x_2621_);
lean_dec(v_stx_2619_);
v___x_2634_ = lean_box(0);
v___x_2635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2635_, 0, v___x_2634_);
return v___x_2635_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1___boxed(lean_object* v_x_2636_, lean_object* v_a_2637_, lean_object* v_a_2638_, lean_object* v_a_2639_, lean_object* v_a_2640_, lean_object* v_a_2641_, lean_object* v_a_2642_, lean_object* v_a_2643_, lean_object* v_a_2644_, lean_object* v_a_2645_){
_start:
{
lean_object* v_res_2646_; 
v_res_2646_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Widget__Conv______elabRules__Mathlib__Tactic__Conv__tacticConv_x3f__1(v_x_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_, v_a_2643_, v_a_2644_);
lean_dec(v_a_2644_);
lean_dec_ref(v_a_2643_);
lean_dec(v_a_2642_);
lean_dec_ref(v_a_2641_);
lean_dec(v_a_2640_);
lean_dec_ref(v_a_2639_);
lean_dec(v_a_2638_);
lean_dec_ref(v_a_2637_);
return v_res_2646_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Name(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_Conv(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Widget_Conv(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped = _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel_rpc___rpc__wrapped);
lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel = _init_lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_SelectionPanel);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Name(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Widget_Conv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Widget_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Widget_Conv(builtin);
}
#ifdef __cplusplus
}
#endif
