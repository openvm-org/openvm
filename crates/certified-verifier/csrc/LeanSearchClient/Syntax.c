// Lean compiler output
// Module: LeanSearchClient.Syntax
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Meta public meta import Lean.Meta.Tactic.TryThis public meta import LeanSearchClient.Basic public meta import Lean.Server.Utils public meta import Lean.Elab.Command
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
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Json_getArr_x3f(lean_object*);
lean_object* l_Lean_Json_getObjVal_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_IO_throwServerError___redArg(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_string_hash(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_io_getenv(lean_object*);
extern lean_object* lp_LeanSearchClient_leansearchclient_useragent;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_System_Uri_escapeUri(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_IO_Process_output(lean_object*, lean_object*);
lean_object* l_Lean_Json_parse(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Json_getStr_x3f(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_runTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_Elab_Term_saveState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_SavedState_restore(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_TSyntax_getString(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_String_quote(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
extern lean_object* lp_LeanSearchClient_leansearch_queries;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_LeanSearchClient_leansearchclient_backend;
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_LeanSearchClient_statesearch_queries;
extern lean_object* lp_LeanSearchClient_statesearch_revision;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCache;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchCache;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "LEANSEARCHCLIENT_LEANSEARCH_API_URL"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Could not obtain outer array from "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "; error: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Could not obtain inner array from "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "query"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__4_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "num_results"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "curl"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__7_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "-X"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__8_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "POST"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__9_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "--user-agent"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__10_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "-H"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__11_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "accept: application/json"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__12 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__12_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "Content-Type: application/json"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__13_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "--data"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__14_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__15;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__16;
static const lean_array_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__17 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__17_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Could not parse response from LeanSearch server, error: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__18 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__18_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "https://leansearch.net/search"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__19 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__19_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "error"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "schema"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "description"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "error: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "\ndescription: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__4_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "LEANSEARCHCLIENT_LEANSTATESEARCH_API_URL"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__5_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\?query="};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "&results="};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__7_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "&rev="};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__8_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "GET"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__9_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__10;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__11;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__12;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Could not contact LeanStateSearch server"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__13_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "https://premise-search.com/api/search"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__14_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__1 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__1_value;
static const lean_string_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "some "};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__2 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__3 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__3_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Nat_cast___at___00LeanSearchClient_instReprSearchResult_repr_spec__1(lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "name"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__5_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__6_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__7;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__9_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "type\?"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__10_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__10_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__11_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__12;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docString\?"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__13_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__13_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__14_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__15;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doc_url\?"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__16 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__16_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__16_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__17 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__17_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__18;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "kind\?"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__19 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__19_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__19_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__20 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__20_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__21 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__21_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__22;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__23;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__24 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__24_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__21_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__25 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__25_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_LeanSearchClient_LeanSearchClient_instReprSearchResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult___closed__0_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected JSON array, got '"};
static const lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__0 = (const lean_object*)&lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__0_value;
static const lean_string_object lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__1 = (const lean_object*)&lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0 = (const lean_object*)&lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0_value;
static const lean_string_object lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1 = (const lean_object*)&lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "result"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "kind"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "doc_url"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "docstring"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "type"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__4_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f(lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "doc"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f(lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "formal_type"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f(lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#check "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " -- "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion(lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = " (type: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion(lean_object*);
static const lean_array_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "apply "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "have : "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "rw ["};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__4_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "rw [← "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__5_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__2_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(177, 152, 123, 219, 220, 182, 189, 250)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__2_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "sorryAx"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__4_value),LEAN_SCALAR_PTR_LITERAL(196, 190, 164, 146, 38, 179, 69, 72)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__5_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__6_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__8_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__7_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__8_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__9;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_LeanSearchClient_LeanSearchClient_leanSearchServer___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "LeanSearch"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "https://leansearch.net/"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "#leansearch"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__3_value;
static const lean_closure_object lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__1_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__2_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__5_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__5_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedSearchServer = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__5_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getCommandSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getCommandSuggestions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTermSuggestions_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTermSuggestions_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTermSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTermSuggestions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTacticSuggestionGroups(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTacticSuggestionGroups___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 117, .m_capacity = 117, .m_length = 116, .m_data = " query should be a string that ends with a `.` or `\?`.\nNote this command sends your query to an external service at "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__0_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__1 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__1_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__2 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__2_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__3 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__3_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__4 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__4_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__5 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__5_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__6 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__6_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__7 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " Search Results"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__1_value;
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__2 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "From: "};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___closed__0 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "LeanSearchClient"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "leansearch_search_cmd"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__1_value),LEAN_SCALAR_PTR_LITERAL(134, 160, 107, 32, 45, 254, 76, 168)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leanSearchServer___closed__3_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__5_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__6_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__7_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__8_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__9_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__10_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__11_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value_aux_1),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__10_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value_aux_2),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__11_value),LEAN_SCALAR_PTR_LITERAL(47, 210, 191, 36, 232, 108, 89, 36)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__12_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__13_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__7_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__13_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__14_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__14_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__15 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__15_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__15_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__16 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__16_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__16_value;
static lean_once_cell_t lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__0;
static lean_once_cell_t lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1;
static lean_once_cell_t lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__2;
static lean_once_cell_t lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__3;
static lean_once_cell_t lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__4;
static lean_once_cell_t lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__0;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__1;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "search_cmd"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 81, 0, 87, 11, 58, 223, 32)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#search"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__14_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__5_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_search__cmd = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__5_value;
static lean_once_cell_t lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0;
static const lean_string_object lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__1_value)}};
static const lean_object* lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__2 = (const lean_object*)&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__2_value;
static lean_once_cell_t lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__1_value;
static lean_once_cell_t lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "leansearch"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Invalid backend "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = ", must be leansearch"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "leansearch_search_term"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 200, 110, 41, 94, 109, 8, 105)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__15_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__2_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__term = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTermImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTermImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_search__term___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "search_term"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__term___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__term___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__term___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__0_value),LEAN_SCALAR_PTR_LITERAL(88, 199, 222, 170, 42, 63, 36, 56)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__term___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__term___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__term___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__2_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_search__term = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__term___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_searchTermImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = ", should be leansearch"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_searchTermImpl___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_searchTermImpl___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTermImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTermImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "leansearch_search_tactic"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 170, 44, 72, 129, 186, 215, 45)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "withPosition"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__2_value),LEAN_SCALAR_PTR_LITERAL(246, 171, 180, 145, 132, 143, 108, 238)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__4_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__5_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__6_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__6_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__13_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__7_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__7_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__9_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__9_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__10_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__10_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__11_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__11_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "statesearch_search_tactic"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(8, 188, 33, 152, 30, 158, 1, 12)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "#statesearch"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__3_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__5_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__5_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Try these:"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "search_tactic"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(179, 52, 9, 158, 10, 226, 124, 222)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__2_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__14_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__3_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__4_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_search__tactic = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__4_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(lean_object* v_opts_1_, lean_object* v_opt_2_){
_start:
{
lean_object* v_name_3_; lean_object* v_defValue_4_; lean_object* v_map_5_; lean_object* v___x_6_; 
v_name_3_ = lean_ctor_get(v_opt_2_, 0);
v_defValue_4_ = lean_ctor_get(v_opt_2_, 1);
v_map_5_ = lean_ctor_get(v_opts_1_, 0);
v___x_6_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_5_, v_name_3_);
if (lean_obj_tag(v___x_6_) == 0)
{
lean_inc(v_defValue_4_);
return v_defValue_4_;
}
else
{
lean_object* v_val_7_; 
v_val_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_val_7_);
lean_dec_ref_known(v___x_6_, 1);
if (lean_obj_tag(v_val_7_) == 0)
{
lean_object* v_v_8_; 
v_v_8_ = lean_ctor_get(v_val_7_, 0);
lean_inc_ref(v_v_8_);
lean_dec_ref_known(v_val_7_, 1);
return v_v_8_;
}
else
{
lean_dec(v_val_7_);
lean_inc(v_defValue_4_);
return v_defValue_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0___boxed(lean_object* v_opts_9_, lean_object* v_opt_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(v_opts_9_, v_opt_10_);
lean_dec_ref(v_opt_10_);
lean_dec_ref(v_opts_9_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___redArg(lean_object* v_a_12_){
_start:
{
lean_object* v_options_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v_options_14_ = lean_ctor_get(v_a_12_, 2);
v___x_15_ = lp_LeanSearchClient_leansearchclient_useragent;
v___x_16_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(v_options_14_, v___x_15_);
v___x_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_17_, 0, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___redArg___boxed(lean_object* v_a_18_, lean_object* v_a_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_LeanSearchClient_LeanSearchClient_useragent___redArg(v_a_18_);
lean_dec_ref(v_a_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent(lean_object* v_a_21_, lean_object* v_a_22_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_LeanSearchClient_LeanSearchClient_useragent___redArg(v_a_21_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___boxed(lean_object* v_a_25_, lean_object* v_a_26_, lean_object* v_a_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_LeanSearchClient_LeanSearchClient_useragent(v_a_25_, v_a_26_);
lean_dec(v_a_26_);
lean_dec_ref(v_a_25_);
return v_res_28_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_box(0);
v___x_30_ = lean_unsigned_to_nat(16u);
v___x_31_ = lean_mk_array(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_32_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_, &lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2__once, _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_);
v___x_33_ = lean_unsigned_to_nat(0u);
v___x_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v___x_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_36_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_, &lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2__once, _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_);
v___x_37_ = lean_st_mk_ref(v___x_36_);
v___x_38_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2____boxed(lean_object* v_a_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_();
return v_res_40_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = lean_box(0);
v___x_42_ = lean_unsigned_to_nat(16u);
v___x_43_ = lean_mk_array(v___x_42_, v___x_41_);
return v___x_43_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_44_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_, &lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2__once, _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_);
v___x_45_ = lean_unsigned_to_nat(0u);
v___x_46_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_46_, 0, v___x_45_);
lean_ctor_set(v___x_46_, 1, v___x_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_48_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_, &lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2__once, _init_lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_);
v___x_49_ = lean_st_mk_ref(v___x_48_);
v___x_50_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2____boxed(lean_object* v_a_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_();
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_53_, lean_object* v_x_54_){
_start:
{
if (lean_obj_tag(v_x_54_) == 0)
{
return v_x_53_;
}
else
{
lean_object* v_key_55_; lean_object* v_value_56_; lean_object* v_tail_57_; lean_object* v___x_59_; uint8_t v_isShared_60_; uint8_t v_isSharedCheck_84_; 
v_key_55_ = lean_ctor_get(v_x_54_, 0);
v_value_56_ = lean_ctor_get(v_x_54_, 1);
v_tail_57_ = lean_ctor_get(v_x_54_, 2);
v_isSharedCheck_84_ = !lean_is_exclusive(v_x_54_);
if (v_isSharedCheck_84_ == 0)
{
v___x_59_ = v_x_54_;
v_isShared_60_ = v_isSharedCheck_84_;
goto v_resetjp_58_;
}
else
{
lean_inc(v_tail_57_);
lean_inc(v_value_56_);
lean_inc(v_key_55_);
lean_dec(v_x_54_);
v___x_59_ = lean_box(0);
v_isShared_60_ = v_isSharedCheck_84_;
goto v_resetjp_58_;
}
v_resetjp_58_:
{
lean_object* v_fst_61_; lean_object* v_snd_62_; lean_object* v___x_63_; uint64_t v___x_64_; uint64_t v___x_65_; uint64_t v___x_66_; uint64_t v___x_67_; uint64_t v___x_68_; uint64_t v_fold_69_; uint64_t v___x_70_; uint64_t v___x_71_; uint64_t v___x_72_; size_t v___x_73_; size_t v___x_74_; size_t v___x_75_; size_t v___x_76_; size_t v___x_77_; lean_object* v___x_78_; lean_object* v___x_80_; 
v_fst_61_ = lean_ctor_get(v_key_55_, 0);
v_snd_62_ = lean_ctor_get(v_key_55_, 1);
v___x_63_ = lean_array_get_size(v_x_53_);
v___x_64_ = lean_string_hash(v_fst_61_);
v___x_65_ = lean_uint64_of_nat(v_snd_62_);
v___x_66_ = lean_uint64_mix_hash(v___x_64_, v___x_65_);
v___x_67_ = 32ULL;
v___x_68_ = lean_uint64_shift_right(v___x_66_, v___x_67_);
v_fold_69_ = lean_uint64_xor(v___x_66_, v___x_68_);
v___x_70_ = 16ULL;
v___x_71_ = lean_uint64_shift_right(v_fold_69_, v___x_70_);
v___x_72_ = lean_uint64_xor(v_fold_69_, v___x_71_);
v___x_73_ = lean_uint64_to_usize(v___x_72_);
v___x_74_ = lean_usize_of_nat(v___x_63_);
v___x_75_ = ((size_t)1ULL);
v___x_76_ = lean_usize_sub(v___x_74_, v___x_75_);
v___x_77_ = lean_usize_land(v___x_73_, v___x_76_);
v___x_78_ = lean_array_uget_borrowed(v_x_53_, v___x_77_);
lean_inc(v___x_78_);
if (v_isShared_60_ == 0)
{
lean_ctor_set(v___x_59_, 2, v___x_78_);
v___x_80_ = v___x_59_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v_key_55_);
lean_ctor_set(v_reuseFailAlloc_83_, 1, v_value_56_);
lean_ctor_set(v_reuseFailAlloc_83_, 2, v___x_78_);
v___x_80_ = v_reuseFailAlloc_83_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
lean_object* v___x_81_; 
v___x_81_ = lean_array_uset(v_x_53_, v___x_77_, v___x_80_);
v_x_53_ = v___x_81_;
v_x_54_ = v_tail_57_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2___redArg(lean_object* v_i_85_, lean_object* v_source_86_, lean_object* v_target_87_){
_start:
{
lean_object* v___x_88_; uint8_t v___x_89_; 
v___x_88_ = lean_array_get_size(v_source_86_);
v___x_89_ = lean_nat_dec_lt(v_i_85_, v___x_88_);
if (v___x_89_ == 0)
{
lean_dec_ref(v_source_86_);
lean_dec(v_i_85_);
return v_target_87_;
}
else
{
lean_object* v_es_90_; lean_object* v___x_91_; lean_object* v_source_92_; lean_object* v_target_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v_es_90_ = lean_array_fget(v_source_86_, v_i_85_);
v___x_91_ = lean_box(0);
v_source_92_ = lean_array_fset(v_source_86_, v_i_85_, v___x_91_);
v_target_93_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(v_target_87_, v_es_90_);
v___x_94_ = lean_unsigned_to_nat(1u);
v___x_95_ = lean_nat_add(v_i_85_, v___x_94_);
lean_dec(v_i_85_);
v_i_85_ = v___x_95_;
v_source_86_ = v_source_92_;
v_target_87_ = v_target_93_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1___redArg(lean_object* v_data_97_){
_start:
{
lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v_nbuckets_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_98_ = lean_array_get_size(v_data_97_);
v___x_99_ = lean_unsigned_to_nat(2u);
v_nbuckets_100_ = lean_nat_mul(v___x_98_, v___x_99_);
v___x_101_ = lean_unsigned_to_nat(0u);
v___x_102_ = lean_box(0);
v___x_103_ = lean_mk_array(v_nbuckets_100_, v___x_102_);
v___x_104_ = lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2___redArg(v___x_101_, v_data_97_, v___x_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2___redArg(lean_object* v_a_105_, lean_object* v_b_106_, lean_object* v_x_107_){
_start:
{
if (lean_obj_tag(v_x_107_) == 0)
{
lean_dec(v_b_106_);
lean_dec_ref(v_a_105_);
return v_x_107_;
}
else
{
lean_object* v_key_108_; lean_object* v_value_109_; lean_object* v_tail_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_129_; 
v_key_108_ = lean_ctor_get(v_x_107_, 0);
v_value_109_ = lean_ctor_get(v_x_107_, 1);
v_tail_110_ = lean_ctor_get(v_x_107_, 2);
v_isSharedCheck_129_ = !lean_is_exclusive(v_x_107_);
if (v_isSharedCheck_129_ == 0)
{
v___x_112_ = v_x_107_;
v_isShared_113_ = v_isSharedCheck_129_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_tail_110_);
lean_inc(v_value_109_);
lean_inc(v_key_108_);
lean_dec(v_x_107_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_129_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
uint8_t v___y_115_; lean_object* v_fst_123_; lean_object* v_snd_124_; lean_object* v_fst_125_; lean_object* v_snd_126_; uint8_t v___x_127_; 
v_fst_123_ = lean_ctor_get(v_key_108_, 0);
v_snd_124_ = lean_ctor_get(v_key_108_, 1);
v_fst_125_ = lean_ctor_get(v_a_105_, 0);
v_snd_126_ = lean_ctor_get(v_a_105_, 1);
v___x_127_ = lean_string_dec_eq(v_fst_123_, v_fst_125_);
if (v___x_127_ == 0)
{
v___y_115_ = v___x_127_;
goto v___jp_114_;
}
else
{
uint8_t v___x_128_; 
v___x_128_ = lean_nat_dec_eq(v_snd_124_, v_snd_126_);
v___y_115_ = v___x_128_;
goto v___jp_114_;
}
v___jp_114_:
{
if (v___y_115_ == 0)
{
lean_object* v___x_116_; lean_object* v___x_118_; 
v___x_116_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2___redArg(v_a_105_, v_b_106_, v_tail_110_);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 2, v___x_116_);
v___x_118_ = v___x_112_;
goto v_reusejp_117_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v_key_108_);
lean_ctor_set(v_reuseFailAlloc_119_, 1, v_value_109_);
lean_ctor_set(v_reuseFailAlloc_119_, 2, v___x_116_);
v___x_118_ = v_reuseFailAlloc_119_;
goto v_reusejp_117_;
}
v_reusejp_117_:
{
return v___x_118_;
}
}
else
{
lean_object* v___x_121_; 
lean_dec(v_value_109_);
lean_dec(v_key_108_);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 1, v_b_106_);
lean_ctor_set(v___x_112_, 0, v_a_105_);
v___x_121_ = v___x_112_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_122_; 
v_reuseFailAlloc_122_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_122_, 0, v_a_105_);
lean_ctor_set(v_reuseFailAlloc_122_, 1, v_b_106_);
lean_ctor_set(v_reuseFailAlloc_122_, 2, v_tail_110_);
v___x_121_ = v_reuseFailAlloc_122_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
return v___x_121_;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg(lean_object* v_a_130_, lean_object* v_x_131_){
_start:
{
if (lean_obj_tag(v_x_131_) == 0)
{
uint8_t v___x_132_; 
v___x_132_ = 0;
return v___x_132_;
}
else
{
lean_object* v_key_133_; lean_object* v_tail_134_; uint8_t v___y_136_; lean_object* v_fst_138_; lean_object* v_snd_139_; lean_object* v_fst_140_; lean_object* v_snd_141_; uint8_t v___x_142_; 
v_key_133_ = lean_ctor_get(v_x_131_, 0);
v_tail_134_ = lean_ctor_get(v_x_131_, 2);
v_fst_138_ = lean_ctor_get(v_key_133_, 0);
v_snd_139_ = lean_ctor_get(v_key_133_, 1);
v_fst_140_ = lean_ctor_get(v_a_130_, 0);
v_snd_141_ = lean_ctor_get(v_a_130_, 1);
v___x_142_ = lean_string_dec_eq(v_fst_138_, v_fst_140_);
if (v___x_142_ == 0)
{
v___y_136_ = v___x_142_;
goto v___jp_135_;
}
else
{
uint8_t v___x_143_; 
v___x_143_ = lean_nat_dec_eq(v_snd_139_, v_snd_141_);
v___y_136_ = v___x_143_;
goto v___jp_135_;
}
v___jp_135_:
{
if (v___y_136_ == 0)
{
v_x_131_ = v_tail_134_;
goto _start;
}
else
{
return v___y_136_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg___boxed(lean_object* v_a_144_, lean_object* v_x_145_){
_start:
{
uint8_t v_res_146_; lean_object* v_r_147_; 
v_res_146_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg(v_a_144_, v_x_145_);
lean_dec(v_x_145_);
lean_dec_ref(v_a_144_);
v_r_147_ = lean_box(v_res_146_);
return v_r_147_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0___redArg(lean_object* v_m_148_, lean_object* v_a_149_, lean_object* v_b_150_){
_start:
{
lean_object* v_size_151_; lean_object* v_buckets_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_199_; 
v_size_151_ = lean_ctor_get(v_m_148_, 0);
v_buckets_152_ = lean_ctor_get(v_m_148_, 1);
v_isSharedCheck_199_ = !lean_is_exclusive(v_m_148_);
if (v_isSharedCheck_199_ == 0)
{
v___x_154_ = v_m_148_;
v_isShared_155_ = v_isSharedCheck_199_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_buckets_152_);
lean_inc(v_size_151_);
lean_dec(v_m_148_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_199_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v_fst_156_; lean_object* v_snd_157_; lean_object* v___x_158_; uint64_t v___x_159_; uint64_t v___x_160_; uint64_t v___x_161_; uint64_t v___x_162_; uint64_t v___x_163_; uint64_t v_fold_164_; uint64_t v___x_165_; uint64_t v___x_166_; uint64_t v___x_167_; size_t v___x_168_; size_t v___x_169_; size_t v___x_170_; size_t v___x_171_; size_t v___x_172_; lean_object* v_bkt_173_; uint8_t v___x_174_; 
v_fst_156_ = lean_ctor_get(v_a_149_, 0);
v_snd_157_ = lean_ctor_get(v_a_149_, 1);
v___x_158_ = lean_array_get_size(v_buckets_152_);
v___x_159_ = lean_string_hash(v_fst_156_);
v___x_160_ = lean_uint64_of_nat(v_snd_157_);
v___x_161_ = lean_uint64_mix_hash(v___x_159_, v___x_160_);
v___x_162_ = 32ULL;
v___x_163_ = lean_uint64_shift_right(v___x_161_, v___x_162_);
v_fold_164_ = lean_uint64_xor(v___x_161_, v___x_163_);
v___x_165_ = 16ULL;
v___x_166_ = lean_uint64_shift_right(v_fold_164_, v___x_165_);
v___x_167_ = lean_uint64_xor(v_fold_164_, v___x_166_);
v___x_168_ = lean_uint64_to_usize(v___x_167_);
v___x_169_ = lean_usize_of_nat(v___x_158_);
v___x_170_ = ((size_t)1ULL);
v___x_171_ = lean_usize_sub(v___x_169_, v___x_170_);
v___x_172_ = lean_usize_land(v___x_168_, v___x_171_);
v_bkt_173_ = lean_array_uget_borrowed(v_buckets_152_, v___x_172_);
v___x_174_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg(v_a_149_, v_bkt_173_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v_size_x27_176_; lean_object* v___x_177_; lean_object* v_buckets_x27_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_175_ = lean_unsigned_to_nat(1u);
v_size_x27_176_ = lean_nat_add(v_size_151_, v___x_175_);
lean_dec(v_size_151_);
lean_inc(v_bkt_173_);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v_a_149_);
lean_ctor_set(v___x_177_, 1, v_b_150_);
lean_ctor_set(v___x_177_, 2, v_bkt_173_);
v_buckets_x27_178_ = lean_array_uset(v_buckets_152_, v___x_172_, v___x_177_);
v___x_179_ = lean_unsigned_to_nat(4u);
v___x_180_ = lean_nat_mul(v_size_x27_176_, v___x_179_);
v___x_181_ = lean_unsigned_to_nat(3u);
v___x_182_ = lean_nat_div(v___x_180_, v___x_181_);
lean_dec(v___x_180_);
v___x_183_ = lean_array_get_size(v_buckets_x27_178_);
v___x_184_ = lean_nat_dec_le(v___x_182_, v___x_183_);
lean_dec(v___x_182_);
if (v___x_184_ == 0)
{
lean_object* v_val_185_; lean_object* v___x_187_; 
v_val_185_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1___redArg(v_buckets_x27_178_);
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v_val_185_);
lean_ctor_set(v___x_154_, 0, v_size_x27_176_);
v___x_187_ = v___x_154_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_size_x27_176_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v_val_185_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
else
{
lean_object* v___x_190_; 
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v_buckets_x27_178_);
lean_ctor_set(v___x_154_, 0, v_size_x27_176_);
v___x_190_ = v___x_154_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v_size_x27_176_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v_buckets_x27_178_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
else
{
lean_object* v___x_192_; lean_object* v_buckets_x27_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_197_; 
lean_inc(v_bkt_173_);
v___x_192_ = lean_box(0);
v_buckets_x27_193_ = lean_array_uset(v_buckets_152_, v___x_172_, v___x_192_);
v___x_194_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2___redArg(v_a_149_, v_b_150_, v_bkt_173_);
v___x_195_ = lean_array_uset(v_buckets_x27_193_, v___x_172_, v___x_194_);
if (v_isShared_155_ == 0)
{
lean_ctor_set(v___x_154_, 1, v___x_195_);
v___x_197_ = v___x_154_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v_size_151_);
lean_ctor_set(v_reuseFailAlloc_198_, 1, v___x_195_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg(lean_object* v_a_200_, lean_object* v_x_201_){
_start:
{
if (lean_obj_tag(v_x_201_) == 0)
{
lean_object* v___x_202_; 
v___x_202_ = lean_box(0);
return v___x_202_;
}
else
{
lean_object* v_key_203_; lean_object* v_value_204_; lean_object* v_tail_205_; uint8_t v___y_207_; lean_object* v_fst_210_; lean_object* v_snd_211_; lean_object* v_fst_212_; lean_object* v_snd_213_; uint8_t v___x_214_; 
v_key_203_ = lean_ctor_get(v_x_201_, 0);
v_value_204_ = lean_ctor_get(v_x_201_, 1);
v_tail_205_ = lean_ctor_get(v_x_201_, 2);
v_fst_210_ = lean_ctor_get(v_key_203_, 0);
v_snd_211_ = lean_ctor_get(v_key_203_, 1);
v_fst_212_ = lean_ctor_get(v_a_200_, 0);
v_snd_213_ = lean_ctor_get(v_a_200_, 1);
v___x_214_ = lean_string_dec_eq(v_fst_210_, v_fst_212_);
if (v___x_214_ == 0)
{
v___y_207_ = v___x_214_;
goto v___jp_206_;
}
else
{
uint8_t v___x_215_; 
v___x_215_ = lean_nat_dec_eq(v_snd_211_, v_snd_213_);
v___y_207_ = v___x_215_;
goto v___jp_206_;
}
v___jp_206_:
{
if (v___y_207_ == 0)
{
v_x_201_ = v_tail_205_;
goto _start;
}
else
{
lean_object* v___x_209_; 
lean_inc(v_value_204_);
v___x_209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_209_, 0, v_value_204_);
return v___x_209_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg___boxed(lean_object* v_a_216_, lean_object* v_x_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg(v_a_216_, v_x_217_);
lean_dec(v_x_217_);
lean_dec_ref(v_a_216_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg(lean_object* v_m_219_, lean_object* v_a_220_){
_start:
{
lean_object* v_buckets_221_; lean_object* v_fst_222_; lean_object* v_snd_223_; lean_object* v___x_224_; uint64_t v___x_225_; uint64_t v___x_226_; uint64_t v___x_227_; uint64_t v___x_228_; uint64_t v___x_229_; uint64_t v_fold_230_; uint64_t v___x_231_; uint64_t v___x_232_; uint64_t v___x_233_; size_t v___x_234_; size_t v___x_235_; size_t v___x_236_; size_t v___x_237_; size_t v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v_buckets_221_ = lean_ctor_get(v_m_219_, 1);
v_fst_222_ = lean_ctor_get(v_a_220_, 0);
v_snd_223_ = lean_ctor_get(v_a_220_, 1);
v___x_224_ = lean_array_get_size(v_buckets_221_);
v___x_225_ = lean_string_hash(v_fst_222_);
v___x_226_ = lean_uint64_of_nat(v_snd_223_);
v___x_227_ = lean_uint64_mix_hash(v___x_225_, v___x_226_);
v___x_228_ = 32ULL;
v___x_229_ = lean_uint64_shift_right(v___x_227_, v___x_228_);
v_fold_230_ = lean_uint64_xor(v___x_227_, v___x_229_);
v___x_231_ = 16ULL;
v___x_232_ = lean_uint64_shift_right(v_fold_230_, v___x_231_);
v___x_233_ = lean_uint64_xor(v_fold_230_, v___x_232_);
v___x_234_ = lean_uint64_to_usize(v___x_233_);
v___x_235_ = lean_usize_of_nat(v___x_224_);
v___x_236_ = ((size_t)1ULL);
v___x_237_ = lean_usize_sub(v___x_235_, v___x_236_);
v___x_238_ = lean_usize_land(v___x_234_, v___x_237_);
v___x_239_ = lean_array_uget_borrowed(v_buckets_221_, v___x_238_);
v___x_240_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg(v_a_220_, v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg___boxed(lean_object* v_m_241_, lean_object* v_a_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg(v_m_241_, v_a_242_);
lean_dec_ref(v_a_242_);
lean_dec_ref(v_m_241_);
return v_res_243_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__15(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_260_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__8));
v___x_261_ = lean_unsigned_to_nat(11u);
v___x_262_ = lean_mk_empty_array_with_capacity(v___x_261_);
v___x_263_ = lean_array_push(v___x_262_, v___x_260_);
return v___x_263_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__16(void){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_264_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__9));
v___x_265_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__15, &lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__15_once, _init_lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__15);
v___x_266_ = lean_array_push(v___x_265_, v___x_264_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg(lean_object* v_s_271_, lean_object* v_num__results_272_, lean_object* v_a_273_){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_275_ = lp_LeanSearchClient_LeanSearchClient_leanSearchCache;
v___x_276_ = lean_st_ref_get(v___x_275_);
lean_inc(v_num__results_272_);
lean_inc_ref(v_s_271_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v_s_271_);
lean_ctor_set(v___x_277_, 1, v_num__results_272_);
v___x_278_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg(v___x_276_, v___x_277_);
lean_dec(v___x_276_);
if (lean_obj_tag(v___x_278_) == 0)
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v_js_283_; lean_object* v___y_284_; lean_object* v___y_374_; 
v___x_279_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__0));
v___x_280_ = lean_io_getenv(v___x_279_);
v___x_281_ = lean_box(0);
if (lean_obj_tag(v___x_280_) == 0)
{
lean_object* v___x_464_; 
v___x_464_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__19));
v___y_374_ = v___x_464_;
goto v___jp_373_;
}
else
{
lean_object* v_val_465_; 
v_val_465_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_val_465_);
lean_dec_ref_known(v___x_280_, 1);
v___y_374_ = v_val_465_;
goto v___jp_373_;
}
v___jp_282_:
{
lean_object* v___x_285_; 
lean_inc(v_js_283_);
v___x_285_ = l_Lean_Json_getArr_x3f(v_js_283_);
if (lean_obj_tag(v___x_285_) == 0)
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_321_; 
lean_dec_ref_known(v___x_277_, 2);
v_a_286_ = lean_ctor_get(v___x_285_, 0);
v_isSharedCheck_321_ = !lean_is_exclusive(v___x_285_);
if (v_isSharedCheck_321_ == 0)
{
v___x_288_ = v___x_285_;
v_isShared_289_ = v_isSharedCheck_321_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_285_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_321_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_290_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__1));
v___x_291_ = lean_unsigned_to_nat(80u);
v___x_292_ = l_Lean_Json_pretty(v_js_283_, v___x_291_);
v___x_293_ = lean_string_append(v___x_290_, v___x_292_);
lean_dec_ref(v___x_292_);
v___x_294_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__2));
v___x_295_ = lean_string_append(v___x_293_, v___x_294_);
v___x_296_ = lean_string_append(v___x_295_, v_a_286_);
lean_dec(v_a_286_);
v___x_297_ = l_Lean_IO_throwServerError___redArg(v___x_296_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_305_; 
lean_del_object(v___x_288_);
v_a_298_ = lean_ctor_get(v___x_297_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_305_ == 0)
{
v___x_300_ = v___x_297_;
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_297_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_301_ == 0)
{
v___x_303_ = v___x_300_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_298_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
}
else
{
lean_object* v_a_306_; lean_object* v___x_308_; uint8_t v_isShared_309_; uint8_t v_isSharedCheck_320_; 
v_a_306_ = lean_ctor_get(v___x_297_, 0);
v_isSharedCheck_320_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_320_ == 0)
{
v___x_308_ = v___x_297_;
v_isShared_309_ = v_isSharedCheck_320_;
goto v_resetjp_307_;
}
else
{
lean_inc(v_a_306_);
lean_dec(v___x_297_);
v___x_308_ = lean_box(0);
v_isShared_309_ = v_isSharedCheck_320_;
goto v_resetjp_307_;
}
v_resetjp_307_:
{
lean_object* v_ref_310_; lean_object* v___x_311_; lean_object* v___x_313_; 
v_ref_310_ = lean_ctor_get(v___y_284_, 5);
v___x_311_ = lean_io_error_to_string(v_a_306_);
if (v_isShared_289_ == 0)
{
lean_ctor_set_tag(v___x_288_, 3);
lean_ctor_set(v___x_288_, 0, v___x_311_);
v___x_313_ = v___x_288_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v___x_311_);
v___x_313_ = v_reuseFailAlloc_319_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_317_; 
v___x_314_ = l_Lean_MessageData_ofFormat(v___x_313_);
lean_inc(v_ref_310_);
v___x_315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_315_, 0, v_ref_310_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
if (v_isShared_309_ == 0)
{
lean_ctor_set(v___x_308_, 0, v___x_315_);
v___x_317_ = v___x_308_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
else
{
lean_object* v_a_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v_a_322_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_a_322_);
lean_dec_ref_known(v___x_285_, 1);
v___x_323_ = lean_unsigned_to_nat(0u);
v___x_324_ = lean_array_get(v___x_281_, v_a_322_, v___x_323_);
lean_dec(v_a_322_);
v___x_325_ = l_Lean_Json_getArr_x3f(v___x_324_);
if (lean_obj_tag(v___x_325_) == 0)
{
lean_object* v_a_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_361_; 
lean_dec_ref_known(v___x_277_, 2);
v_a_326_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_361_ == 0)
{
v___x_328_ = v___x_325_;
v_isShared_329_ = v_isSharedCheck_361_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_a_326_);
lean_dec(v___x_325_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_361_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v___x_330_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__3));
v___x_331_ = lean_unsigned_to_nat(80u);
v___x_332_ = l_Lean_Json_pretty(v_js_283_, v___x_331_);
v___x_333_ = lean_string_append(v___x_330_, v___x_332_);
lean_dec_ref(v___x_332_);
v___x_334_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__2));
v___x_335_ = lean_string_append(v___x_333_, v___x_334_);
v___x_336_ = lean_string_append(v___x_335_, v_a_326_);
lean_dec(v_a_326_);
v___x_337_ = l_Lean_IO_throwServerError___redArg(v___x_336_);
if (lean_obj_tag(v___x_337_) == 0)
{
lean_object* v_a_338_; lean_object* v___x_340_; uint8_t v_isShared_341_; uint8_t v_isSharedCheck_345_; 
lean_del_object(v___x_328_);
v_a_338_ = lean_ctor_get(v___x_337_, 0);
v_isSharedCheck_345_ = !lean_is_exclusive(v___x_337_);
if (v_isSharedCheck_345_ == 0)
{
v___x_340_ = v___x_337_;
v_isShared_341_ = v_isSharedCheck_345_;
goto v_resetjp_339_;
}
else
{
lean_inc(v_a_338_);
lean_dec(v___x_337_);
v___x_340_ = lean_box(0);
v_isShared_341_ = v_isSharedCheck_345_;
goto v_resetjp_339_;
}
v_resetjp_339_:
{
lean_object* v___x_343_; 
if (v_isShared_341_ == 0)
{
v___x_343_ = v___x_340_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v_a_338_);
v___x_343_ = v_reuseFailAlloc_344_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
return v___x_343_;
}
}
}
else
{
lean_object* v_a_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_360_; 
v_a_346_ = lean_ctor_get(v___x_337_, 0);
v_isSharedCheck_360_ = !lean_is_exclusive(v___x_337_);
if (v_isSharedCheck_360_ == 0)
{
v___x_348_ = v___x_337_;
v_isShared_349_ = v_isSharedCheck_360_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_a_346_);
lean_dec(v___x_337_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_360_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v_ref_350_; lean_object* v___x_351_; lean_object* v___x_353_; 
v_ref_350_ = lean_ctor_get(v___y_284_, 5);
v___x_351_ = lean_io_error_to_string(v_a_346_);
if (v_isShared_329_ == 0)
{
lean_ctor_set_tag(v___x_328_, 3);
lean_ctor_set(v___x_328_, 0, v___x_351_);
v___x_353_ = v___x_328_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v___x_351_);
v___x_353_ = v_reuseFailAlloc_359_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_357_; 
v___x_354_ = l_Lean_MessageData_ofFormat(v___x_353_);
lean_inc(v_ref_350_);
v___x_355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_355_, 0, v_ref_350_);
lean_ctor_set(v___x_355_, 1, v___x_354_);
if (v_isShared_349_ == 0)
{
lean_ctor_set(v___x_348_, 0, v___x_355_);
v___x_357_ = v___x_348_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v___x_355_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
}
}
}
else
{
lean_object* v_a_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_372_; 
lean_dec(v_js_283_);
v_a_362_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_372_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_372_ == 0)
{
v___x_364_ = v___x_325_;
v_isShared_365_ = v_isSharedCheck_372_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_a_362_);
lean_dec(v___x_325_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_372_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_370_; 
v___x_366_ = lean_st_ref_take(v___x_275_);
lean_inc(v_a_362_);
v___x_367_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0___redArg(v___x_366_, v___x_277_, v_a_362_);
v___x_368_ = lean_st_ref_set(v___x_275_, v___x_367_);
if (v_isShared_365_ == 0)
{
lean_ctor_set_tag(v___x_364_, 0);
v___x_370_ = v___x_364_;
goto v_reusejp_369_;
}
else
{
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v_a_362_);
v___x_370_ = v_reuseFailAlloc_371_;
goto v_reusejp_369_;
}
v_reusejp_369_:
{
return v___x_370_;
}
}
}
}
}
v___jp_373_:
{
lean_object* v___x_375_; lean_object* v_a_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_463_; 
v___x_375_ = lp_LeanSearchClient_LeanSearchClient_useragent___redArg(v_a_273_);
v_a_376_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_463_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_463_ == 0)
{
v___x_378_ = v___x_375_;
v_isShared_379_ = v_isSharedCheck_463_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_a_376_);
lean_dec(v___x_375_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_463_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_380_; lean_object* v___x_382_; 
v___x_380_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__4));
if (v_isShared_379_ == 0)
{
lean_ctor_set_tag(v___x_378_, 3);
lean_ctor_set(v___x_378_, 0, v_s_271_);
v___x_382_ = v___x_378_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v_s_271_);
v___x_382_ = v_reuseFailAlloc_462_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; uint8_t v___x_417_; uint8_t v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_383_ = lean_unsigned_to_nat(1u);
v___x_384_ = lean_mk_empty_array_with_capacity(v___x_383_);
v___x_385_ = lean_array_push(v___x_384_, v___x_382_);
v___x_386_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
v___x_387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_387_, 0, v___x_380_);
lean_ctor_set(v___x_387_, 1, v___x_386_);
v___x_388_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__5));
v___x_389_ = l_Lean_JsonNumber_fromNat(v_num__results_272_);
v___x_390_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
v___x_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_388_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = lean_box(0);
v___x_393_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_391_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
v___x_394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_394_, 0, v___x_387_);
lean_ctor_set(v___x_394_, 1, v___x_393_);
v___x_395_ = l_Lean_Json_mkObj(v___x_394_);
lean_dec_ref_known(v___x_394_, 2);
v___x_396_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__6));
v___x_397_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__7));
v___x_398_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__10));
v___x_399_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__11));
v___x_400_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__12));
v___x_401_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__13));
v___x_402_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__14));
v___x_403_ = lean_unsigned_to_nat(80u);
v___x_404_ = l_Lean_Json_pretty(v___x_395_, v___x_403_);
v___x_405_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__16, &lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__16_once, _init_lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__16);
v___x_406_ = lean_array_push(v___x_405_, v___y_374_);
v___x_407_ = lean_array_push(v___x_406_, v___x_398_);
v___x_408_ = lean_array_push(v___x_407_, v_a_376_);
v___x_409_ = lean_array_push(v___x_408_, v___x_399_);
v___x_410_ = lean_array_push(v___x_409_, v___x_400_);
v___x_411_ = lean_array_push(v___x_410_, v___x_399_);
v___x_412_ = lean_array_push(v___x_411_, v___x_401_);
v___x_413_ = lean_array_push(v___x_412_, v___x_402_);
v___x_414_ = lean_array_push(v___x_413_, v___x_404_);
v___x_415_ = lean_box(0);
v___x_416_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__17));
v___x_417_ = 1;
v___x_418_ = 0;
v___x_419_ = lean_alloc_ctor(0, 5, 2);
lean_ctor_set(v___x_419_, 0, v___x_396_);
lean_ctor_set(v___x_419_, 1, v___x_397_);
lean_ctor_set(v___x_419_, 2, v___x_414_);
lean_ctor_set(v___x_419_, 3, v___x_415_);
lean_ctor_set(v___x_419_, 4, v___x_416_);
lean_ctor_set_uint8(v___x_419_, sizeof(void*)*5, v___x_417_);
lean_ctor_set_uint8(v___x_419_, sizeof(void*)*5 + 1, v___x_418_);
v___x_420_ = l_IO_Process_output(v___x_419_, v___x_415_);
if (lean_obj_tag(v___x_420_) == 0)
{
lean_object* v_a_421_; lean_object* v_stdout_422_; lean_object* v___x_423_; 
v_a_421_ = lean_ctor_get(v___x_420_, 0);
lean_inc(v_a_421_);
lean_dec_ref_known(v___x_420_, 1);
v_stdout_422_ = lean_ctor_get(v_a_421_, 0);
lean_inc_ref(v_stdout_422_);
lean_dec(v_a_421_);
v___x_423_ = l_Lean_Json_parse(v_stdout_422_);
if (lean_obj_tag(v___x_423_) == 0)
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_447_; 
v_a_424_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_447_ == 0)
{
v___x_426_ = v___x_423_;
v_isShared_427_ = v_isSharedCheck_447_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_423_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_447_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_428_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__18));
v___x_429_ = lean_string_append(v___x_428_, v_a_424_);
lean_dec(v_a_424_);
v___x_430_ = l_Lean_IO_throwServerError___redArg(v___x_429_);
if (lean_obj_tag(v___x_430_) == 0)
{
lean_object* v_a_431_; 
lean_del_object(v___x_426_);
v_a_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc(v_a_431_);
lean_dec_ref_known(v___x_430_, 1);
v_js_283_ = v_a_431_;
v___y_284_ = v_a_273_;
goto v___jp_282_;
}
else
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_446_; 
lean_dec_ref_known(v___x_277_, 2);
v_a_432_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_446_ == 0)
{
v___x_434_ = v___x_430_;
v_isShared_435_ = v_isSharedCheck_446_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_430_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_446_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v_ref_436_; lean_object* v___x_437_; lean_object* v___x_439_; 
v_ref_436_ = lean_ctor_get(v_a_273_, 5);
v___x_437_ = lean_io_error_to_string(v_a_432_);
if (v_isShared_427_ == 0)
{
lean_ctor_set_tag(v___x_426_, 3);
lean_ctor_set(v___x_426_, 0, v___x_437_);
v___x_439_ = v___x_426_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v___x_437_);
v___x_439_ = v_reuseFailAlloc_445_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_443_; 
v___x_440_ = l_Lean_MessageData_ofFormat(v___x_439_);
lean_inc(v_ref_436_);
v___x_441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_441_, 0, v_ref_436_);
lean_ctor_set(v___x_441_, 1, v___x_440_);
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 0, v___x_441_);
v___x_443_ = v___x_434_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v___x_441_);
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
else
{
lean_object* v_a_448_; 
v_a_448_ = lean_ctor_get(v___x_423_, 0);
lean_inc(v_a_448_);
lean_dec_ref_known(v___x_423_, 1);
v_js_283_ = v_a_448_;
v___y_284_ = v_a_273_;
goto v___jp_282_;
}
}
else
{
lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_461_; 
lean_dec_ref_known(v___x_277_, 2);
v_a_449_ = lean_ctor_get(v___x_420_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_420_);
if (v_isSharedCheck_461_ == 0)
{
v___x_451_ = v___x_420_;
v_isShared_452_ = v_isSharedCheck_461_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_420_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_461_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v_ref_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_459_; 
v_ref_453_ = lean_ctor_get(v_a_273_, 5);
v___x_454_ = lean_io_error_to_string(v_a_449_);
v___x_455_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
v___x_456_ = l_Lean_MessageData_ofFormat(v___x_455_);
lean_inc(v_ref_453_);
v___x_457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_457_, 0, v_ref_453_);
lean_ctor_set(v___x_457_, 1, v___x_456_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 0, v___x_457_);
v___x_459_ = v___x_451_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v___x_457_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
}
}
}
else
{
lean_object* v_val_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_473_; 
lean_dec_ref_known(v___x_277_, 2);
lean_dec(v_num__results_272_);
lean_dec_ref(v_s_271_);
v_val_466_ = lean_ctor_get(v___x_278_, 0);
v_isSharedCheck_473_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_473_ == 0)
{
v___x_468_ = v___x_278_;
v_isShared_469_ = v_isSharedCheck_473_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_val_466_);
lean_dec(v___x_278_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_473_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v___x_471_; 
if (v_isShared_469_ == 0)
{
lean_ctor_set_tag(v___x_468_, 0);
v___x_471_ = v___x_468_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_472_; 
v_reuseFailAlloc_472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_472_, 0, v_val_466_);
v___x_471_ = v_reuseFailAlloc_472_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
return v___x_471_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___boxed(lean_object* v_s_474_, lean_object* v_num__results_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg(v_s_474_, v_num__results_475_, v_a_476_);
lean_dec_ref(v_a_476_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson(lean_object* v_s_479_, lean_object* v_num__results_480_, lean_object* v_a_481_, lean_object* v_a_482_){
_start:
{
lean_object* v___x_484_; 
v___x_484_ = lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg(v_s_479_, v_num__results_480_, v_a_481_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___boxed(lean_object* v_s_485_, lean_object* v_num__results_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson(v_s_485_, v_num__results_486_, v_a_487_, v_a_488_);
lean_dec(v_a_488_);
lean_dec_ref(v_a_487_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0(lean_object* v_00_u03b2_491_, lean_object* v_m_492_, lean_object* v_a_493_, lean_object* v_b_494_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0___redArg(v_m_492_, v_a_493_, v_b_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1(lean_object* v_00_u03b2_496_, lean_object* v_m_497_, lean_object* v_a_498_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___redArg(v_m_497_, v_a_498_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1___boxed(lean_object* v_00_u03b2_500_, lean_object* v_m_501_, lean_object* v_a_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1(v_00_u03b2_500_, v_m_501_, v_a_502_);
lean_dec_ref(v_a_502_);
lean_dec_ref(v_m_501_);
return v_res_503_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0(lean_object* v_00_u03b2_504_, lean_object* v_a_505_, lean_object* v_x_506_){
_start:
{
uint8_t v___x_507_; 
v___x_507_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___redArg(v_a_505_, v_x_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0___boxed(lean_object* v_00_u03b2_508_, lean_object* v_a_509_, lean_object* v_x_510_){
_start:
{
uint8_t v_res_511_; lean_object* v_r_512_; 
v_res_511_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__0(v_00_u03b2_508_, v_a_509_, v_x_510_);
lean_dec(v_x_510_);
lean_dec_ref(v_a_509_);
v_r_512_ = lean_box(v_res_511_);
return v_r_512_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1(lean_object* v_00_u03b2_513_, lean_object* v_data_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1___redArg(v_data_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2(lean_object* v_00_u03b2_516_, lean_object* v_a_517_, lean_object* v_b_518_, lean_object* v_x_519_){
_start:
{
lean_object* v___x_520_; 
v___x_520_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__2___redArg(v_a_517_, v_b_518_, v_x_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4(lean_object* v_00_u03b2_521_, lean_object* v_a_522_, lean_object* v_x_523_){
_start:
{
lean_object* v___x_524_; 
v___x_524_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___redArg(v_a_522_, v_x_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4___boxed(lean_object* v_00_u03b2_525_, lean_object* v_a_526_, lean_object* v_x_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLeanSearchQueryJson_spec__1_spec__4(v_00_u03b2_525_, v_a_526_, v_x_527_);
lean_dec(v_x_527_);
lean_dec_ref(v_a_526_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_529_, lean_object* v_i_530_, lean_object* v_source_531_, lean_object* v_target_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2___redArg(v_i_530_, v_source_531_, v_target_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_534_, lean_object* v_x_535_, lean_object* v_x_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLeanSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(v_x_535_, v_x_536_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg(lean_object* v_a_538_, lean_object* v_x_539_){
_start:
{
if (lean_obj_tag(v_x_539_) == 0)
{
lean_object* v___x_540_; 
v___x_540_ = lean_box(0);
return v___x_540_;
}
else
{
lean_object* v_key_541_; lean_object* v_value_542_; lean_object* v_tail_543_; uint8_t v___y_545_; lean_object* v_fst_548_; lean_object* v_snd_549_; lean_object* v_fst_550_; lean_object* v_snd_551_; uint8_t v___x_552_; 
v_key_541_ = lean_ctor_get(v_x_539_, 0);
v_value_542_ = lean_ctor_get(v_x_539_, 1);
v_tail_543_ = lean_ctor_get(v_x_539_, 2);
v_fst_548_ = lean_ctor_get(v_key_541_, 0);
v_snd_549_ = lean_ctor_get(v_key_541_, 1);
v_fst_550_ = lean_ctor_get(v_a_538_, 0);
v_snd_551_ = lean_ctor_get(v_a_538_, 1);
v___x_552_ = lean_string_dec_eq(v_fst_548_, v_fst_550_);
if (v___x_552_ == 0)
{
v___y_545_ = v___x_552_;
goto v___jp_544_;
}
else
{
lean_object* v_fst_553_; lean_object* v_snd_554_; lean_object* v_fst_555_; lean_object* v_snd_556_; uint8_t v___x_557_; 
v_fst_553_ = lean_ctor_get(v_snd_549_, 0);
v_snd_554_ = lean_ctor_get(v_snd_549_, 1);
v_fst_555_ = lean_ctor_get(v_snd_551_, 0);
v_snd_556_ = lean_ctor_get(v_snd_551_, 1);
v___x_557_ = lean_nat_dec_eq(v_fst_553_, v_fst_555_);
if (v___x_557_ == 0)
{
v___y_545_ = v___x_557_;
goto v___jp_544_;
}
else
{
uint8_t v___x_558_; 
v___x_558_ = lean_string_dec_eq(v_snd_554_, v_snd_556_);
v___y_545_ = v___x_558_;
goto v___jp_544_;
}
}
v___jp_544_:
{
if (v___y_545_ == 0)
{
v_x_539_ = v_tail_543_;
goto _start;
}
else
{
lean_object* v___x_547_; 
lean_inc(v_value_542_);
v___x_547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_547_, 0, v_value_542_);
return v___x_547_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg___boxed(lean_object* v_a_559_, lean_object* v_x_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg(v_a_559_, v_x_560_);
lean_dec(v_x_560_);
lean_dec_ref(v_a_559_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg(lean_object* v_m_562_, lean_object* v_a_563_){
_start:
{
lean_object* v_snd_564_; lean_object* v_buckets_565_; lean_object* v_fst_566_; lean_object* v_fst_567_; lean_object* v_snd_568_; lean_object* v___x_569_; uint64_t v___x_570_; uint64_t v___x_571_; uint64_t v___x_572_; uint64_t v___x_573_; uint64_t v___x_574_; uint64_t v___x_575_; uint64_t v___x_576_; uint64_t v_fold_577_; uint64_t v___x_578_; uint64_t v___x_579_; uint64_t v___x_580_; size_t v___x_581_; size_t v___x_582_; size_t v___x_583_; size_t v___x_584_; size_t v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; 
v_snd_564_ = lean_ctor_get(v_a_563_, 1);
v_buckets_565_ = lean_ctor_get(v_m_562_, 1);
v_fst_566_ = lean_ctor_get(v_a_563_, 0);
v_fst_567_ = lean_ctor_get(v_snd_564_, 0);
v_snd_568_ = lean_ctor_get(v_snd_564_, 1);
v___x_569_ = lean_array_get_size(v_buckets_565_);
v___x_570_ = lean_string_hash(v_fst_566_);
v___x_571_ = lean_uint64_of_nat(v_fst_567_);
v___x_572_ = lean_string_hash(v_snd_568_);
v___x_573_ = lean_uint64_mix_hash(v___x_571_, v___x_572_);
v___x_574_ = lean_uint64_mix_hash(v___x_570_, v___x_573_);
v___x_575_ = 32ULL;
v___x_576_ = lean_uint64_shift_right(v___x_574_, v___x_575_);
v_fold_577_ = lean_uint64_xor(v___x_574_, v___x_576_);
v___x_578_ = 16ULL;
v___x_579_ = lean_uint64_shift_right(v_fold_577_, v___x_578_);
v___x_580_ = lean_uint64_xor(v_fold_577_, v___x_579_);
v___x_581_ = lean_uint64_to_usize(v___x_580_);
v___x_582_ = lean_usize_of_nat(v___x_569_);
v___x_583_ = ((size_t)1ULL);
v___x_584_ = lean_usize_sub(v___x_582_, v___x_583_);
v___x_585_ = lean_usize_land(v___x_581_, v___x_584_);
v___x_586_ = lean_array_uget_borrowed(v_buckets_565_, v___x_585_);
v___x_587_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg(v_a_563_, v___x_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg___boxed(lean_object* v_m_588_, lean_object* v_a_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg(v_m_588_, v_a_589_);
lean_dec_ref(v_a_589_);
lean_dec_ref(v_m_588_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_591_, lean_object* v_x_592_){
_start:
{
if (lean_obj_tag(v_x_592_) == 0)
{
return v_x_591_;
}
else
{
lean_object* v_key_593_; lean_object* v_snd_594_; lean_object* v_value_595_; lean_object* v_tail_596_; lean_object* v___x_598_; uint8_t v_isShared_599_; uint8_t v_isSharedCheck_626_; 
v_key_593_ = lean_ctor_get(v_x_592_, 0);
lean_inc(v_key_593_);
v_snd_594_ = lean_ctor_get(v_key_593_, 1);
v_value_595_ = lean_ctor_get(v_x_592_, 1);
v_tail_596_ = lean_ctor_get(v_x_592_, 2);
v_isSharedCheck_626_ = !lean_is_exclusive(v_x_592_);
if (v_isSharedCheck_626_ == 0)
{
lean_object* v_unused_627_; 
v_unused_627_ = lean_ctor_get(v_x_592_, 0);
lean_dec(v_unused_627_);
v___x_598_ = v_x_592_;
v_isShared_599_ = v_isSharedCheck_626_;
goto v_resetjp_597_;
}
else
{
lean_inc(v_tail_596_);
lean_inc(v_value_595_);
lean_dec(v_x_592_);
v___x_598_ = lean_box(0);
v_isShared_599_ = v_isSharedCheck_626_;
goto v_resetjp_597_;
}
v_resetjp_597_:
{
lean_object* v_fst_600_; lean_object* v_fst_601_; lean_object* v_snd_602_; lean_object* v___x_603_; uint64_t v___x_604_; uint64_t v___x_605_; uint64_t v___x_606_; uint64_t v___x_607_; uint64_t v___x_608_; uint64_t v___x_609_; uint64_t v___x_610_; uint64_t v_fold_611_; uint64_t v___x_612_; uint64_t v___x_613_; uint64_t v___x_614_; size_t v___x_615_; size_t v___x_616_; size_t v___x_617_; size_t v___x_618_; size_t v___x_619_; lean_object* v___x_620_; lean_object* v___x_622_; 
v_fst_600_ = lean_ctor_get(v_key_593_, 0);
v_fst_601_ = lean_ctor_get(v_snd_594_, 0);
v_snd_602_ = lean_ctor_get(v_snd_594_, 1);
v___x_603_ = lean_array_get_size(v_x_591_);
v___x_604_ = lean_string_hash(v_fst_600_);
v___x_605_ = lean_uint64_of_nat(v_fst_601_);
v___x_606_ = lean_string_hash(v_snd_602_);
v___x_607_ = lean_uint64_mix_hash(v___x_605_, v___x_606_);
v___x_608_ = lean_uint64_mix_hash(v___x_604_, v___x_607_);
v___x_609_ = 32ULL;
v___x_610_ = lean_uint64_shift_right(v___x_608_, v___x_609_);
v_fold_611_ = lean_uint64_xor(v___x_608_, v___x_610_);
v___x_612_ = 16ULL;
v___x_613_ = lean_uint64_shift_right(v_fold_611_, v___x_612_);
v___x_614_ = lean_uint64_xor(v_fold_611_, v___x_613_);
v___x_615_ = lean_uint64_to_usize(v___x_614_);
v___x_616_ = lean_usize_of_nat(v___x_603_);
v___x_617_ = ((size_t)1ULL);
v___x_618_ = lean_usize_sub(v___x_616_, v___x_617_);
v___x_619_ = lean_usize_land(v___x_615_, v___x_618_);
v___x_620_ = lean_array_uget_borrowed(v_x_591_, v___x_619_);
lean_inc(v___x_620_);
if (v_isShared_599_ == 0)
{
lean_ctor_set(v___x_598_, 2, v___x_620_);
v___x_622_ = v___x_598_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v_key_593_);
lean_ctor_set(v_reuseFailAlloc_625_, 1, v_value_595_);
lean_ctor_set(v_reuseFailAlloc_625_, 2, v___x_620_);
v___x_622_ = v_reuseFailAlloc_625_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
lean_object* v___x_623_; 
v___x_623_ = lean_array_uset(v_x_591_, v___x_619_, v___x_622_);
v_x_591_ = v___x_623_;
v_x_592_ = v_tail_596_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2___redArg(lean_object* v_i_628_, lean_object* v_source_629_, lean_object* v_target_630_){
_start:
{
lean_object* v___x_631_; uint8_t v___x_632_; 
v___x_631_ = lean_array_get_size(v_source_629_);
v___x_632_ = lean_nat_dec_lt(v_i_628_, v___x_631_);
if (v___x_632_ == 0)
{
lean_dec_ref(v_source_629_);
lean_dec(v_i_628_);
return v_target_630_;
}
else
{
lean_object* v_es_633_; lean_object* v___x_634_; lean_object* v_source_635_; lean_object* v_target_636_; lean_object* v___x_637_; lean_object* v___x_638_; 
v_es_633_ = lean_array_fget(v_source_629_, v_i_628_);
v___x_634_ = lean_box(0);
v_source_635_ = lean_array_fset(v_source_629_, v_i_628_, v___x_634_);
v_target_636_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(v_target_630_, v_es_633_);
v___x_637_ = lean_unsigned_to_nat(1u);
v___x_638_ = lean_nat_add(v_i_628_, v___x_637_);
lean_dec(v_i_628_);
v_i_628_ = v___x_638_;
v_source_629_ = v_source_635_;
v_target_630_ = v_target_636_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1___redArg(lean_object* v_data_640_){
_start:
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v_nbuckets_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_641_ = lean_array_get_size(v_data_640_);
v___x_642_ = lean_unsigned_to_nat(2u);
v_nbuckets_643_ = lean_nat_mul(v___x_641_, v___x_642_);
v___x_644_ = lean_unsigned_to_nat(0u);
v___x_645_ = lean_box(0);
v___x_646_ = lean_mk_array(v_nbuckets_643_, v___x_645_);
v___x_647_ = lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2___redArg(v___x_644_, v_data_640_, v___x_646_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2___redArg(lean_object* v_a_648_, lean_object* v_b_649_, lean_object* v_x_650_){
_start:
{
if (lean_obj_tag(v_x_650_) == 0)
{
lean_dec(v_b_649_);
lean_dec_ref(v_a_648_);
return v_x_650_;
}
else
{
lean_object* v_key_651_; lean_object* v_value_652_; lean_object* v_tail_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_677_; 
v_key_651_ = lean_ctor_get(v_x_650_, 0);
v_value_652_ = lean_ctor_get(v_x_650_, 1);
v_tail_653_ = lean_ctor_get(v_x_650_, 2);
v_isSharedCheck_677_ = !lean_is_exclusive(v_x_650_);
if (v_isSharedCheck_677_ == 0)
{
v___x_655_ = v_x_650_;
v_isShared_656_ = v_isSharedCheck_677_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_tail_653_);
lean_inc(v_value_652_);
lean_inc(v_key_651_);
lean_dec(v_x_650_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_677_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
uint8_t v___y_658_; lean_object* v_fst_666_; lean_object* v_snd_667_; lean_object* v_fst_668_; lean_object* v_snd_669_; uint8_t v___x_670_; 
v_fst_666_ = lean_ctor_get(v_key_651_, 0);
v_snd_667_ = lean_ctor_get(v_key_651_, 1);
v_fst_668_ = lean_ctor_get(v_a_648_, 0);
v_snd_669_ = lean_ctor_get(v_a_648_, 1);
v___x_670_ = lean_string_dec_eq(v_fst_666_, v_fst_668_);
if (v___x_670_ == 0)
{
v___y_658_ = v___x_670_;
goto v___jp_657_;
}
else
{
lean_object* v_fst_671_; lean_object* v_snd_672_; lean_object* v_fst_673_; lean_object* v_snd_674_; uint8_t v___x_675_; 
v_fst_671_ = lean_ctor_get(v_snd_667_, 0);
v_snd_672_ = lean_ctor_get(v_snd_667_, 1);
v_fst_673_ = lean_ctor_get(v_snd_669_, 0);
v_snd_674_ = lean_ctor_get(v_snd_669_, 1);
v___x_675_ = lean_nat_dec_eq(v_fst_671_, v_fst_673_);
if (v___x_675_ == 0)
{
v___y_658_ = v___x_675_;
goto v___jp_657_;
}
else
{
uint8_t v___x_676_; 
v___x_676_ = lean_string_dec_eq(v_snd_672_, v_snd_674_);
v___y_658_ = v___x_676_;
goto v___jp_657_;
}
}
v___jp_657_:
{
if (v___y_658_ == 0)
{
lean_object* v___x_659_; lean_object* v___x_661_; 
v___x_659_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2___redArg(v_a_648_, v_b_649_, v_tail_653_);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 2, v___x_659_);
v___x_661_ = v___x_655_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v_key_651_);
lean_ctor_set(v_reuseFailAlloc_662_, 1, v_value_652_);
lean_ctor_set(v_reuseFailAlloc_662_, 2, v___x_659_);
v___x_661_ = v_reuseFailAlloc_662_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
return v___x_661_;
}
}
else
{
lean_object* v___x_664_; 
lean_dec(v_value_652_);
lean_dec(v_key_651_);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 1, v_b_649_);
lean_ctor_set(v___x_655_, 0, v_a_648_);
v___x_664_ = v___x_655_;
goto v_reusejp_663_;
}
else
{
lean_object* v_reuseFailAlloc_665_; 
v_reuseFailAlloc_665_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_665_, 0, v_a_648_);
lean_ctor_set(v_reuseFailAlloc_665_, 1, v_b_649_);
lean_ctor_set(v_reuseFailAlloc_665_, 2, v_tail_653_);
v___x_664_ = v_reuseFailAlloc_665_;
goto v_reusejp_663_;
}
v_reusejp_663_:
{
return v___x_664_;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg(lean_object* v_a_678_, lean_object* v_x_679_){
_start:
{
if (lean_obj_tag(v_x_679_) == 0)
{
uint8_t v___x_680_; 
v___x_680_ = 0;
return v___x_680_;
}
else
{
lean_object* v_key_681_; lean_object* v_tail_682_; uint8_t v___y_684_; lean_object* v_fst_686_; lean_object* v_snd_687_; lean_object* v_fst_688_; lean_object* v_snd_689_; uint8_t v___x_690_; 
v_key_681_ = lean_ctor_get(v_x_679_, 0);
v_tail_682_ = lean_ctor_get(v_x_679_, 2);
v_fst_686_ = lean_ctor_get(v_key_681_, 0);
v_snd_687_ = lean_ctor_get(v_key_681_, 1);
v_fst_688_ = lean_ctor_get(v_a_678_, 0);
v_snd_689_ = lean_ctor_get(v_a_678_, 1);
v___x_690_ = lean_string_dec_eq(v_fst_686_, v_fst_688_);
if (v___x_690_ == 0)
{
v___y_684_ = v___x_690_;
goto v___jp_683_;
}
else
{
lean_object* v_fst_691_; lean_object* v_snd_692_; lean_object* v_fst_693_; lean_object* v_snd_694_; uint8_t v___x_695_; 
v_fst_691_ = lean_ctor_get(v_snd_687_, 0);
v_snd_692_ = lean_ctor_get(v_snd_687_, 1);
v_fst_693_ = lean_ctor_get(v_snd_689_, 0);
v_snd_694_ = lean_ctor_get(v_snd_689_, 1);
v___x_695_ = lean_nat_dec_eq(v_fst_691_, v_fst_693_);
if (v___x_695_ == 0)
{
v___y_684_ = v___x_695_;
goto v___jp_683_;
}
else
{
uint8_t v___x_696_; 
v___x_696_ = lean_string_dec_eq(v_snd_692_, v_snd_694_);
v___y_684_ = v___x_696_;
goto v___jp_683_;
}
}
v___jp_683_:
{
if (v___y_684_ == 0)
{
v_x_679_ = v_tail_682_;
goto _start;
}
else
{
return v___y_684_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg___boxed(lean_object* v_a_697_, lean_object* v_x_698_){
_start:
{
uint8_t v_res_699_; lean_object* v_r_700_; 
v_res_699_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg(v_a_697_, v_x_698_);
lean_dec(v_x_698_);
lean_dec_ref(v_a_697_);
v_r_700_ = lean_box(v_res_699_);
return v_r_700_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0___redArg(lean_object* v_m_701_, lean_object* v_a_702_, lean_object* v_b_703_){
_start:
{
lean_object* v_snd_704_; lean_object* v_size_705_; lean_object* v_buckets_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_756_; 
v_snd_704_ = lean_ctor_get(v_a_702_, 1);
v_size_705_ = lean_ctor_get(v_m_701_, 0);
v_buckets_706_ = lean_ctor_get(v_m_701_, 1);
v_isSharedCheck_756_ = !lean_is_exclusive(v_m_701_);
if (v_isSharedCheck_756_ == 0)
{
v___x_708_ = v_m_701_;
v_isShared_709_ = v_isSharedCheck_756_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_buckets_706_);
lean_inc(v_size_705_);
lean_dec(v_m_701_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_756_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
lean_object* v_fst_710_; lean_object* v_fst_711_; lean_object* v_snd_712_; lean_object* v___x_713_; uint64_t v___x_714_; uint64_t v___x_715_; uint64_t v___x_716_; uint64_t v___x_717_; uint64_t v___x_718_; uint64_t v___x_719_; uint64_t v___x_720_; uint64_t v_fold_721_; uint64_t v___x_722_; uint64_t v___x_723_; uint64_t v___x_724_; size_t v___x_725_; size_t v___x_726_; size_t v___x_727_; size_t v___x_728_; size_t v___x_729_; lean_object* v_bkt_730_; uint8_t v___x_731_; 
v_fst_710_ = lean_ctor_get(v_a_702_, 0);
v_fst_711_ = lean_ctor_get(v_snd_704_, 0);
v_snd_712_ = lean_ctor_get(v_snd_704_, 1);
v___x_713_ = lean_array_get_size(v_buckets_706_);
v___x_714_ = lean_string_hash(v_fst_710_);
v___x_715_ = lean_uint64_of_nat(v_fst_711_);
v___x_716_ = lean_string_hash(v_snd_712_);
v___x_717_ = lean_uint64_mix_hash(v___x_715_, v___x_716_);
v___x_718_ = lean_uint64_mix_hash(v___x_714_, v___x_717_);
v___x_719_ = 32ULL;
v___x_720_ = lean_uint64_shift_right(v___x_718_, v___x_719_);
v_fold_721_ = lean_uint64_xor(v___x_718_, v___x_720_);
v___x_722_ = 16ULL;
v___x_723_ = lean_uint64_shift_right(v_fold_721_, v___x_722_);
v___x_724_ = lean_uint64_xor(v_fold_721_, v___x_723_);
v___x_725_ = lean_uint64_to_usize(v___x_724_);
v___x_726_ = lean_usize_of_nat(v___x_713_);
v___x_727_ = ((size_t)1ULL);
v___x_728_ = lean_usize_sub(v___x_726_, v___x_727_);
v___x_729_ = lean_usize_land(v___x_725_, v___x_728_);
v_bkt_730_ = lean_array_uget_borrowed(v_buckets_706_, v___x_729_);
v___x_731_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg(v_a_702_, v_bkt_730_);
if (v___x_731_ == 0)
{
lean_object* v___x_732_; lean_object* v_size_x27_733_; lean_object* v___x_734_; lean_object* v_buckets_x27_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; uint8_t v___x_741_; 
v___x_732_ = lean_unsigned_to_nat(1u);
v_size_x27_733_ = lean_nat_add(v_size_705_, v___x_732_);
lean_dec(v_size_705_);
lean_inc(v_bkt_730_);
v___x_734_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_734_, 0, v_a_702_);
lean_ctor_set(v___x_734_, 1, v_b_703_);
lean_ctor_set(v___x_734_, 2, v_bkt_730_);
v_buckets_x27_735_ = lean_array_uset(v_buckets_706_, v___x_729_, v___x_734_);
v___x_736_ = lean_unsigned_to_nat(4u);
v___x_737_ = lean_nat_mul(v_size_x27_733_, v___x_736_);
v___x_738_ = lean_unsigned_to_nat(3u);
v___x_739_ = lean_nat_div(v___x_737_, v___x_738_);
lean_dec(v___x_737_);
v___x_740_ = lean_array_get_size(v_buckets_x27_735_);
v___x_741_ = lean_nat_dec_le(v___x_739_, v___x_740_);
lean_dec(v___x_739_);
if (v___x_741_ == 0)
{
lean_object* v_val_742_; lean_object* v___x_744_; 
v_val_742_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1___redArg(v_buckets_x27_735_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 1, v_val_742_);
lean_ctor_set(v___x_708_, 0, v_size_x27_733_);
v___x_744_ = v___x_708_;
goto v_reusejp_743_;
}
else
{
lean_object* v_reuseFailAlloc_745_; 
v_reuseFailAlloc_745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_745_, 0, v_size_x27_733_);
lean_ctor_set(v_reuseFailAlloc_745_, 1, v_val_742_);
v___x_744_ = v_reuseFailAlloc_745_;
goto v_reusejp_743_;
}
v_reusejp_743_:
{
return v___x_744_;
}
}
else
{
lean_object* v___x_747_; 
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 1, v_buckets_x27_735_);
lean_ctor_set(v___x_708_, 0, v_size_x27_733_);
v___x_747_ = v___x_708_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_size_x27_733_);
lean_ctor_set(v_reuseFailAlloc_748_, 1, v_buckets_x27_735_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
else
{
lean_object* v___x_749_; lean_object* v_buckets_x27_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_754_; 
lean_inc(v_bkt_730_);
v___x_749_ = lean_box(0);
v_buckets_x27_750_ = lean_array_uset(v_buckets_706_, v___x_729_, v___x_749_);
v___x_751_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2___redArg(v_a_702_, v_b_703_, v_bkt_730_);
v___x_752_ = lean_array_uset(v_buckets_x27_750_, v___x_729_, v___x_751_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 1, v___x_752_);
v___x_754_ = v___x_708_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v_size_705_);
lean_ctor_set(v_reuseFailAlloc_755_, 1, v___x_752_);
v___x_754_ = v_reuseFailAlloc_755_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
return v___x_754_;
}
}
}
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__10(void){
_start:
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_767_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__8));
v___x_768_ = lean_unsigned_to_nat(5u);
v___x_769_ = lean_mk_empty_array_with_capacity(v___x_768_);
v___x_770_ = lean_array_push(v___x_769_, v___x_767_);
return v___x_770_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__11(void){
_start:
{
lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; 
v___x_771_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__9));
v___x_772_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__10, &lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__10_once, _init_lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__10);
v___x_773_ = lean_array_push(v___x_772_, v___x_771_);
return v___x_773_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__12(void){
_start:
{
lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; 
v___x_774_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__10));
v___x_775_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__11, &lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__11_once, _init_lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__11);
v___x_776_ = lean_array_push(v___x_775_, v___x_774_);
return v___x_776_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg(lean_object* v_s_779_, lean_object* v_num__results_780_, lean_object* v_rev_781_, lean_object* v_a_782_){
_start:
{
lean_object* v___x_784_; lean_object* v_js_786_; lean_object* v___y_787_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v___x_784_ = lp_LeanSearchClient_LeanSearchClient_stateSearchCache;
v___x_932_ = lean_st_ref_get(v___x_784_);
lean_inc_ref(v_rev_781_);
lean_inc(v_num__results_780_);
v___x_933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_933_, 0, v_num__results_780_);
lean_ctor_set(v___x_933_, 1, v_rev_781_);
lean_inc_ref(v_s_779_);
v___x_934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_934_, 0, v_s_779_);
lean_ctor_set(v___x_934_, 1, v___x_933_);
v___x_935_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg(v___x_932_, v___x_934_);
lean_dec_ref_known(v___x_934_, 2);
lean_dec(v___x_932_);
if (lean_obj_tag(v___x_935_) == 0)
{
lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___y_939_; 
v___x_936_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__5));
v___x_937_ = lean_io_getenv(v___x_936_);
if (lean_obj_tag(v___x_937_) == 0)
{
lean_object* v___x_1010_; 
v___x_1010_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__14));
v___y_939_ = v___x_1010_;
goto v___jp_938_;
}
else
{
lean_object* v_val_1011_; 
v_val_1011_ = lean_ctor_get(v___x_937_, 0);
lean_inc(v_val_1011_);
lean_dec_ref_known(v___x_937_, 1);
v___y_939_ = v_val_1011_;
goto v___jp_938_;
}
v___jp_938_:
{
lean_object* v___x_940_; lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_1009_; 
v___x_940_ = lp_LeanSearchClient_LeanSearchClient_useragent___redArg(v_a_782_);
v_a_941_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_1009_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_1009_ == 0)
{
v___x_943_ = v___x_940_;
v_isShared_944_ = v_isSharedCheck_1009_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_940_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_1009_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; uint8_t v___x_963_; uint8_t v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; 
lean_inc_ref(v_s_779_);
v___x_945_ = l_System_Uri_escapeUri(v_s_779_);
v___x_946_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__6));
v___x_947_ = lean_string_append(v___x_946_, v___x_945_);
lean_dec_ref(v___x_945_);
v___x_948_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__7));
v___x_949_ = lean_string_append(v___x_947_, v___x_948_);
lean_inc(v_num__results_780_);
v___x_950_ = l_Nat_reprFast(v_num__results_780_);
v___x_951_ = lean_string_append(v___x_949_, v___x_950_);
lean_dec_ref(v___x_950_);
v___x_952_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__8));
v___x_953_ = lean_string_append(v___x_951_, v___x_952_);
v___x_954_ = lean_string_append(v___x_953_, v_rev_781_);
v___x_955_ = lean_string_append(v___y_939_, v___x_954_);
lean_dec_ref(v___x_954_);
v___x_956_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__6));
v___x_957_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__7));
v___x_958_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__12, &lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__12_once, _init_lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__12);
v___x_959_ = lean_array_push(v___x_958_, v_a_941_);
v___x_960_ = lean_array_push(v___x_959_, v___x_955_);
v___x_961_ = lean_box(0);
v___x_962_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg___closed__17));
v___x_963_ = 1;
v___x_964_ = 0;
v___x_965_ = lean_alloc_ctor(0, 5, 2);
lean_ctor_set(v___x_965_, 0, v___x_956_);
lean_ctor_set(v___x_965_, 1, v___x_957_);
lean_ctor_set(v___x_965_, 2, v___x_960_);
lean_ctor_set(v___x_965_, 3, v___x_961_);
lean_ctor_set(v___x_965_, 4, v___x_962_);
lean_ctor_set_uint8(v___x_965_, sizeof(void*)*5, v___x_963_);
lean_ctor_set_uint8(v___x_965_, sizeof(void*)*5 + 1, v___x_964_);
v___x_966_ = l_IO_Process_output(v___x_965_, v___x_961_);
if (lean_obj_tag(v___x_966_) == 0)
{
lean_object* v_a_967_; lean_object* v_stdout_968_; lean_object* v___x_969_; 
lean_del_object(v___x_943_);
v_a_967_ = lean_ctor_get(v___x_966_, 0);
lean_inc(v_a_967_);
lean_dec_ref_known(v___x_966_, 1);
v_stdout_968_ = lean_ctor_get(v_a_967_, 0);
lean_inc_ref(v_stdout_968_);
lean_dec(v_a_967_);
v___x_969_ = l_Lean_Json_parse(v_stdout_968_);
if (lean_obj_tag(v___x_969_) == 0)
{
lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_991_; 
v_isSharedCheck_991_ = !lean_is_exclusive(v___x_969_);
if (v_isSharedCheck_991_ == 0)
{
lean_object* v_unused_992_; 
v_unused_992_ = lean_ctor_get(v___x_969_, 0);
lean_dec(v_unused_992_);
v___x_971_ = v___x_969_;
v_isShared_972_ = v_isSharedCheck_991_;
goto v_resetjp_970_;
}
else
{
lean_dec(v___x_969_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_991_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_973_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__13));
v___x_974_ = l_Lean_IO_throwServerError___redArg(v___x_973_);
if (lean_obj_tag(v___x_974_) == 0)
{
lean_object* v_a_975_; 
lean_del_object(v___x_971_);
v_a_975_ = lean_ctor_get(v___x_974_, 0);
lean_inc(v_a_975_);
lean_dec_ref_known(v___x_974_, 1);
v_js_786_ = v_a_975_;
v___y_787_ = v_a_782_;
goto v___jp_785_;
}
else
{
lean_object* v_a_976_; lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_990_; 
lean_dec_ref(v_rev_781_);
lean_dec(v_num__results_780_);
lean_dec_ref(v_s_779_);
v_a_976_ = lean_ctor_get(v___x_974_, 0);
v_isSharedCheck_990_ = !lean_is_exclusive(v___x_974_);
if (v_isSharedCheck_990_ == 0)
{
v___x_978_ = v___x_974_;
v_isShared_979_ = v_isSharedCheck_990_;
goto v_resetjp_977_;
}
else
{
lean_inc(v_a_976_);
lean_dec(v___x_974_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_990_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v_ref_980_; lean_object* v___x_981_; lean_object* v___x_983_; 
v_ref_980_ = lean_ctor_get(v_a_782_, 5);
v___x_981_ = lean_io_error_to_string(v_a_976_);
if (v_isShared_972_ == 0)
{
lean_ctor_set_tag(v___x_971_, 3);
lean_ctor_set(v___x_971_, 0, v___x_981_);
v___x_983_ = v___x_971_;
goto v_reusejp_982_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v___x_981_);
v___x_983_ = v_reuseFailAlloc_989_;
goto v_reusejp_982_;
}
v_reusejp_982_:
{
lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_987_; 
v___x_984_ = l_Lean_MessageData_ofFormat(v___x_983_);
lean_inc(v_ref_980_);
v___x_985_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_985_, 0, v_ref_980_);
lean_ctor_set(v___x_985_, 1, v___x_984_);
if (v_isShared_979_ == 0)
{
lean_ctor_set(v___x_978_, 0, v___x_985_);
v___x_987_ = v___x_978_;
goto v_reusejp_986_;
}
else
{
lean_object* v_reuseFailAlloc_988_; 
v_reuseFailAlloc_988_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_988_, 0, v___x_985_);
v___x_987_ = v_reuseFailAlloc_988_;
goto v_reusejp_986_;
}
v_reusejp_986_:
{
return v___x_987_;
}
}
}
}
}
}
else
{
lean_object* v_a_993_; 
v_a_993_ = lean_ctor_get(v___x_969_, 0);
lean_inc(v_a_993_);
lean_dec_ref_known(v___x_969_, 1);
v_js_786_ = v_a_993_;
v___y_787_ = v_a_782_;
goto v___jp_785_;
}
}
else
{
lean_object* v_a_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1008_; 
lean_dec_ref(v_rev_781_);
lean_dec(v_num__results_780_);
lean_dec_ref(v_s_779_);
v_a_994_ = lean_ctor_get(v___x_966_, 0);
v_isSharedCheck_1008_ = !lean_is_exclusive(v___x_966_);
if (v_isSharedCheck_1008_ == 0)
{
v___x_996_ = v___x_966_;
v_isShared_997_ = v_isSharedCheck_1008_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_a_994_);
lean_dec(v___x_966_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1008_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
lean_object* v_ref_998_; lean_object* v___x_999_; lean_object* v___x_1001_; 
v_ref_998_ = lean_ctor_get(v_a_782_, 5);
v___x_999_ = lean_io_error_to_string(v_a_994_);
if (v_isShared_944_ == 0)
{
lean_ctor_set_tag(v___x_943_, 3);
lean_ctor_set(v___x_943_, 0, v___x_999_);
v___x_1001_ = v___x_943_;
goto v_reusejp_1000_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v___x_999_);
v___x_1001_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1000_;
}
v_reusejp_1000_:
{
lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1005_; 
v___x_1002_ = l_Lean_MessageData_ofFormat(v___x_1001_);
lean_inc(v_ref_998_);
v___x_1003_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1003_, 0, v_ref_998_);
lean_ctor_set(v___x_1003_, 1, v___x_1002_);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 0, v___x_1003_);
v___x_1005_ = v___x_996_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v___x_1003_);
v___x_1005_ = v_reuseFailAlloc_1006_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
return v___x_1005_;
}
}
}
}
}
}
}
else
{
lean_object* v_val_1012_; lean_object* v___x_1014_; uint8_t v_isShared_1015_; uint8_t v_isSharedCheck_1019_; 
lean_dec_ref(v_rev_781_);
lean_dec(v_num__results_780_);
lean_dec_ref(v_s_779_);
v_val_1012_ = lean_ctor_get(v___x_935_, 0);
v_isSharedCheck_1019_ = !lean_is_exclusive(v___x_935_);
if (v_isSharedCheck_1019_ == 0)
{
v___x_1014_ = v___x_935_;
v_isShared_1015_ = v_isSharedCheck_1019_;
goto v_resetjp_1013_;
}
else
{
lean_inc(v_val_1012_);
lean_dec(v___x_935_);
v___x_1014_ = lean_box(0);
v_isShared_1015_ = v_isSharedCheck_1019_;
goto v_resetjp_1013_;
}
v_resetjp_1013_:
{
lean_object* v___x_1017_; 
if (v_isShared_1015_ == 0)
{
lean_ctor_set_tag(v___x_1014_, 0);
v___x_1017_ = v___x_1014_;
goto v_reusejp_1016_;
}
else
{
lean_object* v_reuseFailAlloc_1018_; 
v_reuseFailAlloc_1018_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1018_, 0, v_val_1012_);
v___x_1017_ = v_reuseFailAlloc_1018_;
goto v_reusejp_1016_;
}
v_reusejp_1016_:
{
return v___x_1017_;
}
}
}
v___jp_785_:
{
lean_object* v___x_788_; 
lean_inc(v_js_786_);
v___x_788_ = l_Lean_Json_getArr_x3f(v_js_786_);
if (lean_obj_tag(v___x_788_) == 0)
{
lean_object* v_a_789_; lean_object* v___x_791_; uint8_t v_isShared_792_; uint8_t v_isSharedCheck_918_; 
lean_dec_ref(v_rev_781_);
lean_dec(v_num__results_780_);
lean_dec_ref(v_s_779_);
v_a_789_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_918_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_918_ == 0)
{
v___x_791_ = v___x_788_;
v_isShared_792_ = v_isSharedCheck_918_;
goto v_resetjp_790_;
}
else
{
lean_inc(v_a_789_);
lean_dec(v___x_788_);
v___x_791_ = lean_box(0);
v_isShared_792_ = v_isSharedCheck_918_;
goto v_resetjp_790_;
}
v_resetjp_790_:
{
lean_object* v___x_793_; lean_object* v___x_794_; 
v___x_793_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__0));
lean_inc(v_js_786_);
v___x_794_ = l_Lean_Json_getObjVal_x3f(v_js_786_, v___x_793_);
if (lean_obj_tag(v___x_794_) == 1)
{
lean_object* v_a_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_893_; 
lean_del_object(v___x_791_);
v_a_795_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_893_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_893_ == 0)
{
v___x_797_ = v___x_794_;
v_isShared_798_ = v_isSharedCheck_893_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_a_795_);
lean_dec(v___x_794_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_893_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v___x_799_; lean_object* v___x_800_; 
v___x_799_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__1));
v___x_800_ = l_Lean_Json_getObjVal_x3f(v_js_786_, v___x_799_);
if (lean_obj_tag(v___x_800_) == 1)
{
lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_868_; 
lean_del_object(v___x_797_);
v_a_801_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_868_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_868_ == 0)
{
v___x_803_ = v___x_800_;
v_isShared_804_ = v_isSharedCheck_868_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_800_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_868_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v___x_805_; lean_object* v___x_806_; 
v___x_805_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__2));
v___x_806_ = l_Lean_Json_getObjVal_x3f(v_a_801_, v___x_805_);
if (lean_obj_tag(v___x_806_) == 1)
{
lean_object* v_a_807_; lean_object* v___x_809_; uint8_t v_isShared_810_; uint8_t v_isSharedCheck_843_; 
lean_del_object(v___x_803_);
lean_dec(v_a_789_);
v_a_807_ = lean_ctor_get(v___x_806_, 0);
v_isSharedCheck_843_ = !lean_is_exclusive(v___x_806_);
if (v_isSharedCheck_843_ == 0)
{
v___x_809_ = v___x_806_;
v_isShared_810_ = v_isSharedCheck_843_;
goto v_resetjp_808_;
}
else
{
lean_inc(v_a_807_);
lean_dec(v___x_806_);
v___x_809_ = lean_box(0);
v_isShared_810_ = v_isSharedCheck_843_;
goto v_resetjp_808_;
}
v_resetjp_808_:
{
lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_811_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__3));
v___x_812_ = lean_unsigned_to_nat(80u);
v___x_813_ = l_Lean_Json_pretty(v_a_795_, v___x_812_);
v___x_814_ = lean_string_append(v___x_811_, v___x_813_);
lean_dec_ref(v___x_813_);
v___x_815_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___closed__4));
v___x_816_ = lean_string_append(v___x_814_, v___x_815_);
v___x_817_ = l_Lean_Json_pretty(v_a_807_, v___x_812_);
v___x_818_ = lean_string_append(v___x_816_, v___x_817_);
lean_dec_ref(v___x_817_);
v___x_819_ = l_Lean_IO_throwServerError___redArg(v___x_818_);
if (lean_obj_tag(v___x_819_) == 0)
{
lean_object* v_a_820_; lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_827_; 
lean_del_object(v___x_809_);
v_a_820_ = lean_ctor_get(v___x_819_, 0);
v_isSharedCheck_827_ = !lean_is_exclusive(v___x_819_);
if (v_isSharedCheck_827_ == 0)
{
v___x_822_ = v___x_819_;
v_isShared_823_ = v_isSharedCheck_827_;
goto v_resetjp_821_;
}
else
{
lean_inc(v_a_820_);
lean_dec(v___x_819_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_827_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v___x_825_; 
if (v_isShared_823_ == 0)
{
v___x_825_ = v___x_822_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_826_; 
v_reuseFailAlloc_826_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_826_, 0, v_a_820_);
v___x_825_ = v_reuseFailAlloc_826_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
return v___x_825_;
}
}
}
else
{
lean_object* v_a_828_; lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_842_; 
v_a_828_ = lean_ctor_get(v___x_819_, 0);
v_isSharedCheck_842_ = !lean_is_exclusive(v___x_819_);
if (v_isSharedCheck_842_ == 0)
{
v___x_830_ = v___x_819_;
v_isShared_831_ = v_isSharedCheck_842_;
goto v_resetjp_829_;
}
else
{
lean_inc(v_a_828_);
lean_dec(v___x_819_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_842_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
lean_object* v_ref_832_; lean_object* v___x_833_; lean_object* v___x_835_; 
v_ref_832_ = lean_ctor_get(v___y_787_, 5);
v___x_833_ = lean_io_error_to_string(v_a_828_);
if (v_isShared_810_ == 0)
{
lean_ctor_set_tag(v___x_809_, 3);
lean_ctor_set(v___x_809_, 0, v___x_833_);
v___x_835_ = v___x_809_;
goto v_reusejp_834_;
}
else
{
lean_object* v_reuseFailAlloc_841_; 
v_reuseFailAlloc_841_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_841_, 0, v___x_833_);
v___x_835_ = v_reuseFailAlloc_841_;
goto v_reusejp_834_;
}
v_reusejp_834_:
{
lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_839_; 
v___x_836_ = l_Lean_MessageData_ofFormat(v___x_835_);
lean_inc(v_ref_832_);
v___x_837_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_837_, 0, v_ref_832_);
lean_ctor_set(v___x_837_, 1, v___x_836_);
if (v_isShared_831_ == 0)
{
lean_ctor_set(v___x_830_, 0, v___x_837_);
v___x_839_ = v___x_830_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_837_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
}
}
}
}
else
{
lean_object* v___x_844_; 
lean_dec_ref(v___x_806_);
lean_dec(v_a_795_);
v___x_844_ = l_Lean_IO_throwServerError___redArg(v_a_789_);
if (lean_obj_tag(v___x_844_) == 0)
{
lean_object* v_a_845_; lean_object* v___x_847_; uint8_t v_isShared_848_; uint8_t v_isSharedCheck_852_; 
lean_del_object(v___x_803_);
v_a_845_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_852_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_852_ == 0)
{
v___x_847_ = v___x_844_;
v_isShared_848_ = v_isSharedCheck_852_;
goto v_resetjp_846_;
}
else
{
lean_inc(v_a_845_);
lean_dec(v___x_844_);
v___x_847_ = lean_box(0);
v_isShared_848_ = v_isSharedCheck_852_;
goto v_resetjp_846_;
}
v_resetjp_846_:
{
lean_object* v___x_850_; 
if (v_isShared_848_ == 0)
{
v___x_850_ = v___x_847_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_a_845_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
}
else
{
lean_object* v_a_853_; lean_object* v___x_855_; uint8_t v_isShared_856_; uint8_t v_isSharedCheck_867_; 
v_a_853_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_867_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_867_ == 0)
{
v___x_855_ = v___x_844_;
v_isShared_856_ = v_isSharedCheck_867_;
goto v_resetjp_854_;
}
else
{
lean_inc(v_a_853_);
lean_dec(v___x_844_);
v___x_855_ = lean_box(0);
v_isShared_856_ = v_isSharedCheck_867_;
goto v_resetjp_854_;
}
v_resetjp_854_:
{
lean_object* v_ref_857_; lean_object* v___x_858_; lean_object* v___x_860_; 
v_ref_857_ = lean_ctor_get(v___y_787_, 5);
v___x_858_ = lean_io_error_to_string(v_a_853_);
if (v_isShared_804_ == 0)
{
lean_ctor_set_tag(v___x_803_, 3);
lean_ctor_set(v___x_803_, 0, v___x_858_);
v___x_860_ = v___x_803_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_866_; 
v_reuseFailAlloc_866_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_866_, 0, v___x_858_);
v___x_860_ = v_reuseFailAlloc_866_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_864_; 
v___x_861_ = l_Lean_MessageData_ofFormat(v___x_860_);
lean_inc(v_ref_857_);
v___x_862_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_862_, 0, v_ref_857_);
lean_ctor_set(v___x_862_, 1, v___x_861_);
if (v_isShared_856_ == 0)
{
lean_ctor_set(v___x_855_, 0, v___x_862_);
v___x_864_ = v___x_855_;
goto v_reusejp_863_;
}
else
{
lean_object* v_reuseFailAlloc_865_; 
v_reuseFailAlloc_865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_865_, 0, v___x_862_);
v___x_864_ = v_reuseFailAlloc_865_;
goto v_reusejp_863_;
}
v_reusejp_863_:
{
return v___x_864_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_869_; 
lean_dec_ref(v___x_800_);
lean_dec(v_a_795_);
v___x_869_ = l_Lean_IO_throwServerError___redArg(v_a_789_);
if (lean_obj_tag(v___x_869_) == 0)
{
lean_object* v_a_870_; lean_object* v___x_872_; uint8_t v_isShared_873_; uint8_t v_isSharedCheck_877_; 
lean_del_object(v___x_797_);
v_a_870_ = lean_ctor_get(v___x_869_, 0);
v_isSharedCheck_877_ = !lean_is_exclusive(v___x_869_);
if (v_isSharedCheck_877_ == 0)
{
v___x_872_ = v___x_869_;
v_isShared_873_ = v_isSharedCheck_877_;
goto v_resetjp_871_;
}
else
{
lean_inc(v_a_870_);
lean_dec(v___x_869_);
v___x_872_ = lean_box(0);
v_isShared_873_ = v_isSharedCheck_877_;
goto v_resetjp_871_;
}
v_resetjp_871_:
{
lean_object* v___x_875_; 
if (v_isShared_873_ == 0)
{
v___x_875_ = v___x_872_;
goto v_reusejp_874_;
}
else
{
lean_object* v_reuseFailAlloc_876_; 
v_reuseFailAlloc_876_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_876_, 0, v_a_870_);
v___x_875_ = v_reuseFailAlloc_876_;
goto v_reusejp_874_;
}
v_reusejp_874_:
{
return v___x_875_;
}
}
}
else
{
lean_object* v_a_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_892_; 
v_a_878_ = lean_ctor_get(v___x_869_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_869_);
if (v_isSharedCheck_892_ == 0)
{
v___x_880_ = v___x_869_;
v_isShared_881_ = v_isSharedCheck_892_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_a_878_);
lean_dec(v___x_869_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_892_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v_ref_882_; lean_object* v___x_883_; lean_object* v___x_885_; 
v_ref_882_ = lean_ctor_get(v___y_787_, 5);
v___x_883_ = lean_io_error_to_string(v_a_878_);
if (v_isShared_798_ == 0)
{
lean_ctor_set_tag(v___x_797_, 3);
lean_ctor_set(v___x_797_, 0, v___x_883_);
v___x_885_ = v___x_797_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v___x_883_);
v___x_885_ = v_reuseFailAlloc_891_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_889_; 
v___x_886_ = l_Lean_MessageData_ofFormat(v___x_885_);
lean_inc(v_ref_882_);
v___x_887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_887_, 0, v_ref_882_);
lean_ctor_set(v___x_887_, 1, v___x_886_);
if (v_isShared_881_ == 0)
{
lean_ctor_set(v___x_880_, 0, v___x_887_);
v___x_889_ = v___x_880_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_890_; 
v_reuseFailAlloc_890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_890_, 0, v___x_887_);
v___x_889_ = v_reuseFailAlloc_890_;
goto v_reusejp_888_;
}
v_reusejp_888_:
{
return v___x_889_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_894_; 
lean_dec_ref(v___x_794_);
lean_dec(v_js_786_);
v___x_894_ = l_Lean_IO_throwServerError___redArg(v_a_789_);
if (lean_obj_tag(v___x_894_) == 0)
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
lean_del_object(v___x_791_);
v_a_895_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___x_894_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_894_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
else
{
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_917_; 
v_a_903_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_917_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_917_ == 0)
{
v___x_905_ = v___x_894_;
v_isShared_906_ = v_isSharedCheck_917_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_894_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_917_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v_ref_907_; lean_object* v___x_908_; lean_object* v___x_910_; 
v_ref_907_ = lean_ctor_get(v___y_787_, 5);
v___x_908_ = lean_io_error_to_string(v_a_903_);
if (v_isShared_792_ == 0)
{
lean_ctor_set_tag(v___x_791_, 3);
lean_ctor_set(v___x_791_, 0, v___x_908_);
v___x_910_ = v___x_791_;
goto v_reusejp_909_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v___x_908_);
v___x_910_ = v_reuseFailAlloc_916_;
goto v_reusejp_909_;
}
v_reusejp_909_:
{
lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_914_; 
v___x_911_ = l_Lean_MessageData_ofFormat(v___x_910_);
lean_inc(v_ref_907_);
v___x_912_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_912_, 0, v_ref_907_);
lean_ctor_set(v___x_912_, 1, v___x_911_);
if (v_isShared_906_ == 0)
{
lean_ctor_set(v___x_905_, 0, v___x_912_);
v___x_914_ = v___x_905_;
goto v_reusejp_913_;
}
else
{
lean_object* v_reuseFailAlloc_915_; 
v_reuseFailAlloc_915_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_915_, 0, v___x_912_);
v___x_914_ = v_reuseFailAlloc_915_;
goto v_reusejp_913_;
}
v_reusejp_913_:
{
return v___x_914_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_919_; lean_object* v___x_921_; uint8_t v_isShared_922_; uint8_t v_isSharedCheck_931_; 
lean_dec(v_js_786_);
v_a_919_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_931_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_931_ == 0)
{
v___x_921_ = v___x_788_;
v_isShared_922_ = v_isSharedCheck_931_;
goto v_resetjp_920_;
}
else
{
lean_inc(v_a_919_);
lean_dec(v___x_788_);
v___x_921_ = lean_box(0);
v_isShared_922_ = v_isSharedCheck_931_;
goto v_resetjp_920_;
}
v_resetjp_920_:
{
lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_929_; 
v___x_923_ = lean_st_ref_take(v___x_784_);
v___x_924_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_924_, 0, v_num__results_780_);
lean_ctor_set(v___x_924_, 1, v_rev_781_);
v___x_925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_925_, 0, v_s_779_);
lean_ctor_set(v___x_925_, 1, v___x_924_);
lean_inc(v_a_919_);
v___x_926_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0___redArg(v___x_923_, v___x_925_, v_a_919_);
v___x_927_ = lean_st_ref_set(v___x_784_, v___x_926_);
if (v_isShared_922_ == 0)
{
lean_ctor_set_tag(v___x_921_, 0);
v___x_929_ = v___x_921_;
goto v_reusejp_928_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v_a_919_);
v___x_929_ = v_reuseFailAlloc_930_;
goto v_reusejp_928_;
}
v_reusejp_928_:
{
return v___x_929_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg___boxed(lean_object* v_s_1020_, lean_object* v_num__results_1021_, lean_object* v_rev_1022_, lean_object* v_a_1023_, lean_object* v_a_1024_){
_start:
{
lean_object* v_res_1025_; 
v_res_1025_ = lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg(v_s_1020_, v_num__results_1021_, v_rev_1022_, v_a_1023_);
lean_dec_ref(v_a_1023_);
return v_res_1025_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson(lean_object* v_s_1026_, lean_object* v_num__results_1027_, lean_object* v_rev_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg(v_s_1026_, v_num__results_1027_, v_rev_1028_, v_a_1029_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___boxed(lean_object* v_s_1033_, lean_object* v_num__results_1034_, lean_object* v_rev_1035_, lean_object* v_a_1036_, lean_object* v_a_1037_, lean_object* v_a_1038_){
_start:
{
lean_object* v_res_1039_; 
v_res_1039_ = lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson(v_s_1033_, v_num__results_1034_, v_rev_1035_, v_a_1036_, v_a_1037_);
lean_dec(v_a_1037_);
lean_dec_ref(v_a_1036_);
return v_res_1039_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0(lean_object* v_00_u03b2_1040_, lean_object* v_m_1041_, lean_object* v_a_1042_, lean_object* v_b_1043_){
_start:
{
lean_object* v___x_1044_; 
v___x_1044_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0___redArg(v_m_1041_, v_a_1042_, v_b_1043_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1(lean_object* v_00_u03b2_1045_, lean_object* v_m_1046_, lean_object* v_a_1047_){
_start:
{
lean_object* v___x_1048_; 
v___x_1048_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___redArg(v_m_1046_, v_a_1047_);
return v___x_1048_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1___boxed(lean_object* v_00_u03b2_1049_, lean_object* v_m_1050_, lean_object* v_a_1051_){
_start:
{
lean_object* v_res_1052_; 
v_res_1052_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1(v_00_u03b2_1049_, v_m_1050_, v_a_1051_);
lean_dec_ref(v_a_1051_);
lean_dec_ref(v_m_1050_);
return v_res_1052_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0(lean_object* v_00_u03b2_1053_, lean_object* v_a_1054_, lean_object* v_x_1055_){
_start:
{
uint8_t v___x_1056_; 
v___x_1056_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___redArg(v_a_1054_, v_x_1055_);
return v___x_1056_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1057_, lean_object* v_a_1058_, lean_object* v_x_1059_){
_start:
{
uint8_t v_res_1060_; lean_object* v_r_1061_; 
v_res_1060_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__0(v_00_u03b2_1057_, v_a_1058_, v_x_1059_);
lean_dec(v_x_1059_);
lean_dec_ref(v_a_1058_);
v_r_1061_ = lean_box(v_res_1060_);
return v_r_1061_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1(lean_object* v_00_u03b2_1062_, lean_object* v_data_1063_){
_start:
{
lean_object* v___x_1064_; 
v___x_1064_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1___redArg(v_data_1063_);
return v___x_1064_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2(lean_object* v_00_u03b2_1065_, lean_object* v_a_1066_, lean_object* v_b_1067_, lean_object* v_x_1068_){
_start:
{
lean_object* v___x_1069_; 
v___x_1069_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__2___redArg(v_a_1066_, v_b_1067_, v_x_1068_);
return v___x_1069_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4(lean_object* v_00_u03b2_1070_, lean_object* v_a_1071_, lean_object* v_x_1072_){
_start:
{
lean_object* v___x_1073_; 
v___x_1073_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___redArg(v_a_1071_, v_x_1072_);
return v___x_1073_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4___boxed(lean_object* v_00_u03b2_1074_, lean_object* v_a_1075_, lean_object* v_x_1076_){
_start:
{
lean_object* v_res_1077_; 
v_res_1077_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getStateSearchQueryJson_spec__1_spec__4(v_00_u03b2_1074_, v_a_1075_, v_x_1076_);
lean_dec(v_x_1076_);
lean_dec_ref(v_a_1075_);
return v_res_1077_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1078_, lean_object* v_i_1079_, lean_object* v_source_1080_, lean_object* v_target_1081_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2___redArg(v_i_1079_, v_source_1080_, v_target_1081_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1083_, lean_object* v_x_1084_, lean_object* v_x_1085_){
_start:
{
lean_object* v___x_1086_; 
v___x_1086_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getStateSearchQueryJson_spec__0_spec__1_spec__2_spec__4___redArg(v_x_1084_, v_x_1085_);
return v___x_1086_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(lean_object* v_x_1093_, lean_object* v_x_1094_){
_start:
{
if (lean_obj_tag(v_x_1093_) == 0)
{
lean_object* v___x_1095_; 
v___x_1095_ = ((lean_object*)(lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__1));
return v___x_1095_;
}
else
{
lean_object* v_val_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1107_; 
v_val_1096_ = lean_ctor_get(v_x_1093_, 0);
v_isSharedCheck_1107_ = !lean_is_exclusive(v_x_1093_);
if (v_isSharedCheck_1107_ == 0)
{
v___x_1098_ = v_x_1093_;
v_isShared_1099_ = v_isSharedCheck_1107_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_val_1096_);
lean_dec(v_x_1093_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1107_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1103_; 
v___x_1100_ = ((lean_object*)(lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___closed__3));
v___x_1101_ = l_String_quote(v_val_1096_);
if (v_isShared_1099_ == 0)
{
lean_ctor_set_tag(v___x_1098_, 3);
lean_ctor_set(v___x_1098_, 0, v___x_1101_);
v___x_1103_ = v___x_1098_;
goto v_reusejp_1102_;
}
else
{
lean_object* v_reuseFailAlloc_1106_; 
v_reuseFailAlloc_1106_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1106_, 0, v___x_1101_);
v___x_1103_ = v_reuseFailAlloc_1106_;
goto v_reusejp_1102_;
}
v_reusejp_1102_:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1104_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1104_, 0, v___x_1100_);
lean_ctor_set(v___x_1104_, 1, v___x_1103_);
v___x_1105_ = l_Repr_addAppParen(v___x_1104_, v_x_1094_);
return v___x_1105_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0___boxed(lean_object* v_x_1108_, lean_object* v_x_1109_){
_start:
{
lean_object* v_res_1110_; 
v_res_1110_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(v_x_1108_, v_x_1109_);
lean_dec(v_x_1109_);
return v_res_1110_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Nat_cast___at___00LeanSearchClient_instReprSearchResult_repr_spec__1(lean_object* v_a_1111_){
_start:
{
lean_object* v___x_1112_; 
v___x_1112_ = lean_nat_to_int(v_a_1111_);
return v___x_1112_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_1126_; lean_object* v___x_1127_; 
v___x_1126_ = lean_unsigned_to_nat(8u);
v___x_1127_ = lean_nat_to_int(v___x_1126_);
return v___x_1127_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__12(void){
_start:
{
lean_object* v___x_1134_; lean_object* v___x_1135_; 
v___x_1134_ = lean_unsigned_to_nat(9u);
v___x_1135_ = lean_nat_to_int(v___x_1134_);
return v___x_1135_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; 
v___x_1139_ = lean_unsigned_to_nat(14u);
v___x_1140_ = lean_nat_to_int(v___x_1139_);
return v___x_1140_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__18(void){
_start:
{
lean_object* v___x_1144_; lean_object* v___x_1145_; 
v___x_1144_ = lean_unsigned_to_nat(12u);
v___x_1145_ = lean_nat_to_int(v___x_1144_);
return v___x_1145_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__22(void){
_start:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1150_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__0));
v___x_1151_ = lean_string_length(v___x_1150_);
return v___x_1151_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__23(void){
_start:
{
lean_object* v___x_1152_; lean_object* v___x_1153_; 
v___x_1152_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__22, &lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__22_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__22);
v___x_1153_ = lean_nat_to_int(v___x_1152_);
return v___x_1153_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(lean_object* v_x_1158_){
_start:
{
lean_object* v_name_1159_; lean_object* v_type_x3f_1160_; lean_object* v_docString_x3f_1161_; lean_object* v_doc__url_x3f_1162_; lean_object* v_kind_x3f_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; uint8_t v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; 
v_name_1159_ = lean_ctor_get(v_x_1158_, 0);
lean_inc_ref(v_name_1159_);
v_type_x3f_1160_ = lean_ctor_get(v_x_1158_, 1);
lean_inc(v_type_x3f_1160_);
v_docString_x3f_1161_ = lean_ctor_get(v_x_1158_, 2);
lean_inc(v_docString_x3f_1161_);
v_doc__url_x3f_1162_ = lean_ctor_get(v_x_1158_, 3);
lean_inc(v_doc__url_x3f_1162_);
v_kind_x3f_1163_ = lean_ctor_get(v_x_1158_, 4);
lean_inc(v_kind_x3f_1163_);
lean_dec_ref(v_x_1158_);
v___x_1164_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__5));
v___x_1165_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__6));
v___x_1166_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__7, &lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__7_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__7);
v___x_1167_ = l_String_quote(v_name_1159_);
v___x_1168_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1168_, 0, v___x_1167_);
v___x_1169_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1169_, 0, v___x_1166_);
lean_ctor_set(v___x_1169_, 1, v___x_1168_);
v___x_1170_ = 0;
v___x_1171_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1171_, 0, v___x_1169_);
lean_ctor_set_uint8(v___x_1171_, sizeof(void*)*1, v___x_1170_);
v___x_1172_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1172_, 0, v___x_1165_);
lean_ctor_set(v___x_1172_, 1, v___x_1171_);
v___x_1173_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__9));
v___x_1174_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1174_, 0, v___x_1172_);
lean_ctor_set(v___x_1174_, 1, v___x_1173_);
v___x_1175_ = lean_box(1);
v___x_1176_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1176_, 0, v___x_1174_);
lean_ctor_set(v___x_1176_, 1, v___x_1175_);
v___x_1177_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__11));
v___x_1178_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1176_);
lean_ctor_set(v___x_1178_, 1, v___x_1177_);
v___x_1179_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1179_, 0, v___x_1178_);
lean_ctor_set(v___x_1179_, 1, v___x_1164_);
v___x_1180_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__12, &lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__12_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__12);
v___x_1181_ = lean_unsigned_to_nat(0u);
v___x_1182_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(v_type_x3f_1160_, v___x_1181_);
v___x_1183_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1180_);
lean_ctor_set(v___x_1183_, 1, v___x_1182_);
v___x_1184_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1184_, 0, v___x_1183_);
lean_ctor_set_uint8(v___x_1184_, sizeof(void*)*1, v___x_1170_);
v___x_1185_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1185_, 0, v___x_1179_);
lean_ctor_set(v___x_1185_, 1, v___x_1184_);
v___x_1186_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1185_);
lean_ctor_set(v___x_1186_, 1, v___x_1173_);
v___x_1187_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1187_, 0, v___x_1186_);
lean_ctor_set(v___x_1187_, 1, v___x_1175_);
v___x_1188_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__14));
v___x_1189_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1187_);
lean_ctor_set(v___x_1189_, 1, v___x_1188_);
v___x_1190_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1189_);
lean_ctor_set(v___x_1190_, 1, v___x_1164_);
v___x_1191_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__15, &lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__15_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__15);
v___x_1192_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(v_docString_x3f_1161_, v___x_1181_);
v___x_1193_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1191_);
lean_ctor_set(v___x_1193_, 1, v___x_1192_);
v___x_1194_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1194_, 0, v___x_1193_);
lean_ctor_set_uint8(v___x_1194_, sizeof(void*)*1, v___x_1170_);
v___x_1195_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1190_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
v___x_1196_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1196_, 0, v___x_1195_);
lean_ctor_set(v___x_1196_, 1, v___x_1173_);
v___x_1197_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1196_);
lean_ctor_set(v___x_1197_, 1, v___x_1175_);
v___x_1198_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__17));
v___x_1199_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1197_);
lean_ctor_set(v___x_1199_, 1, v___x_1198_);
v___x_1200_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1200_, 0, v___x_1199_);
lean_ctor_set(v___x_1200_, 1, v___x_1164_);
v___x_1201_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__18, &lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__18_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__18);
v___x_1202_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(v_doc__url_x3f_1162_, v___x_1181_);
v___x_1203_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1201_);
lean_ctor_set(v___x_1203_, 1, v___x_1202_);
v___x_1204_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1204_, 0, v___x_1203_);
lean_ctor_set_uint8(v___x_1204_, sizeof(void*)*1, v___x_1170_);
v___x_1205_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1205_, 0, v___x_1200_);
lean_ctor_set(v___x_1205_, 1, v___x_1204_);
v___x_1206_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1206_, 0, v___x_1205_);
lean_ctor_set(v___x_1206_, 1, v___x_1173_);
v___x_1207_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1207_, 0, v___x_1206_);
lean_ctor_set(v___x_1207_, 1, v___x_1175_);
v___x_1208_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__20));
v___x_1209_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1209_, 0, v___x_1207_);
lean_ctor_set(v___x_1209_, 1, v___x_1208_);
v___x_1210_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1210_, 0, v___x_1209_);
lean_ctor_set(v___x_1210_, 1, v___x_1164_);
v___x_1211_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprSearchResult_repr_spec__0(v_kind_x3f_1163_, v___x_1181_);
v___x_1212_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1212_, 0, v___x_1180_);
lean_ctor_set(v___x_1212_, 1, v___x_1211_);
v___x_1213_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1213_, 0, v___x_1212_);
lean_ctor_set_uint8(v___x_1213_, sizeof(void*)*1, v___x_1170_);
v___x_1214_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1214_, 0, v___x_1210_);
lean_ctor_set(v___x_1214_, 1, v___x_1213_);
v___x_1215_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__23, &lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__23_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__23);
v___x_1216_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__24));
v___x_1217_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1217_, 0, v___x_1216_);
lean_ctor_set(v___x_1217_, 1, v___x_1214_);
v___x_1218_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__25));
v___x_1219_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1217_);
lean_ctor_set(v___x_1219_, 1, v___x_1218_);
v___x_1220_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1220_, 0, v___x_1215_);
lean_ctor_set(v___x_1220_, 1, v___x_1219_);
v___x_1221_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1221_, 0, v___x_1220_);
lean_ctor_set_uint8(v___x_1221_, sizeof(void*)*1, v___x_1170_);
return v___x_1221_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr(lean_object* v_x_1222_, lean_object* v_prec_1223_){
_start:
{
lean_object* v___x_1224_; 
v___x_1224_ = lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(v_x_1222_);
return v___x_1224_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___boxed(lean_object* v_x_1225_, lean_object* v_prec_1226_){
_start:
{
lean_object* v_res_1227_; 
v_res_1227_ = lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr(v_x_1225_, v_prec_1226_);
lean_dec(v_prec_1226_);
return v_res_1227_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(lean_object* v_j_1230_, lean_object* v_k_1231_){
_start:
{
lean_object* v___x_1232_; lean_object* v___x_1233_; 
v___x_1232_ = l_Lean_Json_getObjValD(v_j_1230_, v_k_1231_);
v___x_1233_ = l_Lean_Json_getStr_x3f(v___x_1232_);
return v___x_1233_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2___boxed(lean_object* v_j_1234_, lean_object* v_k_1235_){
_start:
{
lean_object* v_res_1236_; 
v_res_1236_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_j_1234_, v_k_1235_);
lean_dec_ref(v_k_1235_);
return v_res_1236_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2_spec__4(size_t v_sz_1237_, size_t v_i_1238_, lean_object* v_bs_1239_){
_start:
{
uint8_t v___x_1240_; 
v___x_1240_ = lean_usize_dec_lt(v_i_1238_, v_sz_1237_);
if (v___x_1240_ == 0)
{
lean_object* v___x_1241_; 
v___x_1241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1241_, 0, v_bs_1239_);
return v___x_1241_;
}
else
{
lean_object* v_v_1242_; lean_object* v___x_1243_; 
v_v_1242_ = lean_array_uget_borrowed(v_bs_1239_, v_i_1238_);
lean_inc(v_v_1242_);
v___x_1243_ = l_Lean_Json_getStr_x3f(v_v_1242_);
if (lean_obj_tag(v___x_1243_) == 0)
{
lean_object* v_a_1244_; lean_object* v___x_1246_; uint8_t v_isShared_1247_; uint8_t v_isSharedCheck_1251_; 
lean_dec_ref(v_bs_1239_);
v_a_1244_ = lean_ctor_get(v___x_1243_, 0);
v_isSharedCheck_1251_ = !lean_is_exclusive(v___x_1243_);
if (v_isSharedCheck_1251_ == 0)
{
v___x_1246_ = v___x_1243_;
v_isShared_1247_ = v_isSharedCheck_1251_;
goto v_resetjp_1245_;
}
else
{
lean_inc(v_a_1244_);
lean_dec(v___x_1243_);
v___x_1246_ = lean_box(0);
v_isShared_1247_ = v_isSharedCheck_1251_;
goto v_resetjp_1245_;
}
v_resetjp_1245_:
{
lean_object* v___x_1249_; 
if (v_isShared_1247_ == 0)
{
v___x_1249_ = v___x_1246_;
goto v_reusejp_1248_;
}
else
{
lean_object* v_reuseFailAlloc_1250_; 
v_reuseFailAlloc_1250_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1250_, 0, v_a_1244_);
v___x_1249_ = v_reuseFailAlloc_1250_;
goto v_reusejp_1248_;
}
v_reusejp_1248_:
{
return v___x_1249_;
}
}
}
else
{
lean_object* v_a_1252_; lean_object* v___x_1253_; lean_object* v_bs_x27_1254_; size_t v___x_1255_; size_t v___x_1256_; lean_object* v___x_1257_; 
v_a_1252_ = lean_ctor_get(v___x_1243_, 0);
lean_inc(v_a_1252_);
lean_dec_ref_known(v___x_1243_, 1);
v___x_1253_ = lean_unsigned_to_nat(0u);
v_bs_x27_1254_ = lean_array_uset(v_bs_1239_, v_i_1238_, v___x_1253_);
v___x_1255_ = ((size_t)1ULL);
v___x_1256_ = lean_usize_add(v_i_1238_, v___x_1255_);
v___x_1257_ = lean_array_uset(v_bs_x27_1254_, v_i_1238_, v_a_1252_);
v_i_1238_ = v___x_1256_;
v_bs_1239_ = v___x_1257_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_sz_1259_, lean_object* v_i_1260_, lean_object* v_bs_1261_){
_start:
{
size_t v_sz_boxed_1262_; size_t v_i_boxed_1263_; lean_object* v_res_1264_; 
v_sz_boxed_1262_ = lean_unbox_usize(v_sz_1259_);
lean_dec(v_sz_1259_);
v_i_boxed_1263_ = lean_unbox_usize(v_i_1260_);
lean_dec(v_i_1260_);
v_res_1264_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2_spec__4(v_sz_boxed_1262_, v_i_boxed_1263_, v_bs_1261_);
return v_res_1264_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2(lean_object* v_x_1267_){
_start:
{
if (lean_obj_tag(v_x_1267_) == 4)
{
lean_object* v_elems_1268_; size_t v_sz_1269_; size_t v___x_1270_; lean_object* v___x_1271_; 
v_elems_1268_ = lean_ctor_get(v_x_1267_, 0);
lean_inc_ref(v_elems_1268_);
lean_dec_ref_known(v_x_1267_, 1);
v_sz_1269_ = lean_array_size(v_elems_1268_);
v___x_1270_ = ((size_t)0ULL);
v___x_1271_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2_spec__4(v_sz_1269_, v___x_1270_, v_elems_1268_);
return v___x_1271_;
}
else
{
lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; 
v___x_1272_ = ((lean_object*)(lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__0));
v___x_1273_ = lean_unsigned_to_nat(80u);
v___x_1274_ = l_Lean_Json_pretty(v_x_1267_, v___x_1273_);
v___x_1275_ = lean_string_append(v___x_1272_, v___x_1274_);
lean_dec_ref(v___x_1274_);
v___x_1276_ = ((lean_object*)(lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2___closed__1));
v___x_1277_ = lean_string_append(v___x_1275_, v___x_1276_);
v___x_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1277_);
return v___x_1278_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0(lean_object* v_j_1279_){
_start:
{
lean_object* v___x_1280_; 
v___x_1280_ = lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0_spec__2(v_j_1279_);
if (lean_obj_tag(v___x_1280_) == 0)
{
lean_object* v_a_1281_; lean_object* v___x_1283_; uint8_t v_isShared_1284_; uint8_t v_isSharedCheck_1288_; 
v_a_1281_ = lean_ctor_get(v___x_1280_, 0);
v_isSharedCheck_1288_ = !lean_is_exclusive(v___x_1280_);
if (v_isSharedCheck_1288_ == 0)
{
v___x_1283_ = v___x_1280_;
v_isShared_1284_ = v_isSharedCheck_1288_;
goto v_resetjp_1282_;
}
else
{
lean_inc(v_a_1281_);
lean_dec(v___x_1280_);
v___x_1283_ = lean_box(0);
v_isShared_1284_ = v_isSharedCheck_1288_;
goto v_resetjp_1282_;
}
v_resetjp_1282_:
{
lean_object* v___x_1286_; 
if (v_isShared_1284_ == 0)
{
v___x_1286_ = v___x_1283_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1287_; 
v_reuseFailAlloc_1287_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1287_, 0, v_a_1281_);
v___x_1286_ = v_reuseFailAlloc_1287_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
return v___x_1286_;
}
}
}
else
{
lean_object* v_a_1289_; lean_object* v___x_1291_; uint8_t v_isShared_1292_; uint8_t v_isSharedCheck_1297_; 
v_a_1289_ = lean_ctor_get(v___x_1280_, 0);
v_isSharedCheck_1297_ = !lean_is_exclusive(v___x_1280_);
if (v_isSharedCheck_1297_ == 0)
{
v___x_1291_ = v___x_1280_;
v_isShared_1292_ = v_isSharedCheck_1297_;
goto v_resetjp_1290_;
}
else
{
lean_inc(v_a_1289_);
lean_dec(v___x_1280_);
v___x_1291_ = lean_box(0);
v_isShared_1292_ = v_isSharedCheck_1297_;
goto v_resetjp_1290_;
}
v_resetjp_1290_:
{
lean_object* v___x_1293_; lean_object* v___x_1295_; 
v___x_1293_ = lean_array_to_list(v_a_1289_);
if (v_isShared_1292_ == 0)
{
lean_ctor_set(v___x_1291_, 0, v___x_1293_);
v___x_1295_ = v___x_1291_;
goto v_reusejp_1294_;
}
else
{
lean_object* v_reuseFailAlloc_1296_; 
v_reuseFailAlloc_1296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1296_, 0, v___x_1293_);
v___x_1295_ = v_reuseFailAlloc_1296_;
goto v_reusejp_1294_;
}
v_reusejp_1294_:
{
return v___x_1295_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0(lean_object* v_j_1298_, lean_object* v_k_1299_){
_start:
{
lean_object* v___x_1300_; lean_object* v___x_1301_; 
v___x_1300_ = l_Lean_Json_getObjValD(v_j_1298_, v_k_1299_);
v___x_1301_ = lp_LeanSearchClient_Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0_spec__0(v___x_1300_);
return v___x_1301_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0___boxed(lean_object* v_j_1302_, lean_object* v_k_1303_){
_start:
{
lean_object* v_res_1304_; 
v_res_1304_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0(v_j_1302_, v_k_1303_);
lean_dec_ref(v_k_1303_);
return v_res_1304_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1(lean_object* v_x_1307_, lean_object* v_x_1308_){
_start:
{
if (lean_obj_tag(v_x_1308_) == 0)
{
return v_x_1307_;
}
else
{
lean_object* v_head_1309_; lean_object* v_tail_1310_; lean_object* v___x_1311_; uint8_t v___x_1312_; 
v_head_1309_ = lean_ctor_get(v_x_1308_, 0);
lean_inc(v_head_1309_);
v_tail_1310_ = lean_ctor_get(v_x_1308_, 1);
lean_inc(v_tail_1310_);
lean_dec_ref_known(v_x_1308_, 2);
v___x_1311_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
v___x_1312_ = lean_string_dec_eq(v_x_1307_, v___x_1311_);
if (v___x_1312_ == 0)
{
lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; 
v___x_1313_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1));
v___x_1314_ = lean_string_append(v_x_1307_, v___x_1313_);
v___x_1315_ = lean_string_append(v___x_1314_, v_head_1309_);
lean_dec(v_head_1309_);
v_x_1307_ = v___x_1315_;
v_x_1308_ = v_tail_1310_;
goto _start;
}
else
{
lean_dec_ref(v_x_1307_);
v_x_1307_ = v_head_1309_;
v_x_1308_ = v_tail_1310_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f(lean_object* v_js_1323_){
_start:
{
lean_object* v___x_1324_; lean_object* v___x_1325_; 
v___x_1324_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__0));
v___x_1325_ = l_Lean_Json_getObjVal_x3f(v_js_1323_, v___x_1324_);
if (lean_obj_tag(v___x_1325_) == 1)
{
lean_object* v_a_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; 
v_a_1326_ = lean_ctor_get(v___x_1325_, 0);
lean_inc_n(v_a_1326_, 2);
lean_dec_ref_known(v___x_1325_, 1);
v___x_1327_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1));
v___x_1328_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__0(v_a_1326_, v___x_1327_);
if (lean_obj_tag(v___x_1328_) == 1)
{
lean_object* v_a_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1399_; 
v_a_1329_ = lean_ctor_get(v___x_1328_, 0);
v_isSharedCheck_1399_ = !lean_is_exclusive(v___x_1328_);
if (v_isSharedCheck_1399_ == 0)
{
v___x_1331_ = v___x_1328_;
v_isShared_1332_ = v_isSharedCheck_1399_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_a_1329_);
lean_dec(v___x_1328_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1399_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v___x_1333_; lean_object* v_name_1334_; lean_object* v___y_1336_; lean_object* v___y_1337_; lean_object* v___y_1338_; lean_object* v___y_1339_; lean_object* v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1347_; lean_object* v___y_1360_; lean_object* v___y_1361_; lean_object* v___y_1374_; lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1333_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
v_name_1334_ = lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1(v___x_1333_, v_a_1329_);
v___x_1388_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__4));
lean_inc(v_a_1326_);
v___x_1389_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_a_1326_, v___x_1388_);
if (lean_obj_tag(v___x_1389_) == 0)
{
lean_object* v___x_1390_; 
lean_dec_ref_known(v___x_1389_, 1);
v___x_1390_ = lean_box(0);
v___y_1374_ = v___x_1390_;
goto v___jp_1373_;
}
else
{
lean_object* v_a_1391_; lean_object* v___x_1393_; uint8_t v_isShared_1394_; uint8_t v_isSharedCheck_1398_; 
v_a_1391_ = lean_ctor_get(v___x_1389_, 0);
v_isSharedCheck_1398_ = !lean_is_exclusive(v___x_1389_);
if (v_isSharedCheck_1398_ == 0)
{
v___x_1393_ = v___x_1389_;
v_isShared_1394_ = v_isSharedCheck_1398_;
goto v_resetjp_1392_;
}
else
{
lean_inc(v_a_1391_);
lean_dec(v___x_1389_);
v___x_1393_ = lean_box(0);
v_isShared_1394_ = v_isSharedCheck_1398_;
goto v_resetjp_1392_;
}
v_resetjp_1392_:
{
lean_object* v___x_1396_; 
if (v_isShared_1394_ == 0)
{
v___x_1396_ = v___x_1393_;
goto v_reusejp_1395_;
}
else
{
lean_object* v_reuseFailAlloc_1397_; 
v_reuseFailAlloc_1397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1397_, 0, v_a_1391_);
v___x_1396_ = v_reuseFailAlloc_1397_;
goto v_reusejp_1395_;
}
v_reusejp_1395_:
{
v___y_1374_ = v___x_1396_;
goto v___jp_1373_;
}
}
}
v___jp_1335_:
{
lean_object* v___x_1340_; lean_object* v___x_1342_; 
v___x_1340_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1340_, 0, v_name_1334_);
lean_ctor_set(v___x_1340_, 1, v___y_1337_);
lean_ctor_set(v___x_1340_, 2, v___y_1338_);
lean_ctor_set(v___x_1340_, 3, v___y_1336_);
lean_ctor_set(v___x_1340_, 4, v___y_1339_);
if (v_isShared_1332_ == 0)
{
lean_ctor_set(v___x_1331_, 0, v___x_1340_);
v___x_1342_ = v___x_1331_;
goto v_reusejp_1341_;
}
else
{
lean_object* v_reuseFailAlloc_1343_; 
v_reuseFailAlloc_1343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1343_, 0, v___x_1340_);
v___x_1342_ = v_reuseFailAlloc_1343_;
goto v_reusejp_1341_;
}
v_reusejp_1341_:
{
return v___x_1342_;
}
}
v___jp_1344_:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1348_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__1));
v___x_1349_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_a_1326_, v___x_1348_);
if (lean_obj_tag(v___x_1349_) == 0)
{
lean_object* v___x_1350_; 
lean_dec_ref_known(v___x_1349_, 1);
v___x_1350_ = lean_box(0);
v___y_1336_ = v___y_1347_;
v___y_1337_ = v___y_1345_;
v___y_1338_ = v___y_1346_;
v___y_1339_ = v___x_1350_;
goto v___jp_1335_;
}
else
{
lean_object* v_a_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1358_; 
v_a_1351_ = lean_ctor_get(v___x_1349_, 0);
v_isSharedCheck_1358_ = !lean_is_exclusive(v___x_1349_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1353_ = v___x_1349_;
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_a_1351_);
lean_dec(v___x_1349_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___x_1356_; 
if (v_isShared_1354_ == 0)
{
v___x_1356_ = v___x_1353_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v_a_1351_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
v___y_1336_ = v___y_1347_;
v___y_1337_ = v___y_1345_;
v___y_1338_ = v___y_1346_;
v___y_1339_ = v___x_1356_;
goto v___jp_1335_;
}
}
}
}
v___jp_1359_:
{
lean_object* v___x_1362_; lean_object* v___x_1363_; 
v___x_1362_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__2));
lean_inc(v_a_1326_);
v___x_1363_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_a_1326_, v___x_1362_);
if (lean_obj_tag(v___x_1363_) == 0)
{
lean_object* v___x_1364_; 
lean_dec_ref_known(v___x_1363_, 1);
v___x_1364_ = lean_box(0);
v___y_1345_ = v___y_1360_;
v___y_1346_ = v___y_1361_;
v___y_1347_ = v___x_1364_;
goto v___jp_1344_;
}
else
{
lean_object* v_a_1365_; lean_object* v___x_1367_; uint8_t v_isShared_1368_; uint8_t v_isSharedCheck_1372_; 
v_a_1365_ = lean_ctor_get(v___x_1363_, 0);
v_isSharedCheck_1372_ = !lean_is_exclusive(v___x_1363_);
if (v_isSharedCheck_1372_ == 0)
{
v___x_1367_ = v___x_1363_;
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
else
{
lean_inc(v_a_1365_);
lean_dec(v___x_1363_);
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
v_reuseFailAlloc_1371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1371_, 0, v_a_1365_);
v___x_1370_ = v_reuseFailAlloc_1371_;
goto v_reusejp_1369_;
}
v_reusejp_1369_:
{
v___y_1345_ = v___y_1360_;
v___y_1346_ = v___y_1361_;
v___y_1347_ = v___x_1370_;
goto v___jp_1344_;
}
}
}
}
v___jp_1373_:
{
lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1375_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__3));
lean_inc(v_a_1326_);
v___x_1376_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_a_1326_, v___x_1375_);
if (lean_obj_tag(v___x_1376_) == 0)
{
lean_object* v___x_1377_; 
lean_dec_ref_known(v___x_1376_, 1);
v___x_1377_ = lean_box(0);
v___y_1360_ = v___y_1374_;
v___y_1361_ = v___x_1377_;
goto v___jp_1359_;
}
else
{
lean_object* v_a_1378_; lean_object* v___x_1380_; uint8_t v_isShared_1381_; uint8_t v_isSharedCheck_1387_; 
v_a_1378_ = lean_ctor_get(v___x_1376_, 0);
v_isSharedCheck_1387_ = !lean_is_exclusive(v___x_1376_);
if (v_isSharedCheck_1387_ == 0)
{
v___x_1380_ = v___x_1376_;
v_isShared_1381_ = v_isSharedCheck_1387_;
goto v_resetjp_1379_;
}
else
{
lean_inc(v_a_1378_);
lean_dec(v___x_1376_);
v___x_1380_ = lean_box(0);
v_isShared_1381_ = v_isSharedCheck_1387_;
goto v_resetjp_1379_;
}
v_resetjp_1379_:
{
uint8_t v___x_1382_; 
v___x_1382_ = lean_string_dec_eq(v_a_1378_, v___x_1333_);
if (v___x_1382_ == 0)
{
lean_object* v___x_1384_; 
if (v_isShared_1381_ == 0)
{
v___x_1384_ = v___x_1380_;
goto v_reusejp_1383_;
}
else
{
lean_object* v_reuseFailAlloc_1385_; 
v_reuseFailAlloc_1385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1385_, 0, v_a_1378_);
v___x_1384_ = v_reuseFailAlloc_1385_;
goto v_reusejp_1383_;
}
v_reusejp_1383_:
{
v___y_1360_ = v___y_1374_;
v___y_1361_ = v___x_1384_;
goto v___jp_1359_;
}
}
else
{
lean_object* v___x_1386_; 
lean_del_object(v___x_1380_);
lean_dec(v_a_1378_);
v___x_1386_ = lean_box(0);
v___y_1360_ = v___y_1374_;
v___y_1361_ = v___x_1386_;
goto v___jp_1359_;
}
}
}
}
}
}
else
{
lean_object* v___x_1400_; 
lean_dec_ref(v___x_1328_);
lean_dec(v_a_1326_);
v___x_1400_ = lean_box(0);
return v___x_1400_;
}
}
else
{
lean_object* v___x_1401_; 
lean_dec_ref(v___x_1325_);
v___x_1401_ = lean_box(0);
return v___x_1401_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f(lean_object* v_js_1403_){
_start:
{
lean_object* v___x_1404_; lean_object* v___x_1405_; 
v___x_1404_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1));
lean_inc(v_js_1403_);
v___x_1405_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1403_, v___x_1404_);
if (lean_obj_tag(v___x_1405_) == 1)
{
lean_object* v_a_1406_; lean_object* v___x_1408_; uint8_t v_isShared_1409_; uint8_t v_isSharedCheck_1445_; 
v_a_1406_ = lean_ctor_get(v___x_1405_, 0);
v_isSharedCheck_1445_ = !lean_is_exclusive(v___x_1405_);
if (v_isSharedCheck_1445_ == 0)
{
v___x_1408_ = v___x_1405_;
v_isShared_1409_ = v_isSharedCheck_1445_;
goto v_resetjp_1407_;
}
else
{
lean_inc(v_a_1406_);
lean_dec(v___x_1405_);
v___x_1408_ = lean_box(0);
v_isShared_1409_ = v_isSharedCheck_1445_;
goto v_resetjp_1407_;
}
v_resetjp_1407_:
{
lean_object* v___y_1411_; lean_object* v___y_1412_; lean_object* v___y_1419_; lean_object* v___x_1434_; lean_object* v___x_1435_; 
v___x_1434_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__4));
lean_inc(v_js_1403_);
v___x_1435_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1403_, v___x_1434_);
if (lean_obj_tag(v___x_1435_) == 0)
{
lean_object* v___x_1436_; 
lean_dec_ref_known(v___x_1435_, 1);
v___x_1436_ = lean_box(0);
v___y_1419_ = v___x_1436_;
goto v___jp_1418_;
}
else
{
lean_object* v_a_1437_; lean_object* v___x_1439_; uint8_t v_isShared_1440_; uint8_t v_isSharedCheck_1444_; 
v_a_1437_ = lean_ctor_get(v___x_1435_, 0);
v_isSharedCheck_1444_ = !lean_is_exclusive(v___x_1435_);
if (v_isSharedCheck_1444_ == 0)
{
v___x_1439_ = v___x_1435_;
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
else
{
lean_inc(v_a_1437_);
lean_dec(v___x_1435_);
v___x_1439_ = lean_box(0);
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
v_resetjp_1438_:
{
lean_object* v___x_1442_; 
if (v_isShared_1440_ == 0)
{
v___x_1442_ = v___x_1439_;
goto v_reusejp_1441_;
}
else
{
lean_object* v_reuseFailAlloc_1443_; 
v_reuseFailAlloc_1443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1443_, 0, v_a_1437_);
v___x_1442_ = v_reuseFailAlloc_1443_;
goto v_reusejp_1441_;
}
v_reusejp_1441_:
{
v___y_1419_ = v___x_1442_;
goto v___jp_1418_;
}
}
}
v___jp_1410_:
{
lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1416_; 
v___x_1413_ = lean_box(0);
v___x_1414_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1414_, 0, v_a_1406_);
lean_ctor_set(v___x_1414_, 1, v___y_1411_);
lean_ctor_set(v___x_1414_, 2, v___y_1412_);
lean_ctor_set(v___x_1414_, 3, v___x_1413_);
lean_ctor_set(v___x_1414_, 4, v___x_1413_);
if (v_isShared_1409_ == 0)
{
lean_ctor_set(v___x_1408_, 0, v___x_1414_);
v___x_1416_ = v___x_1408_;
goto v_reusejp_1415_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v___x_1414_);
v___x_1416_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1415_;
}
v_reusejp_1415_:
{
return v___x_1416_;
}
}
v___jp_1418_:
{
lean_object* v___x_1420_; lean_object* v___x_1421_; 
v___x_1420_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f___closed__0));
v___x_1421_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1403_, v___x_1420_);
if (lean_obj_tag(v___x_1421_) == 0)
{
lean_object* v___x_1422_; 
lean_dec_ref_known(v___x_1421_, 1);
v___x_1422_ = lean_box(0);
v___y_1411_ = v___y_1419_;
v___y_1412_ = v___x_1422_;
goto v___jp_1410_;
}
else
{
lean_object* v_a_1423_; lean_object* v___x_1425_; uint8_t v_isShared_1426_; uint8_t v_isSharedCheck_1433_; 
v_a_1423_ = lean_ctor_get(v___x_1421_, 0);
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1421_);
if (v_isSharedCheck_1433_ == 0)
{
v___x_1425_ = v___x_1421_;
v_isShared_1426_ = v_isSharedCheck_1433_;
goto v_resetjp_1424_;
}
else
{
lean_inc(v_a_1423_);
lean_dec(v___x_1421_);
v___x_1425_ = lean_box(0);
v_isShared_1426_ = v_isSharedCheck_1433_;
goto v_resetjp_1424_;
}
v_resetjp_1424_:
{
lean_object* v___x_1427_; uint8_t v___x_1428_; 
v___x_1427_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
v___x_1428_ = lean_string_dec_eq(v_a_1423_, v___x_1427_);
if (v___x_1428_ == 0)
{
lean_object* v___x_1430_; 
if (v_isShared_1426_ == 0)
{
v___x_1430_ = v___x_1425_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v_a_1423_);
v___x_1430_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
v___y_1411_ = v___y_1419_;
v___y_1412_ = v___x_1430_;
goto v___jp_1410_;
}
}
else
{
lean_object* v___x_1432_; 
lean_del_object(v___x_1425_);
lean_dec(v_a_1423_);
v___x_1432_ = lean_box(0);
v___y_1411_ = v___y_1419_;
v___y_1412_ = v___x_1432_;
goto v___jp_1410_;
}
}
}
}
}
}
else
{
lean_object* v___x_1446_; 
lean_dec_ref(v___x_1405_);
lean_dec(v_js_1403_);
v___x_1446_ = lean_box(0);
return v___x_1446_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f(lean_object* v_js_1448_){
_start:
{
lean_object* v___x_1449_; lean_object* v___x_1450_; 
v___x_1449_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__1));
lean_inc(v_js_1448_);
v___x_1450_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1448_, v___x_1449_);
if (lean_obj_tag(v___x_1450_) == 1)
{
lean_object* v_a_1451_; lean_object* v___x_1453_; uint8_t v_isShared_1454_; uint8_t v_isSharedCheck_1505_; 
v_a_1451_ = lean_ctor_get(v___x_1450_, 0);
v_isSharedCheck_1505_ = !lean_is_exclusive(v___x_1450_);
if (v_isSharedCheck_1505_ == 0)
{
v___x_1453_ = v___x_1450_;
v_isShared_1454_ = v_isSharedCheck_1505_;
goto v_resetjp_1452_;
}
else
{
lean_inc(v_a_1451_);
lean_dec(v___x_1450_);
v___x_1453_ = lean_box(0);
v_isShared_1454_ = v_isSharedCheck_1505_;
goto v_resetjp_1452_;
}
v_resetjp_1452_:
{
lean_object* v___y_1456_; lean_object* v___y_1457_; lean_object* v___y_1458_; lean_object* v___y_1465_; lean_object* v___y_1466_; lean_object* v___y_1479_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1494_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f___closed__0));
lean_inc(v_js_1448_);
v___x_1495_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1448_, v___x_1494_);
if (lean_obj_tag(v___x_1495_) == 0)
{
lean_object* v___x_1496_; 
lean_dec_ref_known(v___x_1495_, 1);
v___x_1496_ = lean_box(0);
v___y_1479_ = v___x_1496_;
goto v___jp_1478_;
}
else
{
lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1504_; 
v_a_1497_ = lean_ctor_get(v___x_1495_, 0);
v_isSharedCheck_1504_ = !lean_is_exclusive(v___x_1495_);
if (v_isSharedCheck_1504_ == 0)
{
v___x_1499_ = v___x_1495_;
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1495_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v___x_1502_; 
if (v_isShared_1500_ == 0)
{
v___x_1502_ = v___x_1499_;
goto v_reusejp_1501_;
}
else
{
lean_object* v_reuseFailAlloc_1503_; 
v_reuseFailAlloc_1503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1503_, 0, v_a_1497_);
v___x_1502_ = v_reuseFailAlloc_1503_;
goto v_reusejp_1501_;
}
v_reusejp_1501_:
{
v___y_1479_ = v___x_1502_;
goto v___jp_1478_;
}
}
}
v___jp_1455_:
{
lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1462_; 
v___x_1459_ = lean_box(0);
v___x_1460_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1460_, 0, v_a_1451_);
lean_ctor_set(v___x_1460_, 1, v___y_1457_);
lean_ctor_set(v___x_1460_, 2, v___y_1456_);
lean_ctor_set(v___x_1460_, 3, v___x_1459_);
lean_ctor_set(v___x_1460_, 4, v___y_1458_);
if (v_isShared_1454_ == 0)
{
lean_ctor_set(v___x_1453_, 0, v___x_1460_);
v___x_1462_ = v___x_1453_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1463_; 
v_reuseFailAlloc_1463_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1463_, 0, v___x_1460_);
v___x_1462_ = v_reuseFailAlloc_1463_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
return v___x_1462_;
}
}
v___jp_1464_:
{
lean_object* v___x_1467_; lean_object* v___x_1468_; 
v___x_1467_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f___closed__1));
v___x_1468_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1448_, v___x_1467_);
if (lean_obj_tag(v___x_1468_) == 0)
{
lean_object* v___x_1469_; 
lean_dec_ref_known(v___x_1468_, 1);
v___x_1469_ = lean_box(0);
v___y_1456_ = v___y_1466_;
v___y_1457_ = v___y_1465_;
v___y_1458_ = v___x_1469_;
goto v___jp_1455_;
}
else
{
lean_object* v_a_1470_; lean_object* v___x_1472_; uint8_t v_isShared_1473_; uint8_t v_isSharedCheck_1477_; 
v_a_1470_ = lean_ctor_get(v___x_1468_, 0);
v_isSharedCheck_1477_ = !lean_is_exclusive(v___x_1468_);
if (v_isSharedCheck_1477_ == 0)
{
v___x_1472_ = v___x_1468_;
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
else
{
lean_inc(v_a_1470_);
lean_dec(v___x_1468_);
v___x_1472_ = lean_box(0);
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
v_resetjp_1471_:
{
lean_object* v___x_1475_; 
if (v_isShared_1473_ == 0)
{
v___x_1475_ = v___x_1472_;
goto v_reusejp_1474_;
}
else
{
lean_object* v_reuseFailAlloc_1476_; 
v_reuseFailAlloc_1476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1476_, 0, v_a_1470_);
v___x_1475_ = v_reuseFailAlloc_1476_;
goto v_reusejp_1474_;
}
v_reusejp_1474_:
{
v___y_1456_ = v___y_1466_;
v___y_1457_ = v___y_1465_;
v___y_1458_ = v___x_1475_;
goto v___jp_1455_;
}
}
}
}
v___jp_1478_:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1480_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLoogleJson_x3f___closed__0));
lean_inc(v_js_1448_);
v___x_1481_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__2(v_js_1448_, v___x_1480_);
if (lean_obj_tag(v___x_1481_) == 0)
{
lean_object* v___x_1482_; 
lean_dec_ref_known(v___x_1481_, 1);
v___x_1482_ = lean_box(0);
v___y_1465_ = v___y_1479_;
v___y_1466_ = v___x_1482_;
goto v___jp_1464_;
}
else
{
lean_object* v_a_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1493_; 
v_a_1483_ = lean_ctor_get(v___x_1481_, 0);
v_isSharedCheck_1493_ = !lean_is_exclusive(v___x_1481_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1485_ = v___x_1481_;
v_isShared_1486_ = v_isSharedCheck_1493_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_a_1483_);
lean_dec(v___x_1481_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1493_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1487_; uint8_t v___x_1488_; 
v___x_1487_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
v___x_1488_ = lean_string_dec_eq(v_a_1483_, v___x_1487_);
if (v___x_1488_ == 0)
{
lean_object* v___x_1490_; 
if (v_isShared_1486_ == 0)
{
v___x_1490_ = v___x_1485_;
goto v_reusejp_1489_;
}
else
{
lean_object* v_reuseFailAlloc_1491_; 
v_reuseFailAlloc_1491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1491_, 0, v_a_1483_);
v___x_1490_ = v_reuseFailAlloc_1491_;
goto v_reusejp_1489_;
}
v_reusejp_1489_:
{
v___y_1465_ = v___y_1479_;
v___y_1466_ = v___x_1490_;
goto v___jp_1464_;
}
}
else
{
lean_object* v___x_1492_; 
lean_del_object(v___x_1485_);
lean_dec(v_a_1483_);
v___x_1492_ = lean_box(0);
v___y_1465_ = v___y_1479_;
v___y_1466_ = v___x_1492_;
goto v___jp_1464_;
}
}
}
}
}
}
else
{
lean_object* v___x_1506_; 
lean_dec_ref(v___x_1450_);
lean_dec(v_js_1448_);
v___x_1506_ = lean_box(0);
return v___x_1506_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion(lean_object* v_sr_1510_){
_start:
{
lean_object* v___y_1512_; lean_object* v___y_1513_; lean_object* v___y_1514_; lean_object* v_name_1517_; lean_object* v_type_x3f_1518_; lean_object* v_docString_x3f_1519_; lean_object* v___y_1521_; 
v_name_1517_ = lean_ctor_get(v_sr_1510_, 0);
lean_inc_ref(v_name_1517_);
v_type_x3f_1518_ = lean_ctor_get(v_sr_1510_, 1);
lean_inc(v_type_x3f_1518_);
v_docString_x3f_1519_ = lean_ctor_get(v_sr_1510_, 2);
lean_inc(v_docString_x3f_1519_);
lean_dec_ref(v_sr_1510_);
if (lean_obj_tag(v_docString_x3f_1519_) == 0)
{
lean_object* v___x_1539_; 
v___x_1539_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
v___y_1521_ = v___x_1539_;
goto v___jp_1520_;
}
else
{
lean_object* v_val_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; 
v_val_1540_ = lean_ctor_get(v_docString_x3f_1519_, 0);
lean_inc(v_val_1540_);
lean_dec_ref_known(v_docString_x3f_1519_, 1);
v___x_1541_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__2));
v___x_1542_ = lean_string_append(v_val_1540_, v___x_1541_);
v___y_1521_ = v___x_1542_;
goto v___jp_1520_;
}
v___jp_1511_:
{
lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1515_ = lean_box(0);
lean_inc(v___y_1512_);
v___x_1516_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1516_, 0, v___y_1513_);
lean_ctor_set(v___x_1516_, 1, v___y_1512_);
lean_ctor_set(v___x_1516_, 2, v___y_1514_);
lean_ctor_set(v___x_1516_, 3, v___x_1515_);
lean_ctor_set(v___x_1516_, 4, v___x_1515_);
lean_ctor_set(v___x_1516_, 5, v___x_1515_);
return v___x_1516_;
}
v___jp_1520_:
{
lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v___x_1524_; lean_object* v___x_1525_; 
v___x_1522_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__0));
v___x_1523_ = lean_string_append(v___x_1522_, v_name_1517_);
lean_dec_ref(v_name_1517_);
v___x_1524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1524_, 0, v___x_1523_);
v___x_1525_ = lean_box(0);
if (lean_obj_tag(v_type_x3f_1518_) == 0)
{
lean_dec_ref(v___y_1521_);
v___y_1512_ = v___x_1525_;
v___y_1513_ = v___x_1524_;
v___y_1514_ = v_type_x3f_1518_;
goto v___jp_1511_;
}
else
{
lean_object* v_val_1526_; lean_object* v___x_1528_; uint8_t v_isShared_1529_; uint8_t v_isSharedCheck_1538_; 
v_val_1526_ = lean_ctor_get(v_type_x3f_1518_, 0);
v_isSharedCheck_1538_ = !lean_is_exclusive(v_type_x3f_1518_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1528_ = v_type_x3f_1518_;
v_isShared_1529_ = v_isSharedCheck_1538_;
goto v_resetjp_1527_;
}
else
{
lean_inc(v_val_1526_);
lean_dec(v_type_x3f_1518_);
v___x_1528_ = lean_box(0);
v_isShared_1529_ = v_isSharedCheck_1538_;
goto v_resetjp_1527_;
}
v_resetjp_1527_:
{
lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1536_; 
v___x_1530_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__1));
v___x_1531_ = lean_string_append(v___x_1530_, v_val_1526_);
lean_dec(v_val_1526_);
v___x_1532_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion___closed__2));
v___x_1533_ = lean_string_append(v___x_1532_, v___y_1521_);
lean_dec_ref(v___y_1521_);
v___x_1534_ = lean_string_append(v___x_1531_, v___x_1533_);
lean_dec_ref(v___x_1533_);
if (v_isShared_1529_ == 0)
{
lean_ctor_set(v___x_1528_, 0, v___x_1534_);
v___x_1536_ = v___x_1528_;
goto v_reusejp_1535_;
}
else
{
lean_object* v_reuseFailAlloc_1537_; 
v_reuseFailAlloc_1537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1537_, 0, v___x_1534_);
v___x_1536_ = v_reuseFailAlloc_1537_;
goto v_reusejp_1535_;
}
v_reusejp_1535_:
{
v___y_1512_ = v___x_1525_;
v___y_1513_ = v___x_1524_;
v___y_1514_ = v___x_1536_;
goto v___jp_1511_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion(lean_object* v_sr_1545_){
_start:
{
lean_object* v_type_x3f_1546_; 
v_type_x3f_1546_ = lean_ctor_get(v_sr_1545_, 1);
lean_inc(v_type_x3f_1546_);
if (lean_obj_tag(v_type_x3f_1546_) == 0)
{
lean_object* v_name_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; 
v_name_1547_ = lean_ctor_get(v_sr_1545_, 0);
lean_inc_ref(v_name_1547_);
lean_dec_ref(v_sr_1545_);
v___x_1548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1548_, 0, v_name_1547_);
v___x_1549_ = lean_box(0);
v___x_1550_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1550_, 0, v___x_1548_);
lean_ctor_set(v___x_1550_, 1, v_type_x3f_1546_);
lean_ctor_set(v___x_1550_, 2, v_type_x3f_1546_);
lean_ctor_set(v___x_1550_, 3, v___x_1549_);
lean_ctor_set(v___x_1550_, 4, v___x_1549_);
lean_ctor_set(v___x_1550_, 5, v___x_1549_);
return v___x_1550_;
}
else
{
lean_object* v_name_1551_; lean_object* v_val_1552_; lean_object* v___x_1554_; uint8_t v_isShared_1555_; uint8_t v_isSharedCheck_1566_; 
v_name_1551_ = lean_ctor_get(v_sr_1545_, 0);
lean_inc_ref(v_name_1551_);
lean_dec_ref(v_sr_1545_);
v_val_1552_ = lean_ctor_get(v_type_x3f_1546_, 0);
v_isSharedCheck_1566_ = !lean_is_exclusive(v_type_x3f_1546_);
if (v_isSharedCheck_1566_ == 0)
{
v___x_1554_ = v_type_x3f_1546_;
v_isShared_1555_ = v_isSharedCheck_1566_;
goto v_resetjp_1553_;
}
else
{
lean_inc(v_val_1552_);
lean_dec(v_type_x3f_1546_);
v___x_1554_ = lean_box(0);
v_isShared_1555_ = v_isSharedCheck_1566_;
goto v_resetjp_1553_;
}
v_resetjp_1553_:
{
lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1563_; 
v___x_1556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1556_, 0, v_name_1551_);
v___x_1557_ = lean_box(0);
v___x_1558_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__0));
v___x_1559_ = lean_string_append(v___x_1558_, v_val_1552_);
lean_dec(v_val_1552_);
v___x_1560_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__1));
v___x_1561_ = lean_string_append(v___x_1559_, v___x_1560_);
if (v_isShared_1555_ == 0)
{
lean_ctor_set(v___x_1554_, 0, v___x_1561_);
v___x_1563_ = v___x_1554_;
goto v_reusejp_1562_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v___x_1561_);
v___x_1563_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1562_;
}
v_reusejp_1562_:
{
lean_object* v___x_1564_; 
v___x_1564_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1564_, 0, v___x_1556_);
lean_ctor_set(v___x_1564_, 1, v___x_1557_);
lean_ctor_set(v___x_1564_, 2, v___x_1563_);
lean_ctor_set(v___x_1564_, 3, v___x_1557_);
lean_ctor_set(v___x_1564_, 4, v___x_1557_);
lean_ctor_set(v___x_1564_, 5, v___x_1557_);
return v___x_1564_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions(lean_object* v_sr_1574_){
_start:
{
lean_object* v_type_x3f_1575_; 
v_type_x3f_1575_ = lean_ctor_get(v_sr_1574_, 1);
lean_inc(v_type_x3f_1575_);
if (lean_obj_tag(v_type_x3f_1575_) == 0)
{
lean_object* v___x_1576_; 
lean_dec_ref(v_sr_1574_);
v___x_1576_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__0));
return v___x_1576_;
}
else
{
lean_object* v_name_1577_; lean_object* v_val_1578_; lean_object* v___x_1580_; uint8_t v_isShared_1581_; uint8_t v_isSharedCheck_1613_; 
v_name_1577_ = lean_ctor_get(v_sr_1574_, 0);
lean_inc_ref(v_name_1577_);
lean_dec_ref(v_sr_1574_);
v_val_1578_ = lean_ctor_get(v_type_x3f_1575_, 0);
v_isSharedCheck_1613_ = !lean_is_exclusive(v_type_x3f_1575_);
if (v_isSharedCheck_1613_ == 0)
{
v___x_1580_ = v_type_x3f_1575_;
v_isShared_1581_ = v_isSharedCheck_1613_;
goto v_resetjp_1579_;
}
else
{
lean_inc(v_val_1578_);
lean_dec(v_type_x3f_1575_);
v___x_1580_ = lean_box(0);
v_isShared_1581_ = v_isSharedCheck_1613_;
goto v_resetjp_1579_;
}
v_resetjp_1579_:
{
lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1585_; 
v___x_1582_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__1));
v___x_1583_ = lean_string_append(v___x_1582_, v_name_1577_);
if (v_isShared_1581_ == 0)
{
lean_ctor_set(v___x_1580_, 0, v___x_1583_);
v___x_1585_ = v___x_1580_;
goto v_reusejp_1584_;
}
else
{
lean_object* v_reuseFailAlloc_1612_; 
v_reuseFailAlloc_1612_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1612_, 0, v___x_1583_);
v___x_1585_ = v_reuseFailAlloc_1612_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; 
v___x_1586_ = lean_box(0);
v___x_1587_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1587_, 0, v___x_1585_);
lean_ctor_set(v___x_1587_, 1, v___x_1586_);
lean_ctor_set(v___x_1587_, 2, v___x_1586_);
lean_ctor_set(v___x_1587_, 3, v___x_1586_);
lean_ctor_set(v___x_1587_, 4, v___x_1586_);
lean_ctor_set(v___x_1587_, 5, v___x_1586_);
v___x_1588_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__2));
v___x_1589_ = lean_string_append(v___x_1588_, v_val_1578_);
lean_dec(v_val_1578_);
v___x_1590_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg___closed__4));
v___x_1591_ = lean_string_append(v___x_1589_, v___x_1590_);
v___x_1592_ = lean_string_append(v___x_1591_, v_name_1577_);
v___x_1593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1593_, 0, v___x_1592_);
v___x_1594_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1594_, 0, v___x_1593_);
lean_ctor_set(v___x_1594_, 1, v___x_1586_);
lean_ctor_set(v___x_1594_, 2, v___x_1586_);
lean_ctor_set(v___x_1594_, 3, v___x_1586_);
lean_ctor_set(v___x_1594_, 4, v___x_1586_);
lean_ctor_set(v___x_1594_, 5, v___x_1586_);
v___x_1595_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__3));
v___x_1596_ = lean_string_append(v___x_1595_, v_name_1577_);
v___x_1597_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__4));
v___x_1598_ = lean_string_append(v___x_1596_, v___x_1597_);
v___x_1599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1599_, 0, v___x_1598_);
v___x_1600_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1600_, 0, v___x_1599_);
lean_ctor_set(v___x_1600_, 1, v___x_1586_);
lean_ctor_set(v___x_1600_, 2, v___x_1586_);
lean_ctor_set(v___x_1600_, 3, v___x_1586_);
lean_ctor_set(v___x_1600_, 4, v___x_1586_);
lean_ctor_set(v___x_1600_, 5, v___x_1586_);
v___x_1601_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__5));
v___x_1602_ = lean_string_append(v___x_1601_, v_name_1577_);
lean_dec_ref(v_name_1577_);
v___x_1603_ = lean_string_append(v___x_1602_, v___x_1597_);
v___x_1604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1604_, 0, v___x_1603_);
v___x_1605_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1605_, 0, v___x_1604_);
lean_ctor_set(v___x_1605_, 1, v___x_1586_);
lean_ctor_set(v___x_1605_, 2, v___x_1586_);
lean_ctor_set(v___x_1605_, 3, v___x_1586_);
lean_ctor_set(v___x_1605_, 4, v___x_1586_);
lean_ctor_set(v___x_1605_, 5, v___x_1586_);
v___x_1606_ = lean_unsigned_to_nat(4u);
v___x_1607_ = lean_mk_empty_array_with_capacity(v___x_1606_);
v___x_1608_ = lean_array_push(v___x_1607_, v___x_1587_);
v___x_1609_ = lean_array_push(v___x_1608_, v___x_1594_);
v___x_1610_ = lean_array_push(v___x_1609_, v___x_1600_);
v___x_1611_ = lean_array_push(v___x_1610_, v___x_1605_);
return v___x_1611_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0(lean_object* v_as_1614_, size_t v_i_1615_, size_t v_stop_1616_, lean_object* v_b_1617_){
_start:
{
lean_object* v___y_1619_; uint8_t v___x_1623_; 
v___x_1623_ = lean_usize_dec_eq(v_i_1615_, v_stop_1616_);
if (v___x_1623_ == 0)
{
lean_object* v___x_1624_; lean_object* v___x_1625_; 
v___x_1624_ = lean_array_uget_borrowed(v_as_1614_, v_i_1615_);
lean_inc(v___x_1624_);
v___x_1625_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_ofLeanSearchJson_x3f(v___x_1624_);
if (lean_obj_tag(v___x_1625_) == 0)
{
v___y_1619_ = v_b_1617_;
goto v___jp_1618_;
}
else
{
lean_object* v_val_1626_; lean_object* v___x_1627_; 
v_val_1626_ = lean_ctor_get(v___x_1625_, 0);
lean_inc(v_val_1626_);
lean_dec_ref_known(v___x_1625_, 1);
v___x_1627_ = lean_array_push(v_b_1617_, v_val_1626_);
v___y_1619_ = v___x_1627_;
goto v___jp_1618_;
}
}
else
{
return v_b_1617_;
}
v___jp_1618_:
{
size_t v___x_1620_; size_t v___x_1621_; 
v___x_1620_ = ((size_t)1ULL);
v___x_1621_ = lean_usize_add(v_i_1615_, v___x_1620_);
v_i_1615_ = v___x_1621_;
v_b_1617_ = v___y_1619_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0___boxed(lean_object* v_as_1628_, lean_object* v_i_1629_, lean_object* v_stop_1630_, lean_object* v_b_1631_){
_start:
{
size_t v_i_boxed_1632_; size_t v_stop_boxed_1633_; lean_object* v_res_1634_; 
v_i_boxed_1632_ = lean_unbox_usize(v_i_1629_);
lean_dec(v_i_1629_);
v_stop_boxed_1633_ = lean_unbox_usize(v_stop_1630_);
lean_dec(v_stop_1630_);
v_res_1634_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0(v_as_1628_, v_i_boxed_1632_, v_stop_boxed_1633_, v_b_1631_);
lean_dec_ref(v_as_1628_);
return v_res_1634_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0(lean_object* v_as_1637_, lean_object* v_start_1638_, lean_object* v_stop_1639_){
_start:
{
lean_object* v___x_1640_; uint8_t v___x_1641_; 
v___x_1640_ = ((lean_object*)(lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___closed__0));
v___x_1641_ = lean_nat_dec_lt(v_start_1638_, v_stop_1639_);
if (v___x_1641_ == 0)
{
return v___x_1640_;
}
else
{
lean_object* v___x_1642_; uint8_t v___x_1643_; 
v___x_1642_ = lean_array_get_size(v_as_1637_);
v___x_1643_ = lean_nat_dec_le(v_stop_1639_, v___x_1642_);
if (v___x_1643_ == 0)
{
uint8_t v___x_1644_; 
v___x_1644_ = lean_nat_dec_lt(v_start_1638_, v___x_1642_);
if (v___x_1644_ == 0)
{
return v___x_1640_;
}
else
{
size_t v___x_1645_; size_t v___x_1646_; lean_object* v___x_1647_; 
v___x_1645_ = lean_usize_of_nat(v_start_1638_);
v___x_1646_ = lean_usize_of_nat(v___x_1642_);
v___x_1647_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0(v_as_1637_, v___x_1645_, v___x_1646_, v___x_1640_);
return v___x_1647_;
}
}
else
{
size_t v___x_1648_; size_t v___x_1649_; lean_object* v___x_1650_; 
v___x_1648_ = lean_usize_of_nat(v_start_1638_);
v___x_1649_ = lean_usize_of_nat(v_stop_1639_);
v___x_1650_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0_spec__0(v_as_1637_, v___x_1648_, v___x_1649_, v___x_1640_);
return v___x_1650_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___boxed(lean_object* v_as_1651_, lean_object* v_start_1652_, lean_object* v_stop_1653_){
_start:
{
lean_object* v_res_1654_; 
v_res_1654_ = lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0(v_as_1651_, v_start_1652_, v_stop_1653_);
lean_dec(v_stop_1653_);
lean_dec(v_start_1652_);
lean_dec_ref(v_as_1651_);
return v_res_1654_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___redArg(lean_object* v_s_1655_, lean_object* v_num__results_1656_, lean_object* v_a_1657_){
_start:
{
lean_object* v___x_1659_; 
v___x_1659_ = lp_LeanSearchClient_LeanSearchClient_getLeanSearchQueryJson___redArg(v_s_1655_, v_num__results_1656_, v_a_1657_);
if (lean_obj_tag(v___x_1659_) == 0)
{
lean_object* v_a_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1670_; 
v_a_1660_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1662_ = v___x_1659_;
v_isShared_1663_ = v_isSharedCheck_1670_;
goto v_resetjp_1661_;
}
else
{
lean_inc(v_a_1660_);
lean_dec(v___x_1659_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1670_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1668_; 
v___x_1664_ = lean_unsigned_to_nat(0u);
v___x_1665_ = lean_array_get_size(v_a_1660_);
v___x_1666_ = lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0(v_a_1660_, v___x_1664_, v___x_1665_);
lean_dec(v_a_1660_);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v___x_1666_);
v___x_1668_ = v___x_1662_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v___x_1666_);
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
v_a_1671_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1673_ = v___x_1659_;
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___x_1659_);
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
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___redArg___boxed(lean_object* v_s_1679_, lean_object* v_num__results_1680_, lean_object* v_a_1681_, lean_object* v_a_1682_){
_start:
{
lean_object* v_res_1683_; 
v_res_1683_ = lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___redArg(v_s_1679_, v_num__results_1680_, v_a_1681_);
lean_dec_ref(v_a_1681_);
return v_res_1683_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch(lean_object* v_s_1684_, lean_object* v_num__results_1685_, lean_object* v_a_1686_, lean_object* v_a_1687_, lean_object* v_a_1688_, lean_object* v_a_1689_){
_start:
{
lean_object* v___x_1691_; 
v___x_1691_ = lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___redArg(v_s_1684_, v_num__results_1685_, v_a_1688_);
return v___x_1691_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryLeanSearch___boxed(lean_object* v_s_1692_, lean_object* v_num__results_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_){
_start:
{
lean_object* v_res_1699_; 
v_res_1699_ = lp_LeanSearchClient_LeanSearchClient_queryLeanSearch(v_s_1692_, v_num__results_1693_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_);
lean_dec(v_a_1697_);
lean_dec_ref(v_a_1696_);
lean_dec(v_a_1695_);
lean_dec_ref(v_a_1694_);
return v_res_1699_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0(lean_object* v_as_1700_, size_t v_i_1701_, size_t v_stop_1702_, lean_object* v_b_1703_){
_start:
{
lean_object* v___y_1705_; uint8_t v___x_1709_; 
v___x_1709_ = lean_usize_dec_eq(v_i_1701_, v_stop_1702_);
if (v___x_1709_ == 0)
{
lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1710_ = lean_array_uget_borrowed(v_as_1700_, v_i_1701_);
lean_inc(v___x_1710_);
v___x_1711_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_ofStateSearchJson_x3f(v___x_1710_);
if (lean_obj_tag(v___x_1711_) == 0)
{
v___y_1705_ = v_b_1703_;
goto v___jp_1704_;
}
else
{
lean_object* v_val_1712_; lean_object* v___x_1713_; 
v_val_1712_ = lean_ctor_get(v___x_1711_, 0);
lean_inc(v_val_1712_);
lean_dec_ref_known(v___x_1711_, 1);
v___x_1713_ = lean_array_push(v_b_1703_, v_val_1712_);
v___y_1705_ = v___x_1713_;
goto v___jp_1704_;
}
}
else
{
return v_b_1703_;
}
v___jp_1704_:
{
size_t v___x_1706_; size_t v___x_1707_; 
v___x_1706_ = ((size_t)1ULL);
v___x_1707_ = lean_usize_add(v_i_1701_, v___x_1706_);
v_i_1701_ = v___x_1707_;
v_b_1703_ = v___y_1705_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0___boxed(lean_object* v_as_1714_, lean_object* v_i_1715_, lean_object* v_stop_1716_, lean_object* v_b_1717_){
_start:
{
size_t v_i_boxed_1718_; size_t v_stop_boxed_1719_; lean_object* v_res_1720_; 
v_i_boxed_1718_ = lean_unbox_usize(v_i_1715_);
lean_dec(v_i_1715_);
v_stop_boxed_1719_ = lean_unbox_usize(v_stop_1716_);
lean_dec(v_stop_1716_);
v_res_1720_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0(v_as_1714_, v_i_boxed_1718_, v_stop_boxed_1719_, v_b_1717_);
lean_dec_ref(v_as_1714_);
return v_res_1720_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0(lean_object* v_as_1721_, lean_object* v_start_1722_, lean_object* v_stop_1723_){
_start:
{
lean_object* v___x_1724_; uint8_t v___x_1725_; 
v___x_1724_ = ((lean_object*)(lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryLeanSearch_spec__0___closed__0));
v___x_1725_ = lean_nat_dec_lt(v_start_1722_, v_stop_1723_);
if (v___x_1725_ == 0)
{
return v___x_1724_;
}
else
{
lean_object* v___x_1726_; uint8_t v___x_1727_; 
v___x_1726_ = lean_array_get_size(v_as_1721_);
v___x_1727_ = lean_nat_dec_le(v_stop_1723_, v___x_1726_);
if (v___x_1727_ == 0)
{
uint8_t v___x_1728_; 
v___x_1728_ = lean_nat_dec_lt(v_start_1722_, v___x_1726_);
if (v___x_1728_ == 0)
{
return v___x_1724_;
}
else
{
size_t v___x_1729_; size_t v___x_1730_; lean_object* v___x_1731_; 
v___x_1729_ = lean_usize_of_nat(v_start_1722_);
v___x_1730_ = lean_usize_of_nat(v___x_1726_);
v___x_1731_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0(v_as_1721_, v___x_1729_, v___x_1730_, v___x_1724_);
return v___x_1731_;
}
}
else
{
size_t v___x_1732_; size_t v___x_1733_; lean_object* v___x_1734_; 
v___x_1732_ = lean_usize_of_nat(v_start_1722_);
v___x_1733_ = lean_usize_of_nat(v_stop_1723_);
v___x_1734_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0_spec__0(v_as_1721_, v___x_1732_, v___x_1733_, v___x_1724_);
return v___x_1734_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0___boxed(lean_object* v_as_1735_, lean_object* v_start_1736_, lean_object* v_stop_1737_){
_start:
{
lean_object* v_res_1738_; 
v_res_1738_ = lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0(v_as_1735_, v_start_1736_, v_stop_1737_);
lean_dec(v_stop_1737_);
lean_dec(v_start_1736_);
lean_dec_ref(v_as_1735_);
return v_res_1738_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg(lean_object* v_s_1739_, lean_object* v_num__results_1740_, lean_object* v_rev_1741_, lean_object* v_a_1742_){
_start:
{
lean_object* v___x_1744_; 
v___x_1744_ = lp_LeanSearchClient_LeanSearchClient_getStateSearchQueryJson___redArg(v_s_1739_, v_num__results_1740_, v_rev_1741_, v_a_1742_);
if (lean_obj_tag(v___x_1744_) == 0)
{
lean_object* v_a_1745_; lean_object* v___x_1747_; uint8_t v_isShared_1748_; uint8_t v_isSharedCheck_1755_; 
v_a_1745_ = lean_ctor_get(v___x_1744_, 0);
v_isSharedCheck_1755_ = !lean_is_exclusive(v___x_1744_);
if (v_isSharedCheck_1755_ == 0)
{
v___x_1747_ = v___x_1744_;
v_isShared_1748_ = v_isSharedCheck_1755_;
goto v_resetjp_1746_;
}
else
{
lean_inc(v_a_1745_);
lean_dec(v___x_1744_);
v___x_1747_ = lean_box(0);
v_isShared_1748_ = v_isSharedCheck_1755_;
goto v_resetjp_1746_;
}
v_resetjp_1746_:
{
lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1753_; 
v___x_1749_ = lean_unsigned_to_nat(0u);
v___x_1750_ = lean_array_get_size(v_a_1745_);
v___x_1751_ = lp_LeanSearchClient_Array_filterMapM___at___00LeanSearchClient_queryStateSearch_spec__0(v_a_1745_, v___x_1749_, v___x_1750_);
lean_dec(v_a_1745_);
if (v_isShared_1748_ == 0)
{
lean_ctor_set(v___x_1747_, 0, v___x_1751_);
v___x_1753_ = v___x_1747_;
goto v_reusejp_1752_;
}
else
{
lean_object* v_reuseFailAlloc_1754_; 
v_reuseFailAlloc_1754_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1754_, 0, v___x_1751_);
v___x_1753_ = v_reuseFailAlloc_1754_;
goto v_reusejp_1752_;
}
v_reusejp_1752_:
{
return v___x_1753_;
}
}
}
else
{
lean_object* v_a_1756_; lean_object* v___x_1758_; uint8_t v_isShared_1759_; uint8_t v_isSharedCheck_1763_; 
v_a_1756_ = lean_ctor_get(v___x_1744_, 0);
v_isSharedCheck_1763_ = !lean_is_exclusive(v___x_1744_);
if (v_isSharedCheck_1763_ == 0)
{
v___x_1758_ = v___x_1744_;
v_isShared_1759_ = v_isSharedCheck_1763_;
goto v_resetjp_1757_;
}
else
{
lean_inc(v_a_1756_);
lean_dec(v___x_1744_);
v___x_1758_ = lean_box(0);
v_isShared_1759_ = v_isSharedCheck_1763_;
goto v_resetjp_1757_;
}
v_resetjp_1757_:
{
lean_object* v___x_1761_; 
if (v_isShared_1759_ == 0)
{
v___x_1761_ = v___x_1758_;
goto v_reusejp_1760_;
}
else
{
lean_object* v_reuseFailAlloc_1762_; 
v_reuseFailAlloc_1762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1762_, 0, v_a_1756_);
v___x_1761_ = v_reuseFailAlloc_1762_;
goto v_reusejp_1760_;
}
v_reusejp_1760_:
{
return v___x_1761_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg___boxed(lean_object* v_s_1764_, lean_object* v_num__results_1765_, lean_object* v_rev_1766_, lean_object* v_a_1767_, lean_object* v_a_1768_){
_start:
{
lean_object* v_res_1769_; 
v_res_1769_ = lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg(v_s_1764_, v_num__results_1765_, v_rev_1766_, v_a_1767_);
lean_dec_ref(v_a_1767_);
return v_res_1769_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch(lean_object* v_s_1770_, lean_object* v_num__results_1771_, lean_object* v_rev_1772_, lean_object* v_a_1773_, lean_object* v_a_1774_, lean_object* v_a_1775_, lean_object* v_a_1776_){
_start:
{
lean_object* v___x_1778_; 
v___x_1778_ = lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg(v_s_1770_, v_num__results_1771_, v_rev_1772_, v_a_1775_);
return v___x_1778_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_queryStateSearch___boxed(lean_object* v_s_1779_, lean_object* v_num__results_1780_, lean_object* v_rev_1781_, lean_object* v_a_1782_, lean_object* v_a_1783_, lean_object* v_a_1784_, lean_object* v_a_1785_, lean_object* v_a_1786_){
_start:
{
lean_object* v_res_1787_; 
v_res_1787_ = lp_LeanSearchClient_LeanSearchClient_queryStateSearch(v_s_1779_, v_num__results_1780_, v_rev_1781_, v_a_1782_, v_a_1783_, v_a_1784_, v_a_1785_);
lean_dec(v_a_1785_);
lean_dec_ref(v_a_1784_);
lean_dec(v_a_1783_);
lean_dec_ref(v_a_1782_);
return v_res_1787_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3(void){
_start:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; 
v___x_1793_ = lean_box(0);
v___x_1794_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__2));
v___x_1795_ = l_Lean_mkConst(v___x_1794_, v___x_1793_);
return v___x_1795_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__9(void){
_start:
{
lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; 
v___x_1804_ = lean_box(0);
v___x_1805_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__8));
v___x_1806_ = l_Lean_mkConst(v___x_1805_, v___x_1804_);
return v___x_1806_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm(lean_object* v_expectedType_x3f_1807_, lean_object* v_a_1808_, lean_object* v_a_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_){
_start:
{
if (lean_obj_tag(v_expectedType_x3f_1807_) == 0)
{
lean_object* v___x_1813_; lean_object* v___x_1814_; 
v___x_1813_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3, &lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3_once, _init_lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3);
v___x_1814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1814_, 0, v___x_1813_);
return v___x_1814_;
}
else
{
lean_object* v_val_1815_; lean_object* v___x_1817_; uint8_t v_isShared_1818_; uint8_t v_isSharedCheck_1831_; 
v_val_1815_ = lean_ctor_get(v_expectedType_x3f_1807_, 0);
v_isSharedCheck_1831_ = !lean_is_exclusive(v_expectedType_x3f_1807_);
if (v_isSharedCheck_1831_ == 0)
{
v___x_1817_ = v_expectedType_x3f_1807_;
v_isShared_1818_ = v_isSharedCheck_1831_;
goto v_resetjp_1816_;
}
else
{
lean_inc(v_val_1815_);
lean_dec(v_expectedType_x3f_1807_);
v___x_1817_ = lean_box(0);
v_isShared_1818_ = v_isSharedCheck_1831_;
goto v_resetjp_1816_;
}
v_resetjp_1816_:
{
uint8_t v___x_1819_; 
v___x_1819_ = l_Lean_Expr_hasExprMVar(v_val_1815_);
if (v___x_1819_ == 0)
{
lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; 
lean_del_object(v___x_1817_);
v___x_1820_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__5));
v___x_1821_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__9, &lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__9_once, _init_lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__9);
v___x_1822_ = lean_unsigned_to_nat(2u);
v___x_1823_ = lean_mk_empty_array_with_capacity(v___x_1822_);
v___x_1824_ = lean_array_push(v___x_1823_, v_val_1815_);
v___x_1825_ = lean_array_push(v___x_1824_, v___x_1821_);
v___x_1826_ = l_Lean_Meta_mkAppM(v___x_1820_, v___x_1825_, v_a_1808_, v_a_1809_, v_a_1810_, v_a_1811_);
return v___x_1826_;
}
else
{
lean_object* v___x_1827_; lean_object* v___x_1829_; 
lean_dec(v_val_1815_);
v___x_1827_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3, &lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3_once, _init_lp_LeanSearchClient_LeanSearchClient_defaultTerm___closed__3);
if (v_isShared_1818_ == 0)
{
lean_ctor_set_tag(v___x_1817_, 0);
lean_ctor_set(v___x_1817_, 0, v___x_1827_);
v___x_1829_ = v___x_1817_;
goto v_reusejp_1828_;
}
else
{
lean_object* v_reuseFailAlloc_1830_; 
v_reuseFailAlloc_1830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1830_, 0, v___x_1827_);
v___x_1829_ = v_reuseFailAlloc_1830_;
goto v_reusejp_1828_;
}
v_reusejp_1828_:
{
return v___x_1829_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm___boxed(lean_object* v_expectedType_x3f_1832_, lean_object* v_a_1833_, lean_object* v_a_1834_, lean_object* v_a_1835_, lean_object* v_a_1836_, lean_object* v_a_1837_){
_start:
{
lean_object* v_res_1838_; 
v_res_1838_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_1832_, v_a_1833_, v_a_1834_, v_a_1835_, v_a_1836_);
lean_dec(v_a_1836_);
lean_dec_ref(v_a_1835_);
lean_dec(v_a_1834_);
lean_dec_ref(v_a_1833_);
return v_res_1838_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___redArg(lean_object* v_a_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_){
_start:
{
lean_object* v___x_1847_; 
v___x_1847_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1839_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_, v___y_1845_);
return v___x_1847_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___redArg___boxed(lean_object* v_a_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_){
_start:
{
lean_object* v_res_1856_; 
v_res_1856_ = lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___redArg(v_a_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_);
lean_dec(v___y_1854_);
lean_dec_ref(v___y_1853_);
lean_dec(v___y_1852_);
lean_dec_ref(v___y_1851_);
lean_dec(v___y_1850_);
lean_dec_ref(v___y_1849_);
return v_res_1856_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0(lean_object* v_00_u03b1_1857_, lean_object* v_a_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_){
_start:
{
lean_object* v___x_1866_; 
v___x_1866_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1858_, v___y_1859_, v___y_1860_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_);
return v___x_1866_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0___boxed(lean_object* v_00_u03b1_1867_, lean_object* v_a_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_){
_start:
{
lean_object* v_res_1876_; 
v_res_1876_ = lp_LeanSearchClient_Lean_Elab_Term_withoutErrToSorry___at___00LeanSearchClient_checkTactic_spec__0(v_00_u03b1_1867_, v_a_1868_, v___y_1869_, v___y_1870_, v___y_1871_, v___y_1872_, v___y_1873_, v___y_1874_);
lean_dec(v___y_1874_);
lean_dec_ref(v___y_1873_);
lean_dec(v___y_1872_);
lean_dec_ref(v___y_1871_);
lean_dec(v___y_1870_);
lean_dec_ref(v___y_1869_);
return v_res_1876_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0(lean_object* v_a_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_, lean_object* v___y_1882_, lean_object* v___y_1883_, lean_object* v_a_x3f_1884_){
_start:
{
uint8_t v___x_1886_; lean_object* v___x_1887_; 
v___x_1886_ = 0;
v___x_1887_ = l_Lean_Elab_Term_SavedState_restore(v_a_1877_, v___x_1886_, v___y_1878_, v___y_1879_, v___y_1880_, v___y_1881_, v___y_1882_, v___y_1883_);
return v___x_1887_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0___boxed(lean_object* v_a_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v_a_x3f_1895_, lean_object* v___y_1896_){
_start:
{
lean_object* v_res_1897_; 
v_res_1897_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0(v_a_1888_, v___y_1889_, v___y_1890_, v___y_1891_, v___y_1892_, v___y_1893_, v___y_1894_, v_a_x3f_1895_);
lean_dec(v_a_x3f_1895_);
lean_dec(v___y_1894_);
lean_dec_ref(v___y_1893_);
lean_dec(v___y_1892_);
lean_dec_ref(v___y_1891_);
lean_dec(v___y_1890_);
lean_dec_ref(v___y_1889_);
return v_res_1897_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg(lean_object* v_x_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_){
_start:
{
lean_object* v___x_1906_; 
v___x_1906_ = l_Lean_Elab_Term_saveState___redArg(v___y_1900_, v___y_1902_, v___y_1904_);
if (lean_obj_tag(v___x_1906_) == 0)
{
lean_object* v_a_1907_; lean_object* v_r_1908_; 
v_a_1907_ = lean_ctor_get(v___x_1906_, 0);
lean_inc(v_a_1907_);
lean_dec_ref_known(v___x_1906_, 1);
lean_inc(v___y_1904_);
lean_inc_ref(v___y_1903_);
lean_inc(v___y_1902_);
lean_inc_ref(v___y_1901_);
lean_inc(v___y_1900_);
lean_inc_ref(v___y_1899_);
v_r_1908_ = lean_apply_7(v_x_1898_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, lean_box(0));
if (lean_obj_tag(v_r_1908_) == 0)
{
lean_object* v_a_1909_; lean_object* v___x_1911_; uint8_t v_isShared_1912_; uint8_t v_isSharedCheck_1933_; 
v_a_1909_ = lean_ctor_get(v_r_1908_, 0);
v_isSharedCheck_1933_ = !lean_is_exclusive(v_r_1908_);
if (v_isSharedCheck_1933_ == 0)
{
v___x_1911_ = v_r_1908_;
v_isShared_1912_ = v_isSharedCheck_1933_;
goto v_resetjp_1910_;
}
else
{
lean_inc(v_a_1909_);
lean_dec(v_r_1908_);
v___x_1911_ = lean_box(0);
v_isShared_1912_ = v_isSharedCheck_1933_;
goto v_resetjp_1910_;
}
v_resetjp_1910_:
{
lean_object* v___x_1914_; 
lean_inc(v_a_1909_);
if (v_isShared_1912_ == 0)
{
lean_ctor_set_tag(v___x_1911_, 1);
v___x_1914_ = v___x_1911_;
goto v_reusejp_1913_;
}
else
{
lean_object* v_reuseFailAlloc_1932_; 
v_reuseFailAlloc_1932_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1932_, 0, v_a_1909_);
v___x_1914_ = v_reuseFailAlloc_1932_;
goto v_reusejp_1913_;
}
v_reusejp_1913_:
{
lean_object* v___x_1915_; 
v___x_1915_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0(v_a_1907_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___x_1914_);
lean_dec_ref(v___x_1914_);
if (lean_obj_tag(v___x_1915_) == 0)
{
lean_object* v___x_1917_; uint8_t v_isShared_1918_; uint8_t v_isSharedCheck_1922_; 
v_isSharedCheck_1922_ = !lean_is_exclusive(v___x_1915_);
if (v_isSharedCheck_1922_ == 0)
{
lean_object* v_unused_1923_; 
v_unused_1923_ = lean_ctor_get(v___x_1915_, 0);
lean_dec(v_unused_1923_);
v___x_1917_ = v___x_1915_;
v_isShared_1918_ = v_isSharedCheck_1922_;
goto v_resetjp_1916_;
}
else
{
lean_dec(v___x_1915_);
v___x_1917_ = lean_box(0);
v_isShared_1918_ = v_isSharedCheck_1922_;
goto v_resetjp_1916_;
}
v_resetjp_1916_:
{
lean_object* v___x_1920_; 
if (v_isShared_1918_ == 0)
{
lean_ctor_set(v___x_1917_, 0, v_a_1909_);
v___x_1920_ = v___x_1917_;
goto v_reusejp_1919_;
}
else
{
lean_object* v_reuseFailAlloc_1921_; 
v_reuseFailAlloc_1921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1921_, 0, v_a_1909_);
v___x_1920_ = v_reuseFailAlloc_1921_;
goto v_reusejp_1919_;
}
v_reusejp_1919_:
{
return v___x_1920_;
}
}
}
else
{
lean_object* v_a_1924_; lean_object* v___x_1926_; uint8_t v_isShared_1927_; uint8_t v_isSharedCheck_1931_; 
lean_dec(v_a_1909_);
v_a_1924_ = lean_ctor_get(v___x_1915_, 0);
v_isSharedCheck_1931_ = !lean_is_exclusive(v___x_1915_);
if (v_isSharedCheck_1931_ == 0)
{
v___x_1926_ = v___x_1915_;
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
else
{
lean_inc(v_a_1924_);
lean_dec(v___x_1915_);
v___x_1926_ = lean_box(0);
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
v_resetjp_1925_:
{
lean_object* v___x_1929_; 
if (v_isShared_1927_ == 0)
{
v___x_1929_ = v___x_1926_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_1930_; 
v_reuseFailAlloc_1930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1930_, 0, v_a_1924_);
v___x_1929_ = v_reuseFailAlloc_1930_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
return v___x_1929_;
}
}
}
}
}
}
else
{
lean_object* v_a_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; 
v_a_1934_ = lean_ctor_get(v_r_1908_, 0);
lean_inc(v_a_1934_);
lean_dec_ref_known(v_r_1908_, 1);
v___x_1935_ = lean_box(0);
v___x_1936_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___lam__0(v_a_1907_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___x_1935_);
if (lean_obj_tag(v___x_1936_) == 0)
{
lean_object* v___x_1938_; uint8_t v_isShared_1939_; uint8_t v_isSharedCheck_1943_; 
v_isSharedCheck_1943_ = !lean_is_exclusive(v___x_1936_);
if (v_isSharedCheck_1943_ == 0)
{
lean_object* v_unused_1944_; 
v_unused_1944_ = lean_ctor_get(v___x_1936_, 0);
lean_dec(v_unused_1944_);
v___x_1938_ = v___x_1936_;
v_isShared_1939_ = v_isSharedCheck_1943_;
goto v_resetjp_1937_;
}
else
{
lean_dec(v___x_1936_);
v___x_1938_ = lean_box(0);
v_isShared_1939_ = v_isSharedCheck_1943_;
goto v_resetjp_1937_;
}
v_resetjp_1937_:
{
lean_object* v___x_1941_; 
if (v_isShared_1939_ == 0)
{
lean_ctor_set_tag(v___x_1938_, 1);
lean_ctor_set(v___x_1938_, 0, v_a_1934_);
v___x_1941_ = v___x_1938_;
goto v_reusejp_1940_;
}
else
{
lean_object* v_reuseFailAlloc_1942_; 
v_reuseFailAlloc_1942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1942_, 0, v_a_1934_);
v___x_1941_ = v_reuseFailAlloc_1942_;
goto v_reusejp_1940_;
}
v_reusejp_1940_:
{
return v___x_1941_;
}
}
}
else
{
lean_object* v_a_1945_; lean_object* v___x_1947_; uint8_t v_isShared_1948_; uint8_t v_isSharedCheck_1952_; 
lean_dec(v_a_1934_);
v_a_1945_ = lean_ctor_get(v___x_1936_, 0);
v_isSharedCheck_1952_ = !lean_is_exclusive(v___x_1936_);
if (v_isSharedCheck_1952_ == 0)
{
v___x_1947_ = v___x_1936_;
v_isShared_1948_ = v_isSharedCheck_1952_;
goto v_resetjp_1946_;
}
else
{
lean_inc(v_a_1945_);
lean_dec(v___x_1936_);
v___x_1947_ = lean_box(0);
v_isShared_1948_ = v_isSharedCheck_1952_;
goto v_resetjp_1946_;
}
v_resetjp_1946_:
{
lean_object* v___x_1950_; 
if (v_isShared_1948_ == 0)
{
v___x_1950_ = v___x_1947_;
goto v_reusejp_1949_;
}
else
{
lean_object* v_reuseFailAlloc_1951_; 
v_reuseFailAlloc_1951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1951_, 0, v_a_1945_);
v___x_1950_ = v_reuseFailAlloc_1951_;
goto v_reusejp_1949_;
}
v_reusejp_1949_:
{
return v___x_1950_;
}
}
}
}
}
else
{
lean_object* v_a_1953_; lean_object* v___x_1955_; uint8_t v_isShared_1956_; uint8_t v_isSharedCheck_1960_; 
lean_dec_ref(v_x_1898_);
v_a_1953_ = lean_ctor_get(v___x_1906_, 0);
v_isSharedCheck_1960_ = !lean_is_exclusive(v___x_1906_);
if (v_isSharedCheck_1960_ == 0)
{
v___x_1955_ = v___x_1906_;
v_isShared_1956_ = v_isSharedCheck_1960_;
goto v_resetjp_1954_;
}
else
{
lean_inc(v_a_1953_);
lean_dec(v___x_1906_);
v___x_1955_ = lean_box(0);
v_isShared_1956_ = v_isSharedCheck_1960_;
goto v_resetjp_1954_;
}
v_resetjp_1954_:
{
lean_object* v___x_1958_; 
if (v_isShared_1956_ == 0)
{
v___x_1958_ = v___x_1955_;
goto v_reusejp_1957_;
}
else
{
lean_object* v_reuseFailAlloc_1959_; 
v_reuseFailAlloc_1959_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1959_, 0, v_a_1953_);
v___x_1958_ = v_reuseFailAlloc_1959_;
goto v_reusejp_1957_;
}
v_reusejp_1957_:
{
return v___x_1958_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg___boxed(lean_object* v_x_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_){
_start:
{
lean_object* v_res_1969_; 
v_res_1969_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg(v_x_1961_, v___y_1962_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_);
lean_dec(v___y_1967_);
lean_dec_ref(v___y_1966_);
lean_dec(v___y_1965_);
lean_dec_ref(v___y_1964_);
lean_dec(v___y_1963_);
lean_dec_ref(v___y_1962_);
return v_res_1969_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1(lean_object* v_00_u03b1_1970_, lean_object* v_x_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_){
_start:
{
lean_object* v___x_1979_; 
v___x_1979_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg(v_x_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_, v___y_1977_);
return v___x_1979_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___boxed(lean_object* v_00_u03b1_1980_, lean_object* v_x_1981_, lean_object* v___y_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_){
_start:
{
lean_object* v_res_1989_; 
v_res_1989_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1(v_00_u03b1_1980_, v_x_1981_, v___y_1982_, v___y_1983_, v___y_1984_, v___y_1985_, v___y_1986_, v___y_1987_);
lean_dec(v___y_1987_);
lean_dec_ref(v___y_1986_);
lean_dec(v___y_1985_);
lean_dec_ref(v___y_1984_);
lean_dec(v___y_1983_);
lean_dec_ref(v___y_1982_);
return v_res_1989_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__0(lean_object* v_a_1990_, lean_object* v_tac_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_){
_start:
{
lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; 
v___x_1999_ = lean_st_ref_get(v___y_1993_);
v___x_2000_ = l_Lean_Expr_mvarId_x21(v_a_1990_);
v___x_2001_ = l_Lean_Elab_runTactic(v___x_2000_, v_tac_1991_, v___y_1992_, v___x_1999_, v___y_1994_, v___y_1995_, v___y_1996_, v___y_1997_);
return v___x_2001_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__0___boxed(lean_object* v_a_2002_, lean_object* v_tac_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_){
_start:
{
lean_object* v_res_2011_; 
v_res_2011_ = lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__0(v_a_2002_, v_tac_2003_, v___y_2004_, v___y_2005_, v___y_2006_, v___y_2007_, v___y_2008_, v___y_2009_);
lean_dec(v___y_2009_);
lean_dec_ref(v___y_2008_);
lean_dec(v___y_2007_);
lean_dec_ref(v___y_2006_);
lean_dec(v___y_2005_);
lean_dec_ref(v_a_2002_);
return v_res_2011_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__1(lean_object* v___x_2012_, uint8_t v___x_2013_, lean_object* v___x_2014_, lean_object* v_tac_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_){
_start:
{
lean_object* v___y_2024_; uint8_t v___y_2025_; lean_object* v_a_2030_; lean_object* v___x_2033_; 
v___x_2033_ = l_Lean_Meta_mkFreshExprMVar(v___x_2012_, v___x_2013_, v___x_2014_, v___y_2018_, v___y_2019_, v___y_2020_, v___y_2021_);
if (lean_obj_tag(v___x_2033_) == 0)
{
lean_object* v_a_2034_; lean_object* v___x_2036_; uint8_t v_isShared_2037_; uint8_t v_isSharedCheck_2054_; 
v_a_2034_ = lean_ctor_get(v___x_2033_, 0);
v_isSharedCheck_2054_ = !lean_is_exclusive(v___x_2033_);
if (v_isSharedCheck_2054_ == 0)
{
v___x_2036_ = v___x_2033_;
v_isShared_2037_ = v_isSharedCheck_2054_;
goto v_resetjp_2035_;
}
else
{
lean_inc(v_a_2034_);
lean_dec(v___x_2033_);
v___x_2036_ = lean_box(0);
v_isShared_2037_ = v_isSharedCheck_2054_;
goto v_resetjp_2035_;
}
v_resetjp_2035_:
{
lean_object* v___f_2038_; lean_object* v___x_2039_; 
v___f_2038_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__0___boxed), 9, 2);
lean_closure_set(v___f_2038_, 0, v_a_2034_);
lean_closure_set(v___f_2038_, 1, v_tac_2015_);
v___x_2039_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___f_2038_, v___y_2016_, v___y_2017_, v___y_2018_, v___y_2019_, v___y_2020_, v___y_2021_);
if (lean_obj_tag(v___x_2039_) == 0)
{
lean_object* v_a_2040_; lean_object* v___x_2042_; uint8_t v_isShared_2043_; uint8_t v_isSharedCheck_2052_; 
v_a_2040_ = lean_ctor_get(v___x_2039_, 0);
v_isSharedCheck_2052_ = !lean_is_exclusive(v___x_2039_);
if (v_isSharedCheck_2052_ == 0)
{
v___x_2042_ = v___x_2039_;
v_isShared_2043_ = v_isSharedCheck_2052_;
goto v_resetjp_2041_;
}
else
{
lean_inc(v_a_2040_);
lean_dec(v___x_2039_);
v___x_2042_ = lean_box(0);
v_isShared_2043_ = v_isSharedCheck_2052_;
goto v_resetjp_2041_;
}
v_resetjp_2041_:
{
lean_object* v_fst_2044_; lean_object* v___x_2045_; lean_object* v___x_2047_; 
v_fst_2044_ = lean_ctor_get(v_a_2040_, 0);
lean_inc(v_fst_2044_);
lean_dec(v_a_2040_);
v___x_2045_ = l_List_lengthTR___redArg(v_fst_2044_);
lean_dec(v_fst_2044_);
if (v_isShared_2037_ == 0)
{
lean_ctor_set_tag(v___x_2036_, 1);
lean_ctor_set(v___x_2036_, 0, v___x_2045_);
v___x_2047_ = v___x_2036_;
goto v_reusejp_2046_;
}
else
{
lean_object* v_reuseFailAlloc_2051_; 
v_reuseFailAlloc_2051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2051_, 0, v___x_2045_);
v___x_2047_ = v_reuseFailAlloc_2051_;
goto v_reusejp_2046_;
}
v_reusejp_2046_:
{
lean_object* v___x_2049_; 
if (v_isShared_2043_ == 0)
{
lean_ctor_set(v___x_2042_, 0, v___x_2047_);
v___x_2049_ = v___x_2042_;
goto v_reusejp_2048_;
}
else
{
lean_object* v_reuseFailAlloc_2050_; 
v_reuseFailAlloc_2050_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2050_, 0, v___x_2047_);
v___x_2049_ = v_reuseFailAlloc_2050_;
goto v_reusejp_2048_;
}
v_reusejp_2048_:
{
return v___x_2049_;
}
}
}
}
else
{
lean_object* v_a_2053_; 
lean_del_object(v___x_2036_);
v_a_2053_ = lean_ctor_get(v___x_2039_, 0);
lean_inc(v_a_2053_);
lean_dec_ref_known(v___x_2039_, 1);
v_a_2030_ = v_a_2053_;
goto v___jp_2029_;
}
}
}
else
{
lean_object* v_a_2055_; 
lean_dec(v_tac_2015_);
v_a_2055_ = lean_ctor_get(v___x_2033_, 0);
lean_inc(v_a_2055_);
lean_dec_ref_known(v___x_2033_, 1);
v_a_2030_ = v_a_2055_;
goto v___jp_2029_;
}
v___jp_2023_:
{
if (v___y_2025_ == 0)
{
lean_object* v___x_2026_; lean_object* v___x_2027_; 
lean_dec_ref(v___y_2024_);
v___x_2026_ = lean_box(0);
v___x_2027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2027_, 0, v___x_2026_);
return v___x_2027_;
}
else
{
lean_object* v___x_2028_; 
v___x_2028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2028_, 0, v___y_2024_);
return v___x_2028_;
}
}
v___jp_2029_:
{
uint8_t v___x_2031_; 
v___x_2031_ = l_Lean_Exception_isInterrupt(v_a_2030_);
if (v___x_2031_ == 0)
{
uint8_t v___x_2032_; 
lean_inc_ref(v_a_2030_);
v___x_2032_ = l_Lean_Exception_isRuntime(v_a_2030_);
v___y_2024_ = v_a_2030_;
v___y_2025_ = v___x_2032_;
goto v___jp_2023_;
}
else
{
v___y_2024_ = v_a_2030_;
v___y_2025_ = v___x_2031_;
goto v___jp_2023_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__1___boxed(lean_object* v___x_2056_, lean_object* v___x_2057_, lean_object* v___x_2058_, lean_object* v_tac_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_){
_start:
{
uint8_t v___x_3604__boxed_2067_; lean_object* v_res_2068_; 
v___x_3604__boxed_2067_ = lean_unbox(v___x_2057_);
v_res_2068_ = lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__1(v___x_2056_, v___x_3604__boxed_2067_, v___x_2058_, v_tac_2059_, v___y_2060_, v___y_2061_, v___y_2062_, v___y_2063_, v___y_2064_, v___y_2065_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
lean_dec(v___y_2063_);
lean_dec_ref(v___y_2062_);
lean_dec(v___y_2061_);
lean_dec_ref(v___y_2060_);
return v_res_2068_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic(lean_object* v_target_2069_, lean_object* v_tac_2070_, lean_object* v_a_2071_, lean_object* v_a_2072_, lean_object* v_a_2073_, lean_object* v_a_2074_, lean_object* v_a_2075_, lean_object* v_a_2076_){
_start:
{
lean_object* v___x_2078_; uint8_t v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___f_2082_; lean_object* v___x_2083_; 
v___x_2078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2078_, 0, v_target_2069_);
v___x_2079_ = 0;
v___x_2080_ = lean_box(0);
v___x_2081_ = lean_box(v___x_2079_);
v___f_2082_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_checkTactic___lam__1___boxed), 11, 4);
lean_closure_set(v___f_2082_, 0, v___x_2078_);
lean_closure_set(v___f_2082_, 1, v___x_2081_);
lean_closure_set(v___f_2082_, 2, v___x_2080_);
lean_closure_set(v___f_2082_, 3, v_tac_2070_);
v___x_2083_ = lp_LeanSearchClient_Lean_withoutModifyingState___at___00LeanSearchClient_checkTactic_spec__1___redArg(v___f_2082_, v_a_2071_, v_a_2072_, v_a_2073_, v_a_2074_, v_a_2075_, v_a_2076_);
return v___x_2083_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic___boxed(lean_object* v_target_2084_, lean_object* v_tac_2085_, lean_object* v_a_2086_, lean_object* v_a_2087_, lean_object* v_a_2088_, lean_object* v_a_2089_, lean_object* v_a_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_){
_start:
{
lean_object* v_res_2093_; 
v_res_2093_ = lp_LeanSearchClient_LeanSearchClient_checkTactic(v_target_2084_, v_tac_2085_, v_a_2086_, v_a_2087_, v_a_2088_, v_a_2089_, v_a_2090_, v_a_2091_);
lean_dec(v_a_2091_);
lean_dec_ref(v_a_2090_);
lean_dec(v_a_2089_);
lean_dec_ref(v_a_2088_);
lean_dec(v_a_2087_);
lean_dec_ref(v_a_2086_);
return v_res_2093_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0(lean_object* v_opts_2094_, lean_object* v_opt_2095_){
_start:
{
lean_object* v_name_2096_; lean_object* v_defValue_2097_; lean_object* v_map_2098_; lean_object* v___x_2099_; 
v_name_2096_ = lean_ctor_get(v_opt_2095_, 0);
v_defValue_2097_ = lean_ctor_get(v_opt_2095_, 1);
v_map_2098_ = lean_ctor_get(v_opts_2094_, 0);
v___x_2099_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2098_, v_name_2096_);
if (lean_obj_tag(v___x_2099_) == 0)
{
lean_inc(v_defValue_2097_);
return v_defValue_2097_;
}
else
{
lean_object* v_val_2100_; 
v_val_2100_ = lean_ctor_get(v___x_2099_, 0);
lean_inc(v_val_2100_);
lean_dec_ref_known(v___x_2099_, 1);
if (lean_obj_tag(v_val_2100_) == 3)
{
lean_object* v_v_2101_; 
v_v_2101_ = lean_ctor_get(v_val_2100_, 0);
lean_inc(v_v_2101_);
lean_dec_ref_known(v_val_2100_, 1);
return v_v_2101_;
}
else
{
lean_dec(v_val_2100_);
lean_inc(v_defValue_2097_);
return v_defValue_2097_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0___boxed(lean_object* v_opts_2102_, lean_object* v_opt_2103_){
_start:
{
lean_object* v_res_2104_; 
v_res_2104_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0(v_opts_2102_, v_opt_2103_);
lean_dec_ref(v_opt_2103_);
lean_dec_ref(v_opts_2102_);
return v_res_2104_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___lam__0(lean_object* v___y_2105_, lean_object* v___y_2106_){
_start:
{
lean_object* v_options_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; 
v_options_2108_ = lean_ctor_get(v___y_2105_, 2);
v___x_2109_ = lp_LeanSearchClient_leansearch_queries;
v___x_2110_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0(v_options_2108_, v___x_2109_);
v___x_2111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2111_, 0, v___x_2110_);
return v___x_2111_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchServer___lam__0___boxed(lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_){
_start:
{
lean_object* v_res_2115_; 
v_res_2115_ = lp_LeanSearchClient_LeanSearchClient_leanSearchServer___lam__0(v___y_2112_, v___y_2113_);
lean_dec(v___y_2113_);
lean_dec_ref(v___y_2112_);
return v_res_2115_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0(size_t v_sz_2129_, size_t v_i_2130_, lean_object* v_bs_2131_){
_start:
{
uint8_t v___x_2132_; 
v___x_2132_ = lean_usize_dec_lt(v_i_2130_, v_sz_2129_);
if (v___x_2132_ == 0)
{
return v_bs_2131_;
}
else
{
lean_object* v_v_2133_; lean_object* v___x_2134_; lean_object* v_bs_x27_2135_; lean_object* v___x_2136_; size_t v___x_2137_; size_t v___x_2138_; lean_object* v___x_2139_; 
v_v_2133_ = lean_array_uget(v_bs_2131_, v_i_2130_);
v___x_2134_ = lean_unsigned_to_nat(0u);
v_bs_x27_2135_ = lean_array_uset(v_bs_2131_, v_i_2130_, v___x_2134_);
v___x_2136_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion(v_v_2133_);
v___x_2137_ = ((size_t)1ULL);
v___x_2138_ = lean_usize_add(v_i_2130_, v___x_2137_);
v___x_2139_ = lean_array_uset(v_bs_x27_2135_, v_i_2130_, v___x_2136_);
v_i_2130_ = v___x_2138_;
v_bs_2131_ = v___x_2139_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0___boxed(lean_object* v_sz_2141_, lean_object* v_i_2142_, lean_object* v_bs_2143_){
_start:
{
size_t v_sz_boxed_2144_; size_t v_i_boxed_2145_; lean_object* v_res_2146_; 
v_sz_boxed_2144_ = lean_unbox_usize(v_sz_2141_);
lean_dec(v_sz_2141_);
v_i_boxed_2145_ = lean_unbox_usize(v_i_2142_);
lean_dec(v_i_2142_);
v_res_2146_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0(v_sz_boxed_2144_, v_i_boxed_2145_, v_bs_2143_);
return v_res_2146_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getCommandSuggestions(lean_object* v_ss_2147_, lean_object* v_s_2148_, lean_object* v_num__results_2149_, lean_object* v_a_2150_, lean_object* v_a_2151_, lean_object* v_a_2152_, lean_object* v_a_2153_){
_start:
{
lean_object* v_query_2155_; lean_object* v___x_2156_; 
v_query_2155_ = lean_ctor_get(v_ss_2147_, 3);
lean_inc_ref(v_query_2155_);
lean_dec_ref(v_ss_2147_);
lean_inc(v_a_2153_);
lean_inc_ref(v_a_2152_);
lean_inc(v_a_2151_);
lean_inc_ref(v_a_2150_);
v___x_2156_ = lean_apply_7(v_query_2155_, v_s_2148_, v_num__results_2149_, v_a_2150_, v_a_2151_, v_a_2152_, v_a_2153_, lean_box(0));
if (lean_obj_tag(v___x_2156_) == 0)
{
lean_object* v_a_2157_; lean_object* v___x_2159_; uint8_t v_isShared_2160_; uint8_t v_isSharedCheck_2167_; 
v_a_2157_ = lean_ctor_get(v___x_2156_, 0);
v_isSharedCheck_2167_ = !lean_is_exclusive(v___x_2156_);
if (v_isSharedCheck_2167_ == 0)
{
v___x_2159_ = v___x_2156_;
v_isShared_2160_ = v_isSharedCheck_2167_;
goto v_resetjp_2158_;
}
else
{
lean_inc(v_a_2157_);
lean_dec(v___x_2156_);
v___x_2159_ = lean_box(0);
v_isShared_2160_ = v_isSharedCheck_2167_;
goto v_resetjp_2158_;
}
v_resetjp_2158_:
{
size_t v_sz_2161_; size_t v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2165_; 
v_sz_2161_ = lean_array_size(v_a_2157_);
v___x_2162_ = ((size_t)0ULL);
v___x_2163_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0(v_sz_2161_, v___x_2162_, v_a_2157_);
if (v_isShared_2160_ == 0)
{
lean_ctor_set(v___x_2159_, 0, v___x_2163_);
v___x_2165_ = v___x_2159_;
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
else
{
lean_object* v_a_2168_; lean_object* v___x_2170_; uint8_t v_isShared_2171_; uint8_t v_isSharedCheck_2175_; 
v_a_2168_ = lean_ctor_get(v___x_2156_, 0);
v_isSharedCheck_2175_ = !lean_is_exclusive(v___x_2156_);
if (v_isSharedCheck_2175_ == 0)
{
v___x_2170_ = v___x_2156_;
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
else
{
lean_inc(v_a_2168_);
lean_dec(v___x_2156_);
v___x_2170_ = lean_box(0);
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
v_resetjp_2169_:
{
lean_object* v___x_2173_; 
if (v_isShared_2171_ == 0)
{
v___x_2173_ = v___x_2170_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v_a_2168_);
v___x_2173_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
return v___x_2173_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getCommandSuggestions___boxed(lean_object* v_ss_2176_, lean_object* v_s_2177_, lean_object* v_num__results_2178_, lean_object* v_a_2179_, lean_object* v_a_2180_, lean_object* v_a_2181_, lean_object* v_a_2182_, lean_object* v_a_2183_){
_start:
{
lean_object* v_res_2184_; 
v_res_2184_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_getCommandSuggestions(v_ss_2176_, v_s_2177_, v_num__results_2178_, v_a_2179_, v_a_2180_, v_a_2181_, v_a_2182_);
lean_dec(v_a_2182_);
lean_dec_ref(v_a_2181_);
lean_dec(v_a_2180_);
lean_dec_ref(v_a_2179_);
return v_res_2184_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTermSuggestions_spec__0(size_t v_sz_2185_, size_t v_i_2186_, lean_object* v_bs_2187_){
_start:
{
uint8_t v___x_2188_; 
v___x_2188_ = lean_usize_dec_lt(v_i_2186_, v_sz_2185_);
if (v___x_2188_ == 0)
{
return v_bs_2187_;
}
else
{
lean_object* v_v_2189_; lean_object* v___x_2190_; lean_object* v_bs_x27_2191_; lean_object* v___x_2192_; size_t v___x_2193_; size_t v___x_2194_; lean_object* v___x_2195_; 
v_v_2189_ = lean_array_uget(v_bs_2187_, v_i_2186_);
v___x_2190_ = lean_unsigned_to_nat(0u);
v_bs_x27_2191_ = lean_array_uset(v_bs_2187_, v_i_2186_, v___x_2190_);
v___x_2192_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion(v_v_2189_);
v___x_2193_ = ((size_t)1ULL);
v___x_2194_ = lean_usize_add(v_i_2186_, v___x_2193_);
v___x_2195_ = lean_array_uset(v_bs_x27_2191_, v_i_2186_, v___x_2192_);
v_i_2186_ = v___x_2194_;
v_bs_2187_ = v___x_2195_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTermSuggestions_spec__0___boxed(lean_object* v_sz_2197_, lean_object* v_i_2198_, lean_object* v_bs_2199_){
_start:
{
size_t v_sz_boxed_2200_; size_t v_i_boxed_2201_; lean_object* v_res_2202_; 
v_sz_boxed_2200_ = lean_unbox_usize(v_sz_2197_);
lean_dec(v_sz_2197_);
v_i_boxed_2201_ = lean_unbox_usize(v_i_2198_);
lean_dec(v_i_2198_);
v_res_2202_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTermSuggestions_spec__0(v_sz_boxed_2200_, v_i_boxed_2201_, v_bs_2199_);
return v_res_2202_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTermSuggestions(lean_object* v_ss_2203_, lean_object* v_s_2204_, lean_object* v_num__results_2205_, lean_object* v_a_2206_, lean_object* v_a_2207_, lean_object* v_a_2208_, lean_object* v_a_2209_){
_start:
{
lean_object* v_query_2211_; lean_object* v___x_2212_; 
v_query_2211_ = lean_ctor_get(v_ss_2203_, 3);
lean_inc_ref(v_query_2211_);
lean_dec_ref(v_ss_2203_);
lean_inc(v_a_2209_);
lean_inc_ref(v_a_2208_);
lean_inc(v_a_2207_);
lean_inc_ref(v_a_2206_);
v___x_2212_ = lean_apply_7(v_query_2211_, v_s_2204_, v_num__results_2205_, v_a_2206_, v_a_2207_, v_a_2208_, v_a_2209_, lean_box(0));
if (lean_obj_tag(v___x_2212_) == 0)
{
lean_object* v_a_2213_; lean_object* v___x_2215_; uint8_t v_isShared_2216_; uint8_t v_isSharedCheck_2223_; 
v_a_2213_ = lean_ctor_get(v___x_2212_, 0);
v_isSharedCheck_2223_ = !lean_is_exclusive(v___x_2212_);
if (v_isSharedCheck_2223_ == 0)
{
v___x_2215_ = v___x_2212_;
v_isShared_2216_ = v_isSharedCheck_2223_;
goto v_resetjp_2214_;
}
else
{
lean_inc(v_a_2213_);
lean_dec(v___x_2212_);
v___x_2215_ = lean_box(0);
v_isShared_2216_ = v_isSharedCheck_2223_;
goto v_resetjp_2214_;
}
v_resetjp_2214_:
{
size_t v_sz_2217_; size_t v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2221_; 
v_sz_2217_ = lean_array_size(v_a_2213_);
v___x_2218_ = ((size_t)0ULL);
v___x_2219_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTermSuggestions_spec__0(v_sz_2217_, v___x_2218_, v_a_2213_);
if (v_isShared_2216_ == 0)
{
lean_ctor_set(v___x_2215_, 0, v___x_2219_);
v___x_2221_ = v___x_2215_;
goto v_reusejp_2220_;
}
else
{
lean_object* v_reuseFailAlloc_2222_; 
v_reuseFailAlloc_2222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2222_, 0, v___x_2219_);
v___x_2221_ = v_reuseFailAlloc_2222_;
goto v_reusejp_2220_;
}
v_reusejp_2220_:
{
return v___x_2221_;
}
}
}
else
{
lean_object* v_a_2224_; lean_object* v___x_2226_; uint8_t v_isShared_2227_; uint8_t v_isSharedCheck_2231_; 
v_a_2224_ = lean_ctor_get(v___x_2212_, 0);
v_isSharedCheck_2231_ = !lean_is_exclusive(v___x_2212_);
if (v_isSharedCheck_2231_ == 0)
{
v___x_2226_ = v___x_2212_;
v_isShared_2227_ = v_isSharedCheck_2231_;
goto v_resetjp_2225_;
}
else
{
lean_inc(v_a_2224_);
lean_dec(v___x_2212_);
v___x_2226_ = lean_box(0);
v_isShared_2227_ = v_isSharedCheck_2231_;
goto v_resetjp_2225_;
}
v_resetjp_2225_:
{
lean_object* v___x_2229_; 
if (v_isShared_2227_ == 0)
{
v___x_2229_ = v___x_2226_;
goto v_reusejp_2228_;
}
else
{
lean_object* v_reuseFailAlloc_2230_; 
v_reuseFailAlloc_2230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2230_, 0, v_a_2224_);
v___x_2229_ = v_reuseFailAlloc_2230_;
goto v_reusejp_2228_;
}
v_reusejp_2228_:
{
return v___x_2229_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTermSuggestions___boxed(lean_object* v_ss_2232_, lean_object* v_s_2233_, lean_object* v_num__results_2234_, lean_object* v_a_2235_, lean_object* v_a_2236_, lean_object* v_a_2237_, lean_object* v_a_2238_, lean_object* v_a_2239_){
_start:
{
lean_object* v_res_2240_; 
v_res_2240_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_getTermSuggestions(v_ss_2232_, v_s_2233_, v_num__results_2234_, v_a_2235_, v_a_2236_, v_a_2237_, v_a_2238_);
lean_dec(v_a_2238_);
lean_dec_ref(v_a_2237_);
lean_dec(v_a_2236_);
lean_dec_ref(v_a_2235_);
return v_res_2240_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0(size_t v_sz_2241_, size_t v_i_2242_, lean_object* v_bs_2243_){
_start:
{
uint8_t v___x_2244_; 
v___x_2244_ = lean_usize_dec_lt(v_i_2242_, v_sz_2241_);
if (v___x_2244_ == 0)
{
return v_bs_2243_;
}
else
{
lean_object* v_v_2245_; lean_object* v_name_2246_; lean_object* v_type_x3f_2247_; lean_object* v___x_2248_; lean_object* v_bs_x27_2249_; lean_object* v___y_2251_; 
v_v_2245_ = lean_array_uget(v_bs_2243_, v_i_2242_);
v_name_2246_ = lean_ctor_get(v_v_2245_, 0);
v_type_x3f_2247_ = lean_ctor_get(v_v_2245_, 1);
v___x_2248_ = lean_unsigned_to_nat(0u);
v_bs_x27_2249_ = lean_array_uset(v_bs_2243_, v_i_2242_, v___x_2248_);
if (lean_obj_tag(v_type_x3f_2247_) == 0)
{
lean_inc_ref(v_name_2246_);
v___y_2251_ = v_name_2246_;
goto v___jp_2250_;
}
else
{
lean_object* v_val_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; 
v_val_2258_ = lean_ctor_get(v_type_x3f_2247_, 0);
v___x_2259_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__0));
lean_inc_ref(v_name_2246_);
v___x_2260_ = lean_string_append(v_name_2246_, v___x_2259_);
v___x_2261_ = lean_string_append(v___x_2260_, v_val_2258_);
v___x_2262_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion___closed__1));
v___x_2263_ = lean_string_append(v___x_2261_, v___x_2262_);
v___y_2251_ = v___x_2263_;
goto v___jp_2250_;
}
v___jp_2250_:
{
lean_object* v___x_2252_; lean_object* v___x_2253_; size_t v___x_2254_; size_t v___x_2255_; lean_object* v___x_2256_; 
v___x_2252_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions(v_v_2245_);
v___x_2253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2253_, 0, v___y_2251_);
lean_ctor_set(v___x_2253_, 1, v___x_2252_);
v___x_2254_ = ((size_t)1ULL);
v___x_2255_ = lean_usize_add(v_i_2242_, v___x_2254_);
v___x_2256_ = lean_array_uset(v_bs_x27_2249_, v_i_2242_, v___x_2253_);
v_i_2242_ = v___x_2255_;
v_bs_2243_ = v___x_2256_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0___boxed(lean_object* v_sz_2264_, lean_object* v_i_2265_, lean_object* v_bs_2266_){
_start:
{
size_t v_sz_boxed_2267_; size_t v_i_boxed_2268_; lean_object* v_res_2269_; 
v_sz_boxed_2267_ = lean_unbox_usize(v_sz_2264_);
lean_dec(v_sz_2264_);
v_i_boxed_2268_ = lean_unbox_usize(v_i_2265_);
lean_dec(v_i_2265_);
v_res_2269_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0(v_sz_boxed_2267_, v_i_boxed_2268_, v_bs_2266_);
return v_res_2269_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTacticSuggestionGroups(lean_object* v_ss_2270_, lean_object* v_s_2271_, lean_object* v_num__results_2272_, lean_object* v_a_2273_, lean_object* v_a_2274_, lean_object* v_a_2275_, lean_object* v_a_2276_){
_start:
{
lean_object* v_query_2278_; lean_object* v___x_2279_; 
v_query_2278_ = lean_ctor_get(v_ss_2270_, 3);
lean_inc_ref(v_query_2278_);
lean_dec_ref(v_ss_2270_);
lean_inc(v_a_2276_);
lean_inc_ref(v_a_2275_);
lean_inc(v_a_2274_);
lean_inc_ref(v_a_2273_);
v___x_2279_ = lean_apply_7(v_query_2278_, v_s_2271_, v_num__results_2272_, v_a_2273_, v_a_2274_, v_a_2275_, v_a_2276_, lean_box(0));
if (lean_obj_tag(v___x_2279_) == 0)
{
lean_object* v_a_2280_; lean_object* v___x_2282_; uint8_t v_isShared_2283_; uint8_t v_isSharedCheck_2290_; 
v_a_2280_ = lean_ctor_get(v___x_2279_, 0);
v_isSharedCheck_2290_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2290_ == 0)
{
v___x_2282_ = v___x_2279_;
v_isShared_2283_ = v_isSharedCheck_2290_;
goto v_resetjp_2281_;
}
else
{
lean_inc(v_a_2280_);
lean_dec(v___x_2279_);
v___x_2282_ = lean_box(0);
v_isShared_2283_ = v_isSharedCheck_2290_;
goto v_resetjp_2281_;
}
v_resetjp_2281_:
{
size_t v_sz_2284_; size_t v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2288_; 
v_sz_2284_ = lean_array_size(v_a_2280_);
v___x_2285_ = ((size_t)0ULL);
v___x_2286_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0(v_sz_2284_, v___x_2285_, v_a_2280_);
if (v_isShared_2283_ == 0)
{
lean_ctor_set(v___x_2282_, 0, v___x_2286_);
v___x_2288_ = v___x_2282_;
goto v_reusejp_2287_;
}
else
{
lean_object* v_reuseFailAlloc_2289_; 
v_reuseFailAlloc_2289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2289_, 0, v___x_2286_);
v___x_2288_ = v_reuseFailAlloc_2289_;
goto v_reusejp_2287_;
}
v_reusejp_2287_:
{
return v___x_2288_;
}
}
}
else
{
lean_object* v_a_2291_; lean_object* v___x_2293_; uint8_t v_isShared_2294_; uint8_t v_isSharedCheck_2298_; 
v_a_2291_ = lean_ctor_get(v___x_2279_, 0);
v_isSharedCheck_2298_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2298_ == 0)
{
v___x_2293_ = v___x_2279_;
v_isShared_2294_ = v_isSharedCheck_2298_;
goto v_resetjp_2292_;
}
else
{
lean_inc(v_a_2291_);
lean_dec(v___x_2279_);
v___x_2293_ = lean_box(0);
v_isShared_2294_ = v_isSharedCheck_2298_;
goto v_resetjp_2292_;
}
v_resetjp_2292_:
{
lean_object* v___x_2296_; 
if (v_isShared_2294_ == 0)
{
v___x_2296_ = v___x_2293_;
goto v_reusejp_2295_;
}
else
{
lean_object* v_reuseFailAlloc_2297_; 
v_reuseFailAlloc_2297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2297_, 0, v_a_2291_);
v___x_2296_ = v_reuseFailAlloc_2297_;
goto v_reusejp_2295_;
}
v_reusejp_2295_:
{
return v___x_2296_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_getTacticSuggestionGroups___boxed(lean_object* v_ss_2299_, lean_object* v_s_2300_, lean_object* v_num__results_2301_, lean_object* v_a_2302_, lean_object* v_a_2303_, lean_object* v_a_2304_, lean_object* v_a_2305_, lean_object* v_a_2306_){
_start:
{
lean_object* v_res_2307_; 
v_res_2307_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_getTacticSuggestionGroups(v_ss_2299_, v_s_2300_, v_num__results_2301_, v_a_2302_, v_a_2303_, v_a_2304_, v_a_2305_);
lean_dec(v_a_2305_);
lean_dec_ref(v_a_2304_);
lean_dec(v_a_2303_);
lean_dec_ref(v_a_2302_);
return v_res_2307_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(lean_object* v_ss_2309_){
_start:
{
lean_object* v_url_2310_; lean_object* v_cmd_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; 
v_url_2310_ = lean_ctor_get(v_ss_2309_, 1);
lean_inc_ref(v_url_2310_);
v_cmd_2311_ = lean_ctor_get(v_ss_2309_, 2);
lean_inc_ref(v_cmd_2311_);
lean_dec_ref(v_ss_2309_);
v___x_2312_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery___closed__0));
v___x_2313_ = lean_string_append(v_cmd_2311_, v___x_2312_);
v___x_2314_ = lean_string_append(v___x_2313_, v_url_2310_);
lean_dec_ref(v_url_2310_);
v___x_2315_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1));
v___x_2316_ = lean_string_append(v___x_2314_, v___x_2315_);
return v___x_2316_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(lean_object* v_opts_2317_, lean_object* v_opt_2318_){
_start:
{
lean_object* v_name_2319_; lean_object* v_defValue_2320_; lean_object* v_map_2321_; lean_object* v___x_2322_; 
v_name_2319_ = lean_ctor_get(v_opt_2318_, 0);
v_defValue_2320_ = lean_ctor_get(v_opt_2318_, 1);
v_map_2321_ = lean_ctor_get(v_opts_2317_, 0);
v___x_2322_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2321_, v_name_2319_);
if (lean_obj_tag(v___x_2322_) == 0)
{
uint8_t v___x_2323_; 
v___x_2323_ = lean_unbox(v_defValue_2320_);
return v___x_2323_;
}
else
{
lean_object* v_val_2324_; 
v_val_2324_ = lean_ctor_get(v___x_2322_, 0);
lean_inc(v_val_2324_);
lean_dec_ref_known(v___x_2322_, 1);
if (lean_obj_tag(v_val_2324_) == 1)
{
uint8_t v_v_2325_; 
v_v_2325_ = lean_ctor_get_uint8(v_val_2324_, 0);
lean_dec_ref_known(v_val_2324_, 0);
return v_v_2325_;
}
else
{
uint8_t v___x_2326_; 
lean_dec(v_val_2324_);
v___x_2326_ = lean_unbox(v_defValue_2320_);
return v___x_2326_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_opts_2327_, lean_object* v_opt_2328_){
_start:
{
uint8_t v_res_2329_; lean_object* v_r_2330_; 
v_res_2329_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(v_opts_2327_, v_opt_2328_);
lean_dec_ref(v_opt_2328_);
lean_dec_ref(v_opts_2327_);
v_r_2330_ = lean_box(v_res_2329_);
return v_r_2330_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(lean_object* v_msgData_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_){
_start:
{
lean_object* v___x_2337_; lean_object* v_env_2338_; lean_object* v___x_2339_; lean_object* v_mctx_2340_; lean_object* v_lctx_2341_; lean_object* v_options_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; 
v___x_2337_ = lean_st_ref_get(v___y_2335_);
v_env_2338_ = lean_ctor_get(v___x_2337_, 0);
lean_inc_ref(v_env_2338_);
lean_dec(v___x_2337_);
v___x_2339_ = lean_st_ref_get(v___y_2333_);
v_mctx_2340_ = lean_ctor_get(v___x_2339_, 0);
lean_inc_ref(v_mctx_2340_);
lean_dec(v___x_2339_);
v_lctx_2341_ = lean_ctor_get(v___y_2332_, 2);
v_options_2342_ = lean_ctor_get(v___y_2334_, 2);
lean_inc_ref(v_options_2342_);
lean_inc_ref(v_lctx_2341_);
v___x_2343_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2343_, 0, v_env_2338_);
lean_ctor_set(v___x_2343_, 1, v_mctx_2340_);
lean_ctor_set(v___x_2343_, 2, v_lctx_2341_);
lean_ctor_set(v___x_2343_, 3, v_options_2342_);
v___x_2344_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2344_, 0, v___x_2343_);
lean_ctor_set(v___x_2344_, 1, v_msgData_2331_);
v___x_2345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2345_, 0, v___x_2344_);
return v___x_2345_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_msgData_2346_, lean_object* v___y_2347_, lean_object* v___y_2348_, lean_object* v___y_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_){
_start:
{
lean_object* v_res_2352_; 
v_res_2352_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(v_msgData_2346_, v___y_2347_, v___y_2348_, v___y_2349_, v___y_2350_);
lean_dec(v___y_2350_);
lean_dec_ref(v___y_2349_);
lean_dec(v___y_2348_);
lean_dec_ref(v___y_2347_);
return v_res_2352_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0(uint8_t v___y_2361_, uint8_t v_suppressElabErrors_2362_, lean_object* v_x_2363_){
_start:
{
if (lean_obj_tag(v_x_2363_) == 1)
{
lean_object* v_pre_2364_; 
v_pre_2364_ = lean_ctor_get(v_x_2363_, 0);
switch(lean_obj_tag(v_pre_2364_))
{
case 1:
{
lean_object* v_pre_2365_; 
v_pre_2365_ = lean_ctor_get(v_pre_2364_, 0);
switch(lean_obj_tag(v_pre_2365_))
{
case 0:
{
lean_object* v_str_2366_; lean_object* v_str_2367_; lean_object* v___x_2368_; uint8_t v___x_2369_; 
v_str_2366_ = lean_ctor_get(v_x_2363_, 1);
v_str_2367_ = lean_ctor_get(v_pre_2364_, 1);
v___x_2368_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__0));
v___x_2369_ = lean_string_dec_eq(v_str_2367_, v___x_2368_);
if (v___x_2369_ == 0)
{
lean_object* v___x_2370_; uint8_t v___x_2371_; 
v___x_2370_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__1));
v___x_2371_ = lean_string_dec_eq(v_str_2367_, v___x_2370_);
if (v___x_2371_ == 0)
{
return v___y_2361_;
}
else
{
lean_object* v___x_2372_; uint8_t v___x_2373_; 
v___x_2372_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__2));
v___x_2373_ = lean_string_dec_eq(v_str_2366_, v___x_2372_);
if (v___x_2373_ == 0)
{
return v___y_2361_;
}
else
{
return v_suppressElabErrors_2362_;
}
}
}
else
{
lean_object* v___x_2374_; uint8_t v___x_2375_; 
v___x_2374_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__3));
v___x_2375_ = lean_string_dec_eq(v_str_2366_, v___x_2374_);
if (v___x_2375_ == 0)
{
return v___y_2361_;
}
else
{
return v_suppressElabErrors_2362_;
}
}
}
case 1:
{
lean_object* v_pre_2376_; 
v_pre_2376_ = lean_ctor_get(v_pre_2365_, 0);
if (lean_obj_tag(v_pre_2376_) == 0)
{
lean_object* v_str_2377_; lean_object* v_str_2378_; lean_object* v_str_2379_; lean_object* v___x_2380_; uint8_t v___x_2381_; 
v_str_2377_ = lean_ctor_get(v_x_2363_, 1);
v_str_2378_ = lean_ctor_get(v_pre_2364_, 1);
v_str_2379_ = lean_ctor_get(v_pre_2365_, 1);
v___x_2380_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__4));
v___x_2381_ = lean_string_dec_eq(v_str_2379_, v___x_2380_);
if (v___x_2381_ == 0)
{
return v___y_2361_;
}
else
{
lean_object* v___x_2382_; uint8_t v___x_2383_; 
v___x_2382_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__5));
v___x_2383_ = lean_string_dec_eq(v_str_2378_, v___x_2382_);
if (v___x_2383_ == 0)
{
return v___y_2361_;
}
else
{
lean_object* v___x_2384_; uint8_t v___x_2385_; 
v___x_2384_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__6));
v___x_2385_ = lean_string_dec_eq(v_str_2377_, v___x_2384_);
if (v___x_2385_ == 0)
{
return v___y_2361_;
}
else
{
return v_suppressElabErrors_2362_;
}
}
}
}
else
{
return v___y_2361_;
}
}
default: 
{
return v___y_2361_;
}
}
}
case 0:
{
lean_object* v_str_2386_; lean_object* v___x_2387_; uint8_t v___x_2388_; 
v_str_2386_ = lean_ctor_get(v_x_2363_, 1);
v___x_2387_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__7));
v___x_2388_ = lean_string_dec_eq(v_str_2386_, v___x_2387_);
if (v___x_2388_ == 0)
{
return v___y_2361_;
}
else
{
return v_suppressElabErrors_2362_;
}
}
default: 
{
return v___y_2361_;
}
}
}
else
{
return v___y_2361_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___boxed(lean_object* v___y_2389_, lean_object* v_suppressElabErrors_2390_, lean_object* v_x_2391_){
_start:
{
uint8_t v___y_4359__boxed_2392_; uint8_t v_suppressElabErrors_boxed_2393_; uint8_t v_res_2394_; lean_object* v_r_2395_; 
v___y_4359__boxed_2392_ = lean_unbox(v___y_2389_);
v_suppressElabErrors_boxed_2393_ = lean_unbox(v_suppressElabErrors_2390_);
v_res_2394_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0(v___y_4359__boxed_2392_, v_suppressElabErrors_boxed_2393_, v_x_2391_);
lean_dec(v_x_2391_);
v_r_2395_ = lean_box(v_res_2394_);
return v_r_2395_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_2396_, lean_object* v_msgData_2397_, uint8_t v_severity_2398_, uint8_t v_isSilent_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_){
_start:
{
uint8_t v___y_2406_; uint8_t v___y_2407_; lean_object* v___y_2408_; lean_object* v___y_2409_; lean_object* v___y_2410_; lean_object* v___y_2411_; lean_object* v___y_2412_; lean_object* v___y_2413_; lean_object* v___y_2414_; lean_object* v___y_2442_; uint8_t v___y_2443_; uint8_t v___y_2444_; lean_object* v___y_2445_; lean_object* v___y_2446_; lean_object* v___y_2447_; uint8_t v___y_2448_; lean_object* v___y_2449_; lean_object* v___y_2467_; uint8_t v___y_2468_; lean_object* v___y_2469_; uint8_t v___y_2470_; lean_object* v___y_2471_; lean_object* v___y_2472_; uint8_t v___y_2473_; lean_object* v___y_2474_; lean_object* v___y_2478_; uint8_t v___y_2479_; lean_object* v___y_2480_; lean_object* v___y_2481_; uint8_t v___y_2482_; lean_object* v___y_2483_; uint8_t v___y_2484_; uint8_t v___x_2489_; lean_object* v___y_2491_; lean_object* v___y_2492_; lean_object* v___y_2493_; uint8_t v___y_2494_; lean_object* v___y_2495_; uint8_t v___y_2496_; uint8_t v___y_2497_; uint8_t v___y_2499_; uint8_t v___x_2514_; 
v___x_2489_ = 2;
v___x_2514_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2398_, v___x_2489_);
if (v___x_2514_ == 0)
{
v___y_2499_ = v___x_2514_;
goto v___jp_2498_;
}
else
{
uint8_t v___x_2515_; 
lean_inc_ref(v_msgData_2397_);
v___x_2515_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2397_);
v___y_2499_ = v___x_2515_;
goto v___jp_2498_;
}
v___jp_2405_:
{
lean_object* v___x_2415_; lean_object* v_currNamespace_2416_; lean_object* v_openDecls_2417_; lean_object* v_env_2418_; lean_object* v_nextMacroScope_2419_; lean_object* v_ngen_2420_; lean_object* v_auxDeclNGen_2421_; lean_object* v_traceState_2422_; lean_object* v_cache_2423_; lean_object* v_messages_2424_; lean_object* v_infoState_2425_; lean_object* v_snapshotTasks_2426_; lean_object* v___x_2428_; uint8_t v_isShared_2429_; uint8_t v_isSharedCheck_2440_; 
v___x_2415_ = lean_st_ref_take(v___y_2414_);
v_currNamespace_2416_ = lean_ctor_get(v___y_2413_, 6);
v_openDecls_2417_ = lean_ctor_get(v___y_2413_, 7);
v_env_2418_ = lean_ctor_get(v___x_2415_, 0);
v_nextMacroScope_2419_ = lean_ctor_get(v___x_2415_, 1);
v_ngen_2420_ = lean_ctor_get(v___x_2415_, 2);
v_auxDeclNGen_2421_ = lean_ctor_get(v___x_2415_, 3);
v_traceState_2422_ = lean_ctor_get(v___x_2415_, 4);
v_cache_2423_ = lean_ctor_get(v___x_2415_, 5);
v_messages_2424_ = lean_ctor_get(v___x_2415_, 6);
v_infoState_2425_ = lean_ctor_get(v___x_2415_, 7);
v_snapshotTasks_2426_ = lean_ctor_get(v___x_2415_, 8);
v_isSharedCheck_2440_ = !lean_is_exclusive(v___x_2415_);
if (v_isSharedCheck_2440_ == 0)
{
v___x_2428_ = v___x_2415_;
v_isShared_2429_ = v_isSharedCheck_2440_;
goto v_resetjp_2427_;
}
else
{
lean_inc(v_snapshotTasks_2426_);
lean_inc(v_infoState_2425_);
lean_inc(v_messages_2424_);
lean_inc(v_cache_2423_);
lean_inc(v_traceState_2422_);
lean_inc(v_auxDeclNGen_2421_);
lean_inc(v_ngen_2420_);
lean_inc(v_nextMacroScope_2419_);
lean_inc(v_env_2418_);
lean_dec(v___x_2415_);
v___x_2428_ = lean_box(0);
v_isShared_2429_ = v_isSharedCheck_2440_;
goto v_resetjp_2427_;
}
v_resetjp_2427_:
{
lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2435_; 
lean_inc(v_openDecls_2417_);
lean_inc(v_currNamespace_2416_);
v___x_2430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2430_, 0, v_currNamespace_2416_);
lean_ctor_set(v___x_2430_, 1, v_openDecls_2417_);
v___x_2431_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2430_);
lean_ctor_set(v___x_2431_, 1, v___y_2410_);
lean_inc_ref(v___y_2408_);
lean_inc_ref(v___y_2409_);
v___x_2432_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2432_, 0, v___y_2409_);
lean_ctor_set(v___x_2432_, 1, v___y_2412_);
lean_ctor_set(v___x_2432_, 2, v___y_2411_);
lean_ctor_set(v___x_2432_, 3, v___y_2408_);
lean_ctor_set(v___x_2432_, 4, v___x_2431_);
lean_ctor_set_uint8(v___x_2432_, sizeof(void*)*5, v___y_2406_);
lean_ctor_set_uint8(v___x_2432_, sizeof(void*)*5 + 1, v___y_2407_);
lean_ctor_set_uint8(v___x_2432_, sizeof(void*)*5 + 2, v_isSilent_2399_);
v___x_2433_ = l_Lean_MessageLog_add(v___x_2432_, v_messages_2424_);
if (v_isShared_2429_ == 0)
{
lean_ctor_set(v___x_2428_, 6, v___x_2433_);
v___x_2435_ = v___x_2428_;
goto v_reusejp_2434_;
}
else
{
lean_object* v_reuseFailAlloc_2439_; 
v_reuseFailAlloc_2439_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2439_, 0, v_env_2418_);
lean_ctor_set(v_reuseFailAlloc_2439_, 1, v_nextMacroScope_2419_);
lean_ctor_set(v_reuseFailAlloc_2439_, 2, v_ngen_2420_);
lean_ctor_set(v_reuseFailAlloc_2439_, 3, v_auxDeclNGen_2421_);
lean_ctor_set(v_reuseFailAlloc_2439_, 4, v_traceState_2422_);
lean_ctor_set(v_reuseFailAlloc_2439_, 5, v_cache_2423_);
lean_ctor_set(v_reuseFailAlloc_2439_, 6, v___x_2433_);
lean_ctor_set(v_reuseFailAlloc_2439_, 7, v_infoState_2425_);
lean_ctor_set(v_reuseFailAlloc_2439_, 8, v_snapshotTasks_2426_);
v___x_2435_ = v_reuseFailAlloc_2439_;
goto v_reusejp_2434_;
}
v_reusejp_2434_:
{
lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; 
v___x_2436_ = lean_st_ref_set(v___y_2414_, v___x_2435_);
v___x_2437_ = lean_box(0);
v___x_2438_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2438_, 0, v___x_2437_);
return v___x_2438_;
}
}
}
v___jp_2441_:
{
lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v_a_2452_; lean_object* v___x_2454_; uint8_t v_isShared_2455_; uint8_t v_isSharedCheck_2465_; 
v___x_2450_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2397_);
v___x_2451_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(v___x_2450_, v___y_2400_, v___y_2401_, v___y_2402_, v___y_2403_);
v_a_2452_ = lean_ctor_get(v___x_2451_, 0);
v_isSharedCheck_2465_ = !lean_is_exclusive(v___x_2451_);
if (v_isSharedCheck_2465_ == 0)
{
v___x_2454_ = v___x_2451_;
v_isShared_2455_ = v_isSharedCheck_2465_;
goto v_resetjp_2453_;
}
else
{
lean_inc(v_a_2452_);
lean_dec(v___x_2451_);
v___x_2454_ = lean_box(0);
v_isShared_2455_ = v_isSharedCheck_2465_;
goto v_resetjp_2453_;
}
v_resetjp_2453_:
{
lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_2459_; 
lean_inc_ref_n(v___y_2447_, 2);
v___x_2456_ = l_Lean_FileMap_toPosition(v___y_2447_, v___y_2446_);
lean_dec(v___y_2446_);
v___x_2457_ = l_Lean_FileMap_toPosition(v___y_2447_, v___y_2449_);
lean_dec(v___y_2449_);
v___x_2458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2458_, 0, v___x_2457_);
v___x_2459_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
if (v___y_2448_ == 0)
{
lean_del_object(v___x_2454_);
lean_dec_ref(v___y_2442_);
v___y_2406_ = v___y_2443_;
v___y_2407_ = v___y_2444_;
v___y_2408_ = v___x_2459_;
v___y_2409_ = v___y_2445_;
v___y_2410_ = v_a_2452_;
v___y_2411_ = v___x_2458_;
v___y_2412_ = v___x_2456_;
v___y_2413_ = v___y_2402_;
v___y_2414_ = v___y_2403_;
goto v___jp_2405_;
}
else
{
uint8_t v___x_2460_; 
lean_inc(v_a_2452_);
v___x_2460_ = l_Lean_MessageData_hasTag(v___y_2442_, v_a_2452_);
if (v___x_2460_ == 0)
{
lean_object* v___x_2461_; lean_object* v___x_2463_; 
lean_dec_ref_known(v___x_2458_, 1);
lean_dec_ref(v___x_2456_);
lean_dec(v_a_2452_);
v___x_2461_ = lean_box(0);
if (v_isShared_2455_ == 0)
{
lean_ctor_set(v___x_2454_, 0, v___x_2461_);
v___x_2463_ = v___x_2454_;
goto v_reusejp_2462_;
}
else
{
lean_object* v_reuseFailAlloc_2464_; 
v_reuseFailAlloc_2464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2464_, 0, v___x_2461_);
v___x_2463_ = v_reuseFailAlloc_2464_;
goto v_reusejp_2462_;
}
v_reusejp_2462_:
{
return v___x_2463_;
}
}
else
{
lean_del_object(v___x_2454_);
v___y_2406_ = v___y_2443_;
v___y_2407_ = v___y_2444_;
v___y_2408_ = v___x_2459_;
v___y_2409_ = v___y_2445_;
v___y_2410_ = v_a_2452_;
v___y_2411_ = v___x_2458_;
v___y_2412_ = v___x_2456_;
v___y_2413_ = v___y_2402_;
v___y_2414_ = v___y_2403_;
goto v___jp_2405_;
}
}
}
}
v___jp_2466_:
{
lean_object* v___x_2475_; 
v___x_2475_ = l_Lean_Syntax_getTailPos_x3f(v___y_2469_, v___y_2468_);
lean_dec(v___y_2469_);
if (lean_obj_tag(v___x_2475_) == 0)
{
lean_inc(v___y_2474_);
v___y_2442_ = v___y_2467_;
v___y_2443_ = v___y_2468_;
v___y_2444_ = v___y_2470_;
v___y_2445_ = v___y_2471_;
v___y_2446_ = v___y_2474_;
v___y_2447_ = v___y_2472_;
v___y_2448_ = v___y_2473_;
v___y_2449_ = v___y_2474_;
goto v___jp_2441_;
}
else
{
lean_object* v_val_2476_; 
v_val_2476_ = lean_ctor_get(v___x_2475_, 0);
lean_inc(v_val_2476_);
lean_dec_ref_known(v___x_2475_, 1);
v___y_2442_ = v___y_2467_;
v___y_2443_ = v___y_2468_;
v___y_2444_ = v___y_2470_;
v___y_2445_ = v___y_2471_;
v___y_2446_ = v___y_2474_;
v___y_2447_ = v___y_2472_;
v___y_2448_ = v___y_2473_;
v___y_2449_ = v_val_2476_;
goto v___jp_2441_;
}
}
v___jp_2477_:
{
lean_object* v_ref_2485_; lean_object* v___x_2486_; 
v_ref_2485_ = l_Lean_replaceRef(v_ref_2396_, v___y_2483_);
v___x_2486_ = l_Lean_Syntax_getPos_x3f(v_ref_2485_, v___y_2479_);
if (lean_obj_tag(v___x_2486_) == 0)
{
lean_object* v___x_2487_; 
v___x_2487_ = lean_unsigned_to_nat(0u);
v___y_2467_ = v___y_2478_;
v___y_2468_ = v___y_2479_;
v___y_2469_ = v_ref_2485_;
v___y_2470_ = v___y_2484_;
v___y_2471_ = v___y_2480_;
v___y_2472_ = v___y_2481_;
v___y_2473_ = v___y_2482_;
v___y_2474_ = v___x_2487_;
goto v___jp_2466_;
}
else
{
lean_object* v_val_2488_; 
v_val_2488_ = lean_ctor_get(v___x_2486_, 0);
lean_inc(v_val_2488_);
lean_dec_ref_known(v___x_2486_, 1);
v___y_2467_ = v___y_2478_;
v___y_2468_ = v___y_2479_;
v___y_2469_ = v_ref_2485_;
v___y_2470_ = v___y_2484_;
v___y_2471_ = v___y_2480_;
v___y_2472_ = v___y_2481_;
v___y_2473_ = v___y_2482_;
v___y_2474_ = v_val_2488_;
goto v___jp_2466_;
}
}
v___jp_2490_:
{
if (v___y_2497_ == 0)
{
v___y_2478_ = v___y_2493_;
v___y_2479_ = v___y_2496_;
v___y_2480_ = v___y_2491_;
v___y_2481_ = v___y_2492_;
v___y_2482_ = v___y_2494_;
v___y_2483_ = v___y_2495_;
v___y_2484_ = v_severity_2398_;
goto v___jp_2477_;
}
else
{
v___y_2478_ = v___y_2493_;
v___y_2479_ = v___y_2496_;
v___y_2480_ = v___y_2491_;
v___y_2481_ = v___y_2492_;
v___y_2482_ = v___y_2494_;
v___y_2483_ = v___y_2495_;
v___y_2484_ = v___x_2489_;
goto v___jp_2477_;
}
}
v___jp_2498_:
{
if (v___y_2499_ == 0)
{
lean_object* v_fileName_2500_; lean_object* v_fileMap_2501_; lean_object* v_options_2502_; lean_object* v_ref_2503_; uint8_t v_suppressElabErrors_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___f_2507_; uint8_t v___x_2508_; uint8_t v___x_2509_; 
v_fileName_2500_ = lean_ctor_get(v___y_2402_, 0);
v_fileMap_2501_ = lean_ctor_get(v___y_2402_, 1);
v_options_2502_ = lean_ctor_get(v___y_2402_, 2);
v_ref_2503_ = lean_ctor_get(v___y_2402_, 5);
v_suppressElabErrors_2504_ = lean_ctor_get_uint8(v___y_2402_, sizeof(void*)*14 + 1);
v___x_2505_ = lean_box(v___y_2499_);
v___x_2506_ = lean_box(v_suppressElabErrors_2504_);
v___f_2507_ = lean_alloc_closure((void*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2507_, 0, v___x_2505_);
lean_closure_set(v___f_2507_, 1, v___x_2506_);
v___x_2508_ = 1;
v___x_2509_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2398_, v___x_2508_);
if (v___x_2509_ == 0)
{
v___y_2491_ = v_fileName_2500_;
v___y_2492_ = v_fileMap_2501_;
v___y_2493_ = v___f_2507_;
v___y_2494_ = v_suppressElabErrors_2504_;
v___y_2495_ = v_ref_2503_;
v___y_2496_ = v___y_2499_;
v___y_2497_ = v___x_2509_;
goto v___jp_2490_;
}
else
{
lean_object* v___x_2510_; uint8_t v___x_2511_; 
v___x_2510_ = l_Lean_warningAsError;
v___x_2511_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(v_options_2502_, v___x_2510_);
v___y_2491_ = v_fileName_2500_;
v___y_2492_ = v_fileMap_2501_;
v___y_2493_ = v___f_2507_;
v___y_2494_ = v_suppressElabErrors_2504_;
v___y_2495_ = v_ref_2503_;
v___y_2496_ = v___y_2499_;
v___y_2497_ = v___x_2511_;
goto v___jp_2490_;
}
}
else
{
lean_object* v___x_2512_; lean_object* v___x_2513_; 
lean_dec_ref(v_msgData_2397_);
v___x_2512_ = lean_box(0);
v___x_2513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2513_, 0, v___x_2512_);
return v___x_2513_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_2516_, lean_object* v_msgData_2517_, lean_object* v_severity_2518_, lean_object* v_isSilent_2519_, lean_object* v___y_2520_, lean_object* v___y_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_){
_start:
{
uint8_t v_severity_boxed_2525_; uint8_t v_isSilent_boxed_2526_; lean_object* v_res_2527_; 
v_severity_boxed_2525_ = lean_unbox(v_severity_2518_);
v_isSilent_boxed_2526_ = lean_unbox(v_isSilent_2519_);
v_res_2527_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg(v_ref_2516_, v_msgData_2517_, v_severity_boxed_2525_, v_isSilent_boxed_2526_, v___y_2520_, v___y_2521_, v___y_2522_, v___y_2523_);
lean_dec(v___y_2523_);
lean_dec_ref(v___y_2522_);
lean_dec(v___y_2521_);
lean_dec_ref(v___y_2520_);
lean_dec(v_ref_2516_);
return v_res_2527_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0(lean_object* v_msgData_2528_, uint8_t v_severity_2529_, uint8_t v_isSilent_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_, lean_object* v___y_2533_, lean_object* v___y_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_){
_start:
{
lean_object* v_ref_2538_; lean_object* v___x_2539_; 
v_ref_2538_ = lean_ctor_get(v___y_2535_, 5);
v___x_2539_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg(v_ref_2538_, v_msgData_2528_, v_severity_2529_, v_isSilent_2530_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
return v___x_2539_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0___boxed(lean_object* v_msgData_2540_, lean_object* v_severity_2541_, lean_object* v_isSilent_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_){
_start:
{
uint8_t v_severity_boxed_2550_; uint8_t v_isSilent_boxed_2551_; lean_object* v_res_2552_; 
v_severity_boxed_2550_ = lean_unbox(v_severity_2541_);
v_isSilent_boxed_2551_ = lean_unbox(v_isSilent_2542_);
v_res_2552_ = lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0(v_msgData_2540_, v_severity_boxed_2550_, v_isSilent_boxed_2551_, v___y_2543_, v___y_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_);
lean_dec(v___y_2548_);
lean_dec_ref(v___y_2547_);
lean_dec(v___y_2546_);
lean_dec_ref(v___y_2545_);
lean_dec(v___y_2544_);
lean_dec_ref(v___y_2543_);
return v_res_2552_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(lean_object* v_msgData_2553_, lean_object* v___y_2554_, lean_object* v___y_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_, lean_object* v___y_2558_, lean_object* v___y_2559_){
_start:
{
uint8_t v___x_2561_; uint8_t v___x_2562_; lean_object* v___x_2563_; 
v___x_2561_ = 1;
v___x_2562_ = 0;
v___x_2563_ = lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0(v_msgData_2553_, v___x_2561_, v___x_2562_, v___y_2554_, v___y_2555_, v___y_2556_, v___y_2557_, v___y_2558_, v___y_2559_);
return v___x_2563_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0___boxed(lean_object* v_msgData_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_, lean_object* v___y_2569_, lean_object* v___y_2570_, lean_object* v___y_2571_){
_start:
{
lean_object* v_res_2572_; 
v_res_2572_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(v_msgData_2564_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_, v___y_2569_, v___y_2570_);
lean_dec(v___y_2570_);
lean_dec_ref(v___y_2569_);
lean_dec(v___y_2568_);
lean_dec_ref(v___y_2567_);
lean_dec(v___y_2566_);
lean_dec_ref(v___y_2565_);
return v_res_2572_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0(uint8_t v___y_2574_, lean_object* v_ss_2575_, lean_object* v_s_2576_, lean_object* v_stx_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_){
_start:
{
if (v___y_2574_ == 0)
{
lean_object* v___x_2585_; lean_object* v___x_2586_; lean_object* v___x_2587_; lean_object* v___x_2588_; 
lean_dec(v_stx_2577_);
lean_dec_ref(v_s_2576_);
v___x_2585_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(v_ss_2575_);
v___x_2586_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2586_, 0, v___x_2585_);
v___x_2587_ = l_Lean_MessageData_ofFormat(v___x_2586_);
v___x_2588_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(v___x_2587_, v___y_2578_, v___y_2579_, v___y_2580_, v___y_2581_, v___y_2582_, v___y_2583_);
return v___x_2588_;
}
else
{
lean_object* v_name_2589_; lean_object* v_queryNum_2590_; lean_object* v___x_2591_; 
v_name_2589_ = lean_ctor_get(v_ss_2575_, 0);
lean_inc_ref(v_name_2589_);
v_queryNum_2590_ = lean_ctor_get(v_ss_2575_, 4);
lean_inc_ref(v_queryNum_2590_);
lean_inc(v___y_2583_);
lean_inc_ref(v___y_2582_);
v___x_2591_ = lean_apply_3(v_queryNum_2590_, v___y_2582_, v___y_2583_, lean_box(0));
if (lean_obj_tag(v___x_2591_) == 0)
{
lean_object* v_a_2592_; lean_object* v___x_2593_; 
v_a_2592_ = lean_ctor_get(v___x_2591_, 0);
lean_inc(v_a_2592_);
lean_dec_ref_known(v___x_2591_, 1);
v___x_2593_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_getCommandSuggestions(v_ss_2575_, v_s_2576_, v_a_2592_, v___y_2580_, v___y_2581_, v___y_2582_, v___y_2583_);
if (lean_obj_tag(v___x_2593_) == 0)
{
lean_object* v_a_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; uint8_t v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; 
v_a_2594_ = lean_ctor_get(v___x_2593_, 0);
lean_inc(v_a_2594_);
lean_dec_ref_known(v___x_2593_, 1);
v___x_2595_ = lean_box(0);
v___x_2596_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___closed__0));
v___x_2597_ = lean_string_append(v_name_2589_, v___x_2596_);
v___x_2598_ = 4;
v___x_2599_ = l_Lean_MessageData_nil;
v___x_2600_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2577_, v_a_2594_, v___x_2595_, v___x_2597_, v___x_2595_, v___x_2598_, v___x_2599_, v___y_2582_, v___y_2583_);
return v___x_2600_;
}
else
{
lean_object* v_a_2601_; lean_object* v___x_2603_; uint8_t v_isShared_2604_; uint8_t v_isSharedCheck_2608_; 
lean_dec_ref(v_name_2589_);
lean_dec(v_stx_2577_);
v_a_2601_ = lean_ctor_get(v___x_2593_, 0);
v_isSharedCheck_2608_ = !lean_is_exclusive(v___x_2593_);
if (v_isSharedCheck_2608_ == 0)
{
v___x_2603_ = v___x_2593_;
v_isShared_2604_ = v_isSharedCheck_2608_;
goto v_resetjp_2602_;
}
else
{
lean_inc(v_a_2601_);
lean_dec(v___x_2593_);
v___x_2603_ = lean_box(0);
v_isShared_2604_ = v_isSharedCheck_2608_;
goto v_resetjp_2602_;
}
v_resetjp_2602_:
{
lean_object* v___x_2606_; 
if (v_isShared_2604_ == 0)
{
v___x_2606_ = v___x_2603_;
goto v_reusejp_2605_;
}
else
{
lean_object* v_reuseFailAlloc_2607_; 
v_reuseFailAlloc_2607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2607_, 0, v_a_2601_);
v___x_2606_ = v_reuseFailAlloc_2607_;
goto v_reusejp_2605_;
}
v_reusejp_2605_:
{
return v___x_2606_;
}
}
}
}
else
{
lean_object* v_a_2609_; lean_object* v___x_2611_; uint8_t v_isShared_2612_; uint8_t v_isSharedCheck_2616_; 
lean_dec_ref(v_name_2589_);
lean_dec(v_stx_2577_);
lean_dec_ref(v_s_2576_);
lean_dec_ref(v_ss_2575_);
v_a_2609_ = lean_ctor_get(v___x_2591_, 0);
v_isSharedCheck_2616_ = !lean_is_exclusive(v___x_2591_);
if (v_isSharedCheck_2616_ == 0)
{
v___x_2611_ = v___x_2591_;
v_isShared_2612_ = v_isSharedCheck_2616_;
goto v_resetjp_2610_;
}
else
{
lean_inc(v_a_2609_);
lean_dec(v___x_2591_);
v___x_2611_ = lean_box(0);
v_isShared_2612_ = v_isSharedCheck_2616_;
goto v_resetjp_2610_;
}
v_resetjp_2610_:
{
lean_object* v___x_2614_; 
if (v_isShared_2612_ == 0)
{
v___x_2614_ = v___x_2611_;
goto v_reusejp_2613_;
}
else
{
lean_object* v_reuseFailAlloc_2615_; 
v_reuseFailAlloc_2615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2615_, 0, v_a_2609_);
v___x_2614_ = v_reuseFailAlloc_2615_;
goto v_reusejp_2613_;
}
v_reusejp_2613_:
{
return v___x_2614_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___boxed(lean_object* v___y_2617_, lean_object* v_ss_2618_, lean_object* v_s_2619_, lean_object* v_stx_2620_, lean_object* v___y_2621_, lean_object* v___y_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_){
_start:
{
uint8_t v___y_4688__boxed_2628_; lean_object* v_res_2629_; 
v___y_4688__boxed_2628_ = lean_unbox(v___y_2617_);
v_res_2629_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0(v___y_4688__boxed_2628_, v_ss_2618_, v_s_2619_, v_stx_2620_, v___y_2621_, v___y_2622_, v___y_2623_, v___y_2624_, v___y_2625_, v___y_2626_);
lean_dec(v___y_2626_);
lean_dec_ref(v___y_2625_);
lean_dec(v___y_2624_);
lean_dec_ref(v___y_2623_);
lean_dec(v___y_2622_);
lean_dec_ref(v___y_2621_);
return v_res_2629_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1(void){
_start:
{
lean_object* v___x_2631_; lean_object* v___x_2632_; 
v___x_2631_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0));
v___x_2632_ = lean_string_utf8_byte_size(v___x_2631_);
return v___x_2632_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2(void){
_start:
{
lean_object* v___x_2633_; lean_object* v___x_2634_; 
v___x_2633_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1));
v___x_2634_ = lean_string_utf8_byte_size(v___x_2633_);
return v___x_2634_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions(lean_object* v_ss_2635_, lean_object* v_stx_2636_, lean_object* v_s_2637_, lean_object* v_a_2638_, lean_object* v_a_2639_){
_start:
{
lean_object* v_s_2641_; uint8_t v___y_2643_; lean_object* v___x_2659_; lean_object* v___x_2660_; lean_object* v___x_2661_; uint8_t v___x_2662_; 
v_s_2641_ = l_Lean_TSyntax_getString(v_s_2637_);
v___x_2659_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1));
v___x_2660_ = lean_string_utf8_byte_size(v_s_2641_);
v___x_2661_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2, &lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2);
v___x_2662_ = lean_nat_dec_le(v___x_2661_, v___x_2660_);
if (v___x_2662_ == 0)
{
goto v___jp_2651_;
}
else
{
lean_object* v___x_2663_; lean_object* v___x_2664_; uint8_t v___x_2665_; 
v___x_2663_ = lean_unsigned_to_nat(0u);
v___x_2664_ = lean_nat_sub(v___x_2660_, v___x_2661_);
v___x_2665_ = lean_string_memcmp(v_s_2641_, v___x_2659_, v___x_2664_, v___x_2663_, v___x_2661_);
lean_dec(v___x_2664_);
if (v___x_2665_ == 0)
{
goto v___jp_2651_;
}
else
{
goto v___jp_2649_;
}
}
v___jp_2642_:
{
lean_object* v___x_2644_; lean_object* v___y_2645_; lean_object* v___x_2646_; 
v___x_2644_ = lean_box(v___y_2643_);
v___y_2645_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___boxed), 11, 4);
lean_closure_set(v___y_2645_, 0, v___x_2644_);
lean_closure_set(v___y_2645_, 1, v_ss_2635_);
lean_closure_set(v___y_2645_, 2, v_s_2641_);
lean_closure_set(v___y_2645_, 3, v_stx_2636_);
v___x_2646_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___y_2645_, v_a_2638_, v_a_2639_);
return v___x_2646_;
}
v___jp_2647_:
{
uint8_t v___x_2648_; 
v___x_2648_ = 0;
v___y_2643_ = v___x_2648_;
goto v___jp_2642_;
}
v___jp_2649_:
{
uint8_t v___x_2650_; 
v___x_2650_ = 1;
v___y_2643_ = v___x_2650_;
goto v___jp_2642_;
}
v___jp_2651_:
{
lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; uint8_t v___x_2655_; 
v___x_2652_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0));
v___x_2653_ = lean_string_utf8_byte_size(v_s_2641_);
v___x_2654_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1, &lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1);
v___x_2655_ = lean_nat_dec_le(v___x_2654_, v___x_2653_);
if (v___x_2655_ == 0)
{
goto v___jp_2647_;
}
else
{
lean_object* v___x_2656_; lean_object* v___x_2657_; uint8_t v___x_2658_; 
v___x_2656_ = lean_unsigned_to_nat(0u);
v___x_2657_ = lean_nat_sub(v___x_2653_, v___x_2654_);
v___x_2658_ = lean_string_memcmp(v_s_2641_, v___x_2652_, v___x_2657_, v___x_2656_, v___x_2654_);
lean_dec(v___x_2657_);
if (v___x_2658_ == 0)
{
goto v___jp_2647_;
}
else
{
goto v___jp_2649_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___boxed(lean_object* v_ss_2666_, lean_object* v_stx_2667_, lean_object* v_s_2668_, lean_object* v_a_2669_, lean_object* v_a_2670_, lean_object* v_a_2671_){
_start:
{
lean_object* v_res_2672_; 
v_res_2672_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions(v_ss_2666_, v_stx_2667_, v_s_2668_, v_a_2669_, v_a_2670_);
lean_dec(v_a_2670_);
lean_dec_ref(v_a_2669_);
lean_dec(v_s_2668_);
return v_res_2672_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1(lean_object* v_ref_2673_, lean_object* v_msgData_2674_, uint8_t v_severity_2675_, uint8_t v_isSilent_2676_, lean_object* v___y_2677_, lean_object* v___y_2678_, lean_object* v___y_2679_, lean_object* v___y_2680_, lean_object* v___y_2681_, lean_object* v___y_2682_){
_start:
{
lean_object* v___x_2684_; 
v___x_2684_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg(v_ref_2673_, v_msgData_2674_, v_severity_2675_, v_isSilent_2676_, v___y_2679_, v___y_2680_, v___y_2681_, v___y_2682_);
return v___x_2684_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___boxed(lean_object* v_ref_2685_, lean_object* v_msgData_2686_, lean_object* v_severity_2687_, lean_object* v_isSilent_2688_, lean_object* v___y_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_, lean_object* v___y_2692_, lean_object* v___y_2693_, lean_object* v___y_2694_, lean_object* v___y_2695_){
_start:
{
uint8_t v_severity_boxed_2696_; uint8_t v_isSilent_boxed_2697_; lean_object* v_res_2698_; 
v_severity_boxed_2696_ = lean_unbox(v_severity_2687_);
v_isSilent_boxed_2697_ = lean_unbox(v_isSilent_2688_);
v_res_2698_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1(v_ref_2685_, v_msgData_2686_, v_severity_boxed_2696_, v_isSilent_boxed_2697_, v___y_2689_, v___y_2690_, v___y_2691_, v___y_2692_, v___y_2693_, v___y_2694_);
lean_dec(v___y_2694_);
lean_dec_ref(v___y_2693_);
lean_dec(v___y_2692_);
lean_dec_ref(v___y_2691_);
lean_dec(v___y_2690_);
lean_dec_ref(v___y_2689_);
lean_dec(v_ref_2685_);
return v_res_2698_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions(lean_object* v_ss_2699_, lean_object* v_stx_2700_, lean_object* v_s_2701_, lean_object* v_a_2702_, lean_object* v_a_2703_, lean_object* v_a_2704_, lean_object* v_a_2705_, lean_object* v_a_2706_, lean_object* v_a_2707_){
_start:
{
lean_object* v_s_2714_; lean_object* v___x_2752_; lean_object* v___x_2753_; lean_object* v___x_2754_; uint8_t v___x_2755_; 
v_s_2714_ = l_Lean_TSyntax_getString(v_s_2701_);
v___x_2752_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1));
v___x_2753_ = lean_string_utf8_byte_size(v_s_2714_);
v___x_2754_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2, &lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2);
v___x_2755_ = lean_nat_dec_le(v___x_2754_, v___x_2753_);
if (v___x_2755_ == 0)
{
goto v___jp_2744_;
}
else
{
lean_object* v___x_2756_; lean_object* v___x_2757_; uint8_t v___x_2758_; 
v___x_2756_ = lean_unsigned_to_nat(0u);
v___x_2757_ = lean_nat_sub(v___x_2753_, v___x_2754_);
v___x_2758_ = lean_string_memcmp(v_s_2714_, v___x_2752_, v___x_2757_, v___x_2756_, v___x_2754_);
lean_dec(v___x_2757_);
if (v___x_2758_ == 0)
{
goto v___jp_2744_;
}
else
{
goto v___jp_2715_;
}
}
v___jp_2709_:
{
lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; 
v___x_2710_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(v_ss_2699_);
v___x_2711_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2711_, 0, v___x_2710_);
v___x_2712_ = l_Lean_MessageData_ofFormat(v___x_2711_);
v___x_2713_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(v___x_2712_, v_a_2702_, v_a_2703_, v_a_2704_, v_a_2705_, v_a_2706_, v_a_2707_);
return v___x_2713_;
}
v___jp_2715_:
{
lean_object* v_name_2716_; lean_object* v_queryNum_2717_; lean_object* v___x_2718_; 
v_name_2716_ = lean_ctor_get(v_ss_2699_, 0);
lean_inc_ref(v_name_2716_);
v_queryNum_2717_ = lean_ctor_get(v_ss_2699_, 4);
lean_inc_ref(v_queryNum_2717_);
lean_inc(v_a_2707_);
lean_inc_ref(v_a_2706_);
v___x_2718_ = lean_apply_3(v_queryNum_2717_, v_a_2706_, v_a_2707_, lean_box(0));
if (lean_obj_tag(v___x_2718_) == 0)
{
lean_object* v_a_2719_; lean_object* v___x_2720_; 
v_a_2719_ = lean_ctor_get(v___x_2718_, 0);
lean_inc(v_a_2719_);
lean_dec_ref_known(v___x_2718_, 1);
v___x_2720_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_getTermSuggestions(v_ss_2699_, v_s_2714_, v_a_2719_, v_a_2704_, v_a_2705_, v_a_2706_, v_a_2707_);
if (lean_obj_tag(v___x_2720_) == 0)
{
lean_object* v_a_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; uint8_t v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; 
v_a_2721_ = lean_ctor_get(v___x_2720_, 0);
lean_inc(v_a_2721_);
lean_dec_ref_known(v___x_2720_, 1);
v___x_2722_ = lean_box(0);
v___x_2723_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___lam__0___closed__0));
v___x_2724_ = lean_string_append(v_name_2716_, v___x_2723_);
v___x_2725_ = 4;
v___x_2726_ = l_Lean_MessageData_nil;
v___x_2727_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2700_, v_a_2721_, v___x_2722_, v___x_2724_, v___x_2722_, v___x_2725_, v___x_2726_, v_a_2706_, v_a_2707_);
return v___x_2727_;
}
else
{
lean_object* v_a_2728_; lean_object* v___x_2730_; uint8_t v_isShared_2731_; uint8_t v_isSharedCheck_2735_; 
lean_dec_ref(v_name_2716_);
lean_dec(v_stx_2700_);
v_a_2728_ = lean_ctor_get(v___x_2720_, 0);
v_isSharedCheck_2735_ = !lean_is_exclusive(v___x_2720_);
if (v_isSharedCheck_2735_ == 0)
{
v___x_2730_ = v___x_2720_;
v_isShared_2731_ = v_isSharedCheck_2735_;
goto v_resetjp_2729_;
}
else
{
lean_inc(v_a_2728_);
lean_dec(v___x_2720_);
v___x_2730_ = lean_box(0);
v_isShared_2731_ = v_isSharedCheck_2735_;
goto v_resetjp_2729_;
}
v_resetjp_2729_:
{
lean_object* v___x_2733_; 
if (v_isShared_2731_ == 0)
{
v___x_2733_ = v___x_2730_;
goto v_reusejp_2732_;
}
else
{
lean_object* v_reuseFailAlloc_2734_; 
v_reuseFailAlloc_2734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2734_, 0, v_a_2728_);
v___x_2733_ = v_reuseFailAlloc_2734_;
goto v_reusejp_2732_;
}
v_reusejp_2732_:
{
return v___x_2733_;
}
}
}
}
else
{
lean_object* v_a_2736_; lean_object* v___x_2738_; uint8_t v_isShared_2739_; uint8_t v_isSharedCheck_2743_; 
lean_dec_ref(v_name_2716_);
lean_dec_ref(v_s_2714_);
lean_dec(v_stx_2700_);
lean_dec_ref(v_ss_2699_);
v_a_2736_ = lean_ctor_get(v___x_2718_, 0);
v_isSharedCheck_2743_ = !lean_is_exclusive(v___x_2718_);
if (v_isSharedCheck_2743_ == 0)
{
v___x_2738_ = v___x_2718_;
v_isShared_2739_ = v_isSharedCheck_2743_;
goto v_resetjp_2737_;
}
else
{
lean_inc(v_a_2736_);
lean_dec(v___x_2718_);
v___x_2738_ = lean_box(0);
v_isShared_2739_ = v_isSharedCheck_2743_;
goto v_resetjp_2737_;
}
v_resetjp_2737_:
{
lean_object* v___x_2741_; 
if (v_isShared_2739_ == 0)
{
v___x_2741_ = v___x_2738_;
goto v_reusejp_2740_;
}
else
{
lean_object* v_reuseFailAlloc_2742_; 
v_reuseFailAlloc_2742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2742_, 0, v_a_2736_);
v___x_2741_ = v_reuseFailAlloc_2742_;
goto v_reusejp_2740_;
}
v_reusejp_2740_:
{
return v___x_2741_;
}
}
}
}
v___jp_2744_:
{
lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2747_; uint8_t v___x_2748_; 
v___x_2745_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0));
v___x_2746_ = lean_string_utf8_byte_size(v_s_2714_);
v___x_2747_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1, &lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1);
v___x_2748_ = lean_nat_dec_le(v___x_2747_, v___x_2746_);
if (v___x_2748_ == 0)
{
lean_dec_ref(v_s_2714_);
lean_dec(v_stx_2700_);
goto v___jp_2709_;
}
else
{
lean_object* v___x_2749_; lean_object* v___x_2750_; uint8_t v___x_2751_; 
v___x_2749_ = lean_unsigned_to_nat(0u);
v___x_2750_ = lean_nat_sub(v___x_2746_, v___x_2747_);
v___x_2751_ = lean_string_memcmp(v_s_2714_, v___x_2745_, v___x_2750_, v___x_2749_, v___x_2747_);
lean_dec(v___x_2750_);
if (v___x_2751_ == 0)
{
lean_dec_ref(v_s_2714_);
lean_dec(v_stx_2700_);
goto v___jp_2709_;
}
else
{
goto v___jp_2715_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions___boxed(lean_object* v_ss_2759_, lean_object* v_stx_2760_, lean_object* v_s_2761_, lean_object* v_a_2762_, lean_object* v_a_2763_, lean_object* v_a_2764_, lean_object* v_a_2765_, lean_object* v_a_2766_, lean_object* v_a_2767_, lean_object* v_a_2768_){
_start:
{
lean_object* v_res_2769_; 
v_res_2769_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions(v_ss_2759_, v_stx_2760_, v_s_2761_, v_a_2762_, v_a_2763_, v_a_2764_, v_a_2765_, v_a_2766_, v_a_2767_);
lean_dec(v_a_2767_);
lean_dec_ref(v_a_2766_);
lean_dec(v_a_2765_);
lean_dec_ref(v_a_2764_);
lean_dec(v_a_2763_);
lean_dec_ref(v_a_2762_);
lean_dec(v_s_2761_);
return v_res_2769_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(lean_object* v_a_2774_, lean_object* v_as_2775_, size_t v_i_2776_, size_t v_stop_2777_, lean_object* v_b_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_){
_start:
{
lean_object* v_a_2787_; uint8_t v___x_2791_; 
v___x_2791_ = lean_usize_dec_eq(v_i_2776_, v_stop_2777_);
if (v___x_2791_ == 0)
{
lean_object* v___x_2792_; lean_object* v_suggestion_2793_; 
v___x_2792_ = lean_array_uget_borrowed(v_as_2775_, v_i_2776_);
v_suggestion_2793_ = lean_ctor_get(v___x_2792_, 0);
if (lean_obj_tag(v_suggestion_2793_) == 1)
{
lean_object* v_a_2794_; lean_object* v___x_2795_; lean_object* v_env_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; 
v_a_2794_ = lean_ctor_get(v_suggestion_2793_, 0);
v___x_2795_ = lean_st_ref_get(v___y_2784_);
v_env_2796_ = lean_ctor_get(v___x_2795_, 0);
lean_inc_ref(v_env_2796_);
lean_dec(v___x_2795_);
v___x_2797_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__1));
v___x_2798_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___closed__2));
lean_inc_ref(v_a_2794_);
v___x_2799_ = l_Lean_Parser_runParserCategory(v_env_2796_, v___x_2797_, v_a_2794_, v___x_2798_);
if (lean_obj_tag(v___x_2799_) == 0)
{
lean_dec_ref_known(v___x_2799_, 1);
v_a_2787_ = v_b_2778_;
goto v___jp_2786_;
}
else
{
lean_object* v_a_2800_; lean_object* v___x_2801_; 
v_a_2800_ = lean_ctor_get(v___x_2799_, 0);
lean_inc(v_a_2800_);
lean_dec_ref_known(v___x_2799_, 1);
lean_inc_ref(v_a_2774_);
v___x_2801_ = lp_LeanSearchClient_LeanSearchClient_checkTactic(v_a_2774_, v_a_2800_, v___y_2779_, v___y_2780_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_);
if (lean_obj_tag(v___x_2801_) == 0)
{
lean_object* v_a_2802_; 
v_a_2802_ = lean_ctor_get(v___x_2801_, 0);
lean_inc(v_a_2802_);
lean_dec_ref_known(v___x_2801_, 1);
if (lean_obj_tag(v_a_2802_) == 0)
{
v_a_2787_ = v_b_2778_;
goto v___jp_2786_;
}
else
{
lean_object* v___x_2803_; 
lean_dec_ref_known(v_a_2802_, 1);
lean_inc(v___x_2792_);
v___x_2803_ = lean_array_push(v_b_2778_, v___x_2792_);
v_a_2787_ = v___x_2803_;
goto v___jp_2786_;
}
}
else
{
lean_object* v_a_2804_; lean_object* v___x_2806_; uint8_t v_isShared_2807_; uint8_t v_isSharedCheck_2811_; 
lean_dec_ref(v_b_2778_);
lean_dec_ref(v_a_2774_);
v_a_2804_ = lean_ctor_get(v___x_2801_, 0);
v_isSharedCheck_2811_ = !lean_is_exclusive(v___x_2801_);
if (v_isSharedCheck_2811_ == 0)
{
v___x_2806_ = v___x_2801_;
v_isShared_2807_ = v_isSharedCheck_2811_;
goto v_resetjp_2805_;
}
else
{
lean_inc(v_a_2804_);
lean_dec(v___x_2801_);
v___x_2806_ = lean_box(0);
v_isShared_2807_ = v_isSharedCheck_2811_;
goto v_resetjp_2805_;
}
v_resetjp_2805_:
{
lean_object* v___x_2809_; 
if (v_isShared_2807_ == 0)
{
v___x_2809_ = v___x_2806_;
goto v_reusejp_2808_;
}
else
{
lean_object* v_reuseFailAlloc_2810_; 
v_reuseFailAlloc_2810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2810_, 0, v_a_2804_);
v___x_2809_ = v_reuseFailAlloc_2810_;
goto v_reusejp_2808_;
}
v_reusejp_2808_:
{
return v___x_2809_;
}
}
}
}
}
else
{
v_a_2787_ = v_b_2778_;
goto v___jp_2786_;
}
}
else
{
lean_object* v___x_2812_; 
lean_dec_ref(v_a_2774_);
v___x_2812_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2812_, 0, v_b_2778_);
return v___x_2812_;
}
v___jp_2786_:
{
size_t v___x_2788_; size_t v___x_2789_; 
v___x_2788_ = ((size_t)1ULL);
v___x_2789_ = lean_usize_add(v_i_2776_, v___x_2788_);
v_i_2776_ = v___x_2789_;
v_b_2778_ = v_a_2787_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg___boxed(lean_object* v_a_2813_, lean_object* v_as_2814_, lean_object* v_i_2815_, lean_object* v_stop_2816_, lean_object* v_b_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_, lean_object* v___y_2822_, lean_object* v___y_2823_, lean_object* v___y_2824_){
_start:
{
size_t v_i_boxed_2825_; size_t v_stop_boxed_2826_; lean_object* v_res_2827_; 
v_i_boxed_2825_ = lean_unbox_usize(v_i_2815_);
lean_dec(v_i_2815_);
v_stop_boxed_2826_ = lean_unbox_usize(v_stop_2816_);
lean_dec(v_stop_2816_);
v_res_2827_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(v_a_2813_, v_as_2814_, v_i_boxed_2825_, v_stop_boxed_2826_, v_b_2817_, v___y_2818_, v___y_2819_, v___y_2820_, v___y_2821_, v___y_2822_, v___y_2823_);
lean_dec(v___y_2823_);
lean_dec_ref(v___y_2822_);
lean_dec(v___y_2821_);
lean_dec_ref(v___y_2820_);
lean_dec(v___y_2819_);
lean_dec_ref(v___y_2818_);
lean_dec_ref(v_as_2814_);
return v_res_2827_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2(lean_object* v_stx_2829_, lean_object* v_a_2830_, lean_object* v_as_2831_, size_t v_sz_2832_, size_t v_i_2833_, lean_object* v_b_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_, lean_object* v___y_2841_, lean_object* v___y_2842_){
_start:
{
lean_object* v_a_2845_; uint8_t v___x_2849_; 
v___x_2849_ = lean_usize_dec_lt(v_i_2833_, v_sz_2832_);
if (v___x_2849_ == 0)
{
lean_object* v___x_2850_; 
lean_dec_ref(v_a_2830_);
lean_dec(v_stx_2829_);
v___x_2850_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2850_, 0, v_b_2834_);
return v___x_2850_;
}
else
{
lean_object* v_a_2851_; lean_object* v_fst_2852_; lean_object* v_snd_2853_; lean_object* v___x_2854_; lean_object* v_a_2856_; lean_object* v___y_2867_; lean_object* v___x_2877_; lean_object* v___x_2878_; lean_object* v___x_2879_; uint8_t v___x_2880_; 
v_a_2851_ = lean_array_uget_borrowed(v_as_2831_, v_i_2833_);
v_fst_2852_ = lean_ctor_get(v_a_2851_, 0);
v_snd_2853_ = lean_ctor_get(v_a_2851_, 1);
v___x_2854_ = lean_box(0);
v___x_2877_ = lean_unsigned_to_nat(0u);
v___x_2878_ = lean_array_get_size(v_snd_2853_);
v___x_2879_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__0));
v___x_2880_ = lean_nat_dec_lt(v___x_2877_, v___x_2878_);
if (v___x_2880_ == 0)
{
v_a_2856_ = v___x_2879_;
goto v___jp_2855_;
}
else
{
uint8_t v___x_2881_; 
v___x_2881_ = lean_nat_dec_le(v___x_2878_, v___x_2878_);
if (v___x_2881_ == 0)
{
if (v___x_2880_ == 0)
{
v_a_2856_ = v___x_2879_;
goto v___jp_2855_;
}
else
{
size_t v___x_2882_; size_t v___x_2883_; lean_object* v___x_2884_; 
v___x_2882_ = ((size_t)0ULL);
v___x_2883_ = lean_usize_of_nat(v___x_2878_);
lean_inc_ref(v_a_2830_);
v___x_2884_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(v_a_2830_, v_snd_2853_, v___x_2882_, v___x_2883_, v___x_2879_, v___y_2837_, v___y_2838_, v___y_2839_, v___y_2840_, v___y_2841_, v___y_2842_);
v___y_2867_ = v___x_2884_;
goto v___jp_2866_;
}
}
else
{
size_t v___x_2885_; size_t v___x_2886_; lean_object* v___x_2887_; 
v___x_2885_ = ((size_t)0ULL);
v___x_2886_ = lean_usize_of_nat(v___x_2878_);
lean_inc_ref(v_a_2830_);
v___x_2887_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(v_a_2830_, v_snd_2853_, v___x_2885_, v___x_2886_, v___x_2879_, v___y_2837_, v___y_2838_, v___y_2839_, v___y_2840_, v___y_2841_, v___y_2842_);
v___y_2867_ = v___x_2887_;
goto v___jp_2866_;
}
}
v___jp_2855_:
{
lean_object* v___x_2857_; lean_object* v___x_2858_; uint8_t v___x_2859_; 
v___x_2857_ = lean_array_get_size(v_a_2856_);
v___x_2858_ = lean_unsigned_to_nat(0u);
v___x_2859_ = lean_nat_dec_eq(v___x_2857_, v___x_2858_);
if (v___x_2859_ == 0)
{
lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; uint8_t v___x_2863_; lean_object* v___x_2864_; lean_object* v___x_2865_; 
v___x_2860_ = lean_box(0);
v___x_2861_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___closed__0));
v___x_2862_ = lean_string_append(v___x_2861_, v_fst_2852_);
v___x_2863_ = 4;
v___x_2864_ = l_Lean_MessageData_nil;
lean_inc(v_stx_2829_);
v___x_2865_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2829_, v_a_2856_, v___x_2860_, v___x_2862_, v___x_2860_, v___x_2863_, v___x_2864_, v___y_2841_, v___y_2842_);
if (lean_obj_tag(v___x_2865_) == 0)
{
lean_dec_ref_known(v___x_2865_, 1);
v_a_2845_ = v___x_2854_;
goto v___jp_2844_;
}
else
{
lean_dec_ref(v_a_2830_);
lean_dec(v_stx_2829_);
return v___x_2865_;
}
}
else
{
lean_dec_ref(v_a_2856_);
v_a_2845_ = v___x_2854_;
goto v___jp_2844_;
}
}
v___jp_2866_:
{
if (lean_obj_tag(v___y_2867_) == 0)
{
lean_object* v_a_2868_; 
v_a_2868_ = lean_ctor_get(v___y_2867_, 0);
lean_inc(v_a_2868_);
lean_dec_ref_known(v___y_2867_, 1);
v_a_2856_ = v_a_2868_;
goto v___jp_2855_;
}
else
{
lean_object* v_a_2869_; lean_object* v___x_2871_; uint8_t v_isShared_2872_; uint8_t v_isSharedCheck_2876_; 
lean_dec_ref(v_a_2830_);
lean_dec(v_stx_2829_);
v_a_2869_ = lean_ctor_get(v___y_2867_, 0);
v_isSharedCheck_2876_ = !lean_is_exclusive(v___y_2867_);
if (v_isSharedCheck_2876_ == 0)
{
v___x_2871_ = v___y_2867_;
v_isShared_2872_ = v_isSharedCheck_2876_;
goto v_resetjp_2870_;
}
else
{
lean_inc(v_a_2869_);
lean_dec(v___y_2867_);
v___x_2871_ = lean_box(0);
v_isShared_2872_ = v_isSharedCheck_2876_;
goto v_resetjp_2870_;
}
v_resetjp_2870_:
{
lean_object* v___x_2874_; 
if (v_isShared_2872_ == 0)
{
v___x_2874_ = v___x_2871_;
goto v_reusejp_2873_;
}
else
{
lean_object* v_reuseFailAlloc_2875_; 
v_reuseFailAlloc_2875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2875_, 0, v_a_2869_);
v___x_2874_ = v_reuseFailAlloc_2875_;
goto v_reusejp_2873_;
}
v_reusejp_2873_:
{
return v___x_2874_;
}
}
}
}
}
v___jp_2844_:
{
size_t v___x_2846_; size_t v___x_2847_; 
v___x_2846_ = ((size_t)1ULL);
v___x_2847_ = lean_usize_add(v_i_2833_, v___x_2846_);
v_i_2833_ = v___x_2847_;
v_b_2834_ = v_a_2845_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___boxed(lean_object* v_stx_2888_, lean_object* v_a_2889_, lean_object* v_as_2890_, lean_object* v_sz_2891_, lean_object* v_i_2892_, lean_object* v_b_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_, lean_object* v___y_2897_, lean_object* v___y_2898_, lean_object* v___y_2899_, lean_object* v___y_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_){
_start:
{
size_t v_sz_boxed_2903_; size_t v_i_boxed_2904_; lean_object* v_res_2905_; 
v_sz_boxed_2903_ = lean_unbox_usize(v_sz_2891_);
lean_dec(v_sz_2891_);
v_i_boxed_2904_ = lean_unbox_usize(v_i_2892_);
lean_dec(v_i_2892_);
v_res_2905_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2(v_stx_2888_, v_a_2889_, v_as_2890_, v_sz_boxed_2903_, v_i_boxed_2904_, v_b_2893_, v___y_2894_, v___y_2895_, v___y_2896_, v___y_2897_, v___y_2898_, v___y_2899_, v___y_2900_, v___y_2901_);
lean_dec(v___y_2901_);
lean_dec_ref(v___y_2900_);
lean_dec(v___y_2899_);
lean_dec_ref(v___y_2898_);
lean_dec(v___y_2897_);
lean_dec_ref(v___y_2896_);
lean_dec(v___y_2895_);
lean_dec_ref(v___y_2894_);
lean_dec_ref(v_as_2890_);
return v_res_2905_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_2906_, lean_object* v_msgData_2907_, uint8_t v_severity_2908_, uint8_t v_isSilent_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_){
_start:
{
lean_object* v___y_2916_; lean_object* v___y_2917_; lean_object* v___y_2918_; uint8_t v___y_2919_; uint8_t v___y_2920_; lean_object* v___y_2921_; lean_object* v___y_2922_; lean_object* v___y_2923_; lean_object* v___y_2924_; lean_object* v___y_2952_; lean_object* v___y_2953_; uint8_t v___y_2954_; lean_object* v___y_2955_; uint8_t v___y_2956_; lean_object* v___y_2957_; uint8_t v___y_2958_; lean_object* v___y_2959_; lean_object* v___y_2977_; lean_object* v___y_2978_; uint8_t v___y_2979_; uint8_t v___y_2980_; lean_object* v___y_2981_; lean_object* v___y_2982_; uint8_t v___y_2983_; lean_object* v___y_2984_; lean_object* v___y_2988_; lean_object* v___y_2989_; lean_object* v___y_2990_; uint8_t v___y_2991_; lean_object* v___y_2992_; uint8_t v___y_2993_; uint8_t v___y_2994_; uint8_t v___x_2999_; lean_object* v___y_3001_; lean_object* v___y_3002_; lean_object* v___y_3003_; lean_object* v___y_3004_; uint8_t v___y_3005_; uint8_t v___y_3006_; uint8_t v___y_3007_; uint8_t v___y_3009_; uint8_t v___x_3024_; 
v___x_2999_ = 2;
v___x_3024_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2908_, v___x_2999_);
if (v___x_3024_ == 0)
{
v___y_3009_ = v___x_3024_;
goto v___jp_3008_;
}
else
{
uint8_t v___x_3025_; 
lean_inc_ref(v_msgData_2907_);
v___x_3025_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2907_);
v___y_3009_ = v___x_3025_;
goto v___jp_3008_;
}
v___jp_2915_:
{
lean_object* v___x_2925_; lean_object* v_currNamespace_2926_; lean_object* v_openDecls_2927_; lean_object* v_env_2928_; lean_object* v_nextMacroScope_2929_; lean_object* v_ngen_2930_; lean_object* v_auxDeclNGen_2931_; lean_object* v_traceState_2932_; lean_object* v_cache_2933_; lean_object* v_messages_2934_; lean_object* v_infoState_2935_; lean_object* v_snapshotTasks_2936_; lean_object* v___x_2938_; uint8_t v_isShared_2939_; uint8_t v_isSharedCheck_2950_; 
v___x_2925_ = lean_st_ref_take(v___y_2924_);
v_currNamespace_2926_ = lean_ctor_get(v___y_2923_, 6);
v_openDecls_2927_ = lean_ctor_get(v___y_2923_, 7);
v_env_2928_ = lean_ctor_get(v___x_2925_, 0);
v_nextMacroScope_2929_ = lean_ctor_get(v___x_2925_, 1);
v_ngen_2930_ = lean_ctor_get(v___x_2925_, 2);
v_auxDeclNGen_2931_ = lean_ctor_get(v___x_2925_, 3);
v_traceState_2932_ = lean_ctor_get(v___x_2925_, 4);
v_cache_2933_ = lean_ctor_get(v___x_2925_, 5);
v_messages_2934_ = lean_ctor_get(v___x_2925_, 6);
v_infoState_2935_ = lean_ctor_get(v___x_2925_, 7);
v_snapshotTasks_2936_ = lean_ctor_get(v___x_2925_, 8);
v_isSharedCheck_2950_ = !lean_is_exclusive(v___x_2925_);
if (v_isSharedCheck_2950_ == 0)
{
v___x_2938_ = v___x_2925_;
v_isShared_2939_ = v_isSharedCheck_2950_;
goto v_resetjp_2937_;
}
else
{
lean_inc(v_snapshotTasks_2936_);
lean_inc(v_infoState_2935_);
lean_inc(v_messages_2934_);
lean_inc(v_cache_2933_);
lean_inc(v_traceState_2932_);
lean_inc(v_auxDeclNGen_2931_);
lean_inc(v_ngen_2930_);
lean_inc(v_nextMacroScope_2929_);
lean_inc(v_env_2928_);
lean_dec(v___x_2925_);
v___x_2938_ = lean_box(0);
v_isShared_2939_ = v_isSharedCheck_2950_;
goto v_resetjp_2937_;
}
v_resetjp_2937_:
{
lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2945_; 
lean_inc(v_openDecls_2927_);
lean_inc(v_currNamespace_2926_);
v___x_2940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2940_, 0, v_currNamespace_2926_);
lean_ctor_set(v___x_2940_, 1, v_openDecls_2927_);
v___x_2941_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2941_, 0, v___x_2940_);
lean_ctor_set(v___x_2941_, 1, v___y_2922_);
lean_inc_ref(v___y_2916_);
lean_inc_ref(v___y_2921_);
v___x_2942_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2942_, 0, v___y_2921_);
lean_ctor_set(v___x_2942_, 1, v___y_2918_);
lean_ctor_set(v___x_2942_, 2, v___y_2917_);
lean_ctor_set(v___x_2942_, 3, v___y_2916_);
lean_ctor_set(v___x_2942_, 4, v___x_2941_);
lean_ctor_set_uint8(v___x_2942_, sizeof(void*)*5, v___y_2920_);
lean_ctor_set_uint8(v___x_2942_, sizeof(void*)*5 + 1, v___y_2919_);
lean_ctor_set_uint8(v___x_2942_, sizeof(void*)*5 + 2, v_isSilent_2909_);
v___x_2943_ = l_Lean_MessageLog_add(v___x_2942_, v_messages_2934_);
if (v_isShared_2939_ == 0)
{
lean_ctor_set(v___x_2938_, 6, v___x_2943_);
v___x_2945_ = v___x_2938_;
goto v_reusejp_2944_;
}
else
{
lean_object* v_reuseFailAlloc_2949_; 
v_reuseFailAlloc_2949_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2949_, 0, v_env_2928_);
lean_ctor_set(v_reuseFailAlloc_2949_, 1, v_nextMacroScope_2929_);
lean_ctor_set(v_reuseFailAlloc_2949_, 2, v_ngen_2930_);
lean_ctor_set(v_reuseFailAlloc_2949_, 3, v_auxDeclNGen_2931_);
lean_ctor_set(v_reuseFailAlloc_2949_, 4, v_traceState_2932_);
lean_ctor_set(v_reuseFailAlloc_2949_, 5, v_cache_2933_);
lean_ctor_set(v_reuseFailAlloc_2949_, 6, v___x_2943_);
lean_ctor_set(v_reuseFailAlloc_2949_, 7, v_infoState_2935_);
lean_ctor_set(v_reuseFailAlloc_2949_, 8, v_snapshotTasks_2936_);
v___x_2945_ = v_reuseFailAlloc_2949_;
goto v_reusejp_2944_;
}
v_reusejp_2944_:
{
lean_object* v___x_2946_; lean_object* v___x_2947_; lean_object* v___x_2948_; 
v___x_2946_ = lean_st_ref_set(v___y_2924_, v___x_2945_);
v___x_2947_ = lean_box(0);
v___x_2948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2948_, 0, v___x_2947_);
return v___x_2948_;
}
}
}
v___jp_2951_:
{
lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v_a_2962_; lean_object* v___x_2964_; uint8_t v_isShared_2965_; uint8_t v_isSharedCheck_2975_; 
v___x_2960_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2907_);
v___x_2961_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(v___x_2960_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_);
v_a_2962_ = lean_ctor_get(v___x_2961_, 0);
v_isSharedCheck_2975_ = !lean_is_exclusive(v___x_2961_);
if (v_isSharedCheck_2975_ == 0)
{
v___x_2964_ = v___x_2961_;
v_isShared_2965_ = v_isSharedCheck_2975_;
goto v_resetjp_2963_;
}
else
{
lean_inc(v_a_2962_);
lean_dec(v___x_2961_);
v___x_2964_ = lean_box(0);
v_isShared_2965_ = v_isSharedCheck_2975_;
goto v_resetjp_2963_;
}
v_resetjp_2963_:
{
lean_object* v___x_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; 
lean_inc_ref_n(v___y_2953_, 2);
v___x_2966_ = l_Lean_FileMap_toPosition(v___y_2953_, v___y_2955_);
lean_dec(v___y_2955_);
v___x_2967_ = l_Lean_FileMap_toPosition(v___y_2953_, v___y_2959_);
lean_dec(v___y_2959_);
v___x_2968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2968_, 0, v___x_2967_);
v___x_2969_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
if (v___y_2958_ == 0)
{
lean_del_object(v___x_2964_);
lean_dec_ref(v___y_2952_);
v___y_2916_ = v___x_2969_;
v___y_2917_ = v___x_2968_;
v___y_2918_ = v___x_2966_;
v___y_2919_ = v___y_2954_;
v___y_2920_ = v___y_2956_;
v___y_2921_ = v___y_2957_;
v___y_2922_ = v_a_2962_;
v___y_2923_ = v___y_2912_;
v___y_2924_ = v___y_2913_;
goto v___jp_2915_;
}
else
{
uint8_t v___x_2970_; 
lean_inc(v_a_2962_);
v___x_2970_ = l_Lean_MessageData_hasTag(v___y_2952_, v_a_2962_);
if (v___x_2970_ == 0)
{
lean_object* v___x_2971_; lean_object* v___x_2973_; 
lean_dec_ref_known(v___x_2968_, 1);
lean_dec_ref(v___x_2966_);
lean_dec(v_a_2962_);
v___x_2971_ = lean_box(0);
if (v_isShared_2965_ == 0)
{
lean_ctor_set(v___x_2964_, 0, v___x_2971_);
v___x_2973_ = v___x_2964_;
goto v_reusejp_2972_;
}
else
{
lean_object* v_reuseFailAlloc_2974_; 
v_reuseFailAlloc_2974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2974_, 0, v___x_2971_);
v___x_2973_ = v_reuseFailAlloc_2974_;
goto v_reusejp_2972_;
}
v_reusejp_2972_:
{
return v___x_2973_;
}
}
else
{
lean_del_object(v___x_2964_);
v___y_2916_ = v___x_2969_;
v___y_2917_ = v___x_2968_;
v___y_2918_ = v___x_2966_;
v___y_2919_ = v___y_2954_;
v___y_2920_ = v___y_2956_;
v___y_2921_ = v___y_2957_;
v___y_2922_ = v_a_2962_;
v___y_2923_ = v___y_2912_;
v___y_2924_ = v___y_2913_;
goto v___jp_2915_;
}
}
}
}
v___jp_2976_:
{
lean_object* v___x_2985_; 
v___x_2985_ = l_Lean_Syntax_getTailPos_x3f(v___y_2981_, v___y_2980_);
lean_dec(v___y_2981_);
if (lean_obj_tag(v___x_2985_) == 0)
{
lean_inc(v___y_2984_);
v___y_2952_ = v___y_2977_;
v___y_2953_ = v___y_2978_;
v___y_2954_ = v___y_2979_;
v___y_2955_ = v___y_2984_;
v___y_2956_ = v___y_2980_;
v___y_2957_ = v___y_2982_;
v___y_2958_ = v___y_2983_;
v___y_2959_ = v___y_2984_;
goto v___jp_2951_;
}
else
{
lean_object* v_val_2986_; 
v_val_2986_ = lean_ctor_get(v___x_2985_, 0);
lean_inc(v_val_2986_);
lean_dec_ref_known(v___x_2985_, 1);
v___y_2952_ = v___y_2977_;
v___y_2953_ = v___y_2978_;
v___y_2954_ = v___y_2979_;
v___y_2955_ = v___y_2984_;
v___y_2956_ = v___y_2980_;
v___y_2957_ = v___y_2982_;
v___y_2958_ = v___y_2983_;
v___y_2959_ = v_val_2986_;
goto v___jp_2951_;
}
}
v___jp_2987_:
{
lean_object* v_ref_2995_; lean_object* v___x_2996_; 
v_ref_2995_ = l_Lean_replaceRef(v_ref_2906_, v___y_2989_);
v___x_2996_ = l_Lean_Syntax_getPos_x3f(v_ref_2995_, v___y_2991_);
if (lean_obj_tag(v___x_2996_) == 0)
{
lean_object* v___x_2997_; 
v___x_2997_ = lean_unsigned_to_nat(0u);
v___y_2977_ = v___y_2988_;
v___y_2978_ = v___y_2990_;
v___y_2979_ = v___y_2994_;
v___y_2980_ = v___y_2991_;
v___y_2981_ = v_ref_2995_;
v___y_2982_ = v___y_2992_;
v___y_2983_ = v___y_2993_;
v___y_2984_ = v___x_2997_;
goto v___jp_2976_;
}
else
{
lean_object* v_val_2998_; 
v_val_2998_ = lean_ctor_get(v___x_2996_, 0);
lean_inc(v_val_2998_);
lean_dec_ref_known(v___x_2996_, 1);
v___y_2977_ = v___y_2988_;
v___y_2978_ = v___y_2990_;
v___y_2979_ = v___y_2994_;
v___y_2980_ = v___y_2991_;
v___y_2981_ = v_ref_2995_;
v___y_2982_ = v___y_2992_;
v___y_2983_ = v___y_2993_;
v___y_2984_ = v_val_2998_;
goto v___jp_2976_;
}
}
v___jp_3000_:
{
if (v___y_3007_ == 0)
{
v___y_2988_ = v___y_3002_;
v___y_2989_ = v___y_3001_;
v___y_2990_ = v___y_3003_;
v___y_2991_ = v___y_3006_;
v___y_2992_ = v___y_3004_;
v___y_2993_ = v___y_3005_;
v___y_2994_ = v_severity_2908_;
goto v___jp_2987_;
}
else
{
v___y_2988_ = v___y_3002_;
v___y_2989_ = v___y_3001_;
v___y_2990_ = v___y_3003_;
v___y_2991_ = v___y_3006_;
v___y_2992_ = v___y_3004_;
v___y_2993_ = v___y_3005_;
v___y_2994_ = v___x_2999_;
goto v___jp_2987_;
}
}
v___jp_3008_:
{
if (v___y_3009_ == 0)
{
lean_object* v_fileName_3010_; lean_object* v_fileMap_3011_; lean_object* v_options_3012_; lean_object* v_ref_3013_; uint8_t v_suppressElabErrors_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; lean_object* v___f_3017_; uint8_t v___x_3018_; uint8_t v___x_3019_; 
v_fileName_3010_ = lean_ctor_get(v___y_2912_, 0);
v_fileMap_3011_ = lean_ctor_get(v___y_2912_, 1);
v_options_3012_ = lean_ctor_get(v___y_2912_, 2);
v_ref_3013_ = lean_ctor_get(v___y_2912_, 5);
v_suppressElabErrors_3014_ = lean_ctor_get_uint8(v___y_2912_, sizeof(void*)*14 + 1);
v___x_3015_ = lean_box(v___y_3009_);
v___x_3016_ = lean_box(v_suppressElabErrors_3014_);
v___f_3017_ = lean_alloc_closure((void*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3017_, 0, v___x_3015_);
lean_closure_set(v___f_3017_, 1, v___x_3016_);
v___x_3018_ = 1;
v___x_3019_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2908_, v___x_3018_);
if (v___x_3019_ == 0)
{
v___y_3001_ = v_ref_3013_;
v___y_3002_ = v___f_3017_;
v___y_3003_ = v_fileMap_3011_;
v___y_3004_ = v_fileName_3010_;
v___y_3005_ = v_suppressElabErrors_3014_;
v___y_3006_ = v___y_3009_;
v___y_3007_ = v___x_3019_;
goto v___jp_3000_;
}
else
{
lean_object* v___x_3020_; uint8_t v___x_3021_; 
v___x_3020_ = l_Lean_warningAsError;
v___x_3021_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(v_options_3012_, v___x_3020_);
v___y_3001_ = v_ref_3013_;
v___y_3002_ = v___f_3017_;
v___y_3003_ = v_fileMap_3011_;
v___y_3004_ = v_fileName_3010_;
v___y_3005_ = v_suppressElabErrors_3014_;
v___y_3006_ = v___y_3009_;
v___y_3007_ = v___x_3021_;
goto v___jp_3000_;
}
}
else
{
lean_object* v___x_3022_; lean_object* v___x_3023_; 
lean_dec_ref(v_msgData_2907_);
v___x_3022_ = lean_box(0);
v___x_3023_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3023_, 0, v___x_3022_);
return v___x_3023_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_3026_, lean_object* v_msgData_3027_, lean_object* v_severity_3028_, lean_object* v_isSilent_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_){
_start:
{
uint8_t v_severity_boxed_3035_; uint8_t v_isSilent_boxed_3036_; lean_object* v_res_3037_; 
v_severity_boxed_3035_ = lean_unbox(v_severity_3028_);
v_isSilent_boxed_3036_ = lean_unbox(v_isSilent_3029_);
v_res_3037_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg(v_ref_3026_, v_msgData_3027_, v_severity_boxed_3035_, v_isSilent_boxed_3036_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_);
lean_dec(v___y_3033_);
lean_dec_ref(v___y_3032_);
lean_dec(v___y_3031_);
lean_dec_ref(v___y_3030_);
lean_dec(v_ref_3026_);
return v_res_3037_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0(lean_object* v_msgData_3038_, uint8_t v_severity_3039_, uint8_t v_isSilent_3040_, lean_object* v___y_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_, lean_object* v___y_3044_, lean_object* v___y_3045_, lean_object* v___y_3046_, lean_object* v___y_3047_, lean_object* v___y_3048_){
_start:
{
lean_object* v_ref_3050_; lean_object* v___x_3051_; 
v_ref_3050_ = lean_ctor_get(v___y_3047_, 5);
v___x_3051_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg(v_ref_3050_, v_msgData_3038_, v_severity_3039_, v_isSilent_3040_, v___y_3045_, v___y_3046_, v___y_3047_, v___y_3048_);
return v___x_3051_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0___boxed(lean_object* v_msgData_3052_, lean_object* v_severity_3053_, lean_object* v_isSilent_3054_, lean_object* v___y_3055_, lean_object* v___y_3056_, lean_object* v___y_3057_, lean_object* v___y_3058_, lean_object* v___y_3059_, lean_object* v___y_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_){
_start:
{
uint8_t v_severity_boxed_3064_; uint8_t v_isSilent_boxed_3065_; lean_object* v_res_3066_; 
v_severity_boxed_3064_ = lean_unbox(v_severity_3053_);
v_isSilent_boxed_3065_ = lean_unbox(v_isSilent_3054_);
v_res_3066_ = lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0(v_msgData_3052_, v_severity_boxed_3064_, v_isSilent_boxed_3065_, v___y_3055_, v___y_3056_, v___y_3057_, v___y_3058_, v___y_3059_, v___y_3060_, v___y_3061_, v___y_3062_);
lean_dec(v___y_3062_);
lean_dec_ref(v___y_3061_);
lean_dec(v___y_3060_);
lean_dec_ref(v___y_3059_);
lean_dec(v___y_3058_);
lean_dec_ref(v___y_3057_);
lean_dec(v___y_3056_);
lean_dec_ref(v___y_3055_);
return v_res_3066_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0(lean_object* v_msgData_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_){
_start:
{
uint8_t v___x_3077_; uint8_t v___x_3078_; lean_object* v___x_3079_; 
v___x_3077_ = 1;
v___x_3078_ = 0;
v___x_3079_ = lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0(v_msgData_3067_, v___x_3077_, v___x_3078_, v___y_3068_, v___y_3069_, v___y_3070_, v___y_3071_, v___y_3072_, v___y_3073_, v___y_3074_, v___y_3075_);
return v___x_3079_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0___boxed(lean_object* v_msgData_3080_, lean_object* v___y_3081_, lean_object* v___y_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_){
_start:
{
lean_object* v_res_3090_; 
v_res_3090_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0(v_msgData_3080_, v___y_3081_, v___y_3082_, v___y_3083_, v___y_3084_, v___y_3085_, v___y_3086_, v___y_3087_, v___y_3088_);
lean_dec(v___y_3088_);
lean_dec_ref(v___y_3087_);
lean_dec(v___y_3086_);
lean_dec_ref(v___y_3085_);
lean_dec(v___y_3084_);
lean_dec_ref(v___y_3083_);
lean_dec(v___y_3082_);
lean_dec_ref(v___y_3081_);
return v_res_3090_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions(lean_object* v_ss_3091_, lean_object* v_stx_3092_, lean_object* v_s_3093_, lean_object* v_a_3094_, lean_object* v_a_3095_, lean_object* v_a_3096_, lean_object* v_a_3097_, lean_object* v_a_3098_, lean_object* v_a_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_){
_start:
{
lean_object* v_s_3108_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; uint8_t v___x_3164_; 
v_s_3108_ = l_Lean_TSyntax_getString(v_s_3093_);
v___x_3161_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__1));
v___x_3162_ = lean_string_utf8_byte_size(v_s_3108_);
v___x_3163_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2, &lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__2);
v___x_3164_ = lean_nat_dec_le(v___x_3163_, v___x_3162_);
if (v___x_3164_ == 0)
{
goto v___jp_3153_;
}
else
{
lean_object* v___x_3165_; lean_object* v___x_3166_; uint8_t v___x_3167_; 
v___x_3165_ = lean_unsigned_to_nat(0u);
v___x_3166_ = lean_nat_sub(v___x_3162_, v___x_3163_);
v___x_3167_ = lean_string_memcmp(v_s_3108_, v___x_3161_, v___x_3166_, v___x_3165_, v___x_3163_);
lean_dec(v___x_3166_);
if (v___x_3167_ == 0)
{
goto v___jp_3153_;
}
else
{
goto v___jp_3109_;
}
}
v___jp_3103_:
{
lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; 
v___x_3104_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(v_ss_3091_);
v___x_3105_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3105_, 0, v___x_3104_);
v___x_3106_ = l_Lean_MessageData_ofFormat(v___x_3105_);
v___x_3107_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0(v___x_3106_, v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_, v_a_3099_, v_a_3100_, v_a_3101_);
return v___x_3107_;
}
v___jp_3109_:
{
lean_object* v___x_3110_; 
v___x_3110_ = l_Lean_Elab_Tactic_getMainTarget(v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_, v_a_3099_, v_a_3100_, v_a_3101_);
if (lean_obj_tag(v___x_3110_) == 0)
{
lean_object* v_a_3111_; lean_object* v_queryNum_3112_; lean_object* v___x_3113_; 
v_a_3111_ = lean_ctor_get(v___x_3110_, 0);
lean_inc(v_a_3111_);
lean_dec_ref_known(v___x_3110_, 1);
v_queryNum_3112_ = lean_ctor_get(v_ss_3091_, 4);
lean_inc_ref(v_queryNum_3112_);
lean_inc(v_a_3101_);
lean_inc_ref(v_a_3100_);
v___x_3113_ = lean_apply_3(v_queryNum_3112_, v_a_3100_, v_a_3101_, lean_box(0));
if (lean_obj_tag(v___x_3113_) == 0)
{
lean_object* v_a_3114_; lean_object* v___x_3115_; 
v_a_3114_ = lean_ctor_get(v___x_3113_, 0);
lean_inc(v_a_3114_);
lean_dec_ref_known(v___x_3113_, 1);
v___x_3115_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_getTacticSuggestionGroups(v_ss_3091_, v_s_3108_, v_a_3114_, v_a_3098_, v_a_3099_, v_a_3100_, v_a_3101_);
if (lean_obj_tag(v___x_3115_) == 0)
{
lean_object* v_a_3116_; lean_object* v___x_3117_; size_t v_sz_3118_; size_t v___x_3119_; lean_object* v___x_3120_; 
v_a_3116_ = lean_ctor_get(v___x_3115_, 0);
lean_inc(v_a_3116_);
lean_dec_ref_known(v___x_3115_, 1);
v___x_3117_ = lean_box(0);
v_sz_3118_ = lean_array_size(v_a_3116_);
v___x_3119_ = ((size_t)0ULL);
v___x_3120_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2(v_stx_3092_, v_a_3111_, v_a_3116_, v_sz_3118_, v___x_3119_, v___x_3117_, v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_, v_a_3099_, v_a_3100_, v_a_3101_);
lean_dec(v_a_3116_);
if (lean_obj_tag(v___x_3120_) == 0)
{
lean_object* v___x_3122_; uint8_t v_isShared_3123_; uint8_t v_isSharedCheck_3127_; 
v_isSharedCheck_3127_ = !lean_is_exclusive(v___x_3120_);
if (v_isSharedCheck_3127_ == 0)
{
lean_object* v_unused_3128_; 
v_unused_3128_ = lean_ctor_get(v___x_3120_, 0);
lean_dec(v_unused_3128_);
v___x_3122_ = v___x_3120_;
v_isShared_3123_ = v_isSharedCheck_3127_;
goto v_resetjp_3121_;
}
else
{
lean_dec(v___x_3120_);
v___x_3122_ = lean_box(0);
v_isShared_3123_ = v_isSharedCheck_3127_;
goto v_resetjp_3121_;
}
v_resetjp_3121_:
{
lean_object* v___x_3125_; 
if (v_isShared_3123_ == 0)
{
lean_ctor_set(v___x_3122_, 0, v___x_3117_);
v___x_3125_ = v___x_3122_;
goto v_reusejp_3124_;
}
else
{
lean_object* v_reuseFailAlloc_3126_; 
v_reuseFailAlloc_3126_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3126_, 0, v___x_3117_);
v___x_3125_ = v_reuseFailAlloc_3126_;
goto v_reusejp_3124_;
}
v_reusejp_3124_:
{
return v___x_3125_;
}
}
}
else
{
return v___x_3120_;
}
}
else
{
lean_object* v_a_3129_; lean_object* v___x_3131_; uint8_t v_isShared_3132_; uint8_t v_isSharedCheck_3136_; 
lean_dec(v_a_3111_);
lean_dec(v_stx_3092_);
v_a_3129_ = lean_ctor_get(v___x_3115_, 0);
v_isSharedCheck_3136_ = !lean_is_exclusive(v___x_3115_);
if (v_isSharedCheck_3136_ == 0)
{
v___x_3131_ = v___x_3115_;
v_isShared_3132_ = v_isSharedCheck_3136_;
goto v_resetjp_3130_;
}
else
{
lean_inc(v_a_3129_);
lean_dec(v___x_3115_);
v___x_3131_ = lean_box(0);
v_isShared_3132_ = v_isSharedCheck_3136_;
goto v_resetjp_3130_;
}
v_resetjp_3130_:
{
lean_object* v___x_3134_; 
if (v_isShared_3132_ == 0)
{
v___x_3134_ = v___x_3131_;
goto v_reusejp_3133_;
}
else
{
lean_object* v_reuseFailAlloc_3135_; 
v_reuseFailAlloc_3135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3135_, 0, v_a_3129_);
v___x_3134_ = v_reuseFailAlloc_3135_;
goto v_reusejp_3133_;
}
v_reusejp_3133_:
{
return v___x_3134_;
}
}
}
}
else
{
lean_object* v_a_3137_; lean_object* v___x_3139_; uint8_t v_isShared_3140_; uint8_t v_isSharedCheck_3144_; 
lean_dec(v_a_3111_);
lean_dec_ref(v_s_3108_);
lean_dec(v_stx_3092_);
lean_dec_ref(v_ss_3091_);
v_a_3137_ = lean_ctor_get(v___x_3113_, 0);
v_isSharedCheck_3144_ = !lean_is_exclusive(v___x_3113_);
if (v_isSharedCheck_3144_ == 0)
{
v___x_3139_ = v___x_3113_;
v_isShared_3140_ = v_isSharedCheck_3144_;
goto v_resetjp_3138_;
}
else
{
lean_inc(v_a_3137_);
lean_dec(v___x_3113_);
v___x_3139_ = lean_box(0);
v_isShared_3140_ = v_isSharedCheck_3144_;
goto v_resetjp_3138_;
}
v_resetjp_3138_:
{
lean_object* v___x_3142_; 
if (v_isShared_3140_ == 0)
{
v___x_3142_ = v___x_3139_;
goto v_reusejp_3141_;
}
else
{
lean_object* v_reuseFailAlloc_3143_; 
v_reuseFailAlloc_3143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3143_, 0, v_a_3137_);
v___x_3142_ = v_reuseFailAlloc_3143_;
goto v_reusejp_3141_;
}
v_reusejp_3141_:
{
return v___x_3142_;
}
}
}
}
else
{
lean_object* v_a_3145_; lean_object* v___x_3147_; uint8_t v_isShared_3148_; uint8_t v_isSharedCheck_3152_; 
lean_dec_ref(v_s_3108_);
lean_dec(v_stx_3092_);
lean_dec_ref(v_ss_3091_);
v_a_3145_ = lean_ctor_get(v___x_3110_, 0);
v_isSharedCheck_3152_ = !lean_is_exclusive(v___x_3110_);
if (v_isSharedCheck_3152_ == 0)
{
v___x_3147_ = v___x_3110_;
v_isShared_3148_ = v_isSharedCheck_3152_;
goto v_resetjp_3146_;
}
else
{
lean_inc(v_a_3145_);
lean_dec(v___x_3110_);
v___x_3147_ = lean_box(0);
v_isShared_3148_ = v_isSharedCheck_3152_;
goto v_resetjp_3146_;
}
v_resetjp_3146_:
{
lean_object* v___x_3150_; 
if (v_isShared_3148_ == 0)
{
v___x_3150_ = v___x_3147_;
goto v_reusejp_3149_;
}
else
{
lean_object* v_reuseFailAlloc_3151_; 
v_reuseFailAlloc_3151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3151_, 0, v_a_3145_);
v___x_3150_ = v_reuseFailAlloc_3151_;
goto v_reusejp_3149_;
}
v_reusejp_3149_:
{
return v___x_3150_;
}
}
}
}
v___jp_3153_:
{
lean_object* v___x_3154_; lean_object* v___x_3155_; lean_object* v___x_3156_; uint8_t v___x_3157_; 
v___x_3154_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__0));
v___x_3155_ = lean_string_utf8_byte_size(v_s_3108_);
v___x_3156_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1, &lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions___closed__1);
v___x_3157_ = lean_nat_dec_le(v___x_3156_, v___x_3155_);
if (v___x_3157_ == 0)
{
lean_dec_ref(v_s_3108_);
lean_dec(v_stx_3092_);
goto v___jp_3103_;
}
else
{
lean_object* v___x_3158_; lean_object* v___x_3159_; uint8_t v___x_3160_; 
v___x_3158_ = lean_unsigned_to_nat(0u);
v___x_3159_ = lean_nat_sub(v___x_3155_, v___x_3156_);
v___x_3160_ = lean_string_memcmp(v_s_3108_, v___x_3154_, v___x_3159_, v___x_3158_, v___x_3156_);
lean_dec(v___x_3159_);
if (v___x_3160_ == 0)
{
lean_dec_ref(v_s_3108_);
lean_dec(v_stx_3092_);
goto v___jp_3103_;
}
else
{
goto v___jp_3109_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions___boxed(lean_object* v_ss_3168_, lean_object* v_stx_3169_, lean_object* v_s_3170_, lean_object* v_a_3171_, lean_object* v_a_3172_, lean_object* v_a_3173_, lean_object* v_a_3174_, lean_object* v_a_3175_, lean_object* v_a_3176_, lean_object* v_a_3177_, lean_object* v_a_3178_, lean_object* v_a_3179_){
_start:
{
lean_object* v_res_3180_; 
v_res_3180_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions(v_ss_3168_, v_stx_3169_, v_s_3170_, v_a_3171_, v_a_3172_, v_a_3173_, v_a_3174_, v_a_3175_, v_a_3176_, v_a_3177_, v_a_3178_);
lean_dec(v_a_3178_);
lean_dec_ref(v_a_3177_);
lean_dec(v_a_3176_);
lean_dec_ref(v_a_3175_);
lean_dec(v_a_3174_);
lean_dec_ref(v_a_3173_);
lean_dec(v_a_3172_);
lean_dec_ref(v_a_3171_);
lean_dec(v_s_3170_);
return v_res_3180_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1(lean_object* v_a_3181_, lean_object* v_as_3182_, size_t v_i_3183_, size_t v_stop_3184_, lean_object* v_b_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_){
_start:
{
lean_object* v___x_3195_; 
v___x_3195_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(v_a_3181_, v_as_3182_, v_i_3183_, v_stop_3184_, v_b_3185_, v___y_3188_, v___y_3189_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_);
return v___x_3195_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___boxed(lean_object* v_a_3196_, lean_object* v_as_3197_, lean_object* v_i_3198_, lean_object* v_stop_3199_, lean_object* v_b_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_, lean_object* v___y_3206_, lean_object* v___y_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_){
_start:
{
size_t v_i_boxed_3210_; size_t v_stop_boxed_3211_; lean_object* v_res_3212_; 
v_i_boxed_3210_ = lean_unbox_usize(v_i_3198_);
lean_dec(v_i_3198_);
v_stop_boxed_3211_ = lean_unbox_usize(v_stop_3199_);
lean_dec(v_stop_3199_);
v_res_3212_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1(v_a_3196_, v_as_3197_, v_i_boxed_3210_, v_stop_boxed_3211_, v_b_3200_, v___y_3201_, v___y_3202_, v___y_3203_, v___y_3204_, v___y_3205_, v___y_3206_, v___y_3207_, v___y_3208_);
lean_dec(v___y_3208_);
lean_dec_ref(v___y_3207_);
lean_dec(v___y_3206_);
lean_dec_ref(v___y_3205_);
lean_dec(v___y_3204_);
lean_dec_ref(v___y_3203_);
lean_dec(v___y_3202_);
lean_dec_ref(v___y_3201_);
lean_dec_ref(v_as_3197_);
return v_res_3212_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1(lean_object* v_ref_3213_, lean_object* v_msgData_3214_, uint8_t v_severity_3215_, uint8_t v_isSilent_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_, lean_object* v___y_3221_, lean_object* v___y_3222_, lean_object* v___y_3223_, lean_object* v___y_3224_){
_start:
{
lean_object* v___x_3226_; 
v___x_3226_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___redArg(v_ref_3213_, v_msgData_3214_, v_severity_3215_, v_isSilent_3216_, v___y_3221_, v___y_3222_, v___y_3223_, v___y_3224_);
return v___x_3226_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1___boxed(lean_object* v_ref_3227_, lean_object* v_msgData_3228_, lean_object* v_severity_3229_, lean_object* v_isSilent_3230_, lean_object* v___y_3231_, lean_object* v___y_3232_, lean_object* v___y_3233_, lean_object* v___y_3234_, lean_object* v___y_3235_, lean_object* v___y_3236_, lean_object* v___y_3237_, lean_object* v___y_3238_, lean_object* v___y_3239_){
_start:
{
uint8_t v_severity_boxed_3240_; uint8_t v_isSilent_boxed_3241_; lean_object* v_res_3242_; 
v_severity_boxed_3240_ = lean_unbox(v_severity_3229_);
v_isSilent_boxed_3241_ = lean_unbox(v_isSilent_3230_);
v_res_3242_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0_spec__0_spec__1(v_ref_3227_, v_msgData_3228_, v_severity_boxed_3240_, v_isSilent_boxed_3241_, v___y_3231_, v___y_3232_, v___y_3233_, v___y_3234_, v___y_3235_, v___y_3236_, v___y_3237_, v___y_3238_);
lean_dec(v___y_3238_);
lean_dec_ref(v___y_3237_);
lean_dec(v___y_3236_);
lean_dec_ref(v___y_3235_);
lean_dec(v___y_3234_);
lean_dec_ref(v___y_3233_);
lean_dec(v___y_3232_);
lean_dec_ref(v___y_3231_);
lean_dec(v_ref_3227_);
return v_res_3242_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3281_; 
v___x_3279_ = lean_box(0);
v___x_3280_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3281_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3281_, 0, v___x_3280_);
lean_ctor_set(v___x_3281_, 1, v___x_3279_);
return v___x_3281_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg(){
_start:
{
lean_object* v___x_3283_; lean_object* v___x_3284_; 
v___x_3283_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0, &lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0_once, _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0);
v___x_3284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3284_, 0, v___x_3283_);
return v___x_3284_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___boxed(lean_object* v___y_3285_){
_start:
{
lean_object* v_res_3286_; 
v_res_3286_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
return v_res_3286_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0(lean_object* v_00_u03b1_3287_, lean_object* v___y_3288_, lean_object* v___y_3289_){
_start:
{
lean_object* v___x_3291_; 
v___x_3291_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
return v___x_3291_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___boxed(lean_object* v_00_u03b1_3292_, lean_object* v___y_3293_, lean_object* v___y_3294_, lean_object* v___y_3295_){
_start:
{
lean_object* v_res_3296_; 
v_res_3296_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0(v_00_u03b1_3292_, v___y_3293_, v___y_3294_);
lean_dec(v___y_3294_);
lean_dec_ref(v___y_3293_);
return v_res_3296_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_3297_; 
v___x_3297_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3297_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_3298_; lean_object* v___x_3299_; 
v___x_3298_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__0, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__0_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__0);
v___x_3299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3299_, 0, v___x_3298_);
return v___x_3299_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_3300_; lean_object* v___x_3301_; lean_object* v___x_3302_; 
v___x_3300_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1);
v___x_3301_ = lean_unsigned_to_nat(0u);
v___x_3302_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3302_, 0, v___x_3301_);
lean_ctor_set(v___x_3302_, 1, v___x_3301_);
lean_ctor_set(v___x_3302_, 2, v___x_3301_);
lean_ctor_set(v___x_3302_, 3, v___x_3301_);
lean_ctor_set(v___x_3302_, 4, v___x_3300_);
lean_ctor_set(v___x_3302_, 5, v___x_3300_);
lean_ctor_set(v___x_3302_, 6, v___x_3300_);
lean_ctor_set(v___x_3302_, 7, v___x_3300_);
lean_ctor_set(v___x_3302_, 8, v___x_3300_);
lean_ctor_set(v___x_3302_, 9, v___x_3300_);
return v___x_3302_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_3303_; lean_object* v___x_3304_; lean_object* v___x_3305_; 
v___x_3303_ = lean_unsigned_to_nat(32u);
v___x_3304_ = lean_mk_empty_array_with_capacity(v___x_3303_);
v___x_3305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3305_, 0, v___x_3304_);
return v___x_3305_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_3306_; lean_object* v___x_3307_; lean_object* v___x_3308_; lean_object* v___x_3309_; lean_object* v___x_3310_; lean_object* v___x_3311_; 
v___x_3306_ = ((size_t)5ULL);
v___x_3307_ = lean_unsigned_to_nat(0u);
v___x_3308_ = lean_unsigned_to_nat(32u);
v___x_3309_ = lean_mk_empty_array_with_capacity(v___x_3308_);
v___x_3310_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__3, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__3_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__3);
v___x_3311_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3311_, 0, v___x_3310_);
lean_ctor_set(v___x_3311_, 1, v___x_3309_);
lean_ctor_set(v___x_3311_, 2, v___x_3307_);
lean_ctor_set(v___x_3311_, 3, v___x_3307_);
lean_ctor_set_usize(v___x_3311_, 4, v___x_3306_);
return v___x_3311_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_3312_; lean_object* v___x_3313_; lean_object* v___x_3314_; lean_object* v___x_3315_; 
v___x_3312_ = lean_box(1);
v___x_3313_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__4, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__4_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__4);
v___x_3314_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__1);
v___x_3315_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3315_, 0, v___x_3314_);
lean_ctor_set(v___x_3315_, 1, v___x_3313_);
lean_ctor_set(v___x_3315_, 2, v___x_3312_);
return v___x_3315_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg(lean_object* v_msgData_3316_, lean_object* v___y_3317_){
_start:
{
lean_object* v___x_3319_; lean_object* v_env_3320_; lean_object* v___x_3321_; lean_object* v_scopes_3322_; lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v_opts_3325_; lean_object* v___x_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; 
v___x_3319_ = lean_st_ref_get(v___y_3317_);
v_env_3320_ = lean_ctor_get(v___x_3319_, 0);
lean_inc_ref(v_env_3320_);
lean_dec(v___x_3319_);
v___x_3321_ = lean_st_ref_get(v___y_3317_);
v_scopes_3322_ = lean_ctor_get(v___x_3321_, 2);
lean_inc(v_scopes_3322_);
lean_dec(v___x_3321_);
v___x_3323_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3324_ = l_List_head_x21___redArg(v___x_3323_, v_scopes_3322_);
lean_dec(v_scopes_3322_);
v_opts_3325_ = lean_ctor_get(v___x_3324_, 1);
lean_inc_ref(v_opts_3325_);
lean_dec(v___x_3324_);
v___x_3326_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__2, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__2_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__2);
v___x_3327_ = lean_obj_once(&lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__5, &lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__5_once, _init_lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___closed__5);
v___x_3328_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3328_, 0, v_env_3320_);
lean_ctor_set(v___x_3328_, 1, v___x_3326_);
lean_ctor_set(v___x_3328_, 2, v___x_3327_);
lean_ctor_set(v___x_3328_, 3, v_opts_3325_);
v___x_3329_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_3329_, 0, v___x_3328_);
lean_ctor_set(v___x_3329_, 1, v_msgData_3316_);
v___x_3330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3330_, 0, v___x_3329_);
return v___x_3330_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_3331_, lean_object* v___y_3332_, lean_object* v___y_3333_){
_start:
{
lean_object* v_res_3334_; 
v_res_3334_ = lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg(v_msgData_3331_, v___y_3332_);
lean_dec(v___y_3332_);
return v_res_3334_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___lam__0(uint8_t v___y_3335_, uint8_t v_suppressElabErrors_3336_, lean_object* v_x_3337_){
_start:
{
if (lean_obj_tag(v_x_3337_) == 1)
{
lean_object* v_pre_3338_; 
v_pre_3338_ = lean_ctor_get(v_x_3337_, 0);
if (lean_obj_tag(v_pre_3338_) == 0)
{
lean_object* v_str_3339_; lean_object* v___x_3340_; uint8_t v___x_3341_; 
v_str_3339_ = lean_ctor_get(v_x_3337_, 1);
v___x_3340_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1___redArg___lam__0___closed__7));
v___x_3341_ = lean_string_dec_eq(v_str_3339_, v___x_3340_);
if (v___x_3341_ == 0)
{
return v___y_3335_;
}
else
{
return v_suppressElabErrors_3336_;
}
}
else
{
return v___y_3335_;
}
}
else
{
return v___y_3335_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___lam__0___boxed(lean_object* v___y_3342_, lean_object* v_suppressElabErrors_3343_, lean_object* v_x_3344_){
_start:
{
uint8_t v___y_2737__boxed_3345_; uint8_t v_suppressElabErrors_boxed_3346_; uint8_t v_res_3347_; lean_object* v_r_3348_; 
v___y_2737__boxed_3345_ = lean_unbox(v___y_3342_);
v_suppressElabErrors_boxed_3346_ = lean_unbox(v_suppressElabErrors_3343_);
v_res_3347_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___lam__0(v___y_2737__boxed_3345_, v_suppressElabErrors_boxed_3346_, v_x_3344_);
lean_dec(v_x_3344_);
v_r_3348_ = lean_box(v_res_3347_);
return v_r_3348_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2(lean_object* v_ref_3349_, lean_object* v_msgData_3350_, uint8_t v_severity_3351_, uint8_t v_isSilent_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_){
_start:
{
uint8_t v___y_3357_; lean_object* v___y_3358_; uint8_t v___y_3359_; lean_object* v___y_3360_; lean_object* v___y_3361_; lean_object* v___y_3362_; lean_object* v___y_3363_; lean_object* v___y_3364_; uint8_t v___y_3421_; uint8_t v___y_3422_; uint8_t v___y_3423_; lean_object* v___y_3424_; lean_object* v___y_3425_; uint8_t v___y_3449_; uint8_t v___y_3450_; uint8_t v___y_3451_; lean_object* v___y_3452_; lean_object* v___y_3453_; uint8_t v___y_3457_; uint8_t v___y_3458_; uint8_t v___y_3459_; uint8_t v___x_3474_; uint8_t v___y_3476_; uint8_t v___y_3477_; uint8_t v___y_3478_; uint8_t v___y_3480_; uint8_t v___x_3492_; 
v___x_3474_ = 2;
v___x_3492_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3351_, v___x_3474_);
if (v___x_3492_ == 0)
{
v___y_3480_ = v___x_3492_;
goto v___jp_3479_;
}
else
{
uint8_t v___x_3493_; 
lean_inc_ref(v_msgData_3350_);
v___x_3493_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_3350_);
v___y_3480_ = v___x_3493_;
goto v___jp_3479_;
}
v___jp_3356_:
{
lean_object* v___x_3365_; 
v___x_3365_ = l_Lean_Elab_Command_getScope___redArg(v___y_3364_);
if (lean_obj_tag(v___x_3365_) == 0)
{
lean_object* v_a_3366_; lean_object* v___x_3367_; 
v_a_3366_ = lean_ctor_get(v___x_3365_, 0);
lean_inc(v_a_3366_);
lean_dec_ref_known(v___x_3365_, 1);
v___x_3367_ = l_Lean_Elab_Command_getScope___redArg(v___y_3364_);
if (lean_obj_tag(v___x_3367_) == 0)
{
lean_object* v_a_3368_; lean_object* v___x_3370_; uint8_t v_isShared_3371_; uint8_t v_isSharedCheck_3403_; 
v_a_3368_ = lean_ctor_get(v___x_3367_, 0);
v_isSharedCheck_3403_ = !lean_is_exclusive(v___x_3367_);
if (v_isSharedCheck_3403_ == 0)
{
v___x_3370_ = v___x_3367_;
v_isShared_3371_ = v_isSharedCheck_3403_;
goto v_resetjp_3369_;
}
else
{
lean_inc(v_a_3368_);
lean_dec(v___x_3367_);
v___x_3370_ = lean_box(0);
v_isShared_3371_ = v_isSharedCheck_3403_;
goto v_resetjp_3369_;
}
v_resetjp_3369_:
{
lean_object* v___x_3372_; lean_object* v_currNamespace_3373_; lean_object* v_openDecls_3374_; lean_object* v_env_3375_; lean_object* v_messages_3376_; lean_object* v_scopes_3377_; lean_object* v_usedQuotCtxts_3378_; lean_object* v_nextMacroScope_3379_; lean_object* v_maxRecDepth_3380_; lean_object* v_ngen_3381_; lean_object* v_auxDeclNGen_3382_; lean_object* v_infoState_3383_; lean_object* v_traceState_3384_; lean_object* v_snapshotTasks_3385_; lean_object* v_prevLinterStates_3386_; lean_object* v___x_3388_; uint8_t v_isShared_3389_; uint8_t v_isSharedCheck_3402_; 
v___x_3372_ = lean_st_ref_take(v___y_3364_);
v_currNamespace_3373_ = lean_ctor_get(v_a_3366_, 2);
lean_inc(v_currNamespace_3373_);
lean_dec(v_a_3366_);
v_openDecls_3374_ = lean_ctor_get(v_a_3368_, 3);
lean_inc(v_openDecls_3374_);
lean_dec(v_a_3368_);
v_env_3375_ = lean_ctor_get(v___x_3372_, 0);
v_messages_3376_ = lean_ctor_get(v___x_3372_, 1);
v_scopes_3377_ = lean_ctor_get(v___x_3372_, 2);
v_usedQuotCtxts_3378_ = lean_ctor_get(v___x_3372_, 3);
v_nextMacroScope_3379_ = lean_ctor_get(v___x_3372_, 4);
v_maxRecDepth_3380_ = lean_ctor_get(v___x_3372_, 5);
v_ngen_3381_ = lean_ctor_get(v___x_3372_, 6);
v_auxDeclNGen_3382_ = lean_ctor_get(v___x_3372_, 7);
v_infoState_3383_ = lean_ctor_get(v___x_3372_, 8);
v_traceState_3384_ = lean_ctor_get(v___x_3372_, 9);
v_snapshotTasks_3385_ = lean_ctor_get(v___x_3372_, 10);
v_prevLinterStates_3386_ = lean_ctor_get(v___x_3372_, 11);
v_isSharedCheck_3402_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3402_ == 0)
{
v___x_3388_ = v___x_3372_;
v_isShared_3389_ = v_isSharedCheck_3402_;
goto v_resetjp_3387_;
}
else
{
lean_inc(v_prevLinterStates_3386_);
lean_inc(v_snapshotTasks_3385_);
lean_inc(v_traceState_3384_);
lean_inc(v_infoState_3383_);
lean_inc(v_auxDeclNGen_3382_);
lean_inc(v_ngen_3381_);
lean_inc(v_maxRecDepth_3380_);
lean_inc(v_nextMacroScope_3379_);
lean_inc(v_usedQuotCtxts_3378_);
lean_inc(v_scopes_3377_);
lean_inc(v_messages_3376_);
lean_inc(v_env_3375_);
lean_dec(v___x_3372_);
v___x_3388_ = lean_box(0);
v_isShared_3389_ = v_isSharedCheck_3402_;
goto v_resetjp_3387_;
}
v_resetjp_3387_:
{
lean_object* v___x_3390_; lean_object* v___x_3391_; lean_object* v___x_3392_; lean_object* v___x_3393_; lean_object* v___x_3395_; 
v___x_3390_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3390_, 0, v_currNamespace_3373_);
lean_ctor_set(v___x_3390_, 1, v_openDecls_3374_);
v___x_3391_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3391_, 0, v___x_3390_);
lean_ctor_set(v___x_3391_, 1, v___y_3362_);
lean_inc_ref(v___y_3361_);
lean_inc_ref(v___y_3363_);
v___x_3392_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_3392_, 0, v___y_3363_);
lean_ctor_set(v___x_3392_, 1, v___y_3360_);
lean_ctor_set(v___x_3392_, 2, v___y_3358_);
lean_ctor_set(v___x_3392_, 3, v___y_3361_);
lean_ctor_set(v___x_3392_, 4, v___x_3391_);
lean_ctor_set_uint8(v___x_3392_, sizeof(void*)*5, v___y_3357_);
lean_ctor_set_uint8(v___x_3392_, sizeof(void*)*5 + 1, v___y_3359_);
lean_ctor_set_uint8(v___x_3392_, sizeof(void*)*5 + 2, v_isSilent_3352_);
v___x_3393_ = l_Lean_MessageLog_add(v___x_3392_, v_messages_3376_);
if (v_isShared_3389_ == 0)
{
lean_ctor_set(v___x_3388_, 1, v___x_3393_);
v___x_3395_ = v___x_3388_;
goto v_reusejp_3394_;
}
else
{
lean_object* v_reuseFailAlloc_3401_; 
v_reuseFailAlloc_3401_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_3401_, 0, v_env_3375_);
lean_ctor_set(v_reuseFailAlloc_3401_, 1, v___x_3393_);
lean_ctor_set(v_reuseFailAlloc_3401_, 2, v_scopes_3377_);
lean_ctor_set(v_reuseFailAlloc_3401_, 3, v_usedQuotCtxts_3378_);
lean_ctor_set(v_reuseFailAlloc_3401_, 4, v_nextMacroScope_3379_);
lean_ctor_set(v_reuseFailAlloc_3401_, 5, v_maxRecDepth_3380_);
lean_ctor_set(v_reuseFailAlloc_3401_, 6, v_ngen_3381_);
lean_ctor_set(v_reuseFailAlloc_3401_, 7, v_auxDeclNGen_3382_);
lean_ctor_set(v_reuseFailAlloc_3401_, 8, v_infoState_3383_);
lean_ctor_set(v_reuseFailAlloc_3401_, 9, v_traceState_3384_);
lean_ctor_set(v_reuseFailAlloc_3401_, 10, v_snapshotTasks_3385_);
lean_ctor_set(v_reuseFailAlloc_3401_, 11, v_prevLinterStates_3386_);
v___x_3395_ = v_reuseFailAlloc_3401_;
goto v_reusejp_3394_;
}
v_reusejp_3394_:
{
lean_object* v___x_3396_; lean_object* v___x_3397_; lean_object* v___x_3399_; 
v___x_3396_ = lean_st_ref_set(v___y_3364_, v___x_3395_);
v___x_3397_ = lean_box(0);
if (v_isShared_3371_ == 0)
{
lean_ctor_set(v___x_3370_, 0, v___x_3397_);
v___x_3399_ = v___x_3370_;
goto v_reusejp_3398_;
}
else
{
lean_object* v_reuseFailAlloc_3400_; 
v_reuseFailAlloc_3400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3400_, 0, v___x_3397_);
v___x_3399_ = v_reuseFailAlloc_3400_;
goto v_reusejp_3398_;
}
v_reusejp_3398_:
{
return v___x_3399_;
}
}
}
}
}
else
{
lean_object* v_a_3404_; lean_object* v___x_3406_; uint8_t v_isShared_3407_; uint8_t v_isSharedCheck_3411_; 
lean_dec(v_a_3366_);
lean_dec_ref(v___y_3362_);
lean_dec_ref(v___y_3360_);
lean_dec(v___y_3358_);
v_a_3404_ = lean_ctor_get(v___x_3367_, 0);
v_isSharedCheck_3411_ = !lean_is_exclusive(v___x_3367_);
if (v_isSharedCheck_3411_ == 0)
{
v___x_3406_ = v___x_3367_;
v_isShared_3407_ = v_isSharedCheck_3411_;
goto v_resetjp_3405_;
}
else
{
lean_inc(v_a_3404_);
lean_dec(v___x_3367_);
v___x_3406_ = lean_box(0);
v_isShared_3407_ = v_isSharedCheck_3411_;
goto v_resetjp_3405_;
}
v_resetjp_3405_:
{
lean_object* v___x_3409_; 
if (v_isShared_3407_ == 0)
{
v___x_3409_ = v___x_3406_;
goto v_reusejp_3408_;
}
else
{
lean_object* v_reuseFailAlloc_3410_; 
v_reuseFailAlloc_3410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3410_, 0, v_a_3404_);
v___x_3409_ = v_reuseFailAlloc_3410_;
goto v_reusejp_3408_;
}
v_reusejp_3408_:
{
return v___x_3409_;
}
}
}
}
else
{
lean_object* v_a_3412_; lean_object* v___x_3414_; uint8_t v_isShared_3415_; uint8_t v_isSharedCheck_3419_; 
lean_dec_ref(v___y_3362_);
lean_dec_ref(v___y_3360_);
lean_dec(v___y_3358_);
v_a_3412_ = lean_ctor_get(v___x_3365_, 0);
v_isSharedCheck_3419_ = !lean_is_exclusive(v___x_3365_);
if (v_isSharedCheck_3419_ == 0)
{
v___x_3414_ = v___x_3365_;
v_isShared_3415_ = v_isSharedCheck_3419_;
goto v_resetjp_3413_;
}
else
{
lean_inc(v_a_3412_);
lean_dec(v___x_3365_);
v___x_3414_ = lean_box(0);
v_isShared_3415_ = v_isSharedCheck_3419_;
goto v_resetjp_3413_;
}
v_resetjp_3413_:
{
lean_object* v___x_3417_; 
if (v_isShared_3415_ == 0)
{
v___x_3417_ = v___x_3414_;
goto v_reusejp_3416_;
}
else
{
lean_object* v_reuseFailAlloc_3418_; 
v_reuseFailAlloc_3418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3418_, 0, v_a_3412_);
v___x_3417_ = v_reuseFailAlloc_3418_;
goto v_reusejp_3416_;
}
v_reusejp_3416_:
{
return v___x_3417_;
}
}
}
}
v___jp_3420_:
{
lean_object* v_fileName_3426_; lean_object* v_fileMap_3427_; uint8_t v_suppressElabErrors_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v_a_3431_; lean_object* v___x_3433_; uint8_t v_isShared_3434_; uint8_t v_isSharedCheck_3447_; 
v_fileName_3426_ = lean_ctor_get(v___y_3353_, 0);
v_fileMap_3427_ = lean_ctor_get(v___y_3353_, 1);
v_suppressElabErrors_3428_ = lean_ctor_get_uint8(v___y_3353_, sizeof(void*)*10);
v___x_3429_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_3350_);
v___x_3430_ = lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg(v___x_3429_, v___y_3354_);
v_a_3431_ = lean_ctor_get(v___x_3430_, 0);
v_isSharedCheck_3447_ = !lean_is_exclusive(v___x_3430_);
if (v_isSharedCheck_3447_ == 0)
{
v___x_3433_ = v___x_3430_;
v_isShared_3434_ = v_isSharedCheck_3447_;
goto v_resetjp_3432_;
}
else
{
lean_inc(v_a_3431_);
lean_dec(v___x_3430_);
v___x_3433_ = lean_box(0);
v_isShared_3434_ = v_isSharedCheck_3447_;
goto v_resetjp_3432_;
}
v_resetjp_3432_:
{
lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; 
lean_inc_ref_n(v_fileMap_3427_, 2);
v___x_3435_ = l_Lean_FileMap_toPosition(v_fileMap_3427_, v___y_3424_);
lean_dec(v___y_3424_);
v___x_3436_ = l_Lean_FileMap_toPosition(v_fileMap_3427_, v___y_3425_);
lean_dec(v___y_3425_);
v___x_3437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3437_, 0, v___x_3436_);
v___x_3438_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00LeanSearchClient_SearchResult_ofLeanSearchJson_x3f_spec__1___closed__0));
if (v_suppressElabErrors_3428_ == 0)
{
lean_del_object(v___x_3433_);
v___y_3357_ = v___y_3422_;
v___y_3358_ = v___x_3437_;
v___y_3359_ = v___y_3423_;
v___y_3360_ = v___x_3435_;
v___y_3361_ = v___x_3438_;
v___y_3362_ = v_a_3431_;
v___y_3363_ = v_fileName_3426_;
v___y_3364_ = v___y_3354_;
goto v___jp_3356_;
}
else
{
lean_object* v___x_3439_; lean_object* v___x_3440_; lean_object* v___f_3441_; uint8_t v___x_3442_; 
v___x_3439_ = lean_box(v___y_3421_);
v___x_3440_ = lean_box(v_suppressElabErrors_3428_);
v___f_3441_ = lean_alloc_closure((void*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3441_, 0, v___x_3439_);
lean_closure_set(v___f_3441_, 1, v___x_3440_);
lean_inc(v_a_3431_);
v___x_3442_ = l_Lean_MessageData_hasTag(v___f_3441_, v_a_3431_);
if (v___x_3442_ == 0)
{
lean_object* v___x_3443_; lean_object* v___x_3445_; 
lean_dec_ref_known(v___x_3437_, 1);
lean_dec_ref(v___x_3435_);
lean_dec(v_a_3431_);
v___x_3443_ = lean_box(0);
if (v_isShared_3434_ == 0)
{
lean_ctor_set(v___x_3433_, 0, v___x_3443_);
v___x_3445_ = v___x_3433_;
goto v_reusejp_3444_;
}
else
{
lean_object* v_reuseFailAlloc_3446_; 
v_reuseFailAlloc_3446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3446_, 0, v___x_3443_);
v___x_3445_ = v_reuseFailAlloc_3446_;
goto v_reusejp_3444_;
}
v_reusejp_3444_:
{
return v___x_3445_;
}
}
else
{
lean_del_object(v___x_3433_);
v___y_3357_ = v___y_3422_;
v___y_3358_ = v___x_3437_;
v___y_3359_ = v___y_3423_;
v___y_3360_ = v___x_3435_;
v___y_3361_ = v___x_3438_;
v___y_3362_ = v_a_3431_;
v___y_3363_ = v_fileName_3426_;
v___y_3364_ = v___y_3354_;
goto v___jp_3356_;
}
}
}
}
v___jp_3448_:
{
lean_object* v___x_3454_; 
v___x_3454_ = l_Lean_Syntax_getTailPos_x3f(v___y_3452_, v___y_3450_);
lean_dec(v___y_3452_);
if (lean_obj_tag(v___x_3454_) == 0)
{
lean_inc(v___y_3453_);
v___y_3421_ = v___y_3449_;
v___y_3422_ = v___y_3450_;
v___y_3423_ = v___y_3451_;
v___y_3424_ = v___y_3453_;
v___y_3425_ = v___y_3453_;
goto v___jp_3420_;
}
else
{
lean_object* v_val_3455_; 
v_val_3455_ = lean_ctor_get(v___x_3454_, 0);
lean_inc(v_val_3455_);
lean_dec_ref_known(v___x_3454_, 1);
v___y_3421_ = v___y_3449_;
v___y_3422_ = v___y_3450_;
v___y_3423_ = v___y_3451_;
v___y_3424_ = v___y_3453_;
v___y_3425_ = v_val_3455_;
goto v___jp_3420_;
}
}
v___jp_3456_:
{
lean_object* v___x_3460_; 
v___x_3460_ = l_Lean_Elab_Command_getRef___redArg(v___y_3353_);
if (lean_obj_tag(v___x_3460_) == 0)
{
lean_object* v_a_3461_; lean_object* v_ref_3462_; lean_object* v___x_3463_; 
v_a_3461_ = lean_ctor_get(v___x_3460_, 0);
lean_inc(v_a_3461_);
lean_dec_ref_known(v___x_3460_, 1);
v_ref_3462_ = l_Lean_replaceRef(v_ref_3349_, v_a_3461_);
lean_dec(v_a_3461_);
v___x_3463_ = l_Lean_Syntax_getPos_x3f(v_ref_3462_, v___y_3458_);
if (lean_obj_tag(v___x_3463_) == 0)
{
lean_object* v___x_3464_; 
v___x_3464_ = lean_unsigned_to_nat(0u);
v___y_3449_ = v___y_3457_;
v___y_3450_ = v___y_3458_;
v___y_3451_ = v___y_3459_;
v___y_3452_ = v_ref_3462_;
v___y_3453_ = v___x_3464_;
goto v___jp_3448_;
}
else
{
lean_object* v_val_3465_; 
v_val_3465_ = lean_ctor_get(v___x_3463_, 0);
lean_inc(v_val_3465_);
lean_dec_ref_known(v___x_3463_, 1);
v___y_3449_ = v___y_3457_;
v___y_3450_ = v___y_3458_;
v___y_3451_ = v___y_3459_;
v___y_3452_ = v_ref_3462_;
v___y_3453_ = v_val_3465_;
goto v___jp_3448_;
}
}
else
{
lean_object* v_a_3466_; lean_object* v___x_3468_; uint8_t v_isShared_3469_; uint8_t v_isSharedCheck_3473_; 
lean_dec_ref(v_msgData_3350_);
v_a_3466_ = lean_ctor_get(v___x_3460_, 0);
v_isSharedCheck_3473_ = !lean_is_exclusive(v___x_3460_);
if (v_isSharedCheck_3473_ == 0)
{
v___x_3468_ = v___x_3460_;
v_isShared_3469_ = v_isSharedCheck_3473_;
goto v_resetjp_3467_;
}
else
{
lean_inc(v_a_3466_);
lean_dec(v___x_3460_);
v___x_3468_ = lean_box(0);
v_isShared_3469_ = v_isSharedCheck_3473_;
goto v_resetjp_3467_;
}
v_resetjp_3467_:
{
lean_object* v___x_3471_; 
if (v_isShared_3469_ == 0)
{
v___x_3471_ = v___x_3468_;
goto v_reusejp_3470_;
}
else
{
lean_object* v_reuseFailAlloc_3472_; 
v_reuseFailAlloc_3472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3472_, 0, v_a_3466_);
v___x_3471_ = v_reuseFailAlloc_3472_;
goto v_reusejp_3470_;
}
v_reusejp_3470_:
{
return v___x_3471_;
}
}
}
}
v___jp_3475_:
{
if (v___y_3478_ == 0)
{
v___y_3457_ = v___y_3476_;
v___y_3458_ = v___y_3477_;
v___y_3459_ = v_severity_3351_;
goto v___jp_3456_;
}
else
{
v___y_3457_ = v___y_3476_;
v___y_3458_ = v___y_3477_;
v___y_3459_ = v___x_3474_;
goto v___jp_3456_;
}
}
v___jp_3479_:
{
if (v___y_3480_ == 0)
{
lean_object* v___x_3481_; lean_object* v_scopes_3482_; lean_object* v___x_3483_; lean_object* v___x_3484_; lean_object* v_opts_3485_; uint8_t v___x_3486_; uint8_t v___x_3487_; 
v___x_3481_ = lean_st_ref_get(v___y_3354_);
v_scopes_3482_ = lean_ctor_get(v___x_3481_, 2);
lean_inc(v_scopes_3482_);
lean_dec(v___x_3481_);
v___x_3483_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3484_ = l_List_head_x21___redArg(v___x_3483_, v_scopes_3482_);
lean_dec(v_scopes_3482_);
v_opts_3485_ = lean_ctor_get(v___x_3484_, 1);
lean_inc_ref(v_opts_3485_);
lean_dec(v___x_3484_);
v___x_3486_ = 1;
v___x_3487_ = l_Lean_instBEqMessageSeverity_beq(v_severity_3351_, v___x_3486_);
if (v___x_3487_ == 0)
{
lean_dec_ref(v_opts_3485_);
v___y_3476_ = v___y_3480_;
v___y_3477_ = v___y_3480_;
v___y_3478_ = v___x_3487_;
goto v___jp_3475_;
}
else
{
lean_object* v___x_3488_; uint8_t v___x_3489_; 
v___x_3488_ = l_Lean_warningAsError;
v___x_3489_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(v_opts_3485_, v___x_3488_);
lean_dec_ref(v_opts_3485_);
v___y_3476_ = v___y_3480_;
v___y_3477_ = v___y_3480_;
v___y_3478_ = v___x_3489_;
goto v___jp_3475_;
}
}
else
{
lean_object* v___x_3490_; lean_object* v___x_3491_; 
lean_dec_ref(v_msgData_3350_);
v___x_3490_ = lean_box(0);
v___x_3491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3491_, 0, v___x_3490_);
return v___x_3491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_3494_, lean_object* v_msgData_3495_, lean_object* v_severity_3496_, lean_object* v_isSilent_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_){
_start:
{
uint8_t v_severity_boxed_3501_; uint8_t v_isSilent_boxed_3502_; lean_object* v_res_3503_; 
v_severity_boxed_3501_ = lean_unbox(v_severity_3496_);
v_isSilent_boxed_3502_ = lean_unbox(v_isSilent_3497_);
v_res_3503_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2(v_ref_3494_, v_msgData_3495_, v_severity_boxed_3501_, v_isSilent_boxed_3502_, v___y_3498_, v___y_3499_);
lean_dec(v___y_3499_);
lean_dec_ref(v___y_3498_);
lean_dec(v_ref_3494_);
return v_res_3503_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1(lean_object* v_msgData_3504_, uint8_t v_severity_3505_, uint8_t v_isSilent_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_){
_start:
{
lean_object* v___x_3510_; 
v___x_3510_ = l_Lean_Elab_Command_getRef___redArg(v___y_3507_);
if (lean_obj_tag(v___x_3510_) == 0)
{
lean_object* v_a_3511_; lean_object* v___x_3512_; 
v_a_3511_ = lean_ctor_get(v___x_3510_, 0);
lean_inc(v_a_3511_);
lean_dec_ref_known(v___x_3510_, 1);
v___x_3512_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2(v_a_3511_, v_msgData_3504_, v_severity_3505_, v_isSilent_3506_, v___y_3507_, v___y_3508_);
lean_dec(v_a_3511_);
return v___x_3512_;
}
else
{
lean_object* v_a_3513_; lean_object* v___x_3515_; uint8_t v_isShared_3516_; uint8_t v_isSharedCheck_3520_; 
lean_dec_ref(v_msgData_3504_);
v_a_3513_ = lean_ctor_get(v___x_3510_, 0);
v_isSharedCheck_3520_ = !lean_is_exclusive(v___x_3510_);
if (v_isSharedCheck_3520_ == 0)
{
v___x_3515_ = v___x_3510_;
v_isShared_3516_ = v_isSharedCheck_3520_;
goto v_resetjp_3514_;
}
else
{
lean_inc(v_a_3513_);
lean_dec(v___x_3510_);
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
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1___boxed(lean_object* v_msgData_3521_, lean_object* v_severity_3522_, lean_object* v_isSilent_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_){
_start:
{
uint8_t v_severity_boxed_3527_; uint8_t v_isSilent_boxed_3528_; lean_object* v_res_3529_; 
v_severity_boxed_3527_ = lean_unbox(v_severity_3522_);
v_isSilent_boxed_3528_ = lean_unbox(v_isSilent_3523_);
v_res_3529_ = lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1(v_msgData_3521_, v_severity_boxed_3527_, v_isSilent_boxed_3528_, v___y_3524_, v___y_3525_);
lean_dec(v___y_3525_);
lean_dec_ref(v___y_3524_);
return v_res_3529_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1(lean_object* v_msgData_3530_, lean_object* v___y_3531_, lean_object* v___y_3532_){
_start:
{
uint8_t v___x_3534_; uint8_t v___x_3535_; lean_object* v___x_3536_; 
v___x_3534_ = 1;
v___x_3535_ = 0;
v___x_3536_ = lp_LeanSearchClient_Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1(v_msgData_3530_, v___x_3534_, v___x_3535_, v___y_3531_, v___y_3532_);
return v___x_3536_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1___boxed(lean_object* v_msgData_3537_, lean_object* v___y_3538_, lean_object* v___y_3539_, lean_object* v___y_3540_){
_start:
{
lean_object* v_res_3541_; 
v_res_3541_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1(v_msgData_3537_, v___y_3538_, v___y_3539_);
lean_dec(v___y_3539_);
lean_dec_ref(v___y_3538_);
return v_res_3541_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__0(void){
_start:
{
lean_object* v___x_3542_; lean_object* v___x_3543_; 
v___x_3542_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v___x_3543_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(v___x_3542_);
return v___x_3543_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__1(void){
_start:
{
lean_object* v___x_3544_; lean_object* v___x_3545_; 
v___x_3544_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__0, &lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__0_once, _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__0);
v___x_3545_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3545_, 0, v___x_3544_);
return v___x_3545_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2(void){
_start:
{
lean_object* v___x_3546_; lean_object* v___x_3547_; 
v___x_3546_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__1, &lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__1);
v___x_3547_ = l_Lean_MessageData_ofFormat(v___x_3546_);
return v___x_3547_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl(lean_object* v_stx_3548_, lean_object* v_a_3549_, lean_object* v_a_3550_){
_start:
{
lean_object* v___x_3552_; uint8_t v___x_3553_; 
v___x_3552_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__2));
lean_inc(v_stx_3548_);
v___x_3553_ = l_Lean_Syntax_isOfKind(v_stx_3548_, v___x_3552_);
if (v___x_3553_ == 0)
{
lean_object* v___x_3554_; 
lean_dec(v_stx_3548_);
v___x_3554_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
return v___x_3554_;
}
else
{
lean_object* v___x_3555_; lean_object* v___x_3556_; lean_object* v___x_3557_; uint8_t v___x_3558_; 
v___x_3555_ = lean_unsigned_to_nat(0u);
v___x_3556_ = lean_unsigned_to_nat(1u);
v___x_3557_ = l_Lean_Syntax_getArg(v_stx_3548_, v___x_3556_);
lean_inc(v___x_3557_);
v___x_3558_ = l_Lean_Syntax_matchesNull(v___x_3557_, v___x_3556_);
if (v___x_3558_ == 0)
{
uint8_t v___x_3559_; 
lean_dec(v_stx_3548_);
v___x_3559_ = l_Lean_Syntax_matchesNull(v___x_3557_, v___x_3555_);
if (v___x_3559_ == 0)
{
lean_object* v___x_3560_; 
v___x_3560_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
return v___x_3560_;
}
else
{
lean_object* v___x_3561_; lean_object* v___x_3562_; 
v___x_3561_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2, &lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2);
v___x_3562_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1(v___x_3561_, v_a_3549_, v_a_3550_);
return v___x_3562_;
}
}
else
{
lean_object* v_s_3563_; lean_object* v___x_3564_; lean_object* v___x_3565_; 
v_s_3563_ = l_Lean_Syntax_getArg(v___x_3557_, v___x_3555_);
lean_dec(v___x_3557_);
v___x_3564_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v___x_3565_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions(v___x_3564_, v_stx_3548_, v_s_3563_, v_a_3549_, v_a_3550_);
lean_dec(v_s_3563_);
return v___x_3565_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___boxed(lean_object* v_stx_3566_, lean_object* v_a_3567_, lean_object* v_a_3568_, lean_object* v_a_3569_){
_start:
{
lean_object* v_res_3570_; 
v_res_3570_ = lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl(v_stx_3566_, v_a_3567_, v_a_3568_);
lean_dec(v_a_3568_);
lean_dec_ref(v_a_3567_);
return v_res_3570_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3(lean_object* v_msgData_3571_, lean_object* v___y_3572_, lean_object* v___y_3573_){
_start:
{
lean_object* v___x_3575_; 
v___x_3575_ = lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg(v_msgData_3571_, v___y_3573_);
return v___x_3575_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___boxed(lean_object* v_msgData_3576_, lean_object* v___y_3577_, lean_object* v___y_3578_, lean_object* v___y_3579_){
_start:
{
lean_object* v_res_3580_; 
v_res_3580_ = lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3(v_msgData_3576_, v___y_3577_, v___y_3578_);
lean_dec(v___y_3578_);
lean_dec_ref(v___y_3577_);
return v_res_3580_;
}
}
static lean_object* _init_lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0(void){
_start:
{
lean_object* v___x_3597_; lean_object* v___x_3598_; 
v___x_3597_ = lean_box(1);
v___x_3598_ = l_Lean_MessageData_ofFormat(v___x_3597_);
return v___x_3598_;
}
}
static lean_object* _init_lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_3602_; lean_object* v___x_3603_; 
v___x_3602_ = ((lean_object*)(lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__2));
v___x_3603_ = l_Lean_MessageData_ofFormat(v___x_3602_);
return v___x_3603_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1(lean_object* v_x_3604_, lean_object* v_x_3605_){
_start:
{
if (lean_obj_tag(v_x_3605_) == 0)
{
return v_x_3604_;
}
else
{
lean_object* v_head_3606_; lean_object* v_tail_3607_; lean_object* v___x_3609_; uint8_t v_isShared_3610_; uint8_t v_isSharedCheck_3629_; 
v_head_3606_ = lean_ctor_get(v_x_3605_, 0);
v_tail_3607_ = lean_ctor_get(v_x_3605_, 1);
v_isSharedCheck_3629_ = !lean_is_exclusive(v_x_3605_);
if (v_isSharedCheck_3629_ == 0)
{
v___x_3609_ = v_x_3605_;
v_isShared_3610_ = v_isSharedCheck_3629_;
goto v_resetjp_3608_;
}
else
{
lean_inc(v_tail_3607_);
lean_inc(v_head_3606_);
lean_dec(v_x_3605_);
v___x_3609_ = lean_box(0);
v_isShared_3610_ = v_isSharedCheck_3629_;
goto v_resetjp_3608_;
}
v_resetjp_3608_:
{
lean_object* v_before_3611_; lean_object* v___x_3613_; uint8_t v_isShared_3614_; uint8_t v_isSharedCheck_3627_; 
v_before_3611_ = lean_ctor_get(v_head_3606_, 0);
v_isSharedCheck_3627_ = !lean_is_exclusive(v_head_3606_);
if (v_isSharedCheck_3627_ == 0)
{
lean_object* v_unused_3628_; 
v_unused_3628_ = lean_ctor_get(v_head_3606_, 1);
lean_dec(v_unused_3628_);
v___x_3613_ = v_head_3606_;
v_isShared_3614_ = v_isSharedCheck_3627_;
goto v_resetjp_3612_;
}
else
{
lean_inc(v_before_3611_);
lean_dec(v_head_3606_);
v___x_3613_ = lean_box(0);
v_isShared_3614_ = v_isSharedCheck_3627_;
goto v_resetjp_3612_;
}
v_resetjp_3612_:
{
lean_object* v___x_3615_; lean_object* v___x_3617_; 
v___x_3615_ = lean_obj_once(&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0, &lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0_once, _init_lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0);
if (v_isShared_3614_ == 0)
{
lean_ctor_set_tag(v___x_3613_, 7);
lean_ctor_set(v___x_3613_, 1, v___x_3615_);
lean_ctor_set(v___x_3613_, 0, v_x_3604_);
v___x_3617_ = v___x_3613_;
goto v_reusejp_3616_;
}
else
{
lean_object* v_reuseFailAlloc_3626_; 
v_reuseFailAlloc_3626_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3626_, 0, v_x_3604_);
lean_ctor_set(v_reuseFailAlloc_3626_, 1, v___x_3615_);
v___x_3617_ = v_reuseFailAlloc_3626_;
goto v_reusejp_3616_;
}
v_reusejp_3616_:
{
lean_object* v___x_3618_; lean_object* v___x_3620_; 
v___x_3618_ = lean_obj_once(&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__3, &lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__3_once, _init_lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__3);
if (v_isShared_3610_ == 0)
{
lean_ctor_set_tag(v___x_3609_, 7);
lean_ctor_set(v___x_3609_, 1, v___x_3618_);
lean_ctor_set(v___x_3609_, 0, v___x_3617_);
v___x_3620_ = v___x_3609_;
goto v_reusejp_3619_;
}
else
{
lean_object* v_reuseFailAlloc_3625_; 
v_reuseFailAlloc_3625_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3625_, 0, v___x_3617_);
lean_ctor_set(v_reuseFailAlloc_3625_, 1, v___x_3618_);
v___x_3620_ = v_reuseFailAlloc_3625_;
goto v_reusejp_3619_;
}
v_reusejp_3619_:
{
lean_object* v___x_3621_; lean_object* v___x_3622_; lean_object* v___x_3623_; 
v___x_3621_ = l_Lean_MessageData_ofSyntax(v_before_3611_);
v___x_3622_ = l_Lean_indentD(v___x_3621_);
v___x_3623_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3623_, 0, v___x_3620_);
lean_ctor_set(v___x_3623_, 1, v___x_3622_);
v_x_3604_ = v___x_3623_;
v_x_3605_ = v_tail_3607_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_3633_; lean_object* v___x_3634_; 
v___x_3633_ = ((lean_object*)(lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__1));
v___x_3634_ = l_Lean_MessageData_ofFormat(v___x_3633_);
return v___x_3634_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg(lean_object* v_msgData_3635_, lean_object* v_macroStack_3636_, lean_object* v___y_3637_){
_start:
{
lean_object* v___x_3639_; lean_object* v_scopes_3640_; lean_object* v___x_3641_; lean_object* v___x_3642_; lean_object* v_opts_3643_; lean_object* v___x_3644_; uint8_t v___x_3645_; 
v___x_3639_ = lean_st_ref_get(v___y_3637_);
v_scopes_3640_ = lean_ctor_get(v___x_3639_, 2);
lean_inc(v_scopes_3640_);
lean_dec(v___x_3639_);
v___x_3641_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3642_ = l_List_head_x21___redArg(v___x_3641_, v_scopes_3640_);
lean_dec(v_scopes_3640_);
v_opts_3643_ = lean_ctor_get(v___x_3642_, 1);
lean_inc_ref(v_opts_3643_);
lean_dec(v___x_3642_);
v___x_3644_ = l_Lean_Elab_pp_macroStack;
v___x_3645_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(v_opts_3643_, v___x_3644_);
lean_dec_ref(v_opts_3643_);
if (v___x_3645_ == 0)
{
lean_object* v___x_3646_; 
lean_dec(v_macroStack_3636_);
v___x_3646_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3646_, 0, v_msgData_3635_);
return v___x_3646_;
}
else
{
if (lean_obj_tag(v_macroStack_3636_) == 0)
{
lean_object* v___x_3647_; 
v___x_3647_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3647_, 0, v_msgData_3635_);
return v___x_3647_;
}
else
{
lean_object* v_head_3648_; lean_object* v_after_3649_; lean_object* v___x_3651_; uint8_t v_isShared_3652_; uint8_t v_isSharedCheck_3664_; 
v_head_3648_ = lean_ctor_get(v_macroStack_3636_, 0);
lean_inc(v_head_3648_);
v_after_3649_ = lean_ctor_get(v_head_3648_, 1);
v_isSharedCheck_3664_ = !lean_is_exclusive(v_head_3648_);
if (v_isSharedCheck_3664_ == 0)
{
lean_object* v_unused_3665_; 
v_unused_3665_ = lean_ctor_get(v_head_3648_, 0);
lean_dec(v_unused_3665_);
v___x_3651_ = v_head_3648_;
v_isShared_3652_ = v_isSharedCheck_3664_;
goto v_resetjp_3650_;
}
else
{
lean_inc(v_after_3649_);
lean_dec(v_head_3648_);
v___x_3651_ = lean_box(0);
v_isShared_3652_ = v_isSharedCheck_3664_;
goto v_resetjp_3650_;
}
v_resetjp_3650_:
{
lean_object* v___x_3653_; lean_object* v___x_3655_; 
v___x_3653_ = lean_obj_once(&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0, &lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0_once, _init_lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0);
if (v_isShared_3652_ == 0)
{
lean_ctor_set_tag(v___x_3651_, 7);
lean_ctor_set(v___x_3651_, 1, v___x_3653_);
lean_ctor_set(v___x_3651_, 0, v_msgData_3635_);
v___x_3655_ = v___x_3651_;
goto v_reusejp_3654_;
}
else
{
lean_object* v_reuseFailAlloc_3663_; 
v_reuseFailAlloc_3663_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3663_, 0, v_msgData_3635_);
lean_ctor_set(v_reuseFailAlloc_3663_, 1, v___x_3653_);
v___x_3655_ = v_reuseFailAlloc_3663_;
goto v_reusejp_3654_;
}
v_reusejp_3654_:
{
lean_object* v___x_3656_; lean_object* v___x_3657_; lean_object* v___x_3658_; lean_object* v___x_3659_; lean_object* v_msgData_3660_; lean_object* v___x_3661_; lean_object* v___x_3662_; 
v___x_3656_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2, &lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2_once, _init_lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2);
v___x_3657_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3657_, 0, v___x_3655_);
lean_ctor_set(v___x_3657_, 1, v___x_3656_);
v___x_3658_ = l_Lean_MessageData_ofSyntax(v_after_3649_);
v___x_3659_ = l_Lean_indentD(v___x_3658_);
v_msgData_3660_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_3660_, 0, v___x_3657_);
lean_ctor_set(v_msgData_3660_, 1, v___x_3659_);
v___x_3661_ = lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1(v_msgData_3660_, v_macroStack_3636_);
v___x_3662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3662_, 0, v___x_3661_);
return v___x_3662_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___boxed(lean_object* v_msgData_3666_, lean_object* v_macroStack_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_){
_start:
{
lean_object* v_res_3670_; 
v_res_3670_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg(v_msgData_3666_, v_macroStack_3667_, v___y_3668_);
lean_dec(v___y_3668_);
return v_res_3670_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg(lean_object* v_msg_3671_, lean_object* v___y_3672_, lean_object* v___y_3673_){
_start:
{
lean_object* v___x_3675_; 
v___x_3675_ = l_Lean_Elab_Command_getRef___redArg(v___y_3672_);
if (lean_obj_tag(v___x_3675_) == 0)
{
lean_object* v_a_3676_; lean_object* v_macroStack_3677_; lean_object* v___x_3678_; lean_object* v_a_3679_; lean_object* v___x_3680_; lean_object* v___x_3681_; lean_object* v_a_3682_; lean_object* v___x_3684_; uint8_t v_isShared_3685_; uint8_t v_isSharedCheck_3690_; 
v_a_3676_ = lean_ctor_get(v___x_3675_, 0);
lean_inc(v_a_3676_);
lean_dec_ref_known(v___x_3675_, 1);
v_macroStack_3677_ = lean_ctor_get(v___y_3672_, 4);
v___x_3678_ = lp_LeanSearchClient_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1_spec__1_spec__2_spec__3___redArg(v_msg_3671_, v___y_3673_);
v_a_3679_ = lean_ctor_get(v___x_3678_, 0);
lean_inc(v_a_3679_);
lean_dec_ref(v___x_3678_);
v___x_3680_ = l_Lean_Elab_getBetterRef(v_a_3676_, v_macroStack_3677_);
lean_dec(v_a_3676_);
lean_inc(v_macroStack_3677_);
v___x_3681_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg(v_a_3679_, v_macroStack_3677_, v___y_3673_);
v_a_3682_ = lean_ctor_get(v___x_3681_, 0);
v_isSharedCheck_3690_ = !lean_is_exclusive(v___x_3681_);
if (v_isSharedCheck_3690_ == 0)
{
v___x_3684_ = v___x_3681_;
v_isShared_3685_ = v_isSharedCheck_3690_;
goto v_resetjp_3683_;
}
else
{
lean_inc(v_a_3682_);
lean_dec(v___x_3681_);
v___x_3684_ = lean_box(0);
v_isShared_3685_ = v_isSharedCheck_3690_;
goto v_resetjp_3683_;
}
v_resetjp_3683_:
{
lean_object* v___x_3686_; lean_object* v___x_3688_; 
v___x_3686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3686_, 0, v___x_3680_);
lean_ctor_set(v___x_3686_, 1, v_a_3682_);
if (v_isShared_3685_ == 0)
{
lean_ctor_set_tag(v___x_3684_, 1);
lean_ctor_set(v___x_3684_, 0, v___x_3686_);
v___x_3688_ = v___x_3684_;
goto v_reusejp_3687_;
}
else
{
lean_object* v_reuseFailAlloc_3689_; 
v_reuseFailAlloc_3689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3689_, 0, v___x_3686_);
v___x_3688_ = v_reuseFailAlloc_3689_;
goto v_reusejp_3687_;
}
v_reusejp_3687_:
{
return v___x_3688_;
}
}
}
else
{
lean_object* v_a_3691_; lean_object* v___x_3693_; uint8_t v_isShared_3694_; uint8_t v_isSharedCheck_3698_; 
lean_dec_ref(v_msg_3671_);
v_a_3691_ = lean_ctor_get(v___x_3675_, 0);
v_isSharedCheck_3698_ = !lean_is_exclusive(v___x_3675_);
if (v_isSharedCheck_3698_ == 0)
{
v___x_3693_ = v___x_3675_;
v_isShared_3694_ = v_isSharedCheck_3698_;
goto v_resetjp_3692_;
}
else
{
lean_inc(v_a_3691_);
lean_dec(v___x_3675_);
v___x_3693_ = lean_box(0);
v_isShared_3694_ = v_isSharedCheck_3698_;
goto v_resetjp_3692_;
}
v_resetjp_3692_:
{
lean_object* v___x_3696_; 
if (v_isShared_3694_ == 0)
{
v___x_3696_ = v___x_3693_;
goto v_reusejp_3695_;
}
else
{
lean_object* v_reuseFailAlloc_3697_; 
v_reuseFailAlloc_3697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3697_, 0, v_a_3691_);
v___x_3696_ = v_reuseFailAlloc_3697_;
goto v_reusejp_3695_;
}
v_reusejp_3695_:
{
return v___x_3696_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg___boxed(lean_object* v_msg_3699_, lean_object* v___y_3700_, lean_object* v___y_3701_, lean_object* v___y_3702_){
_start:
{
lean_object* v_res_3703_; 
v_res_3703_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg(v_msg_3699_, v___y_3700_, v___y_3701_);
lean_dec(v___y_3701_);
lean_dec_ref(v___y_3700_);
return v_res_3703_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl(lean_object* v_stx_3707_, lean_object* v_a_3708_, lean_object* v_a_3709_){
_start:
{
lean_object* v_server_3712_; lean_object* v___y_3713_; lean_object* v___y_3714_; lean_object* v___x_3730_; lean_object* v_scopes_3731_; lean_object* v___x_3732_; lean_object* v___x_3733_; lean_object* v_opts_3734_; lean_object* v___x_3735_; lean_object* v___x_3736_; lean_object* v___x_3737_; uint8_t v___x_3738_; 
v___x_3730_ = lean_st_ref_get(v_a_3709_);
v_scopes_3731_ = lean_ctor_get(v___x_3730_, 2);
lean_inc(v_scopes_3731_);
lean_dec(v___x_3730_);
v___x_3732_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3733_ = l_List_head_x21___redArg(v___x_3732_, v_scopes_3731_);
lean_dec(v_scopes_3731_);
v_opts_3734_ = lean_ctor_get(v___x_3733_, 1);
lean_inc_ref(v_opts_3734_);
lean_dec(v___x_3733_);
v___x_3735_ = lp_LeanSearchClient_leansearchclient_backend;
v___x_3736_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(v_opts_3734_, v___x_3735_);
lean_dec_ref(v_opts_3734_);
v___x_3737_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__0));
v___x_3738_ = lean_string_dec_eq(v___x_3736_, v___x_3737_);
if (v___x_3738_ == 0)
{
lean_object* v___x_3739_; lean_object* v___x_3740_; lean_object* v___x_3741_; lean_object* v___x_3742_; lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v___x_3745_; lean_object* v_a_3746_; lean_object* v___x_3748_; uint8_t v_isShared_3749_; uint8_t v_isSharedCheck_3753_; 
lean_dec(v_stx_3707_);
v___x_3739_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__1));
v___x_3740_ = lean_string_append(v___x_3739_, v___x_3736_);
lean_dec_ref(v___x_3736_);
v___x_3741_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__2));
v___x_3742_ = lean_string_append(v___x_3740_, v___x_3741_);
v___x_3743_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3743_, 0, v___x_3742_);
v___x_3744_ = l_Lean_MessageData_ofFormat(v___x_3743_);
v___x_3745_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg(v___x_3744_, v_a_3708_, v_a_3709_);
v_a_3746_ = lean_ctor_get(v___x_3745_, 0);
v_isSharedCheck_3753_ = !lean_is_exclusive(v___x_3745_);
if (v_isSharedCheck_3753_ == 0)
{
v___x_3748_ = v___x_3745_;
v_isShared_3749_ = v_isSharedCheck_3753_;
goto v_resetjp_3747_;
}
else
{
lean_inc(v_a_3746_);
lean_dec(v___x_3745_);
v___x_3748_ = lean_box(0);
v_isShared_3749_ = v_isSharedCheck_3753_;
goto v_resetjp_3747_;
}
v_resetjp_3747_:
{
lean_object* v___x_3751_; 
if (v_isShared_3749_ == 0)
{
v___x_3751_ = v___x_3748_;
goto v_reusejp_3750_;
}
else
{
lean_object* v_reuseFailAlloc_3752_; 
v_reuseFailAlloc_3752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3752_, 0, v_a_3746_);
v___x_3751_ = v_reuseFailAlloc_3752_;
goto v_reusejp_3750_;
}
v_reusejp_3750_:
{
return v___x_3751_;
}
}
}
else
{
lean_object* v___x_3754_; 
lean_dec_ref(v___x_3736_);
v___x_3754_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v_server_3712_ = v___x_3754_;
v___y_3713_ = v_a_3708_;
v___y_3714_ = v_a_3709_;
goto v___jp_3711_;
}
v___jp_3711_:
{
lean_object* v___x_3715_; uint8_t v___x_3716_; 
v___x_3715_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_search__cmd___closed__1));
lean_inc(v_stx_3707_);
v___x_3716_ = l_Lean_Syntax_isOfKind(v_stx_3707_, v___x_3715_);
if (v___x_3716_ == 0)
{
lean_object* v___x_3717_; 
lean_dec(v_stx_3707_);
v___x_3717_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
return v___x_3717_;
}
else
{
lean_object* v___x_3718_; lean_object* v___x_3719_; lean_object* v___x_3720_; uint8_t v___x_3721_; 
v___x_3718_ = lean_unsigned_to_nat(0u);
v___x_3719_ = lean_unsigned_to_nat(1u);
v___x_3720_ = l_Lean_Syntax_getArg(v_stx_3707_, v___x_3719_);
lean_inc(v___x_3720_);
v___x_3721_ = l_Lean_Syntax_matchesNull(v___x_3720_, v___x_3719_);
if (v___x_3721_ == 0)
{
uint8_t v___x_3722_; 
lean_dec(v_stx_3707_);
v___x_3722_ = l_Lean_Syntax_matchesNull(v___x_3720_, v___x_3718_);
if (v___x_3722_ == 0)
{
lean_object* v___x_3723_; 
v___x_3723_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg();
return v___x_3723_;
}
else
{
lean_object* v___x_3724_; lean_object* v___x_3725_; lean_object* v___x_3726_; lean_object* v___x_3727_; 
lean_inc_ref(v_server_3712_);
v___x_3724_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(v_server_3712_);
v___x_3725_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3725_, 0, v___x_3724_);
v___x_3726_ = l_Lean_MessageData_ofFormat(v___x_3725_);
v___x_3727_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_leanSearchCommandImpl_spec__1(v___x_3726_, v___y_3713_, v___y_3714_);
return v___x_3727_;
}
}
else
{
lean_object* v___x_3728_; lean_object* v___x_3729_; 
v___x_3728_ = l_Lean_Syntax_getArg(v___x_3720_, v___x_3718_);
lean_dec(v___x_3720_);
lean_inc_ref(v_server_3712_);
v___x_3729_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchCommandSuggestions(v_server_3712_, v_stx_3707_, v___x_3728_, v___y_3713_, v___y_3714_);
lean_dec(v___x_3728_);
return v___x_3729_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___boxed(lean_object* v_stx_3755_, lean_object* v_a_3756_, lean_object* v_a_3757_, lean_object* v_a_3758_){
_start:
{
lean_object* v_res_3759_; 
v_res_3759_ = lp_LeanSearchClient_LeanSearchClient_searchCommandImpl(v_stx_3755_, v_a_3756_, v_a_3757_);
lean_dec(v_a_3757_);
lean_dec_ref(v_a_3756_);
return v_res_3759_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0(lean_object* v_00_u03b1_3760_, lean_object* v_msg_3761_, lean_object* v___y_3762_, lean_object* v___y_3763_){
_start:
{
lean_object* v___x_3765_; 
v___x_3765_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___redArg(v_msg_3761_, v___y_3762_, v___y_3763_);
return v___x_3765_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0___boxed(lean_object* v_00_u03b1_3766_, lean_object* v_msg_3767_, lean_object* v___y_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_){
_start:
{
lean_object* v_res_3771_; 
v_res_3771_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0(v_00_u03b1_3766_, v_msg_3767_, v___y_3768_, v___y_3769_);
lean_dec(v___y_3769_);
lean_dec_ref(v___y_3768_);
return v_res_3771_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0(lean_object* v_msgData_3772_, lean_object* v_macroStack_3773_, lean_object* v___y_3774_, lean_object* v___y_3775_){
_start:
{
lean_object* v___x_3777_; 
v___x_3777_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg(v_msgData_3772_, v_macroStack_3773_, v___y_3775_);
return v___x_3777_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___boxed(lean_object* v_msgData_3778_, lean_object* v_macroStack_3779_, lean_object* v___y_3780_, lean_object* v___y_3781_, lean_object* v___y_3782_){
_start:
{
lean_object* v_res_3783_; 
v_res_3783_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0(v_msgData_3778_, v_macroStack_3779_, v___y_3780_, v___y_3781_);
lean_dec(v___y_3781_);
lean_dec_ref(v___y_3780_);
return v_res_3783_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg(){
_start:
{
lean_object* v___x_3794_; lean_object* v___x_3795_; 
v___x_3794_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0, &lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0_once, _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0);
v___x_3795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3795_, 0, v___x_3794_);
return v___x_3795_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg___boxed(lean_object* v___y_3796_){
_start:
{
lean_object* v_res_3797_; 
v_res_3797_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
return v_res_3797_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0(lean_object* v_00_u03b1_3798_, lean_object* v___y_3799_, lean_object* v___y_3800_, lean_object* v___y_3801_, lean_object* v___y_3802_, lean_object* v___y_3803_, lean_object* v___y_3804_){
_start:
{
lean_object* v___x_3806_; 
v___x_3806_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
return v___x_3806_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___boxed(lean_object* v_00_u03b1_3807_, lean_object* v___y_3808_, lean_object* v___y_3809_, lean_object* v___y_3810_, lean_object* v___y_3811_, lean_object* v___y_3812_, lean_object* v___y_3813_, lean_object* v___y_3814_){
_start:
{
lean_object* v_res_3815_; 
v_res_3815_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0(v_00_u03b1_3807_, v___y_3808_, v___y_3809_, v___y_3810_, v___y_3811_, v___y_3812_, v___y_3813_);
lean_dec(v___y_3813_);
lean_dec_ref(v___y_3812_);
lean_dec(v___y_3811_);
lean_dec_ref(v___y_3810_);
lean_dec(v___y_3809_);
lean_dec_ref(v___y_3808_);
return v_res_3815_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTermImpl(lean_object* v_stx_3816_, lean_object* v_expectedType_x3f_3817_, lean_object* v_a_3818_, lean_object* v_a_3819_, lean_object* v_a_3820_, lean_object* v_a_3821_, lean_object* v_a_3822_, lean_object* v_a_3823_){
_start:
{
lean_object* v___x_3825_; uint8_t v___x_3826_; 
v___x_3825_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leansearch__search__term___closed__1));
lean_inc(v_stx_3816_);
v___x_3826_ = l_Lean_Syntax_isOfKind(v_stx_3816_, v___x_3825_);
if (v___x_3826_ == 0)
{
lean_object* v___x_3827_; 
lean_dec(v_expectedType_x3f_3817_);
lean_dec(v_stx_3816_);
v___x_3827_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
return v___x_3827_;
}
else
{
lean_object* v___x_3828_; lean_object* v___x_3829_; lean_object* v___x_3830_; uint8_t v___x_3831_; 
v___x_3828_ = lean_unsigned_to_nat(0u);
v___x_3829_ = lean_unsigned_to_nat(1u);
v___x_3830_ = l_Lean_Syntax_getArg(v_stx_3816_, v___x_3829_);
lean_inc(v___x_3830_);
v___x_3831_ = l_Lean_Syntax_matchesNull(v___x_3830_, v___x_3829_);
if (v___x_3831_ == 0)
{
uint8_t v___x_3832_; 
lean_dec(v_stx_3816_);
v___x_3832_ = l_Lean_Syntax_matchesNull(v___x_3830_, v___x_3828_);
if (v___x_3832_ == 0)
{
lean_object* v___x_3833_; 
lean_dec(v_expectedType_x3f_3817_);
v___x_3833_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
return v___x_3833_;
}
else
{
lean_object* v___x_3834_; lean_object* v___x_3835_; 
v___x_3834_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2, &lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2);
v___x_3835_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(v___x_3834_, v_a_3818_, v_a_3819_, v_a_3820_, v_a_3821_, v_a_3822_, v_a_3823_);
if (lean_obj_tag(v___x_3835_) == 0)
{
lean_object* v___x_3836_; 
lean_dec_ref_known(v___x_3835_, 1);
v___x_3836_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_3817_, v_a_3820_, v_a_3821_, v_a_3822_, v_a_3823_);
return v___x_3836_;
}
else
{
lean_object* v_a_3837_; lean_object* v___x_3839_; uint8_t v_isShared_3840_; uint8_t v_isSharedCheck_3844_; 
lean_dec(v_expectedType_x3f_3817_);
v_a_3837_ = lean_ctor_get(v___x_3835_, 0);
v_isSharedCheck_3844_ = !lean_is_exclusive(v___x_3835_);
if (v_isSharedCheck_3844_ == 0)
{
v___x_3839_ = v___x_3835_;
v_isShared_3840_ = v_isSharedCheck_3844_;
goto v_resetjp_3838_;
}
else
{
lean_inc(v_a_3837_);
lean_dec(v___x_3835_);
v___x_3839_ = lean_box(0);
v_isShared_3840_ = v_isSharedCheck_3844_;
goto v_resetjp_3838_;
}
v_resetjp_3838_:
{
lean_object* v___x_3842_; 
if (v_isShared_3840_ == 0)
{
v___x_3842_ = v___x_3839_;
goto v_reusejp_3841_;
}
else
{
lean_object* v_reuseFailAlloc_3843_; 
v_reuseFailAlloc_3843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3843_, 0, v_a_3837_);
v___x_3842_ = v_reuseFailAlloc_3843_;
goto v_reusejp_3841_;
}
v_reusejp_3841_:
{
return v___x_3842_;
}
}
}
}
}
else
{
lean_object* v_s_3845_; lean_object* v___x_3846_; lean_object* v___x_3847_; 
v_s_3845_ = l_Lean_Syntax_getArg(v___x_3830_, v___x_3828_);
lean_dec(v___x_3830_);
v___x_3846_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v___x_3847_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions(v___x_3846_, v_stx_3816_, v_s_3845_, v_a_3818_, v_a_3819_, v_a_3820_, v_a_3821_, v_a_3822_, v_a_3823_);
lean_dec(v_s_3845_);
if (lean_obj_tag(v___x_3847_) == 0)
{
lean_object* v___x_3848_; 
lean_dec_ref_known(v___x_3847_, 1);
v___x_3848_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_3817_, v_a_3820_, v_a_3821_, v_a_3822_, v_a_3823_);
return v___x_3848_;
}
else
{
lean_object* v_a_3849_; lean_object* v___x_3851_; uint8_t v_isShared_3852_; uint8_t v_isSharedCheck_3856_; 
lean_dec(v_expectedType_x3f_3817_);
v_a_3849_ = lean_ctor_get(v___x_3847_, 0);
v_isSharedCheck_3856_ = !lean_is_exclusive(v___x_3847_);
if (v_isSharedCheck_3856_ == 0)
{
v___x_3851_ = v___x_3847_;
v_isShared_3852_ = v_isSharedCheck_3856_;
goto v_resetjp_3850_;
}
else
{
lean_inc(v_a_3849_);
lean_dec(v___x_3847_);
v___x_3851_ = lean_box(0);
v_isShared_3852_ = v_isSharedCheck_3856_;
goto v_resetjp_3850_;
}
v_resetjp_3850_:
{
lean_object* v___x_3854_; 
if (v_isShared_3852_ == 0)
{
v___x_3854_ = v___x_3851_;
goto v_reusejp_3853_;
}
else
{
lean_object* v_reuseFailAlloc_3855_; 
v_reuseFailAlloc_3855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3855_, 0, v_a_3849_);
v___x_3854_ = v_reuseFailAlloc_3855_;
goto v_reusejp_3853_;
}
v_reusejp_3853_:
{
return v___x_3854_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTermImpl___boxed(lean_object* v_stx_3857_, lean_object* v_expectedType_x3f_3858_, lean_object* v_a_3859_, lean_object* v_a_3860_, lean_object* v_a_3861_, lean_object* v_a_3862_, lean_object* v_a_3863_, lean_object* v_a_3864_, lean_object* v_a_3865_){
_start:
{
lean_object* v_res_3866_; 
v_res_3866_ = lp_LeanSearchClient_LeanSearchClient_leanSearchTermImpl(v_stx_3857_, v_expectedType_x3f_3858_, v_a_3859_, v_a_3860_, v_a_3861_, v_a_3862_, v_a_3863_, v_a_3864_);
lean_dec(v_a_3864_);
lean_dec_ref(v_a_3863_);
lean_dec(v_a_3862_);
lean_dec_ref(v_a_3861_);
lean_dec(v_a_3860_);
lean_dec_ref(v_a_3859_);
return v_res_3866_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg(lean_object* v_msgData_3876_, lean_object* v_macroStack_3877_, lean_object* v___y_3878_){
_start:
{
lean_object* v_options_3880_; lean_object* v___x_3881_; uint8_t v___x_3882_; 
v_options_3880_ = lean_ctor_get(v___y_3878_, 2);
v___x_3881_ = l_Lean_Elab_pp_macroStack;
v___x_3882_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__3(v_options_3880_, v___x_3881_);
if (v___x_3882_ == 0)
{
lean_object* v___x_3883_; 
lean_dec(v_macroStack_3877_);
v___x_3883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3883_, 0, v_msgData_3876_);
return v___x_3883_;
}
else
{
if (lean_obj_tag(v_macroStack_3877_) == 0)
{
lean_object* v___x_3884_; 
v___x_3884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3884_, 0, v_msgData_3876_);
return v___x_3884_;
}
else
{
lean_object* v_head_3885_; lean_object* v_after_3886_; lean_object* v___x_3888_; uint8_t v_isShared_3889_; uint8_t v_isSharedCheck_3901_; 
v_head_3885_ = lean_ctor_get(v_macroStack_3877_, 0);
lean_inc(v_head_3885_);
v_after_3886_ = lean_ctor_get(v_head_3885_, 1);
v_isSharedCheck_3901_ = !lean_is_exclusive(v_head_3885_);
if (v_isSharedCheck_3901_ == 0)
{
lean_object* v_unused_3902_; 
v_unused_3902_ = lean_ctor_get(v_head_3885_, 0);
lean_dec(v_unused_3902_);
v___x_3888_ = v_head_3885_;
v_isShared_3889_ = v_isSharedCheck_3901_;
goto v_resetjp_3887_;
}
else
{
lean_inc(v_after_3886_);
lean_dec(v_head_3885_);
v___x_3888_ = lean_box(0);
v_isShared_3889_ = v_isSharedCheck_3901_;
goto v_resetjp_3887_;
}
v_resetjp_3887_:
{
lean_object* v___x_3890_; lean_object* v___x_3892_; 
v___x_3890_ = lean_obj_once(&lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0, &lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0_once, _init_lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1___closed__0);
if (v_isShared_3889_ == 0)
{
lean_ctor_set_tag(v___x_3888_, 7);
lean_ctor_set(v___x_3888_, 1, v___x_3890_);
lean_ctor_set(v___x_3888_, 0, v_msgData_3876_);
v___x_3892_ = v___x_3888_;
goto v_reusejp_3891_;
}
else
{
lean_object* v_reuseFailAlloc_3900_; 
v_reuseFailAlloc_3900_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3900_, 0, v_msgData_3876_);
lean_ctor_set(v_reuseFailAlloc_3900_, 1, v___x_3890_);
v___x_3892_ = v_reuseFailAlloc_3900_;
goto v_reusejp_3891_;
}
v_reusejp_3891_:
{
lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3895_; lean_object* v___x_3896_; lean_object* v_msgData_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; 
v___x_3893_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2, &lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2_once, _init_lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0___redArg___closed__2);
v___x_3894_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3894_, 0, v___x_3892_);
lean_ctor_set(v___x_3894_, 1, v___x_3893_);
v___x_3895_ = l_Lean_MessageData_ofSyntax(v_after_3886_);
v___x_3896_ = l_Lean_indentD(v___x_3895_);
v_msgData_3897_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_3897_, 0, v___x_3894_);
lean_ctor_set(v_msgData_3897_, 1, v___x_3896_);
v___x_3898_ = lp_LeanSearchClient_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchCommandImpl_spec__0_spec__0_spec__1(v_msgData_3897_, v_macroStack_3877_);
v___x_3899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3899_, 0, v___x_3898_);
return v___x_3899_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg___boxed(lean_object* v_msgData_3903_, lean_object* v_macroStack_3904_, lean_object* v___y_3905_, lean_object* v___y_3906_){
_start:
{
lean_object* v_res_3907_; 
v_res_3907_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg(v_msgData_3903_, v_macroStack_3904_, v___y_3905_);
lean_dec_ref(v___y_3905_);
return v_res_3907_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg(lean_object* v_msg_3908_, lean_object* v___y_3909_, lean_object* v___y_3910_, lean_object* v___y_3911_, lean_object* v___y_3912_, lean_object* v___y_3913_, lean_object* v___y_3914_){
_start:
{
lean_object* v_ref_3916_; lean_object* v___x_3917_; lean_object* v_a_3918_; lean_object* v_macroStack_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v_a_3922_; lean_object* v___x_3924_; uint8_t v_isShared_3925_; uint8_t v_isSharedCheck_3930_; 
v_ref_3916_ = lean_ctor_get(v___y_3913_, 5);
v___x_3917_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(v_msg_3908_, v___y_3911_, v___y_3912_, v___y_3913_, v___y_3914_);
v_a_3918_ = lean_ctor_get(v___x_3917_, 0);
lean_inc(v_a_3918_);
lean_dec_ref(v___x_3917_);
v_macroStack_3919_ = lean_ctor_get(v___y_3909_, 1);
v___x_3920_ = l_Lean_Elab_getBetterRef(v_ref_3916_, v_macroStack_3919_);
lean_inc(v_macroStack_3919_);
v___x_3921_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg(v_a_3918_, v_macroStack_3919_, v___y_3913_);
v_a_3922_ = lean_ctor_get(v___x_3921_, 0);
v_isSharedCheck_3930_ = !lean_is_exclusive(v___x_3921_);
if (v_isSharedCheck_3930_ == 0)
{
v___x_3924_ = v___x_3921_;
v_isShared_3925_ = v_isSharedCheck_3930_;
goto v_resetjp_3923_;
}
else
{
lean_inc(v_a_3922_);
lean_dec(v___x_3921_);
v___x_3924_ = lean_box(0);
v_isShared_3925_ = v_isSharedCheck_3930_;
goto v_resetjp_3923_;
}
v_resetjp_3923_:
{
lean_object* v___x_3926_; lean_object* v___x_3928_; 
v___x_3926_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3926_, 0, v___x_3920_);
lean_ctor_set(v___x_3926_, 1, v_a_3922_);
if (v_isShared_3925_ == 0)
{
lean_ctor_set_tag(v___x_3924_, 1);
lean_ctor_set(v___x_3924_, 0, v___x_3926_);
v___x_3928_ = v___x_3924_;
goto v_reusejp_3927_;
}
else
{
lean_object* v_reuseFailAlloc_3929_; 
v_reuseFailAlloc_3929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3929_, 0, v___x_3926_);
v___x_3928_ = v_reuseFailAlloc_3929_;
goto v_reusejp_3927_;
}
v_reusejp_3927_:
{
return v___x_3928_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg___boxed(lean_object* v_msg_3931_, lean_object* v___y_3932_, lean_object* v___y_3933_, lean_object* v___y_3934_, lean_object* v___y_3935_, lean_object* v___y_3936_, lean_object* v___y_3937_, lean_object* v___y_3938_){
_start:
{
lean_object* v_res_3939_; 
v_res_3939_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg(v_msg_3931_, v___y_3932_, v___y_3933_, v___y_3934_, v___y_3935_, v___y_3936_, v___y_3937_);
lean_dec(v___y_3937_);
lean_dec_ref(v___y_3936_);
lean_dec(v___y_3935_);
lean_dec_ref(v___y_3934_);
lean_dec(v___y_3933_);
lean_dec_ref(v___y_3932_);
return v_res_3939_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTermImpl(lean_object* v_stx_3941_, lean_object* v_expectedType_x3f_3942_, lean_object* v_a_3943_, lean_object* v_a_3944_, lean_object* v_a_3945_, lean_object* v_a_3946_, lean_object* v_a_3947_, lean_object* v_a_3948_){
_start:
{
lean_object* v_server_3951_; lean_object* v___y_3952_; lean_object* v___y_3953_; lean_object* v___y_3954_; lean_object* v___y_3955_; lean_object* v___y_3956_; lean_object* v___y_3957_; lean_object* v_options_3991_; lean_object* v___x_3992_; lean_object* v___x_3993_; lean_object* v___x_3994_; uint8_t v___x_3995_; 
v_options_3991_ = lean_ctor_get(v_a_3947_, 2);
v___x_3992_ = lp_LeanSearchClient_leansearchclient_backend;
v___x_3993_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(v_options_3991_, v___x_3992_);
v___x_3994_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__0));
v___x_3995_ = lean_string_dec_eq(v___x_3993_, v___x_3994_);
if (v___x_3995_ == 0)
{
lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; lean_object* v___x_3999_; lean_object* v___x_4000_; lean_object* v___x_4001_; lean_object* v___x_4002_; lean_object* v_a_4003_; lean_object* v___x_4005_; uint8_t v_isShared_4006_; uint8_t v_isSharedCheck_4010_; 
lean_dec(v_expectedType_x3f_3942_);
lean_dec(v_stx_3941_);
v___x_3996_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__1));
v___x_3997_ = lean_string_append(v___x_3996_, v___x_3993_);
lean_dec_ref(v___x_3993_);
v___x_3998_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchTermImpl___closed__0));
v___x_3999_ = lean_string_append(v___x_3997_, v___x_3998_);
v___x_4000_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4000_, 0, v___x_3999_);
v___x_4001_ = l_Lean_MessageData_ofFormat(v___x_4000_);
v___x_4002_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg(v___x_4001_, v_a_3943_, v_a_3944_, v_a_3945_, v_a_3946_, v_a_3947_, v_a_3948_);
v_a_4003_ = lean_ctor_get(v___x_4002_, 0);
v_isSharedCheck_4010_ = !lean_is_exclusive(v___x_4002_);
if (v_isSharedCheck_4010_ == 0)
{
v___x_4005_ = v___x_4002_;
v_isShared_4006_ = v_isSharedCheck_4010_;
goto v_resetjp_4004_;
}
else
{
lean_inc(v_a_4003_);
lean_dec(v___x_4002_);
v___x_4005_ = lean_box(0);
v_isShared_4006_ = v_isSharedCheck_4010_;
goto v_resetjp_4004_;
}
v_resetjp_4004_:
{
lean_object* v___x_4008_; 
if (v_isShared_4006_ == 0)
{
v___x_4008_ = v___x_4005_;
goto v_reusejp_4007_;
}
else
{
lean_object* v_reuseFailAlloc_4009_; 
v_reuseFailAlloc_4009_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4009_, 0, v_a_4003_);
v___x_4008_ = v_reuseFailAlloc_4009_;
goto v_reusejp_4007_;
}
v_reusejp_4007_:
{
return v___x_4008_;
}
}
}
else
{
lean_object* v___x_4011_; 
lean_dec_ref(v___x_3993_);
v___x_4011_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v_server_3951_ = v___x_4011_;
v___y_3952_ = v_a_3943_;
v___y_3953_ = v_a_3944_;
v___y_3954_ = v_a_3945_;
v___y_3955_ = v_a_3946_;
v___y_3956_ = v_a_3947_;
v___y_3957_ = v_a_3948_;
goto v___jp_3950_;
}
v___jp_3950_:
{
lean_object* v___x_3958_; uint8_t v___x_3959_; 
v___x_3958_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_search__term___closed__1));
lean_inc(v_stx_3941_);
v___x_3959_ = l_Lean_Syntax_isOfKind(v_stx_3941_, v___x_3958_);
if (v___x_3959_ == 0)
{
lean_object* v___x_3960_; 
lean_dec(v_expectedType_x3f_3942_);
lean_dec(v_stx_3941_);
v___x_3960_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
return v___x_3960_;
}
else
{
lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; uint8_t v___x_3964_; 
v___x_3961_ = lean_unsigned_to_nat(0u);
v___x_3962_ = lean_unsigned_to_nat(1u);
v___x_3963_ = l_Lean_Syntax_getArg(v_stx_3941_, v___x_3962_);
lean_inc(v___x_3963_);
v___x_3964_ = l_Lean_Syntax_matchesNull(v___x_3963_, v___x_3962_);
if (v___x_3964_ == 0)
{
uint8_t v___x_3965_; 
lean_dec(v_stx_3941_);
v___x_3965_ = l_Lean_Syntax_matchesNull(v___x_3963_, v___x_3961_);
if (v___x_3965_ == 0)
{
lean_object* v___x_3966_; 
lean_dec(v_expectedType_x3f_3942_);
v___x_3966_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTermImpl_spec__0___redArg();
return v___x_3966_;
}
else
{
lean_object* v___x_3967_; lean_object* v___x_3968_; lean_object* v___x_3969_; lean_object* v___x_3970_; 
lean_inc_ref(v_server_3951_);
v___x_3967_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_incompleteSearchQuery(v_server_3951_);
v___x_3968_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3968_, 0, v___x_3967_);
v___x_3969_ = l_Lean_MessageData_ofFormat(v___x_3968_);
v___x_3970_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0(v___x_3969_, v___y_3952_, v___y_3953_, v___y_3954_, v___y_3955_, v___y_3956_, v___y_3957_);
if (lean_obj_tag(v___x_3970_) == 0)
{
lean_object* v___x_3971_; 
lean_dec_ref_known(v___x_3970_, 1);
v___x_3971_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_3942_, v___y_3954_, v___y_3955_, v___y_3956_, v___y_3957_);
return v___x_3971_;
}
else
{
lean_object* v_a_3972_; lean_object* v___x_3974_; uint8_t v_isShared_3975_; uint8_t v_isSharedCheck_3979_; 
lean_dec(v_expectedType_x3f_3942_);
v_a_3972_ = lean_ctor_get(v___x_3970_, 0);
v_isSharedCheck_3979_ = !lean_is_exclusive(v___x_3970_);
if (v_isSharedCheck_3979_ == 0)
{
v___x_3974_ = v___x_3970_;
v_isShared_3975_ = v_isSharedCheck_3979_;
goto v_resetjp_3973_;
}
else
{
lean_inc(v_a_3972_);
lean_dec(v___x_3970_);
v___x_3974_ = lean_box(0);
v_isShared_3975_ = v_isSharedCheck_3979_;
goto v_resetjp_3973_;
}
v_resetjp_3973_:
{
lean_object* v___x_3977_; 
if (v_isShared_3975_ == 0)
{
v___x_3977_ = v___x_3974_;
goto v_reusejp_3976_;
}
else
{
lean_object* v_reuseFailAlloc_3978_; 
v_reuseFailAlloc_3978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3978_, 0, v_a_3972_);
v___x_3977_ = v_reuseFailAlloc_3978_;
goto v_reusejp_3976_;
}
v_reusejp_3976_:
{
return v___x_3977_;
}
}
}
}
}
else
{
lean_object* v___x_3980_; lean_object* v___x_3981_; 
v___x_3980_ = l_Lean_Syntax_getArg(v___x_3963_, v___x_3961_);
lean_dec(v___x_3963_);
lean_inc_ref(v_server_3951_);
v___x_3981_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTermSuggestions(v_server_3951_, v_stx_3941_, v___x_3980_, v___y_3952_, v___y_3953_, v___y_3954_, v___y_3955_, v___y_3956_, v___y_3957_);
lean_dec(v___x_3980_);
if (lean_obj_tag(v___x_3981_) == 0)
{
lean_object* v___x_3982_; 
lean_dec_ref_known(v___x_3981_, 1);
v___x_3982_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_3942_, v___y_3954_, v___y_3955_, v___y_3956_, v___y_3957_);
return v___x_3982_;
}
else
{
lean_object* v_a_3983_; lean_object* v___x_3985_; uint8_t v_isShared_3986_; uint8_t v_isSharedCheck_3990_; 
lean_dec(v_expectedType_x3f_3942_);
v_a_3983_ = lean_ctor_get(v___x_3981_, 0);
v_isSharedCheck_3990_ = !lean_is_exclusive(v___x_3981_);
if (v_isSharedCheck_3990_ == 0)
{
v___x_3985_ = v___x_3981_;
v_isShared_3986_ = v_isSharedCheck_3990_;
goto v_resetjp_3984_;
}
else
{
lean_inc(v_a_3983_);
lean_dec(v___x_3981_);
v___x_3985_ = lean_box(0);
v_isShared_3986_ = v_isSharedCheck_3990_;
goto v_resetjp_3984_;
}
v_resetjp_3984_:
{
lean_object* v___x_3988_; 
if (v_isShared_3986_ == 0)
{
v___x_3988_ = v___x_3985_;
goto v_reusejp_3987_;
}
else
{
lean_object* v_reuseFailAlloc_3989_; 
v_reuseFailAlloc_3989_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3989_, 0, v_a_3983_);
v___x_3988_ = v_reuseFailAlloc_3989_;
goto v_reusejp_3987_;
}
v_reusejp_3987_:
{
return v___x_3988_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTermImpl___boxed(lean_object* v_stx_4012_, lean_object* v_expectedType_x3f_4013_, lean_object* v_a_4014_, lean_object* v_a_4015_, lean_object* v_a_4016_, lean_object* v_a_4017_, lean_object* v_a_4018_, lean_object* v_a_4019_, lean_object* v_a_4020_){
_start:
{
lean_object* v_res_4021_; 
v_res_4021_ = lp_LeanSearchClient_LeanSearchClient_searchTermImpl(v_stx_4012_, v_expectedType_x3f_4013_, v_a_4014_, v_a_4015_, v_a_4016_, v_a_4017_, v_a_4018_, v_a_4019_);
lean_dec(v_a_4019_);
lean_dec_ref(v_a_4018_);
lean_dec(v_a_4017_);
lean_dec_ref(v_a_4016_);
lean_dec(v_a_4015_);
lean_dec_ref(v_a_4014_);
return v_res_4021_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0(lean_object* v_00_u03b1_4022_, lean_object* v_msg_4023_, lean_object* v___y_4024_, lean_object* v___y_4025_, lean_object* v___y_4026_, lean_object* v___y_4027_, lean_object* v___y_4028_, lean_object* v___y_4029_){
_start:
{
lean_object* v___x_4031_; 
v___x_4031_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___redArg(v_msg_4023_, v___y_4024_, v___y_4025_, v___y_4026_, v___y_4027_, v___y_4028_, v___y_4029_);
return v___x_4031_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0___boxed(lean_object* v_00_u03b1_4032_, lean_object* v_msg_4033_, lean_object* v___y_4034_, lean_object* v___y_4035_, lean_object* v___y_4036_, lean_object* v___y_4037_, lean_object* v___y_4038_, lean_object* v___y_4039_, lean_object* v___y_4040_){
_start:
{
lean_object* v_res_4041_; 
v_res_4041_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0(v_00_u03b1_4032_, v_msg_4033_, v___y_4034_, v___y_4035_, v___y_4036_, v___y_4037_, v___y_4038_, v___y_4039_);
lean_dec(v___y_4039_);
lean_dec_ref(v___y_4038_);
lean_dec(v___y_4037_);
lean_dec_ref(v___y_4036_);
lean_dec(v___y_4035_);
lean_dec_ref(v___y_4034_);
return v_res_4041_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0(lean_object* v_msgData_4042_, lean_object* v_macroStack_4043_, lean_object* v___y_4044_, lean_object* v___y_4045_, lean_object* v___y_4046_, lean_object* v___y_4047_, lean_object* v___y_4048_, lean_object* v___y_4049_){
_start:
{
lean_object* v___x_4051_; 
v___x_4051_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___redArg(v_msgData_4042_, v_macroStack_4043_, v___y_4048_);
return v___x_4051_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0___boxed(lean_object* v_msgData_4052_, lean_object* v_macroStack_4053_, lean_object* v___y_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_, lean_object* v___y_4057_, lean_object* v___y_4058_, lean_object* v___y_4059_, lean_object* v___y_4060_){
_start:
{
lean_object* v_res_4061_; 
v_res_4061_ = lp_LeanSearchClient_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00LeanSearchClient_searchTermImpl_spec__0_spec__0(v_msgData_4052_, v_macroStack_4053_, v___y_4054_, v___y_4055_, v___y_4056_, v___y_4057_, v___y_4058_, v___y_4059_);
lean_dec(v___y_4059_);
lean_dec_ref(v___y_4058_);
lean_dec(v___y_4057_);
lean_dec_ref(v___y_4056_);
lean_dec(v___y_4055_);
lean_dec_ref(v___y_4054_);
return v_res_4061_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg(){
_start:
{
lean_object* v___x_4094_; lean_object* v___x_4095_; 
v___x_4094_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0, &lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0_once, _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchCommandImpl_spec__0___redArg___closed__0);
v___x_4095_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4095_, 0, v___x_4094_);
return v___x_4095_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg___boxed(lean_object* v___y_4096_){
_start:
{
lean_object* v_res_4097_; 
v_res_4097_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v_res_4097_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0(lean_object* v_00_u03b1_4098_, lean_object* v___y_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_, lean_object* v___y_4104_, lean_object* v___y_4105_, lean_object* v___y_4106_){
_start:
{
lean_object* v___x_4108_; 
v___x_4108_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v___x_4108_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___boxed(lean_object* v_00_u03b1_4109_, lean_object* v___y_4110_, lean_object* v___y_4111_, lean_object* v___y_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_, lean_object* v___y_4117_, lean_object* v___y_4118_){
_start:
{
lean_object* v_res_4119_; 
v_res_4119_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0(v_00_u03b1_4109_, v___y_4110_, v___y_4111_, v___y_4112_, v___y_4113_, v___y_4114_, v___y_4115_, v___y_4116_, v___y_4117_);
lean_dec(v___y_4117_);
lean_dec_ref(v___y_4116_);
lean_dec(v___y_4115_);
lean_dec_ref(v___y_4114_);
lean_dec(v___y_4113_);
lean_dec_ref(v___y_4112_);
lean_dec(v___y_4111_);
lean_dec_ref(v___y_4110_);
return v_res_4119_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___lam__0(uint8_t v___x_4120_, lean_object* v_stx_4121_, lean_object* v___y_4122_, lean_object* v___y_4123_, lean_object* v___y_4124_, lean_object* v___y_4125_, lean_object* v___y_4126_, lean_object* v___y_4127_, lean_object* v___y_4128_, lean_object* v___y_4129_){
_start:
{
if (v___x_4120_ == 0)
{
lean_object* v___x_4131_; 
lean_dec(v_stx_4121_);
v___x_4131_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v___x_4131_;
}
else
{
lean_object* v___x_4132_; lean_object* v___x_4133_; lean_object* v___x_4134_; uint8_t v___x_4135_; 
v___x_4132_ = lean_unsigned_to_nat(0u);
v___x_4133_ = lean_unsigned_to_nat(1u);
v___x_4134_ = l_Lean_Syntax_getArg(v_stx_4121_, v___x_4133_);
lean_inc(v___x_4134_);
v___x_4135_ = l_Lean_Syntax_matchesNull(v___x_4134_, v___x_4133_);
if (v___x_4135_ == 0)
{
uint8_t v___x_4136_; 
lean_dec(v_stx_4121_);
v___x_4136_ = l_Lean_Syntax_matchesNull(v___x_4134_, v___x_4132_);
if (v___x_4136_ == 0)
{
lean_object* v___x_4137_; 
v___x_4137_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v___x_4137_;
}
else
{
lean_object* v___x_4138_; lean_object* v___x_4139_; 
v___x_4138_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2, &lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_leanSearchCommandImpl___closed__2);
v___x_4139_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__0(v___x_4138_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_, v___y_4126_, v___y_4127_, v___y_4128_, v___y_4129_);
return v___x_4139_;
}
}
else
{
lean_object* v_s_4140_; lean_object* v___x_4141_; lean_object* v___x_4142_; 
v_s_4140_ = l_Lean_Syntax_getArg(v___x_4134_, v___x_4132_);
lean_dec(v___x_4134_);
v___x_4141_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v___x_4142_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions(v___x_4141_, v_stx_4121_, v_s_4140_, v___y_4122_, v___y_4123_, v___y_4124_, v___y_4125_, v___y_4126_, v___y_4127_, v___y_4128_, v___y_4129_);
lean_dec(v_s_4140_);
return v___x_4142_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___lam__0___boxed(lean_object* v___x_4143_, lean_object* v_stx_4144_, lean_object* v___y_4145_, lean_object* v___y_4146_, lean_object* v___y_4147_, lean_object* v___y_4148_, lean_object* v___y_4149_, lean_object* v___y_4150_, lean_object* v___y_4151_, lean_object* v___y_4152_, lean_object* v___y_4153_){
_start:
{
uint8_t v___x_629__boxed_4154_; lean_object* v_res_4155_; 
v___x_629__boxed_4154_ = lean_unbox(v___x_4143_);
v_res_4155_ = lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___lam__0(v___x_629__boxed_4154_, v_stx_4144_, v___y_4145_, v___y_4146_, v___y_4147_, v___y_4148_, v___y_4149_, v___y_4150_, v___y_4151_, v___y_4152_);
lean_dec(v___y_4152_);
lean_dec_ref(v___y_4151_);
lean_dec(v___y_4150_);
lean_dec_ref(v___y_4149_);
lean_dec(v___y_4148_);
lean_dec_ref(v___y_4147_);
lean_dec(v___y_4146_);
lean_dec_ref(v___y_4145_);
return v_res_4155_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl(lean_object* v_stx_4156_, lean_object* v_a_4157_, lean_object* v_a_4158_, lean_object* v_a_4159_, lean_object* v_a_4160_, lean_object* v_a_4161_, lean_object* v_a_4162_, lean_object* v_a_4163_, lean_object* v_a_4164_){
_start:
{
lean_object* v___x_4166_; uint8_t v___x_4167_; lean_object* v___x_4168_; lean_object* v___y_4169_; lean_object* v___x_4170_; 
v___x_4166_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leansearch__search__tactic___closed__1));
lean_inc(v_stx_4156_);
v___x_4167_ = l_Lean_Syntax_isOfKind(v_stx_4156_, v___x_4166_);
v___x_4168_ = lean_box(v___x_4167_);
v___y_4169_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___lam__0___boxed), 11, 2);
lean_closure_set(v___y_4169_, 0, v___x_4168_);
lean_closure_set(v___y_4169_, 1, v_stx_4156_);
v___x_4170_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_4169_, v_a_4157_, v_a_4158_, v_a_4159_, v_a_4160_, v_a_4161_, v_a_4162_, v_a_4163_, v_a_4164_);
return v___x_4170_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl___boxed(lean_object* v_stx_4171_, lean_object* v_a_4172_, lean_object* v_a_4173_, lean_object* v_a_4174_, lean_object* v_a_4175_, lean_object* v_a_4176_, lean_object* v_a_4177_, lean_object* v_a_4178_, lean_object* v_a_4179_, lean_object* v_a_4180_){
_start:
{
lean_object* v_res_4181_; 
v_res_4181_ = lp_LeanSearchClient_LeanSearchClient_leanSearchTacticImpl(v_stx_4171_, v_a_4172_, v_a_4173_, v_a_4174_, v_a_4175_, v_a_4176_, v_a_4177_, v_a_4178_, v_a_4179_);
lean_dec(v_a_4179_);
lean_dec_ref(v_a_4178_);
lean_dec(v_a_4177_);
lean_dec_ref(v_a_4176_);
lean_dec(v_a_4175_);
lean_dec_ref(v_a_4174_);
lean_dec(v_a_4173_);
lean_dec_ref(v_a_4172_);
return v_res_4181_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg(lean_object* v_stx_4197_, uint8_t v___x_4198_, lean_object* v_a_4199_, lean_object* v_as_4200_, size_t v_sz_4201_, size_t v_i_4202_, uint8_t v_b_4203_, lean_object* v___y_4204_, lean_object* v___y_4205_, lean_object* v___y_4206_, lean_object* v___y_4207_, lean_object* v___y_4208_, lean_object* v___y_4209_){
_start:
{
uint8_t v_a_4212_; uint8_t v___x_4216_; 
v___x_4216_ = lean_usize_dec_lt(v_i_4202_, v_sz_4201_);
if (v___x_4216_ == 0)
{
lean_object* v___x_4217_; lean_object* v___x_4218_; 
lean_dec_ref(v_a_4199_);
lean_dec(v_stx_4197_);
v___x_4217_ = lean_box(v_b_4203_);
v___x_4218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4218_, 0, v___x_4217_);
return v___x_4218_;
}
else
{
lean_object* v_a_4219_; lean_object* v_fst_4220_; lean_object* v_snd_4221_; lean_object* v___x_4222_; lean_object* v_a_4224_; lean_object* v___y_4242_; lean_object* v___x_4252_; lean_object* v___x_4253_; uint8_t v___x_4254_; 
v_a_4219_ = lean_array_uget_borrowed(v_as_4200_, v_i_4202_);
v_fst_4220_ = lean_ctor_get(v_a_4219_, 0);
v_snd_4221_ = lean_ctor_get(v_a_4219_, 1);
v___x_4222_ = lean_unsigned_to_nat(0u);
v___x_4252_ = lean_array_get_size(v_snd_4221_);
v___x_4253_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions___closed__0));
v___x_4254_ = lean_nat_dec_lt(v___x_4222_, v___x_4252_);
if (v___x_4254_ == 0)
{
v_a_4224_ = v___x_4253_;
goto v___jp_4223_;
}
else
{
uint8_t v___x_4255_; 
v___x_4255_ = lean_nat_dec_le(v___x_4252_, v___x_4252_);
if (v___x_4255_ == 0)
{
if (v___x_4254_ == 0)
{
v_a_4224_ = v___x_4253_;
goto v___jp_4223_;
}
else
{
size_t v___x_4256_; size_t v___x_4257_; lean_object* v___x_4258_; 
v___x_4256_ = ((size_t)0ULL);
v___x_4257_ = lean_usize_of_nat(v___x_4252_);
lean_inc_ref(v_a_4199_);
v___x_4258_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(v_a_4199_, v_snd_4221_, v___x_4256_, v___x_4257_, v___x_4253_, v___y_4204_, v___y_4205_, v___y_4206_, v___y_4207_, v___y_4208_, v___y_4209_);
v___y_4242_ = v___x_4258_;
goto v___jp_4241_;
}
}
else
{
size_t v___x_4259_; size_t v___x_4260_; lean_object* v___x_4261_; 
v___x_4259_ = ((size_t)0ULL);
v___x_4260_ = lean_usize_of_nat(v___x_4252_);
lean_inc_ref(v_a_4199_);
v___x_4261_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__1___redArg(v_a_4199_, v_snd_4221_, v___x_4259_, v___x_4260_, v___x_4253_, v___y_4204_, v___y_4205_, v___y_4206_, v___y_4207_, v___y_4208_, v___y_4209_);
v___y_4242_ = v___x_4261_;
goto v___jp_4241_;
}
}
v___jp_4223_:
{
lean_object* v___x_4225_; uint8_t v___x_4226_; 
v___x_4225_ = lean_array_get_size(v_a_4224_);
v___x_4226_ = lean_nat_dec_eq(v___x_4225_, v___x_4222_);
if (v___x_4226_ == 0)
{
lean_object* v___x_4227_; lean_object* v___x_4228_; lean_object* v___x_4229_; uint8_t v___x_4230_; lean_object* v___x_4231_; lean_object* v___x_4232_; 
v___x_4227_ = lean_box(0);
v___x_4228_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_SearchServer_searchTacticSuggestions_spec__2___closed__0));
v___x_4229_ = lean_string_append(v___x_4228_, v_fst_4220_);
v___x_4230_ = 4;
v___x_4231_ = l_Lean_MessageData_nil;
lean_inc(v_stx_4197_);
v___x_4232_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_4197_, v_a_4224_, v___x_4227_, v___x_4229_, v___x_4227_, v___x_4230_, v___x_4231_, v___y_4208_, v___y_4209_);
if (lean_obj_tag(v___x_4232_) == 0)
{
lean_dec_ref_known(v___x_4232_, 1);
v_a_4212_ = v___x_4198_;
goto v___jp_4211_;
}
else
{
lean_object* v_a_4233_; lean_object* v___x_4235_; uint8_t v_isShared_4236_; uint8_t v_isSharedCheck_4240_; 
lean_dec_ref(v_a_4199_);
lean_dec(v_stx_4197_);
v_a_4233_ = lean_ctor_get(v___x_4232_, 0);
v_isSharedCheck_4240_ = !lean_is_exclusive(v___x_4232_);
if (v_isSharedCheck_4240_ == 0)
{
v___x_4235_ = v___x_4232_;
v_isShared_4236_ = v_isSharedCheck_4240_;
goto v_resetjp_4234_;
}
else
{
lean_inc(v_a_4233_);
lean_dec(v___x_4232_);
v___x_4235_ = lean_box(0);
v_isShared_4236_ = v_isSharedCheck_4240_;
goto v_resetjp_4234_;
}
v_resetjp_4234_:
{
lean_object* v___x_4238_; 
if (v_isShared_4236_ == 0)
{
v___x_4238_ = v___x_4235_;
goto v_reusejp_4237_;
}
else
{
lean_object* v_reuseFailAlloc_4239_; 
v_reuseFailAlloc_4239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4239_, 0, v_a_4233_);
v___x_4238_ = v_reuseFailAlloc_4239_;
goto v_reusejp_4237_;
}
v_reusejp_4237_:
{
return v___x_4238_;
}
}
}
}
else
{
lean_dec_ref(v_a_4224_);
v_a_4212_ = v_b_4203_;
goto v___jp_4211_;
}
}
v___jp_4241_:
{
if (lean_obj_tag(v___y_4242_) == 0)
{
lean_object* v_a_4243_; 
v_a_4243_ = lean_ctor_get(v___y_4242_, 0);
lean_inc(v_a_4243_);
lean_dec_ref_known(v___y_4242_, 1);
v_a_4224_ = v_a_4243_;
goto v___jp_4223_;
}
else
{
lean_object* v_a_4244_; lean_object* v___x_4246_; uint8_t v_isShared_4247_; uint8_t v_isSharedCheck_4251_; 
lean_dec_ref(v_a_4199_);
lean_dec(v_stx_4197_);
v_a_4244_ = lean_ctor_get(v___y_4242_, 0);
v_isSharedCheck_4251_ = !lean_is_exclusive(v___y_4242_);
if (v_isSharedCheck_4251_ == 0)
{
v___x_4246_ = v___y_4242_;
v_isShared_4247_ = v_isSharedCheck_4251_;
goto v_resetjp_4245_;
}
else
{
lean_inc(v_a_4244_);
lean_dec(v___y_4242_);
v___x_4246_ = lean_box(0);
v_isShared_4247_ = v_isSharedCheck_4251_;
goto v_resetjp_4245_;
}
v_resetjp_4245_:
{
lean_object* v___x_4249_; 
if (v_isShared_4247_ == 0)
{
v___x_4249_ = v___x_4246_;
goto v_reusejp_4248_;
}
else
{
lean_object* v_reuseFailAlloc_4250_; 
v_reuseFailAlloc_4250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4250_, 0, v_a_4244_);
v___x_4249_ = v_reuseFailAlloc_4250_;
goto v_reusejp_4248_;
}
v_reusejp_4248_:
{
return v___x_4249_;
}
}
}
}
}
v___jp_4211_:
{
size_t v___x_4213_; size_t v___x_4214_; 
v___x_4213_ = ((size_t)1ULL);
v___x_4214_ = lean_usize_add(v_i_4202_, v___x_4213_);
v_i_4202_ = v___x_4214_;
v_b_4203_ = v_a_4212_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg___boxed(lean_object* v_stx_4262_, lean_object* v___x_4263_, lean_object* v_a_4264_, lean_object* v_as_4265_, lean_object* v_sz_4266_, lean_object* v_i_4267_, lean_object* v_b_4268_, lean_object* v___y_4269_, lean_object* v___y_4270_, lean_object* v___y_4271_, lean_object* v___y_4272_, lean_object* v___y_4273_, lean_object* v___y_4274_, lean_object* v___y_4275_){
_start:
{
uint8_t v___x_7202__boxed_4276_; size_t v_sz_boxed_4277_; size_t v_i_boxed_4278_; uint8_t v_b_boxed_4279_; lean_object* v_res_4280_; 
v___x_7202__boxed_4276_ = lean_unbox(v___x_4263_);
v_sz_boxed_4277_ = lean_unbox_usize(v_sz_4266_);
lean_dec(v_sz_4266_);
v_i_boxed_4278_ = lean_unbox_usize(v_i_4267_);
lean_dec(v_i_4267_);
v_b_boxed_4279_ = lean_unbox(v_b_4268_);
v_res_4280_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg(v_stx_4262_, v___x_7202__boxed_4276_, v_a_4264_, v_as_4265_, v_sz_boxed_4277_, v_i_boxed_4278_, v_b_boxed_4279_, v___y_4269_, v___y_4270_, v___y_4271_, v___y_4272_, v___y_4273_, v___y_4274_);
lean_dec(v___y_4274_);
lean_dec_ref(v___y_4273_);
lean_dec(v___y_4272_);
lean_dec_ref(v___y_4271_);
lean_dec(v___y_4270_);
lean_dec_ref(v___y_4269_);
lean_dec_ref(v_as_4265_);
return v_res_4280_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0(lean_object* v_stx_4282_, lean_object* v___y_4283_, lean_object* v___y_4284_, lean_object* v___y_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_, lean_object* v___y_4288_, lean_object* v___y_4289_, lean_object* v___y_4290_){
_start:
{
lean_object* v___x_4292_; 
v___x_4292_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4284_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_);
if (lean_obj_tag(v___x_4292_) == 0)
{
lean_object* v_a_4293_; lean_object* v___x_4294_; 
v_a_4293_ = lean_ctor_get(v___x_4292_, 0);
lean_inc(v_a_4293_);
lean_dec_ref_known(v___x_4292_, 1);
v___x_4294_ = l_Lean_Elab_Tactic_getMainTarget(v___y_4283_, v___y_4284_, v___y_4285_, v___y_4286_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_);
if (lean_obj_tag(v___x_4294_) == 0)
{
lean_object* v_a_4295_; lean_object* v___x_4296_; 
v_a_4295_ = lean_ctor_get(v___x_4294_, 0);
lean_inc(v_a_4295_);
lean_dec_ref_known(v___x_4294_, 1);
v___x_4296_ = l_Lean_Meta_ppGoal(v_a_4293_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_);
lean_dec(v_a_4293_);
if (lean_obj_tag(v___x_4296_) == 0)
{
lean_object* v_a_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; lean_object* v___x_4300_; lean_object* v___x_4301_; uint8_t v___x_4302_; 
v_a_4297_ = lean_ctor_get(v___x_4296_, 0);
lean_inc(v_a_4297_);
lean_dec_ref_known(v___x_4296_, 1);
v___x_4298_ = l_Std_Format_defWidth;
v___x_4299_ = lean_unsigned_to_nat(0u);
v___x_4300_ = l_Std_Format_pretty(v_a_4297_, v___x_4298_, v___x_4299_, v___x_4299_);
v___x_4301_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__1));
lean_inc(v_stx_4282_);
v___x_4302_ = l_Lean_Syntax_isOfKind(v_stx_4282_, v___x_4301_);
if (v___x_4302_ == 0)
{
lean_object* v___x_4303_; 
lean_dec_ref(v___x_4300_);
lean_dec(v_a_4295_);
lean_dec(v_stx_4282_);
v___x_4303_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v___x_4303_;
}
else
{
lean_object* v_options_4304_; lean_object* v___x_4305_; lean_object* v___x_4306_; lean_object* v___x_4307_; lean_object* v___x_4308_; lean_object* v___x_4309_; 
v_options_4304_ = lean_ctor_get(v___y_4289_, 2);
v___x_4305_ = lp_LeanSearchClient_statesearch_queries;
v___x_4306_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_leanSearchServer_spec__0(v_options_4304_, v___x_4305_);
v___x_4307_ = lp_LeanSearchClient_statesearch_revision;
v___x_4308_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(v_options_4304_, v___x_4307_);
v___x_4309_ = lp_LeanSearchClient_LeanSearchClient_queryStateSearch___redArg(v___x_4300_, v___x_4306_, v___x_4308_, v___y_4289_);
if (lean_obj_tag(v___x_4309_) == 0)
{
lean_object* v_a_4310_; size_t v_sz_4311_; size_t v___x_4312_; lean_object* v___x_4313_; uint8_t v___x_4314_; size_t v_sz_4315_; lean_object* v___x_4316_; 
v_a_4310_ = lean_ctor_get(v___x_4309_, 0);
lean_inc_n(v_a_4310_, 2);
lean_dec_ref_known(v___x_4309_, 1);
v_sz_4311_ = lean_array_size(v_a_4310_);
v___x_4312_ = ((size_t)0ULL);
v___x_4313_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getTacticSuggestionGroups_spec__0(v_sz_4311_, v___x_4312_, v_a_4310_);
v___x_4314_ = 0;
v_sz_4315_ = lean_array_size(v___x_4313_);
lean_inc(v_stx_4282_);
v___x_4316_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg(v_stx_4282_, v___x_4302_, v_a_4295_, v___x_4313_, v_sz_4315_, v___x_4312_, v___x_4314_, v___y_4285_, v___y_4286_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_);
lean_dec_ref(v___x_4313_);
if (lean_obj_tag(v___x_4316_) == 0)
{
lean_object* v_a_4317_; lean_object* v___x_4319_; uint8_t v_isShared_4320_; uint8_t v_isSharedCheck_4332_; 
v_a_4317_ = lean_ctor_get(v___x_4316_, 0);
v_isSharedCheck_4332_ = !lean_is_exclusive(v___x_4316_);
if (v_isSharedCheck_4332_ == 0)
{
v___x_4319_ = v___x_4316_;
v_isShared_4320_ = v_isSharedCheck_4332_;
goto v_resetjp_4318_;
}
else
{
lean_inc(v_a_4317_);
lean_dec(v___x_4316_);
v___x_4319_ = lean_box(0);
v_isShared_4320_ = v_isSharedCheck_4332_;
goto v_resetjp_4318_;
}
v_resetjp_4318_:
{
uint8_t v___x_4321_; 
v___x_4321_ = lean_unbox(v_a_4317_);
lean_dec(v_a_4317_);
if (v___x_4321_ == 0)
{
lean_object* v___x_4322_; lean_object* v___x_4323_; lean_object* v___x_4324_; uint8_t v___x_4325_; lean_object* v___x_4326_; lean_object* v___x_4327_; 
lean_del_object(v___x_4319_);
v___x_4322_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_SearchServer_getCommandSuggestions_spec__0(v_sz_4311_, v___x_4312_, v_a_4310_);
v___x_4323_ = lean_box(0);
v___x_4324_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___closed__0));
v___x_4325_ = 4;
v___x_4326_ = l_Lean_MessageData_nil;
v___x_4327_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_4282_, v___x_4322_, v___x_4323_, v___x_4324_, v___x_4323_, v___x_4325_, v___x_4326_, v___y_4289_, v___y_4290_);
return v___x_4327_;
}
else
{
lean_object* v___x_4328_; lean_object* v___x_4330_; 
lean_dec(v_a_4310_);
lean_dec(v_stx_4282_);
v___x_4328_ = lean_box(0);
if (v_isShared_4320_ == 0)
{
lean_ctor_set(v___x_4319_, 0, v___x_4328_);
v___x_4330_ = v___x_4319_;
goto v_reusejp_4329_;
}
else
{
lean_object* v_reuseFailAlloc_4331_; 
v_reuseFailAlloc_4331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4331_, 0, v___x_4328_);
v___x_4330_ = v_reuseFailAlloc_4331_;
goto v_reusejp_4329_;
}
v_reusejp_4329_:
{
return v___x_4330_;
}
}
}
}
else
{
lean_object* v_a_4333_; lean_object* v___x_4335_; uint8_t v_isShared_4336_; uint8_t v_isSharedCheck_4340_; 
lean_dec(v_a_4310_);
lean_dec(v_stx_4282_);
v_a_4333_ = lean_ctor_get(v___x_4316_, 0);
v_isSharedCheck_4340_ = !lean_is_exclusive(v___x_4316_);
if (v_isSharedCheck_4340_ == 0)
{
v___x_4335_ = v___x_4316_;
v_isShared_4336_ = v_isSharedCheck_4340_;
goto v_resetjp_4334_;
}
else
{
lean_inc(v_a_4333_);
lean_dec(v___x_4316_);
v___x_4335_ = lean_box(0);
v_isShared_4336_ = v_isSharedCheck_4340_;
goto v_resetjp_4334_;
}
v_resetjp_4334_:
{
lean_object* v___x_4338_; 
if (v_isShared_4336_ == 0)
{
v___x_4338_ = v___x_4335_;
goto v_reusejp_4337_;
}
else
{
lean_object* v_reuseFailAlloc_4339_; 
v_reuseFailAlloc_4339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4339_, 0, v_a_4333_);
v___x_4338_ = v_reuseFailAlloc_4339_;
goto v_reusejp_4337_;
}
v_reusejp_4337_:
{
return v___x_4338_;
}
}
}
}
else
{
lean_object* v_a_4341_; lean_object* v___x_4343_; uint8_t v_isShared_4344_; uint8_t v_isSharedCheck_4348_; 
lean_dec(v_a_4295_);
lean_dec(v_stx_4282_);
v_a_4341_ = lean_ctor_get(v___x_4309_, 0);
v_isSharedCheck_4348_ = !lean_is_exclusive(v___x_4309_);
if (v_isSharedCheck_4348_ == 0)
{
v___x_4343_ = v___x_4309_;
v_isShared_4344_ = v_isSharedCheck_4348_;
goto v_resetjp_4342_;
}
else
{
lean_inc(v_a_4341_);
lean_dec(v___x_4309_);
v___x_4343_ = lean_box(0);
v_isShared_4344_ = v_isSharedCheck_4348_;
goto v_resetjp_4342_;
}
v_resetjp_4342_:
{
lean_object* v___x_4346_; 
if (v_isShared_4344_ == 0)
{
v___x_4346_ = v___x_4343_;
goto v_reusejp_4345_;
}
else
{
lean_object* v_reuseFailAlloc_4347_; 
v_reuseFailAlloc_4347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4347_, 0, v_a_4341_);
v___x_4346_ = v_reuseFailAlloc_4347_;
goto v_reusejp_4345_;
}
v_reusejp_4345_:
{
return v___x_4346_;
}
}
}
}
}
else
{
lean_object* v_a_4349_; lean_object* v___x_4351_; uint8_t v_isShared_4352_; uint8_t v_isSharedCheck_4356_; 
lean_dec(v_a_4295_);
lean_dec(v_stx_4282_);
v_a_4349_ = lean_ctor_get(v___x_4296_, 0);
v_isSharedCheck_4356_ = !lean_is_exclusive(v___x_4296_);
if (v_isSharedCheck_4356_ == 0)
{
v___x_4351_ = v___x_4296_;
v_isShared_4352_ = v_isSharedCheck_4356_;
goto v_resetjp_4350_;
}
else
{
lean_inc(v_a_4349_);
lean_dec(v___x_4296_);
v___x_4351_ = lean_box(0);
v_isShared_4352_ = v_isSharedCheck_4356_;
goto v_resetjp_4350_;
}
v_resetjp_4350_:
{
lean_object* v___x_4354_; 
if (v_isShared_4352_ == 0)
{
v___x_4354_ = v___x_4351_;
goto v_reusejp_4353_;
}
else
{
lean_object* v_reuseFailAlloc_4355_; 
v_reuseFailAlloc_4355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4355_, 0, v_a_4349_);
v___x_4354_ = v_reuseFailAlloc_4355_;
goto v_reusejp_4353_;
}
v_reusejp_4353_:
{
return v___x_4354_;
}
}
}
}
else
{
lean_object* v_a_4357_; lean_object* v___x_4359_; uint8_t v_isShared_4360_; uint8_t v_isSharedCheck_4364_; 
lean_dec(v_a_4293_);
lean_dec(v_stx_4282_);
v_a_4357_ = lean_ctor_get(v___x_4294_, 0);
v_isSharedCheck_4364_ = !lean_is_exclusive(v___x_4294_);
if (v_isSharedCheck_4364_ == 0)
{
v___x_4359_ = v___x_4294_;
v_isShared_4360_ = v_isSharedCheck_4364_;
goto v_resetjp_4358_;
}
else
{
lean_inc(v_a_4357_);
lean_dec(v___x_4294_);
v___x_4359_ = lean_box(0);
v_isShared_4360_ = v_isSharedCheck_4364_;
goto v_resetjp_4358_;
}
v_resetjp_4358_:
{
lean_object* v___x_4362_; 
if (v_isShared_4360_ == 0)
{
v___x_4362_ = v___x_4359_;
goto v_reusejp_4361_;
}
else
{
lean_object* v_reuseFailAlloc_4363_; 
v_reuseFailAlloc_4363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4363_, 0, v_a_4357_);
v___x_4362_ = v_reuseFailAlloc_4363_;
goto v_reusejp_4361_;
}
v_reusejp_4361_:
{
return v___x_4362_;
}
}
}
}
else
{
lean_object* v_a_4365_; lean_object* v___x_4367_; uint8_t v_isShared_4368_; uint8_t v_isSharedCheck_4372_; 
lean_dec(v_stx_4282_);
v_a_4365_ = lean_ctor_get(v___x_4292_, 0);
v_isSharedCheck_4372_ = !lean_is_exclusive(v___x_4292_);
if (v_isSharedCheck_4372_ == 0)
{
v___x_4367_ = v___x_4292_;
v_isShared_4368_ = v_isSharedCheck_4372_;
goto v_resetjp_4366_;
}
else
{
lean_inc(v_a_4365_);
lean_dec(v___x_4292_);
v___x_4367_ = lean_box(0);
v_isShared_4368_ = v_isSharedCheck_4372_;
goto v_resetjp_4366_;
}
v_resetjp_4366_:
{
lean_object* v___x_4370_; 
if (v_isShared_4368_ == 0)
{
v___x_4370_ = v___x_4367_;
goto v_reusejp_4369_;
}
else
{
lean_object* v_reuseFailAlloc_4371_; 
v_reuseFailAlloc_4371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4371_, 0, v_a_4365_);
v___x_4370_ = v_reuseFailAlloc_4371_;
goto v_reusejp_4369_;
}
v_reusejp_4369_:
{
return v___x_4370_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___boxed(lean_object* v_stx_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_, lean_object* v___y_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_, lean_object* v___y_4380_, lean_object* v___y_4381_, lean_object* v___y_4382_){
_start:
{
lean_object* v_res_4383_; 
v_res_4383_ = lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0(v_stx_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_, v___y_4379_, v___y_4380_, v___y_4381_);
lean_dec(v___y_4381_);
lean_dec_ref(v___y_4380_);
lean_dec(v___y_4379_);
lean_dec_ref(v___y_4378_);
lean_dec(v___y_4377_);
lean_dec_ref(v___y_4376_);
lean_dec(v___y_4375_);
lean_dec_ref(v___y_4374_);
return v_res_4383_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl(lean_object* v_stx_4384_, lean_object* v_a_4385_, lean_object* v_a_4386_, lean_object* v_a_4387_, lean_object* v_a_4388_, lean_object* v_a_4389_, lean_object* v_a_4390_, lean_object* v_a_4391_, lean_object* v_a_4392_){
_start:
{
lean_object* v___f_4394_; lean_object* v___x_4395_; 
v___f_4394_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___lam__0___boxed), 10, 1);
lean_closure_set(v___f_4394_, 0, v_stx_4384_);
v___x_4395_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4394_, v_a_4385_, v_a_4386_, v_a_4387_, v_a_4388_, v_a_4389_, v_a_4390_, v_a_4391_, v_a_4392_);
return v___x_4395_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl___boxed(lean_object* v_stx_4396_, lean_object* v_a_4397_, lean_object* v_a_4398_, lean_object* v_a_4399_, lean_object* v_a_4400_, lean_object* v_a_4401_, lean_object* v_a_4402_, lean_object* v_a_4403_, lean_object* v_a_4404_, lean_object* v_a_4405_){
_start:
{
lean_object* v_res_4406_; 
v_res_4406_ = lp_LeanSearchClient_LeanSearchClient_stateSearchTacticImpl(v_stx_4396_, v_a_4397_, v_a_4398_, v_a_4399_, v_a_4400_, v_a_4401_, v_a_4402_, v_a_4403_, v_a_4404_);
lean_dec(v_a_4404_);
lean_dec_ref(v_a_4403_);
lean_dec(v_a_4402_);
lean_dec_ref(v_a_4401_);
lean_dec(v_a_4400_);
lean_dec_ref(v_a_4399_);
lean_dec(v_a_4398_);
lean_dec_ref(v_a_4397_);
return v_res_4406_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0(lean_object* v_stx_4407_, uint8_t v___x_4408_, lean_object* v_a_4409_, lean_object* v_as_4410_, size_t v_sz_4411_, size_t v_i_4412_, uint8_t v_b_4413_, lean_object* v___y_4414_, lean_object* v___y_4415_, lean_object* v___y_4416_, lean_object* v___y_4417_, lean_object* v___y_4418_, lean_object* v___y_4419_, lean_object* v___y_4420_, lean_object* v___y_4421_){
_start:
{
lean_object* v___x_4423_; 
v___x_4423_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___redArg(v_stx_4407_, v___x_4408_, v_a_4409_, v_as_4410_, v_sz_4411_, v_i_4412_, v_b_4413_, v___y_4416_, v___y_4417_, v___y_4418_, v___y_4419_, v___y_4420_, v___y_4421_);
return v___x_4423_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0___boxed(lean_object* v_stx_4424_, lean_object* v___x_4425_, lean_object* v_a_4426_, lean_object* v_as_4427_, lean_object* v_sz_4428_, lean_object* v_i_4429_, lean_object* v_b_4430_, lean_object* v___y_4431_, lean_object* v___y_4432_, lean_object* v___y_4433_, lean_object* v___y_4434_, lean_object* v___y_4435_, lean_object* v___y_4436_, lean_object* v___y_4437_, lean_object* v___y_4438_, lean_object* v___y_4439_){
_start:
{
uint8_t v___x_7528__boxed_4440_; size_t v_sz_boxed_4441_; size_t v_i_boxed_4442_; uint8_t v_b_boxed_4443_; lean_object* v_res_4444_; 
v___x_7528__boxed_4440_ = lean_unbox(v___x_4425_);
v_sz_boxed_4441_ = lean_unbox_usize(v_sz_4428_);
lean_dec(v_sz_4428_);
v_i_boxed_4442_ = lean_unbox_usize(v_i_4429_);
lean_dec(v_i_4429_);
v_b_boxed_4443_ = lean_unbox(v_b_4430_);
v_res_4444_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_stateSearchTacticImpl_spec__0(v_stx_4424_, v___x_7528__boxed_4440_, v_a_4426_, v_as_4427_, v_sz_boxed_4441_, v_i_boxed_4442_, v_b_boxed_4443_, v___y_4431_, v___y_4432_, v___y_4433_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_, v___y_4438_);
lean_dec(v___y_4438_);
lean_dec_ref(v___y_4437_);
lean_dec(v___y_4436_);
lean_dec_ref(v___y_4435_);
lean_dec(v___y_4434_);
lean_dec_ref(v___y_4433_);
lean_dec(v___y_4432_);
lean_dec_ref(v___y_4431_);
lean_dec_ref(v_as_4427_);
return v_res_4444_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg(lean_object* v_msg_4461_, lean_object* v___y_4462_, lean_object* v___y_4463_, lean_object* v___y_4464_, lean_object* v___y_4465_){
_start:
{
lean_object* v_ref_4467_; lean_object* v___x_4468_; lean_object* v_a_4469_; lean_object* v___x_4471_; uint8_t v_isShared_4472_; uint8_t v_isSharedCheck_4477_; 
v_ref_4467_ = lean_ctor_get(v___y_4464_, 5);
v___x_4468_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00LeanSearchClient_SearchServer_searchCommandSuggestions_spec__0_spec__0_spec__1_spec__2(v_msg_4461_, v___y_4462_, v___y_4463_, v___y_4464_, v___y_4465_);
v_a_4469_ = lean_ctor_get(v___x_4468_, 0);
v_isSharedCheck_4477_ = !lean_is_exclusive(v___x_4468_);
if (v_isSharedCheck_4477_ == 0)
{
v___x_4471_ = v___x_4468_;
v_isShared_4472_ = v_isSharedCheck_4477_;
goto v_resetjp_4470_;
}
else
{
lean_inc(v_a_4469_);
lean_dec(v___x_4468_);
v___x_4471_ = lean_box(0);
v_isShared_4472_ = v_isSharedCheck_4477_;
goto v_resetjp_4470_;
}
v_resetjp_4470_:
{
lean_object* v___x_4473_; lean_object* v___x_4475_; 
lean_inc(v_ref_4467_);
v___x_4473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4473_, 0, v_ref_4467_);
lean_ctor_set(v___x_4473_, 1, v_a_4469_);
if (v_isShared_4472_ == 0)
{
lean_ctor_set_tag(v___x_4471_, 1);
lean_ctor_set(v___x_4471_, 0, v___x_4473_);
v___x_4475_ = v___x_4471_;
goto v_reusejp_4474_;
}
else
{
lean_object* v_reuseFailAlloc_4476_; 
v_reuseFailAlloc_4476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4476_, 0, v___x_4473_);
v___x_4475_ = v_reuseFailAlloc_4476_;
goto v_reusejp_4474_;
}
v_reusejp_4474_:
{
return v___x_4475_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg___boxed(lean_object* v_msg_4478_, lean_object* v___y_4479_, lean_object* v___y_4480_, lean_object* v___y_4481_, lean_object* v___y_4482_, lean_object* v___y_4483_){
_start:
{
lean_object* v_res_4484_; 
v_res_4484_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg(v_msg_4478_, v___y_4479_, v___y_4480_, v___y_4481_, v___y_4482_);
lean_dec(v___y_4482_);
lean_dec_ref(v___y_4481_);
lean_dec(v___y_4480_);
lean_dec_ref(v___y_4479_);
return v_res_4484_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___lam__0(uint8_t v___x_4485_, lean_object* v_stx_4486_, lean_object* v___x_4487_, lean_object* v___y_4488_, lean_object* v___y_4489_, lean_object* v___y_4490_, lean_object* v___y_4491_, lean_object* v___y_4492_, lean_object* v___y_4493_, lean_object* v___y_4494_, lean_object* v___y_4495_){
_start:
{
if (v___x_4485_ == 0)
{
lean_object* v___x_4497_; 
lean_dec_ref(v___x_4487_);
lean_dec(v_stx_4486_);
v___x_4497_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v___x_4497_;
}
else
{
lean_object* v___x_4498_; lean_object* v___x_4499_; lean_object* v___x_4500_; uint8_t v___x_4501_; 
v___x_4498_ = lean_unsigned_to_nat(0u);
v___x_4499_ = lean_unsigned_to_nat(1u);
v___x_4500_ = l_Lean_Syntax_getArg(v_stx_4486_, v___x_4499_);
lean_inc(v___x_4500_);
v___x_4501_ = l_Lean_Syntax_matchesNull(v___x_4500_, v___x_4499_);
if (v___x_4501_ == 0)
{
uint8_t v___x_4502_; 
lean_dec(v_stx_4486_);
v___x_4502_ = l_Lean_Syntax_matchesNull(v___x_4500_, v___x_4498_);
if (v___x_4502_ == 0)
{
lean_object* v___x_4503_; 
lean_dec_ref(v___x_4487_);
v___x_4503_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_leanSearchTacticImpl_spec__0___redArg();
return v___x_4503_;
}
else
{
lean_object* v_ref_4504_; lean_object* v___x_4505_; lean_object* v___x_4506_; lean_object* v___x_4507_; lean_object* v___x_4508_; lean_object* v___x_4509_; lean_object* v___x_4510_; lean_object* v___x_4511_; 
v_ref_4504_ = lean_ctor_get(v___y_4494_, 5);
v___x_4505_ = l_Lean_SourceInfo_fromRef(v_ref_4504_, v___x_4501_);
v___x_4506_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__0));
v___x_4507_ = l_Lean_Name_mkStr2(v___x_4487_, v___x_4506_);
v___x_4508_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_statesearch__search__tactic___closed__2));
lean_inc(v___x_4505_);
v___x_4509_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4509_, 0, v___x_4505_);
lean_ctor_set(v___x_4509_, 1, v___x_4508_);
v___x_4510_ = l_Lean_Syntax_node1(v___x_4505_, v___x_4507_, v___x_4509_);
v___x_4511_ = l_Lean_Elab_Tactic_evalTactic(v___x_4510_, v___y_4488_, v___y_4489_, v___y_4490_, v___y_4491_, v___y_4492_, v___y_4493_, v___y_4494_, v___y_4495_);
return v___x_4511_;
}
}
else
{
lean_object* v_options_4512_; lean_object* v_s_4513_; lean_object* v___x_4514_; lean_object* v___x_4515_; lean_object* v___x_4516_; uint8_t v___x_4517_; 
lean_dec_ref(v___x_4487_);
v_options_4512_ = lean_ctor_get(v___y_4494_, 2);
v_s_4513_ = l_Lean_Syntax_getArg(v___x_4500_, v___x_4498_);
lean_dec(v___x_4500_);
v___x_4514_ = lp_LeanSearchClient_leansearchclient_backend;
v___x_4515_ = lp_LeanSearchClient_Lean_Option_get___at___00LeanSearchClient_useragent_spec__0(v_options_4512_, v___x_4514_);
v___x_4516_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__0));
v___x_4517_ = lean_string_dec_eq(v___x_4515_, v___x_4516_);
if (v___x_4517_ == 0)
{
lean_object* v___x_4518_; lean_object* v___x_4519_; lean_object* v___x_4520_; lean_object* v___x_4521_; lean_object* v___x_4522_; lean_object* v___x_4523_; lean_object* v___x_4524_; lean_object* v_a_4525_; lean_object* v___x_4527_; uint8_t v_isShared_4528_; uint8_t v_isSharedCheck_4532_; 
lean_dec(v_s_4513_);
lean_dec(v_stx_4486_);
v___x_4518_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchCommandImpl___closed__1));
v___x_4519_ = lean_string_append(v___x_4518_, v___x_4515_);
lean_dec_ref(v___x_4515_);
v___x_4520_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_searchTermImpl___closed__0));
v___x_4521_ = lean_string_append(v___x_4519_, v___x_4520_);
v___x_4522_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4522_, 0, v___x_4521_);
v___x_4523_ = l_Lean_MessageData_ofFormat(v___x_4522_);
v___x_4524_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg(v___x_4523_, v___y_4492_, v___y_4493_, v___y_4494_, v___y_4495_);
v_a_4525_ = lean_ctor_get(v___x_4524_, 0);
v_isSharedCheck_4532_ = !lean_is_exclusive(v___x_4524_);
if (v_isSharedCheck_4532_ == 0)
{
v___x_4527_ = v___x_4524_;
v_isShared_4528_ = v_isSharedCheck_4532_;
goto v_resetjp_4526_;
}
else
{
lean_inc(v_a_4525_);
lean_dec(v___x_4524_);
v___x_4527_ = lean_box(0);
v_isShared_4528_ = v_isSharedCheck_4532_;
goto v_resetjp_4526_;
}
v_resetjp_4526_:
{
lean_object* v___x_4530_; 
if (v_isShared_4528_ == 0)
{
v___x_4530_ = v___x_4527_;
goto v_reusejp_4529_;
}
else
{
lean_object* v_reuseFailAlloc_4531_; 
v_reuseFailAlloc_4531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4531_, 0, v_a_4525_);
v___x_4530_ = v_reuseFailAlloc_4531_;
goto v_reusejp_4529_;
}
v_reusejp_4529_:
{
return v___x_4530_;
}
}
}
else
{
lean_object* v___x_4533_; lean_object* v___x_4534_; 
lean_dec_ref(v___x_4515_);
v___x_4533_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leanSearchServer));
v___x_4534_ = lp_LeanSearchClient_LeanSearchClient_SearchServer_searchTacticSuggestions(v___x_4533_, v_stx_4486_, v_s_4513_, v___y_4488_, v___y_4489_, v___y_4490_, v___y_4491_, v___y_4492_, v___y_4493_, v___y_4494_, v___y_4495_);
lean_dec(v_s_4513_);
return v___x_4534_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___lam__0___boxed(lean_object* v___x_4535_, lean_object* v_stx_4536_, lean_object* v___x_4537_, lean_object* v___y_4538_, lean_object* v___y_4539_, lean_object* v___y_4540_, lean_object* v___y_4541_, lean_object* v___y_4542_, lean_object* v___y_4543_, lean_object* v___y_4544_, lean_object* v___y_4545_, lean_object* v___y_4546_){
_start:
{
uint8_t v___x_4229__boxed_4547_; lean_object* v_res_4548_; 
v___x_4229__boxed_4547_ = lean_unbox(v___x_4535_);
v_res_4548_ = lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___lam__0(v___x_4229__boxed_4547_, v_stx_4536_, v___x_4537_, v___y_4538_, v___y_4539_, v___y_4540_, v___y_4541_, v___y_4542_, v___y_4543_, v___y_4544_, v___y_4545_);
lean_dec(v___y_4545_);
lean_dec_ref(v___y_4544_);
lean_dec(v___y_4543_);
lean_dec_ref(v___y_4542_);
lean_dec(v___y_4541_);
lean_dec_ref(v___y_4540_);
lean_dec(v___y_4539_);
lean_dec_ref(v___y_4538_);
return v_res_4548_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl(lean_object* v_stx_4549_, lean_object* v_a_4550_, lean_object* v_a_4551_, lean_object* v_a_4552_, lean_object* v_a_4553_, lean_object* v_a_4554_, lean_object* v_a_4555_, lean_object* v_a_4556_, lean_object* v_a_4557_){
_start:
{
lean_object* v___x_4559_; lean_object* v___x_4560_; uint8_t v___x_4561_; lean_object* v___x_4562_; lean_object* v___y_4563_; lean_object* v___x_4564_; 
v___x_4559_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_leansearch__search__cmd___closed__0));
v___x_4560_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_search__tactic___closed__1));
lean_inc(v_stx_4549_);
v___x_4561_ = l_Lean_Syntax_isOfKind(v_stx_4549_, v___x_4560_);
v___x_4562_ = lean_box(v___x_4561_);
v___y_4563_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___lam__0___boxed), 12, 3);
lean_closure_set(v___y_4563_, 0, v___x_4562_);
lean_closure_set(v___y_4563_, 1, v_stx_4549_);
lean_closure_set(v___y_4563_, 2, v___x_4559_);
v___x_4564_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_4563_, v_a_4550_, v_a_4551_, v_a_4552_, v_a_4553_, v_a_4554_, v_a_4555_, v_a_4556_, v_a_4557_);
return v___x_4564_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_searchTacticImpl___boxed(lean_object* v_stx_4565_, lean_object* v_a_4566_, lean_object* v_a_4567_, lean_object* v_a_4568_, lean_object* v_a_4569_, lean_object* v_a_4570_, lean_object* v_a_4571_, lean_object* v_a_4572_, lean_object* v_a_4573_, lean_object* v_a_4574_){
_start:
{
lean_object* v_res_4575_; 
v_res_4575_ = lp_LeanSearchClient_LeanSearchClient_searchTacticImpl(v_stx_4565_, v_a_4566_, v_a_4567_, v_a_4568_, v_a_4569_, v_a_4570_, v_a_4571_, v_a_4572_, v_a_4573_);
lean_dec(v_a_4573_);
lean_dec_ref(v_a_4572_);
lean_dec(v_a_4571_);
lean_dec_ref(v_a_4570_);
lean_dec(v_a_4569_);
lean_dec_ref(v_a_4568_);
lean_dec(v_a_4567_);
lean_dec_ref(v_a_4566_);
return v_res_4575_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0(lean_object* v_00_u03b1_4576_, lean_object* v_msg_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_, lean_object* v___y_4580_, lean_object* v___y_4581_, lean_object* v___y_4582_, lean_object* v___y_4583_, lean_object* v___y_4584_, lean_object* v___y_4585_){
_start:
{
lean_object* v___x_4587_; 
v___x_4587_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___redArg(v_msg_4577_, v___y_4582_, v___y_4583_, v___y_4584_, v___y_4585_);
return v___x_4587_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0___boxed(lean_object* v_00_u03b1_4588_, lean_object* v_msg_4589_, lean_object* v___y_4590_, lean_object* v___y_4591_, lean_object* v___y_4592_, lean_object* v___y_4593_, lean_object* v___y_4594_, lean_object* v___y_4595_, lean_object* v___y_4596_, lean_object* v___y_4597_, lean_object* v___y_4598_){
_start:
{
lean_object* v_res_4599_; 
v_res_4599_ = lp_LeanSearchClient_Lean_throwError___at___00LeanSearchClient_searchTacticImpl_spec__0(v_00_u03b1_4588_, v_msg_4589_, v___y_4590_, v___y_4591_, v___y_4592_, v___y_4593_, v___y_4594_, v___y_4595_, v___y_4596_, v___y_4597_);
lean_dec(v___y_4597_);
lean_dec_ref(v___y_4596_);
lean_dec(v___y_4595_);
lean_dec_ref(v___y_4594_);
lean_dec(v___y_4593_);
lean_dec_ref(v___y_4592_);
lean_dec(v___y_4591_);
lean_dec_ref(v___y_4590_);
return v_res_4599_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient_Syntax(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_Utils(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_LeanSearchClient_LeanSearchClient_Syntax(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_Utils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_709949654____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_LeanSearchClient_leanSearchCache = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_leanSearchCache);
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Syntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_Syntax_857704034____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_LeanSearchClient_stateSearchCache = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_stateSearchCache);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin);
lean_object* initialize_Lean_Server_Utils(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LeanSearchClient_LeanSearchClient_Syntax(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_Utils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_LeanSearchClient_LeanSearchClient_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_LeanSearchClient_LeanSearchClient_Syntax(builtin);
}
#ifdef __cplusplus
}
#endif
