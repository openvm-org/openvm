// Lean compiler output
// Module: LeanSearchClient.LoogleSyntax
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Meta public meta import Lean.Parser.Basic public meta import Lean.Meta.Tactic.TryThis public meta import LeanSearchClient.Basic public meta import LeanSearchClient.Syntax
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
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
lean_object* l_String_Slice_slice_x21(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
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
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Json_getStr_x3f(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_IO_throwServerError___redArg(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_io_getenv(lean_object*);
lean_object* l_System_Uri_escapeUri(lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_useragent___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_IO_Process_output(lean_object*, lean_object*);
lean_object* l_Lean_Json_parse(lean_object*);
uint8_t l___private_Lean_Data_Json_Basic_0__Lean_Json_beq_x27(lean_object*, lean_object*);
lean_object* l_Lean_Json_getArr_x3f(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion(lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(lean_object*);
lean_object* l_Lean_Parser_nonReservedSymbol_parenthesizer___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_checkTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_nonReservedSymbol(lean_object*, uint8_t);
lean_object* l_String_quote(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_Lean_Parser_nonReservedSymbol_formatter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions(lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion(lean_object*);
lean_object* lp_LeanSearchClient_LeanSearchClient_defaultTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__1_value;
static const lean_string_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__1 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__1_value;
static const lean_string_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "some "};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__2 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__3 = (const lean_object*)&lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__3_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Nat_cast___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__1(lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "name"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__1_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__5_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__6_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__7;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__9_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "type"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__10_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__10_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__11_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "doc\?"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__12 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__12_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__12_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__13_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__14_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__15;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__16;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__17 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__17_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__14_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__18 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__18_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_empty_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_empty_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_success_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_success_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_failure_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_failure_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult_default;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult;
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "#["};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__9_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__1 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__1_value;
static const lean_string_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__2 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__2_value;
static lean_once_cell_t lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__3;
static lean_once_cell_t lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__4;
static const lean_ctor_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__5 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__6 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__6_value;
static const lean_string_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "#[]"};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__7 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__7_value)}};
static const lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__8 = (const lean_object*)&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__8_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4_spec__6_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__1_value;
static const lean_string_object lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__3;
static lean_once_cell_t lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__4;
static const lean_ctor_object lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__5 = (const lean_object*)&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "LeanSearchClient.LoogleResult.empty"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__1_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "LeanSearchClient.LoogleResult.success"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "LeanSearchClient.LoogleResult.failure"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__7_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__8_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__9_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult___closed__0_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCache;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00LeanSearchClient_getLoogleQueryJson_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12_spec__16(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12_spec__16___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected JSON array, got '"};
static const lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__0 = (const lean_object*)&lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__0_value;
static const lean_string_object lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__1 = (const lean_object*)&lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__2;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__4;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__5;
static const lean_ctor_object lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__6 = (const lean_object*)&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Could not obtain name and type from "};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__0_value;
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "doc"};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__1 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "/-"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "error"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "suggestions"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Could not obtain hits or error from "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__4_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "LEANSEARCHCLIENT_LOOGLE_API_URL"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__5_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\?q="};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__6_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__7_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "curl"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__8_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "-X"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__9_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "GET"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__10_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "--user-agent"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__11_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__12;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__13;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__14;
static const lean_array_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__15 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__15_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Could not contact Loogle server"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__16 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__16_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hits"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__17 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__17_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Could not obtain array from "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__18 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__18_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "; error: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__19 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__19_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = ", query :"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__20 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__20_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", hits: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__21 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__21_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "https://loogle.lean-lang.org/json"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__22 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__22_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00LeanSearchClient_getLoogleQueryJson_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8_spec__12(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogleUsage___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1551, .m_capacity = 1551, .m_length = 1515, .m_data = "Loogle Usage\n\nLoogle finds definitions and lemmas in various ways:\n\nBy constant:\n🔍 Real.sin\nfinds all lemmas whose statement somehow mentions the sine function.\n\nBy lemma name substring:\n🔍 \"differ\"\nfinds all lemmas that have \"differ\" somewhere in their lemma name.\n\nBy subexpression:\n🔍 _ * (_ ^ _)\nfinds all lemmas whose statements somewhere include a product where the second argument is raised to some power.\n\nThe pattern can also be non-linear, as in\n🔍 Real.sqrt \?a * Real.sqrt \?a\n\nIf the pattern has parameters, they are matched in any order. Both of these will find List.map:\n🔍 (\?a -> \?b) -> List \?a -> List \?b\n🔍 List \?a -> (\?a -> \?b) -> List \?b\n\nBy main conclusion:\n🔍 |- tsum _ = _ * tsum _\nfinds all lemmas where the conclusion (the subexpression to the right of all → and ∀) has the given shape.\n\nAs before, if the pattern has parameters, they are matched against the hypotheses of the lemma in any order; for example,\n🔍 |- _ < _ → tsum _ < tsum _\nwill find tsum_lt_tsum even though the hypothesis f i < g i is not the last.\n\nIf you pass more than one such search filter, separated by commas Loogle will return lemmas which match all of them. The search\n🔍 Real.sin, \"two\", tsum, _ * _, _ ^ _, |- _ < _ → _\nwould find all lemmas which mention the constants Real.sin and tsum, have \"two\" as a substring of the lemma name, include a product and a power somewhere in the type, and have a hypothesis of the form _ < _ (if there were any such lemmas). Metavariables (\?a) are assigned independently in each filter."};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleUsage___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleUsage___closed__0_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleUsage = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleUsage___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⊢ "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__1;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "|- "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__1;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "turnstyle"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__0_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "LeanSearchClient"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__2_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 46, 91, 12, 106, 122, 200, 192)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "patternIgnore"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__3_value),LEAN_SCALAR_PTR_LITERAL(195, 83, 213, 191, 208, 4, 123, 240)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__4_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__5_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "unicode_turnstile"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__8_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__7_value),LEAN_SCALAR_PTR_LITERAL(78, 65, 222, 66, 191, 234, 68, 129)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__9_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "ascii_turnstile"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__10_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__11_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__10_value),LEAN_SCALAR_PTR_LITERAL(214, 112, 16, 174, 104, 223, 56, 14)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__11_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__11_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__12 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__12_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__6_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__9_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__12_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__13_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__4_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__13_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__14_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__0_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__2_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__14_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__15 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__15_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_turnstyle = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__15_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "loogle_filter"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 13, 22, 24, 49, 104, 212, 182)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__6_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__15_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__9_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__9_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__10_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__6_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__10_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__11_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__0_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__1_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__11_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__12 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__12_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filter = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__12_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "loogle_filters"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__0_value),LEAN_SCALAR_PTR_LITERAL(33, 128, 119, 139, 25, 227, 154, 143)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__12_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__8_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__0_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__5_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__filters = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__5_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "loogle_cmd"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(98, 19, 224, 144, 227, 215, 57, 206)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#loogle"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__5_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__5_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__cmd = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__5_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg();
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleCmdImpl_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleCmdImpl_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "#loogle "};
static const lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4___closed__0 = (const lean_object*)&lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__0_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__1 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__1_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__2 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__2_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__3 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__3_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__4 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__4_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__5 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__5_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__6 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__6_value;
static const lean_string_object lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__7 = (const lean_object*)&lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleUsage___closed__0_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Loogle Search Results"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__2_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Loogle search returned no results"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__3_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__3_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__4_value;
static lean_once_cell_t lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Loogle search failed with error: "};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Did you maybe mean"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__7_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "just_loogle_cmd"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(60, 188, 208, 76, 21, 152, 23, 73)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__2_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__cmd___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___redArg();
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "loogle_term"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__0_value),LEAN_SCALAR_PTR_LITERAL(164, 130, 106, 1, 24, 250, 136, 94)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__4_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__2_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__term = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTermImpl_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTermImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\""};
static const lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0 = (const lean_object*)&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__2;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__4;
static lean_once_cell_t lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__5;
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\\\""};
static const lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__0 = (const lean_object*)&lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__0_value;
static const lean_string_object lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "#loogle \""};
static const lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__1 = (const lean_object*)&lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTermImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTermImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "loogle_tactic"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 148, 81, 178, 108, 87, 252, 44)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "withPosition"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__2_value),LEAN_SCALAR_PTR_LITERAL(246, 171, 180, 145, 132, 143, 108, 238)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__3_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__4 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__4_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__4_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__5 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__5_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__5_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__6 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__6_value;
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__7 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__7_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__7_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__8 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__8_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__8_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__9 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__9_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__6_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__9_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__10 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__10_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__10_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__5_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__11 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__11_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__filter___closed__5_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__11_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__12 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__12_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__3_value),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__12_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__13 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__13_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__13_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__14 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__14_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_loogle__tactic = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__14_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg();
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTacticImpl_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTacticImpl_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTacticImpl_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__0 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__1 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__1_value;
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__2 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__2_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "From: "};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__0 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__0_value;
static const lean_array_object lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__1 = (const lean_object*)&lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__1_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTacticImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTacticImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "just_loogle_tactic"};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__0 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__0_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1_value),LEAN_SCALAR_PTR_LITERAL(174, 39, 126, 241, 34, 66, 12, 142)}};
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__1_value_aux_0),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(53, 36, 232, 150, 236, 43, 255, 217)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__1 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__1_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__2 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__2_value;
static const lean_ctor_object lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__2_value)}};
static const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__3 = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__3_value;
LEAN_EXPORT const lean_object* lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic = (const lean_object*)&lp_LeanSearchClient_LeanSearchClient_just__loogle__tactic___closed__3_value;
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0(lean_object* v_x_13_, lean_object* v_x_14_){
_start:
{
if (lean_obj_tag(v_x_13_) == 0)
{
lean_object* v___x_15_; 
v___x_15_ = ((lean_object*)(lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__1));
return v___x_15_;
}
else
{
lean_object* v_val_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_27_; 
v_val_16_ = lean_ctor_get(v_x_13_, 0);
v_isSharedCheck_27_ = !lean_is_exclusive(v_x_13_);
if (v_isSharedCheck_27_ == 0)
{
v___x_18_ = v_x_13_;
v_isShared_19_ = v_isSharedCheck_27_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_val_16_);
lean_dec(v_x_13_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_27_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_23_; 
v___x_20_ = ((lean_object*)(lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__3));
v___x_21_ = l_String_quote(v_val_16_);
if (v_isShared_19_ == 0)
{
lean_ctor_set_tag(v___x_18_, 3);
lean_ctor_set(v___x_18_, 0, v___x_21_);
v___x_23_ = v___x_18_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v___x_21_);
v___x_23_ = v_reuseFailAlloc_26_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_24_, 0, v___x_20_);
lean_ctor_set(v___x_24_, 1, v___x_23_);
v___x_25_ = l_Repr_addAppParen(v___x_24_, v_x_14_);
return v___x_25_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___boxed(lean_object* v_x_28_, lean_object* v_x_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0(v_x_28_, v_x_29_);
lean_dec(v_x_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Nat_cast___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__1(lean_object* v_a_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_nat_to_int(v_a_31_);
return v___x_32_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_unsigned_to_nat(8u);
v___x_47_ = lean_nat_to_int(v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__15(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_58_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__0));
v___x_59_ = lean_string_length(v___x_58_);
return v___x_59_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__16(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_60_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__15, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__15_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__15);
v___x_61_ = lean_nat_to_int(v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg(lean_object* v_x_66_){
_start:
{
lean_object* v_name_67_; lean_object* v_type_68_; lean_object* v_doc_x3f_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; uint8_t v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_name_67_ = lean_ctor_get(v_x_66_, 0);
lean_inc_ref(v_name_67_);
v_type_68_ = lean_ctor_get(v_x_66_, 1);
lean_inc_ref(v_type_68_);
v_doc_x3f_69_ = lean_ctor_get(v_x_66_, 2);
lean_inc(v_doc_x3f_69_);
lean_dec_ref(v_x_66_);
v___x_70_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__5));
v___x_71_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__6));
v___x_72_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__7, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__7_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__7);
v___x_73_ = l_String_quote(v_name_67_);
v___x_74_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
v___x_75_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_72_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
v___x_76_ = 0;
v___x_77_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_77_, 0, v___x_75_);
lean_ctor_set_uint8(v___x_77_, sizeof(void*)*1, v___x_76_);
v___x_78_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_71_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
v___x_79_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__9));
v___x_80_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_78_);
lean_ctor_set(v___x_80_, 1, v___x_79_);
v___x_81_ = lean_box(1);
v___x_82_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_80_);
lean_ctor_set(v___x_82_, 1, v___x_81_);
v___x_83_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__11));
v___x_84_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_82_);
lean_ctor_set(v___x_84_, 1, v___x_83_);
v___x_85_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v___x_70_);
v___x_86_ = l_String_quote(v_type_68_);
v___x_87_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
v___x_88_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_72_);
lean_ctor_set(v___x_88_, 1, v___x_87_);
v___x_89_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set_uint8(v___x_89_, sizeof(void*)*1, v___x_76_);
v___x_90_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_85_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_79_);
v___x_92_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v___x_81_);
v___x_93_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__13));
v___x_94_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v___x_70_);
v___x_96_ = lean_unsigned_to_nat(0u);
v___x_97_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0(v_doc_x3f_69_, v___x_96_);
v___x_98_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_72_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
v___x_99_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set_uint8(v___x_99_, sizeof(void*)*1, v___x_76_);
v___x_100_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_100_, 0, v___x_95_);
lean_ctor_set(v___x_100_, 1, v___x_99_);
v___x_101_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__16, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__16_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__16);
v___x_102_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__17));
v___x_103_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v___x_100_);
v___x_104_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__18));
v___x_105_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_103_);
lean_ctor_set(v___x_105_, 1, v___x_104_);
v___x_106_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_101_);
lean_ctor_set(v___x_106_, 1, v___x_105_);
v___x_107_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set_uint8(v___x_107_, sizeof(void*)*1, v___x_76_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr(lean_object* v_x_108_, lean_object* v_prec_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg(v_x_108_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___boxed(lean_object* v_x_111_, lean_object* v_prec_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr(v_x_111_, v_prec_112_);
lean_dec(v_prec_112_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorIdx(lean_object* v_x_116_){
_start:
{
switch(lean_obj_tag(v_x_116_))
{
case 0:
{
lean_object* v___x_117_; 
v___x_117_ = lean_unsigned_to_nat(0u);
return v___x_117_;
}
case 1:
{
lean_object* v___x_118_; 
v___x_118_ = lean_unsigned_to_nat(1u);
return v___x_118_;
}
default: 
{
lean_object* v___x_119_; 
v___x_119_ = lean_unsigned_to_nat(2u);
return v___x_119_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorIdx___boxed(lean_object* v_x_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorIdx(v_x_120_);
lean_dec(v_x_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(lean_object* v_t_122_, lean_object* v_k_123_){
_start:
{
switch(lean_obj_tag(v_t_122_))
{
case 0:
{
return v_k_123_;
}
case 1:
{
lean_object* v_a_124_; lean_object* v___x_125_; 
v_a_124_ = lean_ctor_get(v_t_122_, 0);
lean_inc_ref(v_a_124_);
lean_dec_ref_known(v_t_122_, 1);
v___x_125_ = lean_apply_1(v_k_123_, v_a_124_);
return v___x_125_;
}
default: 
{
lean_object* v_error_126_; lean_object* v_suggestions_127_; lean_object* v___x_128_; 
v_error_126_ = lean_ctor_get(v_t_122_, 0);
lean_inc_ref(v_error_126_);
v_suggestions_127_ = lean_ctor_get(v_t_122_, 1);
lean_inc(v_suggestions_127_);
lean_dec_ref_known(v_t_122_, 2);
v___x_128_ = lean_apply_2(v_k_123_, v_error_126_, v_suggestions_127_);
return v___x_128_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim(lean_object* v_motive_129_, lean_object* v_ctorIdx_130_, lean_object* v_t_131_, lean_object* v_h_132_, lean_object* v_k_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_131_, v_k_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___boxed(lean_object* v_motive_135_, lean_object* v_ctorIdx_136_, lean_object* v_t_137_, lean_object* v_h_138_, lean_object* v_k_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim(v_motive_135_, v_ctorIdx_136_, v_t_137_, v_h_138_, v_k_139_);
lean_dec(v_ctorIdx_136_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_empty_elim___redArg(lean_object* v_t_141_, lean_object* v_empty_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_141_, v_empty_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_empty_elim(lean_object* v_motive_144_, lean_object* v_t_145_, lean_object* v_h_146_, lean_object* v_empty_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_145_, v_empty_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_success_elim___redArg(lean_object* v_t_149_, lean_object* v_success_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_149_, v_success_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_success_elim(lean_object* v_motive_152_, lean_object* v_t_153_, lean_object* v_h_154_, lean_object* v_success_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_153_, v_success_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_failure_elim___redArg(lean_object* v_t_157_, lean_object* v_failure_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_157_, v_failure_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_LoogleResult_failure_elim(lean_object* v_motive_160_, lean_object* v_t_161_, lean_object* v_h_162_, lean_object* v_failure_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_LeanSearchClient_LeanSearchClient_LoogleResult_ctorElim___redArg(v_t_161_, v_failure_163_);
return v___x_164_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult_default(void){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lean_box(0);
return v___x_165_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult(void){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_box(0);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0_spec__1_spec__3(lean_object* v_x_167_, lean_object* v_x_168_, lean_object* v_x_169_){
_start:
{
if (lean_obj_tag(v_x_169_) == 0)
{
lean_dec(v_x_167_);
return v_x_168_;
}
else
{
lean_object* v_head_170_; lean_object* v_tail_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_181_; 
v_head_170_ = lean_ctor_get(v_x_169_, 0);
v_tail_171_ = lean_ctor_get(v_x_169_, 1);
v_isSharedCheck_181_ = !lean_is_exclusive(v_x_169_);
if (v_isSharedCheck_181_ == 0)
{
v___x_173_ = v_x_169_;
v_isShared_174_ = v_isSharedCheck_181_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_tail_171_);
lean_inc(v_head_170_);
lean_dec(v_x_169_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_181_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_176_; 
lean_inc(v_x_167_);
if (v_isShared_174_ == 0)
{
lean_ctor_set_tag(v___x_173_, 5);
lean_ctor_set(v___x_173_, 1, v_x_167_);
lean_ctor_set(v___x_173_, 0, v_x_168_);
v___x_176_ = v___x_173_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_180_; 
v_reuseFailAlloc_180_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_180_, 0, v_x_168_);
lean_ctor_set(v_reuseFailAlloc_180_, 1, v_x_167_);
v___x_176_ = v_reuseFailAlloc_180_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(v_head_170_);
v___x_178_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_176_);
lean_ctor_set(v___x_178_, 1, v___x_177_);
v_x_168_ = v___x_178_;
v_x_169_ = v_tail_171_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0_spec__1(lean_object* v_x_182_, lean_object* v_x_183_, lean_object* v_x_184_){
_start:
{
if (lean_obj_tag(v_x_184_) == 0)
{
lean_dec(v_x_182_);
return v_x_183_;
}
else
{
lean_object* v_head_185_; lean_object* v_tail_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_196_; 
v_head_185_ = lean_ctor_get(v_x_184_, 0);
v_tail_186_ = lean_ctor_get(v_x_184_, 1);
v_isSharedCheck_196_ = !lean_is_exclusive(v_x_184_);
if (v_isSharedCheck_196_ == 0)
{
v___x_188_ = v_x_184_;
v_isShared_189_ = v_isSharedCheck_196_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_tail_186_);
lean_inc(v_head_185_);
lean_dec(v_x_184_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_196_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v___x_191_; 
lean_inc(v_x_182_);
if (v_isShared_189_ == 0)
{
lean_ctor_set_tag(v___x_188_, 5);
lean_ctor_set(v___x_188_, 1, v_x_182_);
lean_ctor_set(v___x_188_, 0, v_x_183_);
v___x_191_ = v___x_188_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v_x_183_);
lean_ctor_set(v_reuseFailAlloc_195_, 1, v_x_182_);
v___x_191_ = v_reuseFailAlloc_195_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_192_ = lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(v_head_185_);
v___x_193_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_191_);
lean_ctor_set(v___x_193_, 1, v___x_192_);
v___x_194_ = lp_LeanSearchClient_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0_spec__1_spec__3(v_x_182_, v___x_193_, v_tail_186_);
return v___x_194_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0(lean_object* v_x_197_, lean_object* v_x_198_){
_start:
{
if (lean_obj_tag(v_x_197_) == 0)
{
lean_object* v___x_199_; 
lean_dec(v_x_198_);
v___x_199_ = lean_box(0);
return v___x_199_;
}
else
{
lean_object* v_tail_200_; 
v_tail_200_ = lean_ctor_get(v_x_197_, 1);
if (lean_obj_tag(v_tail_200_) == 0)
{
lean_object* v_head_201_; lean_object* v___x_202_; 
lean_dec(v_x_198_);
v_head_201_ = lean_ctor_get(v_x_197_, 0);
lean_inc(v_head_201_);
lean_dec_ref_known(v_x_197_, 2);
v___x_202_ = lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(v_head_201_);
return v___x_202_;
}
else
{
lean_object* v_head_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
lean_inc(v_tail_200_);
v_head_203_ = lean_ctor_get(v_x_197_, 0);
lean_inc(v_head_203_);
lean_dec_ref_known(v_x_197_, 2);
v___x_204_ = lp_LeanSearchClient_LeanSearchClient_instReprSearchResult_repr___redArg(v_head_203_);
v___x_205_ = lp_LeanSearchClient_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0_spec__1(v_x_198_, v___x_204_, v_tail_200_);
return v___x_205_;
}
}
}
}
static lean_object* _init_lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__3(void){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__0));
v___x_212_ = lean_string_length(v___x_211_);
return v___x_212_;
}
}
static lean_object* _init_lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__4(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_213_ = lean_obj_once(&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__3, &lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__3_once, _init_lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__3);
v___x_214_ = lean_nat_to_int(v___x_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0(lean_object* v_xs_222_){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_223_ = lean_array_get_size(v_xs_222_);
v___x_224_ = lean_unsigned_to_nat(0u);
v___x_225_ = lean_nat_dec_eq(v___x_223_, v___x_224_);
if (v___x_225_ == 0)
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_226_ = lean_array_to_list(v_xs_222_);
v___x_227_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__1));
v___x_228_ = lp_LeanSearchClient_Std_Format_joinSep___at___00Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0_spec__0(v___x_226_, v___x_227_);
v___x_229_ = lean_obj_once(&lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__4, &lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__4_once, _init_lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__4);
v___x_230_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__5));
v___x_231_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
lean_ctor_set(v___x_231_, 1, v___x_228_);
v___x_232_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__6));
v___x_233_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_231_);
lean_ctor_set(v___x_233_, 1, v___x_232_);
v___x_234_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_229_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
v___x_235_ = l_Std_Format_fill(v___x_234_);
return v___x_235_;
}
else
{
lean_object* v___x_236_; 
lean_dec_ref(v_xs_222_);
v___x_236_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__8));
return v___x_236_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4_spec__6_spec__7(lean_object* v_x_237_, lean_object* v_x_238_, lean_object* v_x_239_){
_start:
{
if (lean_obj_tag(v_x_239_) == 0)
{
lean_dec(v_x_237_);
return v_x_238_;
}
else
{
lean_object* v_head_240_; lean_object* v_tail_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_252_; 
v_head_240_ = lean_ctor_get(v_x_239_, 0);
v_tail_241_ = lean_ctor_get(v_x_239_, 1);
v_isSharedCheck_252_ = !lean_is_exclusive(v_x_239_);
if (v_isSharedCheck_252_ == 0)
{
v___x_243_ = v_x_239_;
v_isShared_244_ = v_isSharedCheck_252_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_tail_241_);
lean_inc(v_head_240_);
lean_dec(v_x_239_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_252_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
lean_inc(v_x_237_);
if (v_isShared_244_ == 0)
{
lean_ctor_set_tag(v___x_243_, 5);
lean_ctor_set(v___x_243_, 1, v_x_237_);
lean_ctor_set(v___x_243_, 0, v_x_238_);
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_251_; 
v_reuseFailAlloc_251_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_251_, 0, v_x_238_);
lean_ctor_set(v_reuseFailAlloc_251_, 1, v_x_237_);
v___x_246_ = v_reuseFailAlloc_251_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_247_ = l_String_quote(v_head_240_);
v___x_248_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_248_, 0, v___x_247_);
v___x_249_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_246_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
v_x_238_ = v___x_249_;
v_x_239_ = v_tail_241_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4_spec__6(lean_object* v_x_253_, lean_object* v_x_254_, lean_object* v_x_255_){
_start:
{
if (lean_obj_tag(v_x_255_) == 0)
{
lean_dec(v_x_253_);
return v_x_254_;
}
else
{
lean_object* v_head_256_; lean_object* v_tail_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_268_; 
v_head_256_ = lean_ctor_get(v_x_255_, 0);
v_tail_257_ = lean_ctor_get(v_x_255_, 1);
v_isSharedCheck_268_ = !lean_is_exclusive(v_x_255_);
if (v_isSharedCheck_268_ == 0)
{
v___x_259_ = v_x_255_;
v_isShared_260_ = v_isSharedCheck_268_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_tail_257_);
lean_inc(v_head_256_);
lean_dec(v_x_255_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_268_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
lean_inc(v_x_253_);
if (v_isShared_260_ == 0)
{
lean_ctor_set_tag(v___x_259_, 5);
lean_ctor_set(v___x_259_, 1, v_x_253_);
lean_ctor_set(v___x_259_, 0, v_x_254_);
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v_x_254_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_x_253_);
v___x_262_ = v_reuseFailAlloc_267_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_263_ = l_String_quote(v_head_256_);
v___x_264_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_264_, 0, v___x_263_);
v___x_265_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_262_);
lean_ctor_set(v___x_265_, 1, v___x_264_);
v___x_266_ = lp_LeanSearchClient_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4_spec__6_spec__7(v_x_253_, v___x_265_, v_tail_257_);
return v___x_266_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4___lam__0(lean_object* v___y_269_){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = l_String_quote(v___y_269_);
v___x_271_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4(lean_object* v_x_272_, lean_object* v_x_273_){
_start:
{
if (lean_obj_tag(v_x_272_) == 0)
{
lean_object* v___x_274_; 
lean_dec(v_x_273_);
v___x_274_ = lean_box(0);
return v___x_274_;
}
else
{
lean_object* v_tail_275_; 
v_tail_275_ = lean_ctor_get(v_x_272_, 1);
if (lean_obj_tag(v_tail_275_) == 0)
{
lean_object* v_head_276_; lean_object* v___x_277_; 
lean_dec(v_x_273_);
v_head_276_ = lean_ctor_get(v_x_272_, 0);
lean_inc(v_head_276_);
lean_dec_ref_known(v_x_272_, 2);
v___x_277_ = lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4___lam__0(v_head_276_);
return v___x_277_;
}
else
{
lean_object* v_head_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
lean_inc(v_tail_275_);
v_head_278_ = lean_ctor_get(v_x_272_, 0);
lean_inc(v_head_278_);
lean_dec_ref_known(v_x_272_, 2);
v___x_279_ = lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4___lam__0(v_head_278_);
v___x_280_ = lp_LeanSearchClient_List_foldl___at___00Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4_spec__6(v_x_273_, v___x_279_, v_tail_275_);
return v___x_280_;
}
}
}
}
static lean_object* _init_lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_285_ = ((lean_object*)(lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__2));
v___x_286_ = lean_string_length(v___x_285_);
return v___x_286_;
}
}
static lean_object* _init_lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__4(void){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_287_ = lean_obj_once(&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__3, &lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__3_once, _init_lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__3);
v___x_288_ = lean_nat_to_int(v___x_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg(lean_object* v_a_291_){
_start:
{
if (lean_obj_tag(v_a_291_) == 0)
{
lean_object* v___x_292_; 
v___x_292_ = ((lean_object*)(lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__1));
return v___x_292_;
}
else
{
lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_293_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__1));
v___x_294_ = lp_LeanSearchClient_Std_Format_joinSep___at___00List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2_spec__4(v_a_291_, v___x_293_);
v___x_295_ = lean_obj_once(&lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__4, &lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__4_once, _init_lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__4);
v___x_296_ = ((lean_object*)(lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg___closed__5));
v___x_297_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v___x_294_);
v___x_298_ = ((lean_object*)(lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0___closed__6));
v___x_299_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_297_);
lean_ctor_set(v___x_299_, 1, v___x_298_);
v___x_300_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_300_, 0, v___x_295_);
lean_ctor_set(v___x_300_, 1, v___x_299_);
v___x_301_ = l_Std_Format_fill(v___x_300_);
return v___x_301_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1(lean_object* v_x_302_, lean_object* v_x_303_){
_start:
{
if (lean_obj_tag(v_x_302_) == 0)
{
lean_object* v___x_304_; 
v___x_304_ = ((lean_object*)(lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__1));
return v___x_304_;
}
else
{
lean_object* v_val_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v_val_305_ = lean_ctor_get(v_x_302_, 0);
lean_inc(v_val_305_);
lean_dec_ref_known(v_x_302_, 1);
v___x_306_ = ((lean_object*)(lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleMatch_repr_spec__0___closed__3));
v___x_307_ = lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg(v_val_305_);
v___x_308_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_308_, 0, v___x_306_);
lean_ctor_set(v___x_308_, 1, v___x_307_);
v___x_309_ = l_Repr_addAppParen(v___x_308_, v_x_303_);
return v___x_309_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1___boxed(lean_object* v_x_310_, lean_object* v_x_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1(v_x_310_, v_x_311_);
lean_dec(v_x_311_);
return v_res_312_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2(void){
_start:
{
lean_object* v___x_316_; lean_object* v___x_317_; 
v___x_316_ = lean_unsigned_to_nat(2u);
v___x_317_ = lean_nat_to_int(v___x_316_);
return v___x_317_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3(void){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_318_ = lean_unsigned_to_nat(1u);
v___x_319_ = lean_nat_to_int(v___x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr(lean_object* v_x_332_, lean_object* v_prec_333_){
_start:
{
lean_object* v___y_335_; 
switch(lean_obj_tag(v_x_332_))
{
case 0:
{
lean_object* v___x_341_; uint8_t v___x_342_; 
v___x_341_ = lean_unsigned_to_nat(1024u);
v___x_342_ = lean_nat_dec_le(v___x_341_, v_prec_333_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; 
v___x_343_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2);
v___y_335_ = v___x_343_;
goto v___jp_334_;
}
else
{
lean_object* v___x_344_; 
v___x_344_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3);
v___y_335_ = v___x_344_;
goto v___jp_334_;
}
}
case 1:
{
lean_object* v_a_345_; lean_object* v___y_347_; lean_object* v___x_355_; uint8_t v___x_356_; 
v_a_345_ = lean_ctor_get(v_x_332_, 0);
lean_inc_ref(v_a_345_);
lean_dec_ref_known(v_x_332_, 1);
v___x_355_ = lean_unsigned_to_nat(1024u);
v___x_356_ = lean_nat_dec_le(v___x_355_, v_prec_333_);
if (v___x_356_ == 0)
{
lean_object* v___x_357_; 
v___x_357_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2);
v___y_347_ = v___x_357_;
goto v___jp_346_;
}
else
{
lean_object* v___x_358_; 
v___x_358_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3);
v___y_347_ = v___x_358_;
goto v___jp_346_;
}
v___jp_346_:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; uint8_t v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_348_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__6));
v___x_349_ = lp_LeanSearchClient_Array_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__0(v_a_345_);
v___x_350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_348_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
lean_inc(v___y_347_);
v___x_351_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_351_, 0, v___y_347_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
v___x_352_ = 0;
v___x_353_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_353_, 0, v___x_351_);
lean_ctor_set_uint8(v___x_353_, sizeof(void*)*1, v___x_352_);
v___x_354_ = l_Repr_addAppParen(v___x_353_, v_prec_333_);
return v___x_354_;
}
}
default: 
{
lean_object* v_error_359_; lean_object* v_suggestions_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_385_; 
v_error_359_ = lean_ctor_get(v_x_332_, 0);
v_suggestions_360_ = lean_ctor_get(v_x_332_, 1);
v_isSharedCheck_385_ = !lean_is_exclusive(v_x_332_);
if (v_isSharedCheck_385_ == 0)
{
v___x_362_ = v_x_332_;
v_isShared_363_ = v_isSharedCheck_385_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_suggestions_360_);
lean_inc(v_error_359_);
lean_dec(v_x_332_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_385_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___y_365_; lean_object* v___x_381_; uint8_t v___x_382_; 
v___x_381_ = lean_unsigned_to_nat(1024u);
v___x_382_ = lean_nat_dec_le(v___x_381_, v_prec_333_);
if (v___x_382_ == 0)
{
lean_object* v___x_383_; 
v___x_383_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__2);
v___y_365_ = v___x_383_;
goto v___jp_364_;
}
else
{
lean_object* v___x_384_; 
v___x_384_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3, &lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3_once, _init_lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__3);
v___y_365_ = v___x_384_;
goto v___jp_364_;
}
v___jp_364_:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_371_; 
v___x_366_ = lean_box(1);
v___x_367_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__9));
v___x_368_ = l_String_quote(v_error_359_);
v___x_369_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
if (v_isShared_363_ == 0)
{
lean_ctor_set_tag(v___x_362_, 5);
lean_ctor_set(v___x_362_, 1, v___x_369_);
lean_ctor_set(v___x_362_, 0, v___x_367_);
v___x_371_ = v___x_362_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v___x_367_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v___x_369_);
v___x_371_ = v_reuseFailAlloc_380_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; uint8_t v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_372_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_366_);
v___x_373_ = lean_unsigned_to_nat(1024u);
v___x_374_ = lp_LeanSearchClient_Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1(v_suggestions_360_, v___x_373_);
v___x_375_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_372_);
lean_ctor_set(v___x_375_, 1, v___x_374_);
lean_inc(v___y_365_);
v___x_376_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_376_, 0, v___y_365_);
lean_ctor_set(v___x_376_, 1, v___x_375_);
v___x_377_ = 0;
v___x_378_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_378_, 0, v___x_376_);
lean_ctor_set_uint8(v___x_378_, sizeof(void*)*1, v___x_377_);
v___x_379_ = l_Repr_addAppParen(v___x_378_, v_prec_333_);
return v___x_379_;
}
}
}
}
}
v___jp_334_:
{
lean_object* v___x_336_; lean_object* v___x_337_; uint8_t v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_336_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___closed__1));
lean_inc(v___y_335_);
v___x_337_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_337_, 0, v___y_335_);
lean_ctor_set(v___x_337_, 1, v___x_336_);
v___x_338_ = 0;
v___x_339_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_339_, 0, v___x_337_);
lean_ctor_set_uint8(v___x_339_, sizeof(void*)*1, v___x_338_);
v___x_340_ = l_Repr_addAppParen(v___x_339_, v_prec_333_);
return v___x_340_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr___boxed(lean_object* v_x_386_, lean_object* v_prec_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_LeanSearchClient_LeanSearchClient_instReprLoogleResult_repr(v_x_386_, v_prec_387_);
lean_dec(v_prec_387_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2(lean_object* v_a_389_, lean_object* v_n_390_){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___redArg(v_a_389_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2___boxed(lean_object* v_a_392_, lean_object* v_n_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_LeanSearchClient_List_repr_x27___at___00Option_repr___at___00LeanSearchClient_instReprLoogleResult_repr_spec__1_spec__2(v_a_392_, v_n_393_);
lean_dec(v_n_393_);
return v_res_394_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_397_ = lean_box(0);
v___x_398_ = lean_unsigned_to_nat(16u);
v___x_399_ = lean_mk_array(v___x_398_, v___x_397_);
return v___x_399_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_400_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_, &lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2__once, _init_lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__0_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_);
v___x_401_ = lean_unsigned_to_nat(0u);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
lean_ctor_set(v___x_402_, 1, v___x_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_404_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_, &lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2__once, _init_lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn___closed__1_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_);
v___x_405_ = lean_st_mk_ref(v___x_404_);
v___x_406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_406_, 0, v___x_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2____boxed(lean_object* v_a_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_();
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(lean_object* v_j_409_, lean_object* v_k_410_){
_start:
{
lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_411_ = l_Lean_Json_getObjValD(v_j_409_, v_k_410_);
v___x_412_ = l_Lean_Json_getStr_x3f(v___x_411_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2___boxed(lean_object* v_j_413_, lean_object* v_k_414_){
_start:
{
lean_object* v_res_415_; 
v_res_415_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(v_j_413_, v_k_414_);
lean_dec_ref(v_k_414_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__5(lean_object* v_j_416_, lean_object* v_k_417_){
_start:
{
lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_418_ = l_Lean_Json_getObjValD(v_j_416_, v_k_417_);
v___x_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__5___boxed(lean_object* v_j_420_, lean_object* v_k_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__5(v_j_420_, v_k_421_);
lean_dec_ref(v_k_421_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00LeanSearchClient_getLoogleQueryJson_spec__6___redArg(lean_object* v_a_423_, lean_object* v_b_424_){
_start:
{
lean_object* v_array_425_; lean_object* v_start_426_; lean_object* v_stop_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_440_; 
v_array_425_ = lean_ctor_get(v_a_423_, 0);
v_start_426_ = lean_ctor_get(v_a_423_, 1);
v_stop_427_ = lean_ctor_get(v_a_423_, 2);
v_isSharedCheck_440_ = !lean_is_exclusive(v_a_423_);
if (v_isSharedCheck_440_ == 0)
{
v___x_429_ = v_a_423_;
v_isShared_430_ = v_isSharedCheck_440_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_stop_427_);
lean_inc(v_start_426_);
lean_inc(v_array_425_);
lean_dec(v_a_423_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_440_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
uint8_t v___x_431_; 
v___x_431_ = lean_nat_dec_lt(v_start_426_, v_stop_427_);
if (v___x_431_ == 0)
{
lean_del_object(v___x_429_);
lean_dec(v_stop_427_);
lean_dec(v_start_426_);
lean_dec_ref(v_array_425_);
return v_b_424_;
}
else
{
lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_435_; 
v___x_432_ = lean_unsigned_to_nat(1u);
v___x_433_ = lean_nat_add(v_start_426_, v___x_432_);
lean_inc_ref(v_array_425_);
if (v_isShared_430_ == 0)
{
lean_ctor_set(v___x_429_, 1, v___x_433_);
v___x_435_ = v___x_429_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v_array_425_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v___x_433_);
lean_ctor_set(v_reuseFailAlloc_439_, 2, v_stop_427_);
v___x_435_ = v_reuseFailAlloc_439_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_436_ = lean_array_fget(v_array_425_, v_start_426_);
lean_dec(v_start_426_);
lean_dec_ref(v_array_425_);
v___x_437_ = lean_array_push(v_b_424_, v___x_436_);
v_a_423_ = v___x_435_;
v_b_424_ = v___x_437_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg(lean_object* v_a_441_, lean_object* v_x_442_){
_start:
{
if (lean_obj_tag(v_x_442_) == 0)
{
lean_object* v___x_443_; 
v___x_443_ = lean_box(0);
return v___x_443_;
}
else
{
lean_object* v_key_444_; lean_object* v_value_445_; lean_object* v_tail_446_; uint8_t v___y_448_; lean_object* v_fst_451_; lean_object* v_snd_452_; lean_object* v_fst_453_; lean_object* v_snd_454_; uint8_t v___x_455_; 
v_key_444_ = lean_ctor_get(v_x_442_, 0);
v_value_445_ = lean_ctor_get(v_x_442_, 1);
v_tail_446_ = lean_ctor_get(v_x_442_, 2);
v_fst_451_ = lean_ctor_get(v_key_444_, 0);
v_snd_452_ = lean_ctor_get(v_key_444_, 1);
v_fst_453_ = lean_ctor_get(v_a_441_, 0);
v_snd_454_ = lean_ctor_get(v_a_441_, 1);
v___x_455_ = lean_string_dec_eq(v_fst_451_, v_fst_453_);
if (v___x_455_ == 0)
{
v___y_448_ = v___x_455_;
goto v___jp_447_;
}
else
{
uint8_t v___x_456_; 
v___x_456_ = lean_nat_dec_eq(v_snd_452_, v_snd_454_);
v___y_448_ = v___x_456_;
goto v___jp_447_;
}
v___jp_447_:
{
if (v___y_448_ == 0)
{
v_x_442_ = v_tail_446_;
goto _start;
}
else
{
lean_object* v___x_450_; 
lean_inc(v_value_445_);
v___x_450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_450_, 0, v_value_445_);
return v___x_450_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg___boxed(lean_object* v_a_457_, lean_object* v_x_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg(v_a_457_, v_x_458_);
lean_dec(v_x_458_);
lean_dec_ref(v_a_457_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg(lean_object* v_m_460_, lean_object* v_a_461_){
_start:
{
lean_object* v_buckets_462_; lean_object* v_fst_463_; lean_object* v_snd_464_; lean_object* v___x_465_; uint64_t v___x_466_; uint64_t v___x_467_; uint64_t v___x_468_; uint64_t v___x_469_; uint64_t v___x_470_; uint64_t v_fold_471_; uint64_t v___x_472_; uint64_t v___x_473_; uint64_t v___x_474_; size_t v___x_475_; size_t v___x_476_; size_t v___x_477_; size_t v___x_478_; size_t v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v_buckets_462_ = lean_ctor_get(v_m_460_, 1);
v_fst_463_ = lean_ctor_get(v_a_461_, 0);
v_snd_464_ = lean_ctor_get(v_a_461_, 1);
v___x_465_ = lean_array_get_size(v_buckets_462_);
v___x_466_ = lean_string_hash(v_fst_463_);
v___x_467_ = lean_uint64_of_nat(v_snd_464_);
v___x_468_ = lean_uint64_mix_hash(v___x_466_, v___x_467_);
v___x_469_ = 32ULL;
v___x_470_ = lean_uint64_shift_right(v___x_468_, v___x_469_);
v_fold_471_ = lean_uint64_xor(v___x_468_, v___x_470_);
v___x_472_ = 16ULL;
v___x_473_ = lean_uint64_shift_right(v_fold_471_, v___x_472_);
v___x_474_ = lean_uint64_xor(v_fold_471_, v___x_473_);
v___x_475_ = lean_uint64_to_usize(v___x_474_);
v___x_476_ = lean_usize_of_nat(v___x_465_);
v___x_477_ = ((size_t)1ULL);
v___x_478_ = lean_usize_sub(v___x_476_, v___x_477_);
v___x_479_ = lean_usize_land(v___x_475_, v___x_478_);
v___x_480_ = lean_array_uget_borrowed(v_buckets_462_, v___x_479_);
v___x_481_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg(v_a_461_, v___x_480_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg___boxed(lean_object* v_m_482_, lean_object* v_a_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg(v_m_482_, v_a_483_);
lean_dec_ref(v_a_483_);
lean_dec_ref(v_m_482_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12_spec__16(size_t v_sz_485_, size_t v_i_486_, lean_object* v_bs_487_){
_start:
{
uint8_t v___x_488_; 
v___x_488_ = lean_usize_dec_lt(v_i_486_, v_sz_485_);
if (v___x_488_ == 0)
{
lean_object* v___x_489_; 
v___x_489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_489_, 0, v_bs_487_);
return v___x_489_;
}
else
{
lean_object* v_v_490_; lean_object* v___x_491_; 
v_v_490_ = lean_array_uget_borrowed(v_bs_487_, v_i_486_);
lean_inc(v_v_490_);
v___x_491_ = l_Lean_Json_getStr_x3f(v_v_490_);
if (lean_obj_tag(v___x_491_) == 0)
{
lean_object* v_a_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
lean_dec_ref(v_bs_487_);
v_a_492_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_499_ == 0)
{
v___x_494_ = v___x_491_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_491_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_497_; 
if (v_isShared_495_ == 0)
{
v___x_497_ = v___x_494_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v_a_492_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
else
{
lean_object* v_a_500_; lean_object* v___x_501_; lean_object* v_bs_x27_502_; size_t v___x_503_; size_t v___x_504_; lean_object* v___x_505_; 
v_a_500_ = lean_ctor_get(v___x_491_, 0);
lean_inc(v_a_500_);
lean_dec_ref_known(v___x_491_, 1);
v___x_501_ = lean_unsigned_to_nat(0u);
v_bs_x27_502_ = lean_array_uset(v_bs_487_, v_i_486_, v___x_501_);
v___x_503_ = ((size_t)1ULL);
v___x_504_ = lean_usize_add(v_i_486_, v___x_503_);
v___x_505_ = lean_array_uset(v_bs_x27_502_, v_i_486_, v_a_500_);
v_i_486_ = v___x_504_;
v_bs_487_ = v___x_505_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12_spec__16___boxed(lean_object* v_sz_507_, lean_object* v_i_508_, lean_object* v_bs_509_){
_start:
{
size_t v_sz_boxed_510_; size_t v_i_boxed_511_; lean_object* v_res_512_; 
v_sz_boxed_510_ = lean_unbox_usize(v_sz_507_);
lean_dec(v_sz_507_);
v_i_boxed_511_ = lean_unbox_usize(v_i_508_);
lean_dec(v_i_508_);
v_res_512_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12_spec__16(v_sz_boxed_510_, v_i_boxed_511_, v_bs_509_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12(lean_object* v_x_515_){
_start:
{
if (lean_obj_tag(v_x_515_) == 4)
{
lean_object* v_elems_516_; size_t v_sz_517_; size_t v___x_518_; lean_object* v___x_519_; 
v_elems_516_ = lean_ctor_get(v_x_515_, 0);
lean_inc_ref(v_elems_516_);
lean_dec_ref_known(v_x_515_, 1);
v_sz_517_ = lean_array_size(v_elems_516_);
v___x_518_ = ((size_t)0ULL);
v___x_519_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12_spec__16(v_sz_517_, v___x_518_, v_elems_516_);
return v___x_519_;
}
else
{
lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; 
v___x_520_ = ((lean_object*)(lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__0));
v___x_521_ = lean_unsigned_to_nat(80u);
v___x_522_ = l_Lean_Json_pretty(v_x_515_, v___x_521_);
v___x_523_ = lean_string_append(v___x_520_, v___x_522_);
lean_dec_ref(v___x_522_);
v___x_524_ = ((lean_object*)(lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12___closed__1));
v___x_525_ = lean_string_append(v___x_523_, v___x_524_);
v___x_526_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_526_, 0, v___x_525_);
return v___x_526_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9(lean_object* v_j_527_){
_start:
{
lean_object* v___x_528_; 
v___x_528_ = lp_LeanSearchClient_Lean_Array_fromJson_x3f___at___00Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9_spec__12(v_j_527_);
if (lean_obj_tag(v___x_528_) == 0)
{
lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
v_a_529_ = lean_ctor_get(v___x_528_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_528_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_528_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_528_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
else
{
lean_object* v_a_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_545_; 
v_a_537_ = lean_ctor_get(v___x_528_, 0);
v_isSharedCheck_545_ = !lean_is_exclusive(v___x_528_);
if (v_isSharedCheck_545_ == 0)
{
v___x_539_ = v___x_528_;
v_isShared_540_ = v_isSharedCheck_545_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_a_537_);
lean_dec(v___x_528_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_545_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_541_; lean_object* v___x_543_; 
v___x_541_ = lean_array_to_list(v_a_537_);
if (v_isShared_540_ == 0)
{
lean_ctor_set(v___x_539_, 0, v___x_541_);
v___x_543_ = v___x_539_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v___x_541_);
v___x_543_ = v_reuseFailAlloc_544_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
return v___x_543_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4(lean_object* v_j_546_, lean_object* v_k_547_){
_start:
{
lean_object* v___x_548_; lean_object* v___x_549_; 
v___x_548_ = l_Lean_Json_getObjValD(v_j_546_, v_k_547_);
v___x_549_ = lp_LeanSearchClient_Lean_List_fromJson_x3f___at___00Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4_spec__9(v___x_548_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4___boxed(lean_object* v_j_550_, lean_object* v_k_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4(v_j_550_, v_k_551_);
lean_dec_ref(v_k_551_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7___redArg(lean_object* v_a_553_, lean_object* v_b_554_, lean_object* v_x_555_){
_start:
{
if (lean_obj_tag(v_x_555_) == 0)
{
lean_dec(v_b_554_);
lean_dec_ref(v_a_553_);
return v_x_555_;
}
else
{
lean_object* v_key_556_; lean_object* v_value_557_; lean_object* v_tail_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_577_; 
v_key_556_ = lean_ctor_get(v_x_555_, 0);
v_value_557_ = lean_ctor_get(v_x_555_, 1);
v_tail_558_ = lean_ctor_get(v_x_555_, 2);
v_isSharedCheck_577_ = !lean_is_exclusive(v_x_555_);
if (v_isSharedCheck_577_ == 0)
{
v___x_560_ = v_x_555_;
v_isShared_561_ = v_isSharedCheck_577_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_tail_558_);
lean_inc(v_value_557_);
lean_inc(v_key_556_);
lean_dec(v_x_555_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_577_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
uint8_t v___y_563_; lean_object* v_fst_571_; lean_object* v_snd_572_; lean_object* v_fst_573_; lean_object* v_snd_574_; uint8_t v___x_575_; 
v_fst_571_ = lean_ctor_get(v_key_556_, 0);
v_snd_572_ = lean_ctor_get(v_key_556_, 1);
v_fst_573_ = lean_ctor_get(v_a_553_, 0);
v_snd_574_ = lean_ctor_get(v_a_553_, 1);
v___x_575_ = lean_string_dec_eq(v_fst_571_, v_fst_573_);
if (v___x_575_ == 0)
{
v___y_563_ = v___x_575_;
goto v___jp_562_;
}
else
{
uint8_t v___x_576_; 
v___x_576_ = lean_nat_dec_eq(v_snd_572_, v_snd_574_);
v___y_563_ = v___x_576_;
goto v___jp_562_;
}
v___jp_562_:
{
if (v___y_563_ == 0)
{
lean_object* v___x_564_; lean_object* v___x_566_; 
v___x_564_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7___redArg(v_a_553_, v_b_554_, v_tail_558_);
if (v_isShared_561_ == 0)
{
lean_ctor_set(v___x_560_, 2, v___x_564_);
v___x_566_ = v___x_560_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v_key_556_);
lean_ctor_set(v_reuseFailAlloc_567_, 1, v_value_557_);
lean_ctor_set(v_reuseFailAlloc_567_, 2, v___x_564_);
v___x_566_ = v_reuseFailAlloc_567_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
return v___x_566_;
}
}
else
{
lean_object* v___x_569_; 
lean_dec(v_value_557_);
lean_dec(v_key_556_);
if (v_isShared_561_ == 0)
{
lean_ctor_set(v___x_560_, 1, v_b_554_);
lean_ctor_set(v___x_560_, 0, v_a_553_);
v___x_569_ = v___x_560_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_a_553_);
lean_ctor_set(v_reuseFailAlloc_570_, 1, v_b_554_);
lean_ctor_set(v_reuseFailAlloc_570_, 2, v_tail_558_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8_spec__12___redArg(lean_object* v_x_578_, lean_object* v_x_579_){
_start:
{
if (lean_obj_tag(v_x_579_) == 0)
{
return v_x_578_;
}
else
{
lean_object* v_key_580_; lean_object* v_value_581_; lean_object* v_tail_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_609_; 
v_key_580_ = lean_ctor_get(v_x_579_, 0);
v_value_581_ = lean_ctor_get(v_x_579_, 1);
v_tail_582_ = lean_ctor_get(v_x_579_, 2);
v_isSharedCheck_609_ = !lean_is_exclusive(v_x_579_);
if (v_isSharedCheck_609_ == 0)
{
v___x_584_ = v_x_579_;
v_isShared_585_ = v_isSharedCheck_609_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_tail_582_);
lean_inc(v_value_581_);
lean_inc(v_key_580_);
lean_dec(v_x_579_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_609_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v_fst_586_; lean_object* v_snd_587_; lean_object* v___x_588_; uint64_t v___x_589_; uint64_t v___x_590_; uint64_t v___x_591_; uint64_t v___x_592_; uint64_t v___x_593_; uint64_t v_fold_594_; uint64_t v___x_595_; uint64_t v___x_596_; uint64_t v___x_597_; size_t v___x_598_; size_t v___x_599_; size_t v___x_600_; size_t v___x_601_; size_t v___x_602_; lean_object* v___x_603_; lean_object* v___x_605_; 
v_fst_586_ = lean_ctor_get(v_key_580_, 0);
v_snd_587_ = lean_ctor_get(v_key_580_, 1);
v___x_588_ = lean_array_get_size(v_x_578_);
v___x_589_ = lean_string_hash(v_fst_586_);
v___x_590_ = lean_uint64_of_nat(v_snd_587_);
v___x_591_ = lean_uint64_mix_hash(v___x_589_, v___x_590_);
v___x_592_ = 32ULL;
v___x_593_ = lean_uint64_shift_right(v___x_591_, v___x_592_);
v_fold_594_ = lean_uint64_xor(v___x_591_, v___x_593_);
v___x_595_ = 16ULL;
v___x_596_ = lean_uint64_shift_right(v_fold_594_, v___x_595_);
v___x_597_ = lean_uint64_xor(v_fold_594_, v___x_596_);
v___x_598_ = lean_uint64_to_usize(v___x_597_);
v___x_599_ = lean_usize_of_nat(v___x_588_);
v___x_600_ = ((size_t)1ULL);
v___x_601_ = lean_usize_sub(v___x_599_, v___x_600_);
v___x_602_ = lean_usize_land(v___x_598_, v___x_601_);
v___x_603_ = lean_array_uget_borrowed(v_x_578_, v___x_602_);
lean_inc(v___x_603_);
if (v_isShared_585_ == 0)
{
lean_ctor_set(v___x_584_, 2, v___x_603_);
v___x_605_ = v___x_584_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v_key_580_);
lean_ctor_set(v_reuseFailAlloc_608_, 1, v_value_581_);
lean_ctor_set(v_reuseFailAlloc_608_, 2, v___x_603_);
v___x_605_ = v_reuseFailAlloc_608_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
lean_object* v___x_606_; 
v___x_606_ = lean_array_uset(v_x_578_, v___x_602_, v___x_605_);
v_x_578_ = v___x_606_;
v_x_579_ = v_tail_582_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8___redArg(lean_object* v_i_610_, lean_object* v_source_611_, lean_object* v_target_612_){
_start:
{
lean_object* v___x_613_; uint8_t v___x_614_; 
v___x_613_ = lean_array_get_size(v_source_611_);
v___x_614_ = lean_nat_dec_lt(v_i_610_, v___x_613_);
if (v___x_614_ == 0)
{
lean_dec_ref(v_source_611_);
lean_dec(v_i_610_);
return v_target_612_;
}
else
{
lean_object* v_es_615_; lean_object* v___x_616_; lean_object* v_source_617_; lean_object* v_target_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v_es_615_ = lean_array_fget(v_source_611_, v_i_610_);
v___x_616_ = lean_box(0);
v_source_617_ = lean_array_fset(v_source_611_, v_i_610_, v___x_616_);
v_target_618_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8_spec__12___redArg(v_target_612_, v_es_615_);
v___x_619_ = lean_unsigned_to_nat(1u);
v___x_620_ = lean_nat_add(v_i_610_, v___x_619_);
lean_dec(v_i_610_);
v_i_610_ = v___x_620_;
v_source_611_ = v_source_617_;
v_target_612_ = v_target_618_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6___redArg(lean_object* v_data_622_){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v_nbuckets_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_623_ = lean_array_get_size(v_data_622_);
v___x_624_ = lean_unsigned_to_nat(2u);
v_nbuckets_625_ = lean_nat_mul(v___x_623_, v___x_624_);
v___x_626_ = lean_unsigned_to_nat(0u);
v___x_627_ = lean_box(0);
v___x_628_ = lean_mk_array(v_nbuckets_625_, v___x_627_);
v___x_629_ = lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8___redArg(v___x_626_, v_data_622_, v___x_628_);
return v___x_629_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg(lean_object* v_a_630_, lean_object* v_x_631_){
_start:
{
if (lean_obj_tag(v_x_631_) == 0)
{
uint8_t v___x_632_; 
v___x_632_ = 0;
return v___x_632_;
}
else
{
lean_object* v_key_633_; lean_object* v_tail_634_; uint8_t v___y_636_; lean_object* v_fst_638_; lean_object* v_snd_639_; lean_object* v_fst_640_; lean_object* v_snd_641_; uint8_t v___x_642_; 
v_key_633_ = lean_ctor_get(v_x_631_, 0);
v_tail_634_ = lean_ctor_get(v_x_631_, 2);
v_fst_638_ = lean_ctor_get(v_key_633_, 0);
v_snd_639_ = lean_ctor_get(v_key_633_, 1);
v_fst_640_ = lean_ctor_get(v_a_630_, 0);
v_snd_641_ = lean_ctor_get(v_a_630_, 1);
v___x_642_ = lean_string_dec_eq(v_fst_638_, v_fst_640_);
if (v___x_642_ == 0)
{
v___y_636_ = v___x_642_;
goto v___jp_635_;
}
else
{
uint8_t v___x_643_; 
v___x_643_ = lean_nat_dec_eq(v_snd_639_, v_snd_641_);
v___y_636_ = v___x_643_;
goto v___jp_635_;
}
v___jp_635_:
{
if (v___y_636_ == 0)
{
v_x_631_ = v_tail_634_;
goto _start;
}
else
{
return v___y_636_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg___boxed(lean_object* v_a_644_, lean_object* v_x_645_){
_start:
{
uint8_t v_res_646_; lean_object* v_r_647_; 
v_res_646_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg(v_a_644_, v_x_645_);
lean_dec(v_x_645_);
lean_dec_ref(v_a_644_);
v_r_647_ = lean_box(v_res_646_);
return v_r_647_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3___redArg(lean_object* v_m_648_, lean_object* v_a_649_, lean_object* v_b_650_){
_start:
{
lean_object* v_size_651_; lean_object* v_buckets_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_699_; 
v_size_651_ = lean_ctor_get(v_m_648_, 0);
v_buckets_652_ = lean_ctor_get(v_m_648_, 1);
v_isSharedCheck_699_ = !lean_is_exclusive(v_m_648_);
if (v_isSharedCheck_699_ == 0)
{
v___x_654_ = v_m_648_;
v_isShared_655_ = v_isSharedCheck_699_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_buckets_652_);
lean_inc(v_size_651_);
lean_dec(v_m_648_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_699_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v_fst_656_; lean_object* v_snd_657_; lean_object* v___x_658_; uint64_t v___x_659_; uint64_t v___x_660_; uint64_t v___x_661_; uint64_t v___x_662_; uint64_t v___x_663_; uint64_t v_fold_664_; uint64_t v___x_665_; uint64_t v___x_666_; uint64_t v___x_667_; size_t v___x_668_; size_t v___x_669_; size_t v___x_670_; size_t v___x_671_; size_t v___x_672_; lean_object* v_bkt_673_; uint8_t v___x_674_; 
v_fst_656_ = lean_ctor_get(v_a_649_, 0);
v_snd_657_ = lean_ctor_get(v_a_649_, 1);
v___x_658_ = lean_array_get_size(v_buckets_652_);
v___x_659_ = lean_string_hash(v_fst_656_);
v___x_660_ = lean_uint64_of_nat(v_snd_657_);
v___x_661_ = lean_uint64_mix_hash(v___x_659_, v___x_660_);
v___x_662_ = 32ULL;
v___x_663_ = lean_uint64_shift_right(v___x_661_, v___x_662_);
v_fold_664_ = lean_uint64_xor(v___x_661_, v___x_663_);
v___x_665_ = 16ULL;
v___x_666_ = lean_uint64_shift_right(v_fold_664_, v___x_665_);
v___x_667_ = lean_uint64_xor(v_fold_664_, v___x_666_);
v___x_668_ = lean_uint64_to_usize(v___x_667_);
v___x_669_ = lean_usize_of_nat(v___x_658_);
v___x_670_ = ((size_t)1ULL);
v___x_671_ = lean_usize_sub(v___x_669_, v___x_670_);
v___x_672_ = lean_usize_land(v___x_668_, v___x_671_);
v_bkt_673_ = lean_array_uget_borrowed(v_buckets_652_, v___x_672_);
v___x_674_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg(v_a_649_, v_bkt_673_);
if (v___x_674_ == 0)
{
lean_object* v___x_675_; lean_object* v_size_x27_676_; lean_object* v___x_677_; lean_object* v_buckets_x27_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; uint8_t v___x_684_; 
v___x_675_ = lean_unsigned_to_nat(1u);
v_size_x27_676_ = lean_nat_add(v_size_651_, v___x_675_);
lean_dec(v_size_651_);
lean_inc(v_bkt_673_);
v___x_677_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_677_, 0, v_a_649_);
lean_ctor_set(v___x_677_, 1, v_b_650_);
lean_ctor_set(v___x_677_, 2, v_bkt_673_);
v_buckets_x27_678_ = lean_array_uset(v_buckets_652_, v___x_672_, v___x_677_);
v___x_679_ = lean_unsigned_to_nat(4u);
v___x_680_ = lean_nat_mul(v_size_x27_676_, v___x_679_);
v___x_681_ = lean_unsigned_to_nat(3u);
v___x_682_ = lean_nat_div(v___x_680_, v___x_681_);
lean_dec(v___x_680_);
v___x_683_ = lean_array_get_size(v_buckets_x27_678_);
v___x_684_ = lean_nat_dec_le(v___x_682_, v___x_683_);
lean_dec(v___x_682_);
if (v___x_684_ == 0)
{
lean_object* v_val_685_; lean_object* v___x_687_; 
v_val_685_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6___redArg(v_buckets_x27_678_);
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 1, v_val_685_);
lean_ctor_set(v___x_654_, 0, v_size_x27_676_);
v___x_687_ = v___x_654_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v_size_x27_676_);
lean_ctor_set(v_reuseFailAlloc_688_, 1, v_val_685_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
else
{
lean_object* v___x_690_; 
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 1, v_buckets_x27_678_);
lean_ctor_set(v___x_654_, 0, v_size_x27_676_);
v___x_690_ = v___x_654_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_size_x27_676_);
lean_ctor_set(v_reuseFailAlloc_691_, 1, v_buckets_x27_678_);
v___x_690_ = v_reuseFailAlloc_691_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
return v___x_690_;
}
}
}
else
{
lean_object* v___x_692_; lean_object* v_buckets_x27_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_697_; 
lean_inc(v_bkt_673_);
v___x_692_ = lean_box(0);
v_buckets_x27_693_ = lean_array_uset(v_buckets_652_, v___x_672_, v___x_692_);
v___x_694_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7___redArg(v_a_649_, v_b_650_, v_bkt_673_);
v___x_695_ = lean_array_uset(v_buckets_x27_693_, v___x_672_, v___x_694_);
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 1, v___x_695_);
v___x_697_ = v___x_654_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v_size_651_);
lean_ctor_set(v_reuseFailAlloc_698_, 1, v___x_695_);
v___x_697_ = v_reuseFailAlloc_698_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
return v___x_697_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(lean_object* v_s_700_, lean_object* v_replacement_701_, lean_object* v_a_702_, lean_object* v_b_703_){
_start:
{
lean_object* v_it_705_; lean_object* v_startPos_706_; lean_object* v_endPos_707_; lean_object* v_it_716_; 
switch(lean_obj_tag(v_a_702_))
{
case 0:
{
lean_object* v_pos_722_; lean_object* v___x_724_; uint8_t v_isShared_725_; uint8_t v_isSharedCheck_734_; 
v_pos_722_ = lean_ctor_get(v_a_702_, 0);
v_isSharedCheck_734_ = !lean_is_exclusive(v_a_702_);
if (v_isSharedCheck_734_ == 0)
{
v___x_724_ = v_a_702_;
v_isShared_725_ = v_isSharedCheck_734_;
goto v_resetjp_723_;
}
else
{
lean_inc(v_pos_722_);
lean_dec(v_a_702_);
v___x_724_ = lean_box(0);
v_isShared_725_ = v_isSharedCheck_734_;
goto v_resetjp_723_;
}
v_resetjp_723_:
{
lean_object* v_startInclusive_726_; lean_object* v_endExclusive_727_; lean_object* v___x_728_; uint8_t v___x_729_; 
v_startInclusive_726_ = lean_ctor_get(v_s_700_, 1);
v_endExclusive_727_ = lean_ctor_get(v_s_700_, 2);
v___x_728_ = lean_nat_sub(v_endExclusive_727_, v_startInclusive_726_);
v___x_729_ = lean_nat_dec_eq(v_pos_722_, v___x_728_);
lean_dec(v___x_728_);
if (v___x_729_ == 0)
{
lean_object* v___x_731_; 
if (v_isShared_725_ == 0)
{
lean_ctor_set_tag(v___x_724_, 1);
v___x_731_ = v___x_724_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v_pos_722_);
v___x_731_ = v_reuseFailAlloc_732_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
v_it_716_ = v___x_731_;
goto v___jp_715_;
}
}
else
{
lean_object* v___x_733_; 
lean_del_object(v___x_724_);
lean_dec(v_pos_722_);
v___x_733_ = lean_box(3);
v_it_716_ = v___x_733_;
goto v___jp_715_;
}
}
}
case 1:
{
lean_object* v_pos_735_; lean_object* v___x_737_; uint8_t v_isShared_738_; uint8_t v_isSharedCheck_747_; 
v_pos_735_ = lean_ctor_get(v_a_702_, 0);
v_isSharedCheck_747_ = !lean_is_exclusive(v_a_702_);
if (v_isSharedCheck_747_ == 0)
{
v___x_737_ = v_a_702_;
v_isShared_738_ = v_isSharedCheck_747_;
goto v_resetjp_736_;
}
else
{
lean_inc(v_pos_735_);
lean_dec(v_a_702_);
v___x_737_ = lean_box(0);
v_isShared_738_ = v_isSharedCheck_747_;
goto v_resetjp_736_;
}
v_resetjp_736_:
{
lean_object* v_str_739_; lean_object* v_startInclusive_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_745_; 
v_str_739_ = lean_ctor_get(v_s_700_, 0);
v_startInclusive_740_ = lean_ctor_get(v_s_700_, 1);
v___x_741_ = lean_nat_add(v_startInclusive_740_, v_pos_735_);
v___x_742_ = lean_string_utf8_next_fast(v_str_739_, v___x_741_);
lean_dec(v___x_741_);
v___x_743_ = lean_nat_sub(v___x_742_, v_startInclusive_740_);
lean_inc(v___x_743_);
if (v_isShared_738_ == 0)
{
lean_ctor_set_tag(v___x_737_, 0);
lean_ctor_set(v___x_737_, 0, v___x_743_);
v___x_745_ = v___x_737_;
goto v_reusejp_744_;
}
else
{
lean_object* v_reuseFailAlloc_746_; 
v_reuseFailAlloc_746_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_746_, 0, v___x_743_);
v___x_745_ = v_reuseFailAlloc_746_;
goto v_reusejp_744_;
}
v_reusejp_744_:
{
v_it_705_ = v___x_745_;
v_startPos_706_ = v_pos_735_;
v_endPos_707_ = v___x_743_;
goto v___jp_704_;
}
}
}
case 2:
{
lean_object* v_needle_748_; lean_object* v_table_749_; lean_object* v_stackPos_750_; lean_object* v_needlePos_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_810_; 
v_needle_748_ = lean_ctor_get(v_a_702_, 0);
v_table_749_ = lean_ctor_get(v_a_702_, 1);
v_stackPos_750_ = lean_ctor_get(v_a_702_, 2);
v_needlePos_751_ = lean_ctor_get(v_a_702_, 3);
v_isSharedCheck_810_ = !lean_is_exclusive(v_a_702_);
if (v_isSharedCheck_810_ == 0)
{
v___x_753_ = v_a_702_;
v_isShared_754_ = v_isSharedCheck_810_;
goto v_resetjp_752_;
}
else
{
lean_inc(v_needlePos_751_);
lean_inc(v_stackPos_750_);
lean_inc(v_table_749_);
lean_inc(v_needle_748_);
lean_dec(v_a_702_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_810_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
lean_object* v_str_755_; lean_object* v_startInclusive_756_; lean_object* v_endExclusive_757_; lean_object* v_str_758_; lean_object* v_startInclusive_759_; lean_object* v_endExclusive_760_; lean_object* v_basePos_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; uint8_t v___x_765_; 
v_str_755_ = lean_ctor_get(v_needle_748_, 0);
v_startInclusive_756_ = lean_ctor_get(v_needle_748_, 1);
v_endExclusive_757_ = lean_ctor_get(v_needle_748_, 2);
v_str_758_ = lean_ctor_get(v_s_700_, 0);
v_startInclusive_759_ = lean_ctor_get(v_s_700_, 1);
v_endExclusive_760_ = lean_ctor_get(v_s_700_, 2);
v_basePos_761_ = lean_nat_sub(v_stackPos_750_, v_needlePos_751_);
v___x_762_ = lean_nat_sub(v_endExclusive_757_, v_startInclusive_756_);
v___x_763_ = lean_nat_add(v_basePos_761_, v___x_762_);
v___x_764_ = lean_nat_sub(v_endExclusive_760_, v_startInclusive_759_);
v___x_765_ = lean_nat_dec_le(v___x_763_, v___x_764_);
lean_dec(v___x_763_);
if (v___x_765_ == 0)
{
uint8_t v___x_766_; 
lean_dec(v___x_762_);
lean_del_object(v___x_753_);
lean_dec(v_needlePos_751_);
lean_dec(v_stackPos_750_);
lean_dec_ref(v_table_749_);
lean_dec_ref(v_needle_748_);
v___x_766_ = lean_nat_dec_lt(v_basePos_761_, v___x_764_);
if (v___x_766_ == 0)
{
lean_dec(v___x_764_);
lean_dec(v_basePos_761_);
lean_dec_ref(v_s_700_);
return v_b_703_;
}
else
{
lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_767_ = l_String_Slice_pos_x21(v_s_700_, v_basePos_761_);
lean_dec(v_basePos_761_);
v___x_768_ = lean_box(3);
v_it_705_ = v___x_768_;
v_startPos_706_ = v___x_767_;
v_endPos_707_ = v___x_764_;
goto v___jp_704_;
}
}
else
{
lean_object* v___x_769_; uint8_t v_stackByte_770_; lean_object* v___x_771_; uint8_t v_patByte_772_; uint8_t v___x_773_; 
lean_dec(v___x_764_);
v___x_769_ = lean_nat_add(v_startInclusive_759_, v_stackPos_750_);
v_stackByte_770_ = lean_string_get_byte_fast(v_str_758_, v___x_769_);
v___x_771_ = lean_nat_add(v_startInclusive_756_, v_needlePos_751_);
v_patByte_772_ = lean_string_get_byte_fast(v_str_755_, v___x_771_);
v___x_773_ = lean_uint8_dec_eq(v_stackByte_770_, v_patByte_772_);
if (v___x_773_ == 0)
{
lean_object* v___x_774_; uint8_t v___x_775_; 
lean_dec(v___x_762_);
v___x_774_ = lean_unsigned_to_nat(0u);
v___x_775_ = lean_nat_dec_eq(v_needlePos_751_, v___x_774_);
if (v___x_775_ == 0)
{
lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v_newNeedlePos_778_; uint8_t v___x_779_; 
v___x_776_ = lean_unsigned_to_nat(1u);
v___x_777_ = lean_nat_sub(v_needlePos_751_, v___x_776_);
lean_dec(v_needlePos_751_);
v_newNeedlePos_778_ = lean_array_fget_borrowed(v_table_749_, v___x_777_);
lean_dec(v___x_777_);
v___x_779_ = lean_nat_dec_eq(v_newNeedlePos_778_, v___x_774_);
if (v___x_779_ == 0)
{
lean_object* v_oldBasePos_780_; lean_object* v___x_781_; lean_object* v_newBasePos_782_; lean_object* v___x_784_; 
lean_inc(v_newNeedlePos_778_);
v_oldBasePos_780_ = l_String_Slice_pos_x21(v_s_700_, v_basePos_761_);
lean_dec(v_basePos_761_);
v___x_781_ = lean_nat_sub(v_stackPos_750_, v_newNeedlePos_778_);
v_newBasePos_782_ = l_String_Slice_pos_x21(v_s_700_, v___x_781_);
lean_dec(v___x_781_);
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 3, v_newNeedlePos_778_);
v___x_784_ = v___x_753_;
goto v_reusejp_783_;
}
else
{
lean_object* v_reuseFailAlloc_785_; 
v_reuseFailAlloc_785_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_785_, 0, v_needle_748_);
lean_ctor_set(v_reuseFailAlloc_785_, 1, v_table_749_);
lean_ctor_set(v_reuseFailAlloc_785_, 2, v_stackPos_750_);
lean_ctor_set(v_reuseFailAlloc_785_, 3, v_newNeedlePos_778_);
v___x_784_ = v_reuseFailAlloc_785_;
goto v_reusejp_783_;
}
v_reusejp_783_:
{
v_it_705_ = v___x_784_;
v_startPos_706_ = v_oldBasePos_780_;
v_endPos_707_ = v_newBasePos_782_;
goto v___jp_704_;
}
}
else
{
lean_object* v_basePos_786_; lean_object* v_nextStackPos_787_; lean_object* v___x_789_; 
v_basePos_786_ = l_String_Slice_pos_x21(v_s_700_, v_basePos_761_);
lean_dec(v_basePos_761_);
v_nextStackPos_787_ = l_String_Slice_posGE___redArg(v_s_700_, v_stackPos_750_);
lean_inc(v_nextStackPos_787_);
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 3, v___x_774_);
lean_ctor_set(v___x_753_, 2, v_nextStackPos_787_);
v___x_789_ = v___x_753_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v_needle_748_);
lean_ctor_set(v_reuseFailAlloc_790_, 1, v_table_749_);
lean_ctor_set(v_reuseFailAlloc_790_, 2, v_nextStackPos_787_);
lean_ctor_set(v_reuseFailAlloc_790_, 3, v___x_774_);
v___x_789_ = v_reuseFailAlloc_790_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
v_it_705_ = v___x_789_;
v_startPos_706_ = v_basePos_786_;
v_endPos_707_ = v_nextStackPos_787_;
goto v___jp_704_;
}
}
}
else
{
lean_object* v_basePos_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v_nextStackPos_794_; lean_object* v___x_796_; 
lean_dec(v_basePos_761_);
lean_dec(v_needlePos_751_);
v_basePos_791_ = l_String_Slice_pos_x21(v_s_700_, v_stackPos_750_);
v___x_792_ = lean_unsigned_to_nat(1u);
v___x_793_ = lean_nat_add(v_stackPos_750_, v___x_792_);
lean_dec(v_stackPos_750_);
v_nextStackPos_794_ = l_String_Slice_posGE___redArg(v_s_700_, v___x_793_);
lean_inc(v_nextStackPos_794_);
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 3, v___x_774_);
lean_ctor_set(v___x_753_, 2, v_nextStackPos_794_);
v___x_796_ = v___x_753_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_797_; 
v_reuseFailAlloc_797_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_797_, 0, v_needle_748_);
lean_ctor_set(v_reuseFailAlloc_797_, 1, v_table_749_);
lean_ctor_set(v_reuseFailAlloc_797_, 2, v_nextStackPos_794_);
lean_ctor_set(v_reuseFailAlloc_797_, 3, v___x_774_);
v___x_796_ = v_reuseFailAlloc_797_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
v_it_705_ = v___x_796_;
v_startPos_706_ = v_basePos_791_;
v_endPos_707_ = v_nextStackPos_794_;
goto v___jp_704_;
}
}
}
else
{
lean_object* v___x_798_; lean_object* v_nextStackPos_799_; lean_object* v_nextNeedlePos_800_; uint8_t v___x_801_; 
lean_dec(v_basePos_761_);
v___x_798_ = lean_unsigned_to_nat(1u);
v_nextStackPos_799_ = lean_nat_add(v_stackPos_750_, v___x_798_);
lean_dec(v_stackPos_750_);
v_nextNeedlePos_800_ = lean_nat_add(v_needlePos_751_, v___x_798_);
lean_dec(v_needlePos_751_);
v___x_801_ = lean_nat_dec_eq(v_nextNeedlePos_800_, v___x_762_);
lean_dec(v___x_762_);
if (v___x_801_ == 0)
{
lean_object* v___x_803_; 
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 3, v_nextNeedlePos_800_);
lean_ctor_set(v___x_753_, 2, v_nextStackPos_799_);
v___x_803_ = v___x_753_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v_needle_748_);
lean_ctor_set(v_reuseFailAlloc_805_, 1, v_table_749_);
lean_ctor_set(v_reuseFailAlloc_805_, 2, v_nextStackPos_799_);
lean_ctor_set(v_reuseFailAlloc_805_, 3, v_nextNeedlePos_800_);
v___x_803_ = v_reuseFailAlloc_805_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
v_a_702_ = v___x_803_;
goto _start;
}
}
else
{
lean_object* v___x_806_; lean_object* v___x_808_; 
lean_dec(v_nextNeedlePos_800_);
v___x_806_ = lean_unsigned_to_nat(0u);
if (v_isShared_754_ == 0)
{
lean_ctor_set(v___x_753_, 3, v___x_806_);
lean_ctor_set(v___x_753_, 2, v_nextStackPos_799_);
v___x_808_ = v___x_753_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_needle_748_);
lean_ctor_set(v_reuseFailAlloc_809_, 1, v_table_749_);
lean_ctor_set(v_reuseFailAlloc_809_, 2, v_nextStackPos_799_);
lean_ctor_set(v_reuseFailAlloc_809_, 3, v___x_806_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
v_it_716_ = v___x_808_;
goto v___jp_715_;
}
}
}
}
}
}
default: 
{
lean_dec_ref(v_s_700_);
return v_b_703_;
}
}
v___jp_704_:
{
lean_object* v___x_708_; lean_object* v_str_709_; lean_object* v_startInclusive_710_; lean_object* v_endExclusive_711_; lean_object* v___x_712_; lean_object* v___x_713_; 
lean_inc_ref(v_s_700_);
v___x_708_ = l_String_Slice_slice_x21(v_s_700_, v_startPos_706_, v_endPos_707_);
lean_dec(v_endPos_707_);
lean_dec(v_startPos_706_);
v_str_709_ = lean_ctor_get(v___x_708_, 0);
lean_inc_ref(v_str_709_);
v_startInclusive_710_ = lean_ctor_get(v___x_708_, 1);
lean_inc(v_startInclusive_710_);
v_endExclusive_711_ = lean_ctor_get(v___x_708_, 2);
lean_inc(v_endExclusive_711_);
lean_dec_ref(v___x_708_);
v___x_712_ = lean_string_utf8_extract_fast(v_str_709_, v_startInclusive_710_, v_endExclusive_711_);
lean_dec(v_endExclusive_711_);
lean_dec(v_startInclusive_710_);
lean_dec_ref(v_str_709_);
v___x_713_ = lean_string_append(v_b_703_, v___x_712_);
lean_dec_ref(v___x_712_);
v_a_702_ = v_it_705_;
v_b_703_ = v___x_713_;
goto _start;
}
v___jp_715_:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_717_ = lean_unsigned_to_nat(0u);
v___x_718_ = lean_string_utf8_byte_size(v_replacement_701_);
v___x_719_ = lean_string_utf8_extract_fast(v_replacement_701_, v___x_717_, v___x_718_);
v___x_720_ = lean_string_append(v_b_703_, v___x_719_);
lean_dec_ref(v___x_719_);
v_a_702_ = v_it_716_;
v_b_703_ = v___x_720_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg___boxed(lean_object* v_s_811_, lean_object* v_replacement_812_, lean_object* v_a_813_, lean_object* v_b_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(v_s_811_, v_replacement_812_, v_a_813_, v_b_814_);
lean_dec_ref(v_replacement_812_);
return v_res_815_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_817_; lean_object* v___x_818_; 
v___x_817_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__0));
v___x_818_ = lean_string_utf8_byte_size(v___x_817_);
return v___x_818_;
}
}
static uint8_t _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_819_; lean_object* v___x_820_; uint8_t v___x_821_; 
v___x_819_ = lean_unsigned_to_nat(0u);
v___x_820_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1);
v___x_821_ = lean_nat_dec_eq(v___x_820_, v___x_819_);
return v___x_821_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; 
v___x_822_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__1);
v___x_823_ = lean_unsigned_to_nat(0u);
v___x_824_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__0));
v___x_825_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_825_, 0, v___x_824_);
lean_ctor_set(v___x_825_, 1, v___x_823_);
lean_ctor_set(v___x_825_, 2, v___x_822_);
return v___x_825_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_826_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3);
v___x_827_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_826_);
return v___x_827_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_828_ = lean_unsigned_to_nat(0u);
v___x_829_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__4, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__4_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__4);
v___x_830_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__3);
v___x_831_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_831_, 0, v___x_830_);
lean_ctor_set(v___x_831_, 1, v___x_829_);
lean_ctor_set(v___x_831_, 2, v___x_828_);
lean_ctor_set(v___x_831_, 3, v___x_828_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg(lean_object* v_s_834_, lean_object* v_replacement_835_){
_start:
{
lean_object* v___x_836_; uint8_t v___x_837_; 
v___x_836_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0));
v___x_837_ = lean_uint8_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__2, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__2_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__2);
if (v___x_837_ == 0)
{
lean_object* v___x_838_; lean_object* v___x_839_; 
v___x_838_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__5, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__5_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__5);
v___x_839_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(v_s_834_, v_replacement_835_, v___x_838_, v___x_836_);
return v___x_839_;
}
else
{
lean_object* v___x_840_; lean_object* v___x_841_; 
v___x_840_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__6));
v___x_841_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(v_s_834_, v_replacement_835_, v___x_840_, v___x_836_);
return v___x_841_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___boxed(lean_object* v_s_842_, lean_object* v_replacement_843_){
_start:
{
lean_object* v_res_844_; 
v_res_844_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg(v_s_842_, v_replacement_843_);
lean_dec_ref(v_replacement_843_);
return v_res_844_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg(size_t v_sz_847_, size_t v_i_848_, lean_object* v_bs_849_, lean_object* v___y_850_){
_start:
{
uint8_t v___x_852_; 
v___x_852_ = lean_usize_dec_lt(v_i_848_, v_sz_847_);
if (v___x_852_ == 0)
{
lean_object* v___x_853_; 
v___x_853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_853_, 0, v_bs_849_);
return v___x_853_;
}
else
{
lean_object* v___x_854_; lean_object* v_v_855_; lean_object* v___x_856_; lean_object* v_bs_x27_857_; lean_object* v_a_859_; lean_object* v___y_865_; lean_object* v___y_886_; lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_854_ = lean_box(0);
v_v_855_ = lean_array_uget(v_bs_849_, v_i_848_);
v___x_856_ = lean_unsigned_to_nat(0u);
v_bs_x27_857_ = lean_array_uset(v_bs_849_, v_i_848_, v___x_856_);
v___x_901_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__1));
lean_inc(v_v_855_);
v___x_902_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(v_v_855_, v___x_901_);
if (lean_obj_tag(v___x_902_) == 0)
{
lean_dec_ref_known(v___x_902_, 1);
v___y_886_ = v___x_854_;
goto v___jp_885_;
}
else
{
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_910_; 
v_a_903_ = lean_ctor_get(v___x_902_, 0);
v_isSharedCheck_910_ = !lean_is_exclusive(v___x_902_);
if (v_isSharedCheck_910_ == 0)
{
v___x_905_ = v___x_902_;
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_902_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v___x_908_; 
if (v_isShared_906_ == 0)
{
v___x_908_ = v___x_905_;
goto v_reusejp_907_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_a_903_);
v___x_908_ = v_reuseFailAlloc_909_;
goto v_reusejp_907_;
}
v_reusejp_907_:
{
v___y_886_ = v___x_908_;
goto v___jp_885_;
}
}
}
v___jp_858_:
{
size_t v___x_860_; size_t v___x_861_; lean_object* v___x_862_; 
v___x_860_ = ((size_t)1ULL);
v___x_861_ = lean_usize_add(v_i_848_, v___x_860_);
v___x_862_ = lean_array_uset(v_bs_x27_857_, v_i_848_, v_a_859_);
v_i_848_ = v___x_861_;
v_bs_849_ = v___x_862_;
goto _start;
}
v___jp_864_:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_866_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___closed__0));
v___x_867_ = lean_unsigned_to_nat(80u);
v___x_868_ = l_Lean_Json_pretty(v_v_855_, v___x_867_);
v___x_869_ = lean_string_append(v___x_866_, v___x_868_);
lean_dec_ref(v___x_868_);
v___x_870_ = l_Lean_IO_throwServerError___redArg(v___x_869_);
if (lean_obj_tag(v___x_870_) == 0)
{
lean_object* v_a_871_; 
v_a_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_a_871_);
lean_dec_ref_known(v___x_870_, 1);
v_a_859_ = v_a_871_;
goto v___jp_858_;
}
else
{
lean_object* v_a_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_884_; 
lean_dec_ref(v_bs_x27_857_);
v_a_872_ = lean_ctor_get(v___x_870_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_870_);
if (v_isSharedCheck_884_ == 0)
{
v___x_874_ = v___x_870_;
v_isShared_875_ = v_isSharedCheck_884_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_a_872_);
lean_dec(v___x_870_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_884_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v_ref_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_882_; 
v_ref_876_ = lean_ctor_get(v___y_865_, 5);
v___x_877_ = lean_io_error_to_string(v_a_872_);
v___x_878_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_878_, 0, v___x_877_);
v___x_879_ = l_Lean_MessageData_ofFormat(v___x_878_);
lean_inc(v_ref_876_);
v___x_880_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_880_, 0, v_ref_876_);
lean_ctor_set(v___x_880_, 1, v___x_879_);
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 0, v___x_880_);
v___x_882_ = v___x_874_;
goto v_reusejp_881_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v___x_880_);
v___x_882_ = v_reuseFailAlloc_883_;
goto v_reusejp_881_;
}
v_reusejp_881_:
{
return v___x_882_;
}
}
}
}
v___jp_885_:
{
lean_object* v___x_887_; lean_object* v___x_888_; 
v___x_887_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__1));
lean_inc(v_v_855_);
v___x_888_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(v_v_855_, v___x_887_);
if (lean_obj_tag(v___x_888_) == 1)
{
lean_object* v_a_889_; lean_object* v___x_890_; lean_object* v___x_891_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc(v_a_889_);
lean_dec_ref_known(v___x_888_, 1);
v___x_890_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instReprLoogleMatch_repr___redArg___closed__10));
lean_inc(v_v_855_);
v___x_891_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(v_v_855_, v___x_890_);
if (lean_obj_tag(v___x_891_) == 1)
{
lean_object* v_a_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_900_; 
lean_dec(v_v_855_);
v_a_892_ = lean_ctor_get(v___x_891_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___x_891_);
if (v_isSharedCheck_900_ == 0)
{
v___x_894_ = v___x_891_;
v_isShared_895_ = v_isSharedCheck_900_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_a_892_);
lean_dec(v___x_891_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_900_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_897_; 
if (v_isShared_895_ == 0)
{
v___x_897_ = v___x_894_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_a_892_);
v___x_897_ = v_reuseFailAlloc_899_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
lean_object* v___x_898_; 
v___x_898_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_898_, 0, v_a_889_);
lean_ctor_set(v___x_898_, 1, v___x_897_);
lean_ctor_set(v___x_898_, 2, v___y_886_);
lean_ctor_set(v___x_898_, 3, v___x_854_);
lean_ctor_set(v___x_898_, 4, v___x_854_);
v_a_859_ = v___x_898_;
goto v___jp_858_;
}
}
}
else
{
lean_dec_ref(v___x_891_);
lean_dec(v_a_889_);
lean_dec(v___y_886_);
v___y_865_ = v___y_850_;
goto v___jp_864_;
}
}
else
{
lean_dec_ref(v___x_888_);
lean_dec(v___y_886_);
v___y_865_ = v___y_850_;
goto v___jp_864_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg___boxed(lean_object* v_sz_911_, lean_object* v_i_912_, lean_object* v_bs_913_, lean_object* v___y_914_, lean_object* v___y_915_){
_start:
{
size_t v_sz_boxed_916_; size_t v_i_boxed_917_; lean_object* v_res_918_; 
v_sz_boxed_916_ = lean_unbox_usize(v_sz_911_);
lean_dec(v_sz_911_);
v_i_boxed_917_ = lean_unbox_usize(v_i_912_);
lean_dec(v_i_912_);
v_res_918_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg(v_sz_boxed_916_, v_i_boxed_917_, v_bs_913_, v___y_914_);
lean_dec_ref(v___y_914_);
return v_res_918_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__12(void){
_start:
{
lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v___x_932_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__9));
v___x_933_ = lean_unsigned_to_nat(5u);
v___x_934_ = lean_mk_empty_array_with_capacity(v___x_933_);
v___x_935_ = lean_array_push(v___x_934_, v___x_932_);
return v___x_935_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__13(void){
_start:
{
lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_936_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__10));
v___x_937_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__12, &lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__12_once, _init_lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__12);
v___x_938_ = lean_array_push(v___x_937_, v___x_936_);
return v___x_938_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__14(void){
_start:
{
lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v___x_939_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__11));
v___x_940_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__13, &lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__13_once, _init_lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__13);
v___x_941_ = lean_array_push(v___x_940_, v___x_939_);
return v___x_941_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson(lean_object* v_s_951_, lean_object* v_num__results_952_, lean_object* v_a_953_, lean_object* v_a_954_){
_start:
{
lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v_s_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v_s_970_; lean_object* v___x_971_; lean_object* v___y_973_; lean_object* v___y_974_; lean_object* v___y_981_; lean_object* v___x_1022_; 
v___x_956_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__0));
v___x_957_ = lean_unsigned_to_nat(0u);
v___x_958_ = lean_box(0);
v___x_959_ = l_String_splitOnAux(v_s_951_, v___x_956_, v___x_957_, v___x_957_, v___x_957_, v___x_958_);
v___x_960_ = lp_LeanSearchClient_LeanSearchClient_loogleCache;
v___x_961_ = lean_st_ref_get(v___x_960_);
v___x_962_ = l_List_getD___redArg(v___x_959_, v___x_957_, v_s_951_);
lean_dec(v___x_959_);
v___x_963_ = lean_string_utf8_byte_size(v___x_962_);
v___x_964_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_964_, 0, v___x_962_);
lean_ctor_set(v___x_964_, 1, v___x_957_);
lean_ctor_set(v___x_964_, 2, v___x_963_);
v___x_965_ = l_String_Slice_trimAscii(v___x_964_);
v_s_966_ = l_String_Slice_toString(v___x_965_);
lean_dec_ref(v___x_965_);
v___x_967_ = lean_string_utf8_byte_size(v_s_966_);
v___x_968_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__1));
v___x_969_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_969_, 0, v_s_966_);
lean_ctor_set(v___x_969_, 1, v___x_957_);
lean_ctor_set(v___x_969_, 2, v___x_967_);
v_s_970_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg(v___x_969_, v___x_968_);
lean_inc(v_num__results_952_);
lean_inc_ref(v_s_970_);
v___x_971_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_971_, 0, v_s_970_);
lean_ctor_set(v___x_971_, 1, v_num__results_952_);
v___x_1022_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg(v___x_961_, v___x_971_);
lean_dec(v___x_961_);
if (lean_obj_tag(v___x_1022_) == 0)
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___y_1026_; 
v___x_1023_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__5));
v___x_1024_ = lean_io_getenv(v___x_1023_);
if (lean_obj_tag(v___x_1024_) == 0)
{
lean_object* v___x_1187_; 
v___x_1187_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__22));
v___y_1026_ = v___x_1187_;
goto v___jp_1025_;
}
else
{
lean_object* v_val_1188_; 
v_val_1188_ = lean_ctor_get(v___x_1024_, 0);
lean_inc(v_val_1188_);
lean_dec_ref_known(v___x_1024_, 1);
v___y_1026_ = v_val_1188_;
goto v___jp_1025_;
}
v___jp_1025_:
{
lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; uint8_t v___x_1033_; 
lean_inc_ref(v_s_970_);
v___x_1027_ = l_System_Uri_escapeUri(v_s_970_);
v___x_1028_ = lean_string_utf8_byte_size(v_s_970_);
v___x_1029_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1029_, 0, v_s_970_);
lean_ctor_set(v___x_1029_, 1, v___x_957_);
lean_ctor_set(v___x_1029_, 2, v___x_1028_);
v___x_1030_ = l_String_Slice_trimAscii(v___x_1029_);
v___x_1031_ = l_String_Slice_toString(v___x_1030_);
lean_dec_ref(v___x_1030_);
v___x_1032_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0));
v___x_1033_ = lean_string_dec_eq(v___x_1031_, v___x_1032_);
lean_dec_ref(v___x_1031_);
if (v___x_1033_ == 0)
{
lean_object* v___x_1034_; 
v___x_1034_ = lp_LeanSearchClient_LeanSearchClient_useragent___redArg(v_a_953_);
if (lean_obj_tag(v___x_1034_) == 0)
{
lean_object* v_a_1035_; uint8_t v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v_a_1035_ = lean_ctor_get(v___x_1034_, 0);
lean_inc(v_a_1035_);
lean_dec_ref_known(v___x_1034_, 1);
v___x_1036_ = 1;
v___x_1037_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__6));
v___x_1038_ = lean_string_append(v___x_1037_, v___x_1027_);
v___x_1039_ = lean_string_append(v___y_1026_, v___x_1038_);
lean_dec_ref(v___x_1038_);
v___x_1040_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__7));
v___x_1041_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__8));
v___x_1042_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__14, &lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__14_once, _init_lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__14);
v___x_1043_ = lean_array_push(v___x_1042_, v_a_1035_);
v___x_1044_ = lean_array_push(v___x_1043_, v___x_1039_);
v___x_1045_ = lean_box(0);
v___x_1046_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__15));
v___x_1047_ = lean_alloc_ctor(0, 5, 2);
lean_ctor_set(v___x_1047_, 0, v___x_1040_);
lean_ctor_set(v___x_1047_, 1, v___x_1041_);
lean_ctor_set(v___x_1047_, 2, v___x_1044_);
lean_ctor_set(v___x_1047_, 3, v___x_1045_);
lean_ctor_set(v___x_1047_, 4, v___x_1046_);
lean_ctor_set_uint8(v___x_1047_, sizeof(void*)*5, v___x_1036_);
lean_ctor_set_uint8(v___x_1047_, sizeof(void*)*5 + 1, v___x_1033_);
v___x_1048_ = l_IO_Process_output(v___x_1047_, v___x_1045_);
if (lean_obj_tag(v___x_1048_) == 0)
{
lean_object* v_a_1049_; lean_object* v_stdout_1050_; lean_object* v___x_1051_; 
v_a_1049_ = lean_ctor_get(v___x_1048_, 0);
lean_inc(v_a_1049_);
lean_dec_ref_known(v___x_1048_, 1);
v_stdout_1050_ = lean_ctor_get(v_a_1049_, 0);
lean_inc_ref(v_stdout_1050_);
lean_dec(v_a_1049_);
v___x_1051_ = l_Lean_Json_parse(v_stdout_1050_);
if (lean_obj_tag(v___x_1051_) == 0)
{
lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1080_; 
lean_dec_ref(v___x_1027_);
lean_dec_ref_known(v___x_971_, 2);
lean_dec(v_num__results_952_);
v_isSharedCheck_1080_ = !lean_is_exclusive(v___x_1051_);
if (v_isSharedCheck_1080_ == 0)
{
lean_object* v_unused_1081_; 
v_unused_1081_ = lean_ctor_get(v___x_1051_, 0);
lean_dec(v_unused_1081_);
v___x_1053_ = v___x_1051_;
v_isShared_1054_ = v_isSharedCheck_1080_;
goto v_resetjp_1052_;
}
else
{
lean_dec(v___x_1051_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1080_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; 
v___x_1055_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__16));
v___x_1056_ = l_Lean_IO_throwServerError___redArg(v___x_1055_);
if (lean_obj_tag(v___x_1056_) == 0)
{
lean_object* v_a_1057_; lean_object* v___x_1059_; uint8_t v_isShared_1060_; uint8_t v_isSharedCheck_1064_; 
lean_del_object(v___x_1053_);
v_a_1057_ = lean_ctor_get(v___x_1056_, 0);
v_isSharedCheck_1064_ = !lean_is_exclusive(v___x_1056_);
if (v_isSharedCheck_1064_ == 0)
{
v___x_1059_ = v___x_1056_;
v_isShared_1060_ = v_isSharedCheck_1064_;
goto v_resetjp_1058_;
}
else
{
lean_inc(v_a_1057_);
lean_dec(v___x_1056_);
v___x_1059_ = lean_box(0);
v_isShared_1060_ = v_isSharedCheck_1064_;
goto v_resetjp_1058_;
}
v_resetjp_1058_:
{
lean_object* v___x_1062_; 
if (v_isShared_1060_ == 0)
{
v___x_1062_ = v___x_1059_;
goto v_reusejp_1061_;
}
else
{
lean_object* v_reuseFailAlloc_1063_; 
v_reuseFailAlloc_1063_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1063_, 0, v_a_1057_);
v___x_1062_ = v_reuseFailAlloc_1063_;
goto v_reusejp_1061_;
}
v_reusejp_1061_:
{
return v___x_1062_;
}
}
}
else
{
lean_object* v_a_1065_; lean_object* v___x_1067_; uint8_t v_isShared_1068_; uint8_t v_isSharedCheck_1079_; 
v_a_1065_ = lean_ctor_get(v___x_1056_, 0);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1056_);
if (v_isSharedCheck_1079_ == 0)
{
v___x_1067_ = v___x_1056_;
v_isShared_1068_ = v_isSharedCheck_1079_;
goto v_resetjp_1066_;
}
else
{
lean_inc(v_a_1065_);
lean_dec(v___x_1056_);
v___x_1067_ = lean_box(0);
v_isShared_1068_ = v_isSharedCheck_1079_;
goto v_resetjp_1066_;
}
v_resetjp_1066_:
{
lean_object* v_ref_1069_; lean_object* v___x_1070_; lean_object* v___x_1072_; 
v_ref_1069_ = lean_ctor_get(v_a_953_, 5);
v___x_1070_ = lean_io_error_to_string(v_a_1065_);
if (v_isShared_1054_ == 0)
{
lean_ctor_set_tag(v___x_1053_, 3);
lean_ctor_set(v___x_1053_, 0, v___x_1070_);
v___x_1072_ = v___x_1053_;
goto v_reusejp_1071_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v___x_1070_);
v___x_1072_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1071_;
}
v_reusejp_1071_:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1076_; 
v___x_1073_ = l_Lean_MessageData_ofFormat(v___x_1072_);
lean_inc(v_ref_1069_);
v___x_1074_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1074_, 0, v_ref_1069_);
lean_ctor_set(v___x_1074_, 1, v___x_1073_);
if (v_isShared_1068_ == 0)
{
lean_ctor_set(v___x_1067_, 0, v___x_1074_);
v___x_1076_ = v___x_1067_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v___x_1074_);
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
else
{
lean_object* v_a_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v_a_1085_; lean_object* v___x_1086_; uint8_t v___x_1087_; 
v_a_1082_ = lean_ctor_get(v___x_1051_, 0);
lean_inc_n(v_a_1082_, 2);
lean_dec_ref_known(v___x_1051_, 1);
v___x_1083_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__17));
v___x_1084_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__5(v_a_1082_, v___x_1083_);
v_a_1085_ = lean_ctor_get(v___x_1084_, 0);
lean_inc(v_a_1085_);
lean_dec_ref(v___x_1084_);
v___x_1086_ = lean_box(0);
v___x_1087_ = l___private_Lean_Data_Json_Basic_0__Lean_Json_beq_x27(v_a_1085_, v___x_1086_);
if (v___x_1087_ == 0)
{
lean_object* v___x_1088_; 
lean_inc(v_a_1085_);
v___x_1088_ = l_Lean_Json_getArr_x3f(v_a_1085_);
if (lean_obj_tag(v___x_1088_) == 0)
{
lean_object* v_a_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1131_; 
lean_dec_ref_known(v___x_971_, 2);
lean_dec(v_num__results_952_);
v_a_1089_ = lean_ctor_get(v___x_1088_, 0);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1088_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1091_ = v___x_1088_;
v_isShared_1092_ = v_isSharedCheck_1131_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_a_1089_);
lean_dec(v___x_1088_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1131_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; 
v___x_1093_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__18));
v___x_1094_ = lean_unsigned_to_nat(80u);
v___x_1095_ = l_Lean_Json_pretty(v_a_1082_, v___x_1094_);
v___x_1096_ = lean_string_append(v___x_1093_, v___x_1095_);
lean_dec_ref(v___x_1095_);
v___x_1097_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__19));
v___x_1098_ = lean_string_append(v___x_1096_, v___x_1097_);
v___x_1099_ = lean_string_append(v___x_1098_, v_a_1089_);
lean_dec(v_a_1089_);
v___x_1100_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__20));
v___x_1101_ = lean_string_append(v___x_1099_, v___x_1100_);
v___x_1102_ = lean_string_append(v___x_1101_, v___x_1027_);
lean_dec_ref(v___x_1027_);
v___x_1103_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__21));
v___x_1104_ = lean_string_append(v___x_1102_, v___x_1103_);
v___x_1105_ = l_Lean_Json_pretty(v_a_1085_, v___x_1094_);
v___x_1106_ = lean_string_append(v___x_1104_, v___x_1105_);
lean_dec_ref(v___x_1105_);
v___x_1107_ = l_Lean_IO_throwServerError___redArg(v___x_1106_);
if (lean_obj_tag(v___x_1107_) == 0)
{
lean_object* v_a_1108_; lean_object* v___x_1110_; uint8_t v_isShared_1111_; uint8_t v_isSharedCheck_1115_; 
lean_del_object(v___x_1091_);
v_a_1108_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1115_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1115_ == 0)
{
v___x_1110_ = v___x_1107_;
v_isShared_1111_ = v_isSharedCheck_1115_;
goto v_resetjp_1109_;
}
else
{
lean_inc(v_a_1108_);
lean_dec(v___x_1107_);
v___x_1110_ = lean_box(0);
v_isShared_1111_ = v_isSharedCheck_1115_;
goto v_resetjp_1109_;
}
v_resetjp_1109_:
{
lean_object* v___x_1113_; 
if (v_isShared_1111_ == 0)
{
v___x_1113_ = v___x_1110_;
goto v_reusejp_1112_;
}
else
{
lean_object* v_reuseFailAlloc_1114_; 
v_reuseFailAlloc_1114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1114_, 0, v_a_1108_);
v___x_1113_ = v_reuseFailAlloc_1114_;
goto v_reusejp_1112_;
}
v_reusejp_1112_:
{
return v___x_1113_;
}
}
}
else
{
lean_object* v_a_1116_; lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1130_; 
v_a_1116_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1118_ = v___x_1107_;
v_isShared_1119_ = v_isSharedCheck_1130_;
goto v_resetjp_1117_;
}
else
{
lean_inc(v_a_1116_);
lean_dec(v___x_1107_);
v___x_1118_ = lean_box(0);
v_isShared_1119_ = v_isSharedCheck_1130_;
goto v_resetjp_1117_;
}
v_resetjp_1117_:
{
lean_object* v_ref_1120_; lean_object* v___x_1121_; lean_object* v___x_1123_; 
v_ref_1120_ = lean_ctor_get(v_a_953_, 5);
v___x_1121_ = lean_io_error_to_string(v_a_1116_);
if (v_isShared_1092_ == 0)
{
lean_ctor_set_tag(v___x_1091_, 3);
lean_ctor_set(v___x_1091_, 0, v___x_1121_);
v___x_1123_ = v___x_1091_;
goto v_reusejp_1122_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v___x_1121_);
v___x_1123_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1122_;
}
v_reusejp_1122_:
{
lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1127_; 
v___x_1124_ = l_Lean_MessageData_ofFormat(v___x_1123_);
lean_inc(v_ref_1120_);
v___x_1125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1125_, 0, v_ref_1120_);
lean_ctor_set(v___x_1125_, 1, v___x_1124_);
if (v_isShared_1119_ == 0)
{
lean_ctor_set(v___x_1118_, 0, v___x_1125_);
v___x_1127_ = v___x_1118_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v___x_1125_);
v___x_1127_ = v_reuseFailAlloc_1128_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
return v___x_1127_;
}
}
}
}
}
}
else
{
lean_object* v_a_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1163_; 
lean_dec(v_a_1085_);
lean_dec(v_a_1082_);
lean_dec_ref(v___x_1027_);
v_a_1132_ = lean_ctor_get(v___x_1088_, 0);
v_isSharedCheck_1163_ = !lean_is_exclusive(v___x_1088_);
if (v_isSharedCheck_1163_ == 0)
{
v___x_1134_ = v___x_1088_;
v_isShared_1135_ = v_isSharedCheck_1163_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_a_1132_);
lean_dec(v___x_1088_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1163_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1136_; lean_object* v___x_1137_; size_t v_sz_1138_; size_t v___x_1139_; lean_object* v___x_1140_; 
v___x_1136_ = l_Array_toSubarray___redArg(v_a_1132_, v___x_957_, v_num__results_952_);
v___x_1137_ = lp_LeanSearchClient___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00LeanSearchClient_getLoogleQueryJson_spec__6___redArg(v___x_1136_, v___x_1046_);
v_sz_1138_ = lean_array_size(v___x_1137_);
v___x_1139_ = ((size_t)0ULL);
v___x_1140_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg(v_sz_1138_, v___x_1139_, v___x_1137_, v_a_953_);
if (lean_obj_tag(v___x_1140_) == 0)
{
lean_object* v_a_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1154_; 
v_a_1141_ = lean_ctor_get(v___x_1140_, 0);
v_isSharedCheck_1154_ = !lean_is_exclusive(v___x_1140_);
if (v_isSharedCheck_1154_ == 0)
{
v___x_1143_ = v___x_1140_;
v_isShared_1144_ = v_isSharedCheck_1154_;
goto v_resetjp_1142_;
}
else
{
lean_inc(v_a_1141_);
lean_dec(v___x_1140_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1154_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v___x_1145_; lean_object* v___x_1147_; 
v___x_1145_ = lean_st_ref_take(v___x_960_);
if (v_isShared_1135_ == 0)
{
lean_ctor_set(v___x_1134_, 0, v_a_1141_);
v___x_1147_ = v___x_1134_;
goto v_reusejp_1146_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v_a_1141_);
v___x_1147_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1146_;
}
v_reusejp_1146_:
{
lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1151_; 
lean_inc_ref(v___x_1147_);
v___x_1148_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3___redArg(v___x_1145_, v___x_971_, v___x_1147_);
v___x_1149_ = lean_st_ref_set(v___x_960_, v___x_1148_);
if (v_isShared_1144_ == 0)
{
lean_ctor_set(v___x_1143_, 0, v___x_1147_);
v___x_1151_ = v___x_1143_;
goto v_reusejp_1150_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v___x_1147_);
v___x_1151_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1150_;
}
v_reusejp_1150_:
{
return v___x_1151_;
}
}
}
}
else
{
lean_object* v_a_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1162_; 
lean_del_object(v___x_1134_);
lean_dec_ref_known(v___x_971_, 2);
v_a_1155_ = lean_ctor_get(v___x_1140_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1140_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1157_ = v___x_1140_;
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_a_1155_);
lean_dec(v___x_1140_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1160_; 
if (v_isShared_1158_ == 0)
{
v___x_1160_ = v___x_1157_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v_a_1155_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
}
}
}
else
{
lean_dec(v_a_1085_);
lean_dec_ref(v___x_1027_);
lean_dec(v_num__results_952_);
v___y_981_ = v_a_1082_;
goto v___jp_980_;
}
}
}
else
{
lean_object* v_a_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1176_; 
lean_dec_ref(v___x_1027_);
lean_dec_ref_known(v___x_971_, 2);
lean_dec(v_num__results_952_);
v_a_1164_ = lean_ctor_get(v___x_1048_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v___x_1048_);
if (v_isSharedCheck_1176_ == 0)
{
v___x_1166_ = v___x_1048_;
v_isShared_1167_ = v_isSharedCheck_1176_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_a_1164_);
lean_dec(v___x_1048_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1176_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v_ref_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1174_; 
v_ref_1168_ = lean_ctor_get(v_a_953_, 5);
v___x_1169_ = lean_io_error_to_string(v_a_1164_);
v___x_1170_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1170_, 0, v___x_1169_);
v___x_1171_ = l_Lean_MessageData_ofFormat(v___x_1170_);
lean_inc(v_ref_1168_);
v___x_1172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1172_, 0, v_ref_1168_);
lean_ctor_set(v___x_1172_, 1, v___x_1171_);
if (v_isShared_1167_ == 0)
{
lean_ctor_set(v___x_1166_, 0, v___x_1172_);
v___x_1174_ = v___x_1166_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v___x_1172_);
v___x_1174_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
return v___x_1174_;
}
}
}
}
else
{
lean_object* v_a_1177_; lean_object* v___x_1179_; uint8_t v_isShared_1180_; uint8_t v_isSharedCheck_1184_; 
lean_dec_ref(v___x_1027_);
lean_dec_ref(v___y_1026_);
lean_dec_ref_known(v___x_971_, 2);
lean_dec(v_num__results_952_);
v_a_1177_ = lean_ctor_get(v___x_1034_, 0);
v_isSharedCheck_1184_ = !lean_is_exclusive(v___x_1034_);
if (v_isSharedCheck_1184_ == 0)
{
v___x_1179_ = v___x_1034_;
v_isShared_1180_ = v_isSharedCheck_1184_;
goto v_resetjp_1178_;
}
else
{
lean_inc(v_a_1177_);
lean_dec(v___x_1034_);
v___x_1179_ = lean_box(0);
v_isShared_1180_ = v_isSharedCheck_1184_;
goto v_resetjp_1178_;
}
v_resetjp_1178_:
{
lean_object* v___x_1182_; 
if (v_isShared_1180_ == 0)
{
v___x_1182_ = v___x_1179_;
goto v_reusejp_1181_;
}
else
{
lean_object* v_reuseFailAlloc_1183_; 
v_reuseFailAlloc_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1183_, 0, v_a_1177_);
v___x_1182_ = v_reuseFailAlloc_1183_;
goto v_reusejp_1181_;
}
v_reusejp_1181_:
{
return v___x_1182_;
}
}
}
}
else
{
lean_object* v___x_1185_; lean_object* v___x_1186_; 
lean_dec_ref(v___x_1027_);
lean_dec_ref(v___y_1026_);
lean_dec_ref_known(v___x_971_, 2);
lean_dec(v_num__results_952_);
v___x_1185_ = lean_box(0);
v___x_1186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1185_);
return v___x_1186_;
}
}
}
else
{
lean_object* v_val_1189_; lean_object* v___x_1191_; uint8_t v_isShared_1192_; uint8_t v_isSharedCheck_1196_; 
lean_dec_ref_known(v___x_971_, 2);
lean_dec_ref(v_s_970_);
lean_dec(v_num__results_952_);
v_val_1189_ = lean_ctor_get(v___x_1022_, 0);
v_isSharedCheck_1196_ = !lean_is_exclusive(v___x_1022_);
if (v_isSharedCheck_1196_ == 0)
{
v___x_1191_ = v___x_1022_;
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
else
{
lean_inc(v_val_1189_);
lean_dec(v___x_1022_);
v___x_1191_ = lean_box(0);
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
v_resetjp_1190_:
{
lean_object* v___x_1194_; 
if (v_isShared_1192_ == 0)
{
lean_ctor_set_tag(v___x_1191_, 0);
v___x_1194_ = v___x_1191_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_val_1189_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
v___jp_972_:
{
lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; 
v___x_975_ = lean_st_ref_take(v___x_960_);
v___x_976_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_976_, 0, v___y_973_);
lean_ctor_set(v___x_976_, 1, v___y_974_);
lean_inc_ref(v___x_976_);
v___x_977_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3___redArg(v___x_975_, v___x_971_, v___x_976_);
v___x_978_ = lean_st_ref_set(v___x_960_, v___x_977_);
v___x_979_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_979_, 0, v___x_976_);
return v___x_979_;
}
v___jp_980_:
{
lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_982_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__2));
lean_inc(v___y_981_);
v___x_983_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__2(v___y_981_, v___x_982_);
if (lean_obj_tag(v___x_983_) == 1)
{
lean_object* v_a_984_; lean_object* v___x_985_; lean_object* v___x_986_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
lean_inc(v_a_984_);
lean_dec_ref_known(v___x_983_, 1);
v___x_985_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__3));
v___x_986_ = lp_LeanSearchClient_Lean_Json_getObjValAs_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__4(v___y_981_, v___x_985_);
if (lean_obj_tag(v___x_986_) == 0)
{
lean_object* v___x_987_; 
lean_dec_ref_known(v___x_986_, 1);
v___x_987_ = lean_box(0);
v___y_973_ = v_a_984_;
v___y_974_ = v___x_987_;
goto v___jp_972_;
}
else
{
lean_object* v_a_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_995_; 
v_a_988_ = lean_ctor_get(v___x_986_, 0);
v_isSharedCheck_995_ = !lean_is_exclusive(v___x_986_);
if (v_isSharedCheck_995_ == 0)
{
v___x_990_ = v___x_986_;
v_isShared_991_ = v_isSharedCheck_995_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_a_988_);
lean_dec(v___x_986_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_995_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v___x_993_; 
if (v_isShared_991_ == 0)
{
v___x_993_ = v___x_990_;
goto v_reusejp_992_;
}
else
{
lean_object* v_reuseFailAlloc_994_; 
v_reuseFailAlloc_994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_994_, 0, v_a_988_);
v___x_993_ = v_reuseFailAlloc_994_;
goto v_reusejp_992_;
}
v_reusejp_992_:
{
v___y_973_ = v_a_984_;
v___y_974_ = v___x_993_;
goto v___jp_972_;
}
}
}
}
else
{
lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
lean_dec_ref(v___x_983_);
lean_dec_ref_known(v___x_971_, 2);
v___x_996_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___closed__4));
v___x_997_ = lean_unsigned_to_nat(80u);
v___x_998_ = l_Lean_Json_pretty(v___y_981_, v___x_997_);
v___x_999_ = lean_string_append(v___x_996_, v___x_998_);
lean_dec_ref(v___x_998_);
v___x_1000_ = l_Lean_IO_throwServerError___redArg(v___x_999_);
if (lean_obj_tag(v___x_1000_) == 0)
{
lean_object* v_a_1001_; lean_object* v___x_1003_; uint8_t v_isShared_1004_; uint8_t v_isSharedCheck_1008_; 
v_a_1001_ = lean_ctor_get(v___x_1000_, 0);
v_isSharedCheck_1008_ = !lean_is_exclusive(v___x_1000_);
if (v_isSharedCheck_1008_ == 0)
{
v___x_1003_ = v___x_1000_;
v_isShared_1004_ = v_isSharedCheck_1008_;
goto v_resetjp_1002_;
}
else
{
lean_inc(v_a_1001_);
lean_dec(v___x_1000_);
v___x_1003_ = lean_box(0);
v_isShared_1004_ = v_isSharedCheck_1008_;
goto v_resetjp_1002_;
}
v_resetjp_1002_:
{
lean_object* v___x_1006_; 
if (v_isShared_1004_ == 0)
{
v___x_1006_ = v___x_1003_;
goto v_reusejp_1005_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v_a_1001_);
v___x_1006_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1005_;
}
v_reusejp_1005_:
{
return v___x_1006_;
}
}
}
else
{
lean_object* v_a_1009_; lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1021_; 
v_a_1009_ = lean_ctor_get(v___x_1000_, 0);
v_isSharedCheck_1021_ = !lean_is_exclusive(v___x_1000_);
if (v_isSharedCheck_1021_ == 0)
{
v___x_1011_ = v___x_1000_;
v_isShared_1012_ = v_isSharedCheck_1021_;
goto v_resetjp_1010_;
}
else
{
lean_inc(v_a_1009_);
lean_dec(v___x_1000_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1021_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
lean_object* v_ref_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1019_; 
v_ref_1013_ = lean_ctor_get(v_a_953_, 5);
v___x_1014_ = lean_io_error_to_string(v_a_1009_);
v___x_1015_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1014_);
v___x_1016_ = l_Lean_MessageData_ofFormat(v___x_1015_);
lean_inc(v_ref_1013_);
v___x_1017_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1017_, 0, v_ref_1013_);
lean_ctor_set(v___x_1017_, 1, v___x_1016_);
if (v_isShared_1012_ == 0)
{
lean_ctor_set(v___x_1011_, 0, v___x_1017_);
v___x_1019_ = v___x_1011_;
goto v_reusejp_1018_;
}
else
{
lean_object* v_reuseFailAlloc_1020_; 
v_reuseFailAlloc_1020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1020_, 0, v___x_1017_);
v___x_1019_ = v_reuseFailAlloc_1020_;
goto v_reusejp_1018_;
}
v_reusejp_1018_:
{
return v___x_1019_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson___boxed(lean_object* v_s_1197_, lean_object* v_num__results_1198_, lean_object* v_a_1199_, lean_object* v_a_1200_, lean_object* v_a_1201_){
_start:
{
lean_object* v_res_1202_; 
v_res_1202_ = lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson(v_s_1197_, v_num__results_1198_, v_a_1199_, v_a_1200_);
lean_dec(v_a_1200_);
lean_dec_ref(v_a_1199_);
lean_dec_ref(v_s_1197_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0(lean_object* v_s_1203_, lean_object* v_pattern_1204_, lean_object* v_replacement_1205_){
_start:
{
lean_object* v___x_1206_; 
v___x_1206_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg(v_s_1203_, v_replacement_1205_);
return v___x_1206_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___boxed(lean_object* v_s_1207_, lean_object* v_pattern_1208_, lean_object* v_replacement_1209_){
_start:
{
lean_object* v_res_1210_; 
v_res_1210_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0(v_s_1207_, v_pattern_1208_, v_replacement_1209_);
lean_dec_ref(v_replacement_1209_);
lean_dec_ref(v_pattern_1208_);
return v_res_1210_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1(lean_object* v_00_u03b2_1211_, lean_object* v_m_1212_, lean_object* v_a_1213_){
_start:
{
lean_object* v___x_1214_; 
v___x_1214_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___redArg(v_m_1212_, v_a_1213_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1___boxed(lean_object* v_00_u03b2_1215_, lean_object* v_m_1216_, lean_object* v_a_1217_){
_start:
{
lean_object* v_res_1218_; 
v_res_1218_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1(v_00_u03b2_1215_, v_m_1216_, v_a_1217_);
lean_dec_ref(v_a_1217_);
lean_dec_ref(v_m_1216_);
return v_res_1218_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3(lean_object* v_00_u03b2_1219_, lean_object* v_m_1220_, lean_object* v_a_1221_, lean_object* v_b_1222_){
_start:
{
lean_object* v___x_1223_; 
v___x_1223_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3___redArg(v_m_1220_, v_a_1221_, v_b_1222_);
return v___x_1223_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00LeanSearchClient_getLoogleQueryJson_spec__6(lean_object* v_inst_1224_, lean_object* v_R_1225_, lean_object* v_a_1226_, lean_object* v_b_1227_){
_start:
{
lean_object* v___x_1228_; 
v___x_1228_ = lp_LeanSearchClient___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00LeanSearchClient_getLoogleQueryJson_spec__6___redArg(v_a_1226_, v_b_1227_);
return v___x_1228_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7(size_t v_sz_1229_, size_t v_i_1230_, lean_object* v_bs_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
lean_object* v___x_1235_; 
v___x_1235_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___redArg(v_sz_1229_, v_i_1230_, v_bs_1231_, v___y_1232_);
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7___boxed(lean_object* v_sz_1236_, lean_object* v_i_1237_, lean_object* v_bs_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
size_t v_sz_boxed_1242_; size_t v_i_boxed_1243_; lean_object* v_res_1244_; 
v_sz_boxed_1242_ = lean_unbox_usize(v_sz_1236_);
lean_dec(v_sz_1236_);
v_i_boxed_1243_ = lean_unbox_usize(v_i_1237_);
lean_dec(v_i_1237_);
v_res_1244_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_getLoogleQueryJson_spec__7(v_sz_boxed_1242_, v_i_boxed_1243_, v_bs_1238_, v___y_1239_, v___y_1240_);
lean_dec(v___y_1240_);
lean_dec_ref(v___y_1239_);
return v_res_1244_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0(lean_object* v_s_1245_, lean_object* v_replacement_1246_, lean_object* v_inst_1247_, lean_object* v_R_1248_, lean_object* v_a_1249_, lean_object* v_b_1250_, lean_object* v_c_1251_){
_start:
{
lean_object* v___x_1252_; 
v___x_1252_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(v_s_1245_, v_replacement_1246_, v_a_1249_, v_b_1250_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___boxed(lean_object* v_s_1253_, lean_object* v_replacement_1254_, lean_object* v_inst_1255_, lean_object* v_R_1256_, lean_object* v_a_1257_, lean_object* v_b_1258_, lean_object* v_c_1259_){
_start:
{
lean_object* v_res_1260_; 
v_res_1260_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0(v_s_1253_, v_replacement_1254_, v_inst_1255_, v_R_1256_, v_a_1257_, v_b_1258_, v_c_1259_);
lean_dec_ref(v_replacement_1254_);
return v_res_1260_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2(lean_object* v_00_u03b2_1261_, lean_object* v_a_1262_, lean_object* v_x_1263_){
_start:
{
lean_object* v___x_1264_; 
v___x_1264_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___redArg(v_a_1262_, v_x_1263_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1265_, lean_object* v_a_1266_, lean_object* v_x_1267_){
_start:
{
lean_object* v_res_1268_; 
v_res_1268_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00LeanSearchClient_getLoogleQueryJson_spec__1_spec__2(v_00_u03b2_1265_, v_a_1266_, v_x_1267_);
lean_dec(v_x_1267_);
lean_dec_ref(v_a_1266_);
return v_res_1268_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5(lean_object* v_00_u03b2_1269_, lean_object* v_a_1270_, lean_object* v_x_1271_){
_start:
{
uint8_t v___x_1272_; 
v___x_1272_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___redArg(v_a_1270_, v_x_1271_);
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1273_, lean_object* v_a_1274_, lean_object* v_x_1275_){
_start:
{
uint8_t v_res_1276_; lean_object* v_r_1277_; 
v_res_1276_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__5(v_00_u03b2_1273_, v_a_1274_, v_x_1275_);
lean_dec(v_x_1275_);
lean_dec_ref(v_a_1274_);
v_r_1277_ = lean_box(v_res_1276_);
return v_r_1277_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6(lean_object* v_00_u03b2_1278_, lean_object* v_data_1279_){
_start:
{
lean_object* v___x_1280_; 
v___x_1280_ = lp_LeanSearchClient_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6___redArg(v_data_1279_);
return v___x_1280_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7(lean_object* v_00_u03b2_1281_, lean_object* v_a_1282_, lean_object* v_b_1283_, lean_object* v_x_1284_){
_start:
{
lean_object* v___x_1285_; 
v___x_1285_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__7___redArg(v_a_1282_, v_b_1283_, v_x_1284_);
return v___x_1285_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8(lean_object* v_00_u03b2_1286_, lean_object* v_i_1287_, lean_object* v_source_1288_, lean_object* v_target_1289_){
_start:
{
lean_object* v___x_1290_; 
v___x_1290_ = lp_LeanSearchClient___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8___redArg(v_i_1287_, v_source_1288_, v_target_1289_);
return v___x_1290_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8_spec__12(lean_object* v_00_u03b2_1291_, lean_object* v_x_1292_, lean_object* v_x_1293_){
_start:
{
lean_object* v___x_1294_; 
v___x_1294_ = lp_LeanSearchClient_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00LeanSearchClient_getLoogleQueryJson_spec__3_spec__6_spec__8_spec__12___redArg(v_x_1292_, v_x_1293_);
return v___x_1294_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__1(void){
_start:
{
uint8_t v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; 
v___x_1298_ = 0;
v___x_1299_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__0));
v___x_1300_ = l_Lean_Parser_nonReservedSymbol(v___x_1299_, v___x_1298_);
return v___x_1300_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_unicode__turnstile(void){
_start:
{
lean_object* v___x_1301_; 
v___x_1301_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__1, &lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__1);
return v___x_1301_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__1(void){
_start:
{
uint8_t v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; 
v___x_1303_ = 0;
v___x_1304_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__0));
v___x_1305_ = l_Lean_Parser_nonReservedSymbol(v___x_1304_, v___x_1303_);
return v___x_1305_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_ascii__turnstile(void){
_start:
{
lean_object* v___x_1306_; 
v___x_1306_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__1, &lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__1);
return v___x_1306_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_formatter(lean_object* v_a_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_, lean_object* v_a_1410_){
_start:
{
lean_object* v___x_1412_; lean_object* v___x_1413_; 
v___x_1412_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__0));
v___x_1413_ = l_Lean_Parser_nonReservedSymbol_formatter___redArg(v___x_1412_, v_a_1407_, v_a_1408_, v_a_1409_, v_a_1410_);
return v___x_1413_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_formatter___boxed(lean_object* v_a_1414_, lean_object* v_a_1415_, lean_object* v_a_1416_, lean_object* v_a_1417_, lean_object* v_a_1418_){
_start:
{
lean_object* v_res_1419_; 
v_res_1419_ = lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_formatter(v_a_1414_, v_a_1415_, v_a_1416_, v_a_1417_);
lean_dec(v_a_1417_);
lean_dec_ref(v_a_1416_);
lean_dec(v_a_1415_);
lean_dec_ref(v_a_1414_);
return v_res_1419_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_formatter(lean_object* v_a_1420_, lean_object* v_a_1421_, lean_object* v_a_1422_, lean_object* v_a_1423_){
_start:
{
lean_object* v___x_1425_; lean_object* v___x_1426_; 
v___x_1425_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__0));
v___x_1426_ = l_Lean_Parser_nonReservedSymbol_formatter___redArg(v___x_1425_, v_a_1420_, v_a_1421_, v_a_1422_, v_a_1423_);
return v___x_1426_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_formatter___boxed(lean_object* v_a_1427_, lean_object* v_a_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_, lean_object* v_a_1431_){
_start:
{
lean_object* v_res_1432_; 
v_res_1432_ = lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_formatter(v_a_1427_, v_a_1428_, v_a_1429_, v_a_1430_);
lean_dec(v_a_1430_);
lean_dec_ref(v_a_1429_);
lean_dec(v_a_1428_);
lean_dec_ref(v_a_1427_);
return v_res_1432_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_parenthesizer(lean_object* v_a_1433_, lean_object* v_a_1434_, lean_object* v_a_1435_, lean_object* v_a_1436_){
_start:
{
lean_object* v___x_1438_; lean_object* v___x_1439_; 
v___x_1438_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_unicode__turnstile___closed__0));
v___x_1439_ = l_Lean_Parser_nonReservedSymbol_parenthesizer___redArg(v___x_1438_, v_a_1433_, v_a_1434_, v_a_1435_, v_a_1436_);
return v___x_1439_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_parenthesizer___boxed(lean_object* v_a_1440_, lean_object* v_a_1441_, lean_object* v_a_1442_, lean_object* v_a_1443_, lean_object* v_a_1444_){
_start:
{
lean_object* v_res_1445_; 
v_res_1445_ = lp_LeanSearchClient_LeanSearchClient_unicode__turnstile_parenthesizer(v_a_1440_, v_a_1441_, v_a_1442_, v_a_1443_);
lean_dec(v_a_1443_);
lean_dec_ref(v_a_1442_);
lean_dec(v_a_1441_);
lean_dec_ref(v_a_1440_);
return v_res_1445_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_parenthesizer(lean_object* v_a_1446_, lean_object* v_a_1447_, lean_object* v_a_1448_, lean_object* v_a_1449_){
_start:
{
lean_object* v___x_1451_; lean_object* v___x_1452_; 
v___x_1451_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_ascii__turnstile___closed__0));
v___x_1452_ = l_Lean_Parser_nonReservedSymbol_parenthesizer___redArg(v___x_1451_, v_a_1446_, v_a_1447_, v_a_1448_, v_a_1449_);
return v___x_1452_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_parenthesizer___boxed(lean_object* v_a_1453_, lean_object* v_a_1454_, lean_object* v_a_1455_, lean_object* v_a_1456_, lean_object* v_a_1457_){
_start:
{
lean_object* v_res_1458_; 
v_res_1458_ = lp_LeanSearchClient_LeanSearchClient_ascii__turnstile_parenthesizer(v_a_1453_, v_a_1454_, v_a_1455_, v_a_1456_);
lean_dec(v_a_1456_);
lean_dec_ref(v_a_1455_);
lean_dec(v_a_1454_);
lean_dec_ref(v_a_1453_);
return v_res_1458_;
}
}
static lean_object* _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; 
v___x_1459_ = lean_box(0);
v___x_1460_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1461_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1461_, 0, v___x_1460_);
lean_ctor_set(v___x_1461_, 1, v___x_1459_);
return v___x_1461_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg(){
_start:
{
lean_object* v___x_1463_; lean_object* v___x_1464_; 
v___x_1463_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0, &lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0_once, _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0);
v___x_1464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1464_, 0, v___x_1463_);
return v___x_1464_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___boxed(lean_object* v___y_1465_){
_start:
{
lean_object* v_res_1466_; 
v_res_1466_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg();
return v_res_1466_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0(lean_object* v_00_u03b1_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
lean_object* v___x_1475_; 
v___x_1475_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg();
return v___x_1475_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___boxed(lean_object* v_00_u03b1_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_){
_start:
{
lean_object* v_res_1484_; 
v_res_1484_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0(v_00_u03b1_1476_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_, v___y_1482_);
lean_dec(v___y_1482_);
lean_dec_ref(v___y_1481_);
lean_dec(v___y_1480_);
lean_dec_ref(v___y_1479_);
lean_dec(v___y_1478_);
lean_dec_ref(v___y_1477_);
return v_res_1484_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleCmdImpl_spec__2(size_t v_sz_1485_, size_t v_i_1486_, lean_object* v_bs_1487_){
_start:
{
uint8_t v___x_1488_; 
v___x_1488_ = lean_usize_dec_lt(v_i_1486_, v_sz_1485_);
if (v___x_1488_ == 0)
{
return v_bs_1487_;
}
else
{
lean_object* v_v_1489_; lean_object* v___x_1490_; lean_object* v_bs_x27_1491_; lean_object* v___x_1492_; size_t v___x_1493_; size_t v___x_1494_; lean_object* v___x_1495_; 
v_v_1489_ = lean_array_uget(v_bs_1487_, v_i_1486_);
v___x_1490_ = lean_unsigned_to_nat(0u);
v_bs_x27_1491_ = lean_array_uset(v_bs_1487_, v_i_1486_, v___x_1490_);
v___x_1492_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_toCommandSuggestion(v_v_1489_);
v___x_1493_ = ((size_t)1ULL);
v___x_1494_ = lean_usize_add(v_i_1486_, v___x_1493_);
v___x_1495_ = lean_array_uset(v_bs_x27_1491_, v_i_1486_, v___x_1492_);
v_i_1486_ = v___x_1494_;
v_bs_1487_ = v___x_1495_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleCmdImpl_spec__2___boxed(lean_object* v_sz_1497_, lean_object* v_i_1498_, lean_object* v_bs_1499_){
_start:
{
size_t v_sz_boxed_1500_; size_t v_i_boxed_1501_; lean_object* v_res_1502_; 
v_sz_boxed_1500_ = lean_unbox_usize(v_sz_1497_);
lean_dec(v_sz_1497_);
v_i_boxed_1501_ = lean_unbox_usize(v_i_1498_);
lean_dec(v_i_1498_);
v_res_1502_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleCmdImpl_spec__2(v_sz_boxed_1500_, v_i_boxed_1501_, v_bs_1499_);
return v_res_1502_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4(lean_object* v_a_1504_, lean_object* v_a_1505_){
_start:
{
if (lean_obj_tag(v_a_1504_) == 0)
{
lean_object* v___x_1506_; 
v___x_1506_ = l_List_reverse___redArg(v_a_1505_);
return v___x_1506_;
}
else
{
lean_object* v_head_1507_; lean_object* v_tail_1508_; lean_object* v___x_1510_; uint8_t v_isShared_1511_; uint8_t v_isSharedCheck_1521_; 
v_head_1507_ = lean_ctor_get(v_a_1504_, 0);
v_tail_1508_ = lean_ctor_get(v_a_1504_, 1);
v_isSharedCheck_1521_ = !lean_is_exclusive(v_a_1504_);
if (v_isSharedCheck_1521_ == 0)
{
v___x_1510_ = v_a_1504_;
v_isShared_1511_ = v_isSharedCheck_1521_;
goto v_resetjp_1509_;
}
else
{
lean_inc(v_tail_1508_);
lean_inc(v_head_1507_);
lean_dec(v_a_1504_);
v___x_1510_ = lean_box(0);
v_isShared_1511_ = v_isSharedCheck_1521_;
goto v_resetjp_1509_;
}
v_resetjp_1509_:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1518_; 
v___x_1512_ = ((lean_object*)(lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4___closed__0));
v___x_1513_ = lean_string_append(v___x_1512_, v_head_1507_);
lean_dec(v_head_1507_);
v___x_1514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1514_, 0, v___x_1513_);
v___x_1515_ = lean_box(0);
v___x_1516_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1516_, 0, v___x_1514_);
lean_ctor_set(v___x_1516_, 1, v___x_1515_);
lean_ctor_set(v___x_1516_, 2, v___x_1515_);
lean_ctor_set(v___x_1516_, 3, v___x_1515_);
lean_ctor_set(v___x_1516_, 4, v___x_1515_);
lean_ctor_set(v___x_1516_, 5, v___x_1515_);
if (v_isShared_1511_ == 0)
{
lean_ctor_set(v___x_1510_, 1, v_a_1505_);
lean_ctor_set(v___x_1510_, 0, v___x_1516_);
v___x_1518_ = v___x_1510_;
goto v_reusejp_1517_;
}
else
{
lean_object* v_reuseFailAlloc_1520_; 
v_reuseFailAlloc_1520_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1520_, 0, v___x_1516_);
lean_ctor_set(v_reuseFailAlloc_1520_, 1, v_a_1505_);
v___x_1518_ = v_reuseFailAlloc_1520_;
goto v_reusejp_1517_;
}
v_reusejp_1517_:
{
v_a_1504_ = v_tail_1508_;
v_a_1505_ = v___x_1518_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7(lean_object* v_opts_1522_, lean_object* v_opt_1523_){
_start:
{
lean_object* v_name_1524_; lean_object* v_defValue_1525_; lean_object* v_map_1526_; lean_object* v___x_1527_; 
v_name_1524_ = lean_ctor_get(v_opt_1523_, 0);
v_defValue_1525_ = lean_ctor_get(v_opt_1523_, 1);
v_map_1526_ = lean_ctor_get(v_opts_1522_, 0);
v___x_1527_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1526_, v_name_1524_);
if (lean_obj_tag(v___x_1527_) == 0)
{
uint8_t v___x_1528_; 
v___x_1528_ = lean_unbox(v_defValue_1525_);
return v___x_1528_;
}
else
{
lean_object* v_val_1529_; 
v_val_1529_ = lean_ctor_get(v___x_1527_, 0);
lean_inc(v_val_1529_);
lean_dec_ref_known(v___x_1527_, 1);
if (lean_obj_tag(v_val_1529_) == 1)
{
uint8_t v_v_1530_; 
v_v_1530_ = lean_ctor_get_uint8(v_val_1529_, 0);
lean_dec_ref_known(v_val_1529_, 0);
return v_v_1530_;
}
else
{
uint8_t v___x_1531_; 
lean_dec(v_val_1529_);
v___x_1531_ = lean_unbox(v_defValue_1525_);
return v___x_1531_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7___boxed(lean_object* v_opts_1532_, lean_object* v_opt_1533_){
_start:
{
uint8_t v_res_1534_; lean_object* v_r_1535_; 
v_res_1534_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7(v_opts_1532_, v_opt_1533_);
lean_dec_ref(v_opt_1533_);
lean_dec_ref(v_opts_1532_);
v_r_1535_ = lean_box(v_res_1534_);
return v_r_1535_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6(lean_object* v_msgData_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_, lean_object* v___y_1540_){
_start:
{
lean_object* v___x_1542_; lean_object* v_env_1543_; lean_object* v___x_1544_; lean_object* v_mctx_1545_; lean_object* v_lctx_1546_; lean_object* v_options_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; 
v___x_1542_ = lean_st_ref_get(v___y_1540_);
v_env_1543_ = lean_ctor_get(v___x_1542_, 0);
lean_inc_ref(v_env_1543_);
lean_dec(v___x_1542_);
v___x_1544_ = lean_st_ref_get(v___y_1538_);
v_mctx_1545_ = lean_ctor_get(v___x_1544_, 0);
lean_inc_ref(v_mctx_1545_);
lean_dec(v___x_1544_);
v_lctx_1546_ = lean_ctor_get(v___y_1537_, 2);
v_options_1547_ = lean_ctor_get(v___y_1539_, 2);
lean_inc_ref(v_options_1547_);
lean_inc_ref(v_lctx_1546_);
v___x_1548_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1548_, 0, v_env_1543_);
lean_ctor_set(v___x_1548_, 1, v_mctx_1545_);
lean_ctor_set(v___x_1548_, 2, v_lctx_1546_);
lean_ctor_set(v___x_1548_, 3, v_options_1547_);
v___x_1549_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1549_, 0, v___x_1548_);
lean_ctor_set(v___x_1549_, 1, v_msgData_1536_);
v___x_1550_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1550_, 0, v___x_1549_);
return v___x_1550_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6___boxed(lean_object* v_msgData_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_){
_start:
{
lean_object* v_res_1557_; 
v_res_1557_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6(v_msgData_1551_, v___y_1552_, v___y_1553_, v___y_1554_, v___y_1555_);
lean_dec(v___y_1555_);
lean_dec_ref(v___y_1554_);
lean_dec(v___y_1553_);
lean_dec_ref(v___y_1552_);
return v_res_1557_;
}
}
LEAN_EXPORT uint8_t lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0(uint8_t v___y_1566_, uint8_t v_suppressElabErrors_1567_, lean_object* v_x_1568_){
_start:
{
if (lean_obj_tag(v_x_1568_) == 1)
{
lean_object* v_pre_1569_; 
v_pre_1569_ = lean_ctor_get(v_x_1568_, 0);
switch(lean_obj_tag(v_pre_1569_))
{
case 1:
{
lean_object* v_pre_1570_; 
v_pre_1570_ = lean_ctor_get(v_pre_1569_, 0);
switch(lean_obj_tag(v_pre_1570_))
{
case 0:
{
lean_object* v_str_1571_; lean_object* v_str_1572_; lean_object* v___x_1573_; uint8_t v___x_1574_; 
v_str_1571_ = lean_ctor_get(v_x_1568_, 1);
v_str_1572_ = lean_ctor_get(v_pre_1569_, 1);
v___x_1573_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__0));
v___x_1574_ = lean_string_dec_eq(v_str_1572_, v___x_1573_);
if (v___x_1574_ == 0)
{
lean_object* v___x_1575_; uint8_t v___x_1576_; 
v___x_1575_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__1));
v___x_1576_ = lean_string_dec_eq(v_str_1572_, v___x_1575_);
if (v___x_1576_ == 0)
{
return v___y_1566_;
}
else
{
lean_object* v___x_1577_; uint8_t v___x_1578_; 
v___x_1577_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__2));
v___x_1578_ = lean_string_dec_eq(v_str_1571_, v___x_1577_);
if (v___x_1578_ == 0)
{
return v___y_1566_;
}
else
{
return v_suppressElabErrors_1567_;
}
}
}
else
{
lean_object* v___x_1579_; uint8_t v___x_1580_; 
v___x_1579_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__3));
v___x_1580_ = lean_string_dec_eq(v_str_1571_, v___x_1579_);
if (v___x_1580_ == 0)
{
return v___y_1566_;
}
else
{
return v_suppressElabErrors_1567_;
}
}
}
case 1:
{
lean_object* v_pre_1581_; 
v_pre_1581_ = lean_ctor_get(v_pre_1570_, 0);
if (lean_obj_tag(v_pre_1581_) == 0)
{
lean_object* v_str_1582_; lean_object* v_str_1583_; lean_object* v_str_1584_; lean_object* v___x_1585_; uint8_t v___x_1586_; 
v_str_1582_ = lean_ctor_get(v_x_1568_, 1);
v_str_1583_ = lean_ctor_get(v_pre_1569_, 1);
v_str_1584_ = lean_ctor_get(v_pre_1570_, 1);
v___x_1585_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__4));
v___x_1586_ = lean_string_dec_eq(v_str_1584_, v___x_1585_);
if (v___x_1586_ == 0)
{
return v___y_1566_;
}
else
{
lean_object* v___x_1587_; uint8_t v___x_1588_; 
v___x_1587_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__5));
v___x_1588_ = lean_string_dec_eq(v_str_1583_, v___x_1587_);
if (v___x_1588_ == 0)
{
return v___y_1566_;
}
else
{
lean_object* v___x_1589_; uint8_t v___x_1590_; 
v___x_1589_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__6));
v___x_1590_ = lean_string_dec_eq(v_str_1582_, v___x_1589_);
if (v___x_1590_ == 0)
{
return v___y_1566_;
}
else
{
return v_suppressElabErrors_1567_;
}
}
}
}
else
{
return v___y_1566_;
}
}
default: 
{
return v___y_1566_;
}
}
}
case 0:
{
lean_object* v_str_1591_; lean_object* v___x_1592_; uint8_t v___x_1593_; 
v_str_1591_ = lean_ctor_get(v_x_1568_, 1);
v___x_1592_ = ((lean_object*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___closed__7));
v___x_1593_ = lean_string_dec_eq(v_str_1591_, v___x_1592_);
if (v___x_1593_ == 0)
{
return v___y_1566_;
}
else
{
return v_suppressElabErrors_1567_;
}
}
default: 
{
return v___y_1566_;
}
}
}
else
{
return v___y_1566_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___boxed(lean_object* v___y_1594_, lean_object* v_suppressElabErrors_1595_, lean_object* v_x_1596_){
_start:
{
uint8_t v___y_6770__boxed_1597_; uint8_t v_suppressElabErrors_boxed_1598_; uint8_t v_res_1599_; lean_object* v_r_1600_; 
v___y_6770__boxed_1597_ = lean_unbox(v___y_1594_);
v_suppressElabErrors_boxed_1598_ = lean_unbox(v_suppressElabErrors_1595_);
v_res_1599_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0(v___y_6770__boxed_1597_, v_suppressElabErrors_boxed_1598_, v_x_1596_);
lean_dec(v_x_1596_);
v_r_1600_ = lean_box(v_res_1599_);
return v_r_1600_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg(lean_object* v_ref_1601_, lean_object* v_msgData_1602_, uint8_t v_severity_1603_, uint8_t v_isSilent_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_){
_start:
{
uint8_t v___y_1611_; uint8_t v___y_1612_; lean_object* v___y_1613_; lean_object* v___y_1614_; lean_object* v___y_1615_; lean_object* v___y_1616_; lean_object* v___y_1617_; lean_object* v___y_1618_; lean_object* v___y_1619_; lean_object* v___y_1647_; uint8_t v___y_1648_; uint8_t v___y_1649_; uint8_t v___y_1650_; lean_object* v___y_1651_; lean_object* v___y_1652_; lean_object* v___y_1653_; lean_object* v___y_1654_; lean_object* v___y_1672_; uint8_t v___y_1673_; uint8_t v___y_1674_; lean_object* v___y_1675_; uint8_t v___y_1676_; lean_object* v___y_1677_; lean_object* v___y_1678_; lean_object* v___y_1679_; lean_object* v___y_1683_; uint8_t v___y_1684_; uint8_t v___y_1685_; lean_object* v___y_1686_; lean_object* v___y_1687_; lean_object* v___y_1688_; uint8_t v___y_1689_; uint8_t v___x_1694_; uint8_t v___y_1696_; lean_object* v___y_1697_; lean_object* v___y_1698_; lean_object* v___y_1699_; lean_object* v___y_1700_; uint8_t v___y_1701_; uint8_t v___y_1702_; uint8_t v___y_1704_; uint8_t v___x_1719_; 
v___x_1694_ = 2;
v___x_1719_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1603_, v___x_1694_);
if (v___x_1719_ == 0)
{
v___y_1704_ = v___x_1719_;
goto v___jp_1703_;
}
else
{
uint8_t v___x_1720_; 
lean_inc_ref(v_msgData_1602_);
v___x_1720_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1602_);
v___y_1704_ = v___x_1720_;
goto v___jp_1703_;
}
v___jp_1610_:
{
lean_object* v___x_1620_; lean_object* v_currNamespace_1621_; lean_object* v_openDecls_1622_; lean_object* v_env_1623_; lean_object* v_nextMacroScope_1624_; lean_object* v_ngen_1625_; lean_object* v_auxDeclNGen_1626_; lean_object* v_traceState_1627_; lean_object* v_cache_1628_; lean_object* v_messages_1629_; lean_object* v_infoState_1630_; lean_object* v_snapshotTasks_1631_; lean_object* v___x_1633_; uint8_t v_isShared_1634_; uint8_t v_isSharedCheck_1645_; 
v___x_1620_ = lean_st_ref_take(v___y_1619_);
v_currNamespace_1621_ = lean_ctor_get(v___y_1618_, 6);
v_openDecls_1622_ = lean_ctor_get(v___y_1618_, 7);
v_env_1623_ = lean_ctor_get(v___x_1620_, 0);
v_nextMacroScope_1624_ = lean_ctor_get(v___x_1620_, 1);
v_ngen_1625_ = lean_ctor_get(v___x_1620_, 2);
v_auxDeclNGen_1626_ = lean_ctor_get(v___x_1620_, 3);
v_traceState_1627_ = lean_ctor_get(v___x_1620_, 4);
v_cache_1628_ = lean_ctor_get(v___x_1620_, 5);
v_messages_1629_ = lean_ctor_get(v___x_1620_, 6);
v_infoState_1630_ = lean_ctor_get(v___x_1620_, 7);
v_snapshotTasks_1631_ = lean_ctor_get(v___x_1620_, 8);
v_isSharedCheck_1645_ = !lean_is_exclusive(v___x_1620_);
if (v_isSharedCheck_1645_ == 0)
{
v___x_1633_ = v___x_1620_;
v_isShared_1634_ = v_isSharedCheck_1645_;
goto v_resetjp_1632_;
}
else
{
lean_inc(v_snapshotTasks_1631_);
lean_inc(v_infoState_1630_);
lean_inc(v_messages_1629_);
lean_inc(v_cache_1628_);
lean_inc(v_traceState_1627_);
lean_inc(v_auxDeclNGen_1626_);
lean_inc(v_ngen_1625_);
lean_inc(v_nextMacroScope_1624_);
lean_inc(v_env_1623_);
lean_dec(v___x_1620_);
v___x_1633_ = lean_box(0);
v_isShared_1634_ = v_isSharedCheck_1645_;
goto v_resetjp_1632_;
}
v_resetjp_1632_:
{
lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1640_; 
lean_inc(v_openDecls_1622_);
lean_inc(v_currNamespace_1621_);
v___x_1635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1635_, 0, v_currNamespace_1621_);
lean_ctor_set(v___x_1635_, 1, v_openDecls_1622_);
v___x_1636_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1636_, 0, v___x_1635_);
lean_ctor_set(v___x_1636_, 1, v___y_1614_);
lean_inc_ref(v___y_1617_);
lean_inc_ref(v___y_1613_);
v___x_1637_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1637_, 0, v___y_1613_);
lean_ctor_set(v___x_1637_, 1, v___y_1615_);
lean_ctor_set(v___x_1637_, 2, v___y_1616_);
lean_ctor_set(v___x_1637_, 3, v___y_1617_);
lean_ctor_set(v___x_1637_, 4, v___x_1636_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*5, v___y_1611_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*5 + 1, v___y_1612_);
lean_ctor_set_uint8(v___x_1637_, sizeof(void*)*5 + 2, v_isSilent_1604_);
v___x_1638_ = l_Lean_MessageLog_add(v___x_1637_, v_messages_1629_);
if (v_isShared_1634_ == 0)
{
lean_ctor_set(v___x_1633_, 6, v___x_1638_);
v___x_1640_ = v___x_1633_;
goto v_reusejp_1639_;
}
else
{
lean_object* v_reuseFailAlloc_1644_; 
v_reuseFailAlloc_1644_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1644_, 0, v_env_1623_);
lean_ctor_set(v_reuseFailAlloc_1644_, 1, v_nextMacroScope_1624_);
lean_ctor_set(v_reuseFailAlloc_1644_, 2, v_ngen_1625_);
lean_ctor_set(v_reuseFailAlloc_1644_, 3, v_auxDeclNGen_1626_);
lean_ctor_set(v_reuseFailAlloc_1644_, 4, v_traceState_1627_);
lean_ctor_set(v_reuseFailAlloc_1644_, 5, v_cache_1628_);
lean_ctor_set(v_reuseFailAlloc_1644_, 6, v___x_1638_);
lean_ctor_set(v_reuseFailAlloc_1644_, 7, v_infoState_1630_);
lean_ctor_set(v_reuseFailAlloc_1644_, 8, v_snapshotTasks_1631_);
v___x_1640_ = v_reuseFailAlloc_1644_;
goto v_reusejp_1639_;
}
v_reusejp_1639_:
{
lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; 
v___x_1641_ = lean_st_ref_set(v___y_1619_, v___x_1640_);
v___x_1642_ = lean_box(0);
v___x_1643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1643_, 0, v___x_1642_);
return v___x_1643_;
}
}
}
v___jp_1646_:
{
lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v_a_1657_; lean_object* v___x_1659_; uint8_t v_isShared_1660_; uint8_t v_isSharedCheck_1670_; 
v___x_1655_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1602_);
v___x_1656_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6(v___x_1655_, v___y_1605_, v___y_1606_, v___y_1607_, v___y_1608_);
v_a_1657_ = lean_ctor_get(v___x_1656_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1656_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1659_ = v___x_1656_;
v_isShared_1660_ = v_isSharedCheck_1670_;
goto v_resetjp_1658_;
}
else
{
lean_inc(v_a_1657_);
lean_dec(v___x_1656_);
v___x_1659_ = lean_box(0);
v_isShared_1660_ = v_isSharedCheck_1670_;
goto v_resetjp_1658_;
}
v_resetjp_1658_:
{
lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; 
lean_inc_ref_n(v___y_1653_, 2);
v___x_1661_ = l_Lean_FileMap_toPosition(v___y_1653_, v___y_1652_);
lean_dec(v___y_1652_);
v___x_1662_ = l_Lean_FileMap_toPosition(v___y_1653_, v___y_1654_);
lean_dec(v___y_1654_);
v___x_1663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1663_, 0, v___x_1662_);
v___x_1664_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0));
if (v___y_1649_ == 0)
{
lean_del_object(v___x_1659_);
lean_dec_ref(v___y_1647_);
v___y_1611_ = v___y_1648_;
v___y_1612_ = v___y_1650_;
v___y_1613_ = v___y_1651_;
v___y_1614_ = v_a_1657_;
v___y_1615_ = v___x_1661_;
v___y_1616_ = v___x_1663_;
v___y_1617_ = v___x_1664_;
v___y_1618_ = v___y_1607_;
v___y_1619_ = v___y_1608_;
goto v___jp_1610_;
}
else
{
uint8_t v___x_1665_; 
lean_inc(v_a_1657_);
v___x_1665_ = l_Lean_MessageData_hasTag(v___y_1647_, v_a_1657_);
if (v___x_1665_ == 0)
{
lean_object* v___x_1666_; lean_object* v___x_1668_; 
lean_dec_ref_known(v___x_1663_, 1);
lean_dec_ref(v___x_1661_);
lean_dec(v_a_1657_);
v___x_1666_ = lean_box(0);
if (v_isShared_1660_ == 0)
{
lean_ctor_set(v___x_1659_, 0, v___x_1666_);
v___x_1668_ = v___x_1659_;
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
else
{
lean_del_object(v___x_1659_);
v___y_1611_ = v___y_1648_;
v___y_1612_ = v___y_1650_;
v___y_1613_ = v___y_1651_;
v___y_1614_ = v_a_1657_;
v___y_1615_ = v___x_1661_;
v___y_1616_ = v___x_1663_;
v___y_1617_ = v___x_1664_;
v___y_1618_ = v___y_1607_;
v___y_1619_ = v___y_1608_;
goto v___jp_1610_;
}
}
}
}
v___jp_1671_:
{
lean_object* v___x_1680_; 
v___x_1680_ = l_Lean_Syntax_getTailPos_x3f(v___y_1675_, v___y_1673_);
lean_dec(v___y_1675_);
if (lean_obj_tag(v___x_1680_) == 0)
{
lean_inc(v___y_1679_);
v___y_1647_ = v___y_1672_;
v___y_1648_ = v___y_1673_;
v___y_1649_ = v___y_1674_;
v___y_1650_ = v___y_1676_;
v___y_1651_ = v___y_1677_;
v___y_1652_ = v___y_1679_;
v___y_1653_ = v___y_1678_;
v___y_1654_ = v___y_1679_;
goto v___jp_1646_;
}
else
{
lean_object* v_val_1681_; 
v_val_1681_ = lean_ctor_get(v___x_1680_, 0);
lean_inc(v_val_1681_);
lean_dec_ref_known(v___x_1680_, 1);
v___y_1647_ = v___y_1672_;
v___y_1648_ = v___y_1673_;
v___y_1649_ = v___y_1674_;
v___y_1650_ = v___y_1676_;
v___y_1651_ = v___y_1677_;
v___y_1652_ = v___y_1679_;
v___y_1653_ = v___y_1678_;
v___y_1654_ = v_val_1681_;
goto v___jp_1646_;
}
}
v___jp_1682_:
{
lean_object* v_ref_1690_; lean_object* v___x_1691_; 
v_ref_1690_ = l_Lean_replaceRef(v_ref_1601_, v___y_1687_);
v___x_1691_ = l_Lean_Syntax_getPos_x3f(v_ref_1690_, v___y_1684_);
if (lean_obj_tag(v___x_1691_) == 0)
{
lean_object* v___x_1692_; 
v___x_1692_ = lean_unsigned_to_nat(0u);
v___y_1672_ = v___y_1683_;
v___y_1673_ = v___y_1684_;
v___y_1674_ = v___y_1685_;
v___y_1675_ = v_ref_1690_;
v___y_1676_ = v___y_1689_;
v___y_1677_ = v___y_1686_;
v___y_1678_ = v___y_1688_;
v___y_1679_ = v___x_1692_;
goto v___jp_1671_;
}
else
{
lean_object* v_val_1693_; 
v_val_1693_ = lean_ctor_get(v___x_1691_, 0);
lean_inc(v_val_1693_);
lean_dec_ref_known(v___x_1691_, 1);
v___y_1672_ = v___y_1683_;
v___y_1673_ = v___y_1684_;
v___y_1674_ = v___y_1685_;
v___y_1675_ = v_ref_1690_;
v___y_1676_ = v___y_1689_;
v___y_1677_ = v___y_1686_;
v___y_1678_ = v___y_1688_;
v___y_1679_ = v_val_1693_;
goto v___jp_1671_;
}
}
v___jp_1695_:
{
if (v___y_1702_ == 0)
{
v___y_1683_ = v___y_1700_;
v___y_1684_ = v___y_1701_;
v___y_1685_ = v___y_1696_;
v___y_1686_ = v___y_1698_;
v___y_1687_ = v___y_1697_;
v___y_1688_ = v___y_1699_;
v___y_1689_ = v_severity_1603_;
goto v___jp_1682_;
}
else
{
v___y_1683_ = v___y_1700_;
v___y_1684_ = v___y_1701_;
v___y_1685_ = v___y_1696_;
v___y_1686_ = v___y_1698_;
v___y_1687_ = v___y_1697_;
v___y_1688_ = v___y_1699_;
v___y_1689_ = v___x_1694_;
goto v___jp_1682_;
}
}
v___jp_1703_:
{
if (v___y_1704_ == 0)
{
lean_object* v_fileName_1705_; lean_object* v_fileMap_1706_; lean_object* v_options_1707_; lean_object* v_ref_1708_; uint8_t v_suppressElabErrors_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___f_1712_; uint8_t v___x_1713_; uint8_t v___x_1714_; 
v_fileName_1705_ = lean_ctor_get(v___y_1607_, 0);
v_fileMap_1706_ = lean_ctor_get(v___y_1607_, 1);
v_options_1707_ = lean_ctor_get(v___y_1607_, 2);
v_ref_1708_ = lean_ctor_get(v___y_1607_, 5);
v_suppressElabErrors_1709_ = lean_ctor_get_uint8(v___y_1607_, sizeof(void*)*14 + 1);
v___x_1710_ = lean_box(v___y_1704_);
v___x_1711_ = lean_box(v_suppressElabErrors_1709_);
v___f_1712_ = lean_alloc_closure((void*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1712_, 0, v___x_1710_);
lean_closure_set(v___f_1712_, 1, v___x_1711_);
v___x_1713_ = 1;
v___x_1714_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1603_, v___x_1713_);
if (v___x_1714_ == 0)
{
v___y_1696_ = v_suppressElabErrors_1709_;
v___y_1697_ = v_ref_1708_;
v___y_1698_ = v_fileName_1705_;
v___y_1699_ = v_fileMap_1706_;
v___y_1700_ = v___f_1712_;
v___y_1701_ = v___y_1704_;
v___y_1702_ = v___x_1714_;
goto v___jp_1695_;
}
else
{
lean_object* v___x_1715_; uint8_t v___x_1716_; 
v___x_1715_ = l_Lean_warningAsError;
v___x_1716_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7(v_options_1707_, v___x_1715_);
v___y_1696_ = v_suppressElabErrors_1709_;
v___y_1697_ = v_ref_1708_;
v___y_1698_ = v_fileName_1705_;
v___y_1699_ = v_fileMap_1706_;
v___y_1700_ = v___f_1712_;
v___y_1701_ = v___y_1704_;
v___y_1702_ = v___x_1716_;
goto v___jp_1695_;
}
}
else
{
lean_object* v___x_1717_; lean_object* v___x_1718_; 
lean_dec_ref(v_msgData_1602_);
v___x_1717_ = lean_box(0);
v___x_1718_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1718_, 0, v___x_1717_);
return v___x_1718_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_ref_1721_, lean_object* v_msgData_1722_, lean_object* v_severity_1723_, lean_object* v_isSilent_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_, lean_object* v___y_1729_){
_start:
{
uint8_t v_severity_boxed_1730_; uint8_t v_isSilent_boxed_1731_; lean_object* v_res_1732_; 
v_severity_boxed_1730_ = lean_unbox(v_severity_1723_);
v_isSilent_boxed_1731_ = lean_unbox(v_isSilent_1724_);
v_res_1732_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg(v_ref_1721_, v_msgData_1722_, v_severity_boxed_1730_, v_isSilent_boxed_1731_, v___y_1725_, v___y_1726_, v___y_1727_, v___y_1728_);
lean_dec(v___y_1728_);
lean_dec_ref(v___y_1727_);
lean_dec(v___y_1726_);
lean_dec_ref(v___y_1725_);
lean_dec(v_ref_1721_);
return v_res_1732_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1(lean_object* v_msgData_1733_, uint8_t v_severity_1734_, uint8_t v_isSilent_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_){
_start:
{
lean_object* v_ref_1743_; lean_object* v___x_1744_; 
v_ref_1743_ = lean_ctor_get(v___y_1740_, 5);
v___x_1744_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg(v_ref_1743_, v_msgData_1733_, v_severity_1734_, v_isSilent_1735_, v___y_1738_, v___y_1739_, v___y_1740_, v___y_1741_);
return v___x_1744_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1___boxed(lean_object* v_msgData_1745_, lean_object* v_severity_1746_, lean_object* v_isSilent_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_){
_start:
{
uint8_t v_severity_boxed_1755_; uint8_t v_isSilent_boxed_1756_; lean_object* v_res_1757_; 
v_severity_boxed_1755_ = lean_unbox(v_severity_1746_);
v_isSilent_boxed_1756_ = lean_unbox(v_isSilent_1747_);
v_res_1757_ = lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1(v_msgData_1745_, v_severity_boxed_1755_, v_isSilent_boxed_1756_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
return v_res_1757_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(lean_object* v_msgData_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_){
_start:
{
uint8_t v___x_1766_; uint8_t v___x_1767_; lean_object* v___x_1768_; 
v___x_1766_ = 1;
v___x_1767_ = 0;
v___x_1768_ = lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1(v_msgData_1758_, v___x_1766_, v___x_1767_, v___y_1759_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_);
return v___x_1768_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3___boxed(lean_object* v_msgData_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_){
_start:
{
lean_object* v_res_1777_; 
v_res_1777_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(v_msgData_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_, v___y_1774_, v___y_1775_);
lean_dec(v___y_1775_);
lean_dec_ref(v___y_1774_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
return v_res_1777_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(lean_object* v_msgData_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_){
_start:
{
uint8_t v___x_1786_; uint8_t v___x_1787_; lean_object* v___x_1788_; 
v___x_1786_ = 0;
v___x_1787_ = 0;
v___x_1788_ = lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1(v_msgData_1778_, v___x_1786_, v___x_1787_, v___y_1779_, v___y_1780_, v___y_1781_, v___y_1782_, v___y_1783_, v___y_1784_);
return v___x_1788_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1___boxed(lean_object* v_msgData_1789_, lean_object* v___y_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_){
_start:
{
lean_object* v_res_1797_; 
v_res_1797_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v_msgData_1789_, v___y_1790_, v___y_1791_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_);
lean_dec(v___y_1795_);
lean_dec_ref(v___y_1794_);
lean_dec(v___y_1793_);
lean_dec_ref(v___y_1792_);
lean_dec(v___y_1791_);
lean_dec_ref(v___y_1790_);
return v_res_1797_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1800_; lean_object* v___x_1801_; 
v___x_1800_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__0));
v___x_1801_ = l_Lean_MessageData_ofFormat(v___x_1800_);
return v___x_1801_;
}
}
static lean_object* _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1806_; lean_object* v___x_1807_; 
v___x_1806_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__4));
v___x_1807_ = l_Lean_MessageData_ofFormat(v___x_1806_);
return v___x_1807_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0(uint8_t v___x_1810_, lean_object* v_stx_1811_, lean_object* v___x_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_){
_start:
{
if (v___x_1810_ == 0)
{
lean_object* v___x_1820_; 
lean_dec_ref(v___x_1812_);
lean_dec(v_stx_1811_);
v___x_1820_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg();
return v___x_1820_;
}
else
{
lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; uint8_t v___x_1825_; 
v___x_1821_ = lean_unsigned_to_nat(1u);
v___x_1822_ = l_Lean_Syntax_getArg(v_stx_1811_, v___x_1821_);
v___x_1823_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__0));
v___x_1824_ = l_Lean_Name_mkStr2(v___x_1812_, v___x_1823_);
lean_inc(v___x_1822_);
v___x_1825_ = l_Lean_Syntax_isOfKind(v___x_1822_, v___x_1824_);
if (v___x_1825_ == 0)
{
lean_object* v___x_1826_; 
lean_dec(v___x_1824_);
lean_dec(v___x_1822_);
lean_dec(v_stx_1811_);
v___x_1826_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg();
return v___x_1826_;
}
else
{
lean_object* v___x_1827_; 
v___x_1827_ = l_Lean_PrettyPrinter_ppCategory(v___x_1824_, v___x_1822_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1827_) == 0)
{
lean_object* v_a_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; 
v_a_1828_ = lean_ctor_get(v___x_1827_, 0);
lean_inc(v_a_1828_);
lean_dec_ref_known(v___x_1827_, 1);
v___x_1829_ = lean_unsigned_to_nat(0u);
v___x_1830_ = l_Std_Format_defWidth;
v___x_1831_ = l_Std_Format_pretty(v_a_1828_, v___x_1830_, v___x_1829_, v___x_1829_);
v___x_1832_ = lean_unsigned_to_nat(6u);
v___x_1833_ = lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson(v___x_1831_, v___x_1832_, v___y_1817_, v___y_1818_);
lean_dec_ref(v___x_1831_);
if (lean_obj_tag(v___x_1833_) == 0)
{
lean_object* v_a_1834_; 
v_a_1834_ = lean_ctor_get(v___x_1833_, 0);
lean_inc(v_a_1834_);
lean_dec_ref_known(v___x_1833_, 1);
switch(lean_obj_tag(v_a_1834_))
{
case 0:
{
lean_object* v___x_1835_; lean_object* v___x_1836_; 
lean_dec(v_stx_1811_);
v___x_1835_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_1836_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v___x_1835_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
return v___x_1836_;
}
case 1:
{
lean_object* v_a_1837_; size_t v_sz_1838_; size_t v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; uint8_t v___x_1842_; 
v_a_1837_ = lean_ctor_get(v_a_1834_, 0);
lean_inc_ref(v_a_1837_);
lean_dec_ref_known(v_a_1834_, 1);
v_sz_1838_ = lean_array_size(v_a_1837_);
v___x_1839_ = ((size_t)0ULL);
v___x_1840_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleCmdImpl_spec__2(v_sz_1838_, v___x_1839_, v_a_1837_);
v___x_1841_ = lean_array_get_size(v___x_1840_);
v___x_1842_ = lean_nat_dec_eq(v___x_1841_, v___x_1829_);
if (v___x_1842_ == 0)
{
lean_object* v___x_1843_; lean_object* v___x_1844_; uint8_t v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; 
v___x_1843_ = lean_box(0);
v___x_1844_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__2));
v___x_1845_ = 4;
v___x_1846_ = l_Lean_MessageData_nil;
v___x_1847_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_1811_, v___x_1840_, v___x_1843_, v___x_1844_, v___x_1843_, v___x_1845_, v___x_1846_, v___y_1817_, v___y_1818_);
return v___x_1847_;
}
else
{
lean_object* v___x_1848_; lean_object* v___x_1849_; 
lean_dec_ref(v___x_1840_);
lean_dec(v_stx_1811_);
v___x_1848_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5);
v___x_1849_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(v___x_1848_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1849_) == 0)
{
lean_object* v___x_1850_; lean_object* v___x_1851_; 
lean_dec_ref_known(v___x_1849_, 1);
v___x_1850_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_1851_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v___x_1850_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
return v___x_1851_;
}
else
{
return v___x_1849_;
}
}
}
default: 
{
lean_object* v_error_1852_; lean_object* v_suggestions_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; 
v_error_1852_ = lean_ctor_get(v_a_1834_, 0);
lean_inc_ref(v_error_1852_);
v_suggestions_1853_ = lean_ctor_get(v_a_1834_, 1);
lean_inc(v_suggestions_1853_);
lean_dec_ref_known(v_a_1834_, 2);
v___x_1854_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__6));
v___x_1855_ = lean_string_append(v___x_1854_, v_error_1852_);
lean_dec_ref(v_error_1852_);
v___x_1856_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1856_, 0, v___x_1855_);
v___x_1857_ = l_Lean_MessageData_ofFormat(v___x_1856_);
v___x_1858_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(v___x_1857_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1858_) == 0)
{
lean_object* v___x_1859_; lean_object* v___x_1860_; 
lean_dec_ref_known(v___x_1858_, 1);
v___x_1859_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_1860_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v___x_1859_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_);
if (lean_obj_tag(v___x_1860_) == 0)
{
lean_object* v___x_1862_; uint8_t v_isShared_1863_; uint8_t v_isSharedCheck_1882_; 
v_isSharedCheck_1882_ = !lean_is_exclusive(v___x_1860_);
if (v_isSharedCheck_1882_ == 0)
{
lean_object* v_unused_1883_; 
v_unused_1883_ = lean_ctor_get(v___x_1860_, 0);
lean_dec(v_unused_1883_);
v___x_1862_ = v___x_1860_;
v_isShared_1863_ = v_isSharedCheck_1882_;
goto v_resetjp_1861_;
}
else
{
lean_dec(v___x_1860_);
v___x_1862_ = lean_box(0);
v_isShared_1863_ = v_isSharedCheck_1882_;
goto v_resetjp_1861_;
}
v_resetjp_1861_:
{
if (lean_obj_tag(v_suggestions_1853_) == 0)
{
lean_object* v___x_1864_; lean_object* v___x_1866_; 
lean_dec(v_stx_1811_);
v___x_1864_ = lean_box(0);
if (v_isShared_1863_ == 0)
{
lean_ctor_set(v___x_1862_, 0, v___x_1864_);
v___x_1866_ = v___x_1862_;
goto v_reusejp_1865_;
}
else
{
lean_object* v_reuseFailAlloc_1867_; 
v_reuseFailAlloc_1867_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1867_, 0, v___x_1864_);
v___x_1866_ = v_reuseFailAlloc_1867_;
goto v_reusejp_1865_;
}
v_reusejp_1865_:
{
return v___x_1866_;
}
}
else
{
lean_object* v_val_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; uint8_t v___x_1871_; 
v_val_1868_ = lean_ctor_get(v_suggestions_1853_, 0);
lean_inc(v_val_1868_);
lean_dec_ref_known(v_suggestions_1853_, 1);
v___x_1869_ = lean_box(0);
v___x_1870_ = lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleCmdImpl_spec__4(v_val_1868_, v___x_1869_);
v___x_1871_ = l_List_isEmpty___redArg(v___x_1870_);
if (v___x_1871_ == 0)
{
lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; uint8_t v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; 
lean_del_object(v___x_1862_);
v___x_1872_ = lean_array_mk(v___x_1870_);
v___x_1873_ = lean_box(0);
v___x_1874_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__7));
v___x_1875_ = 4;
v___x_1876_ = l_Lean_MessageData_nil;
v___x_1877_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_1811_, v___x_1872_, v___x_1873_, v___x_1874_, v___x_1873_, v___x_1875_, v___x_1876_, v___y_1817_, v___y_1818_);
return v___x_1877_;
}
else
{
lean_object* v___x_1878_; lean_object* v___x_1880_; 
lean_dec(v___x_1870_);
lean_dec(v_stx_1811_);
v___x_1878_ = lean_box(0);
if (v_isShared_1863_ == 0)
{
lean_ctor_set(v___x_1862_, 0, v___x_1878_);
v___x_1880_ = v___x_1862_;
goto v_reusejp_1879_;
}
else
{
lean_object* v_reuseFailAlloc_1881_; 
v_reuseFailAlloc_1881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1881_, 0, v___x_1878_);
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
}
else
{
lean_dec(v_suggestions_1853_);
lean_dec(v_stx_1811_);
return v___x_1860_;
}
}
else
{
lean_dec(v_suggestions_1853_);
lean_dec(v_stx_1811_);
return v___x_1858_;
}
}
}
}
else
{
lean_object* v_a_1884_; lean_object* v___x_1886_; uint8_t v_isShared_1887_; uint8_t v_isSharedCheck_1891_; 
lean_dec(v_stx_1811_);
v_a_1884_ = lean_ctor_get(v___x_1833_, 0);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1833_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1886_ = v___x_1833_;
v_isShared_1887_ = v_isSharedCheck_1891_;
goto v_resetjp_1885_;
}
else
{
lean_inc(v_a_1884_);
lean_dec(v___x_1833_);
v___x_1886_ = lean_box(0);
v_isShared_1887_ = v_isSharedCheck_1891_;
goto v_resetjp_1885_;
}
v_resetjp_1885_:
{
lean_object* v___x_1889_; 
if (v_isShared_1887_ == 0)
{
v___x_1889_ = v___x_1886_;
goto v_reusejp_1888_;
}
else
{
lean_object* v_reuseFailAlloc_1890_; 
v_reuseFailAlloc_1890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1890_, 0, v_a_1884_);
v___x_1889_ = v_reuseFailAlloc_1890_;
goto v_reusejp_1888_;
}
v_reusejp_1888_:
{
return v___x_1889_;
}
}
}
}
else
{
lean_object* v_a_1892_; lean_object* v___x_1894_; uint8_t v_isShared_1895_; uint8_t v_isSharedCheck_1899_; 
lean_dec(v_stx_1811_);
v_a_1892_ = lean_ctor_get(v___x_1827_, 0);
v_isSharedCheck_1899_ = !lean_is_exclusive(v___x_1827_);
if (v_isSharedCheck_1899_ == 0)
{
v___x_1894_ = v___x_1827_;
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
else
{
lean_inc(v_a_1892_);
lean_dec(v___x_1827_);
v___x_1894_ = lean_box(0);
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
v_resetjp_1893_:
{
lean_object* v___x_1897_; 
if (v_isShared_1895_ == 0)
{
v___x_1897_ = v___x_1894_;
goto v_reusejp_1896_;
}
else
{
lean_object* v_reuseFailAlloc_1898_; 
v_reuseFailAlloc_1898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1898_, 0, v_a_1892_);
v___x_1897_ = v_reuseFailAlloc_1898_;
goto v_reusejp_1896_;
}
v_reusejp_1896_:
{
return v___x_1897_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___boxed(lean_object* v___x_1900_, lean_object* v_stx_1901_, lean_object* v___x_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_){
_start:
{
uint8_t v___x_7160__boxed_1910_; lean_object* v_res_1911_; 
v___x_7160__boxed_1910_ = lean_unbox(v___x_1900_);
v_res_1911_ = lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0(v___x_7160__boxed_1910_, v_stx_1901_, v___x_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_);
lean_dec(v___y_1908_);
lean_dec_ref(v___y_1907_);
lean_dec(v___y_1906_);
lean_dec_ref(v___y_1905_);
lean_dec(v___y_1904_);
lean_dec_ref(v___y_1903_);
return v_res_1911_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl(lean_object* v_stx_1912_, lean_object* v_a_1913_, lean_object* v_a_1914_){
_start:
{
lean_object* v___x_1916_; lean_object* v___x_1917_; uint8_t v___x_1918_; lean_object* v___x_1919_; lean_object* v___y_1920_; lean_object* v___x_1921_; 
v___x_1916_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_turnstyle___closed__1));
v___x_1917_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogle__cmd___closed__1));
lean_inc(v_stx_1912_);
v___x_1918_ = l_Lean_Syntax_isOfKind(v_stx_1912_, v___x_1917_);
v___x_1919_ = lean_box(v___x_1918_);
v___y_1920_ = lean_alloc_closure((void*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___boxed), 10, 3);
lean_closure_set(v___y_1920_, 0, v___x_1919_);
lean_closure_set(v___y_1920_, 1, v_stx_1912_);
lean_closure_set(v___y_1920_, 2, v___x_1916_);
v___x_1921_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___y_1920_, v_a_1913_, v_a_1914_);
return v___x_1921_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___boxed(lean_object* v_stx_1922_, lean_object* v_a_1923_, lean_object* v_a_1924_, lean_object* v_a_1925_){
_start:
{
lean_object* v_res_1926_; 
v_res_1926_ = lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl(v_stx_1922_, v_a_1923_, v_a_1924_);
lean_dec(v_a_1924_);
lean_dec_ref(v_a_1923_);
return v_res_1926_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2(lean_object* v_ref_1927_, lean_object* v_msgData_1928_, uint8_t v_severity_1929_, uint8_t v_isSilent_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_){
_start:
{
lean_object* v___x_1938_; 
v___x_1938_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg(v_ref_1927_, v_msgData_1928_, v_severity_1929_, v_isSilent_1930_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_);
return v___x_1938_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_1939_, lean_object* v_msgData_1940_, lean_object* v_severity_1941_, lean_object* v_isSilent_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
uint8_t v_severity_boxed_1950_; uint8_t v_isSilent_boxed_1951_; lean_object* v_res_1952_; 
v_severity_boxed_1950_ = lean_unbox(v_severity_1941_);
v_isSilent_boxed_1951_ = lean_unbox(v_isSilent_1942_);
v_res_1952_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2(v_ref_1939_, v_msgData_1940_, v_severity_boxed_1950_, v_isSilent_boxed_1951_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_);
lean_dec(v___y_1948_);
lean_dec_ref(v___y_1947_);
lean_dec(v___y_1946_);
lean_dec_ref(v___y_1945_);
lean_dec(v___y_1944_);
lean_dec_ref(v___y_1943_);
lean_dec(v_ref_1939_);
return v_res_1952_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___redArg(){
_start:
{
lean_object* v___x_1963_; lean_object* v___x_1964_; 
v___x_1963_ = lean_box(0);
v___x_1964_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1964_, 0, v___x_1963_);
return v___x_1964_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___redArg___boxed(lean_object* v_a_1965_){
_start:
{
lean_object* v_res_1966_; 
v_res_1966_ = lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___redArg();
return v_res_1966_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl(lean_object* v_x_1967_, lean_object* v_a_1968_, lean_object* v_a_1969_){
_start:
{
lean_object* v___x_1971_; 
v___x_1971_ = lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___redArg();
return v___x_1971_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl___boxed(lean_object* v_x_1972_, lean_object* v_a_1973_, lean_object* v_a_1974_, lean_object* v_a_1975_){
_start:
{
lean_object* v_res_1976_; 
v_res_1976_ = lp_LeanSearchClient_LeanSearchClient_justLoogleCmdImpl(v_x_1972_, v_a_1973_, v_a_1974_);
lean_dec(v_a_1974_);
lean_dec_ref(v_a_1973_);
lean_dec(v_x_1972_);
return v_res_1976_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTermImpl_spec__1(size_t v_sz_1986_, size_t v_i_1987_, lean_object* v_bs_1988_){
_start:
{
uint8_t v___x_1989_; 
v___x_1989_ = lean_usize_dec_lt(v_i_1987_, v_sz_1986_);
if (v___x_1989_ == 0)
{
return v_bs_1988_;
}
else
{
lean_object* v_v_1990_; lean_object* v___x_1991_; lean_object* v_bs_x27_1992_; lean_object* v___x_1993_; size_t v___x_1994_; size_t v___x_1995_; lean_object* v___x_1996_; 
v_v_1990_ = lean_array_uget(v_bs_1988_, v_i_1987_);
v___x_1991_ = lean_unsigned_to_nat(0u);
v_bs_x27_1992_ = lean_array_uset(v_bs_1988_, v_i_1987_, v___x_1991_);
v___x_1993_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_toTermSuggestion(v_v_1990_);
v___x_1994_ = ((size_t)1ULL);
v___x_1995_ = lean_usize_add(v_i_1987_, v___x_1994_);
v___x_1996_ = lean_array_uset(v_bs_x27_1992_, v_i_1987_, v___x_1993_);
v_i_1987_ = v___x_1995_;
v_bs_1988_ = v___x_1996_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTermImpl_spec__1___boxed(lean_object* v_sz_1998_, lean_object* v_i_1999_, lean_object* v_bs_2000_){
_start:
{
size_t v_sz_boxed_2001_; size_t v_i_boxed_2002_; lean_object* v_res_2003_; 
v_sz_boxed_2001_ = lean_unbox_usize(v_sz_1998_);
lean_dec(v_sz_1998_);
v_i_boxed_2002_ = lean_unbox_usize(v_i_1999_);
lean_dec(v_i_1999_);
v_res_2003_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTermImpl_spec__1(v_sz_boxed_2001_, v_i_boxed_2002_, v_bs_2000_);
return v_res_2003_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_2005_; lean_object* v___x_2006_; 
v___x_2005_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0));
v___x_2006_ = lean_string_utf8_byte_size(v___x_2005_);
return v___x_2006_;
}
}
static uint8_t _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_2007_; lean_object* v___x_2008_; uint8_t v___x_2009_; 
v___x_2007_ = lean_unsigned_to_nat(0u);
v___x_2008_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1);
v___x_2009_ = lean_nat_dec_eq(v___x_2008_, v___x_2007_);
return v___x_2009_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; lean_object* v___x_2013_; 
v___x_2010_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__1);
v___x_2011_ = lean_unsigned_to_nat(0u);
v___x_2012_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0));
v___x_2013_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2013_, 0, v___x_2012_);
lean_ctor_set(v___x_2013_, 1, v___x_2011_);
lean_ctor_set(v___x_2013_, 2, v___x_2010_);
return v___x_2013_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_2014_; lean_object* v___x_2015_; 
v___x_2014_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3);
v___x_2015_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_2014_);
return v___x_2015_;
}
}
static lean_object* _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; 
v___x_2016_ = lean_unsigned_to_nat(0u);
v___x_2017_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__4, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__4_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__4);
v___x_2018_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__3);
v___x_2019_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_2019_, 0, v___x_2018_);
lean_ctor_set(v___x_2019_, 1, v___x_2017_);
lean_ctor_set(v___x_2019_, 2, v___x_2016_);
lean_ctor_set(v___x_2019_, 3, v___x_2016_);
return v___x_2019_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg(lean_object* v_s_2020_, lean_object* v_replacement_2021_){
_start:
{
lean_object* v___x_2022_; uint8_t v___x_2023_; 
v___x_2022_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0));
v___x_2023_ = lean_uint8_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__2, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__2_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__2);
if (v___x_2023_ == 0)
{
lean_object* v___x_2024_; lean_object* v___x_2025_; 
v___x_2024_ = lean_obj_once(&lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__5, &lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__5_once, _init_lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__5);
v___x_2025_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(v_s_2020_, v_replacement_2021_, v___x_2024_, v___x_2022_);
return v___x_2025_;
}
else
{
lean_object* v___x_2026_; lean_object* v___x_2027_; 
v___x_2026_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0___redArg___closed__6));
v___x_2027_ = lp_LeanSearchClient_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00LeanSearchClient_getLoogleQueryJson_spec__0_spec__0___redArg(v_s_2020_, v_replacement_2021_, v___x_2026_, v___x_2022_);
return v___x_2027_;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___boxed(lean_object* v_s_2028_, lean_object* v_replacement_2029_){
_start:
{
lean_object* v_res_2030_; 
v_res_2030_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg(v_s_2028_, v_replacement_2029_);
lean_dec_ref(v_replacement_2029_);
return v_res_2030_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2(lean_object* v_a_2033_, lean_object* v_a_2034_){
_start:
{
if (lean_obj_tag(v_a_2033_) == 0)
{
lean_object* v___x_2035_; 
v___x_2035_ = l_List_reverse___redArg(v_a_2034_);
return v___x_2035_;
}
else
{
lean_object* v_head_2036_; lean_object* v_tail_2037_; lean_object* v___x_2039_; uint8_t v_isShared_2040_; uint8_t v_isSharedCheck_2057_; 
v_head_2036_ = lean_ctor_get(v_a_2033_, 0);
v_tail_2037_ = lean_ctor_get(v_a_2033_, 1);
v_isSharedCheck_2057_ = !lean_is_exclusive(v_a_2033_);
if (v_isSharedCheck_2057_ == 0)
{
v___x_2039_ = v_a_2033_;
v_isShared_2040_ = v_isSharedCheck_2057_;
goto v_resetjp_2038_;
}
else
{
lean_inc(v_tail_2037_);
lean_inc(v_head_2036_);
lean_dec(v_a_2033_);
v___x_2039_ = lean_box(0);
v_isShared_2040_ = v_isSharedCheck_2057_;
goto v_resetjp_2038_;
}
v_resetjp_2038_:
{
lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2054_; 
v___x_2041_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0));
v___x_2042_ = ((lean_object*)(lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__0));
v___x_2043_ = lean_unsigned_to_nat(0u);
v___x_2044_ = lean_string_utf8_byte_size(v_head_2036_);
v___x_2045_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2045_, 0, v_head_2036_);
lean_ctor_set(v___x_2045_, 1, v___x_2043_);
lean_ctor_set(v___x_2045_, 2, v___x_2044_);
v___x_2046_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg(v___x_2045_, v___x_2042_);
v___x_2047_ = ((lean_object*)(lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__1));
v___x_2048_ = lean_string_append(v___x_2047_, v___x_2046_);
lean_dec_ref(v___x_2046_);
v___x_2049_ = lean_string_append(v___x_2048_, v___x_2041_);
v___x_2050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2050_, 0, v___x_2049_);
v___x_2051_ = lean_box(0);
v___x_2052_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2052_, 0, v___x_2050_);
lean_ctor_set(v___x_2052_, 1, v___x_2051_);
lean_ctor_set(v___x_2052_, 2, v___x_2051_);
lean_ctor_set(v___x_2052_, 3, v___x_2051_);
lean_ctor_set(v___x_2052_, 4, v___x_2051_);
lean_ctor_set(v___x_2052_, 5, v___x_2051_);
if (v_isShared_2040_ == 0)
{
lean_ctor_set(v___x_2039_, 1, v_a_2034_);
lean_ctor_set(v___x_2039_, 0, v___x_2052_);
v___x_2054_ = v___x_2039_;
goto v_reusejp_2053_;
}
else
{
lean_object* v_reuseFailAlloc_2056_; 
v_reuseFailAlloc_2056_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2056_, 0, v___x_2052_);
lean_ctor_set(v_reuseFailAlloc_2056_, 1, v_a_2034_);
v___x_2054_ = v_reuseFailAlloc_2056_;
goto v_reusejp_2053_;
}
v_reusejp_2053_:
{
v_a_2033_ = v_tail_2037_;
v_a_2034_ = v___x_2054_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTermImpl(lean_object* v_stx_2058_, lean_object* v_expectedType_x3f_2059_, lean_object* v_a_2060_, lean_object* v_a_2061_, lean_object* v_a_2062_, lean_object* v_a_2063_, lean_object* v_a_2064_, lean_object* v_a_2065_){
_start:
{
lean_object* v___x_2067_; uint8_t v___x_2068_; 
v___x_2067_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogle__term___closed__1));
lean_inc(v_stx_2058_);
v___x_2068_ = l_Lean_Syntax_isOfKind(v_stx_2058_, v___x_2067_);
if (v___x_2068_ == 0)
{
lean_object* v___x_2069_; 
lean_dec(v_expectedType_x3f_2059_);
lean_dec(v_stx_2058_);
v___x_2069_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg();
return v___x_2069_;
}
else
{
lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; 
v___x_2070_ = lean_unsigned_to_nat(1u);
v___x_2071_ = l_Lean_Syntax_getArg(v_stx_2058_, v___x_2070_);
v___x_2072_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1));
v___x_2073_ = l_Lean_PrettyPrinter_ppCategory(v___x_2072_, v___x_2071_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2073_) == 0)
{
lean_object* v_a_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; 
v_a_2074_ = lean_ctor_get(v___x_2073_, 0);
lean_inc(v_a_2074_);
lean_dec_ref_known(v___x_2073_, 1);
v___x_2075_ = lean_unsigned_to_nat(0u);
v___x_2076_ = l_Std_Format_defWidth;
v___x_2077_ = l_Std_Format_pretty(v_a_2074_, v___x_2076_, v___x_2075_, v___x_2075_);
v___x_2078_ = lean_unsigned_to_nat(6u);
v___x_2079_ = lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson(v___x_2077_, v___x_2078_, v_a_2064_, v_a_2065_);
lean_dec_ref(v___x_2077_);
if (lean_obj_tag(v___x_2079_) == 0)
{
lean_object* v_a_2080_; 
v_a_2080_ = lean_ctor_get(v___x_2079_, 0);
lean_inc(v_a_2080_);
lean_dec_ref_known(v___x_2079_, 1);
switch(lean_obj_tag(v_a_2080_))
{
case 0:
{
lean_object* v___x_2081_; lean_object* v___x_2082_; 
lean_dec(v_stx_2058_);
v___x_2081_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_2082_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v___x_2081_, v_a_2060_, v_a_2061_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2082_) == 0)
{
lean_object* v___x_2083_; 
lean_dec_ref_known(v___x_2082_, 1);
v___x_2083_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_2059_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
return v___x_2083_;
}
else
{
lean_object* v_a_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2091_; 
lean_dec(v_expectedType_x3f_2059_);
v_a_2084_ = lean_ctor_get(v___x_2082_, 0);
v_isSharedCheck_2091_ = !lean_is_exclusive(v___x_2082_);
if (v_isSharedCheck_2091_ == 0)
{
v___x_2086_ = v___x_2082_;
v_isShared_2087_ = v_isSharedCheck_2091_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_a_2084_);
lean_dec(v___x_2082_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2091_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v___x_2089_; 
if (v_isShared_2087_ == 0)
{
v___x_2089_ = v___x_2086_;
goto v_reusejp_2088_;
}
else
{
lean_object* v_reuseFailAlloc_2090_; 
v_reuseFailAlloc_2090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2090_, 0, v_a_2084_);
v___x_2089_ = v_reuseFailAlloc_2090_;
goto v_reusejp_2088_;
}
v_reusejp_2088_:
{
return v___x_2089_;
}
}
}
}
case 1:
{
lean_object* v_a_2092_; size_t v_sz_2093_; size_t v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; uint8_t v___x_2097_; 
v_a_2092_ = lean_ctor_get(v_a_2080_, 0);
lean_inc_ref(v_a_2092_);
lean_dec_ref_known(v_a_2080_, 1);
v_sz_2093_ = lean_array_size(v_a_2092_);
v___x_2094_ = ((size_t)0ULL);
v___x_2095_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTermImpl_spec__1(v_sz_2093_, v___x_2094_, v_a_2092_);
v___x_2096_ = lean_array_get_size(v___x_2095_);
v___x_2097_ = lean_nat_dec_eq(v___x_2096_, v___x_2075_);
if (v___x_2097_ == 0)
{
lean_object* v___x_2098_; lean_object* v___x_2099_; uint8_t v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; 
v___x_2098_ = lean_box(0);
v___x_2099_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__2));
v___x_2100_ = 4;
v___x_2101_ = l_Lean_MessageData_nil;
v___x_2102_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2058_, v___x_2095_, v___x_2098_, v___x_2099_, v___x_2098_, v___x_2100_, v___x_2101_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2102_) == 0)
{
lean_object* v___x_2103_; 
lean_dec_ref_known(v___x_2102_, 1);
v___x_2103_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_2059_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
return v___x_2103_;
}
else
{
lean_object* v_a_2104_; lean_object* v___x_2106_; uint8_t v_isShared_2107_; uint8_t v_isSharedCheck_2111_; 
lean_dec(v_expectedType_x3f_2059_);
v_a_2104_ = lean_ctor_get(v___x_2102_, 0);
v_isSharedCheck_2111_ = !lean_is_exclusive(v___x_2102_);
if (v_isSharedCheck_2111_ == 0)
{
v___x_2106_ = v___x_2102_;
v_isShared_2107_ = v_isSharedCheck_2111_;
goto v_resetjp_2105_;
}
else
{
lean_inc(v_a_2104_);
lean_dec(v___x_2102_);
v___x_2106_ = lean_box(0);
v_isShared_2107_ = v_isSharedCheck_2111_;
goto v_resetjp_2105_;
}
v_resetjp_2105_:
{
lean_object* v___x_2109_; 
if (v_isShared_2107_ == 0)
{
v___x_2109_ = v___x_2106_;
goto v_reusejp_2108_;
}
else
{
lean_object* v_reuseFailAlloc_2110_; 
v_reuseFailAlloc_2110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2110_, 0, v_a_2104_);
v___x_2109_ = v_reuseFailAlloc_2110_;
goto v_reusejp_2108_;
}
v_reusejp_2108_:
{
return v___x_2109_;
}
}
}
}
else
{
lean_object* v___x_2112_; lean_object* v___x_2113_; 
lean_dec_ref(v___x_2095_);
lean_dec(v_stx_2058_);
v___x_2112_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__5);
v___x_2113_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(v___x_2112_, v_a_2060_, v_a_2061_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2113_) == 0)
{
lean_object* v___x_2114_; lean_object* v___x_2115_; 
lean_dec_ref_known(v___x_2113_, 1);
v___x_2114_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_2115_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v___x_2114_, v_a_2060_, v_a_2061_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2115_) == 0)
{
lean_object* v___x_2116_; 
lean_dec_ref_known(v___x_2115_, 1);
v___x_2116_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_2059_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
return v___x_2116_;
}
else
{
lean_object* v_a_2117_; lean_object* v___x_2119_; uint8_t v_isShared_2120_; uint8_t v_isSharedCheck_2124_; 
lean_dec(v_expectedType_x3f_2059_);
v_a_2117_ = lean_ctor_get(v___x_2115_, 0);
v_isSharedCheck_2124_ = !lean_is_exclusive(v___x_2115_);
if (v_isSharedCheck_2124_ == 0)
{
v___x_2119_ = v___x_2115_;
v_isShared_2120_ = v_isSharedCheck_2124_;
goto v_resetjp_2118_;
}
else
{
lean_inc(v_a_2117_);
lean_dec(v___x_2115_);
v___x_2119_ = lean_box(0);
v_isShared_2120_ = v_isSharedCheck_2124_;
goto v_resetjp_2118_;
}
v_resetjp_2118_:
{
lean_object* v___x_2122_; 
if (v_isShared_2120_ == 0)
{
v___x_2122_ = v___x_2119_;
goto v_reusejp_2121_;
}
else
{
lean_object* v_reuseFailAlloc_2123_; 
v_reuseFailAlloc_2123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2123_, 0, v_a_2117_);
v___x_2122_ = v_reuseFailAlloc_2123_;
goto v_reusejp_2121_;
}
v_reusejp_2121_:
{
return v___x_2122_;
}
}
}
}
else
{
lean_object* v_a_2125_; lean_object* v___x_2127_; uint8_t v_isShared_2128_; uint8_t v_isSharedCheck_2132_; 
lean_dec(v_expectedType_x3f_2059_);
v_a_2125_ = lean_ctor_get(v___x_2113_, 0);
v_isSharedCheck_2132_ = !lean_is_exclusive(v___x_2113_);
if (v_isSharedCheck_2132_ == 0)
{
v___x_2127_ = v___x_2113_;
v_isShared_2128_ = v_isSharedCheck_2132_;
goto v_resetjp_2126_;
}
else
{
lean_inc(v_a_2125_);
lean_dec(v___x_2113_);
v___x_2127_ = lean_box(0);
v_isShared_2128_ = v_isSharedCheck_2132_;
goto v_resetjp_2126_;
}
v_resetjp_2126_:
{
lean_object* v___x_2130_; 
if (v_isShared_2128_ == 0)
{
v___x_2130_ = v___x_2127_;
goto v_reusejp_2129_;
}
else
{
lean_object* v_reuseFailAlloc_2131_; 
v_reuseFailAlloc_2131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2131_, 0, v_a_2125_);
v___x_2130_ = v_reuseFailAlloc_2131_;
goto v_reusejp_2129_;
}
v_reusejp_2129_:
{
return v___x_2130_;
}
}
}
}
}
default: 
{
lean_object* v_error_2133_; lean_object* v_suggestions_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; 
v_error_2133_ = lean_ctor_get(v_a_2080_, 0);
lean_inc_ref(v_error_2133_);
v_suggestions_2134_ = lean_ctor_get(v_a_2080_, 1);
lean_inc(v_suggestions_2134_);
lean_dec_ref_known(v_a_2080_, 2);
v___x_2135_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__6));
v___x_2136_ = lean_string_append(v___x_2135_, v_error_2133_);
lean_dec_ref(v_error_2133_);
v___x_2137_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2137_, 0, v___x_2136_);
v___x_2138_ = l_Lean_MessageData_ofFormat(v___x_2137_);
v___x_2139_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleCmdImpl_spec__3(v___x_2138_, v_a_2060_, v_a_2061_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2139_) == 0)
{
lean_object* v___x_2140_; lean_object* v___x_2141_; 
lean_dec_ref_known(v___x_2139_, 1);
v___x_2140_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_2141_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1(v___x_2140_, v_a_2060_, v_a_2061_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2141_) == 0)
{
lean_dec_ref_known(v___x_2141_, 1);
if (lean_obj_tag(v_suggestions_2134_) == 0)
{
lean_object* v___x_2142_; 
lean_dec(v_stx_2058_);
v___x_2142_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_2059_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
return v___x_2142_;
}
else
{
lean_object* v_val_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; uint8_t v___x_2146_; 
v_val_2143_ = lean_ctor_get(v_suggestions_2134_, 0);
lean_inc(v_val_2143_);
lean_dec_ref_known(v_suggestions_2134_, 1);
v___x_2144_ = lean_box(0);
v___x_2145_ = lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2(v_val_2143_, v___x_2144_);
v___x_2146_ = l_List_isEmpty___redArg(v___x_2145_);
if (v___x_2146_ == 0)
{
lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; uint8_t v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; 
v___x_2147_ = lean_array_mk(v___x_2145_);
v___x_2148_ = lean_box(0);
v___x_2149_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__7));
v___x_2150_ = 4;
v___x_2151_ = l_Lean_MessageData_nil;
v___x_2152_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2058_, v___x_2147_, v___x_2148_, v___x_2149_, v___x_2148_, v___x_2150_, v___x_2151_, v_a_2064_, v_a_2065_);
if (lean_obj_tag(v___x_2152_) == 0)
{
lean_object* v___x_2153_; 
lean_dec_ref_known(v___x_2152_, 1);
v___x_2153_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_2059_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
return v___x_2153_;
}
else
{
lean_object* v_a_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2161_; 
lean_dec(v_expectedType_x3f_2059_);
v_a_2154_ = lean_ctor_get(v___x_2152_, 0);
v_isSharedCheck_2161_ = !lean_is_exclusive(v___x_2152_);
if (v_isSharedCheck_2161_ == 0)
{
v___x_2156_ = v___x_2152_;
v_isShared_2157_ = v_isSharedCheck_2161_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_a_2154_);
lean_dec(v___x_2152_);
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
else
{
lean_object* v___x_2162_; 
lean_dec(v___x_2145_);
lean_dec(v_stx_2058_);
v___x_2162_ = lp_LeanSearchClient_LeanSearchClient_defaultTerm(v_expectedType_x3f_2059_, v_a_2062_, v_a_2063_, v_a_2064_, v_a_2065_);
return v___x_2162_;
}
}
}
else
{
lean_object* v_a_2163_; lean_object* v___x_2165_; uint8_t v_isShared_2166_; uint8_t v_isSharedCheck_2170_; 
lean_dec(v_suggestions_2134_);
lean_dec(v_expectedType_x3f_2059_);
lean_dec(v_stx_2058_);
v_a_2163_ = lean_ctor_get(v___x_2141_, 0);
v_isSharedCheck_2170_ = !lean_is_exclusive(v___x_2141_);
if (v_isSharedCheck_2170_ == 0)
{
v___x_2165_ = v___x_2141_;
v_isShared_2166_ = v_isSharedCheck_2170_;
goto v_resetjp_2164_;
}
else
{
lean_inc(v_a_2163_);
lean_dec(v___x_2141_);
v___x_2165_ = lean_box(0);
v_isShared_2166_ = v_isSharedCheck_2170_;
goto v_resetjp_2164_;
}
v_resetjp_2164_:
{
lean_object* v___x_2168_; 
if (v_isShared_2166_ == 0)
{
v___x_2168_ = v___x_2165_;
goto v_reusejp_2167_;
}
else
{
lean_object* v_reuseFailAlloc_2169_; 
v_reuseFailAlloc_2169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2169_, 0, v_a_2163_);
v___x_2168_ = v_reuseFailAlloc_2169_;
goto v_reusejp_2167_;
}
v_reusejp_2167_:
{
return v___x_2168_;
}
}
}
}
else
{
lean_object* v_a_2171_; lean_object* v___x_2173_; uint8_t v_isShared_2174_; uint8_t v_isSharedCheck_2178_; 
lean_dec(v_suggestions_2134_);
lean_dec(v_expectedType_x3f_2059_);
lean_dec(v_stx_2058_);
v_a_2171_ = lean_ctor_get(v___x_2139_, 0);
v_isSharedCheck_2178_ = !lean_is_exclusive(v___x_2139_);
if (v_isSharedCheck_2178_ == 0)
{
v___x_2173_ = v___x_2139_;
v_isShared_2174_ = v_isSharedCheck_2178_;
goto v_resetjp_2172_;
}
else
{
lean_inc(v_a_2171_);
lean_dec(v___x_2139_);
v___x_2173_ = lean_box(0);
v_isShared_2174_ = v_isSharedCheck_2178_;
goto v_resetjp_2172_;
}
v_resetjp_2172_:
{
lean_object* v___x_2176_; 
if (v_isShared_2174_ == 0)
{
v___x_2176_ = v___x_2173_;
goto v_reusejp_2175_;
}
else
{
lean_object* v_reuseFailAlloc_2177_; 
v_reuseFailAlloc_2177_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2177_, 0, v_a_2171_);
v___x_2176_ = v_reuseFailAlloc_2177_;
goto v_reusejp_2175_;
}
v_reusejp_2175_:
{
return v___x_2176_;
}
}
}
}
}
}
else
{
lean_object* v_a_2179_; lean_object* v___x_2181_; uint8_t v_isShared_2182_; uint8_t v_isSharedCheck_2186_; 
lean_dec(v_expectedType_x3f_2059_);
lean_dec(v_stx_2058_);
v_a_2179_ = lean_ctor_get(v___x_2079_, 0);
v_isSharedCheck_2186_ = !lean_is_exclusive(v___x_2079_);
if (v_isSharedCheck_2186_ == 0)
{
v___x_2181_ = v___x_2079_;
v_isShared_2182_ = v_isSharedCheck_2186_;
goto v_resetjp_2180_;
}
else
{
lean_inc(v_a_2179_);
lean_dec(v___x_2079_);
v___x_2181_ = lean_box(0);
v_isShared_2182_ = v_isSharedCheck_2186_;
goto v_resetjp_2180_;
}
v_resetjp_2180_:
{
lean_object* v___x_2184_; 
if (v_isShared_2182_ == 0)
{
v___x_2184_ = v___x_2181_;
goto v_reusejp_2183_;
}
else
{
lean_object* v_reuseFailAlloc_2185_; 
v_reuseFailAlloc_2185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2185_, 0, v_a_2179_);
v___x_2184_ = v_reuseFailAlloc_2185_;
goto v_reusejp_2183_;
}
v_reusejp_2183_:
{
return v___x_2184_;
}
}
}
}
else
{
lean_object* v_a_2187_; lean_object* v___x_2189_; uint8_t v_isShared_2190_; uint8_t v_isSharedCheck_2194_; 
lean_dec(v_expectedType_x3f_2059_);
lean_dec(v_stx_2058_);
v_a_2187_ = lean_ctor_get(v___x_2073_, 0);
v_isSharedCheck_2194_ = !lean_is_exclusive(v___x_2073_);
if (v_isSharedCheck_2194_ == 0)
{
v___x_2189_ = v___x_2073_;
v_isShared_2190_ = v_isSharedCheck_2194_;
goto v_resetjp_2188_;
}
else
{
lean_inc(v_a_2187_);
lean_dec(v___x_2073_);
v___x_2189_ = lean_box(0);
v_isShared_2190_ = v_isSharedCheck_2194_;
goto v_resetjp_2188_;
}
v_resetjp_2188_:
{
lean_object* v___x_2192_; 
if (v_isShared_2190_ == 0)
{
v___x_2192_ = v___x_2189_;
goto v_reusejp_2191_;
}
else
{
lean_object* v_reuseFailAlloc_2193_; 
v_reuseFailAlloc_2193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2193_, 0, v_a_2187_);
v___x_2192_ = v_reuseFailAlloc_2193_;
goto v_reusejp_2191_;
}
v_reusejp_2191_:
{
return v___x_2192_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTermImpl___boxed(lean_object* v_stx_2195_, lean_object* v_expectedType_x3f_2196_, lean_object* v_a_2197_, lean_object* v_a_2198_, lean_object* v_a_2199_, lean_object* v_a_2200_, lean_object* v_a_2201_, lean_object* v_a_2202_, lean_object* v_a_2203_){
_start:
{
lean_object* v_res_2204_; 
v_res_2204_ = lp_LeanSearchClient_LeanSearchClient_loogleTermImpl(v_stx_2195_, v_expectedType_x3f_2196_, v_a_2197_, v_a_2198_, v_a_2199_, v_a_2200_, v_a_2201_, v_a_2202_);
lean_dec(v_a_2202_);
lean_dec_ref(v_a_2201_);
lean_dec(v_a_2200_);
lean_dec_ref(v_a_2199_);
lean_dec(v_a_2198_);
lean_dec_ref(v_a_2197_);
return v_res_2204_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0(lean_object* v_s_2205_, lean_object* v_pattern_2206_, lean_object* v_replacement_2207_){
_start:
{
lean_object* v___x_2208_; 
v___x_2208_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg(v_s_2205_, v_replacement_2207_);
return v___x_2208_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___boxed(lean_object* v_s_2209_, lean_object* v_pattern_2210_, lean_object* v_replacement_2211_){
_start:
{
lean_object* v_res_2212_; 
v_res_2212_ = lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0(v_s_2209_, v_pattern_2210_, v_replacement_2211_);
lean_dec_ref(v_replacement_2211_);
lean_dec_ref(v_pattern_2210_);
return v_res_2212_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg(){
_start:
{
lean_object* v___x_2251_; lean_object* v___x_2252_; 
v___x_2251_ = lean_obj_once(&lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0, &lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0_once, _init_lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleCmdImpl_spec__0___redArg___closed__0);
v___x_2252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2252_, 0, v___x_2251_);
return v___x_2252_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg___boxed(lean_object* v___y_2253_){
_start:
{
lean_object* v_res_2254_; 
v_res_2254_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg();
return v_res_2254_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0(lean_object* v_00_u03b1_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_){
_start:
{
lean_object* v___x_2265_; 
v___x_2265_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg();
return v___x_2265_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___boxed(lean_object* v_00_u03b1_2266_, lean_object* v___y_2267_, lean_object* v___y_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_){
_start:
{
lean_object* v_res_2276_; 
v_res_2276_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0(v_00_u03b1_2266_, v___y_2267_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_, v___y_2274_);
lean_dec(v___y_2274_);
lean_dec_ref(v___y_2273_);
lean_dec(v___y_2272_);
lean_dec_ref(v___y_2271_);
lean_dec(v___y_2270_);
lean_dec_ref(v___y_2269_);
lean_dec(v___y_2268_);
lean_dec_ref(v___y_2267_);
return v_res_2276_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTacticImpl_spec__6(lean_object* v_a_2277_, lean_object* v_a_2278_){
_start:
{
if (lean_obj_tag(v_a_2277_) == 0)
{
lean_object* v___x_2279_; 
v___x_2279_ = l_List_reverse___redArg(v_a_2278_);
return v___x_2279_;
}
else
{
lean_object* v_head_2280_; lean_object* v_tail_2281_; lean_object* v___x_2283_; uint8_t v_isShared_2284_; uint8_t v_isSharedCheck_2296_; 
v_head_2280_ = lean_ctor_get(v_a_2277_, 0);
v_tail_2281_ = lean_ctor_get(v_a_2277_, 1);
v_isSharedCheck_2296_ = !lean_is_exclusive(v_a_2277_);
if (v_isSharedCheck_2296_ == 0)
{
v___x_2283_ = v_a_2277_;
v_isShared_2284_ = v_isSharedCheck_2296_;
goto v_resetjp_2282_;
}
else
{
lean_inc(v_tail_2281_);
lean_inc(v_head_2280_);
lean_dec(v_a_2277_);
v___x_2283_ = lean_box(0);
v_isShared_2284_ = v_isSharedCheck_2296_;
goto v_resetjp_2282_;
}
v_resetjp_2282_:
{
lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2293_; 
v___x_2285_ = ((lean_object*)(lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTermImpl_spec__2___closed__1));
v___x_2286_ = lean_string_append(v___x_2285_, v_head_2280_);
lean_dec(v_head_2280_);
v___x_2287_ = ((lean_object*)(lp_LeanSearchClient_String_Slice_replace___at___00LeanSearchClient_loogleTermImpl_spec__0___redArg___closed__0));
v___x_2288_ = lean_string_append(v___x_2286_, v___x_2287_);
v___x_2289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2289_, 0, v___x_2288_);
v___x_2290_ = lean_box(0);
v___x_2291_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2291_, 0, v___x_2289_);
lean_ctor_set(v___x_2291_, 1, v___x_2290_);
lean_ctor_set(v___x_2291_, 2, v___x_2290_);
lean_ctor_set(v___x_2291_, 3, v___x_2290_);
lean_ctor_set(v___x_2291_, 4, v___x_2290_);
lean_ctor_set(v___x_2291_, 5, v___x_2290_);
if (v_isShared_2284_ == 0)
{
lean_ctor_set(v___x_2283_, 1, v_a_2278_);
lean_ctor_set(v___x_2283_, 0, v___x_2291_);
v___x_2293_ = v___x_2283_;
goto v_reusejp_2292_;
}
else
{
lean_object* v_reuseFailAlloc_2295_; 
v_reuseFailAlloc_2295_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2295_, 0, v___x_2291_);
lean_ctor_set(v_reuseFailAlloc_2295_, 1, v_a_2278_);
v___x_2293_ = v_reuseFailAlloc_2295_;
goto v_reusejp_2292_;
}
v_reusejp_2292_:
{
v_a_2277_ = v_tail_2281_;
v_a_2278_ = v___x_2293_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg(lean_object* v_ref_2297_, lean_object* v_msgData_2298_, uint8_t v_severity_2299_, uint8_t v_isSilent_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_, lean_object* v___y_2304_){
_start:
{
lean_object* v___y_2307_; lean_object* v___y_2308_; uint8_t v___y_2309_; lean_object* v___y_2310_; uint8_t v___y_2311_; lean_object* v___y_2312_; lean_object* v___y_2313_; lean_object* v___y_2314_; lean_object* v___y_2315_; lean_object* v___y_2343_; lean_object* v___y_2344_; lean_object* v___y_2345_; uint8_t v___y_2346_; uint8_t v___y_2347_; lean_object* v___y_2348_; uint8_t v___y_2349_; lean_object* v___y_2350_; lean_object* v___y_2368_; lean_object* v___y_2369_; lean_object* v___y_2370_; uint8_t v___y_2371_; uint8_t v___y_2372_; lean_object* v___y_2373_; uint8_t v___y_2374_; lean_object* v___y_2375_; lean_object* v___y_2379_; lean_object* v___y_2380_; uint8_t v___y_2381_; uint8_t v___y_2382_; lean_object* v___y_2383_; lean_object* v___y_2384_; uint8_t v___y_2385_; uint8_t v___x_2390_; lean_object* v___y_2392_; uint8_t v___y_2393_; lean_object* v___y_2394_; lean_object* v___y_2395_; lean_object* v___y_2396_; uint8_t v___y_2397_; uint8_t v___y_2398_; uint8_t v___y_2400_; uint8_t v___x_2415_; 
v___x_2390_ = 2;
v___x_2415_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2299_, v___x_2390_);
if (v___x_2415_ == 0)
{
v___y_2400_ = v___x_2415_;
goto v___jp_2399_;
}
else
{
uint8_t v___x_2416_; 
lean_inc_ref(v_msgData_2298_);
v___x_2416_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2298_);
v___y_2400_ = v___x_2416_;
goto v___jp_2399_;
}
v___jp_2306_:
{
lean_object* v___x_2316_; lean_object* v_currNamespace_2317_; lean_object* v_openDecls_2318_; lean_object* v_env_2319_; lean_object* v_nextMacroScope_2320_; lean_object* v_ngen_2321_; lean_object* v_auxDeclNGen_2322_; lean_object* v_traceState_2323_; lean_object* v_cache_2324_; lean_object* v_messages_2325_; lean_object* v_infoState_2326_; lean_object* v_snapshotTasks_2327_; lean_object* v___x_2329_; uint8_t v_isShared_2330_; uint8_t v_isSharedCheck_2341_; 
v___x_2316_ = lean_st_ref_take(v___y_2315_);
v_currNamespace_2317_ = lean_ctor_get(v___y_2314_, 6);
v_openDecls_2318_ = lean_ctor_get(v___y_2314_, 7);
v_env_2319_ = lean_ctor_get(v___x_2316_, 0);
v_nextMacroScope_2320_ = lean_ctor_get(v___x_2316_, 1);
v_ngen_2321_ = lean_ctor_get(v___x_2316_, 2);
v_auxDeclNGen_2322_ = lean_ctor_get(v___x_2316_, 3);
v_traceState_2323_ = lean_ctor_get(v___x_2316_, 4);
v_cache_2324_ = lean_ctor_get(v___x_2316_, 5);
v_messages_2325_ = lean_ctor_get(v___x_2316_, 6);
v_infoState_2326_ = lean_ctor_get(v___x_2316_, 7);
v_snapshotTasks_2327_ = lean_ctor_get(v___x_2316_, 8);
v_isSharedCheck_2341_ = !lean_is_exclusive(v___x_2316_);
if (v_isSharedCheck_2341_ == 0)
{
v___x_2329_ = v___x_2316_;
v_isShared_2330_ = v_isSharedCheck_2341_;
goto v_resetjp_2328_;
}
else
{
lean_inc(v_snapshotTasks_2327_);
lean_inc(v_infoState_2326_);
lean_inc(v_messages_2325_);
lean_inc(v_cache_2324_);
lean_inc(v_traceState_2323_);
lean_inc(v_auxDeclNGen_2322_);
lean_inc(v_ngen_2321_);
lean_inc(v_nextMacroScope_2320_);
lean_inc(v_env_2319_);
lean_dec(v___x_2316_);
v___x_2329_ = lean_box(0);
v_isShared_2330_ = v_isSharedCheck_2341_;
goto v_resetjp_2328_;
}
v_resetjp_2328_:
{
lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2336_; 
lean_inc(v_openDecls_2318_);
lean_inc(v_currNamespace_2317_);
v___x_2331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2331_, 0, v_currNamespace_2317_);
lean_ctor_set(v___x_2331_, 1, v_openDecls_2318_);
v___x_2332_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2332_, 0, v___x_2331_);
lean_ctor_set(v___x_2332_, 1, v___y_2308_);
lean_inc_ref(v___y_2312_);
lean_inc_ref(v___y_2310_);
v___x_2333_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2333_, 0, v___y_2310_);
lean_ctor_set(v___x_2333_, 1, v___y_2307_);
lean_ctor_set(v___x_2333_, 2, v___y_2313_);
lean_ctor_set(v___x_2333_, 3, v___y_2312_);
lean_ctor_set(v___x_2333_, 4, v___x_2332_);
lean_ctor_set_uint8(v___x_2333_, sizeof(void*)*5, v___y_2309_);
lean_ctor_set_uint8(v___x_2333_, sizeof(void*)*5 + 1, v___y_2311_);
lean_ctor_set_uint8(v___x_2333_, sizeof(void*)*5 + 2, v_isSilent_2300_);
v___x_2334_ = l_Lean_MessageLog_add(v___x_2333_, v_messages_2325_);
if (v_isShared_2330_ == 0)
{
lean_ctor_set(v___x_2329_, 6, v___x_2334_);
v___x_2336_ = v___x_2329_;
goto v_reusejp_2335_;
}
else
{
lean_object* v_reuseFailAlloc_2340_; 
v_reuseFailAlloc_2340_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2340_, 0, v_env_2319_);
lean_ctor_set(v_reuseFailAlloc_2340_, 1, v_nextMacroScope_2320_);
lean_ctor_set(v_reuseFailAlloc_2340_, 2, v_ngen_2321_);
lean_ctor_set(v_reuseFailAlloc_2340_, 3, v_auxDeclNGen_2322_);
lean_ctor_set(v_reuseFailAlloc_2340_, 4, v_traceState_2323_);
lean_ctor_set(v_reuseFailAlloc_2340_, 5, v_cache_2324_);
lean_ctor_set(v_reuseFailAlloc_2340_, 6, v___x_2334_);
lean_ctor_set(v_reuseFailAlloc_2340_, 7, v_infoState_2326_);
lean_ctor_set(v_reuseFailAlloc_2340_, 8, v_snapshotTasks_2327_);
v___x_2336_ = v_reuseFailAlloc_2340_;
goto v_reusejp_2335_;
}
v_reusejp_2335_:
{
lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; 
v___x_2337_ = lean_st_ref_set(v___y_2315_, v___x_2336_);
v___x_2338_ = lean_box(0);
v___x_2339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2339_, 0, v___x_2338_);
return v___x_2339_;
}
}
}
v___jp_2342_:
{
lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v_a_2353_; lean_object* v___x_2355_; uint8_t v_isShared_2356_; uint8_t v_isSharedCheck_2366_; 
v___x_2351_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2298_);
v___x_2352_ = lp_LeanSearchClient_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__6(v___x_2351_, v___y_2301_, v___y_2302_, v___y_2303_, v___y_2304_);
v_a_2353_ = lean_ctor_get(v___x_2352_, 0);
v_isSharedCheck_2366_ = !lean_is_exclusive(v___x_2352_);
if (v_isSharedCheck_2366_ == 0)
{
v___x_2355_ = v___x_2352_;
v_isShared_2356_ = v_isSharedCheck_2366_;
goto v_resetjp_2354_;
}
else
{
lean_inc(v_a_2353_);
lean_dec(v___x_2352_);
v___x_2355_ = lean_box(0);
v_isShared_2356_ = v_isSharedCheck_2366_;
goto v_resetjp_2354_;
}
v_resetjp_2354_:
{
lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; 
lean_inc_ref_n(v___y_2345_, 2);
v___x_2357_ = l_Lean_FileMap_toPosition(v___y_2345_, v___y_2344_);
lean_dec(v___y_2344_);
v___x_2358_ = l_Lean_FileMap_toPosition(v___y_2345_, v___y_2350_);
lean_dec(v___y_2350_);
v___x_2359_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2359_, 0, v___x_2358_);
v___x_2360_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleMatch_default___closed__0));
if (v___y_2346_ == 0)
{
lean_del_object(v___x_2355_);
lean_dec_ref(v___y_2343_);
v___y_2307_ = v___x_2357_;
v___y_2308_ = v_a_2353_;
v___y_2309_ = v___y_2347_;
v___y_2310_ = v___y_2348_;
v___y_2311_ = v___y_2349_;
v___y_2312_ = v___x_2360_;
v___y_2313_ = v___x_2359_;
v___y_2314_ = v___y_2303_;
v___y_2315_ = v___y_2304_;
goto v___jp_2306_;
}
else
{
uint8_t v___x_2361_; 
lean_inc(v_a_2353_);
v___x_2361_ = l_Lean_MessageData_hasTag(v___y_2343_, v_a_2353_);
if (v___x_2361_ == 0)
{
lean_object* v___x_2362_; lean_object* v___x_2364_; 
lean_dec_ref_known(v___x_2359_, 1);
lean_dec_ref(v___x_2357_);
lean_dec(v_a_2353_);
v___x_2362_ = lean_box(0);
if (v_isShared_2356_ == 0)
{
lean_ctor_set(v___x_2355_, 0, v___x_2362_);
v___x_2364_ = v___x_2355_;
goto v_reusejp_2363_;
}
else
{
lean_object* v_reuseFailAlloc_2365_; 
v_reuseFailAlloc_2365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2365_, 0, v___x_2362_);
v___x_2364_ = v_reuseFailAlloc_2365_;
goto v_reusejp_2363_;
}
v_reusejp_2363_:
{
return v___x_2364_;
}
}
else
{
lean_del_object(v___x_2355_);
v___y_2307_ = v___x_2357_;
v___y_2308_ = v_a_2353_;
v___y_2309_ = v___y_2347_;
v___y_2310_ = v___y_2348_;
v___y_2311_ = v___y_2349_;
v___y_2312_ = v___x_2360_;
v___y_2313_ = v___x_2359_;
v___y_2314_ = v___y_2303_;
v___y_2315_ = v___y_2304_;
goto v___jp_2306_;
}
}
}
}
v___jp_2367_:
{
lean_object* v___x_2376_; 
v___x_2376_ = l_Lean_Syntax_getTailPos_x3f(v___y_2370_, v___y_2371_);
lean_dec(v___y_2370_);
if (lean_obj_tag(v___x_2376_) == 0)
{
lean_inc(v___y_2375_);
v___y_2343_ = v___y_2368_;
v___y_2344_ = v___y_2375_;
v___y_2345_ = v___y_2369_;
v___y_2346_ = v___y_2372_;
v___y_2347_ = v___y_2371_;
v___y_2348_ = v___y_2373_;
v___y_2349_ = v___y_2374_;
v___y_2350_ = v___y_2375_;
goto v___jp_2342_;
}
else
{
lean_object* v_val_2377_; 
v_val_2377_ = lean_ctor_get(v___x_2376_, 0);
lean_inc(v_val_2377_);
lean_dec_ref_known(v___x_2376_, 1);
v___y_2343_ = v___y_2368_;
v___y_2344_ = v___y_2375_;
v___y_2345_ = v___y_2369_;
v___y_2346_ = v___y_2372_;
v___y_2347_ = v___y_2371_;
v___y_2348_ = v___y_2373_;
v___y_2349_ = v___y_2374_;
v___y_2350_ = v_val_2377_;
goto v___jp_2342_;
}
}
v___jp_2378_:
{
lean_object* v_ref_2386_; lean_object* v___x_2387_; 
v_ref_2386_ = l_Lean_replaceRef(v_ref_2297_, v___y_2384_);
v___x_2387_ = l_Lean_Syntax_getPos_x3f(v_ref_2386_, v___y_2382_);
if (lean_obj_tag(v___x_2387_) == 0)
{
lean_object* v___x_2388_; 
v___x_2388_ = lean_unsigned_to_nat(0u);
v___y_2368_ = v___y_2379_;
v___y_2369_ = v___y_2380_;
v___y_2370_ = v_ref_2386_;
v___y_2371_ = v___y_2382_;
v___y_2372_ = v___y_2381_;
v___y_2373_ = v___y_2383_;
v___y_2374_ = v___y_2385_;
v___y_2375_ = v___x_2388_;
goto v___jp_2367_;
}
else
{
lean_object* v_val_2389_; 
v_val_2389_ = lean_ctor_get(v___x_2387_, 0);
lean_inc(v_val_2389_);
lean_dec_ref_known(v___x_2387_, 1);
v___y_2368_ = v___y_2379_;
v___y_2369_ = v___y_2380_;
v___y_2370_ = v_ref_2386_;
v___y_2371_ = v___y_2382_;
v___y_2372_ = v___y_2381_;
v___y_2373_ = v___y_2383_;
v___y_2374_ = v___y_2385_;
v___y_2375_ = v_val_2389_;
goto v___jp_2367_;
}
}
v___jp_2391_:
{
if (v___y_2398_ == 0)
{
v___y_2379_ = v___y_2394_;
v___y_2380_ = v___y_2392_;
v___y_2381_ = v___y_2393_;
v___y_2382_ = v___y_2397_;
v___y_2383_ = v___y_2395_;
v___y_2384_ = v___y_2396_;
v___y_2385_ = v_severity_2299_;
goto v___jp_2378_;
}
else
{
v___y_2379_ = v___y_2394_;
v___y_2380_ = v___y_2392_;
v___y_2381_ = v___y_2393_;
v___y_2382_ = v___y_2397_;
v___y_2383_ = v___y_2395_;
v___y_2384_ = v___y_2396_;
v___y_2385_ = v___x_2390_;
goto v___jp_2378_;
}
}
v___jp_2399_:
{
if (v___y_2400_ == 0)
{
lean_object* v_fileName_2401_; lean_object* v_fileMap_2402_; lean_object* v_options_2403_; lean_object* v_ref_2404_; uint8_t v_suppressElabErrors_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___f_2408_; uint8_t v___x_2409_; uint8_t v___x_2410_; 
v_fileName_2401_ = lean_ctor_get(v___y_2303_, 0);
v_fileMap_2402_ = lean_ctor_get(v___y_2303_, 1);
v_options_2403_ = lean_ctor_get(v___y_2303_, 2);
v_ref_2404_ = lean_ctor_get(v___y_2303_, 5);
v_suppressElabErrors_2405_ = lean_ctor_get_uint8(v___y_2303_, sizeof(void*)*14 + 1);
v___x_2406_ = lean_box(v___y_2400_);
v___x_2407_ = lean_box(v_suppressElabErrors_2405_);
v___f_2408_ = lean_alloc_closure((void*)(lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2408_, 0, v___x_2406_);
lean_closure_set(v___f_2408_, 1, v___x_2407_);
v___x_2409_ = 1;
v___x_2410_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2299_, v___x_2409_);
if (v___x_2410_ == 0)
{
v___y_2392_ = v_fileMap_2402_;
v___y_2393_ = v_suppressElabErrors_2405_;
v___y_2394_ = v___f_2408_;
v___y_2395_ = v_fileName_2401_;
v___y_2396_ = v_ref_2404_;
v___y_2397_ = v___y_2400_;
v___y_2398_ = v___x_2410_;
goto v___jp_2391_;
}
else
{
lean_object* v___x_2411_; uint8_t v___x_2412_; 
v___x_2411_ = l_Lean_warningAsError;
v___x_2412_ = lp_LeanSearchClient_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleCmdImpl_spec__1_spec__1_spec__2_spec__7(v_options_2403_, v___x_2411_);
v___y_2392_ = v_fileMap_2402_;
v___y_2393_ = v_suppressElabErrors_2405_;
v___y_2394_ = v___f_2408_;
v___y_2395_ = v_fileName_2401_;
v___y_2396_ = v_ref_2404_;
v___y_2397_ = v___y_2400_;
v___y_2398_ = v___x_2412_;
goto v___jp_2391_;
}
}
else
{
lean_object* v___x_2413_; lean_object* v___x_2414_; 
lean_dec_ref(v_msgData_2298_);
v___x_2413_ = lean_box(0);
v___x_2414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2414_, 0, v___x_2413_);
return v___x_2414_;
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_ref_2417_, lean_object* v_msgData_2418_, lean_object* v_severity_2419_, lean_object* v_isSilent_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
uint8_t v_severity_boxed_2426_; uint8_t v_isSilent_boxed_2427_; lean_object* v_res_2428_; 
v_severity_boxed_2426_ = lean_unbox(v_severity_2419_);
v_isSilent_boxed_2427_ = lean_unbox(v_isSilent_2420_);
v_res_2428_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg(v_ref_2417_, v_msgData_2418_, v_severity_boxed_2426_, v_isSilent_boxed_2427_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
lean_dec(v___y_2424_);
lean_dec_ref(v___y_2423_);
lean_dec(v___y_2422_);
lean_dec_ref(v___y_2421_);
lean_dec(v_ref_2417_);
return v_res_2428_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1(lean_object* v_msgData_2429_, uint8_t v_severity_2430_, uint8_t v_isSilent_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_, lean_object* v___y_2435_, lean_object* v___y_2436_, lean_object* v___y_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_){
_start:
{
lean_object* v_ref_2441_; lean_object* v___x_2442_; 
v_ref_2441_ = lean_ctor_get(v___y_2438_, 5);
v___x_2442_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg(v_ref_2441_, v_msgData_2429_, v_severity_2430_, v_isSilent_2431_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_);
return v___x_2442_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1___boxed(lean_object* v_msgData_2443_, lean_object* v_severity_2444_, lean_object* v_isSilent_2445_, lean_object* v___y_2446_, lean_object* v___y_2447_, lean_object* v___y_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_){
_start:
{
uint8_t v_severity_boxed_2455_; uint8_t v_isSilent_boxed_2456_; lean_object* v_res_2457_; 
v_severity_boxed_2455_ = lean_unbox(v_severity_2444_);
v_isSilent_boxed_2456_ = lean_unbox(v_isSilent_2445_);
v_res_2457_ = lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1(v_msgData_2443_, v_severity_boxed_2455_, v_isSilent_boxed_2456_, v___y_2446_, v___y_2447_, v___y_2448_, v___y_2449_, v___y_2450_, v___y_2451_, v___y_2452_, v___y_2453_);
lean_dec(v___y_2453_);
lean_dec_ref(v___y_2452_);
lean_dec(v___y_2451_);
lean_dec_ref(v___y_2450_);
lean_dec(v___y_2449_);
lean_dec_ref(v___y_2448_);
lean_dec(v___y_2447_);
lean_dec_ref(v___y_2446_);
return v_res_2457_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1(lean_object* v_msgData_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_){
_start:
{
uint8_t v___x_2468_; uint8_t v___x_2469_; lean_object* v___x_2470_; 
v___x_2468_ = 0;
v___x_2469_ = 0;
v___x_2470_ = lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1(v_msgData_2458_, v___x_2468_, v___x_2469_, v___y_2459_, v___y_2460_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_, v___y_2465_, v___y_2466_);
return v___x_2470_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1___boxed(lean_object* v_msgData_2471_, lean_object* v___y_2472_, lean_object* v___y_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_){
_start:
{
lean_object* v_res_2481_; 
v_res_2481_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1(v_msgData_2471_, v___y_2472_, v___y_2473_, v___y_2474_, v___y_2475_, v___y_2476_, v___y_2477_, v___y_2478_, v___y_2479_);
lean_dec(v___y_2479_);
lean_dec_ref(v___y_2478_);
lean_dec(v___y_2477_);
lean_dec_ref(v___y_2476_);
lean_dec(v___y_2475_);
lean_dec_ref(v___y_2474_);
lean_dec(v___y_2473_);
lean_dec_ref(v___y_2472_);
return v_res_2481_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTacticImpl_spec__2(size_t v_sz_2482_, size_t v_i_2483_, lean_object* v_bs_2484_){
_start:
{
uint8_t v___x_2485_; 
v___x_2485_ = lean_usize_dec_lt(v_i_2483_, v_sz_2482_);
if (v___x_2485_ == 0)
{
return v_bs_2484_;
}
else
{
lean_object* v_v_2486_; lean_object* v_name_2487_; lean_object* v___x_2488_; lean_object* v_bs_x27_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; size_t v___x_2492_; size_t v___x_2493_; lean_object* v___x_2494_; 
v_v_2486_ = lean_array_uget(v_bs_2484_, v_i_2483_);
v_name_2487_ = lean_ctor_get(v_v_2486_, 0);
lean_inc_ref(v_name_2487_);
v___x_2488_ = lean_unsigned_to_nat(0u);
v_bs_x27_2489_ = lean_array_uset(v_bs_2484_, v_i_2483_, v___x_2488_);
v___x_2490_ = lp_LeanSearchClient_LeanSearchClient_SearchResult_toTacticSuggestions(v_v_2486_);
v___x_2491_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2491_, 0, v_name_2487_);
lean_ctor_set(v___x_2491_, 1, v___x_2490_);
v___x_2492_ = ((size_t)1ULL);
v___x_2493_ = lean_usize_add(v_i_2483_, v___x_2492_);
v___x_2494_ = lean_array_uset(v_bs_x27_2489_, v_i_2483_, v___x_2491_);
v_i_2483_ = v___x_2493_;
v_bs_2484_ = v___x_2494_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTacticImpl_spec__2___boxed(lean_object* v_sz_2496_, lean_object* v_i_2497_, lean_object* v_bs_2498_){
_start:
{
size_t v_sz_boxed_2499_; size_t v_i_boxed_2500_; lean_object* v_res_2501_; 
v_sz_boxed_2499_ = lean_unbox_usize(v_sz_2496_);
lean_dec(v_sz_2496_);
v_i_boxed_2500_ = lean_unbox_usize(v_i_2497_);
lean_dec(v_i_2497_);
v_res_2501_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTacticImpl_spec__2(v_sz_boxed_2499_, v_i_boxed_2500_, v_bs_2498_);
return v_res_2501_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5(lean_object* v_msgData_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_){
_start:
{
uint8_t v___x_2512_; uint8_t v___x_2513_; lean_object* v___x_2514_; 
v___x_2512_ = 1;
v___x_2513_ = 0;
v___x_2514_ = lp_LeanSearchClient_Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1(v_msgData_2502_, v___x_2512_, v___x_2513_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_);
return v___x_2514_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5___boxed(lean_object* v_msgData_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_, lean_object* v___y_2519_, lean_object* v___y_2520_, lean_object* v___y_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_){
_start:
{
lean_object* v_res_2525_; 
v_res_2525_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5(v_msgData_2515_, v___y_2516_, v___y_2517_, v___y_2518_, v___y_2519_, v___y_2520_, v___y_2521_, v___y_2522_, v___y_2523_);
lean_dec(v___y_2523_);
lean_dec_ref(v___y_2522_);
lean_dec(v___y_2521_);
lean_dec_ref(v___y_2520_);
lean_dec(v___y_2519_);
lean_dec_ref(v___y_2518_);
lean_dec(v___y_2517_);
lean_dec_ref(v___y_2516_);
return v_res_2525_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3(lean_object* v_as_2530_, size_t v_i_2531_, size_t v_stop_2532_, lean_object* v_b_2533_, lean_object* v___y_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_){
_start:
{
lean_object* v_a_2544_; uint8_t v___x_2548_; 
v___x_2548_ = lean_usize_dec_eq(v_i_2531_, v_stop_2532_);
if (v___x_2548_ == 0)
{
lean_object* v___x_2549_; lean_object* v_suggestion_2550_; 
v___x_2549_ = lean_array_uget_borrowed(v_as_2530_, v_i_2531_);
v_suggestion_2550_ = lean_ctor_get(v___x_2549_, 0);
if (lean_obj_tag(v_suggestion_2550_) == 1)
{
lean_object* v_a_2551_; lean_object* v___x_2552_; lean_object* v_env_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; 
v_a_2551_ = lean_ctor_get(v_suggestion_2550_, 0);
v___x_2552_ = lean_st_ref_get(v___y_2541_);
v_env_2553_ = lean_ctor_get(v___x_2552_, 0);
lean_inc_ref(v_env_2553_);
lean_dec(v___x_2552_);
v___x_2554_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__1));
v___x_2555_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___closed__2));
lean_inc_ref(v_a_2551_);
v___x_2556_ = l_Lean_Parser_runParserCategory(v_env_2553_, v___x_2554_, v_a_2551_, v___x_2555_);
if (lean_obj_tag(v___x_2556_) == 0)
{
lean_dec_ref_known(v___x_2556_, 1);
v_a_2544_ = v_b_2533_;
goto v___jp_2543_;
}
else
{
lean_object* v_a_2557_; lean_object* v___x_2558_; 
v_a_2557_ = lean_ctor_get(v___x_2556_, 0);
lean_inc(v_a_2557_);
lean_dec_ref_known(v___x_2556_, 1);
v___x_2558_ = l_Lean_Elab_Tactic_getMainTarget(v___y_2534_, v___y_2535_, v___y_2536_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_);
if (lean_obj_tag(v___x_2558_) == 0)
{
lean_object* v_a_2559_; lean_object* v___x_2560_; 
v_a_2559_ = lean_ctor_get(v___x_2558_, 0);
lean_inc(v_a_2559_);
lean_dec_ref_known(v___x_2558_, 1);
v___x_2560_ = lp_LeanSearchClient_LeanSearchClient_checkTactic(v_a_2559_, v_a_2557_, v___y_2536_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_);
if (lean_obj_tag(v___x_2560_) == 0)
{
lean_object* v_a_2561_; 
v_a_2561_ = lean_ctor_get(v___x_2560_, 0);
lean_inc(v_a_2561_);
lean_dec_ref_known(v___x_2560_, 1);
if (lean_obj_tag(v_a_2561_) == 0)
{
v_a_2544_ = v_b_2533_;
goto v___jp_2543_;
}
else
{
lean_object* v___x_2562_; 
lean_dec_ref_known(v_a_2561_, 1);
lean_inc(v___x_2549_);
v___x_2562_ = lean_array_push(v_b_2533_, v___x_2549_);
v_a_2544_ = v___x_2562_;
goto v___jp_2543_;
}
}
else
{
lean_object* v_a_2563_; lean_object* v___x_2565_; uint8_t v_isShared_2566_; uint8_t v_isSharedCheck_2570_; 
lean_dec_ref(v_b_2533_);
v_a_2563_ = lean_ctor_get(v___x_2560_, 0);
v_isSharedCheck_2570_ = !lean_is_exclusive(v___x_2560_);
if (v_isSharedCheck_2570_ == 0)
{
v___x_2565_ = v___x_2560_;
v_isShared_2566_ = v_isSharedCheck_2570_;
goto v_resetjp_2564_;
}
else
{
lean_inc(v_a_2563_);
lean_dec(v___x_2560_);
v___x_2565_ = lean_box(0);
v_isShared_2566_ = v_isSharedCheck_2570_;
goto v_resetjp_2564_;
}
v_resetjp_2564_:
{
lean_object* v___x_2568_; 
if (v_isShared_2566_ == 0)
{
v___x_2568_ = v___x_2565_;
goto v_reusejp_2567_;
}
else
{
lean_object* v_reuseFailAlloc_2569_; 
v_reuseFailAlloc_2569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2569_, 0, v_a_2563_);
v___x_2568_ = v_reuseFailAlloc_2569_;
goto v_reusejp_2567_;
}
v_reusejp_2567_:
{
return v___x_2568_;
}
}
}
}
else
{
lean_object* v_a_2571_; lean_object* v___x_2573_; uint8_t v_isShared_2574_; uint8_t v_isSharedCheck_2578_; 
lean_dec(v_a_2557_);
lean_dec_ref(v_b_2533_);
v_a_2571_ = lean_ctor_get(v___x_2558_, 0);
v_isSharedCheck_2578_ = !lean_is_exclusive(v___x_2558_);
if (v_isSharedCheck_2578_ == 0)
{
v___x_2573_ = v___x_2558_;
v_isShared_2574_ = v_isSharedCheck_2578_;
goto v_resetjp_2572_;
}
else
{
lean_inc(v_a_2571_);
lean_dec(v___x_2558_);
v___x_2573_ = lean_box(0);
v_isShared_2574_ = v_isSharedCheck_2578_;
goto v_resetjp_2572_;
}
v_resetjp_2572_:
{
lean_object* v___x_2576_; 
if (v_isShared_2574_ == 0)
{
v___x_2576_ = v___x_2573_;
goto v_reusejp_2575_;
}
else
{
lean_object* v_reuseFailAlloc_2577_; 
v_reuseFailAlloc_2577_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2577_, 0, v_a_2571_);
v___x_2576_ = v_reuseFailAlloc_2577_;
goto v_reusejp_2575_;
}
v_reusejp_2575_:
{
return v___x_2576_;
}
}
}
}
}
else
{
v_a_2544_ = v_b_2533_;
goto v___jp_2543_;
}
}
else
{
lean_object* v___x_2579_; 
v___x_2579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2579_, 0, v_b_2533_);
return v___x_2579_;
}
v___jp_2543_:
{
size_t v___x_2545_; size_t v___x_2546_; 
v___x_2545_ = ((size_t)1ULL);
v___x_2546_ = lean_usize_add(v_i_2531_, v___x_2545_);
v_i_2531_ = v___x_2546_;
v_b_2533_ = v_a_2544_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3___boxed(lean_object* v_as_2580_, lean_object* v_i_2581_, lean_object* v_stop_2582_, lean_object* v_b_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_, lean_object* v___y_2590_, lean_object* v___y_2591_, lean_object* v___y_2592_){
_start:
{
size_t v_i_boxed_2593_; size_t v_stop_boxed_2594_; lean_object* v_res_2595_; 
v_i_boxed_2593_ = lean_unbox_usize(v_i_2581_);
lean_dec(v_i_2581_);
v_stop_boxed_2594_ = lean_unbox_usize(v_stop_2582_);
lean_dec(v_stop_2582_);
v_res_2595_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3(v_as_2580_, v_i_boxed_2593_, v_stop_boxed_2594_, v_b_2583_, v___y_2584_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_, v___y_2591_);
lean_dec(v___y_2591_);
lean_dec_ref(v___y_2590_);
lean_dec(v___y_2589_);
lean_dec_ref(v___y_2588_);
lean_dec(v___y_2587_);
lean_dec_ref(v___y_2586_);
lean_dec(v___y_2585_);
lean_dec_ref(v___y_2584_);
lean_dec_ref(v_as_2580_);
return v_res_2595_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4(lean_object* v_stx_2599_, lean_object* v_as_2600_, size_t v_sz_2601_, size_t v_i_2602_, lean_object* v_b_2603_, lean_object* v___y_2604_, lean_object* v___y_2605_, lean_object* v___y_2606_, lean_object* v___y_2607_, lean_object* v___y_2608_, lean_object* v___y_2609_, lean_object* v___y_2610_, lean_object* v___y_2611_){
_start:
{
lean_object* v_a_2614_; uint8_t v___x_2618_; 
v___x_2618_ = lean_usize_dec_lt(v_i_2602_, v_sz_2601_);
if (v___x_2618_ == 0)
{
lean_object* v___x_2619_; 
lean_dec(v_stx_2599_);
v___x_2619_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2619_, 0, v_b_2603_);
return v___x_2619_;
}
else
{
lean_object* v_a_2620_; lean_object* v_fst_2621_; lean_object* v_snd_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v_a_2626_; lean_object* v___y_2636_; lean_object* v___x_2646_; lean_object* v___x_2647_; uint8_t v___x_2648_; 
v_a_2620_ = lean_array_uget_borrowed(v_as_2600_, v_i_2602_);
v_fst_2621_ = lean_ctor_get(v_a_2620_, 0);
v_snd_2622_ = lean_ctor_get(v_a_2620_, 1);
v___x_2623_ = lean_box(0);
v___x_2624_ = lean_unsigned_to_nat(0u);
v___x_2646_ = lean_array_get_size(v_snd_2622_);
v___x_2647_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__1));
v___x_2648_ = lean_nat_dec_lt(v___x_2624_, v___x_2646_);
if (v___x_2648_ == 0)
{
v_a_2626_ = v___x_2647_;
goto v___jp_2625_;
}
else
{
uint8_t v___x_2649_; 
v___x_2649_ = lean_nat_dec_le(v___x_2646_, v___x_2646_);
if (v___x_2649_ == 0)
{
if (v___x_2648_ == 0)
{
v_a_2626_ = v___x_2647_;
goto v___jp_2625_;
}
else
{
size_t v___x_2650_; size_t v___x_2651_; lean_object* v___x_2652_; 
v___x_2650_ = ((size_t)0ULL);
v___x_2651_ = lean_usize_of_nat(v___x_2646_);
v___x_2652_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3(v_snd_2622_, v___x_2650_, v___x_2651_, v___x_2647_, v___y_2604_, v___y_2605_, v___y_2606_, v___y_2607_, v___y_2608_, v___y_2609_, v___y_2610_, v___y_2611_);
v___y_2636_ = v___x_2652_;
goto v___jp_2635_;
}
}
else
{
size_t v___x_2653_; size_t v___x_2654_; lean_object* v___x_2655_; 
v___x_2653_ = ((size_t)0ULL);
v___x_2654_ = lean_usize_of_nat(v___x_2646_);
v___x_2655_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00LeanSearchClient_loogleTacticImpl_spec__3(v_snd_2622_, v___x_2653_, v___x_2654_, v___x_2647_, v___y_2604_, v___y_2605_, v___y_2606_, v___y_2607_, v___y_2608_, v___y_2609_, v___y_2610_, v___y_2611_);
v___y_2636_ = v___x_2655_;
goto v___jp_2635_;
}
}
v___jp_2625_:
{
lean_object* v___x_2627_; uint8_t v___x_2628_; 
v___x_2627_ = lean_array_get_size(v_a_2626_);
v___x_2628_ = lean_nat_dec_eq(v___x_2627_, v___x_2624_);
if (v___x_2628_ == 0)
{
lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; uint8_t v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; 
v___x_2629_ = lean_box(0);
v___x_2630_ = ((lean_object*)(lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___closed__0));
v___x_2631_ = lean_string_append(v___x_2630_, v_fst_2621_);
v___x_2632_ = 4;
v___x_2633_ = l_Lean_MessageData_nil;
lean_inc(v_stx_2599_);
v___x_2634_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2599_, v_a_2626_, v___x_2629_, v___x_2631_, v___x_2629_, v___x_2632_, v___x_2633_, v___y_2610_, v___y_2611_);
if (lean_obj_tag(v___x_2634_) == 0)
{
lean_dec_ref_known(v___x_2634_, 1);
v_a_2614_ = v___x_2623_;
goto v___jp_2613_;
}
else
{
lean_dec(v_stx_2599_);
return v___x_2634_;
}
}
else
{
lean_dec_ref(v_a_2626_);
v_a_2614_ = v___x_2623_;
goto v___jp_2613_;
}
}
v___jp_2635_:
{
if (lean_obj_tag(v___y_2636_) == 0)
{
lean_object* v_a_2637_; 
v_a_2637_ = lean_ctor_get(v___y_2636_, 0);
lean_inc(v_a_2637_);
lean_dec_ref_known(v___y_2636_, 1);
v_a_2626_ = v_a_2637_;
goto v___jp_2625_;
}
else
{
lean_object* v_a_2638_; lean_object* v___x_2640_; uint8_t v_isShared_2641_; uint8_t v_isSharedCheck_2645_; 
lean_dec(v_stx_2599_);
v_a_2638_ = lean_ctor_get(v___y_2636_, 0);
v_isSharedCheck_2645_ = !lean_is_exclusive(v___y_2636_);
if (v_isSharedCheck_2645_ == 0)
{
v___x_2640_ = v___y_2636_;
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
else
{
lean_inc(v_a_2638_);
lean_dec(v___y_2636_);
v___x_2640_ = lean_box(0);
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
v_resetjp_2639_:
{
lean_object* v___x_2643_; 
if (v_isShared_2641_ == 0)
{
v___x_2643_ = v___x_2640_;
goto v_reusejp_2642_;
}
else
{
lean_object* v_reuseFailAlloc_2644_; 
v_reuseFailAlloc_2644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2644_, 0, v_a_2638_);
v___x_2643_ = v_reuseFailAlloc_2644_;
goto v_reusejp_2642_;
}
v_reusejp_2642_:
{
return v___x_2643_;
}
}
}
}
}
v___jp_2613_:
{
size_t v___x_2615_; size_t v___x_2616_; 
v___x_2615_ = ((size_t)1ULL);
v___x_2616_ = lean_usize_add(v_i_2602_, v___x_2615_);
v_i_2602_ = v___x_2616_;
v_b_2603_ = v_a_2614_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4___boxed(lean_object* v_stx_2656_, lean_object* v_as_2657_, lean_object* v_sz_2658_, lean_object* v_i_2659_, lean_object* v_b_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_){
_start:
{
size_t v_sz_boxed_2670_; size_t v_i_boxed_2671_; lean_object* v_res_2672_; 
v_sz_boxed_2670_ = lean_unbox_usize(v_sz_2658_);
lean_dec(v_sz_2658_);
v_i_boxed_2671_ = lean_unbox_usize(v_i_2659_);
lean_dec(v_i_2659_);
v_res_2672_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4(v_stx_2656_, v_as_2657_, v_sz_boxed_2670_, v_i_boxed_2671_, v_b_2660_, v___y_2661_, v___y_2662_, v___y_2663_, v___y_2664_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_);
lean_dec(v___y_2668_);
lean_dec_ref(v___y_2667_);
lean_dec(v___y_2666_);
lean_dec_ref(v___y_2665_);
lean_dec(v___y_2664_);
lean_dec_ref(v___y_2663_);
lean_dec(v___y_2662_);
lean_dec_ref(v___y_2661_);
lean_dec_ref(v_as_2657_);
return v_res_2672_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTacticImpl(lean_object* v_stx_2673_, lean_object* v_a_2674_, lean_object* v_a_2675_, lean_object* v_a_2676_, lean_object* v_a_2677_, lean_object* v_a_2678_, lean_object* v_a_2679_, lean_object* v_a_2680_, lean_object* v_a_2681_){
_start:
{
lean_object* v___x_2683_; uint8_t v___x_2684_; 
v___x_2683_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogle__tactic___closed__1));
lean_inc(v_stx_2673_);
v___x_2684_ = l_Lean_Syntax_isOfKind(v_stx_2673_, v___x_2683_);
if (v___x_2684_ == 0)
{
lean_object* v___x_2685_; 
lean_dec(v_stx_2673_);
v___x_2685_ = lp_LeanSearchClient_Lean_Elab_throwUnsupportedSyntax___at___00LeanSearchClient_loogleTacticImpl_spec__0___redArg();
return v___x_2685_;
}
else
{
lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; 
v___x_2686_ = lean_unsigned_to_nat(1u);
v___x_2687_ = l_Lean_Syntax_getArg(v_stx_2673_, v___x_2686_);
v___x_2688_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogle__filters___closed__1));
v___x_2689_ = l_Lean_PrettyPrinter_ppCategory(v___x_2688_, v___x_2687_, v_a_2680_, v_a_2681_);
if (lean_obj_tag(v___x_2689_) == 0)
{
lean_object* v_a_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; 
v_a_2690_ = lean_ctor_get(v___x_2689_, 0);
lean_inc(v_a_2690_);
lean_dec_ref_known(v___x_2689_, 1);
v___x_2691_ = lean_unsigned_to_nat(0u);
v___x_2692_ = l_Std_Format_defWidth;
v___x_2693_ = l_Std_Format_pretty(v_a_2690_, v___x_2692_, v___x_2691_, v___x_2691_);
v___x_2694_ = lean_unsigned_to_nat(6u);
v___x_2695_ = lp_LeanSearchClient_LeanSearchClient_getLoogleQueryJson(v___x_2693_, v___x_2694_, v_a_2680_, v_a_2681_);
lean_dec_ref(v___x_2693_);
if (lean_obj_tag(v___x_2695_) == 0)
{
lean_object* v_a_2696_; 
v_a_2696_ = lean_ctor_get(v___x_2695_, 0);
lean_inc(v_a_2696_);
lean_dec_ref_known(v___x_2695_, 1);
switch(lean_obj_tag(v_a_2696_))
{
case 0:
{
lean_object* v___x_2697_; lean_object* v___x_2698_; 
lean_dec(v_stx_2673_);
v___x_2697_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_2698_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1(v___x_2697_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_, v_a_2678_, v_a_2679_, v_a_2680_, v_a_2681_);
return v___x_2698_;
}
case 1:
{
lean_object* v_a_2699_; size_t v_sz_2700_; size_t v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; size_t v_sz_2704_; lean_object* v___x_2705_; 
v_a_2699_ = lean_ctor_get(v_a_2696_, 0);
lean_inc_ref(v_a_2699_);
lean_dec_ref_known(v_a_2696_, 1);
v_sz_2700_ = lean_array_size(v_a_2699_);
v___x_2701_ = ((size_t)0ULL);
v___x_2702_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00LeanSearchClient_loogleTacticImpl_spec__2(v_sz_2700_, v___x_2701_, v_a_2699_);
v___x_2703_ = lean_box(0);
v_sz_2704_ = lean_array_size(v___x_2702_);
v___x_2705_ = lp_LeanSearchClient___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00LeanSearchClient_loogleTacticImpl_spec__4(v_stx_2673_, v___x_2702_, v_sz_2704_, v___x_2701_, v___x_2703_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_, v_a_2678_, v_a_2679_, v_a_2680_, v_a_2681_);
lean_dec_ref(v___x_2702_);
if (lean_obj_tag(v___x_2705_) == 0)
{
lean_object* v___x_2707_; uint8_t v_isShared_2708_; uint8_t v_isSharedCheck_2712_; 
v_isSharedCheck_2712_ = !lean_is_exclusive(v___x_2705_);
if (v_isSharedCheck_2712_ == 0)
{
lean_object* v_unused_2713_; 
v_unused_2713_ = lean_ctor_get(v___x_2705_, 0);
lean_dec(v_unused_2713_);
v___x_2707_ = v___x_2705_;
v_isShared_2708_ = v_isSharedCheck_2712_;
goto v_resetjp_2706_;
}
else
{
lean_dec(v___x_2705_);
v___x_2707_ = lean_box(0);
v_isShared_2708_ = v_isSharedCheck_2712_;
goto v_resetjp_2706_;
}
v_resetjp_2706_:
{
lean_object* v___x_2710_; 
if (v_isShared_2708_ == 0)
{
lean_ctor_set(v___x_2707_, 0, v___x_2703_);
v___x_2710_ = v___x_2707_;
goto v_reusejp_2709_;
}
else
{
lean_object* v_reuseFailAlloc_2711_; 
v_reuseFailAlloc_2711_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2711_, 0, v___x_2703_);
v___x_2710_ = v_reuseFailAlloc_2711_;
goto v_reusejp_2709_;
}
v_reusejp_2709_:
{
return v___x_2710_;
}
}
}
else
{
return v___x_2705_;
}
}
default: 
{
lean_object* v_error_2714_; lean_object* v_suggestions_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; 
v_error_2714_ = lean_ctor_get(v_a_2696_, 0);
lean_inc_ref(v_error_2714_);
v_suggestions_2715_ = lean_ctor_get(v_a_2696_, 1);
lean_inc(v_suggestions_2715_);
lean_dec_ref_known(v_a_2696_, 2);
v___x_2716_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__6));
v___x_2717_ = lean_string_append(v___x_2716_, v_error_2714_);
lean_dec_ref(v_error_2714_);
v___x_2718_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2718_, 0, v___x_2717_);
v___x_2719_ = l_Lean_MessageData_ofFormat(v___x_2718_);
v___x_2720_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5(v___x_2719_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_, v_a_2678_, v_a_2679_, v_a_2680_, v_a_2681_);
if (lean_obj_tag(v___x_2720_) == 0)
{
lean_object* v___x_2721_; lean_object* v___x_2722_; 
lean_dec_ref_known(v___x_2720_, 1);
v___x_2721_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_2722_ = lp_LeanSearchClient_Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1(v___x_2721_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_, v_a_2678_, v_a_2679_, v_a_2680_, v_a_2681_);
if (lean_obj_tag(v___x_2722_) == 0)
{
lean_object* v___x_2724_; uint8_t v_isShared_2725_; uint8_t v_isSharedCheck_2744_; 
v_isSharedCheck_2744_ = !lean_is_exclusive(v___x_2722_);
if (v_isSharedCheck_2744_ == 0)
{
lean_object* v_unused_2745_; 
v_unused_2745_ = lean_ctor_get(v___x_2722_, 0);
lean_dec(v_unused_2745_);
v___x_2724_ = v___x_2722_;
v_isShared_2725_ = v_isSharedCheck_2744_;
goto v_resetjp_2723_;
}
else
{
lean_dec(v___x_2722_);
v___x_2724_ = lean_box(0);
v_isShared_2725_ = v_isSharedCheck_2744_;
goto v_resetjp_2723_;
}
v_resetjp_2723_:
{
if (lean_obj_tag(v_suggestions_2715_) == 0)
{
lean_object* v___x_2726_; lean_object* v___x_2728_; 
lean_dec(v_stx_2673_);
v___x_2726_ = lean_box(0);
if (v_isShared_2725_ == 0)
{
lean_ctor_set(v___x_2724_, 0, v___x_2726_);
v___x_2728_ = v___x_2724_;
goto v_reusejp_2727_;
}
else
{
lean_object* v_reuseFailAlloc_2729_; 
v_reuseFailAlloc_2729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2729_, 0, v___x_2726_);
v___x_2728_ = v_reuseFailAlloc_2729_;
goto v_reusejp_2727_;
}
v_reusejp_2727_:
{
return v___x_2728_;
}
}
else
{
lean_object* v_val_2730_; lean_object* v___x_2731_; lean_object* v___x_2732_; uint8_t v___x_2733_; 
v_val_2730_ = lean_ctor_get(v_suggestions_2715_, 0);
lean_inc(v_val_2730_);
lean_dec_ref_known(v_suggestions_2715_, 1);
v___x_2731_ = lean_box(0);
v___x_2732_ = lp_LeanSearchClient_List_mapTR_loop___at___00LeanSearchClient_loogleTacticImpl_spec__6(v_val_2730_, v___x_2731_);
v___x_2733_ = l_List_isEmpty___redArg(v___x_2732_);
if (v___x_2733_ == 0)
{
lean_object* v___x_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; uint8_t v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; 
lean_del_object(v___x_2724_);
v___x_2734_ = lean_array_mk(v___x_2732_);
v___x_2735_ = lean_box(0);
v___x_2736_ = ((lean_object*)(lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__7));
v___x_2737_ = 4;
v___x_2738_ = l_Lean_MessageData_nil;
v___x_2739_ = l_Lean_Meta_Tactic_TryThis_addSuggestions___redArg(v_stx_2673_, v___x_2734_, v___x_2735_, v___x_2736_, v___x_2735_, v___x_2737_, v___x_2738_, v_a_2680_, v_a_2681_);
return v___x_2739_;
}
else
{
lean_object* v___x_2740_; lean_object* v___x_2742_; 
lean_dec(v___x_2732_);
lean_dec(v_stx_2673_);
v___x_2740_ = lean_box(0);
if (v_isShared_2725_ == 0)
{
lean_ctor_set(v___x_2724_, 0, v___x_2740_);
v___x_2742_ = v___x_2724_;
goto v_reusejp_2741_;
}
else
{
lean_object* v_reuseFailAlloc_2743_; 
v_reuseFailAlloc_2743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2743_, 0, v___x_2740_);
v___x_2742_ = v_reuseFailAlloc_2743_;
goto v_reusejp_2741_;
}
v_reusejp_2741_:
{
return v___x_2742_;
}
}
}
}
}
else
{
lean_dec(v_suggestions_2715_);
lean_dec(v_stx_2673_);
return v___x_2722_;
}
}
else
{
lean_dec(v_suggestions_2715_);
lean_dec(v_stx_2673_);
return v___x_2720_;
}
}
}
}
else
{
lean_object* v_a_2746_; lean_object* v___x_2748_; uint8_t v_isShared_2749_; uint8_t v_isSharedCheck_2753_; 
lean_dec(v_stx_2673_);
v_a_2746_ = lean_ctor_get(v___x_2695_, 0);
v_isSharedCheck_2753_ = !lean_is_exclusive(v___x_2695_);
if (v_isSharedCheck_2753_ == 0)
{
v___x_2748_ = v___x_2695_;
v_isShared_2749_ = v_isSharedCheck_2753_;
goto v_resetjp_2747_;
}
else
{
lean_inc(v_a_2746_);
lean_dec(v___x_2695_);
v___x_2748_ = lean_box(0);
v_isShared_2749_ = v_isSharedCheck_2753_;
goto v_resetjp_2747_;
}
v_resetjp_2747_:
{
lean_object* v___x_2751_; 
if (v_isShared_2749_ == 0)
{
v___x_2751_ = v___x_2748_;
goto v_reusejp_2750_;
}
else
{
lean_object* v_reuseFailAlloc_2752_; 
v_reuseFailAlloc_2752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2752_, 0, v_a_2746_);
v___x_2751_ = v_reuseFailAlloc_2752_;
goto v_reusejp_2750_;
}
v_reusejp_2750_:
{
return v___x_2751_;
}
}
}
}
else
{
lean_object* v_a_2754_; lean_object* v___x_2756_; uint8_t v_isShared_2757_; uint8_t v_isSharedCheck_2761_; 
lean_dec(v_stx_2673_);
v_a_2754_ = lean_ctor_get(v___x_2689_, 0);
v_isSharedCheck_2761_ = !lean_is_exclusive(v___x_2689_);
if (v_isSharedCheck_2761_ == 0)
{
v___x_2756_ = v___x_2689_;
v_isShared_2757_ = v_isSharedCheck_2761_;
goto v_resetjp_2755_;
}
else
{
lean_inc(v_a_2754_);
lean_dec(v___x_2689_);
v___x_2756_ = lean_box(0);
v_isShared_2757_ = v_isSharedCheck_2761_;
goto v_resetjp_2755_;
}
v_resetjp_2755_:
{
lean_object* v___x_2759_; 
if (v_isShared_2757_ == 0)
{
v___x_2759_ = v___x_2756_;
goto v_reusejp_2758_;
}
else
{
lean_object* v_reuseFailAlloc_2760_; 
v_reuseFailAlloc_2760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2760_, 0, v_a_2754_);
v___x_2759_ = v_reuseFailAlloc_2760_;
goto v_reusejp_2758_;
}
v_reusejp_2758_:
{
return v___x_2759_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_loogleTacticImpl___boxed(lean_object* v_stx_2762_, lean_object* v_a_2763_, lean_object* v_a_2764_, lean_object* v_a_2765_, lean_object* v_a_2766_, lean_object* v_a_2767_, lean_object* v_a_2768_, lean_object* v_a_2769_, lean_object* v_a_2770_, lean_object* v_a_2771_){
_start:
{
lean_object* v_res_2772_; 
v_res_2772_ = lp_LeanSearchClient_LeanSearchClient_loogleTacticImpl(v_stx_2762_, v_a_2763_, v_a_2764_, v_a_2765_, v_a_2766_, v_a_2767_, v_a_2768_, v_a_2769_, v_a_2770_);
lean_dec(v_a_2770_);
lean_dec_ref(v_a_2769_);
lean_dec(v_a_2768_);
lean_dec_ref(v_a_2767_);
lean_dec(v_a_2766_);
lean_dec_ref(v_a_2765_);
lean_dec(v_a_2764_);
lean_dec_ref(v_a_2763_);
return v_res_2772_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2(lean_object* v_ref_2773_, lean_object* v_msgData_2774_, uint8_t v_severity_2775_, uint8_t v_isSilent_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_){
_start:
{
lean_object* v___x_2786_; 
v___x_2786_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___redArg(v_ref_2773_, v_msgData_2774_, v_severity_2775_, v_isSilent_2776_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_);
return v___x_2786_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_2787_, lean_object* v_msgData_2788_, lean_object* v_severity_2789_, lean_object* v_isSilent_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_){
_start:
{
uint8_t v_severity_boxed_2800_; uint8_t v_isSilent_boxed_2801_; lean_object* v_res_2802_; 
v_severity_boxed_2800_ = lean_unbox(v_severity_2789_);
v_isSilent_boxed_2801_ = lean_unbox(v_isSilent_2790_);
v_res_2802_ = lp_LeanSearchClient_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00LeanSearchClient_loogleTacticImpl_spec__1_spec__1_spec__2(v_ref_2787_, v_msgData_2788_, v_severity_boxed_2800_, v_isSilent_boxed_2801_, v___y_2791_, v___y_2792_, v___y_2793_, v___y_2794_, v___y_2795_, v___y_2796_, v___y_2797_, v___y_2798_);
lean_dec(v___y_2798_);
lean_dec_ref(v___y_2797_);
lean_dec(v___y_2796_);
lean_dec_ref(v___y_2795_);
lean_dec(v___y_2794_);
lean_dec_ref(v___y_2793_);
lean_dec(v___y_2792_);
lean_dec_ref(v___y_2791_);
lean_dec(v_ref_2787_);
return v_res_2802_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___redArg(lean_object* v_a_2815_, lean_object* v_a_2816_, lean_object* v_a_2817_, lean_object* v_a_2818_, lean_object* v_a_2819_, lean_object* v_a_2820_, lean_object* v_a_2821_, lean_object* v_a_2822_){
_start:
{
lean_object* v___x_2824_; lean_object* v___x_2825_; 
v___x_2824_ = lean_obj_once(&lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1, &lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1_once, _init_lp_LeanSearchClient_LeanSearchClient_loogleCmdImpl___lam__0___closed__1);
v___x_2825_ = lp_LeanSearchClient_Lean_logWarning___at___00LeanSearchClient_loogleTacticImpl_spec__5(v___x_2824_, v_a_2815_, v_a_2816_, v_a_2817_, v_a_2818_, v_a_2819_, v_a_2820_, v_a_2821_, v_a_2822_);
return v___x_2825_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___redArg___boxed(lean_object* v_a_2826_, lean_object* v_a_2827_, lean_object* v_a_2828_, lean_object* v_a_2829_, lean_object* v_a_2830_, lean_object* v_a_2831_, lean_object* v_a_2832_, lean_object* v_a_2833_, lean_object* v_a_2834_){
_start:
{
lean_object* v_res_2835_; 
v_res_2835_ = lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___redArg(v_a_2826_, v_a_2827_, v_a_2828_, v_a_2829_, v_a_2830_, v_a_2831_, v_a_2832_, v_a_2833_);
lean_dec(v_a_2833_);
lean_dec_ref(v_a_2832_);
lean_dec(v_a_2831_);
lean_dec_ref(v_a_2830_);
lean_dec(v_a_2829_);
lean_dec_ref(v_a_2828_);
lean_dec(v_a_2827_);
lean_dec_ref(v_a_2826_);
return v_res_2835_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl(lean_object* v_x_2836_, lean_object* v_a_2837_, lean_object* v_a_2838_, lean_object* v_a_2839_, lean_object* v_a_2840_, lean_object* v_a_2841_, lean_object* v_a_2842_, lean_object* v_a_2843_, lean_object* v_a_2844_){
_start:
{
lean_object* v___x_2846_; 
v___x_2846_ = lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___redArg(v_a_2837_, v_a_2838_, v_a_2839_, v_a_2840_, v_a_2841_, v_a_2842_, v_a_2843_, v_a_2844_);
return v___x_2846_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl___boxed(lean_object* v_x_2847_, lean_object* v_a_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_, lean_object* v_a_2852_, lean_object* v_a_2853_, lean_object* v_a_2854_, lean_object* v_a_2855_, lean_object* v_a_2856_){
_start:
{
lean_object* v_res_2857_; 
v_res_2857_ = lp_LeanSearchClient_LeanSearchClient_justLoogleTacticImpl(v_x_2847_, v_a_2848_, v_a_2849_, v_a_2850_, v_a_2851_, v_a_2852_, v_a_2853_, v_a_2854_, v_a_2855_);
lean_dec(v_a_2855_);
lean_dec_ref(v_a_2854_);
lean_dec(v_a_2853_);
lean_dec_ref(v_a_2852_);
lean_dec(v_a_2851_);
lean_dec_ref(v_a_2850_);
lean_dec(v_a_2849_);
lean_dec_ref(v_a_2848_);
lean_dec(v_x_2847_);
return v_res_2857_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient_LoogleSyntax(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Parser_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin);
lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient_Syntax(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_LeanSearchClient_LeanSearchClient_LoogleSyntax(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult_default = _init_lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult_default();
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult_default);
lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult = _init_lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult();
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_instInhabitedLoogleResult);
res = lp_LeanSearchClient___private_LeanSearchClient_LoogleSyntax_0__LeanSearchClient_initFn_00___x40_LeanSearchClient_LoogleSyntax_2643959438____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_LeanSearchClient_loogleCache = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_loogleCache);
lean_dec_ref(res);
lp_LeanSearchClient_LeanSearchClient_unicode__turnstile = _init_lp_LeanSearchClient_LeanSearchClient_unicode__turnstile();
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_unicode__turnstile);
lp_LeanSearchClient_LeanSearchClient_ascii__turnstile = _init_lp_LeanSearchClient_LeanSearchClient_ascii__turnstile();
lean_mark_persistent(lp_LeanSearchClient_LeanSearchClient_ascii__turnstile);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Meta(uint8_t builtin);
lean_object* initialize_Lean_Parser_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin);
lean_object* initialize_LeanSearchClient_LeanSearchClient_Syntax(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LeanSearchClient_LeanSearchClient_LoogleSyntax(uint8_t builtin) {
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
res = initialize_Lean_Parser_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_LeanSearchClient_LeanSearchClient_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient_LoogleSyntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_LeanSearchClient_LeanSearchClient_LoogleSyntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_LeanSearchClient_LeanSearchClient_LoogleSyntax(builtin);
}
#ifdef __cplusplus
}
#endif
