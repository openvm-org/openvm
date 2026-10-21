// Lean compiler output
// Module: Mathlib.Tactic.Linter.DocString
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.Linter.Header public meta import Std.Data.Iterators.Combinators.Zip public import Lean.Parser.Command meta import Std.Data.Iterators.Producers.Range
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
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_string_get_byte_fast(lean_object*, lean_object*);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_String_Slice_posGE___redArg(lean_object*, lean_object*);
lean_object* l_String_Slice_slice_x21(lean_object*, lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_length(lean_object*);
lean_object* l_String_Slice_subslice_x21(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_FileMap_ofString(lean_object*);
lean_object* l_Lean_Parser_mkParserState(lean_object*);
lean_object* l_Lean_Doc_Parser_document(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_getTokenTable(lean_object*);
lean_object* l_Lean_Parser_ParserFn_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_ParserState_allErrors(lean_object*);
lean_object* l_instDecidableEqString___boxed(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Parser_SyntaxStack_back(lean_object*);
lean_object* l_Lean_Parser_Error_toString(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
extern lean_object* l_Lean_doc_verso_module;
extern lean_object* l_Lean_doc_verso;
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isMissing(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lean_string_mk(lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_prevn(lean_object*, lean_object*, lean_object*);
uint8_t l_String_Slice_beq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_getTrailing_x3f(lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "docString"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(54, 182, 27, 106, 156, 224, 166, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "enable the style.docString linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(249, 64, 33, 172, 114, 16, 197, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_docString;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "empty"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(54, 182, 27, 106, 156, 224, 166, 223)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(254, 188, 170, 130, 193, 241, 187, 203)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "enable the style.docString.empty linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(249, 64, 33, 172, 114, 16, 197, 138)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(125, 122, 67, 167, 94, 47, 79, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_docString_empty;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "docStringVerso"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(32, 25, 128, 145, 108, 134, 150, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "enable the style.docStringVerso linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(135, 97, 5, 70, 160, 20, 62, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
static const lean_array_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5_value;
static const lean_array_object lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_getDeclModifiers_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_getDeclModifiers_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Array_map__unattach_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Array_map__unattach_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Doc_Parser_document, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "link target '(url)' or '[ref]' (use '\\[' for a literal '[')"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___boxed(lean_object*);
static const lean_ctor_object lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3___boxed(lean_object*);
static const lean_string_object lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "$$"};
static const lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__5;
static lean_once_cell_t lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__6;
static const lean_ctor_object lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__7 = (const lean_object*)&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "LaTeX"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg___closed__0 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "http://"};
static const lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__5;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___boxed(lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "(http"};
static const lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__5;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "https://"};
static const lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__5;
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___boxed(lean_object*);
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "URL"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__0_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2(uint8_t, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8_spec__12(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "unexpected doc string"};
static const lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1;
static const lean_string_object lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "versoCommentBody"};
static const lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(44, 76, 179, 33, 27, 4, 201, 125)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "error: doc-strings should end with a single space or newline"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__2_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__4;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__6;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__7;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "error: doc-strings should not end with a comma"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__8_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__10;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "error: doc-strings should start with a single space or newline"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__11_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__13;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__14 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__14_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 246}, .m_size = 2, .m_capacity = 2, .m_data = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___closed__0_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__15_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "warning: this doc-string is empty"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__16 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__17;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__5_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "DocString"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__8_value),LEAN_SCALAR_PTR_LITERAL(30, 169, 75, 138, 63, 18, 205, 96)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(231, 19, 1, 126, 12, 185, 51, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(106, 159, 59, 32, 19, 46, 183, 16)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(212, 129, 123, 207, 101, 253, 96, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__13_value),LEAN_SCALAR_PTR_LITERAL(108, 147, 118, 162, 115, 181, 140, 66)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "docStringLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__15_value),LEAN_SCALAR_PTR_LITERAL(236, 203, 115, 3, 25, 146, 124, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3364454516____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3364454516____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get_x3f___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get_x3f___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "moduleDoc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "moduleDocVersoLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(165, 167, 92, 38, 199, 241, 187, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3183647073____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3183647073____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_));
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_));
v___x_59_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0(v___x_56_, v___x_57_, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4____boxed(lean_object* v_a_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_();
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_82_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_));
v___x_83_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_));
v___x_84_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_));
v___x_85_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0(v___x_82_, v___x_83_, v___x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4____boxed(lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_();
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_));
v___x_107_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_));
v___x_108_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_));
v___x_109_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4__spec__0(v___x_106_, v___x_107_, v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4____boxed(lean_object* v_a_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_();
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getDeclModifiers(lean_object* v_x_125_){
_start:
{
if (lean_obj_tag(v_x_125_) == 1)
{
lean_object* v_kind_126_; lean_object* v_args_127_; lean_object* v___y_129_; lean_object* v___x_145_; uint8_t v___x_146_; 
v_kind_126_ = lean_ctor_get(v_x_125_, 1);
v_args_127_ = lean_ctor_get(v_x_125_, 2);
lean_inc_ref(v_args_127_);
v___x_145_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__5));
v___x_146_ = lean_name_eq(v_kind_126_, v___x_145_);
if (v___x_146_ == 0)
{
lean_object* v___x_147_; 
lean_dec_ref_known(v_x_125_, 3);
v___x_147_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__6));
v___y_129_ = v___x_147_;
goto v___jp_128_;
}
else
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_148_ = lean_unsigned_to_nat(1u);
v___x_149_ = lean_mk_empty_array_with_capacity(v___x_148_);
v___x_150_ = lean_array_push(v___x_149_, v_x_125_);
v___y_129_ = v___x_150_;
goto v___jp_128_;
}
v___jp_128_:
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_130_ = lean_unsigned_to_nat(0u);
v___x_131_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__0));
v___x_132_ = lean_array_get_size(v_args_127_);
v___x_133_ = lean_nat_dec_lt(v___x_130_, v___x_132_);
if (v___x_133_ == 0)
{
lean_object* v___x_134_; 
lean_dec_ref(v_args_127_);
v___x_134_ = l_Array_append___redArg(v___y_129_, v___x_131_);
return v___x_134_;
}
else
{
uint8_t v___x_135_; 
v___x_135_ = lean_nat_dec_le(v___x_132_, v___x_132_);
if (v___x_135_ == 0)
{
if (v___x_133_ == 0)
{
lean_object* v___x_136_; 
lean_dec_ref(v_args_127_);
v___x_136_ = l_Array_append___redArg(v___y_129_, v___x_131_);
return v___x_136_;
}
else
{
size_t v___x_137_; size_t v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_137_ = ((size_t)0ULL);
v___x_138_ = lean_usize_of_nat(v___x_132_);
v___x_139_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0(v_args_127_, v___x_137_, v___x_138_, v___x_131_);
lean_dec_ref(v_args_127_);
v___x_140_ = l_Array_append___redArg(v___y_129_, v___x_139_);
lean_dec_ref(v___x_139_);
return v___x_140_;
}
}
else
{
size_t v___x_141_; size_t v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_141_ = ((size_t)0ULL);
v___x_142_ = lean_usize_of_nat(v___x_132_);
v___x_143_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0(v_args_127_, v___x_141_, v___x_142_, v___x_131_);
lean_dec_ref(v_args_127_);
v___x_144_ = l_Array_append___redArg(v___y_129_, v___x_143_);
lean_dec_ref(v___x_143_);
return v___x_144_;
}
}
}
}
else
{
lean_object* v___x_151_; 
lean_dec(v_x_125_);
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__6));
return v___x_151_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0(lean_object* v_as_152_, size_t v_i_153_, size_t v_stop_154_, lean_object* v_b_155_){
_start:
{
uint8_t v___x_156_; 
v___x_156_ = lean_usize_dec_eq(v_i_153_, v_stop_154_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; size_t v___x_160_; size_t v___x_161_; 
v___x_157_ = lean_array_uget_borrowed(v_as_152_, v_i_153_);
lean_inc(v___x_157_);
v___x_158_ = lp_mathlib_Mathlib_Linter_getDeclModifiers(v___x_157_);
v___x_159_ = l_Array_append___redArg(v_b_155_, v___x_158_);
lean_dec_ref(v___x_158_);
v___x_160_ = ((size_t)1ULL);
v___x_161_ = lean_usize_add(v_i_153_, v___x_160_);
v_i_153_ = v___x_161_;
v_b_155_ = v___x_159_;
goto _start;
}
else
{
return v_b_155_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0___boxed(lean_object* v_as_163_, lean_object* v_i_164_, lean_object* v_stop_165_, lean_object* v_b_166_){
_start:
{
size_t v_i_boxed_167_; size_t v_stop_boxed_168_; lean_object* v_res_169_; 
v_i_boxed_167_ = lean_unbox_usize(v_i_164_);
lean_dec(v_i_164_);
v_stop_boxed_168_ = lean_unbox_usize(v_stop_165_);
lean_dec(v_stop_165_);
v_res_169_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Linter_getDeclModifiers_spec__0(v_as_163_, v_i_boxed_167_, v_stop_boxed_168_, v_b_166_);
lean_dec_ref(v_as_163_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_getDeclModifiers_match__1_splitter___redArg(lean_object* v_x_170_, lean_object* v_h__1_171_, lean_object* v_h__2_172_){
_start:
{
if (lean_obj_tag(v_x_170_) == 1)
{
lean_object* v_info_173_; lean_object* v_kind_174_; lean_object* v_args_175_; lean_object* v___x_176_; 
lean_dec(v_h__2_172_);
v_info_173_ = lean_ctor_get(v_x_170_, 0);
lean_inc(v_info_173_);
v_kind_174_ = lean_ctor_get(v_x_170_, 1);
lean_inc(v_kind_174_);
v_args_175_ = lean_ctor_get(v_x_170_, 2);
lean_inc_ref(v_args_175_);
lean_dec_ref_known(v_x_170_, 3);
v___x_176_ = lean_apply_3(v_h__1_171_, v_info_173_, v_kind_174_, v_args_175_);
return v___x_176_;
}
else
{
lean_object* v___x_177_; 
lean_dec(v_h__1_171_);
v___x_177_ = lean_apply_2(v_h__2_172_, v_x_170_, lean_box(0));
return v___x_177_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_getDeclModifiers_match__1_splitter(lean_object* v_motive_178_, lean_object* v_x_179_, lean_object* v_h__1_180_, lean_object* v_h__2_181_){
_start:
{
if (lean_obj_tag(v_x_179_) == 1)
{
lean_object* v_info_182_; lean_object* v_kind_183_; lean_object* v_args_184_; lean_object* v___x_185_; 
lean_dec(v_h__2_181_);
v_info_182_ = lean_ctor_get(v_x_179_, 0);
lean_inc(v_info_182_);
v_kind_183_ = lean_ctor_get(v_x_179_, 1);
lean_inc(v_kind_183_);
v_args_184_ = lean_ctor_get(v_x_179_, 2);
lean_inc_ref(v_args_184_);
lean_dec_ref_known(v_x_179_, 3);
v___x_185_ = lean_apply_3(v_h__1_180_, v_info_182_, v_kind_183_, v_args_184_);
return v___x_185_;
}
else
{
lean_object* v___x_186_; 
lean_dec(v_h__1_180_);
v___x_186_ = lean_apply_2(v_h__2_181_, v_x_179_, lean_box(0));
return v___x_186_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Array_map__unattach_match__1_splitter___redArg(lean_object* v_x_187_, lean_object* v_h__1_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lean_apply_2(v_h__1_188_, v_x_187_, lean_box(0));
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Array_map__unattach_match__1_splitter(lean_object* v_00_u03b1_190_, lean_object* v_P_191_, lean_object* v_motive_192_, lean_object* v_x_193_, lean_object* v_h__1_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lean_apply_2(v_h__1_194_, v_x_193_, lean_box(0));
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg(lean_object* v_s_196_, lean_object* v_replacement_197_, lean_object* v_a_198_, lean_object* v_b_199_){
_start:
{
lean_object* v_it_201_; lean_object* v_startPos_202_; lean_object* v_endPos_203_; lean_object* v_it_212_; 
switch(lean_obj_tag(v_a_198_))
{
case 0:
{
lean_object* v_pos_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_230_; 
v_pos_218_ = lean_ctor_get(v_a_198_, 0);
v_isSharedCheck_230_ = !lean_is_exclusive(v_a_198_);
if (v_isSharedCheck_230_ == 0)
{
v___x_220_ = v_a_198_;
v_isShared_221_ = v_isSharedCheck_230_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_pos_218_);
lean_dec(v_a_198_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_230_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
lean_object* v_startInclusive_222_; lean_object* v_endExclusive_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v_startInclusive_222_ = lean_ctor_get(v_s_196_, 1);
v_endExclusive_223_ = lean_ctor_get(v_s_196_, 2);
v___x_224_ = lean_nat_sub(v_endExclusive_223_, v_startInclusive_222_);
v___x_225_ = lean_nat_dec_eq(v_pos_218_, v___x_224_);
lean_dec(v___x_224_);
if (v___x_225_ == 0)
{
lean_object* v___x_227_; 
if (v_isShared_221_ == 0)
{
lean_ctor_set_tag(v___x_220_, 1);
v___x_227_ = v___x_220_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_pos_218_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
v_it_212_ = v___x_227_;
goto v___jp_211_;
}
}
else
{
lean_object* v___x_229_; 
lean_del_object(v___x_220_);
lean_dec(v_pos_218_);
v___x_229_ = lean_box(3);
v_it_212_ = v___x_229_;
goto v___jp_211_;
}
}
}
case 1:
{
lean_object* v_pos_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_243_; 
v_pos_231_ = lean_ctor_get(v_a_198_, 0);
v_isSharedCheck_243_ = !lean_is_exclusive(v_a_198_);
if (v_isSharedCheck_243_ == 0)
{
v___x_233_ = v_a_198_;
v_isShared_234_ = v_isSharedCheck_243_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_pos_231_);
lean_dec(v_a_198_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_243_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v_str_235_; lean_object* v_startInclusive_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_241_; 
v_str_235_ = lean_ctor_get(v_s_196_, 0);
v_startInclusive_236_ = lean_ctor_get(v_s_196_, 1);
v___x_237_ = lean_nat_add(v_startInclusive_236_, v_pos_231_);
v___x_238_ = lean_string_utf8_next_fast(v_str_235_, v___x_237_);
lean_dec(v___x_237_);
v___x_239_ = lean_nat_sub(v___x_238_, v_startInclusive_236_);
lean_inc(v___x_239_);
if (v_isShared_234_ == 0)
{
lean_ctor_set_tag(v___x_233_, 0);
lean_ctor_set(v___x_233_, 0, v___x_239_);
v___x_241_ = v___x_233_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v___x_239_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
v_it_201_ = v___x_241_;
v_startPos_202_ = v_pos_231_;
v_endPos_203_ = v___x_239_;
goto v___jp_200_;
}
}
}
case 2:
{
lean_object* v_needle_244_; lean_object* v_table_245_; lean_object* v_stackPos_246_; lean_object* v_needlePos_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_306_; 
v_needle_244_ = lean_ctor_get(v_a_198_, 0);
v_table_245_ = lean_ctor_get(v_a_198_, 1);
v_stackPos_246_ = lean_ctor_get(v_a_198_, 2);
v_needlePos_247_ = lean_ctor_get(v_a_198_, 3);
v_isSharedCheck_306_ = !lean_is_exclusive(v_a_198_);
if (v_isSharedCheck_306_ == 0)
{
v___x_249_ = v_a_198_;
v_isShared_250_ = v_isSharedCheck_306_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_needlePos_247_);
lean_inc(v_stackPos_246_);
lean_inc(v_table_245_);
lean_inc(v_needle_244_);
lean_dec(v_a_198_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_306_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v_str_251_; lean_object* v_startInclusive_252_; lean_object* v_endExclusive_253_; lean_object* v_str_254_; lean_object* v_startInclusive_255_; lean_object* v_endExclusive_256_; lean_object* v_basePos_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; uint8_t v___x_261_; 
v_str_251_ = lean_ctor_get(v_needle_244_, 0);
v_startInclusive_252_ = lean_ctor_get(v_needle_244_, 1);
v_endExclusive_253_ = lean_ctor_get(v_needle_244_, 2);
v_str_254_ = lean_ctor_get(v_s_196_, 0);
v_startInclusive_255_ = lean_ctor_get(v_s_196_, 1);
v_endExclusive_256_ = lean_ctor_get(v_s_196_, 2);
v_basePos_257_ = lean_nat_sub(v_stackPos_246_, v_needlePos_247_);
v___x_258_ = lean_nat_sub(v_endExclusive_253_, v_startInclusive_252_);
v___x_259_ = lean_nat_add(v_basePos_257_, v___x_258_);
v___x_260_ = lean_nat_sub(v_endExclusive_256_, v_startInclusive_255_);
v___x_261_ = lean_nat_dec_le(v___x_259_, v___x_260_);
lean_dec(v___x_259_);
if (v___x_261_ == 0)
{
uint8_t v___x_262_; 
lean_dec(v___x_258_);
lean_del_object(v___x_249_);
lean_dec(v_needlePos_247_);
lean_dec(v_stackPos_246_);
lean_dec_ref(v_table_245_);
lean_dec_ref(v_needle_244_);
v___x_262_ = lean_nat_dec_lt(v_basePos_257_, v___x_260_);
if (v___x_262_ == 0)
{
lean_dec(v___x_260_);
lean_dec(v_basePos_257_);
lean_dec_ref(v_s_196_);
return v_b_199_;
}
else
{
lean_object* v___x_263_; lean_object* v___x_264_; 
v___x_263_ = l_String_Slice_pos_x21(v_s_196_, v_basePos_257_);
lean_dec(v_basePos_257_);
v___x_264_ = lean_box(3);
v_it_201_ = v___x_264_;
v_startPos_202_ = v___x_263_;
v_endPos_203_ = v___x_260_;
goto v___jp_200_;
}
}
else
{
lean_object* v___x_265_; uint8_t v_stackByte_266_; lean_object* v___x_267_; uint8_t v_patByte_268_; uint8_t v___x_269_; 
lean_dec(v___x_260_);
v___x_265_ = lean_nat_add(v_startInclusive_255_, v_stackPos_246_);
v_stackByte_266_ = lean_string_get_byte_fast(v_str_254_, v___x_265_);
v___x_267_ = lean_nat_add(v_startInclusive_252_, v_needlePos_247_);
v_patByte_268_ = lean_string_get_byte_fast(v_str_251_, v___x_267_);
v___x_269_ = lean_uint8_dec_eq(v_stackByte_266_, v_patByte_268_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; uint8_t v___x_271_; 
lean_dec(v___x_258_);
v___x_270_ = lean_unsigned_to_nat(0u);
v___x_271_ = lean_nat_dec_eq(v_needlePos_247_, v___x_270_);
if (v___x_271_ == 0)
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v_newNeedlePos_274_; uint8_t v___x_275_; 
v___x_272_ = lean_unsigned_to_nat(1u);
v___x_273_ = lean_nat_sub(v_needlePos_247_, v___x_272_);
lean_dec(v_needlePos_247_);
v_newNeedlePos_274_ = lean_array_fget_borrowed(v_table_245_, v___x_273_);
lean_dec(v___x_273_);
v___x_275_ = lean_nat_dec_eq(v_newNeedlePos_274_, v___x_270_);
if (v___x_275_ == 0)
{
lean_object* v_oldBasePos_276_; lean_object* v___x_277_; lean_object* v_newBasePos_278_; lean_object* v___x_280_; 
lean_inc(v_newNeedlePos_274_);
v_oldBasePos_276_ = l_String_Slice_pos_x21(v_s_196_, v_basePos_257_);
lean_dec(v_basePos_257_);
v___x_277_ = lean_nat_sub(v_stackPos_246_, v_newNeedlePos_274_);
v_newBasePos_278_ = l_String_Slice_pos_x21(v_s_196_, v___x_277_);
lean_dec(v___x_277_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 3, v_newNeedlePos_274_);
v___x_280_ = v___x_249_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v_needle_244_);
lean_ctor_set(v_reuseFailAlloc_281_, 1, v_table_245_);
lean_ctor_set(v_reuseFailAlloc_281_, 2, v_stackPos_246_);
lean_ctor_set(v_reuseFailAlloc_281_, 3, v_newNeedlePos_274_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
v_it_201_ = v___x_280_;
v_startPos_202_ = v_oldBasePos_276_;
v_endPos_203_ = v_newBasePos_278_;
goto v___jp_200_;
}
}
else
{
lean_object* v_basePos_282_; lean_object* v_nextStackPos_283_; lean_object* v___x_285_; 
v_basePos_282_ = l_String_Slice_pos_x21(v_s_196_, v_basePos_257_);
lean_dec(v_basePos_257_);
v_nextStackPos_283_ = l_String_Slice_posGE___redArg(v_s_196_, v_stackPos_246_);
lean_inc(v_nextStackPos_283_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 3, v___x_270_);
lean_ctor_set(v___x_249_, 2, v_nextStackPos_283_);
v___x_285_ = v___x_249_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v_needle_244_);
lean_ctor_set(v_reuseFailAlloc_286_, 1, v_table_245_);
lean_ctor_set(v_reuseFailAlloc_286_, 2, v_nextStackPos_283_);
lean_ctor_set(v_reuseFailAlloc_286_, 3, v___x_270_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
v_it_201_ = v___x_285_;
v_startPos_202_ = v_basePos_282_;
v_endPos_203_ = v_nextStackPos_283_;
goto v___jp_200_;
}
}
}
else
{
lean_object* v_basePos_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v_nextStackPos_290_; lean_object* v___x_292_; 
lean_dec(v_basePos_257_);
lean_dec(v_needlePos_247_);
v_basePos_287_ = l_String_Slice_pos_x21(v_s_196_, v_stackPos_246_);
v___x_288_ = lean_unsigned_to_nat(1u);
v___x_289_ = lean_nat_add(v_stackPos_246_, v___x_288_);
lean_dec(v_stackPos_246_);
v_nextStackPos_290_ = l_String_Slice_posGE___redArg(v_s_196_, v___x_289_);
lean_inc(v_nextStackPos_290_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 3, v___x_270_);
lean_ctor_set(v___x_249_, 2, v_nextStackPos_290_);
v___x_292_ = v___x_249_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v_needle_244_);
lean_ctor_set(v_reuseFailAlloc_293_, 1, v_table_245_);
lean_ctor_set(v_reuseFailAlloc_293_, 2, v_nextStackPos_290_);
lean_ctor_set(v_reuseFailAlloc_293_, 3, v___x_270_);
v___x_292_ = v_reuseFailAlloc_293_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
v_it_201_ = v___x_292_;
v_startPos_202_ = v_basePos_287_;
v_endPos_203_ = v_nextStackPos_290_;
goto v___jp_200_;
}
}
}
else
{
lean_object* v___x_294_; lean_object* v_nextStackPos_295_; lean_object* v_nextNeedlePos_296_; uint8_t v___x_297_; 
lean_dec(v_basePos_257_);
v___x_294_ = lean_unsigned_to_nat(1u);
v_nextStackPos_295_ = lean_nat_add(v_stackPos_246_, v___x_294_);
lean_dec(v_stackPos_246_);
v_nextNeedlePos_296_ = lean_nat_add(v_needlePos_247_, v___x_294_);
lean_dec(v_needlePos_247_);
v___x_297_ = lean_nat_dec_eq(v_nextNeedlePos_296_, v___x_258_);
lean_dec(v___x_258_);
if (v___x_297_ == 0)
{
lean_object* v___x_299_; 
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 3, v_nextNeedlePos_296_);
lean_ctor_set(v___x_249_, 2, v_nextStackPos_295_);
v___x_299_ = v___x_249_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_needle_244_);
lean_ctor_set(v_reuseFailAlloc_301_, 1, v_table_245_);
lean_ctor_set(v_reuseFailAlloc_301_, 2, v_nextStackPos_295_);
lean_ctor_set(v_reuseFailAlloc_301_, 3, v_nextNeedlePos_296_);
v___x_299_ = v_reuseFailAlloc_301_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
v_a_198_ = v___x_299_;
goto _start;
}
}
else
{
lean_object* v___x_302_; lean_object* v___x_304_; 
lean_dec(v_nextNeedlePos_296_);
v___x_302_ = lean_unsigned_to_nat(0u);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 3, v___x_302_);
lean_ctor_set(v___x_249_, 2, v_nextStackPos_295_);
v___x_304_ = v___x_249_;
goto v_reusejp_303_;
}
else
{
lean_object* v_reuseFailAlloc_305_; 
v_reuseFailAlloc_305_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_305_, 0, v_needle_244_);
lean_ctor_set(v_reuseFailAlloc_305_, 1, v_table_245_);
lean_ctor_set(v_reuseFailAlloc_305_, 2, v_nextStackPos_295_);
lean_ctor_set(v_reuseFailAlloc_305_, 3, v___x_302_);
v___x_304_ = v_reuseFailAlloc_305_;
goto v_reusejp_303_;
}
v_reusejp_303_:
{
v_it_212_ = v___x_304_;
goto v___jp_211_;
}
}
}
}
}
}
default: 
{
lean_dec_ref(v_s_196_);
return v_b_199_;
}
}
v___jp_200_:
{
lean_object* v___x_204_; lean_object* v_str_205_; lean_object* v_startInclusive_206_; lean_object* v_endExclusive_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
lean_inc_ref(v_s_196_);
v___x_204_ = l_String_Slice_slice_x21(v_s_196_, v_startPos_202_, v_endPos_203_);
lean_dec(v_endPos_203_);
lean_dec(v_startPos_202_);
v_str_205_ = lean_ctor_get(v___x_204_, 0);
lean_inc_ref(v_str_205_);
v_startInclusive_206_ = lean_ctor_get(v___x_204_, 1);
lean_inc(v_startInclusive_206_);
v_endExclusive_207_ = lean_ctor_get(v___x_204_, 2);
lean_inc(v_endExclusive_207_);
lean_dec_ref(v___x_204_);
v___x_208_ = lean_string_utf8_extract_fast(v_str_205_, v_startInclusive_206_, v_endExclusive_207_);
lean_dec(v_endExclusive_207_);
lean_dec(v_startInclusive_206_);
lean_dec_ref(v_str_205_);
v___x_209_ = lean_string_append(v_b_199_, v___x_208_);
lean_dec_ref(v___x_208_);
v_a_198_ = v_it_201_;
v_b_199_ = v___x_209_;
goto _start;
}
v___jp_211_:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_213_ = lean_unsigned_to_nat(0u);
v___x_214_ = lean_string_utf8_byte_size(v_replacement_197_);
v___x_215_ = lean_string_utf8_extract_fast(v_replacement_197_, v___x_213_, v___x_214_);
v___x_216_ = lean_string_append(v_b_199_, v___x_215_);
lean_dec_ref(v___x_215_);
v_a_198_ = v_it_212_;
v_b_199_ = v___x_216_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg___boxed(lean_object* v_s_307_, lean_object* v_replacement_308_, lean_object* v_a_309_, lean_object* v_b_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg(v_s_307_, v_replacement_308_, v_a_309_, v_b_310_);
lean_dec_ref(v_replacement_308_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg(lean_object* v_indent_315_, lean_object* v_s_316_, lean_object* v_replacement_317_){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; uint8_t v___x_321_; 
v___x_318_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0));
v___x_319_ = lean_unsigned_to_nat(0u);
v___x_320_ = lean_string_utf8_byte_size(v_indent_315_);
v___x_321_ = lean_nat_dec_eq(v___x_320_, v___x_319_);
if (v___x_321_ == 0)
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_322_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_322_, 0, v_indent_315_);
lean_ctor_set(v___x_322_, 1, v___x_319_);
lean_ctor_set(v___x_322_, 2, v___x_320_);
v___x_323_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_322_);
v___x_324_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_324_, 0, v___x_322_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
lean_ctor_set(v___x_324_, 2, v___x_319_);
lean_ctor_set(v___x_324_, 3, v___x_319_);
v___x_325_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg(v_s_316_, v_replacement_317_, v___x_324_, v___x_318_);
return v___x_325_;
}
else
{
lean_object* v___x_326_; lean_object* v___x_327_; 
lean_dec_ref(v_indent_315_);
v___x_326_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1));
v___x_327_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg(v_s_316_, v_replacement_317_, v___x_326_, v___x_318_);
return v___x_327_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___boxed(lean_object* v_indent_328_, lean_object* v_s_329_, lean_object* v_replacement_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg(v_indent_328_, v_s_329_, v_replacement_330_);
lean_dec_ref(v_replacement_330_);
return v_res_331_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__1(void){
_start:
{
uint32_t v___x_333_; lean_object* v___x_334_; 
v___x_333_ = 10;
v___x_334_ = lean_box_uint32(v___x_333_);
return v___x_334_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__2(void){
_start:
{
uint32_t v___x_335_; lean_object* v___x_336_; 
v___x_335_ = 32;
v___x_336_ = lean_box_uint32(v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString(lean_object* v_currIndent_337_, lean_object* v_docString_338_){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v_indent_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_339_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__2;
v___x_340_ = l_List_replicateTR___redArg(v_currIndent_337_, v___x_339_);
v___x_341_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__1;
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v___x_340_);
v_indent_343_ = lean_string_mk(v___x_342_);
v___x_344_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___closed__0));
v___x_345_ = lean_unsigned_to_nat(0u);
v___x_346_ = lean_string_utf8_byte_size(v_docString_338_);
v___x_347_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_347_, 0, v_docString_338_);
lean_ctor_set(v___x_347_, 1, v___x_345_);
lean_ctor_set(v___x_347_, 2, v___x_346_);
v___x_348_ = lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg(v_indent_343_, v___x_347_, v___x_344_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0(lean_object* v_indent_349_, lean_object* v_s_350_, lean_object* v_pattern_351_, lean_object* v_replacement_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg(v_indent_349_, v_s_350_, v_replacement_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___boxed(lean_object* v_indent_354_, lean_object* v_s_355_, lean_object* v_pattern_356_, lean_object* v_replacement_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0(v_indent_354_, v_s_355_, v_pattern_356_, v_replacement_357_);
lean_dec_ref(v_replacement_357_);
lean_dec_ref(v_pattern_356_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0(lean_object* v_s_359_, lean_object* v_replacement_360_, lean_object* v_inst_361_, lean_object* v_R_362_, lean_object* v_a_363_, lean_object* v_b_364_, lean_object* v_c_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___redArg(v_s_359_, v_replacement_360_, v_a_363_, v_b_364_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0___boxed(lean_object* v_s_367_, lean_object* v_replacement_368_, lean_object* v_inst_369_, lean_object* v_R_370_, lean_object* v_a_371_, lean_object* v_b_372_, lean_object* v_c_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0_spec__0(v_s_367_, v_replacement_368_, v_inst_369_, v_R_370_, v_a_371_, v_b_372_, v_c_373_);
lean_dec_ref(v_replacement_368_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax(lean_object* v_docComment_385_, lean_object* v_fileName_386_, lean_object* v_a_387_, lean_object* v_a_388_){
_start:
{
lean_object* v___y_391_; 
if (lean_obj_tag(v_fileName_386_) == 0)
{
lean_object* v_fileName_437_; 
v_fileName_437_ = lean_ctor_get(v_a_387_, 0);
lean_inc_ref(v_fileName_437_);
v___y_391_ = v_fileName_437_;
goto v___jp_390_;
}
else
{
lean_object* v_val_438_; 
v_val_438_ = lean_ctor_get(v_fileName_386_, 0);
lean_inc(v_val_438_);
lean_dec_ref_known(v_fileName_386_, 1);
v___y_391_ = v_val_438_;
goto v___jp_390_;
}
v___jp_390_:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_392_ = lean_st_ref_get(v_a_388_);
v___x_393_ = lean_st_ref_get(v_a_388_);
v___x_394_ = l_Lean_Elab_Command_getScope___redArg(v_a_388_);
if (lean_obj_tag(v___x_394_) == 0)
{
lean_object* v_a_395_; lean_object* v___x_396_; 
v_a_395_ = lean_ctor_get(v___x_394_, 0);
lean_inc(v_a_395_);
lean_dec_ref_known(v___x_394_, 1);
v___x_396_ = l_Lean_Elab_Command_getScope___redArg(v_a_388_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_420_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_420_ == 0)
{
v___x_399_ = v___x_396_;
v_isShared_400_ = v_isSharedCheck_420_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_396_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_420_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v_env_401_; lean_object* v___x_402_; lean_object* v_scopes_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v_opts_406_; lean_object* v_currNamespace_407_; lean_object* v_openDecls_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_418_; 
v_env_401_ = lean_ctor_get(v___x_392_, 0);
lean_inc_ref_n(v_env_401_, 2);
lean_dec(v___x_392_);
v___x_402_ = lean_string_utf8_byte_size(v_docComment_385_);
v_scopes_403_ = lean_ctor_get(v___x_393_, 2);
lean_inc(v_scopes_403_);
lean_dec(v___x_393_);
v___x_404_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_405_ = l_List_head_x21___redArg(v___x_404_, v_scopes_403_);
lean_dec(v_scopes_403_);
v_opts_406_ = lean_ctor_get(v___x_405_, 1);
lean_inc_ref(v_opts_406_);
lean_dec(v___x_405_);
v_currNamespace_407_ = lean_ctor_get(v_a_395_, 2);
lean_inc(v_currNamespace_407_);
lean_dec(v_a_395_);
v_openDecls_408_ = lean_ctor_get(v_a_397_, 3);
lean_inc(v_openDecls_408_);
lean_dec(v_a_397_);
lean_inc_ref_n(v_docComment_385_, 2);
v___x_409_ = l_Lean_FileMap_ofString(v_docComment_385_);
v___x_410_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_410_, 0, v_docComment_385_);
lean_ctor_set(v___x_410_, 1, v___y_391_);
lean_ctor_set(v___x_410_, 2, v___x_409_);
lean_ctor_set(v___x_410_, 3, v___x_402_);
v___x_411_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_411_, 0, v_env_401_);
lean_ctor_set(v___x_411_, 1, v_opts_406_);
lean_ctor_set(v___x_411_, 2, v_currNamespace_407_);
lean_ctor_set(v___x_411_, 3, v_openDecls_408_);
v___x_412_ = l_Lean_Parser_mkParserState(v_docComment_385_);
lean_dec_ref(v_docComment_385_);
v___x_413_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___closed__2));
v___x_414_ = l_Lean_Parser_getTokenTable(v_env_401_);
v___x_415_ = l_Lean_Parser_ParserFn_run(v___x_413_, v___x_410_, v___x_411_, v___x_414_, v___x_412_);
v___x_416_ = l_Lean_Parser_ParserState_allErrors(v___x_415_);
if (v_isShared_400_ == 0)
{
lean_ctor_set(v___x_399_, 0, v___x_416_);
v___x_418_ = v___x_399_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___x_416_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
else
{
lean_object* v_a_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_428_; 
lean_dec(v_a_395_);
lean_dec(v___x_393_);
lean_dec(v___x_392_);
lean_dec_ref(v___y_391_);
lean_dec_ref(v_docComment_385_);
v_a_421_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_428_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_428_ == 0)
{
v___x_423_ = v___x_396_;
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_a_421_);
lean_dec(v___x_396_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_428_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v___x_426_; 
if (v_isShared_424_ == 0)
{
v___x_426_ = v___x_423_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v_a_421_);
v___x_426_ = v_reuseFailAlloc_427_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
return v___x_426_;
}
}
}
}
else
{
lean_object* v_a_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_436_; 
lean_dec(v___x_393_);
lean_dec(v___x_392_);
lean_dec_ref(v___y_391_);
lean_dec_ref(v_docComment_385_);
v_a_429_ = lean_ctor_get(v___x_394_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v___x_394_);
if (v_isSharedCheck_436_ == 0)
{
v___x_431_ = v___x_394_;
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_a_429_);
lean_dec(v___x_394_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_436_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v___x_434_; 
if (v_isShared_432_ == 0)
{
v___x_434_ = v___x_431_;
goto v_reusejp_433_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v_a_429_);
v___x_434_ = v_reuseFailAlloc_435_;
goto v_reusejp_433_;
}
v_reusejp_433_:
{
return v___x_434_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax___boxed(lean_object* v_docComment_439_, lean_object* v_fileName_440_, lean_object* v_a_441_, lean_object* v_a_442_, lean_object* v_a_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax(v_docComment_439_, v_fileName_440_, v_a_441_, v_a_442_);
lean_dec(v_a_442_);
lean_dec_ref(v_a_441_);
return v_res_444_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__0(void){
_start:
{
lean_object* v___x_445_; lean_object* v___f_446_; 
v___x_445_ = lean_alloc_closure((void*)(l_instDecidableEqString___boxed), 2, 0);
v___f_446_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_446_, 0, v___x_445_);
return v___f_446_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning(lean_object* v_err_448_){
_start:
{
lean_object* v_expected_449_; lean_object* v___f_450_; lean_object* v___x_451_; uint8_t v___x_452_; 
v_expected_449_ = lean_ctor_get(v_err_448_, 2);
lean_inc(v_expected_449_);
lean_dec_ref(v_err_448_);
v___f_450_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__0);
v___x_451_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___closed__1));
v___x_452_ = l_List_elem___redArg(v___f_450_, v___x_451_, v_expected_449_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning___boxed(lean_object* v_err_453_){
_start:
{
uint8_t v_res_454_; lean_object* v_r_455_; 
v_res_454_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning(v_err_453_);
v_r_455_ = lean_box(v_res_454_);
return v_r_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3(lean_object* v_s_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = ((lean_object*)(lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3___closed__0));
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3___boxed(lean_object* v_s_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3(v_s_460_);
lean_dec_ref(v_s_460_);
return v_res_461_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1(void){
_start:
{
lean_object* v___x_463_; lean_object* v___x_464_; 
v___x_463_ = ((lean_object*)(lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__0));
v___x_464_ = lean_string_utf8_byte_size(v___x_463_);
return v___x_464_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__2(void){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v___x_465_ = lean_unsigned_to_nat(0u);
v___x_466_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1);
v___x_467_ = lean_nat_dec_eq(v___x_466_, v___x_465_);
return v___x_467_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3(void){
_start:
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_468_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__1);
v___x_469_ = lean_unsigned_to_nat(0u);
v___x_470_ = ((lean_object*)(lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__0));
v___x_471_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_471_, 0, v___x_470_);
lean_ctor_set(v___x_471_, 1, v___x_469_);
lean_ctor_set(v___x_471_, 2, v___x_468_);
return v___x_471_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__4(void){
_start:
{
lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_472_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3);
v___x_473_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_472_);
return v___x_473_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__5(void){
_start:
{
lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_474_ = lean_unsigned_to_nat(0u);
v___x_475_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__4, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__4_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__4);
v___x_476_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__3);
v___x_477_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_477_, 0, v___x_476_);
lean_ctor_set(v___x_477_, 1, v___x_475_);
lean_ctor_set(v___x_477_, 2, v___x_474_);
lean_ctor_set(v___x_477_, 3, v___x_474_);
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__6(void){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_478_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__5, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__5_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__5);
v___x_479_ = lean_unsigned_to_nat(0u);
v___x_480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_480_, 0, v___x_479_);
lean_ctor_set(v___x_480_, 1, v___x_478_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5(lean_object* v_s_484_){
_start:
{
uint8_t v___x_485_; 
v___x_485_ = lean_uint8_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__2, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__2_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__2);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; 
v___x_486_ = lean_obj_once(&lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__6, &lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__6_once, _init_lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__6);
return v___x_486_;
}
else
{
lean_object* v___x_487_; 
v___x_487_ = ((lean_object*)(lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___closed__7));
return v___x_487_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5___boxed(lean_object* v_s_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5(v_s_488_);
lean_dec_ref(v_s_488_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7(lean_object* v_as_490_, size_t v_i_491_, size_t v_stop_492_, lean_object* v_b_493_){
_start:
{
lean_object* v___y_495_; uint8_t v___x_499_; 
v___x_499_ = lean_usize_dec_eq(v_i_491_, v_stop_492_);
if (v___x_499_ == 0)
{
lean_object* v___x_500_; lean_object* v_snd_501_; lean_object* v_snd_502_; uint8_t v___x_503_; 
v___x_500_ = lean_array_uget_borrowed(v_as_490_, v_i_491_);
v_snd_501_ = lean_ctor_get(v___x_500_, 1);
v_snd_502_ = lean_ctor_get(v_snd_501_, 1);
lean_inc(v_snd_502_);
v___x_503_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_isSilencedVersoWarning(v_snd_502_);
if (v___x_503_ == 0)
{
lean_object* v___x_504_; 
lean_inc(v___x_500_);
v___x_504_ = lean_array_push(v_b_493_, v___x_500_);
v___y_495_ = v___x_504_;
goto v___jp_494_;
}
else
{
v___y_495_ = v_b_493_;
goto v___jp_494_;
}
}
else
{
return v_b_493_;
}
v___jp_494_:
{
size_t v___x_496_; size_t v___x_497_; 
v___x_496_ = ((size_t)1ULL);
v___x_497_ = lean_usize_add(v_i_491_, v___x_496_);
v_i_491_ = v___x_497_;
v_b_493_ = v___y_495_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7___boxed(lean_object* v_as_505_, lean_object* v_i_506_, lean_object* v_stop_507_, lean_object* v_b_508_){
_start:
{
size_t v_i_boxed_509_; size_t v_stop_boxed_510_; lean_object* v_res_511_; 
v_i_boxed_509_ = lean_unbox_usize(v_i_506_);
lean_dec(v_i_506_);
v_stop_boxed_510_ = lean_unbox_usize(v_stop_507_);
lean_dec(v_stop_507_);
v_res_511_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7(v_as_505_, v_i_boxed_509_, v_stop_boxed_510_, v_b_508_);
lean_dec_ref(v_as_505_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg(lean_object* v___x_513_, lean_object* v_trimmedStr_514_, lean_object* v___x_515_, lean_object* v_a_516_, lean_object* v_b_517_){
_start:
{
lean_object* v_it_519_; lean_object* v_out_520_; lean_object* v_left_523_; lean_object* v_memoizedLeft_524_; lean_object* v_right_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_677_; 
v_left_523_ = lean_ctor_get(v_a_516_, 0);
v_memoizedLeft_524_ = lean_ctor_get(v_a_516_, 1);
v_right_525_ = lean_ctor_get(v_a_516_, 2);
v_isSharedCheck_677_ = !lean_is_exclusive(v_a_516_);
if (v_isSharedCheck_677_ == 0)
{
v___x_527_ = v_a_516_;
v_isShared_528_ = v_isSharedCheck_677_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_right_525_);
lean_inc(v_memoizedLeft_524_);
lean_inc(v_left_523_);
lean_dec(v_a_516_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_677_;
goto v_resetjp_526_;
}
v___jp_518_:
{
lean_object* v___x_521_; 
v___x_521_ = lean_string_append(v_b_517_, v_out_520_);
lean_dec_ref(v_out_520_);
v_a_516_ = v_it_519_;
v_b_517_ = v___x_521_;
goto _start;
}
v_resetjp_526_:
{
lean_object* v_it_530_; lean_object* v_startInclusive_531_; lean_object* v_endExclusive_532_; lean_object* v_it_540_; 
if (lean_obj_tag(v_memoizedLeft_524_) == 0)
{
if (lean_obj_tag(v_left_523_) == 0)
{
lean_object* v_currPos_543_; lean_object* v_searcher_544_; lean_object* v___x_546_; uint8_t v_isShared_547_; uint8_t v_isSharedCheck_646_; 
v_currPos_543_ = lean_ctor_get(v_left_523_, 0);
v_searcher_544_ = lean_ctor_get(v_left_523_, 1);
v_isSharedCheck_646_ = !lean_is_exclusive(v_left_523_);
if (v_isSharedCheck_646_ == 0)
{
v___x_546_ = v_left_523_;
v_isShared_547_ = v_isSharedCheck_646_;
goto v_resetjp_545_;
}
else
{
lean_inc(v_searcher_544_);
lean_inc(v_currPos_543_);
lean_dec(v_left_523_);
v___x_546_ = lean_box(0);
v_isShared_547_ = v_isSharedCheck_646_;
goto v_resetjp_545_;
}
v_resetjp_545_:
{
lean_object* v_it_549_; lean_object* v_it_554_; lean_object* v_startPos_555_; lean_object* v_endPos_556_; 
switch(lean_obj_tag(v_searcher_544_))
{
case 0:
{
lean_object* v_pos_569_; lean_object* v___x_571_; uint8_t v_isShared_572_; uint8_t v_isSharedCheck_581_; 
lean_del_object(v___x_546_);
v_pos_569_ = lean_ctor_get(v_searcher_544_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v_searcher_544_);
if (v_isSharedCheck_581_ == 0)
{
v___x_571_ = v_searcher_544_;
v_isShared_572_ = v_isSharedCheck_581_;
goto v_resetjp_570_;
}
else
{
lean_inc(v_pos_569_);
lean_dec(v_searcher_544_);
v___x_571_ = lean_box(0);
v_isShared_572_ = v_isSharedCheck_581_;
goto v_resetjp_570_;
}
v_resetjp_570_:
{
lean_object* v_startInclusive_573_; lean_object* v_endExclusive_574_; lean_object* v___x_575_; uint8_t v___x_576_; 
v_startInclusive_573_ = lean_ctor_get(v___x_513_, 1);
v_endExclusive_574_ = lean_ctor_get(v___x_513_, 2);
v___x_575_ = lean_nat_sub(v_endExclusive_574_, v_startInclusive_573_);
v___x_576_ = lean_nat_dec_eq(v_pos_569_, v___x_575_);
lean_dec(v___x_575_);
if (v___x_576_ == 0)
{
lean_object* v___x_578_; 
lean_inc(v_pos_569_);
if (v_isShared_572_ == 0)
{
lean_ctor_set_tag(v___x_571_, 1);
v___x_578_ = v___x_571_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v_pos_569_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
lean_inc(v_pos_569_);
v_it_554_ = v___x_578_;
v_startPos_555_ = v_pos_569_;
v_endPos_556_ = v_pos_569_;
goto v___jp_553_;
}
}
else
{
lean_object* v___x_580_; 
lean_del_object(v___x_571_);
v___x_580_ = lean_box(3);
lean_inc(v_pos_569_);
v_it_554_ = v___x_580_;
v_startPos_555_ = v_pos_569_;
v_endPos_556_ = v_pos_569_;
goto v___jp_553_;
}
}
}
case 1:
{
lean_object* v_pos_582_; lean_object* v___x_584_; uint8_t v_isShared_585_; uint8_t v_isSharedCheck_590_; 
lean_del_object(v___x_527_);
v_pos_582_ = lean_ctor_get(v_searcher_544_, 0);
v_isSharedCheck_590_ = !lean_is_exclusive(v_searcher_544_);
if (v_isSharedCheck_590_ == 0)
{
v___x_584_ = v_searcher_544_;
v_isShared_585_ = v_isSharedCheck_590_;
goto v_resetjp_583_;
}
else
{
lean_inc(v_pos_582_);
lean_dec(v_searcher_544_);
v___x_584_ = lean_box(0);
v_isShared_585_ = v_isSharedCheck_590_;
goto v_resetjp_583_;
}
v_resetjp_583_:
{
lean_object* v___x_586_; lean_object* v___x_588_; 
v___x_586_ = lean_string_utf8_next_fast(v_trimmedStr_514_, v_pos_582_);
lean_dec(v_pos_582_);
if (v_isShared_585_ == 0)
{
lean_ctor_set_tag(v___x_584_, 0);
lean_ctor_set(v___x_584_, 0, v___x_586_);
v___x_588_ = v___x_584_;
goto v_reusejp_587_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v___x_586_);
v___x_588_ = v_reuseFailAlloc_589_;
goto v_reusejp_587_;
}
v_reusejp_587_:
{
v_it_549_ = v___x_588_;
goto v___jp_548_;
}
}
}
case 2:
{
lean_object* v_needle_591_; lean_object* v_table_592_; lean_object* v_stackPos_593_; lean_object* v_needlePos_594_; lean_object* v___x_596_; uint8_t v_isShared_597_; uint8_t v_isSharedCheck_645_; 
v_needle_591_ = lean_ctor_get(v_searcher_544_, 0);
v_table_592_ = lean_ctor_get(v_searcher_544_, 1);
v_stackPos_593_ = lean_ctor_get(v_searcher_544_, 2);
v_needlePos_594_ = lean_ctor_get(v_searcher_544_, 3);
v_isSharedCheck_645_ = !lean_is_exclusive(v_searcher_544_);
if (v_isSharedCheck_645_ == 0)
{
v___x_596_ = v_searcher_544_;
v_isShared_597_ = v_isSharedCheck_645_;
goto v_resetjp_595_;
}
else
{
lean_inc(v_needlePos_594_);
lean_inc(v_stackPos_593_);
lean_inc(v_table_592_);
lean_inc(v_needle_591_);
lean_dec(v_searcher_544_);
v___x_596_ = lean_box(0);
v_isShared_597_ = v_isSharedCheck_645_;
goto v_resetjp_595_;
}
v_resetjp_595_:
{
lean_object* v_str_598_; lean_object* v_startInclusive_599_; lean_object* v_endExclusive_600_; lean_object* v_basePos_601_; lean_object* v___x_602_; lean_object* v___x_603_; uint8_t v___x_604_; 
v_str_598_ = lean_ctor_get(v_needle_591_, 0);
v_startInclusive_599_ = lean_ctor_get(v_needle_591_, 1);
v_endExclusive_600_ = lean_ctor_get(v_needle_591_, 2);
v_basePos_601_ = lean_nat_sub(v_stackPos_593_, v_needlePos_594_);
v___x_602_ = lean_nat_sub(v_endExclusive_600_, v_startInclusive_599_);
v___x_603_ = lean_nat_add(v_basePos_601_, v___x_602_);
v___x_604_ = lean_nat_dec_le(v___x_603_, v___x_515_);
lean_dec(v___x_603_);
if (v___x_604_ == 0)
{
uint8_t v___x_605_; 
lean_dec(v___x_602_);
lean_del_object(v___x_596_);
lean_dec(v_needlePos_594_);
lean_dec(v_stackPos_593_);
lean_dec_ref(v_table_592_);
lean_dec_ref(v_needle_591_);
v___x_605_ = lean_nat_dec_lt(v_basePos_601_, v___x_515_);
lean_dec(v_basePos_601_);
if (v___x_605_ == 0)
{
lean_del_object(v___x_546_);
goto v___jp_567_;
}
else
{
lean_object* v___x_606_; 
lean_del_object(v___x_527_);
v___x_606_ = lean_box(3);
v_it_549_ = v___x_606_;
goto v___jp_548_;
}
}
else
{
uint8_t v_stackByte_607_; lean_object* v___x_608_; uint8_t v_patByte_609_; uint8_t v___x_610_; 
lean_dec(v_basePos_601_);
lean_inc(v_stackPos_593_);
v_stackByte_607_ = lean_string_get_byte_fast(v_trimmedStr_514_, v_stackPos_593_);
v___x_608_ = lean_nat_add(v_startInclusive_599_, v_needlePos_594_);
v_patByte_609_ = lean_string_get_byte_fast(v_str_598_, v___x_608_);
v___x_610_ = lean_uint8_dec_eq(v_stackByte_607_, v_patByte_609_);
if (v___x_610_ == 0)
{
lean_object* v___x_611_; uint8_t v___x_612_; 
lean_dec(v___x_602_);
lean_del_object(v___x_527_);
v___x_611_ = lean_unsigned_to_nat(0u);
v___x_612_ = lean_nat_dec_eq(v_needlePos_594_, v___x_611_);
if (v___x_612_ == 0)
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v_newNeedlePos_615_; uint8_t v___x_616_; 
v___x_613_ = lean_unsigned_to_nat(1u);
v___x_614_ = lean_nat_sub(v_needlePos_594_, v___x_613_);
lean_dec(v_needlePos_594_);
v_newNeedlePos_615_ = lean_array_fget_borrowed(v_table_592_, v___x_614_);
lean_dec(v___x_614_);
v___x_616_ = lean_nat_dec_eq(v_newNeedlePos_615_, v___x_611_);
if (v___x_616_ == 0)
{
lean_object* v___x_618_; 
lean_inc(v_newNeedlePos_615_);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 3, v_newNeedlePos_615_);
v___x_618_ = v___x_596_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_needle_591_);
lean_ctor_set(v_reuseFailAlloc_619_, 1, v_table_592_);
lean_ctor_set(v_reuseFailAlloc_619_, 2, v_stackPos_593_);
lean_ctor_set(v_reuseFailAlloc_619_, 3, v_newNeedlePos_615_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
v_it_549_ = v___x_618_;
goto v___jp_548_;
}
}
else
{
lean_object* v_nextStackPos_620_; lean_object* v___x_622_; 
v_nextStackPos_620_ = l_String_Slice_posGE___redArg(v___x_513_, v_stackPos_593_);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 3, v___x_611_);
lean_ctor_set(v___x_596_, 2, v_nextStackPos_620_);
v___x_622_ = v___x_596_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_623_; 
v_reuseFailAlloc_623_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_623_, 0, v_needle_591_);
lean_ctor_set(v_reuseFailAlloc_623_, 1, v_table_592_);
lean_ctor_set(v_reuseFailAlloc_623_, 2, v_nextStackPos_620_);
lean_ctor_set(v_reuseFailAlloc_623_, 3, v___x_611_);
v___x_622_ = v_reuseFailAlloc_623_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
v_it_549_ = v___x_622_;
goto v___jp_548_;
}
}
}
else
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v_nextStackPos_626_; lean_object* v___x_628_; 
lean_dec(v_needlePos_594_);
v___x_624_ = lean_unsigned_to_nat(1u);
v___x_625_ = lean_nat_add(v_stackPos_593_, v___x_624_);
lean_dec(v_stackPos_593_);
v_nextStackPos_626_ = l_String_Slice_posGE___redArg(v___x_513_, v___x_625_);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 3, v___x_611_);
lean_ctor_set(v___x_596_, 2, v_nextStackPos_626_);
v___x_628_ = v___x_596_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_needle_591_);
lean_ctor_set(v_reuseFailAlloc_629_, 1, v_table_592_);
lean_ctor_set(v_reuseFailAlloc_629_, 2, v_nextStackPos_626_);
lean_ctor_set(v_reuseFailAlloc_629_, 3, v___x_611_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
v_it_549_ = v___x_628_;
goto v___jp_548_;
}
}
}
else
{
lean_object* v___x_630_; lean_object* v_nextStackPos_631_; lean_object* v_nextNeedlePos_632_; uint8_t v___x_633_; 
lean_del_object(v___x_546_);
v___x_630_ = lean_unsigned_to_nat(1u);
v_nextStackPos_631_ = lean_nat_add(v_stackPos_593_, v___x_630_);
lean_dec(v_stackPos_593_);
v_nextNeedlePos_632_ = lean_nat_add(v_needlePos_594_, v___x_630_);
lean_dec(v_needlePos_594_);
v___x_633_ = lean_nat_dec_eq(v_nextNeedlePos_632_, v___x_602_);
lean_dec(v___x_602_);
if (v___x_633_ == 0)
{
lean_object* v___x_635_; 
lean_del_object(v___x_527_);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 3, v_nextNeedlePos_632_);
lean_ctor_set(v___x_596_, 2, v_nextStackPos_631_);
v___x_635_ = v___x_596_;
goto v_reusejp_634_;
}
else
{
lean_object* v_reuseFailAlloc_637_; 
v_reuseFailAlloc_637_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_637_, 0, v_needle_591_);
lean_ctor_set(v_reuseFailAlloc_637_, 1, v_table_592_);
lean_ctor_set(v_reuseFailAlloc_637_, 2, v_nextStackPos_631_);
lean_ctor_set(v_reuseFailAlloc_637_, 3, v_nextNeedlePos_632_);
v___x_635_ = v_reuseFailAlloc_637_;
goto v_reusejp_634_;
}
v_reusejp_634_:
{
lean_object* v___x_636_; 
v___x_636_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_636_, 0, v_currPos_543_);
lean_ctor_set(v___x_636_, 1, v___x_635_);
v_it_540_ = v___x_636_;
goto v___jp_539_;
}
}
else
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_643_; 
v___x_638_ = lean_nat_sub(v_nextStackPos_631_, v_nextNeedlePos_632_);
lean_dec(v_nextNeedlePos_632_);
v___x_639_ = l_String_Slice_pos_x21(v___x_513_, v___x_638_);
lean_dec(v___x_638_);
v___x_640_ = l_String_Slice_pos_x21(v___x_513_, v_nextStackPos_631_);
v___x_641_ = lean_unsigned_to_nat(0u);
if (v_isShared_597_ == 0)
{
lean_ctor_set(v___x_596_, 3, v___x_641_);
lean_ctor_set(v___x_596_, 2, v_nextStackPos_631_);
v___x_643_ = v___x_596_;
goto v_reusejp_642_;
}
else
{
lean_object* v_reuseFailAlloc_644_; 
v_reuseFailAlloc_644_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_644_, 0, v_needle_591_);
lean_ctor_set(v_reuseFailAlloc_644_, 1, v_table_592_);
lean_ctor_set(v_reuseFailAlloc_644_, 2, v_nextStackPos_631_);
lean_ctor_set(v_reuseFailAlloc_644_, 3, v___x_641_);
v___x_643_ = v_reuseFailAlloc_644_;
goto v_reusejp_642_;
}
v_reusejp_642_:
{
v_it_554_ = v___x_643_;
v_startPos_555_ = v___x_639_;
v_endPos_556_ = v___x_640_;
goto v___jp_553_;
}
}
}
}
}
}
default: 
{
lean_del_object(v___x_546_);
goto v___jp_567_;
}
}
v___jp_548_:
{
lean_object* v___x_551_; 
if (v_isShared_547_ == 0)
{
lean_ctor_set(v___x_546_, 1, v_it_549_);
v___x_551_ = v___x_546_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v_currPos_543_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v_it_549_);
v___x_551_ = v_reuseFailAlloc_552_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
v_it_540_ = v___x_551_;
goto v___jp_539_;
}
}
v___jp_553_:
{
lean_object* v_slice_557_; lean_object* v_startInclusive_558_; lean_object* v_endExclusive_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_566_; 
v_slice_557_ = l_String_Slice_subslice_x21(v___x_513_, v_currPos_543_, v_startPos_555_);
v_startInclusive_558_ = lean_ctor_get(v_slice_557_, 0);
v_endExclusive_559_ = lean_ctor_get(v_slice_557_, 1);
v_isSharedCheck_566_ = !lean_is_exclusive(v_slice_557_);
if (v_isSharedCheck_566_ == 0)
{
v___x_561_ = v_slice_557_;
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_endExclusive_559_);
lean_inc(v_startInclusive_558_);
lean_dec(v_slice_557_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v_nextIt_564_; 
if (v_isShared_562_ == 0)
{
lean_ctor_set(v___x_561_, 1, v_it_554_);
lean_ctor_set(v___x_561_, 0, v_endPos_556_);
v_nextIt_564_ = v___x_561_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_endPos_556_);
lean_ctor_set(v_reuseFailAlloc_565_, 1, v_it_554_);
v_nextIt_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
v_it_530_ = v_nextIt_564_;
v_startInclusive_531_ = v_startInclusive_558_;
v_endExclusive_532_ = v_endExclusive_559_;
goto v___jp_529_;
}
}
}
v___jp_567_:
{
lean_object* v___x_568_; 
v___x_568_ = lean_box(1);
lean_inc(v___x_515_);
v_it_530_ = v___x_568_;
v_startInclusive_531_ = v_currPos_543_;
v_endExclusive_532_ = v___x_515_;
goto v___jp_529_;
}
}
}
else
{
lean_del_object(v___x_527_);
lean_dec(v_right_525_);
lean_dec(v___x_515_);
lean_dec_ref(v_trimmedStr_514_);
return v_b_517_;
}
}
else
{
lean_object* v_next_647_; 
lean_del_object(v___x_527_);
v_next_647_ = lean_ctor_get(v_right_525_, 0);
lean_inc(v_next_647_);
if (lean_obj_tag(v_next_647_) == 0)
{
lean_dec_ref_known(v_memoizedLeft_524_, 1);
lean_dec(v_right_525_);
lean_dec(v_left_523_);
lean_dec(v___x_515_);
lean_dec_ref(v_trimmedStr_514_);
return v_b_517_;
}
else
{
lean_object* v_val_648_; lean_object* v_upperBound_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_675_; 
v_val_648_ = lean_ctor_get(v_memoizedLeft_524_, 0);
lean_inc(v_val_648_);
lean_dec_ref_known(v_memoizedLeft_524_, 1);
v_upperBound_649_ = lean_ctor_get(v_right_525_, 1);
v_isSharedCheck_675_ = !lean_is_exclusive(v_right_525_);
if (v_isSharedCheck_675_ == 0)
{
lean_object* v_unused_676_; 
v_unused_676_ = lean_ctor_get(v_right_525_, 0);
lean_dec(v_unused_676_);
v___x_651_ = v_right_525_;
v_isShared_652_ = v_isSharedCheck_675_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_upperBound_649_);
lean_dec(v_right_525_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_675_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v_val_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_674_; 
v_val_653_ = lean_ctor_get(v_next_647_, 0);
v_isSharedCheck_674_ = !lean_is_exclusive(v_next_647_);
if (v_isSharedCheck_674_ == 0)
{
v___x_655_ = v_next_647_;
v_isShared_656_ = v_isSharedCheck_674_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_val_653_);
lean_dec(v_next_647_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_674_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
uint8_t v___x_657_; 
v___x_657_ = lean_nat_dec_lt(v_val_653_, v_upperBound_649_);
if (v___x_657_ == 0)
{
lean_del_object(v___x_655_);
lean_dec(v_val_653_);
lean_del_object(v___x_651_);
lean_dec(v_upperBound_649_);
lean_dec(v_val_648_);
lean_dec(v_left_523_);
lean_dec(v___x_515_);
lean_dec_ref(v_trimmedStr_514_);
return v_b_517_;
}
else
{
lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_661_; 
v___x_658_ = lean_unsigned_to_nat(1u);
v___x_659_ = lean_nat_add(v_val_653_, v___x_658_);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 0, v___x_659_);
v___x_661_ = v___x_655_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v___x_659_);
v___x_661_ = v_reuseFailAlloc_673_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
lean_object* v___x_663_; 
if (v_isShared_652_ == 0)
{
lean_ctor_set(v___x_651_, 0, v___x_661_);
v___x_663_ = v___x_651_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v___x_661_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_upperBound_649_);
v___x_663_ = v_reuseFailAlloc_672_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; uint8_t v___x_669_; 
v___x_664_ = lean_box(0);
v___x_665_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_665_, 0, v_left_523_);
lean_ctor_set(v___x_665_, 1, v___x_664_);
lean_ctor_set(v___x_665_, 2, v___x_663_);
v___x_666_ = lean_unsigned_to_nat(2u);
v___x_667_ = lean_nat_mod(v_val_653_, v___x_666_);
lean_dec(v_val_653_);
v___x_668_ = lean_unsigned_to_nat(0u);
v___x_669_ = lean_nat_dec_eq(v___x_667_, v___x_668_);
lean_dec(v___x_667_);
if (v___x_669_ == 0)
{
lean_object* v___x_670_; 
lean_dec(v_val_648_);
v___x_670_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg___closed__0));
v_it_519_ = v___x_665_;
v_out_520_ = v___x_670_;
goto v___jp_518_;
}
else
{
lean_object* v___x_671_; 
v___x_671_ = l_String_Slice_toString(v_val_648_);
lean_dec(v_val_648_);
v_it_519_ = v___x_665_;
v_out_520_ = v___x_671_;
goto v___jp_518_;
}
}
}
}
}
}
}
}
v___jp_529_:
{
lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_536_; 
lean_inc_ref(v_trimmedStr_514_);
v___x_533_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_533_, 0, v_trimmedStr_514_);
lean_ctor_set(v___x_533_, 1, v_startInclusive_531_);
lean_ctor_set(v___x_533_, 2, v_endExclusive_532_);
v___x_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 1, v___x_534_);
lean_ctor_set(v___x_527_, 0, v_it_530_);
v___x_536_ = v___x_527_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_538_; 
v_reuseFailAlloc_538_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_538_, 0, v_it_530_);
lean_ctor_set(v_reuseFailAlloc_538_, 1, v___x_534_);
lean_ctor_set(v_reuseFailAlloc_538_, 2, v_right_525_);
v___x_536_ = v_reuseFailAlloc_538_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
v_a_516_ = v___x_536_;
goto _start;
}
}
v___jp_539_:
{
lean_object* v___x_541_; 
v___x_541_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_541_, 0, v_it_540_);
lean_ctor_set(v___x_541_, 1, v_memoizedLeft_524_);
lean_ctor_set(v___x_541_, 2, v_right_525_);
v_a_516_ = v___x_541_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg___boxed(lean_object* v___x_678_, lean_object* v_trimmedStr_679_, lean_object* v___x_680_, lean_object* v_a_681_, lean_object* v_b_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg(v___x_678_, v_trimmedStr_679_, v___x_680_, v_a_681_, v_b_682_);
lean_dec_ref(v___x_678_);
return v_res_683_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(lean_object* v_s_684_, lean_object* v_a_685_, uint8_t v_b_686_){
_start:
{
uint8_t v___x_687_; 
v___x_687_ = 0;
switch(lean_obj_tag(v_a_685_))
{
case 0:
{
uint8_t v___x_688_; 
lean_dec_ref_known(v_a_685_, 1);
v___x_688_ = 1;
return v___x_688_;
}
case 1:
{
lean_object* v_pos_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_702_; 
v_pos_689_ = lean_ctor_get(v_a_685_, 0);
v_isSharedCheck_702_ = !lean_is_exclusive(v_a_685_);
if (v_isSharedCheck_702_ == 0)
{
v___x_691_ = v_a_685_;
v_isShared_692_ = v_isSharedCheck_702_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_pos_689_);
lean_dec(v_a_685_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_702_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v_str_693_; lean_object* v_startInclusive_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_699_; 
v_str_693_ = lean_ctor_get(v_s_684_, 0);
v_startInclusive_694_ = lean_ctor_get(v_s_684_, 1);
v___x_695_ = lean_nat_add(v_startInclusive_694_, v_pos_689_);
lean_dec(v_pos_689_);
v___x_696_ = lean_string_utf8_next_fast(v_str_693_, v___x_695_);
lean_dec(v___x_695_);
v___x_697_ = lean_nat_sub(v___x_696_, v_startInclusive_694_);
if (v_isShared_692_ == 0)
{
lean_ctor_set_tag(v___x_691_, 0);
lean_ctor_set(v___x_691_, 0, v___x_697_);
v___x_699_ = v___x_691_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v___x_697_);
v___x_699_ = v_reuseFailAlloc_701_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
v_a_685_ = v___x_699_;
v_b_686_ = v___x_687_;
goto _start;
}
}
}
case 2:
{
lean_object* v_needle_703_; lean_object* v_table_704_; lean_object* v_stackPos_705_; lean_object* v_needlePos_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_759_; 
v_needle_703_ = lean_ctor_get(v_a_685_, 0);
v_table_704_ = lean_ctor_get(v_a_685_, 1);
v_stackPos_705_ = lean_ctor_get(v_a_685_, 2);
v_needlePos_706_ = lean_ctor_get(v_a_685_, 3);
v_isSharedCheck_759_ = !lean_is_exclusive(v_a_685_);
if (v_isSharedCheck_759_ == 0)
{
v___x_708_ = v_a_685_;
v_isShared_709_ = v_isSharedCheck_759_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_needlePos_706_);
lean_inc(v_stackPos_705_);
lean_inc(v_table_704_);
lean_inc(v_needle_703_);
lean_dec(v_a_685_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_759_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
lean_object* v_str_710_; lean_object* v_startInclusive_711_; lean_object* v_endExclusive_712_; lean_object* v_str_713_; lean_object* v_startInclusive_714_; lean_object* v_endExclusive_715_; lean_object* v_basePos_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; uint8_t v___x_720_; 
v_str_710_ = lean_ctor_get(v_needle_703_, 0);
v_startInclusive_711_ = lean_ctor_get(v_needle_703_, 1);
v_endExclusive_712_ = lean_ctor_get(v_needle_703_, 2);
v_str_713_ = lean_ctor_get(v_s_684_, 0);
v_startInclusive_714_ = lean_ctor_get(v_s_684_, 1);
v_endExclusive_715_ = lean_ctor_get(v_s_684_, 2);
v_basePos_716_ = lean_nat_sub(v_stackPos_705_, v_needlePos_706_);
v___x_717_ = lean_nat_sub(v_endExclusive_712_, v_startInclusive_711_);
v___x_718_ = lean_nat_add(v_basePos_716_, v___x_717_);
v___x_719_ = lean_nat_sub(v_endExclusive_715_, v_startInclusive_714_);
v___x_720_ = lean_nat_dec_le(v___x_718_, v___x_719_);
lean_dec(v___x_718_);
if (v___x_720_ == 0)
{
uint8_t v___x_721_; 
lean_dec(v___x_717_);
lean_del_object(v___x_708_);
lean_dec(v_needlePos_706_);
lean_dec(v_stackPos_705_);
lean_dec_ref(v_table_704_);
lean_dec_ref(v_needle_703_);
v___x_721_ = lean_nat_dec_lt(v_basePos_716_, v___x_719_);
lean_dec(v___x_719_);
lean_dec(v_basePos_716_);
if (v___x_721_ == 0)
{
return v_b_686_;
}
else
{
lean_object* v___x_722_; 
v___x_722_ = lean_box(3);
v_a_685_ = v___x_722_;
v_b_686_ = v___x_687_;
goto _start;
}
}
else
{
lean_object* v___x_724_; uint8_t v_stackByte_725_; lean_object* v___x_726_; uint8_t v_patByte_727_; uint8_t v___x_728_; 
lean_dec(v___x_719_);
lean_dec(v_basePos_716_);
v___x_724_ = lean_nat_add(v_startInclusive_714_, v_stackPos_705_);
v_stackByte_725_ = lean_string_get_byte_fast(v_str_713_, v___x_724_);
v___x_726_ = lean_nat_add(v_startInclusive_711_, v_needlePos_706_);
v_patByte_727_ = lean_string_get_byte_fast(v_str_710_, v___x_726_);
v___x_728_ = lean_uint8_dec_eq(v_stackByte_725_, v_patByte_727_);
if (v___x_728_ == 0)
{
lean_object* v___x_729_; uint8_t v___x_730_; 
lean_dec(v___x_717_);
v___x_729_ = lean_unsigned_to_nat(0u);
v___x_730_ = lean_nat_dec_eq(v_needlePos_706_, v___x_729_);
if (v___x_730_ == 0)
{
lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v_newNeedlePos_733_; uint8_t v___x_734_; 
v___x_731_ = lean_unsigned_to_nat(1u);
v___x_732_ = lean_nat_sub(v_needlePos_706_, v___x_731_);
lean_dec(v_needlePos_706_);
v_newNeedlePos_733_ = lean_array_fget_borrowed(v_table_704_, v___x_732_);
lean_dec(v___x_732_);
v___x_734_ = lean_nat_dec_eq(v_newNeedlePos_733_, v___x_729_);
if (v___x_734_ == 0)
{
lean_object* v___x_736_; 
lean_inc(v_newNeedlePos_733_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 3, v_newNeedlePos_733_);
v___x_736_ = v___x_708_;
goto v_reusejp_735_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v_needle_703_);
lean_ctor_set(v_reuseFailAlloc_738_, 1, v_table_704_);
lean_ctor_set(v_reuseFailAlloc_738_, 2, v_stackPos_705_);
lean_ctor_set(v_reuseFailAlloc_738_, 3, v_newNeedlePos_733_);
v___x_736_ = v_reuseFailAlloc_738_;
goto v_reusejp_735_;
}
v_reusejp_735_:
{
v_a_685_ = v___x_736_;
v_b_686_ = v___x_687_;
goto _start;
}
}
else
{
lean_object* v_nextStackPos_739_; lean_object* v___x_741_; 
v_nextStackPos_739_ = l_String_Slice_posGE___redArg(v_s_684_, v_stackPos_705_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 3, v___x_729_);
lean_ctor_set(v___x_708_, 2, v_nextStackPos_739_);
v___x_741_ = v___x_708_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v_needle_703_);
lean_ctor_set(v_reuseFailAlloc_743_, 1, v_table_704_);
lean_ctor_set(v_reuseFailAlloc_743_, 2, v_nextStackPos_739_);
lean_ctor_set(v_reuseFailAlloc_743_, 3, v___x_729_);
v___x_741_ = v_reuseFailAlloc_743_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
v_a_685_ = v___x_741_;
v_b_686_ = v___x_687_;
goto _start;
}
}
}
else
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v_nextStackPos_746_; lean_object* v___x_748_; 
lean_dec(v_needlePos_706_);
v___x_744_ = lean_unsigned_to_nat(1u);
v___x_745_ = lean_nat_add(v_stackPos_705_, v___x_744_);
lean_dec(v_stackPos_705_);
v_nextStackPos_746_ = l_String_Slice_posGE___redArg(v_s_684_, v___x_745_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 3, v___x_729_);
lean_ctor_set(v___x_708_, 2, v_nextStackPos_746_);
v___x_748_ = v___x_708_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v_needle_703_);
lean_ctor_set(v_reuseFailAlloc_750_, 1, v_table_704_);
lean_ctor_set(v_reuseFailAlloc_750_, 2, v_nextStackPos_746_);
lean_ctor_set(v_reuseFailAlloc_750_, 3, v___x_729_);
v___x_748_ = v_reuseFailAlloc_750_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
v_a_685_ = v___x_748_;
v_b_686_ = v___x_687_;
goto _start;
}
}
}
else
{
lean_object* v___x_751_; lean_object* v_nextNeedlePos_752_; uint8_t v___x_753_; 
v___x_751_ = lean_unsigned_to_nat(1u);
v_nextNeedlePos_752_ = lean_nat_add(v_needlePos_706_, v___x_751_);
lean_dec(v_needlePos_706_);
v___x_753_ = lean_nat_dec_eq(v_nextNeedlePos_752_, v___x_717_);
lean_dec(v___x_717_);
if (v___x_753_ == 0)
{
lean_object* v_nextStackPos_754_; lean_object* v___x_756_; 
v_nextStackPos_754_ = lean_nat_add(v_stackPos_705_, v___x_751_);
lean_dec(v_stackPos_705_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 3, v_nextNeedlePos_752_);
lean_ctor_set(v___x_708_, 2, v_nextStackPos_754_);
v___x_756_ = v___x_708_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_758_; 
v_reuseFailAlloc_758_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_758_, 0, v_needle_703_);
lean_ctor_set(v_reuseFailAlloc_758_, 1, v_table_704_);
lean_ctor_set(v_reuseFailAlloc_758_, 2, v_nextStackPos_754_);
lean_ctor_set(v_reuseFailAlloc_758_, 3, v_nextNeedlePos_752_);
v___x_756_ = v_reuseFailAlloc_758_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
v_a_685_ = v___x_756_;
goto _start;
}
}
else
{
lean_dec(v_nextNeedlePos_752_);
lean_del_object(v___x_708_);
lean_dec(v_stackPos_705_);
lean_dec_ref(v_table_704_);
lean_dec_ref(v_needle_703_);
return v___x_753_;
}
}
}
}
}
default: 
{
return v_b_686_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg___boxed(lean_object* v_s_760_, lean_object* v_a_761_, lean_object* v_b_762_){
_start:
{
uint8_t v_b_boxed_763_; uint8_t v_res_764_; lean_object* v_r_765_; 
v_b_boxed_763_ = lean_unbox(v_b_762_);
v_res_764_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(v_s_760_, v_a_761_, v_b_boxed_763_);
lean_dec_ref(v_s_760_);
v_r_765_ = lean_box(v_res_764_);
return v_r_765_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1(void){
_start:
{
lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_767_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__0));
v___x_768_ = lean_string_utf8_byte_size(v___x_767_);
return v___x_768_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__2(void){
_start:
{
lean_object* v___x_769_; lean_object* v___x_770_; uint8_t v___x_771_; 
v___x_769_ = lean_unsigned_to_nat(0u);
v___x_770_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1);
v___x_771_ = lean_nat_dec_eq(v___x_770_, v___x_769_);
return v___x_771_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3(void){
_start:
{
lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; 
v___x_772_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__1);
v___x_773_ = lean_unsigned_to_nat(0u);
v___x_774_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__0));
v___x_775_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_775_, 0, v___x_774_);
lean_ctor_set(v___x_775_, 1, v___x_773_);
lean_ctor_set(v___x_775_, 2, v___x_772_);
return v___x_775_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__4(void){
_start:
{
lean_object* v___x_776_; lean_object* v___x_777_; 
v___x_776_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3);
v___x_777_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_776_);
return v___x_777_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__5(void){
_start:
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
v___x_778_ = lean_unsigned_to_nat(0u);
v___x_779_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__4, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__4);
v___x_780_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__3);
v___x_781_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_781_, 0, v___x_780_);
lean_ctor_set(v___x_781_, 1, v___x_779_);
lean_ctor_set(v___x_781_, 2, v___x_778_);
lean_ctor_set(v___x_781_, 3, v___x_778_);
return v___x_781_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1(lean_object* v_s_782_){
_start:
{
lean_object* v___y_784_; uint8_t v___x_787_; 
v___x_787_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__2, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__2);
if (v___x_787_ == 0)
{
lean_object* v___x_788_; 
v___x_788_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__5, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___closed__5);
v___y_784_ = v___x_788_;
goto v___jp_783_;
}
else
{
lean_object* v___x_789_; 
v___x_789_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1));
v___y_784_ = v___x_789_;
goto v___jp_783_;
}
v___jp_783_:
{
uint8_t v___x_785_; uint8_t v___x_786_; 
v___x_785_ = 0;
lean_inc(v___y_784_);
v___x_786_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(v_s_782_, v___y_784_, v___x_785_);
return v___x_786_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1___boxed(lean_object* v_s_790_){
_start:
{
uint8_t v_res_791_; lean_object* v_r_792_; 
v_res_791_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1(v_s_790_);
lean_dec_ref(v_s_790_);
v_r_792_ = lean_box(v_res_791_);
return v_r_792_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1(void){
_start:
{
lean_object* v___x_794_; lean_object* v___x_795_; 
v___x_794_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__0));
v___x_795_ = lean_string_utf8_byte_size(v___x_794_);
return v___x_795_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__2(void){
_start:
{
lean_object* v___x_796_; lean_object* v___x_797_; uint8_t v___x_798_; 
v___x_796_ = lean_unsigned_to_nat(0u);
v___x_797_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1);
v___x_798_ = lean_nat_dec_eq(v___x_797_, v___x_796_);
return v___x_798_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3(void){
_start:
{
lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_799_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__1);
v___x_800_ = lean_unsigned_to_nat(0u);
v___x_801_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__0));
v___x_802_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_802_, 0, v___x_801_);
lean_ctor_set(v___x_802_, 1, v___x_800_);
lean_ctor_set(v___x_802_, 2, v___x_799_);
return v___x_802_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__4(void){
_start:
{
lean_object* v___x_803_; lean_object* v___x_804_; 
v___x_803_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3);
v___x_804_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_803_);
return v___x_804_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__5(void){
_start:
{
lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; 
v___x_805_ = lean_unsigned_to_nat(0u);
v___x_806_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__4, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__4);
v___x_807_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__3);
v___x_808_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_808_, 0, v___x_807_);
lean_ctor_set(v___x_808_, 1, v___x_806_);
lean_ctor_set(v___x_808_, 2, v___x_805_);
lean_ctor_set(v___x_808_, 3, v___x_805_);
return v___x_808_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0(lean_object* v_s_809_){
_start:
{
lean_object* v___y_811_; uint8_t v___x_814_; 
v___x_814_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__2, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__2);
if (v___x_814_ == 0)
{
lean_object* v___x_815_; 
v___x_815_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__5, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___closed__5);
v___y_811_ = v___x_815_;
goto v___jp_810_;
}
else
{
lean_object* v___x_816_; 
v___x_816_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1));
v___y_811_ = v___x_816_;
goto v___jp_810_;
}
v___jp_810_:
{
uint8_t v___x_812_; uint8_t v___x_813_; 
v___x_812_ = 0;
lean_inc(v___y_811_);
v___x_813_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(v_s_809_, v___y_811_, v___x_812_);
return v___x_813_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0___boxed(lean_object* v_s_817_){
_start:
{
uint8_t v_res_818_; lean_object* v_r_819_; 
v_res_818_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0(v_s_817_);
lean_dec_ref(v_s_817_);
v_r_819_ = lean_box(v_res_818_);
return v_r_819_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1(void){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_821_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__0));
v___x_822_ = lean_string_utf8_byte_size(v___x_821_);
return v___x_822_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__2(void){
_start:
{
lean_object* v___x_823_; lean_object* v___x_824_; uint8_t v___x_825_; 
v___x_823_ = lean_unsigned_to_nat(0u);
v___x_824_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1);
v___x_825_ = lean_nat_dec_eq(v___x_824_, v___x_823_);
return v___x_825_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3(void){
_start:
{
lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_826_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__1);
v___x_827_ = lean_unsigned_to_nat(0u);
v___x_828_ = ((lean_object*)(lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__0));
v___x_829_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_829_, 0, v___x_828_);
lean_ctor_set(v___x_829_, 1, v___x_827_);
lean_ctor_set(v___x_829_, 2, v___x_826_);
return v___x_829_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__4(void){
_start:
{
lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_830_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3);
v___x_831_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_830_);
return v___x_831_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__5(void){
_start:
{
lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_832_ = lean_unsigned_to_nat(0u);
v___x_833_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__4, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__4_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__4);
v___x_834_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__3);
v___x_835_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_835_, 0, v___x_834_);
lean_ctor_set(v___x_835_, 1, v___x_833_);
lean_ctor_set(v___x_835_, 2, v___x_832_);
lean_ctor_set(v___x_835_, 3, v___x_832_);
return v___x_835_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2(lean_object* v_s_836_){
_start:
{
lean_object* v___y_838_; uint8_t v___x_841_; 
v___x_841_ = lean_uint8_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__2, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__2_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__2);
if (v___x_841_ == 0)
{
lean_object* v___x_842_; 
v___x_842_ = lean_obj_once(&lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__5, &lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__5_once, _init_lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___closed__5);
v___y_838_ = v___x_842_;
goto v___jp_837_;
}
else
{
lean_object* v___x_843_; 
v___x_843_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__1));
v___y_838_ = v___x_843_;
goto v___jp_837_;
}
v___jp_837_:
{
uint8_t v___x_839_; uint8_t v___x_840_; 
v___x_839_ = 0;
lean_inc(v___y_838_);
v___x_840_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(v_s_836_, v___y_838_, v___x_839_);
return v___x_840_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2___boxed(lean_object* v_s_844_){
_start:
{
uint8_t v_res_845_; lean_object* v_r_846_; 
v_res_845_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2(v_s_844_);
lean_dec_ref(v_s_844_);
v_r_846_ = lean_box(v_res_845_);
return v_r_846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg(lean_object* v___x_848_, lean_object* v___x_849_, lean_object* v_docComment_850_, lean_object* v___x_851_, lean_object* v_a_852_, lean_object* v_b_853_){
_start:
{
lean_object* v___y_855_; lean_object* v___y_856_; lean_object* v___y_860_; lean_object* v___y_861_; uint8_t v___y_862_; lean_object* v_it_868_; lean_object* v_out_869_; 
if (lean_obj_tag(v_a_852_) == 0)
{
lean_object* v_currPos_872_; lean_object* v_searcher_873_; lean_object* v___x_875_; uint8_t v_isShared_876_; uint8_t v_isSharedCheck_908_; 
v_currPos_872_ = lean_ctor_get(v_a_852_, 0);
v_searcher_873_ = lean_ctor_get(v_a_852_, 1);
v_isSharedCheck_908_ = !lean_is_exclusive(v_a_852_);
if (v_isSharedCheck_908_ == 0)
{
v___x_875_ = v_a_852_;
v_isShared_876_ = v_isSharedCheck_908_;
goto v_resetjp_874_;
}
else
{
lean_inc(v_searcher_873_);
lean_inc(v_currPos_872_);
lean_dec(v_a_852_);
v___x_875_ = lean_box(0);
v_isShared_876_ = v_isSharedCheck_908_;
goto v_resetjp_874_;
}
v_resetjp_874_:
{
uint8_t v___y_886_; lean_object* v_startInclusive_890_; lean_object* v_endExclusive_891_; lean_object* v___x_892_; uint8_t v___x_893_; 
v_startInclusive_890_ = lean_ctor_get(v___x_848_, 1);
v_endExclusive_891_ = lean_ctor_get(v___x_848_, 2);
v___x_892_ = lean_nat_sub(v_endExclusive_891_, v_startInclusive_890_);
v___x_893_ = lean_nat_dec_eq(v_searcher_873_, v___x_892_);
lean_dec(v___x_892_);
if (v___x_893_ == 0)
{
uint32_t v___x_894_; uint8_t v___y_896_; uint32_t v___x_901_; uint8_t v___x_902_; 
v___x_894_ = lean_string_utf8_get_fast(v_docComment_850_, v_searcher_873_);
v___x_901_ = 32;
v___x_902_ = lean_uint32_dec_eq(v___x_894_, v___x_901_);
if (v___x_902_ == 0)
{
uint32_t v___x_903_; uint8_t v___x_904_; 
v___x_903_ = 9;
v___x_904_ = lean_uint32_dec_eq(v___x_894_, v___x_903_);
v___y_896_ = v___x_904_;
goto v___jp_895_;
}
else
{
v___y_896_ = v___x_902_;
goto v___jp_895_;
}
v___jp_895_:
{
if (v___y_896_ == 0)
{
uint32_t v___x_897_; uint8_t v___x_898_; 
v___x_897_ = 13;
v___x_898_ = lean_uint32_dec_eq(v___x_894_, v___x_897_);
if (v___x_898_ == 0)
{
uint32_t v___x_899_; uint8_t v___x_900_; 
v___x_899_ = 10;
v___x_900_ = lean_uint32_dec_eq(v___x_894_, v___x_899_);
v___y_886_ = v___x_900_;
goto v___jp_885_;
}
else
{
v___y_886_ = v___x_898_;
goto v___jp_885_;
}
}
else
{
goto v___jp_877_;
}
}
}
else
{
uint8_t v___x_905_; 
lean_del_object(v___x_875_);
lean_dec(v_searcher_873_);
v___x_905_ = lean_nat_dec_eq(v_currPos_872_, v___x_849_);
if (v___x_905_ == 0)
{
lean_object* v_slice_906_; lean_object* v___x_907_; 
lean_inc(v___x_851_);
lean_inc_ref(v_docComment_850_);
v_slice_906_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_slice_906_, 0, v_docComment_850_);
lean_ctor_set(v_slice_906_, 1, v_currPos_872_);
lean_ctor_set(v_slice_906_, 2, v___x_851_);
v___x_907_ = lean_box(1);
v_it_868_ = v___x_907_;
v_out_869_ = v_slice_906_;
goto v___jp_867_;
}
else
{
lean_dec(v_currPos_872_);
lean_dec(v___x_851_);
lean_dec_ref(v_docComment_850_);
lean_dec_ref(v___x_848_);
return v_b_853_;
}
}
v___jp_877_:
{
lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v_slice_881_; lean_object* v_nextIt_883_; 
v___x_878_ = lean_string_utf8_next_fast(v_docComment_850_, v_searcher_873_);
v___x_879_ = lean_nat_sub(v___x_878_, v_searcher_873_);
v___x_880_ = lean_nat_add(v_searcher_873_, v___x_879_);
lean_dec(v___x_879_);
lean_dec(v_searcher_873_);
lean_inc_ref(v___x_848_);
v_slice_881_ = l_String_Slice_slice_x21(v___x_848_, v_currPos_872_, v___x_880_);
lean_dec(v_currPos_872_);
lean_inc(v___x_880_);
if (v_isShared_876_ == 0)
{
lean_ctor_set(v___x_875_, 1, v___x_880_);
lean_ctor_set(v___x_875_, 0, v___x_880_);
v_nextIt_883_ = v___x_875_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v___x_880_);
lean_ctor_set(v_reuseFailAlloc_884_, 1, v___x_880_);
v_nextIt_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
v_it_868_ = v_nextIt_883_;
v_out_869_ = v_slice_881_;
goto v___jp_867_;
}
}
v___jp_885_:
{
if (v___y_886_ == 0)
{
lean_object* v___x_887_; lean_object* v___x_888_; 
lean_del_object(v___x_875_);
v___x_887_ = lean_string_utf8_next_fast(v_docComment_850_, v_searcher_873_);
lean_dec(v_searcher_873_);
v___x_888_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_888_, 0, v_currPos_872_);
lean_ctor_set(v___x_888_, 1, v___x_887_);
v_a_852_ = v___x_888_;
goto _start;
}
else
{
goto v___jp_877_;
}
}
}
}
else
{
lean_dec(v___x_851_);
lean_dec_ref(v_docComment_850_);
lean_dec_ref(v___x_848_);
return v_b_853_;
}
v___jp_854_:
{
lean_object* v___x_857_; 
v___x_857_ = lean_string_append(v_b_853_, v___y_856_);
lean_dec_ref(v___y_856_);
v_a_852_ = v___y_855_;
v_b_853_ = v___x_857_;
goto _start;
}
v___jp_859_:
{
if (v___y_862_ == 0)
{
lean_object* v___x_863_; 
v___x_863_ = l_String_Slice_toString(v___y_861_);
lean_dec_ref(v___y_861_);
v___y_855_ = v___y_860_;
v___y_856_ = v___x_863_;
goto v___jp_854_;
}
else
{
uint8_t v___x_864_; 
v___x_864_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0(v___y_861_);
if (v___x_864_ == 0)
{
lean_object* v___x_865_; 
lean_dec_ref(v___y_861_);
v___x_865_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg___closed__0));
v___y_855_ = v___y_860_;
v___y_856_ = v___x_865_;
goto v___jp_854_;
}
else
{
lean_object* v___x_866_; 
v___x_866_ = l_String_Slice_toString(v___y_861_);
lean_dec_ref(v___y_861_);
v___y_855_ = v___y_860_;
v___y_856_ = v___x_866_;
goto v___jp_854_;
}
}
}
v___jp_867_:
{
uint8_t v___x_870_; 
v___x_870_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__1(v_out_869_);
if (v___x_870_ == 0)
{
uint8_t v___x_871_; 
v___x_871_ = lp_mathlib_String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__2(v_out_869_);
v___y_860_ = v_it_868_;
v___y_861_ = v_out_869_;
v___y_862_ = v___x_871_;
goto v___jp_859_;
}
else
{
v___y_860_ = v_it_868_;
v___y_861_ = v_out_869_;
v___y_862_ = v___x_870_;
goto v___jp_859_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg___boxed(lean_object* v___x_909_, lean_object* v___x_910_, lean_object* v_docComment_911_, lean_object* v___x_912_, lean_object* v_a_913_, lean_object* v_b_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg(v___x_909_, v___x_910_, v_docComment_911_, v___x_912_, v_a_913_, v_b_914_);
lean_dec(v___x_910_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax(lean_object* v_docComment_920_, lean_object* v_fileName_921_, lean_object* v_a_922_, lean_object* v_a_923_){
_start:
{
lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v_trimmedStr_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v_trimmedStr_939_; lean_object* v___x_940_; 
v___x_925_ = lean_unsigned_to_nat(0u);
v___x_926_ = lean_string_utf8_byte_size(v_docComment_920_);
lean_inc_ref_n(v_docComment_920_, 2);
v___x_927_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_927_, 0, v_docComment_920_);
lean_ctor_set(v___x_927_, 1, v___x_925_);
lean_ctor_set(v___x_927_, 2, v___x_926_);
v___x_928_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0));
v___x_929_ = lp_mathlib_String_Slice_splitInclusive___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__3(v___x_927_);
v_trimmedStr_930_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg(v___x_927_, v___x_926_, v_docComment_920_, v___x_926_, v___x_929_, v___x_928_);
v___x_931_ = lean_string_utf8_byte_size(v_trimmedStr_930_);
lean_inc_ref(v_trimmedStr_930_);
v___x_932_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_932_, 0, v_trimmedStr_930_);
lean_ctor_set(v___x_932_, 1, v___x_925_);
lean_ctor_set(v___x_932_, 2, v___x_931_);
v___x_933_ = lp_mathlib_String_Slice_splitToSubslice___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__5(v___x_932_);
v___x_934_ = lean_string_length(v_docComment_920_);
lean_dec_ref(v_docComment_920_);
v___x_935_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__0));
v___x_936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_936_, 0, v___x_935_);
lean_ctor_set(v___x_936_, 1, v___x_934_);
v___x_937_ = lean_box(0);
v___x_938_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_938_, 0, v___x_933_);
lean_ctor_set(v___x_938_, 1, v___x_937_);
lean_ctor_set(v___x_938_, 2, v___x_936_);
v_trimmedStr_939_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg(v___x_932_, v_trimmedStr_930_, v___x_931_, v___x_938_, v___x_928_);
lean_dec_ref_known(v___x_932_, 3);
v___x_940_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_checkVersoSyntax(v_trimmedStr_939_, v_fileName_921_, v_a_922_, v_a_923_);
if (lean_obj_tag(v___x_940_) == 0)
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_967_; 
v_a_941_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_967_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_967_ == 0)
{
v___x_943_ = v___x_940_;
v_isShared_944_ = v_isSharedCheck_967_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_940_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_967_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_945_; lean_object* v___x_946_; uint8_t v___x_947_; 
v___x_945_ = lean_array_get_size(v_a_941_);
v___x_946_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___closed__1));
v___x_947_ = lean_nat_dec_lt(v___x_925_, v___x_945_);
if (v___x_947_ == 0)
{
lean_object* v___x_949_; 
lean_dec(v_a_941_);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 0, v___x_946_);
v___x_949_ = v___x_943_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_950_; 
v_reuseFailAlloc_950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_950_, 0, v___x_946_);
v___x_949_ = v_reuseFailAlloc_950_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
return v___x_949_;
}
}
else
{
uint8_t v___x_951_; 
v___x_951_ = lean_nat_dec_le(v___x_945_, v___x_945_);
if (v___x_951_ == 0)
{
if (v___x_947_ == 0)
{
lean_object* v___x_953_; 
lean_dec(v_a_941_);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 0, v___x_946_);
v___x_953_ = v___x_943_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v___x_946_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
return v___x_953_;
}
}
else
{
size_t v___x_955_; size_t v___x_956_; lean_object* v___x_957_; lean_object* v___x_959_; 
v___x_955_ = ((size_t)0ULL);
v___x_956_ = lean_usize_of_nat(v___x_945_);
v___x_957_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7(v_a_941_, v___x_955_, v___x_956_, v___x_946_);
lean_dec(v_a_941_);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 0, v___x_957_);
v___x_959_ = v___x_943_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v___x_957_);
v___x_959_ = v_reuseFailAlloc_960_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
return v___x_959_;
}
}
}
else
{
size_t v___x_961_; size_t v___x_962_; lean_object* v___x_963_; lean_object* v___x_965_; 
v___x_961_ = ((size_t)0ULL);
v___x_962_ = lean_usize_of_nat(v___x_945_);
v___x_963_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__7(v_a_941_, v___x_961_, v___x_962_, v___x_946_);
lean_dec(v_a_941_);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 0, v___x_963_);
v___x_965_ = v___x_943_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v___x_963_);
v___x_965_ = v_reuseFailAlloc_966_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
return v___x_965_;
}
}
}
}
}
else
{
return v___x_940_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax___boxed(lean_object* v_docComment_968_, lean_object* v_fileName_969_, lean_object* v_a_970_, lean_object* v_a_971_, lean_object* v_a_972_){
_start:
{
lean_object* v_res_973_; 
v_res_973_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax(v_docComment_968_, v_fileName_969_, v_a_970_, v_a_971_);
lean_dec(v_a_971_);
lean_dec_ref(v_a_970_);
return v_res_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4(lean_object* v___x_974_, lean_object* v___x_975_, lean_object* v_docComment_976_, lean_object* v___x_977_, lean_object* v_inst_978_, lean_object* v_R_979_, lean_object* v_a_980_, lean_object* v_b_981_, lean_object* v_c_982_){
_start:
{
lean_object* v___x_983_; 
v___x_983_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___redArg(v___x_974_, v___x_975_, v_docComment_976_, v___x_977_, v_a_980_, v_b_981_);
return v___x_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4___boxed(lean_object* v___x_984_, lean_object* v___x_985_, lean_object* v_docComment_986_, lean_object* v___x_987_, lean_object* v_inst_988_, lean_object* v_R_989_, lean_object* v_a_990_, lean_object* v_b_991_, lean_object* v_c_992_){
_start:
{
lean_object* v_res_993_; 
v_res_993_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__4(v___x_984_, v___x_985_, v_docComment_986_, v___x_987_, v_inst_988_, v_R_989_, v_a_990_, v_b_991_, v_c_992_);
lean_dec(v___x_985_);
return v_res_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6(lean_object* v___x_994_, lean_object* v_trimmedStr_995_, lean_object* v___x_996_, lean_object* v_inst_997_, lean_object* v_R_998_, lean_object* v_a_999_, lean_object* v_b_1000_, lean_object* v_c_1001_){
_start:
{
lean_object* v___x_1002_; 
v___x_1002_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___redArg(v___x_994_, v_trimmedStr_995_, v___x_996_, v_a_999_, v_b_1000_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6___boxed(lean_object* v___x_1003_, lean_object* v_trimmedStr_1004_, lean_object* v___x_1005_, lean_object* v_inst_1006_, lean_object* v_R_1007_, lean_object* v_a_1008_, lean_object* v_b_1009_, lean_object* v_c_1010_){
_start:
{
lean_object* v_res_1011_; 
v_res_1011_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__6(v___x_1003_, v_trimmedStr_1004_, v___x_1005_, v_inst_1006_, v_R_1007_, v_a_1008_, v_b_1009_, v_c_1010_);
lean_dec_ref(v___x_1003_);
return v_res_1011_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0(lean_object* v_s_1012_, lean_object* v_inst_1013_, lean_object* v_R_1014_, lean_object* v_a_1015_, uint8_t v_b_1016_, lean_object* v_c_1017_){
_start:
{
uint8_t v___x_1018_; 
v___x_1018_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___redArg(v_s_1012_, v_a_1015_, v_b_1016_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0___boxed(lean_object* v_s_1019_, lean_object* v_inst_1020_, lean_object* v_R_1021_, lean_object* v_a_1022_, lean_object* v_b_1023_, lean_object* v_c_1024_){
_start:
{
uint8_t v_b_boxed_1025_; uint8_t v_res_1026_; lean_object* v_r_1027_; 
v_b_boxed_1025_ = lean_unbox(v_b_1023_);
v_res_1026_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax_spec__0_spec__0(v_s_1019_, v_inst_1020_, v_R_1021_, v_a_1022_, v_b_boxed_1025_, v_c_1024_);
lean_dec_ref(v_s_1019_);
v_r_1027_ = lean_box(v_res_1026_);
return v_r_1027_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(lean_object* v_opts_1028_, lean_object* v_opt_1029_){
_start:
{
lean_object* v_name_1030_; lean_object* v_defValue_1031_; lean_object* v_map_1032_; lean_object* v___x_1033_; 
v_name_1030_ = lean_ctor_get(v_opt_1029_, 0);
v_defValue_1031_ = lean_ctor_get(v_opt_1029_, 1);
v_map_1032_ = lean_ctor_get(v_opts_1028_, 0);
v___x_1033_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1032_, v_name_1030_);
if (lean_obj_tag(v___x_1033_) == 0)
{
uint8_t v___x_1034_; 
v___x_1034_ = lean_unbox(v_defValue_1031_);
return v___x_1034_;
}
else
{
lean_object* v_val_1035_; 
v_val_1035_ = lean_ctor_get(v___x_1033_, 0);
lean_inc(v_val_1035_);
lean_dec_ref_known(v___x_1033_, 1);
if (lean_obj_tag(v_val_1035_) == 1)
{
uint8_t v_v_1036_; 
v_v_1036_ = lean_ctor_get_uint8(v_val_1035_, 0);
lean_dec_ref_known(v_val_1035_, 0);
return v_v_1036_;
}
else
{
uint8_t v___x_1037_; 
lean_dec(v_val_1035_);
v___x_1037_ = lean_unbox(v_defValue_1031_);
return v___x_1037_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4___boxed(lean_object* v_opts_1038_, lean_object* v_opt_1039_){
_start:
{
uint8_t v_res_1040_; lean_object* v_r_1041_; 
v_res_1040_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(v_opts_1038_, v_opt_1039_);
lean_dec_ref(v_opt_1039_);
lean_dec_ref(v_opts_1038_);
v_r_1041_ = lean_box(v_res_1040_);
return v_r_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__5(lean_object* v_msg_1042_){
_start:
{
lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1043_ = lean_unsigned_to_nat(0u);
v___x_1044_ = lean_panic_fn_borrowed(v___x_1043_, v_msg_1042_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg(lean_object* v_o_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v___x_1048_; lean_object* v_env_1049_; lean_object* v___x_1050_; lean_object* v_toEnvExtension_1051_; lean_object* v_asyncMode_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v_merged_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1064_; 
v___x_1048_ = lean_st_ref_get(v___y_1046_);
v_env_1049_ = lean_ctor_get(v___x_1048_, 0);
lean_inc_ref(v_env_1049_);
lean_dec(v___x_1048_);
v___x_1050_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1051_ = lean_ctor_get(v___x_1050_, 0);
v_asyncMode_1052_ = lean_ctor_get(v_toEnvExtension_1051_, 2);
v___x_1053_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1054_ = lean_box(0);
v___x_1055_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1053_, v___x_1050_, v_env_1049_, v_asyncMode_1052_, v___x_1054_);
v_merged_1056_ = lean_ctor_get(v___x_1055_, 0);
v_isSharedCheck_1064_ = !lean_is_exclusive(v___x_1055_);
if (v_isSharedCheck_1064_ == 0)
{
lean_object* v_unused_1065_; 
v_unused_1065_ = lean_ctor_get(v___x_1055_, 1);
lean_dec(v_unused_1065_);
v___x_1058_ = v___x_1055_;
v_isShared_1059_ = v_isSharedCheck_1064_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_merged_1056_);
lean_dec(v___x_1055_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1064_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
lean_object* v___x_1061_; 
if (v_isShared_1059_ == 0)
{
lean_ctor_set(v___x_1058_, 1, v_merged_1056_);
lean_ctor_set(v___x_1058_, 0, v_o_1045_);
v___x_1061_ = v___x_1058_;
goto v_reusejp_1060_;
}
else
{
lean_object* v_reuseFailAlloc_1063_; 
v_reuseFailAlloc_1063_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1063_, 0, v_o_1045_);
lean_ctor_set(v_reuseFailAlloc_1063_, 1, v_merged_1056_);
v___x_1061_ = v_reuseFailAlloc_1063_;
goto v_reusejp_1060_;
}
v_reusejp_1060_:
{
lean_object* v___x_1062_; 
v___x_1062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1061_);
return v___x_1062_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
lean_object* v_res_1069_; 
v_res_1069_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg(v_o_1066_, v___y_1067_);
lean_dec(v___y_1067_);
return v_res_1069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(lean_object* v___y_1070_, lean_object* v___y_1071_){
_start:
{
lean_object* v___x_1073_; lean_object* v_scopes_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v_opts_1077_; lean_object* v___x_1078_; 
v___x_1073_ = lean_st_ref_get(v___y_1071_);
v_scopes_1074_ = lean_ctor_get(v___x_1073_, 2);
lean_inc(v_scopes_1074_);
lean_dec(v___x_1073_);
v___x_1075_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1076_ = l_List_head_x21___redArg(v___x_1075_, v_scopes_1074_);
lean_dec(v_scopes_1074_);
v_opts_1077_ = lean_ctor_get(v___x_1076_, 1);
lean_inc_ref(v_opts_1077_);
lean_dec(v___x_1076_);
v___x_1078_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg(v_opts_1077_, v___y_1071_);
return v___x_1078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0___boxed(lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_){
_start:
{
lean_object* v_res_1082_; 
v_res_1082_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_1079_, v___y_1080_);
lean_dec(v___y_1080_);
lean_dec_ref(v___y_1079_);
return v_res_1082_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__0(void){
_start:
{
lean_object* v___x_1083_; 
v___x_1083_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1083_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1(void){
_start:
{
lean_object* v___x_1084_; lean_object* v___x_1085_; 
v___x_1084_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__0);
v___x_1085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1085_, 0, v___x_1084_);
return v___x_1085_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__2(void){
_start:
{
lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; 
v___x_1086_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1);
v___x_1087_ = lean_unsigned_to_nat(0u);
v___x_1088_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1088_, 0, v___x_1087_);
lean_ctor_set(v___x_1088_, 1, v___x_1087_);
lean_ctor_set(v___x_1088_, 2, v___x_1087_);
lean_ctor_set(v___x_1088_, 3, v___x_1087_);
lean_ctor_set(v___x_1088_, 4, v___x_1086_);
lean_ctor_set(v___x_1088_, 5, v___x_1086_);
lean_ctor_set(v___x_1088_, 6, v___x_1086_);
lean_ctor_set(v___x_1088_, 7, v___x_1086_);
lean_ctor_set(v___x_1088_, 8, v___x_1086_);
lean_ctor_set(v___x_1088_, 9, v___x_1086_);
return v___x_1088_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__3(void){
_start:
{
lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; 
v___x_1089_ = lean_unsigned_to_nat(32u);
v___x_1090_ = lean_mk_empty_array_with_capacity(v___x_1089_);
v___x_1091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1091_, 0, v___x_1090_);
return v___x_1091_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__4(void){
_start:
{
size_t v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; 
v___x_1092_ = ((size_t)5ULL);
v___x_1093_ = lean_unsigned_to_nat(0u);
v___x_1094_ = lean_unsigned_to_nat(32u);
v___x_1095_ = lean_mk_empty_array_with_capacity(v___x_1094_);
v___x_1096_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__3);
v___x_1097_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1097_, 0, v___x_1096_);
lean_ctor_set(v___x_1097_, 1, v___x_1095_);
lean_ctor_set(v___x_1097_, 2, v___x_1093_);
lean_ctor_set(v___x_1097_, 3, v___x_1093_);
lean_ctor_set_usize(v___x_1097_, 4, v___x_1092_);
return v___x_1097_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__5(void){
_start:
{
lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; 
v___x_1098_ = lean_box(1);
v___x_1099_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__4);
v___x_1100_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__1);
v___x_1101_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1101_, 0, v___x_1100_);
lean_ctor_set(v___x_1101_, 1, v___x_1099_);
lean_ctor_set(v___x_1101_, 2, v___x_1098_);
return v___x_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg(lean_object* v_msgData_1102_, lean_object* v___y_1103_){
_start:
{
lean_object* v___x_1105_; lean_object* v_env_1106_; lean_object* v___x_1107_; lean_object* v_scopes_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v_opts_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; 
v___x_1105_ = lean_st_ref_get(v___y_1103_);
v_env_1106_ = lean_ctor_get(v___x_1105_, 0);
lean_inc_ref(v_env_1106_);
lean_dec(v___x_1105_);
v___x_1107_ = lean_st_ref_get(v___y_1103_);
v_scopes_1108_ = lean_ctor_get(v___x_1107_, 2);
lean_inc(v_scopes_1108_);
lean_dec(v___x_1107_);
v___x_1109_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1110_ = l_List_head_x21___redArg(v___x_1109_, v_scopes_1108_);
lean_dec(v_scopes_1108_);
v_opts_1111_ = lean_ctor_get(v___x_1110_, 1);
lean_inc_ref(v_opts_1111_);
lean_dec(v___x_1110_);
v___x_1112_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__2);
v___x_1113_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___closed__5);
v___x_1114_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1114_, 0, v_env_1106_);
lean_ctor_set(v___x_1114_, 1, v___x_1112_);
lean_ctor_set(v___x_1114_, 2, v___x_1113_);
lean_ctor_set(v___x_1114_, 3, v_opts_1111_);
v___x_1115_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1115_, 0, v___x_1114_);
lean_ctor_set(v___x_1115_, 1, v_msgData_1102_);
v___x_1116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1116_, 0, v___x_1115_);
return v___x_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg___boxed(lean_object* v_msgData_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_){
_start:
{
lean_object* v_res_1120_; 
v_res_1120_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg(v_msgData_1117_, v___y_1118_);
lean_dec(v___y_1118_);
return v_res_1120_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0(uint8_t v___y_1122_, uint8_t v_suppressElabErrors_1123_, lean_object* v_x_1124_){
_start:
{
if (lean_obj_tag(v_x_1124_) == 1)
{
lean_object* v_pre_1125_; 
v_pre_1125_ = lean_ctor_get(v_x_1124_, 0);
if (lean_obj_tag(v_pre_1125_) == 0)
{
lean_object* v_str_1126_; lean_object* v___x_1127_; uint8_t v___x_1128_; 
v_str_1126_ = lean_ctor_get(v_x_1124_, 1);
v___x_1127_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___closed__0));
v___x_1128_ = lean_string_dec_eq(v_str_1126_, v___x_1127_);
if (v___x_1128_ == 0)
{
return v___y_1122_;
}
else
{
return v_suppressElabErrors_1123_;
}
}
else
{
return v___y_1122_;
}
}
else
{
return v___y_1122_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___boxed(lean_object* v___y_1129_, lean_object* v_suppressElabErrors_1130_, lean_object* v_x_1131_){
_start:
{
uint8_t v___y_21043__boxed_1132_; uint8_t v_suppressElabErrors_boxed_1133_; uint8_t v_res_1134_; lean_object* v_r_1135_; 
v___y_21043__boxed_1132_ = lean_unbox(v___y_1129_);
v_suppressElabErrors_boxed_1133_ = lean_unbox(v_suppressElabErrors_1130_);
v_res_1134_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0(v___y_21043__boxed_1132_, v_suppressElabErrors_boxed_1133_, v_x_1131_);
lean_dec(v_x_1131_);
v_r_1135_ = lean_box(v_res_1134_);
return v_r_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5(lean_object* v_ref_1136_, lean_object* v_msgData_1137_, uint8_t v_severity_1138_, uint8_t v_isSilent_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
lean_object* v___y_1144_; uint8_t v___y_1145_; lean_object* v___y_1146_; lean_object* v___y_1147_; lean_object* v___y_1148_; uint8_t v___y_1149_; lean_object* v___y_1150_; lean_object* v___y_1151_; uint8_t v___y_1208_; lean_object* v___y_1209_; uint8_t v___y_1210_; uint8_t v___y_1211_; lean_object* v___y_1212_; uint8_t v___y_1236_; uint8_t v___y_1237_; lean_object* v___y_1238_; uint8_t v___y_1239_; lean_object* v___y_1240_; uint8_t v___y_1244_; uint8_t v___y_1245_; uint8_t v___y_1246_; uint8_t v___x_1261_; uint8_t v___y_1263_; uint8_t v___y_1264_; uint8_t v___y_1265_; uint8_t v___y_1267_; uint8_t v___x_1279_; 
v___x_1261_ = 2;
v___x_1279_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1138_, v___x_1261_);
if (v___x_1279_ == 0)
{
v___y_1267_ = v___x_1279_;
goto v___jp_1266_;
}
else
{
uint8_t v___x_1280_; 
lean_inc_ref(v_msgData_1137_);
v___x_1280_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1137_);
v___y_1267_ = v___x_1280_;
goto v___jp_1266_;
}
v___jp_1143_:
{
lean_object* v___x_1152_; 
v___x_1152_ = l_Lean_Elab_Command_getScope___redArg(v___y_1151_);
if (lean_obj_tag(v___x_1152_) == 0)
{
lean_object* v_a_1153_; lean_object* v___x_1154_; 
v_a_1153_ = lean_ctor_get(v___x_1152_, 0);
lean_inc(v_a_1153_);
lean_dec_ref_known(v___x_1152_, 1);
v___x_1154_ = l_Lean_Elab_Command_getScope___redArg(v___y_1151_);
if (lean_obj_tag(v___x_1154_) == 0)
{
lean_object* v_a_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1190_; 
v_a_1155_ = lean_ctor_get(v___x_1154_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___x_1154_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1157_ = v___x_1154_;
v_isShared_1158_ = v_isSharedCheck_1190_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_a_1155_);
lean_dec(v___x_1154_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1190_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1159_; lean_object* v_currNamespace_1160_; lean_object* v_openDecls_1161_; lean_object* v_env_1162_; lean_object* v_messages_1163_; lean_object* v_scopes_1164_; lean_object* v_usedQuotCtxts_1165_; lean_object* v_nextMacroScope_1166_; lean_object* v_maxRecDepth_1167_; lean_object* v_ngen_1168_; lean_object* v_auxDeclNGen_1169_; lean_object* v_infoState_1170_; lean_object* v_traceState_1171_; lean_object* v_snapshotTasks_1172_; lean_object* v_prevLinterStates_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1189_; 
v___x_1159_ = lean_st_ref_take(v___y_1151_);
v_currNamespace_1160_ = lean_ctor_get(v_a_1153_, 2);
lean_inc(v_currNamespace_1160_);
lean_dec(v_a_1153_);
v_openDecls_1161_ = lean_ctor_get(v_a_1155_, 3);
lean_inc(v_openDecls_1161_);
lean_dec(v_a_1155_);
v_env_1162_ = lean_ctor_get(v___x_1159_, 0);
v_messages_1163_ = lean_ctor_get(v___x_1159_, 1);
v_scopes_1164_ = lean_ctor_get(v___x_1159_, 2);
v_usedQuotCtxts_1165_ = lean_ctor_get(v___x_1159_, 3);
v_nextMacroScope_1166_ = lean_ctor_get(v___x_1159_, 4);
v_maxRecDepth_1167_ = lean_ctor_get(v___x_1159_, 5);
v_ngen_1168_ = lean_ctor_get(v___x_1159_, 6);
v_auxDeclNGen_1169_ = lean_ctor_get(v___x_1159_, 7);
v_infoState_1170_ = lean_ctor_get(v___x_1159_, 8);
v_traceState_1171_ = lean_ctor_get(v___x_1159_, 9);
v_snapshotTasks_1172_ = lean_ctor_get(v___x_1159_, 10);
v_prevLinterStates_1173_ = lean_ctor_get(v___x_1159_, 11);
v_isSharedCheck_1189_ = !lean_is_exclusive(v___x_1159_);
if (v_isSharedCheck_1189_ == 0)
{
v___x_1175_ = v___x_1159_;
v_isShared_1176_ = v_isSharedCheck_1189_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_prevLinterStates_1173_);
lean_inc(v_snapshotTasks_1172_);
lean_inc(v_traceState_1171_);
lean_inc(v_infoState_1170_);
lean_inc(v_auxDeclNGen_1169_);
lean_inc(v_ngen_1168_);
lean_inc(v_maxRecDepth_1167_);
lean_inc(v_nextMacroScope_1166_);
lean_inc(v_usedQuotCtxts_1165_);
lean_inc(v_scopes_1164_);
lean_inc(v_messages_1163_);
lean_inc(v_env_1162_);
lean_dec(v___x_1159_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1189_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1182_; 
v___x_1177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1177_, 0, v_currNamespace_1160_);
lean_ctor_set(v___x_1177_, 1, v_openDecls_1161_);
v___x_1178_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1177_);
lean_ctor_set(v___x_1178_, 1, v___y_1150_);
lean_inc_ref(v___y_1148_);
lean_inc_ref(v___y_1144_);
v___x_1179_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1179_, 0, v___y_1144_);
lean_ctor_set(v___x_1179_, 1, v___y_1146_);
lean_ctor_set(v___x_1179_, 2, v___y_1147_);
lean_ctor_set(v___x_1179_, 3, v___y_1148_);
lean_ctor_set(v___x_1179_, 4, v___x_1178_);
lean_ctor_set_uint8(v___x_1179_, sizeof(void*)*5, v___y_1149_);
lean_ctor_set_uint8(v___x_1179_, sizeof(void*)*5 + 1, v___y_1145_);
lean_ctor_set_uint8(v___x_1179_, sizeof(void*)*5 + 2, v_isSilent_1139_);
v___x_1180_ = l_Lean_MessageLog_add(v___x_1179_, v_messages_1163_);
if (v_isShared_1176_ == 0)
{
lean_ctor_set(v___x_1175_, 1, v___x_1180_);
v___x_1182_ = v___x_1175_;
goto v_reusejp_1181_;
}
else
{
lean_object* v_reuseFailAlloc_1188_; 
v_reuseFailAlloc_1188_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1188_, 0, v_env_1162_);
lean_ctor_set(v_reuseFailAlloc_1188_, 1, v___x_1180_);
lean_ctor_set(v_reuseFailAlloc_1188_, 2, v_scopes_1164_);
lean_ctor_set(v_reuseFailAlloc_1188_, 3, v_usedQuotCtxts_1165_);
lean_ctor_set(v_reuseFailAlloc_1188_, 4, v_nextMacroScope_1166_);
lean_ctor_set(v_reuseFailAlloc_1188_, 5, v_maxRecDepth_1167_);
lean_ctor_set(v_reuseFailAlloc_1188_, 6, v_ngen_1168_);
lean_ctor_set(v_reuseFailAlloc_1188_, 7, v_auxDeclNGen_1169_);
lean_ctor_set(v_reuseFailAlloc_1188_, 8, v_infoState_1170_);
lean_ctor_set(v_reuseFailAlloc_1188_, 9, v_traceState_1171_);
lean_ctor_set(v_reuseFailAlloc_1188_, 10, v_snapshotTasks_1172_);
lean_ctor_set(v_reuseFailAlloc_1188_, 11, v_prevLinterStates_1173_);
v___x_1182_ = v_reuseFailAlloc_1188_;
goto v_reusejp_1181_;
}
v_reusejp_1181_:
{
lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1186_; 
v___x_1183_ = lean_st_ref_set(v___y_1151_, v___x_1182_);
v___x_1184_ = lean_box(0);
if (v_isShared_1158_ == 0)
{
lean_ctor_set(v___x_1157_, 0, v___x_1184_);
v___x_1186_ = v___x_1157_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1187_; 
v_reuseFailAlloc_1187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1187_, 0, v___x_1184_);
v___x_1186_ = v_reuseFailAlloc_1187_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
return v___x_1186_;
}
}
}
}
}
else
{
lean_object* v_a_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1198_; 
lean_dec(v_a_1153_);
lean_dec_ref(v___y_1150_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
v_a_1191_ = lean_ctor_get(v___x_1154_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1154_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1193_ = v___x_1154_;
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_a_1191_);
lean_dec(v___x_1154_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1196_; 
if (v_isShared_1194_ == 0)
{
v___x_1196_ = v___x_1193_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1191_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
return v___x_1196_;
}
}
}
}
else
{
lean_object* v_a_1199_; lean_object* v___x_1201_; uint8_t v_isShared_1202_; uint8_t v_isSharedCheck_1206_; 
lean_dec_ref(v___y_1150_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
v_a_1199_ = lean_ctor_get(v___x_1152_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v___x_1152_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1201_ = v___x_1152_;
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
else
{
lean_inc(v_a_1199_);
lean_dec(v___x_1152_);
v___x_1201_ = lean_box(0);
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
v_resetjp_1200_:
{
lean_object* v___x_1204_; 
if (v_isShared_1202_ == 0)
{
v___x_1204_ = v___x_1201_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v_a_1199_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
}
}
v___jp_1207_:
{
lean_object* v_fileName_1213_; lean_object* v_fileMap_1214_; uint8_t v_suppressElabErrors_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v_a_1218_; lean_object* v___x_1220_; uint8_t v_isShared_1221_; uint8_t v_isSharedCheck_1234_; 
v_fileName_1213_ = lean_ctor_get(v___y_1140_, 0);
v_fileMap_1214_ = lean_ctor_get(v___y_1140_, 1);
v_suppressElabErrors_1215_ = lean_ctor_get_uint8(v___y_1140_, sizeof(void*)*10);
v___x_1216_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1137_);
v___x_1217_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg(v___x_1216_, v___y_1141_);
v_a_1218_ = lean_ctor_get(v___x_1217_, 0);
v_isSharedCheck_1234_ = !lean_is_exclusive(v___x_1217_);
if (v_isSharedCheck_1234_ == 0)
{
v___x_1220_ = v___x_1217_;
v_isShared_1221_ = v_isSharedCheck_1234_;
goto v_resetjp_1219_;
}
else
{
lean_inc(v_a_1218_);
lean_dec(v___x_1217_);
v___x_1220_ = lean_box(0);
v_isShared_1221_ = v_isSharedCheck_1234_;
goto v_resetjp_1219_;
}
v_resetjp_1219_:
{
lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; 
lean_inc_ref_n(v_fileMap_1214_, 2);
v___x_1222_ = l_Lean_FileMap_toPosition(v_fileMap_1214_, v___y_1209_);
lean_dec(v___y_1209_);
v___x_1223_ = l_Lean_FileMap_toPosition(v_fileMap_1214_, v___y_1212_);
lean_dec(v___y_1212_);
v___x_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1224_, 0, v___x_1223_);
v___x_1225_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0));
if (v_suppressElabErrors_1215_ == 0)
{
lean_del_object(v___x_1220_);
v___y_1144_ = v_fileName_1213_;
v___y_1145_ = v___y_1210_;
v___y_1146_ = v___x_1222_;
v___y_1147_ = v___x_1224_;
v___y_1148_ = v___x_1225_;
v___y_1149_ = v___y_1211_;
v___y_1150_ = v_a_1218_;
v___y_1151_ = v___y_1141_;
goto v___jp_1143_;
}
else
{
lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___f_1228_; uint8_t v___x_1229_; 
v___x_1226_ = lean_box(v___y_1208_);
v___x_1227_ = lean_box(v_suppressElabErrors_1215_);
v___f_1228_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1228_, 0, v___x_1226_);
lean_closure_set(v___f_1228_, 1, v___x_1227_);
lean_inc(v_a_1218_);
v___x_1229_ = l_Lean_MessageData_hasTag(v___f_1228_, v_a_1218_);
if (v___x_1229_ == 0)
{
lean_object* v___x_1230_; lean_object* v___x_1232_; 
lean_dec_ref_known(v___x_1224_, 1);
lean_dec_ref(v___x_1222_);
lean_dec(v_a_1218_);
v___x_1230_ = lean_box(0);
if (v_isShared_1221_ == 0)
{
lean_ctor_set(v___x_1220_, 0, v___x_1230_);
v___x_1232_ = v___x_1220_;
goto v_reusejp_1231_;
}
else
{
lean_object* v_reuseFailAlloc_1233_; 
v_reuseFailAlloc_1233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1233_, 0, v___x_1230_);
v___x_1232_ = v_reuseFailAlloc_1233_;
goto v_reusejp_1231_;
}
v_reusejp_1231_:
{
return v___x_1232_;
}
}
else
{
lean_del_object(v___x_1220_);
v___y_1144_ = v_fileName_1213_;
v___y_1145_ = v___y_1210_;
v___y_1146_ = v___x_1222_;
v___y_1147_ = v___x_1224_;
v___y_1148_ = v___x_1225_;
v___y_1149_ = v___y_1211_;
v___y_1150_ = v_a_1218_;
v___y_1151_ = v___y_1141_;
goto v___jp_1143_;
}
}
}
}
v___jp_1235_:
{
lean_object* v___x_1241_; 
v___x_1241_ = l_Lean_Syntax_getTailPos_x3f(v___y_1238_, v___y_1239_);
lean_dec(v___y_1238_);
if (lean_obj_tag(v___x_1241_) == 0)
{
lean_inc(v___y_1240_);
v___y_1208_ = v___y_1236_;
v___y_1209_ = v___y_1240_;
v___y_1210_ = v___y_1237_;
v___y_1211_ = v___y_1239_;
v___y_1212_ = v___y_1240_;
goto v___jp_1207_;
}
else
{
lean_object* v_val_1242_; 
v_val_1242_ = lean_ctor_get(v___x_1241_, 0);
lean_inc(v_val_1242_);
lean_dec_ref_known(v___x_1241_, 1);
v___y_1208_ = v___y_1236_;
v___y_1209_ = v___y_1240_;
v___y_1210_ = v___y_1237_;
v___y_1211_ = v___y_1239_;
v___y_1212_ = v_val_1242_;
goto v___jp_1207_;
}
}
v___jp_1243_:
{
lean_object* v___x_1247_; 
v___x_1247_ = l_Lean_Elab_Command_getRef___redArg(v___y_1140_);
if (lean_obj_tag(v___x_1247_) == 0)
{
lean_object* v_a_1248_; lean_object* v_ref_1249_; lean_object* v___x_1250_; 
v_a_1248_ = lean_ctor_get(v___x_1247_, 0);
lean_inc(v_a_1248_);
lean_dec_ref_known(v___x_1247_, 1);
v_ref_1249_ = l_Lean_replaceRef(v_ref_1136_, v_a_1248_);
lean_dec(v_a_1248_);
v___x_1250_ = l_Lean_Syntax_getPos_x3f(v_ref_1249_, v___y_1245_);
if (lean_obj_tag(v___x_1250_) == 0)
{
lean_object* v___x_1251_; 
v___x_1251_ = lean_unsigned_to_nat(0u);
v___y_1236_ = v___y_1244_;
v___y_1237_ = v___y_1246_;
v___y_1238_ = v_ref_1249_;
v___y_1239_ = v___y_1245_;
v___y_1240_ = v___x_1251_;
goto v___jp_1235_;
}
else
{
lean_object* v_val_1252_; 
v_val_1252_ = lean_ctor_get(v___x_1250_, 0);
lean_inc(v_val_1252_);
lean_dec_ref_known(v___x_1250_, 1);
v___y_1236_ = v___y_1244_;
v___y_1237_ = v___y_1246_;
v___y_1238_ = v_ref_1249_;
v___y_1239_ = v___y_1245_;
v___y_1240_ = v_val_1252_;
goto v___jp_1235_;
}
}
else
{
lean_object* v_a_1253_; lean_object* v___x_1255_; uint8_t v_isShared_1256_; uint8_t v_isSharedCheck_1260_; 
lean_dec_ref(v_msgData_1137_);
v_a_1253_ = lean_ctor_get(v___x_1247_, 0);
v_isSharedCheck_1260_ = !lean_is_exclusive(v___x_1247_);
if (v_isSharedCheck_1260_ == 0)
{
v___x_1255_ = v___x_1247_;
v_isShared_1256_ = v_isSharedCheck_1260_;
goto v_resetjp_1254_;
}
else
{
lean_inc(v_a_1253_);
lean_dec(v___x_1247_);
v___x_1255_ = lean_box(0);
v_isShared_1256_ = v_isSharedCheck_1260_;
goto v_resetjp_1254_;
}
v_resetjp_1254_:
{
lean_object* v___x_1258_; 
if (v_isShared_1256_ == 0)
{
v___x_1258_ = v___x_1255_;
goto v_reusejp_1257_;
}
else
{
lean_object* v_reuseFailAlloc_1259_; 
v_reuseFailAlloc_1259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1259_, 0, v_a_1253_);
v___x_1258_ = v_reuseFailAlloc_1259_;
goto v_reusejp_1257_;
}
v_reusejp_1257_:
{
return v___x_1258_;
}
}
}
}
v___jp_1262_:
{
if (v___y_1265_ == 0)
{
v___y_1244_ = v___y_1263_;
v___y_1245_ = v___y_1264_;
v___y_1246_ = v_severity_1138_;
goto v___jp_1243_;
}
else
{
v___y_1244_ = v___y_1263_;
v___y_1245_ = v___y_1264_;
v___y_1246_ = v___x_1261_;
goto v___jp_1243_;
}
}
v___jp_1266_:
{
if (v___y_1267_ == 0)
{
lean_object* v___x_1268_; lean_object* v_scopes_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v_opts_1272_; uint8_t v___x_1273_; uint8_t v___x_1274_; 
v___x_1268_ = lean_st_ref_get(v___y_1141_);
v_scopes_1269_ = lean_ctor_get(v___x_1268_, 2);
lean_inc(v_scopes_1269_);
lean_dec(v___x_1268_);
v___x_1270_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1271_ = l_List_head_x21___redArg(v___x_1270_, v_scopes_1269_);
lean_dec(v_scopes_1269_);
v_opts_1272_ = lean_ctor_get(v___x_1271_, 1);
lean_inc_ref(v_opts_1272_);
lean_dec(v___x_1271_);
v___x_1273_ = 1;
v___x_1274_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1138_, v___x_1273_);
if (v___x_1274_ == 0)
{
lean_dec_ref(v_opts_1272_);
v___y_1263_ = v___y_1267_;
v___y_1264_ = v___y_1267_;
v___y_1265_ = v___x_1274_;
goto v___jp_1262_;
}
else
{
lean_object* v___x_1275_; uint8_t v___x_1276_; 
v___x_1275_ = l_Lean_warningAsError;
v___x_1276_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(v_opts_1272_, v___x_1275_);
lean_dec_ref(v_opts_1272_);
v___y_1263_ = v___y_1267_;
v___y_1264_ = v___y_1267_;
v___y_1265_ = v___x_1276_;
goto v___jp_1262_;
}
}
else
{
lean_object* v___x_1277_; lean_object* v___x_1278_; 
lean_dec_ref(v_msgData_1137_);
v___x_1277_ = lean_box(0);
v___x_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1277_);
return v___x_1278_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5___boxed(lean_object* v_ref_1281_, lean_object* v_msgData_1282_, lean_object* v_severity_1283_, lean_object* v_isSilent_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_){
_start:
{
uint8_t v_severity_boxed_1288_; uint8_t v_isSilent_boxed_1289_; lean_object* v_res_1290_; 
v_severity_boxed_1288_ = lean_unbox(v_severity_1283_);
v_isSilent_boxed_1289_ = lean_unbox(v_isSilent_1284_);
v_res_1290_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5(v_ref_1281_, v_msgData_1282_, v_severity_boxed_1288_, v_isSilent_boxed_1289_, v___y_1285_, v___y_1286_);
lean_dec(v___y_1286_);
lean_dec_ref(v___y_1285_);
lean_dec(v_ref_1281_);
return v_res_1290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2(lean_object* v_ref_1291_, lean_object* v_msgData_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_){
_start:
{
uint8_t v___x_1296_; uint8_t v___x_1297_; lean_object* v___x_1298_; 
v___x_1296_ = 1;
v___x_1297_ = 0;
v___x_1298_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5(v_ref_1291_, v_msgData_1292_, v___x_1296_, v___x_1297_, v___y_1293_, v___y_1294_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2___boxed(lean_object* v_ref_1299_, lean_object* v_msgData_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_){
_start:
{
lean_object* v_res_1304_; 
v_res_1304_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2(v_ref_1299_, v_msgData_1300_, v___y_1301_, v___y_1302_);
lean_dec(v___y_1302_);
lean_dec_ref(v___y_1301_);
lean_dec(v_ref_1299_);
return v_res_1304_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1306_; lean_object* v___x_1307_; 
v___x_1306_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__0));
v___x_1307_ = l_Lean_stringToMessageData(v___x_1306_);
return v___x_1307_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1309_; lean_object* v___x_1310_; 
v___x_1309_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__2));
v___x_1310_ = l_Lean_stringToMessageData(v___x_1309_);
return v___x_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(lean_object* v_linterOption_1311_, lean_object* v_stx_1312_, lean_object* v_msg_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_){
_start:
{
lean_object* v_name_1317_; lean_object* v___x_1319_; uint8_t v_isShared_1320_; uint8_t v_isSharedCheck_1335_; 
v_name_1317_ = lean_ctor_get(v_linterOption_1311_, 0);
v_isSharedCheck_1335_ = !lean_is_exclusive(v_linterOption_1311_);
if (v_isSharedCheck_1335_ == 0)
{
lean_object* v_unused_1336_; 
v_unused_1336_ = lean_ctor_get(v_linterOption_1311_, 1);
lean_dec(v_unused_1336_);
v___x_1319_ = v_linterOption_1311_;
v_isShared_1320_ = v_isSharedCheck_1335_;
goto v_resetjp_1318_;
}
else
{
lean_inc(v_name_1317_);
lean_dec(v_linterOption_1311_);
v___x_1319_ = lean_box(0);
v_isShared_1320_ = v_isSharedCheck_1335_;
goto v_resetjp_1318_;
}
v_resetjp_1318_:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1324_; 
v___x_1321_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__1);
lean_inc(v_name_1317_);
v___x_1322_ = l_Lean_MessageData_ofName(v_name_1317_);
if (v_isShared_1320_ == 0)
{
lean_ctor_set_tag(v___x_1319_, 7);
lean_ctor_set(v___x_1319_, 1, v___x_1322_);
lean_ctor_set(v___x_1319_, 0, v___x_1321_);
v___x_1324_ = v___x_1319_;
goto v_reusejp_1323_;
}
else
{
lean_object* v_reuseFailAlloc_1334_; 
v_reuseFailAlloc_1334_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1334_, 0, v___x_1321_);
lean_ctor_set(v_reuseFailAlloc_1334_, 1, v___x_1322_);
v___x_1324_ = v_reuseFailAlloc_1334_;
goto v_reusejp_1323_;
}
v_reusejp_1323_:
{
lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v_disable_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; 
v___x_1325_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___closed__3);
v___x_1326_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1326_, 0, v___x_1324_);
lean_ctor_set(v___x_1326_, 1, v___x_1325_);
v_disable_1327_ = l_Lean_MessageData_note(v___x_1326_);
v___x_1328_ = l_Lean_Linter_linterMessageTag;
v___x_1329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1329_, 0, v_msg_1313_);
lean_ctor_set(v___x_1329_, 1, v_disable_1327_);
v___x_1330_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1328_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
v___x_1331_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1331_, 0, v_name_1317_);
lean_ctor_set(v___x_1331_, 1, v___x_1330_);
lean_inc(v_stx_1312_);
v___x_1332_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1332_, 0, v_stx_1312_);
lean_ctor_set(v___x_1332_, 1, v___x_1331_);
v___x_1333_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2(v_stx_1312_, v___x_1332_, v___y_1314_, v___y_1315_);
lean_dec(v_stx_1312_);
return v___x_1333_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1___boxed(lean_object* v_linterOption_1337_, lean_object* v_stx_1338_, lean_object* v_msg_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_){
_start:
{
lean_object* v_res_1343_; 
v_res_1343_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(v_linterOption_1337_, v_stx_1338_, v_msg_1339_, v___y_1340_, v___y_1341_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
return v_res_1343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3_spec__6(lean_object* v_as_1344_, size_t v_sz_1345_, size_t v_i_1346_, lean_object* v_b_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_){
_start:
{
uint8_t v___x_1351_; 
v___x_1351_ = lean_usize_dec_lt(v_i_1346_, v_sz_1345_);
if (v___x_1351_ == 0)
{
lean_object* v___x_1352_; 
v___x_1352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1352_, 0, v_b_1347_);
return v___x_1352_;
}
else
{
lean_object* v_a_1353_; lean_object* v_snd_1354_; lean_object* v_fst_1355_; lean_object* v_snd_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; 
v_a_1353_ = lean_array_uget_borrowed(v_as_1344_, v_i_1346_);
v_snd_1354_ = lean_ctor_get(v_a_1353_, 1);
v_fst_1355_ = lean_ctor_get(v_snd_1354_, 0);
v_snd_1356_ = lean_ctor_get(v_snd_1354_, 1);
v___x_1357_ = lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
v___x_1358_ = l_Lean_Parser_SyntaxStack_back(v_fst_1355_);
lean_inc(v_snd_1356_);
v___x_1359_ = l_Lean_Parser_Error_toString(v_snd_1356_);
v___x_1360_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1360_, 0, v___x_1359_);
v___x_1361_ = l_Lean_MessageData_ofFormat(v___x_1360_);
v___x_1362_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(v___x_1357_, v___x_1358_, v___x_1361_, v___y_1348_, v___y_1349_);
if (lean_obj_tag(v___x_1362_) == 0)
{
lean_object* v___x_1363_; size_t v___x_1364_; size_t v___x_1365_; 
lean_dec_ref_known(v___x_1362_, 1);
v___x_1363_ = lean_box(0);
v___x_1364_ = ((size_t)1ULL);
v___x_1365_ = lean_usize_add(v_i_1346_, v___x_1364_);
v_i_1346_ = v___x_1365_;
v_b_1347_ = v___x_1363_;
goto _start;
}
else
{
return v___x_1362_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3_spec__6___boxed(lean_object* v_as_1367_, lean_object* v_sz_1368_, lean_object* v_i_1369_, lean_object* v_b_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_){
_start:
{
size_t v_sz_boxed_1374_; size_t v_i_boxed_1375_; lean_object* v_res_1376_; 
v_sz_boxed_1374_ = lean_unbox_usize(v_sz_1368_);
lean_dec(v_sz_1368_);
v_i_boxed_1375_ = lean_unbox_usize(v_i_1369_);
lean_dec(v_i_1369_);
v_res_1376_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3_spec__6(v_as_1367_, v_sz_boxed_1374_, v_i_boxed_1375_, v_b_1370_, v___y_1371_, v___y_1372_);
lean_dec(v___y_1372_);
lean_dec_ref(v___y_1371_);
lean_dec_ref(v_as_1367_);
return v_res_1376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3(lean_object* v_as_1377_, size_t v_sz_1378_, size_t v_i_1379_, lean_object* v_b_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_){
_start:
{
uint8_t v___x_1384_; 
v___x_1384_ = lean_usize_dec_lt(v_i_1379_, v_sz_1378_);
if (v___x_1384_ == 0)
{
lean_object* v___x_1385_; 
v___x_1385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1385_, 0, v_b_1380_);
return v___x_1385_;
}
else
{
lean_object* v_a_1386_; lean_object* v_snd_1387_; lean_object* v_fst_1388_; lean_object* v_snd_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; 
v_a_1386_ = lean_array_uget_borrowed(v_as_1377_, v_i_1379_);
v_snd_1387_ = lean_ctor_get(v_a_1386_, 1);
v_fst_1388_ = lean_ctor_get(v_snd_1387_, 0);
v_snd_1389_ = lean_ctor_get(v_snd_1387_, 1);
v___x_1390_ = lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
v___x_1391_ = l_Lean_Parser_SyntaxStack_back(v_fst_1388_);
lean_inc(v_snd_1389_);
v___x_1392_ = l_Lean_Parser_Error_toString(v_snd_1389_);
v___x_1393_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1393_, 0, v___x_1392_);
v___x_1394_ = l_Lean_MessageData_ofFormat(v___x_1393_);
v___x_1395_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(v___x_1390_, v___x_1391_, v___x_1394_, v___y_1381_, v___y_1382_);
if (lean_obj_tag(v___x_1395_) == 0)
{
lean_object* v___x_1396_; size_t v___x_1397_; size_t v___x_1398_; lean_object* v___x_1399_; 
lean_dec_ref_known(v___x_1395_, 1);
v___x_1396_ = lean_box(0);
v___x_1397_ = ((size_t)1ULL);
v___x_1398_ = lean_usize_add(v_i_1379_, v___x_1397_);
v___x_1399_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3_spec__6(v_as_1377_, v_sz_1378_, v___x_1398_, v___x_1396_, v___y_1381_, v___y_1382_);
return v___x_1399_;
}
else
{
return v___x_1395_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3___boxed(lean_object* v_as_1400_, lean_object* v_sz_1401_, lean_object* v_i_1402_, lean_object* v_b_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_){
_start:
{
size_t v_sz_boxed_1407_; size_t v_i_boxed_1408_; lean_object* v_res_1409_; 
v_sz_boxed_1407_ = lean_unbox_usize(v_sz_1401_);
lean_dec(v_sz_1401_);
v_i_boxed_1408_ = lean_unbox_usize(v_i_1402_);
lean_dec(v_i_1402_);
v_res_1409_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3(v_as_1400_, v_sz_boxed_1407_, v_i_boxed_1408_, v_b_1403_, v___y_1404_, v___y_1405_);
lean_dec(v___y_1405_);
lean_dec_ref(v___y_1404_);
lean_dec_ref(v_as_1400_);
return v_res_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1(lean_object* v_x_1413_){
_start:
{
lean_object* v___x_1414_; 
v___x_1414_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1___closed__0));
return v___x_1414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1___boxed(lean_object* v_x_1415_){
_start:
{
lean_object* v_res_1416_; 
v_res_1416_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1(v_x_1415_);
lean_dec(v_x_1415_);
return v_res_1416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0(lean_object* v___x_1417_, lean_object* v_r_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_){
_start:
{
lean_object* v___x_1422_; lean_object* v___x_1423_; 
v___x_1422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1422_, 0, v___x_1417_);
v___x_1423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1422_);
return v___x_1423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0___boxed(lean_object* v___x_1424_, lean_object* v_r_1425_, lean_object* v___y_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_){
_start:
{
lean_object* v_res_1429_; 
v_res_1429_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0(v___x_1424_, v_r_1425_, v___y_1426_, v___y_1427_);
lean_dec(v___y_1427_);
lean_dec_ref(v___y_1426_);
return v_res_1429_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__3(void){
_start:
{
lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; 
v___x_1433_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__2));
v___x_1434_ = lean_unsigned_to_nat(14u);
v___x_1435_ = lean_unsigned_to_nat(22u);
v___x_1436_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__1));
v___x_1437_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__0));
v___x_1438_ = l_mkPanicMessageWithDecl(v___x_1437_, v___x_1436_, v___x_1435_, v___x_1434_, v___x_1433_);
return v___x_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2(uint8_t v___x_1439_, lean_object* v___x_1440_, uint8_t v___x_1441_, lean_object* v_n_1442_){
_start:
{
lean_object* v___y_1444_; lean_object* v___x_1448_; 
v___x_1448_ = l_Lean_Syntax_getTailPos_x3f(v___x_1440_, v___x_1441_);
if (lean_obj_tag(v___x_1448_) == 0)
{
lean_object* v___x_1449_; lean_object* v___x_1450_; 
v___x_1449_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___closed__3);
v___x_1450_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__5(v___x_1449_);
v___y_1444_ = v___x_1450_;
goto v___jp_1443_;
}
else
{
lean_object* v_val_1451_; 
v_val_1451_ = lean_ctor_get(v___x_1448_, 0);
lean_inc(v_val_1451_);
lean_dec_ref_known(v___x_1448_, 1);
v___y_1444_ = v_val_1451_;
goto v___jp_1443_;
}
v___jp_1443_:
{
lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; 
v___x_1445_ = lean_nat_sub(v___y_1444_, v_n_1442_);
lean_dec(v___y_1444_);
lean_inc(v___x_1445_);
v___x_1446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1446_, 0, v___x_1445_);
lean_ctor_set(v___x_1446_, 1, v___x_1445_);
v___x_1447_ = l_Lean_Syntax_ofRange(v___x_1446_, v___x_1439_);
return v___x_1447_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2___boxed(lean_object* v___x_1452_, lean_object* v___x_1453_, lean_object* v___x_1454_, lean_object* v_n_1455_){
_start:
{
uint8_t v___x_21523__boxed_1456_; uint8_t v___x_21525__boxed_1457_; lean_object* v_res_1458_; 
v___x_21523__boxed_1456_ = lean_unbox(v___x_1452_);
v___x_21525__boxed_1457_ = lean_unbox(v___x_1454_);
v_res_1458_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2(v___x_21523__boxed_1456_, v___x_1453_, v___x_21525__boxed_1457_, v_n_1455_);
lean_dec(v_n_1455_);
lean_dec(v___x_1453_);
return v_res_1458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__6(lean_object* v_s_1459_, lean_object* v_pos_1460_){
_start:
{
lean_object* v_str_1461_; lean_object* v_startInclusive_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; uint8_t v___x_1466_; 
v_str_1461_ = lean_ctor_get(v_s_1459_, 0);
v_startInclusive_1462_ = lean_ctor_get(v_s_1459_, 1);
v___x_1463_ = lean_nat_add(v_startInclusive_1462_, v_pos_1460_);
v___x_1464_ = lean_nat_sub(v___x_1463_, v_startInclusive_1462_);
v___x_1465_ = lean_unsigned_to_nat(0u);
v___x_1466_ = lean_nat_dec_eq(v___x_1464_, v___x_1465_);
if (v___x_1466_ == 0)
{
lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; uint8_t v___y_1475_; lean_object* v___x_1476_; uint32_t v___x_1477_; uint8_t v___y_1479_; uint32_t v___x_1484_; uint8_t v___x_1485_; 
lean_inc(v_startInclusive_1462_);
lean_inc_ref(v_str_1461_);
v___x_1467_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1467_, 0, v_str_1461_);
lean_ctor_set(v___x_1467_, 1, v_startInclusive_1462_);
lean_ctor_set(v___x_1467_, 2, v___x_1463_);
v___x_1468_ = lean_unsigned_to_nat(1u);
v___x_1469_ = lean_nat_sub(v___x_1464_, v___x_1468_);
lean_dec(v___x_1464_);
v___x_1470_ = l_String_Slice_posLE(v___x_1467_, v___x_1469_);
lean_dec_ref_known(v___x_1467_, 3);
v___x_1476_ = lean_nat_add(v_startInclusive_1462_, v___x_1470_);
v___x_1477_ = lean_string_utf8_get_fast(v_str_1461_, v___x_1476_);
lean_dec(v___x_1476_);
v___x_1484_ = 32;
v___x_1485_ = lean_uint32_dec_eq(v___x_1477_, v___x_1484_);
if (v___x_1485_ == 0)
{
uint32_t v___x_1486_; uint8_t v___x_1487_; 
v___x_1486_ = 9;
v___x_1487_ = lean_uint32_dec_eq(v___x_1477_, v___x_1486_);
v___y_1479_ = v___x_1487_;
goto v___jp_1478_;
}
else
{
v___y_1479_ = v___x_1485_;
goto v___jp_1478_;
}
v___jp_1471_:
{
uint8_t v___x_1472_; 
v___x_1472_ = lean_nat_dec_lt(v___x_1470_, v_pos_1460_);
if (v___x_1472_ == 0)
{
lean_dec(v___x_1470_);
return v_pos_1460_;
}
else
{
lean_dec(v_pos_1460_);
v_pos_1460_ = v___x_1470_;
goto _start;
}
}
v___jp_1474_:
{
if (v___y_1475_ == 0)
{
lean_dec(v___x_1470_);
return v_pos_1460_;
}
else
{
goto v___jp_1471_;
}
}
v___jp_1478_:
{
if (v___y_1479_ == 0)
{
uint32_t v___x_1480_; uint8_t v___x_1481_; 
v___x_1480_ = 13;
v___x_1481_ = lean_uint32_dec_eq(v___x_1477_, v___x_1480_);
if (v___x_1481_ == 0)
{
uint32_t v___x_1482_; uint8_t v___x_1483_; 
v___x_1482_ = 10;
v___x_1483_ = lean_uint32_dec_eq(v___x_1477_, v___x_1482_);
v___y_1475_ = v___x_1483_;
goto v___jp_1474_;
}
else
{
v___y_1475_ = v___x_1481_;
goto v___jp_1474_;
}
}
else
{
goto v___jp_1471_;
}
}
}
else
{
lean_dec(v___x_1464_);
lean_dec(v___x_1463_);
return v_pos_1460_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__6___boxed(lean_object* v_s_1488_, lean_object* v_pos_1489_){
_start:
{
lean_object* v_res_1490_; 
v_res_1490_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__6(v_s_1488_, v_pos_1489_);
lean_dec_ref(v_s_1488_);
return v_res_1490_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8_spec__12(lean_object* v_a_1491_, lean_object* v_as_1492_, size_t v_i_1493_, size_t v_stop_1494_){
_start:
{
uint8_t v___x_1495_; 
v___x_1495_ = lean_usize_dec_eq(v_i_1493_, v_stop_1494_);
if (v___x_1495_ == 0)
{
lean_object* v___x_1496_; uint8_t v___x_1497_; 
v___x_1496_ = lean_array_uget_borrowed(v_as_1492_, v_i_1493_);
v___x_1497_ = lean_string_dec_eq(v_a_1491_, v___x_1496_);
if (v___x_1497_ == 0)
{
size_t v___x_1498_; size_t v___x_1499_; 
v___x_1498_ = ((size_t)1ULL);
v___x_1499_ = lean_usize_add(v_i_1493_, v___x_1498_);
v_i_1493_ = v___x_1499_;
goto _start;
}
else
{
return v___x_1497_;
}
}
else
{
uint8_t v___x_1501_; 
v___x_1501_ = 0;
return v___x_1501_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8_spec__12___boxed(lean_object* v_a_1502_, lean_object* v_as_1503_, lean_object* v_i_1504_, lean_object* v_stop_1505_){
_start:
{
size_t v_i_boxed_1506_; size_t v_stop_boxed_1507_; uint8_t v_res_1508_; lean_object* v_r_1509_; 
v_i_boxed_1506_ = lean_unbox_usize(v_i_1504_);
lean_dec(v_i_1504_);
v_stop_boxed_1507_ = lean_unbox_usize(v_stop_1505_);
lean_dec(v_stop_1505_);
v_res_1508_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8_spec__12(v_a_1502_, v_as_1503_, v_i_boxed_1506_, v_stop_boxed_1507_);
lean_dec_ref(v_as_1503_);
lean_dec_ref(v_a_1502_);
v_r_1509_ = lean_box(v_res_1508_);
return v_r_1509_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8(lean_object* v_as_1510_, lean_object* v_a_1511_){
_start:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; uint8_t v___x_1514_; 
v___x_1512_ = lean_unsigned_to_nat(0u);
v___x_1513_ = lean_array_get_size(v_as_1510_);
v___x_1514_ = lean_nat_dec_lt(v___x_1512_, v___x_1513_);
if (v___x_1514_ == 0)
{
return v___x_1514_;
}
else
{
if (v___x_1514_ == 0)
{
return v___x_1514_;
}
else
{
size_t v___x_1515_; size_t v___x_1516_; uint8_t v___x_1517_; 
v___x_1515_ = ((size_t)0ULL);
v___x_1516_ = lean_usize_of_nat(v___x_1513_);
v___x_1517_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8_spec__12(v_a_1511_, v_as_1510_, v___x_1515_, v___x_1516_);
return v___x_1517_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8___boxed(lean_object* v_as_1518_, lean_object* v_a_1519_){
_start:
{
uint8_t v_res_1520_; lean_object* v_r_1521_; 
v_res_1520_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8(v_as_1518_, v_a_1519_);
lean_dec_ref(v_a_1519_);
lean_dec_ref(v_as_1518_);
v_r_1521_ = lean_box(v_res_1520_);
return v_r_1521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(lean_object* v_linterOption_1522_, lean_object* v_stx_1523_, lean_object* v_msg_1524_, lean_object* v___y_1525_, lean_object* v___y_1526_){
_start:
{
lean_object* v___x_1528_; lean_object* v_a_1529_; lean_object* v___x_1531_; uint8_t v_isShared_1532_; uint8_t v_isSharedCheck_1539_; 
v___x_1528_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_1525_, v___y_1526_);
v_a_1529_ = lean_ctor_get(v___x_1528_, 0);
v_isSharedCheck_1539_ = !lean_is_exclusive(v___x_1528_);
if (v_isSharedCheck_1539_ == 0)
{
v___x_1531_ = v___x_1528_;
v_isShared_1532_ = v_isSharedCheck_1539_;
goto v_resetjp_1530_;
}
else
{
lean_inc(v_a_1529_);
lean_dec(v___x_1528_);
v___x_1531_ = lean_box(0);
v_isShared_1532_ = v_isSharedCheck_1539_;
goto v_resetjp_1530_;
}
v_resetjp_1530_:
{
uint8_t v___x_1533_; 
v___x_1533_ = l_Lean_Linter_getLinterValue(v_linterOption_1522_, v_a_1529_);
lean_dec(v_a_1529_);
if (v___x_1533_ == 0)
{
lean_object* v___x_1534_; lean_object* v___x_1536_; 
lean_dec_ref(v_msg_1524_);
lean_dec(v_stx_1523_);
lean_dec_ref(v_linterOption_1522_);
v___x_1534_ = lean_box(0);
if (v_isShared_1532_ == 0)
{
lean_ctor_set(v___x_1531_, 0, v___x_1534_);
v___x_1536_ = v___x_1531_;
goto v_reusejp_1535_;
}
else
{
lean_object* v_reuseFailAlloc_1537_; 
v_reuseFailAlloc_1537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1537_, 0, v___x_1534_);
v___x_1536_ = v_reuseFailAlloc_1537_;
goto v_reusejp_1535_;
}
v_reusejp_1535_:
{
return v___x_1536_;
}
}
else
{
lean_object* v___x_1538_; 
lean_del_object(v___x_1531_);
v___x_1538_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(v_linterOption_1522_, v_stx_1523_, v_msg_1524_, v___y_1525_, v___y_1526_);
return v___x_1538_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7___boxed(lean_object* v_linterOption_1540_, lean_object* v_stx_1541_, lean_object* v_msg_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_){
_start:
{
lean_object* v_res_1546_; 
v_res_1546_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(v_linterOption_1540_, v_stx_1541_, v_msg_1542_, v___y_1543_, v___y_1544_);
lean_dec(v___y_1544_);
lean_dec_ref(v___y_1543_);
return v_res_1546_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0(void){
_start:
{
lean_object* v___x_1547_; lean_object* v___x_1548_; 
v___x_1547_ = lean_box(1);
v___x_1548_ = l_Lean_MessageData_ofFormat(v___x_1547_);
return v___x_1548_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__3(void){
_start:
{
lean_object* v___x_1552_; lean_object* v___x_1553_; 
v___x_1552_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__2));
v___x_1553_ = l_Lean_MessageData_ofFormat(v___x_1552_);
return v___x_1553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18(lean_object* v_x_1554_, lean_object* v_x_1555_){
_start:
{
if (lean_obj_tag(v_x_1555_) == 0)
{
return v_x_1554_;
}
else
{
lean_object* v_head_1556_; lean_object* v_tail_1557_; lean_object* v___x_1559_; uint8_t v_isShared_1560_; uint8_t v_isSharedCheck_1579_; 
v_head_1556_ = lean_ctor_get(v_x_1555_, 0);
v_tail_1557_ = lean_ctor_get(v_x_1555_, 1);
v_isSharedCheck_1579_ = !lean_is_exclusive(v_x_1555_);
if (v_isSharedCheck_1579_ == 0)
{
v___x_1559_ = v_x_1555_;
v_isShared_1560_ = v_isSharedCheck_1579_;
goto v_resetjp_1558_;
}
else
{
lean_inc(v_tail_1557_);
lean_inc(v_head_1556_);
lean_dec(v_x_1555_);
v___x_1559_ = lean_box(0);
v_isShared_1560_ = v_isSharedCheck_1579_;
goto v_resetjp_1558_;
}
v_resetjp_1558_:
{
lean_object* v_before_1561_; lean_object* v___x_1563_; uint8_t v_isShared_1564_; uint8_t v_isSharedCheck_1577_; 
v_before_1561_ = lean_ctor_get(v_head_1556_, 0);
v_isSharedCheck_1577_ = !lean_is_exclusive(v_head_1556_);
if (v_isSharedCheck_1577_ == 0)
{
lean_object* v_unused_1578_; 
v_unused_1578_ = lean_ctor_get(v_head_1556_, 1);
lean_dec(v_unused_1578_);
v___x_1563_ = v_head_1556_;
v_isShared_1564_ = v_isSharedCheck_1577_;
goto v_resetjp_1562_;
}
else
{
lean_inc(v_before_1561_);
lean_dec(v_head_1556_);
v___x_1563_ = lean_box(0);
v_isShared_1564_ = v_isSharedCheck_1577_;
goto v_resetjp_1562_;
}
v_resetjp_1562_:
{
lean_object* v___x_1565_; lean_object* v___x_1567_; 
v___x_1565_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0);
if (v_isShared_1564_ == 0)
{
lean_ctor_set_tag(v___x_1563_, 7);
lean_ctor_set(v___x_1563_, 1, v___x_1565_);
lean_ctor_set(v___x_1563_, 0, v_x_1554_);
v___x_1567_ = v___x_1563_;
goto v_reusejp_1566_;
}
else
{
lean_object* v_reuseFailAlloc_1576_; 
v_reuseFailAlloc_1576_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1576_, 0, v_x_1554_);
lean_ctor_set(v_reuseFailAlloc_1576_, 1, v___x_1565_);
v___x_1567_ = v_reuseFailAlloc_1576_;
goto v_reusejp_1566_;
}
v_reusejp_1566_:
{
lean_object* v___x_1568_; lean_object* v___x_1570_; 
v___x_1568_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__3);
if (v_isShared_1560_ == 0)
{
lean_ctor_set_tag(v___x_1559_, 7);
lean_ctor_set(v___x_1559_, 1, v___x_1568_);
lean_ctor_set(v___x_1559_, 0, v___x_1567_);
v___x_1570_ = v___x_1559_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1575_; 
v_reuseFailAlloc_1575_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1575_, 0, v___x_1567_);
lean_ctor_set(v_reuseFailAlloc_1575_, 1, v___x_1568_);
v___x_1570_ = v_reuseFailAlloc_1575_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; 
v___x_1571_ = l_Lean_MessageData_ofSyntax(v_before_1561_);
v___x_1572_ = l_Lean_indentD(v___x_1571_);
v___x_1573_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1573_, 0, v___x_1570_);
lean_ctor_set(v___x_1573_, 1, v___x_1572_);
v_x_1554_ = v___x_1573_;
v_x_1555_ = v_tail_1557_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__2(void){
_start:
{
lean_object* v___x_1583_; lean_object* v___x_1584_; 
v___x_1583_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__1));
v___x_1584_ = l_Lean_MessageData_ofFormat(v___x_1583_);
return v___x_1584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg(lean_object* v_msgData_1585_, lean_object* v_macroStack_1586_, lean_object* v___y_1587_){
_start:
{
lean_object* v___x_1589_; lean_object* v_scopes_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v_opts_1593_; lean_object* v___x_1594_; uint8_t v___x_1595_; 
v___x_1589_ = lean_st_ref_get(v___y_1587_);
v_scopes_1590_ = lean_ctor_get(v___x_1589_, 2);
lean_inc(v_scopes_1590_);
lean_dec(v___x_1589_);
v___x_1591_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1592_ = l_List_head_x21___redArg(v___x_1591_, v_scopes_1590_);
lean_dec(v_scopes_1590_);
v_opts_1593_ = lean_ctor_get(v___x_1592_, 1);
lean_inc_ref(v_opts_1593_);
lean_dec(v___x_1592_);
v___x_1594_ = l_Lean_Elab_pp_macroStack;
v___x_1595_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(v_opts_1593_, v___x_1594_);
lean_dec_ref(v_opts_1593_);
if (v___x_1595_ == 0)
{
lean_object* v___x_1596_; 
lean_dec(v_macroStack_1586_);
v___x_1596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1596_, 0, v_msgData_1585_);
return v___x_1596_;
}
else
{
if (lean_obj_tag(v_macroStack_1586_) == 0)
{
lean_object* v___x_1597_; 
v___x_1597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1597_, 0, v_msgData_1585_);
return v___x_1597_;
}
else
{
lean_object* v_head_1598_; lean_object* v_after_1599_; lean_object* v___x_1601_; uint8_t v_isShared_1602_; uint8_t v_isSharedCheck_1614_; 
v_head_1598_ = lean_ctor_get(v_macroStack_1586_, 0);
lean_inc(v_head_1598_);
v_after_1599_ = lean_ctor_get(v_head_1598_, 1);
v_isSharedCheck_1614_ = !lean_is_exclusive(v_head_1598_);
if (v_isSharedCheck_1614_ == 0)
{
lean_object* v_unused_1615_; 
v_unused_1615_ = lean_ctor_get(v_head_1598_, 0);
lean_dec(v_unused_1615_);
v___x_1601_ = v_head_1598_;
v_isShared_1602_ = v_isSharedCheck_1614_;
goto v_resetjp_1600_;
}
else
{
lean_inc(v_after_1599_);
lean_dec(v_head_1598_);
v___x_1601_ = lean_box(0);
v_isShared_1602_ = v_isSharedCheck_1614_;
goto v_resetjp_1600_;
}
v_resetjp_1600_:
{
lean_object* v___x_1603_; lean_object* v___x_1605_; 
v___x_1603_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18___closed__0);
if (v_isShared_1602_ == 0)
{
lean_ctor_set_tag(v___x_1601_, 7);
lean_ctor_set(v___x_1601_, 1, v___x_1603_);
lean_ctor_set(v___x_1601_, 0, v_msgData_1585_);
v___x_1605_ = v___x_1601_;
goto v_reusejp_1604_;
}
else
{
lean_object* v_reuseFailAlloc_1613_; 
v_reuseFailAlloc_1613_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1613_, 0, v_msgData_1585_);
lean_ctor_set(v_reuseFailAlloc_1613_, 1, v___x_1603_);
v___x_1605_ = v_reuseFailAlloc_1613_;
goto v_reusejp_1604_;
}
v_reusejp_1604_:
{
lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v_msgData_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; 
v___x_1606_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___closed__2);
v___x_1607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1607_, 0, v___x_1605_);
lean_ctor_set(v___x_1607_, 1, v___x_1606_);
v___x_1608_ = l_Lean_MessageData_ofSyntax(v_after_1599_);
v___x_1609_ = l_Lean_indentD(v___x_1608_);
v_msgData_1610_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1610_, 0, v___x_1607_);
lean_ctor_set(v_msgData_1610_, 1, v___x_1609_);
v___x_1611_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15_spec__18(v_msgData_1610_, v_macroStack_1586_);
v___x_1612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1611_);
return v___x_1612_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg___boxed(lean_object* v_msgData_1616_, lean_object* v_macroStack_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_){
_start:
{
lean_object* v_res_1620_; 
v_res_1620_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg(v_msgData_1616_, v_macroStack_1617_, v___y_1618_);
lean_dec(v___y_1618_);
return v_res_1620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg(lean_object* v_msg_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___x_1625_; 
v___x_1625_ = l_Lean_Elab_Command_getRef___redArg(v___y_1622_);
if (lean_obj_tag(v___x_1625_) == 0)
{
lean_object* v_a_1626_; lean_object* v_macroStack_1627_; lean_object* v___x_1628_; lean_object* v_a_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v_a_1632_; lean_object* v___x_1634_; uint8_t v_isShared_1635_; uint8_t v_isSharedCheck_1640_; 
v_a_1626_ = lean_ctor_get(v___x_1625_, 0);
lean_inc(v_a_1626_);
lean_dec_ref_known(v___x_1625_, 1);
v_macroStack_1627_ = lean_ctor_get(v___y_1622_, 4);
v___x_1628_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg(v_msg_1621_, v___y_1623_);
v_a_1629_ = lean_ctor_get(v___x_1628_, 0);
lean_inc(v_a_1629_);
lean_dec_ref(v___x_1628_);
v___x_1630_ = l_Lean_Elab_getBetterRef(v_a_1626_, v_macroStack_1627_);
lean_dec(v_a_1626_);
lean_inc(v_macroStack_1627_);
v___x_1631_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg(v_a_1629_, v_macroStack_1627_, v___y_1623_);
v_a_1632_ = lean_ctor_get(v___x_1631_, 0);
v_isSharedCheck_1640_ = !lean_is_exclusive(v___x_1631_);
if (v_isSharedCheck_1640_ == 0)
{
v___x_1634_ = v___x_1631_;
v_isShared_1635_ = v_isSharedCheck_1640_;
goto v_resetjp_1633_;
}
else
{
lean_inc(v_a_1632_);
lean_dec(v___x_1631_);
v___x_1634_ = lean_box(0);
v_isShared_1635_ = v_isSharedCheck_1640_;
goto v_resetjp_1633_;
}
v_resetjp_1633_:
{
lean_object* v___x_1636_; lean_object* v___x_1638_; 
v___x_1636_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1636_, 0, v___x_1630_);
lean_ctor_set(v___x_1636_, 1, v_a_1632_);
if (v_isShared_1635_ == 0)
{
lean_ctor_set_tag(v___x_1634_, 1);
lean_ctor_set(v___x_1634_, 0, v___x_1636_);
v___x_1638_ = v___x_1634_;
goto v_reusejp_1637_;
}
else
{
lean_object* v_reuseFailAlloc_1639_; 
v_reuseFailAlloc_1639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1639_, 0, v___x_1636_);
v___x_1638_ = v_reuseFailAlloc_1639_;
goto v_reusejp_1637_;
}
v_reusejp_1637_:
{
return v___x_1638_;
}
}
}
else
{
lean_object* v_a_1641_; lean_object* v___x_1643_; uint8_t v_isShared_1644_; uint8_t v_isSharedCheck_1648_; 
lean_dec_ref(v_msg_1621_);
v_a_1641_ = lean_ctor_get(v___x_1625_, 0);
v_isSharedCheck_1648_ = !lean_is_exclusive(v___x_1625_);
if (v_isSharedCheck_1648_ == 0)
{
v___x_1643_ = v___x_1625_;
v_isShared_1644_ = v_isSharedCheck_1648_;
goto v_resetjp_1642_;
}
else
{
lean_inc(v_a_1641_);
lean_dec(v___x_1625_);
v___x_1643_ = lean_box(0);
v_isShared_1644_ = v_isSharedCheck_1648_;
goto v_resetjp_1642_;
}
v_resetjp_1642_:
{
lean_object* v___x_1646_; 
if (v_isShared_1644_ == 0)
{
v___x_1646_ = v___x_1643_;
goto v_reusejp_1645_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v_a_1641_);
v___x_1646_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1645_;
}
v_reusejp_1645_:
{
return v___x_1646_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg___boxed(lean_object* v_msg_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_, lean_object* v___y_1652_){
_start:
{
lean_object* v_res_1653_; 
v_res_1653_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg(v_msg_1649_, v___y_1650_, v___y_1651_);
lean_dec(v___y_1651_);
lean_dec_ref(v___y_1650_);
return v_res_1653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg(lean_object* v_ref_1654_, lean_object* v_msg_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_){
_start:
{
lean_object* v___x_1659_; 
v___x_1659_ = l_Lean_Elab_Command_getRef___redArg(v___y_1656_);
if (lean_obj_tag(v___x_1659_) == 0)
{
lean_object* v_a_1660_; lean_object* v_fileName_1661_; lean_object* v_fileMap_1662_; lean_object* v_currRecDepth_1663_; lean_object* v_cmdPos_1664_; lean_object* v_macroStack_1665_; lean_object* v_quotContext_x3f_1666_; lean_object* v_currMacroScope_1667_; lean_object* v_snap_x3f_1668_; lean_object* v_cancelTk_x3f_1669_; uint8_t v_suppressElabErrors_1670_; lean_object* v_ref_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; 
v_a_1660_ = lean_ctor_get(v___x_1659_, 0);
lean_inc(v_a_1660_);
lean_dec_ref_known(v___x_1659_, 1);
v_fileName_1661_ = lean_ctor_get(v___y_1656_, 0);
v_fileMap_1662_ = lean_ctor_get(v___y_1656_, 1);
v_currRecDepth_1663_ = lean_ctor_get(v___y_1656_, 2);
v_cmdPos_1664_ = lean_ctor_get(v___y_1656_, 3);
v_macroStack_1665_ = lean_ctor_get(v___y_1656_, 4);
v_quotContext_x3f_1666_ = lean_ctor_get(v___y_1656_, 5);
v_currMacroScope_1667_ = lean_ctor_get(v___y_1656_, 6);
v_snap_x3f_1668_ = lean_ctor_get(v___y_1656_, 8);
v_cancelTk_x3f_1669_ = lean_ctor_get(v___y_1656_, 9);
v_suppressElabErrors_1670_ = lean_ctor_get_uint8(v___y_1656_, sizeof(void*)*10);
v_ref_1671_ = l_Lean_replaceRef(v_ref_1654_, v_a_1660_);
lean_dec(v_a_1660_);
lean_inc(v_cancelTk_x3f_1669_);
lean_inc(v_snap_x3f_1668_);
lean_inc(v_currMacroScope_1667_);
lean_inc(v_quotContext_x3f_1666_);
lean_inc(v_macroStack_1665_);
lean_inc(v_cmdPos_1664_);
lean_inc(v_currRecDepth_1663_);
lean_inc_ref(v_fileMap_1662_);
lean_inc_ref(v_fileName_1661_);
v___x_1672_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_1672_, 0, v_fileName_1661_);
lean_ctor_set(v___x_1672_, 1, v_fileMap_1662_);
lean_ctor_set(v___x_1672_, 2, v_currRecDepth_1663_);
lean_ctor_set(v___x_1672_, 3, v_cmdPos_1664_);
lean_ctor_set(v___x_1672_, 4, v_macroStack_1665_);
lean_ctor_set(v___x_1672_, 5, v_quotContext_x3f_1666_);
lean_ctor_set(v___x_1672_, 6, v_currMacroScope_1667_);
lean_ctor_set(v___x_1672_, 7, v_ref_1671_);
lean_ctor_set(v___x_1672_, 8, v_snap_x3f_1668_);
lean_ctor_set(v___x_1672_, 9, v_cancelTk_x3f_1669_);
lean_ctor_set_uint8(v___x_1672_, sizeof(void*)*10, v_suppressElabErrors_1670_);
v___x_1673_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg(v_msg_1655_, v___x_1672_, v___y_1657_);
lean_dec_ref_known(v___x_1672_, 10);
return v___x_1673_;
}
else
{
lean_object* v_a_1674_; lean_object* v___x_1676_; uint8_t v_isShared_1677_; uint8_t v_isSharedCheck_1681_; 
lean_dec_ref(v_msg_1655_);
v_a_1674_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_1681_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_1681_ == 0)
{
v___x_1676_ = v___x_1659_;
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
else
{
lean_inc(v_a_1674_);
lean_dec(v___x_1659_);
v___x_1676_ = lean_box(0);
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
v_resetjp_1675_:
{
lean_object* v___x_1679_; 
if (v_isShared_1677_ == 0)
{
v___x_1679_ = v___x_1676_;
goto v_reusejp_1678_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v_a_1674_);
v___x_1679_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1678_;
}
v_reusejp_1678_:
{
return v___x_1679_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg___boxed(lean_object* v_ref_1682_, lean_object* v_msg_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_){
_start:
{
lean_object* v_res_1687_; 
v_res_1687_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg(v_ref_1682_, v_msg_1683_, v___y_1684_, v___y_1685_);
lean_dec(v___y_1685_);
lean_dec_ref(v___y_1684_);
lean_dec(v_ref_1682_);
return v_res_1687_;
}
}
static lean_object* _init_lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1689_; lean_object* v___x_1690_; 
v___x_1689_ = ((lean_object*)(lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__0));
v___x_1690_ = l_Lean_stringToMessageData(v___x_1689_);
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2(lean_object* v_stx_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_){
_start:
{
lean_object* v_val_1703_; lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1710_ = lean_unsigned_to_nat(1u);
v___x_1711_ = l_Lean_Syntax_getArg(v_stx_1692_, v___x_1710_);
switch(lean_obj_tag(v___x_1711_))
{
case 2:
{
lean_object* v_val_1712_; 
lean_dec(v_stx_1692_);
v_val_1712_ = lean_ctor_get(v___x_1711_, 1);
lean_inc_ref(v_val_1712_);
lean_dec_ref_known(v___x_1711_, 2);
v_val_1703_ = v_val_1712_;
goto v___jp_1702_;
}
case 1:
{
lean_object* v_kind_1713_; 
v_kind_1713_ = lean_ctor_get(v___x_1711_, 1);
lean_inc(v_kind_1713_);
if (lean_obj_tag(v_kind_1713_) == 1)
{
lean_object* v_pre_1714_; 
v_pre_1714_ = lean_ctor_get(v_kind_1713_, 0);
lean_inc(v_pre_1714_);
if (lean_obj_tag(v_pre_1714_) == 1)
{
lean_object* v_pre_1715_; 
v_pre_1715_ = lean_ctor_get(v_pre_1714_, 0);
lean_inc(v_pre_1715_);
if (lean_obj_tag(v_pre_1715_) == 1)
{
lean_object* v_pre_1716_; 
v_pre_1716_ = lean_ctor_get(v_pre_1715_, 0);
lean_inc(v_pre_1716_);
if (lean_obj_tag(v_pre_1716_) == 1)
{
lean_object* v_pre_1717_; 
v_pre_1717_ = lean_ctor_get(v_pre_1716_, 0);
if (lean_obj_tag(v_pre_1717_) == 0)
{
lean_object* v_str_1718_; lean_object* v_str_1719_; lean_object* v_str_1720_; lean_object* v_str_1721_; lean_object* v___x_1722_; uint8_t v___x_1723_; 
v_str_1718_ = lean_ctor_get(v_kind_1713_, 1);
lean_inc_ref(v_str_1718_);
lean_dec_ref_known(v_kind_1713_, 2);
v_str_1719_ = lean_ctor_get(v_pre_1714_, 1);
lean_inc_ref(v_str_1719_);
lean_dec_ref_known(v_pre_1714_, 2);
v_str_1720_ = lean_ctor_get(v_pre_1715_, 1);
lean_inc_ref(v_str_1720_);
lean_dec_ref_known(v_pre_1715_, 2);
v_str_1721_ = lean_ctor_get(v_pre_1716_, 1);
lean_inc_ref(v_str_1721_);
lean_dec_ref_known(v_pre_1716_, 2);
v___x_1722_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1));
v___x_1723_ = lean_string_dec_eq(v_str_1721_, v___x_1722_);
lean_dec_ref(v_str_1721_);
if (v___x_1723_ == 0)
{
lean_dec_ref(v_str_1720_);
lean_dec_ref(v_str_1719_);
lean_dec_ref(v_str_1718_);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
else
{
lean_object* v___x_1724_; uint8_t v___x_1725_; 
v___x_1724_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2));
v___x_1725_ = lean_string_dec_eq(v_str_1720_, v___x_1724_);
lean_dec_ref(v_str_1720_);
if (v___x_1725_ == 0)
{
lean_dec_ref(v_str_1719_);
lean_dec_ref(v_str_1718_);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
else
{
lean_object* v___x_1726_; uint8_t v___x_1727_; 
v___x_1726_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3));
v___x_1727_ = lean_string_dec_eq(v_str_1719_, v___x_1726_);
lean_dec_ref(v_str_1719_);
if (v___x_1727_ == 0)
{
lean_dec_ref(v_str_1718_);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
else
{
lean_object* v___x_1728_; uint8_t v___x_1729_; 
v___x_1728_ = ((lean_object*)(lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__2));
v___x_1729_ = lean_string_dec_eq(v_str_1718_, v___x_1728_);
lean_dec_ref(v_str_1718_);
if (v___x_1729_ == 0)
{
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
else
{
lean_object* v___x_1730_; lean_object* v___x_1731_; 
v___x_1730_ = lean_unsigned_to_nat(0u);
v___x_1731_ = l_Lean_Syntax_getArg(v___x_1711_, v___x_1730_);
lean_dec_ref_known(v___x_1711_, 3);
if (lean_obj_tag(v___x_1731_) == 2)
{
lean_object* v_val_1732_; 
lean_dec(v_stx_1692_);
v_val_1732_ = lean_ctor_get(v___x_1731_, 1);
lean_inc_ref(v_val_1732_);
lean_dec_ref_known(v___x_1731_, 2);
v_val_1703_ = v_val_1732_;
goto v___jp_1702_;
}
else
{
lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; 
lean_dec(v___x_1731_);
v___x_1733_ = lean_obj_once(&lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1, &lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1_once, _init_lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1);
lean_inc(v_stx_1692_);
v___x_1734_ = l_Lean_MessageData_ofSyntax(v_stx_1692_);
v___x_1735_ = l_Lean_indentD(v___x_1734_);
v___x_1736_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1736_, 0, v___x_1733_);
lean_ctor_set(v___x_1736_, 1, v___x_1735_);
v___x_1737_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg(v_stx_1692_, v___x_1736_, v___y_1693_, v___y_1694_);
lean_dec(v_stx_1692_);
return v___x_1737_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_1716_, 2);
lean_dec_ref_known(v_pre_1715_, 2);
lean_dec_ref_known(v_pre_1714_, 2);
lean_dec_ref_known(v_kind_1713_, 2);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
}
else
{
lean_dec_ref_known(v_pre_1715_, 2);
lean_dec(v_pre_1716_);
lean_dec_ref_known(v_pre_1714_, 2);
lean_dec_ref_known(v_kind_1713_, 2);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
}
else
{
lean_dec(v_pre_1715_);
lean_dec_ref_known(v_pre_1714_, 2);
lean_dec_ref_known(v_kind_1713_, 2);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
}
else
{
lean_dec_ref_known(v_kind_1713_, 2);
lean_dec(v_pre_1714_);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
}
else
{
lean_dec(v_kind_1713_);
lean_dec_ref_known(v___x_1711_, 3);
goto v___jp_1696_;
}
}
default: 
{
lean_dec(v___x_1711_);
goto v___jp_1696_;
}
}
v___jp_1696_:
{
lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; 
v___x_1697_ = lean_obj_once(&lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1, &lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1_once, _init_lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___closed__1);
lean_inc(v_stx_1692_);
v___x_1698_ = l_Lean_MessageData_ofSyntax(v_stx_1692_);
v___x_1699_ = l_Lean_indentD(v___x_1698_);
v___x_1700_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1700_, 0, v___x_1697_);
lean_ctor_set(v___x_1700_, 1, v___x_1699_);
v___x_1701_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg(v_stx_1692_, v___x_1700_, v___y_1693_, v___y_1694_);
lean_dec(v_stx_1692_);
return v___x_1701_;
}
v___jp_1702_:
{
lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; 
v___x_1704_ = lean_unsigned_to_nat(0u);
v___x_1705_ = lean_string_utf8_byte_size(v_val_1703_);
v___x_1706_ = lean_unsigned_to_nat(2u);
v___x_1707_ = lean_nat_sub(v___x_1705_, v___x_1706_);
v___x_1708_ = lean_string_utf8_extract(v_val_1703_, v___x_1704_, v___x_1707_);
lean_dec(v___x_1707_);
lean_dec_ref(v_val_1703_);
v___x_1709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1709_, 0, v___x_1708_);
return v___x_1709_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2___boxed(lean_object* v_stx_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_){
_start:
{
lean_object* v_res_1742_; 
v_res_1742_ = lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2(v_stx_1738_, v___y_1739_, v___y_1740_);
lean_dec(v___y_1740_);
lean_dec_ref(v___y_1739_);
return v_res_1742_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__4(void){
_start:
{
lean_object* v___x_1752_; lean_object* v___x_1753_; 
v___x_1752_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__3));
v___x_1753_ = l_Lean_MessageData_ofFormat(v___x_1752_);
return v___x_1753_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__6(void){
_start:
{
lean_object* v___x_1755_; lean_object* v___x_1756_; 
v___x_1755_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__5));
v___x_1756_ = lean_string_utf8_byte_size(v___x_1755_);
return v___x_1756_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__7(void){
_start:
{
lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; 
v___x_1757_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__6, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__6_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__6);
v___x_1758_ = lean_unsigned_to_nat(0u);
v___x_1759_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__5));
v___x_1760_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1760_, 0, v___x_1759_);
lean_ctor_set(v___x_1760_, 1, v___x_1758_);
lean_ctor_set(v___x_1760_, 2, v___x_1757_);
return v___x_1760_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__10(void){
_start:
{
lean_object* v___x_1764_; lean_object* v___x_1765_; 
v___x_1764_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__9));
v___x_1765_ = l_Lean_MessageData_ofFormat(v___x_1764_);
return v___x_1765_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__13(void){
_start:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; 
v___x_1769_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__12));
v___x_1770_ = l_Lean_MessageData_ofFormat(v___x_1769_);
return v___x_1770_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__17(void){
_start:
{
lean_object* v___x_1779_; lean_object* v___x_1780_; 
v___x_1779_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__16));
v___x_1780_ = l_Lean_stringToMessageData(v___x_1779_);
return v___x_1780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9(uint8_t v___x_1781_, lean_object* v___x_1782_, lean_object* v_as_1783_, size_t v_sz_1784_, size_t v_i_1785_, lean_object* v_b_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_){
_start:
{
lean_object* v_a_1791_; lean_object* v___y_1796_; uint8_t v___x_1815_; 
v___x_1815_ = lean_usize_dec_lt(v_i_1785_, v_sz_1784_);
if (v___x_1815_ == 0)
{
lean_object* v___x_1816_; 
lean_dec_ref(v___x_1782_);
v___x_1816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1816_, 0, v_b_1786_);
return v___x_1816_;
}
else
{
lean_object* v___x_1817_; lean_object* v_a_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; 
v___x_1817_ = lean_box(0);
v_a_1818_ = lean_array_uget_borrowed(v_as_1783_, v_i_1785_);
v___x_1819_ = lean_unsigned_to_nat(0u);
v___x_1820_ = l_Lean_Syntax_getArg(v_a_1818_, v___x_1819_);
v___x_1821_ = l_Lean_Syntax_getArg(v___x_1820_, v___x_1819_);
lean_dec(v___x_1820_);
v___x_1822_ = l_Lean_Syntax_getPos_x3f(v___x_1821_, v___x_1781_);
if (lean_obj_tag(v___x_1822_) == 1)
{
lean_object* v_val_1823_; uint8_t v___x_1824_; 
v_val_1823_ = lean_ctor_get(v___x_1822_, 0);
lean_inc(v_val_1823_);
lean_dec_ref_known(v___x_1822_, 1);
v___x_1824_ = l_Lean_Syntax_isMissing(v___x_1821_);
if (v___x_1824_ == 0)
{
lean_object* v___x_1825_; lean_object* v___x_1826_; uint8_t v___x_1827_; 
lean_inc(v___x_1821_);
v___x_1825_ = l_Lean_Syntax_getKind(v___x_1821_);
v___x_1826_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__1));
v___x_1827_ = lean_name_eq(v___x_1825_, v___x_1826_);
lean_dec(v___x_1825_);
if (v___x_1827_ == 0)
{
lean_dec(v_val_1823_);
lean_dec(v___x_1821_);
v_a_1791_ = v___x_1817_;
goto v___jp_1790_;
}
else
{
lean_object* v___x_1828_; lean_object* v_column_1829_; lean_object* v___x_1831_; uint8_t v_isShared_1832_; uint8_t v_isSharedCheck_1989_; 
lean_inc_ref(v___x_1782_);
v___x_1828_ = l_Lean_FileMap_toPosition(v___x_1782_, v_val_1823_);
lean_dec(v_val_1823_);
v_column_1829_ = lean_ctor_get(v___x_1828_, 1);
v_isSharedCheck_1989_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_1989_ == 0)
{
lean_object* v_unused_1990_; 
v_unused_1990_ = lean_ctor_get(v___x_1828_, 0);
lean_dec(v_unused_1990_);
v___x_1831_ = v___x_1828_;
v_isShared_1832_ = v_isSharedCheck_1989_;
goto v_resetjp_1830_;
}
else
{
lean_inc(v_column_1829_);
lean_dec(v___x_1828_);
v___x_1831_ = lean_box(0);
v_isShared_1832_ = v_isSharedCheck_1989_;
goto v_resetjp_1830_;
}
v_resetjp_1830_:
{
lean_object* v___x_1833_; 
lean_inc(v___x_1821_);
v___x_1833_ = lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2(v___x_1821_, v___y_1787_, v___y_1788_);
if (lean_obj_tag(v___x_1833_) == 0)
{
lean_object* v_a_1834_; lean_object* v___y_1836_; lean_object* v___y_1837_; lean_object* v___y_1838_; uint8_t v___y_1839_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v_startInclusive_1859_; lean_object* v_endExclusive_1860_; lean_object* v___x_1862_; uint8_t v_isShared_1863_; uint8_t v_isSharedCheck_1977_; 
v_a_1834_ = lean_ctor_get(v___x_1833_, 0);
lean_inc_n(v_a_1834_, 2);
lean_dec_ref_known(v___x_1833_, 1);
v___x_1856_ = lean_string_utf8_byte_size(v_a_1834_);
v___x_1857_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1857_, 0, v_a_1834_);
lean_ctor_set(v___x_1857_, 1, v___x_1819_);
lean_ctor_set(v___x_1857_, 2, v___x_1856_);
v___x_1858_ = l_String_Slice_trimAscii(v___x_1857_);
v_startInclusive_1859_ = lean_ctor_get(v___x_1858_, 1);
v_endExclusive_1860_ = lean_ctor_get(v___x_1858_, 2);
v_isSharedCheck_1977_ = !lean_is_exclusive(v___x_1858_);
if (v_isSharedCheck_1977_ == 0)
{
lean_object* v_unused_1978_; 
v_unused_1978_ = lean_ctor_get(v___x_1858_, 0);
lean_dec(v_unused_1978_);
v___x_1862_ = v___x_1858_;
v_isShared_1863_ = v_isSharedCheck_1977_;
goto v_resetjp_1861_;
}
else
{
lean_inc(v_endExclusive_1860_);
lean_inc(v_startInclusive_1859_);
lean_dec(v___x_1858_);
v___x_1862_ = lean_box(0);
v_isShared_1863_ = v_isSharedCheck_1977_;
goto v_resetjp_1861_;
}
v___jp_1835_:
{
if (v___y_1839_ == 0)
{
lean_dec_ref(v___y_1838_);
lean_dec(v_a_1834_);
v_a_1791_ = v___x_1817_;
goto v___jp_1790_;
}
else
{
lean_object* v___x_1840_; uint8_t v___x_1841_; 
v___x_1840_ = lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
v___x_1841_ = l_Lean_Linter_getLinterValue(v___x_1840_, v___y_1838_);
lean_dec_ref(v___y_1838_);
if (v___x_1841_ == 0)
{
lean_dec(v_a_1834_);
v_a_1791_ = v___x_1817_;
goto v___jp_1790_;
}
else
{
lean_object* v___x_1842_; lean_object* v___x_1843_; 
v___x_1842_ = lean_box(0);
v___x_1843_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax(v_a_1834_, v___x_1842_, v___y_1837_, v___y_1836_);
if (lean_obj_tag(v___x_1843_) == 0)
{
lean_object* v_a_1844_; size_t v_sz_1845_; size_t v___x_1846_; lean_object* v___x_1847_; 
v_a_1844_ = lean_ctor_get(v___x_1843_, 0);
lean_inc(v_a_1844_);
lean_dec_ref_known(v___x_1843_, 1);
v_sz_1845_ = lean_array_size(v_a_1844_);
v___x_1846_ = ((size_t)0ULL);
v___x_1847_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__3(v_a_1844_, v_sz_1845_, v___x_1846_, v___x_1817_, v___y_1837_, v___y_1836_);
lean_dec(v_a_1844_);
if (lean_obj_tag(v___x_1847_) == 0)
{
lean_dec_ref_known(v___x_1847_, 1);
v_a_1791_ = v___x_1817_;
goto v___jp_1790_;
}
else
{
lean_dec_ref(v___x_1782_);
return v___x_1847_;
}
}
else
{
lean_object* v_a_1848_; lean_object* v___x_1850_; uint8_t v_isShared_1851_; uint8_t v_isSharedCheck_1855_; 
lean_dec_ref(v___x_1782_);
v_a_1848_ = lean_ctor_get(v___x_1843_, 0);
v_isSharedCheck_1855_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1855_ == 0)
{
v___x_1850_ = v___x_1843_;
v_isShared_1851_ = v_isSharedCheck_1855_;
goto v_resetjp_1849_;
}
else
{
lean_inc(v_a_1848_);
lean_dec(v___x_1843_);
v___x_1850_ = lean_box(0);
v_isShared_1851_ = v_isSharedCheck_1855_;
goto v_resetjp_1849_;
}
v_resetjp_1849_:
{
lean_object* v___x_1853_; 
if (v_isShared_1851_ == 0)
{
v___x_1853_ = v___x_1850_;
goto v_reusejp_1852_;
}
else
{
lean_object* v_reuseFailAlloc_1854_; 
v_reuseFailAlloc_1854_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1854_, 0, v_a_1848_);
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
v_resetjp_1861_:
{
lean_object* v___x_1864_; uint8_t v___x_1865_; lean_object* v___y_1867_; lean_object* v___y_1868_; 
v___x_1864_ = lean_nat_sub(v_endExclusive_1860_, v_startInclusive_1859_);
lean_dec(v_startInclusive_1859_);
lean_dec(v_endExclusive_1860_);
v___x_1865_ = lean_nat_dec_eq(v___x_1864_, v___x_1819_);
lean_dec(v___x_1864_);
if (v___x_1865_ == 0)
{
lean_object* v___x_1886_; lean_object* v___y_1888_; lean_object* v___y_1889_; uint8_t v___y_1890_; lean_object* v___y_1896_; lean_object* v___y_1897_; lean_object* v___y_1898_; lean_object* v___y_1899_; lean_object* v___y_1905_; lean_object* v___y_1906_; lean_object* v___y_1930_; lean_object* v___y_1931_; uint8_t v___y_1932_; lean_object* v_str_1940_; lean_object* v_startPos_1941_; lean_object* v_stopPos_1942_; lean_object* v___y_1948_; 
v___x_1886_ = lp_mathlib_Mathlib_Linter_linter_style_docString;
if (lean_obj_tag(v___x_1821_) == 1)
{
lean_object* v_info_1952_; lean_object* v_kind_1953_; lean_object* v_args_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; uint8_t v___x_1957_; 
v_info_1952_ = lean_ctor_get(v___x_1821_, 0);
lean_inc(v_info_1952_);
v_kind_1953_ = lean_ctor_get(v___x_1821_, 1);
lean_inc(v_kind_1953_);
v_args_1954_ = lean_ctor_get(v___x_1821_, 2);
lean_inc_ref(v_args_1954_);
v___x_1955_ = lean_array_get_size(v_args_1954_);
v___x_1956_ = lean_unsigned_to_nat(2u);
v___x_1957_ = lean_nat_dec_eq(v___x_1955_, v___x_1956_);
if (v___x_1957_ == 0)
{
lean_object* v___x_1958_; 
lean_dec_ref(v_args_1954_);
lean_dec(v_kind_1953_);
lean_dec(v_info_1952_);
v___x_1958_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1(v___x_1821_);
v___y_1948_ = v___x_1958_;
goto v___jp_1947_;
}
else
{
lean_object* v___x_1959_; 
v___x_1959_ = lean_array_fget(v_args_1954_, v___x_1819_);
if (lean_obj_tag(v___x_1959_) == 2)
{
lean_object* v_info_1960_; lean_object* v___x_1961_; 
lean_dec_ref(v_args_1954_);
lean_dec(v_kind_1953_);
lean_dec(v_info_1952_);
v_info_1960_ = lean_ctor_get(v___x_1959_, 0);
lean_inc(v_info_1960_);
lean_dec_ref_known(v___x_1959_, 2);
v___x_1961_ = l_Lean_SourceInfo_getTrailing_x3f(v_info_1960_);
lean_dec(v_info_1960_);
if (lean_obj_tag(v___x_1961_) == 0)
{
lean_object* v___x_1962_; 
v___x_1962_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString_spec__0___redArg___closed__0));
v_str_1940_ = v___x_1962_;
v_startPos_1941_ = v___x_1819_;
v_stopPos_1942_ = v___x_1819_;
goto v___jp_1939_;
}
else
{
lean_object* v_val_1963_; 
v_val_1963_ = lean_ctor_get(v___x_1961_, 0);
lean_inc(v_val_1963_);
lean_dec_ref_known(v___x_1961_, 1);
v___y_1948_ = v_val_1963_;
goto v___jp_1947_;
}
}
else
{
lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; 
v___x_1964_ = lean_unsigned_to_nat(1u);
v___x_1965_ = lean_array_fget(v_args_1954_, v___x_1964_);
lean_dec_ref(v_args_1954_);
v___x_1966_ = lean_mk_empty_array_with_capacity(v___x_1956_);
v___x_1967_ = lean_array_push(v___x_1966_, v___x_1959_);
v___x_1968_ = lean_array_push(v___x_1967_, v___x_1965_);
v___x_1969_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1969_, 0, v_info_1952_);
lean_ctor_set(v___x_1969_, 1, v_kind_1953_);
lean_ctor_set(v___x_1969_, 2, v___x_1968_);
v___x_1970_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1(v___x_1969_);
lean_dec_ref_known(v___x_1969_, 3);
v___y_1948_ = v___x_1970_;
goto v___jp_1947_;
}
}
}
else
{
lean_object* v___x_1971_; 
v___x_1971_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__1(v___x_1821_);
v___y_1948_ = v___x_1971_;
goto v___jp_1947_;
}
v___jp_1887_:
{
if (v___y_1890_ == 0)
{
lean_dec(v___x_1821_);
v___y_1867_ = v___y_1889_;
v___y_1868_ = v___y_1888_;
goto v___jp_1866_;
}
else
{
lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; 
v___x_1891_ = lean_unsigned_to_nat(3u);
v___x_1892_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2(v___x_1827_, v___x_1821_, v___x_1865_, v___x_1891_);
lean_dec(v___x_1821_);
v___x_1893_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__4);
v___x_1894_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(v___x_1886_, v___x_1892_, v___x_1893_, v___y_1889_, v___y_1888_);
if (lean_obj_tag(v___x_1894_) == 0)
{
lean_dec_ref_known(v___x_1894_, 1);
v___y_1867_ = v___y_1889_;
v___y_1868_ = v___y_1888_;
goto v___jp_1866_;
}
else
{
lean_dec(v_a_1834_);
lean_dec_ref(v___x_1782_);
return v___x_1894_;
}
}
}
v___jp_1895_:
{
lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; uint8_t v___x_1903_; 
v___x_1900_ = lean_unsigned_to_nat(1u);
v___x_1901_ = lean_nat_add(v___y_1897_, v___x_1900_);
lean_dec(v___y_1897_);
v___x_1902_ = lean_string_length(v___y_1896_);
lean_dec_ref(v___y_1896_);
v___x_1903_ = lean_nat_dec_eq(v___x_1901_, v___x_1902_);
lean_dec(v___x_1901_);
if (v___x_1903_ == 0)
{
v___y_1888_ = v___y_1899_;
v___y_1889_ = v___y_1898_;
v___y_1890_ = v___x_1827_;
goto v___jp_1887_;
}
else
{
v___y_1888_ = v___y_1899_;
v___y_1889_ = v___y_1898_;
v___y_1890_ = v___x_1865_;
goto v___jp_1887_;
}
}
v___jp_1904_:
{
lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1910_; 
lean_inc(v_a_1834_);
v___x_1907_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString(v_column_1829_, v_a_1834_);
v___x_1908_ = lean_string_utf8_byte_size(v___x_1907_);
lean_inc_ref(v___x_1907_);
if (v_isShared_1863_ == 0)
{
lean_ctor_set(v___x_1862_, 2, v___x_1908_);
lean_ctor_set(v___x_1862_, 1, v___x_1819_);
lean_ctor_set(v___x_1862_, 0, v___x_1907_);
v___x_1910_ = v___x_1862_;
goto v_reusejp_1909_;
}
else
{
lean_object* v_reuseFailAlloc_1928_; 
v_reuseFailAlloc_1928_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1928_, 0, v___x_1907_);
lean_ctor_set(v_reuseFailAlloc_1928_, 1, v___x_1819_);
lean_ctor_set(v_reuseFailAlloc_1928_, 2, v___x_1908_);
v___x_1910_ = v_reuseFailAlloc_1928_;
goto v_reusejp_1909_;
}
v_reusejp_1909_:
{
lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; uint8_t v___x_1920_; 
v___x_1911_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__6(v___x_1910_, v___x_1908_);
lean_dec_ref(v___x_1910_);
v___x_1912_ = lean_string_utf8_extract_fast(v___x_1907_, v___x_1819_, v___x_1911_);
lean_dec(v___x_1911_);
v___x_1913_ = lean_string_length(v___x_1912_);
v___x_1914_ = lean_unsigned_to_nat(1u);
v___x_1915_ = lean_string_utf8_byte_size(v___x_1912_);
lean_inc_ref(v___x_1912_);
v___x_1916_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1916_, 0, v___x_1912_);
lean_ctor_set(v___x_1916_, 1, v___x_1819_);
lean_ctor_set(v___x_1916_, 2, v___x_1915_);
v___x_1917_ = l_String_Slice_Pos_prevn(v___x_1916_, v___x_1915_, v___x_1914_);
lean_dec_ref_known(v___x_1916_, 3);
v___x_1918_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1918_, 0, v___x_1912_);
lean_ctor_set(v___x_1918_, 1, v___x_1917_);
lean_ctor_set(v___x_1918_, 2, v___x_1915_);
v___x_1919_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__7);
v___x_1920_ = l_String_Slice_beq(v___x_1918_, v___x_1919_);
lean_dec_ref_known(v___x_1918_, 3);
if (v___x_1920_ == 0)
{
v___y_1896_ = v___x_1907_;
v___y_1897_ = v___x_1913_;
v___y_1898_ = v___y_1905_;
v___y_1899_ = v___y_1906_;
goto v___jp_1895_;
}
else
{
lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; 
v___x_1921_ = lean_string_length(v_a_1834_);
v___x_1922_ = lean_nat_sub(v___x_1921_, v___x_1913_);
v___x_1923_ = lean_unsigned_to_nat(3u);
v___x_1924_ = lean_nat_add(v___x_1922_, v___x_1923_);
lean_dec(v___x_1922_);
v___x_1925_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__2(v___x_1827_, v___x_1821_, v___x_1865_, v___x_1924_);
lean_dec(v___x_1924_);
v___x_1926_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__10, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__10_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__10);
v___x_1927_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(v___x_1886_, v___x_1925_, v___x_1926_, v___y_1905_, v___y_1906_);
if (lean_obj_tag(v___x_1927_) == 0)
{
lean_dec_ref_known(v___x_1927_, 1);
v___y_1896_ = v___x_1907_;
v___y_1897_ = v___x_1913_;
v___y_1898_ = v___y_1905_;
v___y_1899_ = v___y_1906_;
goto v___jp_1895_;
}
else
{
lean_dec_ref(v___x_1907_);
lean_dec(v_a_1834_);
lean_dec(v___x_1821_);
lean_dec_ref(v___x_1782_);
return v___x_1927_;
}
}
}
}
v___jp_1929_:
{
if (v___y_1932_ == 0)
{
lean_dec(v___y_1931_);
lean_dec(v___y_1930_);
lean_del_object(v___x_1831_);
v___y_1905_ = v___y_1787_;
v___y_1906_ = v___y_1788_;
goto v___jp_1904_;
}
else
{
lean_object* v___x_1934_; 
if (v_isShared_1832_ == 0)
{
lean_ctor_set(v___x_1831_, 1, v___y_1930_);
lean_ctor_set(v___x_1831_, 0, v___y_1931_);
v___x_1934_ = v___x_1831_;
goto v_reusejp_1933_;
}
else
{
lean_object* v_reuseFailAlloc_1938_; 
v_reuseFailAlloc_1938_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1938_, 0, v___y_1931_);
lean_ctor_set(v_reuseFailAlloc_1938_, 1, v___y_1930_);
v___x_1934_ = v_reuseFailAlloc_1938_;
goto v_reusejp_1933_;
}
v_reusejp_1933_:
{
lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; 
v___x_1935_ = l_Lean_Syntax_ofRange(v___x_1934_, v___x_1827_);
v___x_1936_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__13, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__13_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__13);
v___x_1937_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(v___x_1886_, v___x_1935_, v___x_1936_, v___y_1787_, v___y_1788_);
if (lean_obj_tag(v___x_1937_) == 0)
{
lean_dec_ref_known(v___x_1937_, 1);
v___y_1905_ = v___y_1787_;
v___y_1906_ = v___y_1788_;
goto v___jp_1904_;
}
else
{
lean_del_object(v___x_1862_);
lean_dec(v_a_1834_);
lean_dec(v_column_1829_);
lean_dec(v___x_1821_);
lean_dec_ref(v___x_1782_);
return v___x_1937_;
}
}
}
}
v___jp_1939_:
{
lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; uint8_t v___x_1946_; 
v___x_1943_ = lean_string_utf8_extract(v_str_1940_, v_startPos_1941_, v_stopPos_1942_);
lean_dec_ref(v_str_1940_);
lean_inc(v_column_1829_);
v___x_1944_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString(v_column_1829_, v___x_1943_);
v___x_1945_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__15));
v___x_1946_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__8(v___x_1945_, v___x_1944_);
lean_dec_ref(v___x_1944_);
if (v___x_1946_ == 0)
{
v___y_1930_ = v_stopPos_1942_;
v___y_1931_ = v_startPos_1941_;
v___y_1932_ = v___x_1827_;
goto v___jp_1929_;
}
else
{
v___y_1930_ = v_stopPos_1942_;
v___y_1931_ = v_startPos_1941_;
v___y_1932_ = v___x_1865_;
goto v___jp_1929_;
}
}
v___jp_1947_:
{
lean_object* v_str_1949_; lean_object* v_startPos_1950_; lean_object* v_stopPos_1951_; 
v_str_1949_ = lean_ctor_get(v___y_1948_, 0);
lean_inc_ref(v_str_1949_);
v_startPos_1950_ = lean_ctor_get(v___y_1948_, 1);
lean_inc(v_startPos_1950_);
v_stopPos_1951_ = lean_ctor_get(v___y_1948_, 2);
lean_inc(v_stopPos_1951_);
lean_dec_ref(v___y_1948_);
v_str_1940_ = v_str_1949_;
v_startPos_1941_ = v_startPos_1950_;
v_stopPos_1942_ = v_stopPos_1951_;
goto v___jp_1939_;
}
}
else
{
lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; 
lean_del_object(v___x_1862_);
lean_dec(v_a_1834_);
lean_del_object(v___x_1831_);
lean_dec(v_column_1829_);
v___x_1972_ = lp_mathlib_Mathlib_Linter_linter_style_docString_empty;
v___x_1973_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__17, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__17_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___closed__17);
v___x_1974_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__7(v___x_1972_, v___x_1821_, v___x_1973_, v___y_1787_, v___y_1788_);
if (lean_obj_tag(v___x_1974_) == 0)
{
lean_object* v_a_1975_; lean_object* v___x_1976_; 
v_a_1975_ = lean_ctor_get(v___x_1974_, 0);
lean_inc(v_a_1975_);
lean_dec_ref_known(v___x_1974_, 1);
v___x_1976_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0(v___x_1817_, v_a_1975_, v___y_1787_, v___y_1788_);
v___y_1796_ = v___x_1976_;
goto v___jp_1795_;
}
else
{
lean_dec_ref(v___x_1782_);
return v___x_1974_;
}
}
v___jp_1866_:
{
lean_object* v___x_1869_; lean_object* v___x_1870_; 
v___x_1869_ = lean_st_ref_get(v___y_1868_);
v___x_1870_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_1867_, v___y_1868_);
if (lean_obj_tag(v___x_1870_) == 0)
{
lean_object* v_a_1871_; lean_object* v_scopes_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v_opts_1875_; lean_object* v___x_1876_; uint8_t v___x_1877_; 
v_a_1871_ = lean_ctor_get(v___x_1870_, 0);
lean_inc(v_a_1871_);
lean_dec_ref_known(v___x_1870_, 1);
v_scopes_1872_ = lean_ctor_get(v___x_1869_, 2);
lean_inc(v_scopes_1872_);
lean_dec(v___x_1869_);
v___x_1873_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1874_ = l_List_head_x21___redArg(v___x_1873_, v_scopes_1872_);
lean_dec(v_scopes_1872_);
v_opts_1875_ = lean_ctor_get(v___x_1874_, 1);
lean_inc_ref(v_opts_1875_);
lean_dec(v___x_1874_);
v___x_1876_ = l_Lean_doc_verso;
v___x_1877_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(v_opts_1875_, v___x_1876_);
lean_dec_ref(v_opts_1875_);
if (v___x_1877_ == 0)
{
v___y_1836_ = v___y_1868_;
v___y_1837_ = v___y_1867_;
v___y_1838_ = v_a_1871_;
v___y_1839_ = v___x_1827_;
goto v___jp_1835_;
}
else
{
v___y_1836_ = v___y_1868_;
v___y_1837_ = v___y_1867_;
v___y_1838_ = v_a_1871_;
v___y_1839_ = v___x_1865_;
goto v___jp_1835_;
}
}
else
{
lean_object* v_a_1878_; lean_object* v___x_1880_; uint8_t v_isShared_1881_; uint8_t v_isSharedCheck_1885_; 
lean_dec(v___x_1869_);
lean_dec(v_a_1834_);
lean_dec_ref(v___x_1782_);
v_a_1878_ = lean_ctor_get(v___x_1870_, 0);
v_isSharedCheck_1885_ = !lean_is_exclusive(v___x_1870_);
if (v_isSharedCheck_1885_ == 0)
{
v___x_1880_ = v___x_1870_;
v_isShared_1881_ = v_isSharedCheck_1885_;
goto v_resetjp_1879_;
}
else
{
lean_inc(v_a_1878_);
lean_dec(v___x_1870_);
v___x_1880_ = lean_box(0);
v_isShared_1881_ = v_isSharedCheck_1885_;
goto v_resetjp_1879_;
}
v_resetjp_1879_:
{
lean_object* v___x_1883_; 
if (v_isShared_1881_ == 0)
{
v___x_1883_ = v___x_1880_;
goto v_reusejp_1882_;
}
else
{
lean_object* v_reuseFailAlloc_1884_; 
v_reuseFailAlloc_1884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1884_, 0, v_a_1878_);
v___x_1883_ = v_reuseFailAlloc_1884_;
goto v_reusejp_1882_;
}
v_reusejp_1882_:
{
return v___x_1883_;
}
}
}
}
}
}
else
{
lean_object* v_a_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_1988_; 
lean_del_object(v___x_1831_);
lean_dec(v_column_1829_);
lean_dec(v___x_1821_);
v_a_1979_ = lean_ctor_get(v___x_1833_, 0);
v_isSharedCheck_1988_ = !lean_is_exclusive(v___x_1833_);
if (v_isSharedCheck_1988_ == 0)
{
v___x_1981_ = v___x_1833_;
v_isShared_1982_ = v_isSharedCheck_1988_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_a_1979_);
lean_dec(v___x_1833_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_1988_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
uint8_t v___x_1983_; 
v___x_1983_ = l_Lean_Exception_isInterrupt(v_a_1979_);
if (v___x_1983_ == 0)
{
lean_object* v___x_1984_; 
lean_del_object(v___x_1981_);
lean_dec(v_a_1979_);
v___x_1984_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___lam__0(v___x_1817_, v___x_1817_, v___y_1787_, v___y_1788_);
v___y_1796_ = v___x_1984_;
goto v___jp_1795_;
}
else
{
lean_object* v___x_1986_; 
lean_dec_ref(v___x_1782_);
if (v_isShared_1982_ == 0)
{
v___x_1986_ = v___x_1981_;
goto v_reusejp_1985_;
}
else
{
lean_object* v_reuseFailAlloc_1987_; 
v_reuseFailAlloc_1987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1987_, 0, v_a_1979_);
v___x_1986_ = v_reuseFailAlloc_1987_;
goto v_reusejp_1985_;
}
v_reusejp_1985_:
{
return v___x_1986_;
}
}
}
}
}
}
}
else
{
lean_dec(v_val_1823_);
lean_dec(v___x_1821_);
v_a_1791_ = v___x_1817_;
goto v___jp_1790_;
}
}
else
{
lean_dec(v___x_1822_);
lean_dec(v___x_1821_);
v_a_1791_ = v___x_1817_;
goto v___jp_1790_;
}
}
v___jp_1790_:
{
size_t v___x_1792_; size_t v___x_1793_; 
v___x_1792_ = ((size_t)1ULL);
v___x_1793_ = lean_usize_add(v_i_1785_, v___x_1792_);
v_i_1785_ = v___x_1793_;
v_b_1786_ = v_a_1791_;
goto _start;
}
v___jp_1795_:
{
if (lean_obj_tag(v___y_1796_) == 0)
{
lean_object* v_a_1797_; lean_object* v___x_1799_; uint8_t v_isShared_1800_; uint8_t v_isSharedCheck_1806_; 
v_a_1797_ = lean_ctor_get(v___y_1796_, 0);
v_isSharedCheck_1806_ = !lean_is_exclusive(v___y_1796_);
if (v_isSharedCheck_1806_ == 0)
{
v___x_1799_ = v___y_1796_;
v_isShared_1800_ = v_isSharedCheck_1806_;
goto v_resetjp_1798_;
}
else
{
lean_inc(v_a_1797_);
lean_dec(v___y_1796_);
v___x_1799_ = lean_box(0);
v_isShared_1800_ = v_isSharedCheck_1806_;
goto v_resetjp_1798_;
}
v_resetjp_1798_:
{
if (lean_obj_tag(v_a_1797_) == 0)
{
lean_object* v_a_1801_; lean_object* v___x_1803_; 
lean_dec_ref(v___x_1782_);
v_a_1801_ = lean_ctor_get(v_a_1797_, 0);
lean_inc(v_a_1801_);
lean_dec_ref_known(v_a_1797_, 1);
if (v_isShared_1800_ == 0)
{
lean_ctor_set(v___x_1799_, 0, v_a_1801_);
v___x_1803_ = v___x_1799_;
goto v_reusejp_1802_;
}
else
{
lean_object* v_reuseFailAlloc_1804_; 
v_reuseFailAlloc_1804_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1804_, 0, v_a_1801_);
v___x_1803_ = v_reuseFailAlloc_1804_;
goto v_reusejp_1802_;
}
v_reusejp_1802_:
{
return v___x_1803_;
}
}
else
{
lean_object* v_a_1805_; 
lean_del_object(v___x_1799_);
v_a_1805_ = lean_ctor_get(v_a_1797_, 0);
lean_inc(v_a_1805_);
lean_dec_ref_known(v_a_1797_, 1);
v_a_1791_ = v_a_1805_;
goto v___jp_1790_;
}
}
}
else
{
lean_object* v_a_1807_; lean_object* v___x_1809_; uint8_t v_isShared_1810_; uint8_t v_isSharedCheck_1814_; 
lean_dec_ref(v___x_1782_);
v_a_1807_ = lean_ctor_get(v___y_1796_, 0);
v_isSharedCheck_1814_ = !lean_is_exclusive(v___y_1796_);
if (v_isSharedCheck_1814_ == 0)
{
v___x_1809_ = v___y_1796_;
v_isShared_1810_ = v_isSharedCheck_1814_;
goto v_resetjp_1808_;
}
else
{
lean_inc(v_a_1807_);
lean_dec(v___y_1796_);
v___x_1809_ = lean_box(0);
v_isShared_1810_ = v_isSharedCheck_1814_;
goto v_resetjp_1808_;
}
v_resetjp_1808_:
{
lean_object* v___x_1812_; 
if (v_isShared_1810_ == 0)
{
v___x_1812_ = v___x_1809_;
goto v_reusejp_1811_;
}
else
{
lean_object* v_reuseFailAlloc_1813_; 
v_reuseFailAlloc_1813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1813_, 0, v_a_1807_);
v___x_1812_ = v_reuseFailAlloc_1813_;
goto v_reusejp_1811_;
}
v_reusejp_1811_:
{
return v___x_1812_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9___boxed(lean_object* v___x_1991_, lean_object* v___x_1992_, lean_object* v_as_1993_, lean_object* v_sz_1994_, lean_object* v_i_1995_, lean_object* v_b_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_){
_start:
{
uint8_t v___x_22091__boxed_2000_; size_t v_sz_boxed_2001_; size_t v_i_boxed_2002_; lean_object* v_res_2003_; 
v___x_22091__boxed_2000_ = lean_unbox(v___x_1991_);
v_sz_boxed_2001_ = lean_unbox_usize(v_sz_1994_);
lean_dec(v_sz_1994_);
v_i_boxed_2002_ = lean_unbox_usize(v_i_1995_);
lean_dec(v_i_1995_);
v_res_2003_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9(v___x_22091__boxed_2000_, v___x_1992_, v_as_1993_, v_sz_boxed_2001_, v_i_boxed_2002_, v_b_1996_, v___y_1997_, v___y_1998_);
lean_dec(v___y_1998_);
lean_dec_ref(v___y_1997_);
lean_dec_ref(v_as_1993_);
return v_res_2003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___lam__0(lean_object* v_stx_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_){
_start:
{
lean_object* v___x_2008_; lean_object* v_a_2009_; lean_object* v___x_2010_; lean_object* v_a_2011_; lean_object* v___x_2013_; uint8_t v_isShared_2014_; uint8_t v_isSharedCheck_2055_; 
v___x_2008_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_2005_, v___y_2006_);
v_a_2009_ = lean_ctor_get(v___x_2008_, 0);
lean_inc(v_a_2009_);
lean_dec_ref(v___x_2008_);
v___x_2010_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_2005_, v___y_2006_);
v_a_2011_ = lean_ctor_get(v___x_2010_, 0);
v_isSharedCheck_2055_ = !lean_is_exclusive(v___x_2010_);
if (v_isSharedCheck_2055_ == 0)
{
v___x_2013_ = v___x_2010_;
v_isShared_2014_ = v_isSharedCheck_2055_;
goto v_resetjp_2012_;
}
else
{
lean_inc(v_a_2011_);
lean_dec(v___x_2010_);
v___x_2013_ = lean_box(0);
v_isShared_2014_ = v_isSharedCheck_2055_;
goto v_resetjp_2012_;
}
v_resetjp_2012_:
{
lean_object* v___x_2015_; lean_object* v_a_2016_; lean_object* v___x_2018_; uint8_t v_isShared_2019_; uint8_t v_isSharedCheck_2054_; 
v___x_2015_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_2005_, v___y_2006_);
v_a_2016_ = lean_ctor_get(v___x_2015_, 0);
v_isSharedCheck_2054_ = !lean_is_exclusive(v___x_2015_);
if (v_isSharedCheck_2054_ == 0)
{
v___x_2018_ = v___x_2015_;
v_isShared_2019_ = v_isSharedCheck_2054_;
goto v_resetjp_2017_;
}
else
{
lean_inc(v_a_2016_);
lean_dec(v___x_2015_);
v___x_2018_ = lean_box(0);
v_isShared_2019_ = v_isSharedCheck_2054_;
goto v_resetjp_2017_;
}
v_resetjp_2017_:
{
uint8_t v___y_2043_; lean_object* v___x_2050_; uint8_t v___x_2051_; 
v___x_2050_ = lp_mathlib_Mathlib_Linter_linter_style_docString;
v___x_2051_ = l_Lean_Linter_getLinterValue(v___x_2050_, v_a_2009_);
lean_dec(v_a_2009_);
if (v___x_2051_ == 0)
{
lean_object* v___x_2052_; uint8_t v___x_2053_; 
v___x_2052_ = lp_mathlib_Mathlib_Linter_linter_style_docString_empty;
v___x_2053_ = l_Lean_Linter_getLinterValue(v___x_2052_, v_a_2011_);
lean_dec(v_a_2011_);
v___y_2043_ = v___x_2053_;
goto v___jp_2042_;
}
else
{
lean_dec(v_a_2011_);
v___y_2043_ = v___x_2051_;
goto v___jp_2042_;
}
v___jp_2020_:
{
lean_object* v___x_2021_; lean_object* v_messages_2022_; uint8_t v___x_2023_; 
v___x_2021_ = lean_st_ref_get(v___y_2006_);
v_messages_2022_ = lean_ctor_get(v___x_2021_, 1);
lean_inc_ref(v_messages_2022_);
lean_dec(v___x_2021_);
v___x_2023_ = l_Lean_MessageLog_hasErrors(v_messages_2022_);
lean_dec_ref(v_messages_2022_);
if (v___x_2023_ == 0)
{
lean_object* v_fileMap_2024_; lean_object* v___x_2025_; lean_object* v___x_2026_; size_t v_sz_2027_; size_t v___x_2028_; lean_object* v___x_2029_; 
lean_del_object(v___x_2018_);
v_fileMap_2024_ = lean_ctor_get(v___y_2005_, 1);
v___x_2025_ = lp_mathlib_Mathlib_Linter_getDeclModifiers(v_stx_2004_);
v___x_2026_ = lean_box(0);
v_sz_2027_ = lean_array_size(v___x_2025_);
v___x_2028_ = ((size_t)0ULL);
lean_inc_ref(v_fileMap_2024_);
v___x_2029_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__9(v___x_2023_, v_fileMap_2024_, v___x_2025_, v_sz_2027_, v___x_2028_, v___x_2026_, v___y_2005_, v___y_2006_);
lean_dec_ref(v___x_2025_);
if (lean_obj_tag(v___x_2029_) == 0)
{
lean_object* v___x_2031_; uint8_t v_isShared_2032_; uint8_t v_isSharedCheck_2036_; 
v_isSharedCheck_2036_ = !lean_is_exclusive(v___x_2029_);
if (v_isSharedCheck_2036_ == 0)
{
lean_object* v_unused_2037_; 
v_unused_2037_ = lean_ctor_get(v___x_2029_, 0);
lean_dec(v_unused_2037_);
v___x_2031_ = v___x_2029_;
v_isShared_2032_ = v_isSharedCheck_2036_;
goto v_resetjp_2030_;
}
else
{
lean_dec(v___x_2029_);
v___x_2031_ = lean_box(0);
v_isShared_2032_ = v_isSharedCheck_2036_;
goto v_resetjp_2030_;
}
v_resetjp_2030_:
{
lean_object* v___x_2034_; 
if (v_isShared_2032_ == 0)
{
lean_ctor_set(v___x_2031_, 0, v___x_2026_);
v___x_2034_ = v___x_2031_;
goto v_reusejp_2033_;
}
else
{
lean_object* v_reuseFailAlloc_2035_; 
v_reuseFailAlloc_2035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2035_, 0, v___x_2026_);
v___x_2034_ = v_reuseFailAlloc_2035_;
goto v_reusejp_2033_;
}
v_reusejp_2033_:
{
return v___x_2034_;
}
}
}
else
{
return v___x_2029_;
}
}
else
{
lean_object* v___x_2038_; lean_object* v___x_2040_; 
lean_dec(v_stx_2004_);
v___x_2038_ = lean_box(0);
if (v_isShared_2019_ == 0)
{
lean_ctor_set(v___x_2018_, 0, v___x_2038_);
v___x_2040_ = v___x_2018_;
goto v_reusejp_2039_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v___x_2038_);
v___x_2040_ = v_reuseFailAlloc_2041_;
goto v_reusejp_2039_;
}
v_reusejp_2039_:
{
return v___x_2040_;
}
}
}
v___jp_2042_:
{
if (v___y_2043_ == 0)
{
lean_object* v___x_2044_; uint8_t v___x_2045_; 
v___x_2044_ = lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
v___x_2045_ = l_Lean_Linter_getLinterValue(v___x_2044_, v_a_2016_);
lean_dec(v_a_2016_);
if (v___x_2045_ == 0)
{
lean_object* v___x_2046_; lean_object* v___x_2048_; 
lean_del_object(v___x_2018_);
lean_dec(v_stx_2004_);
v___x_2046_ = lean_box(0);
if (v_isShared_2014_ == 0)
{
lean_ctor_set(v___x_2013_, 0, v___x_2046_);
v___x_2048_ = v___x_2013_;
goto v_reusejp_2047_;
}
else
{
lean_object* v_reuseFailAlloc_2049_; 
v_reuseFailAlloc_2049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2049_, 0, v___x_2046_);
v___x_2048_ = v_reuseFailAlloc_2049_;
goto v_reusejp_2047_;
}
v_reusejp_2047_:
{
return v___x_2048_;
}
}
else
{
lean_del_object(v___x_2013_);
goto v___jp_2020_;
}
}
else
{
lean_dec(v_a_2016_);
lean_del_object(v___x_2013_);
goto v___jp_2020_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___lam__0___boxed(lean_object* v_stx_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_){
_start:
{
lean_object* v_res_2060_; 
v_res_2060_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter___lam__0(v_stx_2056_, v___y_2057_, v___y_2058_);
lean_dec(v___y_2058_);
lean_dec_ref(v___y_2057_);
return v_res_2060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0(lean_object* v_o_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_){
_start:
{
lean_object* v___x_2107_; 
v___x_2107_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___redArg(v_o_2103_, v___y_2105_);
return v___x_2107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0___boxed(lean_object* v_o_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_){
_start:
{
lean_object* v_res_2112_; 
v_res_2112_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0_spec__0(v_o_2108_, v___y_2109_, v___y_2110_);
lean_dec(v___y_2110_);
lean_dec_ref(v___y_2109_);
return v_res_2112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4(lean_object* v_00_u03b1_2113_, lean_object* v_ref_2114_, lean_object* v_msg_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_){
_start:
{
lean_object* v___x_2119_; 
v___x_2119_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___redArg(v_ref_2114_, v_msg_2115_, v___y_2116_, v___y_2117_);
return v___x_2119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4___boxed(lean_object* v_00_u03b1_2120_, lean_object* v_ref_2121_, lean_object* v_msg_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_){
_start:
{
lean_object* v_res_2126_; 
v_res_2126_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4(v_00_u03b1_2120_, v_ref_2121_, v_msg_2122_, v___y_2123_, v___y_2124_);
lean_dec(v___y_2124_);
lean_dec_ref(v___y_2123_);
lean_dec(v_ref_2121_);
return v_res_2126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12(lean_object* v_msgData_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_){
_start:
{
lean_object* v___x_2131_; 
v___x_2131_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___redArg(v_msgData_2127_, v___y_2129_);
return v___x_2131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12___boxed(lean_object* v_msgData_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_){
_start:
{
lean_object* v_res_2136_; 
v_res_2136_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1_spec__2_spec__5_spec__12(v_msgData_2132_, v___y_2133_, v___y_2134_);
lean_dec(v___y_2134_);
lean_dec_ref(v___y_2133_);
return v_res_2136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8(lean_object* v_00_u03b1_2137_, lean_object* v_msg_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_){
_start:
{
lean_object* v___x_2142_; 
v___x_2142_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___redArg(v_msg_2138_, v___y_2139_, v___y_2140_);
return v___x_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8___boxed(lean_object* v_00_u03b1_2143_, lean_object* v_msg_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_){
_start:
{
lean_object* v_res_2148_; 
v_res_2148_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8(v_00_u03b1_2143_, v_msg_2144_, v___y_2145_, v___y_2146_);
lean_dec(v___y_2146_);
lean_dec_ref(v___y_2145_);
return v_res_2148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15(lean_object* v_msgData_2149_, lean_object* v_macroStack_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_){
_start:
{
lean_object* v___x_2154_; 
v___x_2154_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___redArg(v_msgData_2149_, v_macroStack_2150_, v___y_2152_);
return v___x_2154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15___boxed(lean_object* v_msgData_2155_, lean_object* v_macroStack_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_){
_start:
{
lean_object* v_res_2160_; 
v_res_2160_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2_spec__4_spec__8_spec__15(v_msgData_2155_, v_macroStack_2156_, v___y_2157_, v___y_2158_);
lean_dec(v___y_2158_);
lean_dec_ref(v___y_2157_);
return v_res_2160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3364454516____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2162_; lean_object* v___x_2163_; 
v___x_2162_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter));
v___x_2163_ = l_Lean_Elab_Command_addLinter(v___x_2162_);
return v___x_2163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3364454516____hygCtx___hyg_2____boxed(lean_object* v_a_2164_){
_start:
{
lean_object* v_res_2165_; 
v_res_2165_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3364454516____hygCtx___hyg_2_();
return v_res_2165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get_x3f___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__1(lean_object* v_opts_2166_, lean_object* v_opt_2167_){
_start:
{
lean_object* v_name_2168_; lean_object* v_map_2169_; lean_object* v___x_2170_; 
v_name_2168_ = lean_ctor_get(v_opt_2167_, 0);
v_map_2169_ = lean_ctor_get(v_opts_2166_, 0);
v___x_2170_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2169_, v_name_2168_);
if (lean_obj_tag(v___x_2170_) == 0)
{
lean_object* v___x_2171_; 
v___x_2171_ = lean_box(0);
return v___x_2171_;
}
else
{
lean_object* v_val_2172_; lean_object* v___x_2174_; uint8_t v_isShared_2175_; uint8_t v_isSharedCheck_2182_; 
v_val_2172_ = lean_ctor_get(v___x_2170_, 0);
v_isSharedCheck_2182_ = !lean_is_exclusive(v___x_2170_);
if (v_isSharedCheck_2182_ == 0)
{
v___x_2174_ = v___x_2170_;
v_isShared_2175_ = v_isSharedCheck_2182_;
goto v_resetjp_2173_;
}
else
{
lean_inc(v_val_2172_);
lean_dec(v___x_2170_);
v___x_2174_ = lean_box(0);
v_isShared_2175_ = v_isSharedCheck_2182_;
goto v_resetjp_2173_;
}
v_resetjp_2173_:
{
if (lean_obj_tag(v_val_2172_) == 1)
{
uint8_t v_v_2176_; lean_object* v___x_2177_; lean_object* v___x_2179_; 
v_v_2176_ = lean_ctor_get_uint8(v_val_2172_, 0);
lean_dec_ref_known(v_val_2172_, 0);
v___x_2177_ = lean_box(v_v_2176_);
if (v_isShared_2175_ == 0)
{
lean_ctor_set(v___x_2174_, 0, v___x_2177_);
v___x_2179_ = v___x_2174_;
goto v_reusejp_2178_;
}
else
{
lean_object* v_reuseFailAlloc_2180_; 
v_reuseFailAlloc_2180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2180_, 0, v___x_2177_);
v___x_2179_ = v_reuseFailAlloc_2180_;
goto v_reusejp_2178_;
}
v_reusejp_2178_:
{
return v___x_2179_;
}
}
else
{
lean_object* v___x_2181_; 
lean_del_object(v___x_2174_);
lean_dec(v_val_2172_);
v___x_2181_ = lean_box(0);
return v___x_2181_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get_x3f___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__1___boxed(lean_object* v_opts_2183_, lean_object* v_opt_2184_){
_start:
{
lean_object* v_res_2185_; 
v_res_2185_ = lp_mathlib_Lean_Option_get_x3f___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__1(v_opts_2183_, v_opt_2184_);
lean_dec_ref(v_opt_2184_);
lean_dec_ref(v_opts_2183_);
return v_res_2185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__0(lean_object* v_as_2186_, size_t v_sz_2187_, size_t v_i_2188_, lean_object* v_b_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_){
_start:
{
uint8_t v___x_2193_; 
v___x_2193_ = lean_usize_dec_lt(v_i_2188_, v_sz_2187_);
if (v___x_2193_ == 0)
{
lean_object* v___x_2194_; 
v___x_2194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2194_, 0, v_b_2189_);
return v___x_2194_;
}
else
{
lean_object* v_a_2195_; lean_object* v_snd_2196_; lean_object* v_fst_2197_; lean_object* v_snd_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; 
v_a_2195_ = lean_array_uget_borrowed(v_as_2186_, v_i_2188_);
v_snd_2196_ = lean_ctor_get(v_a_2195_, 1);
v_fst_2197_ = lean_ctor_get(v_snd_2196_, 0);
v_snd_2198_ = lean_ctor_get(v_snd_2196_, 1);
v___x_2199_ = lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
v___x_2200_ = l_Lean_Parser_SyntaxStack_back(v_fst_2197_);
lean_inc(v_snd_2198_);
v___x_2201_ = l_Lean_Parser_Error_toString(v_snd_2198_);
v___x_2202_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2202_, 0, v___x_2201_);
v___x_2203_ = l_Lean_MessageData_ofFormat(v___x_2202_);
v___x_2204_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__1(v___x_2199_, v___x_2200_, v___x_2203_, v___y_2190_, v___y_2191_);
if (lean_obj_tag(v___x_2204_) == 0)
{
lean_object* v___x_2205_; size_t v___x_2206_; size_t v___x_2207_; 
lean_dec_ref_known(v___x_2204_, 1);
v___x_2205_ = lean_box(0);
v___x_2206_ = ((size_t)1ULL);
v___x_2207_ = lean_usize_add(v_i_2188_, v___x_2206_);
v_i_2188_ = v___x_2207_;
v_b_2189_ = v___x_2205_;
goto _start;
}
else
{
return v___x_2204_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__0___boxed(lean_object* v_as_2209_, lean_object* v_sz_2210_, lean_object* v_i_2211_, lean_object* v_b_2212_, lean_object* v___y_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_){
_start:
{
size_t v_sz_boxed_2216_; size_t v_i_boxed_2217_; lean_object* v_res_2218_; 
v_sz_boxed_2216_ = lean_unbox_usize(v_sz_2210_);
lean_dec(v_sz_2210_);
v_i_boxed_2217_ = lean_unbox_usize(v_i_2211_);
lean_dec(v_i_2211_);
v_res_2218_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__0(v_as_2209_, v_sz_boxed_2216_, v_i_boxed_2217_, v_b_2212_, v___y_2213_, v___y_2214_);
lean_dec(v___y_2214_);
lean_dec_ref(v___y_2213_);
lean_dec_ref(v_as_2209_);
return v_res_2218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0(lean_object* v_stx_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_){
_start:
{
lean_object* v_a_2225_; lean_object* v___x_2233_; lean_object* v_a_2234_; lean_object* v___x_2236_; uint8_t v_isShared_2237_; uint8_t v_isSharedCheck_2319_; 
v___x_2233_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__0(v___y_2221_, v___y_2222_);
v_a_2234_ = lean_ctor_get(v___x_2233_, 0);
v_isSharedCheck_2319_ = !lean_is_exclusive(v___x_2233_);
if (v_isSharedCheck_2319_ == 0)
{
v___x_2236_ = v___x_2233_;
v_isShared_2237_ = v_isSharedCheck_2319_;
goto v_resetjp_2235_;
}
else
{
lean_inc(v_a_2234_);
lean_dec(v___x_2233_);
v___x_2236_ = lean_box(0);
v_isShared_2237_ = v_isSharedCheck_2319_;
goto v_resetjp_2235_;
}
v___jp_2224_:
{
uint8_t v___x_2226_; 
v___x_2226_ = l_Lean_Exception_isInterrupt(v_a_2225_);
if (v___x_2226_ == 0)
{
lean_object* v___x_2227_; lean_object* v___x_2228_; 
lean_dec_ref(v_a_2225_);
v___x_2227_ = lean_box(0);
v___x_2228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2228_, 0, v___x_2227_);
return v___x_2228_;
}
else
{
lean_object* v___x_2229_; 
v___x_2229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2229_, 0, v_a_2225_);
return v___x_2229_;
}
}
v___jp_2230_:
{
lean_object* v___x_2231_; lean_object* v___x_2232_; 
v___x_2231_ = lean_box(0);
v___x_2232_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2232_, 0, v___x_2231_);
return v___x_2232_;
}
v_resetjp_2235_:
{
lean_object* v___x_2238_; uint8_t v___x_2239_; 
v___x_2238_ = lp_mathlib_Mathlib_Linter_linter_style_docStringVerso;
v___x_2239_ = l_Lean_Linter_getLinterValue(v___x_2238_, v_a_2234_);
lean_dec(v_a_2234_);
if (v___x_2239_ == 0)
{
lean_object* v___x_2240_; lean_object* v___x_2242_; 
lean_dec(v_stx_2220_);
v___x_2240_ = lean_box(0);
if (v_isShared_2237_ == 0)
{
lean_ctor_set(v___x_2236_, 0, v___x_2240_);
v___x_2242_ = v___x_2236_;
goto v_reusejp_2241_;
}
else
{
lean_object* v_reuseFailAlloc_2243_; 
v_reuseFailAlloc_2243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2243_, 0, v___x_2240_);
v___x_2242_ = v_reuseFailAlloc_2243_;
goto v_reusejp_2241_;
}
v_reusejp_2241_:
{
return v___x_2242_;
}
}
else
{
lean_object* v___x_2244_; uint8_t v___y_2246_; lean_object* v_scopes_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v_opts_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; 
v___x_2244_ = lean_st_ref_get(v___y_2222_);
v_scopes_2309_ = lean_ctor_get(v___x_2244_, 2);
lean_inc(v_scopes_2309_);
lean_dec(v___x_2244_);
v___x_2310_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2311_ = l_List_head_x21___redArg(v___x_2310_, v_scopes_2309_);
lean_dec(v_scopes_2309_);
v_opts_2312_ = lean_ctor_get(v___x_2311_, 1);
lean_inc_ref(v_opts_2312_);
lean_dec(v___x_2311_);
v___x_2313_ = l_Lean_doc_verso_module;
v___x_2314_ = lp_mathlib_Lean_Option_get_x3f___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__1(v_opts_2312_, v___x_2313_);
if (lean_obj_tag(v___x_2314_) == 0)
{
lean_object* v___x_2315_; uint8_t v___x_2316_; 
v___x_2315_ = l_Lean_doc_verso;
v___x_2316_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__4(v_opts_2312_, v___x_2315_);
lean_dec_ref(v_opts_2312_);
v___y_2246_ = v___x_2316_;
goto v___jp_2245_;
}
else
{
lean_object* v_val_2317_; uint8_t v___x_2318_; 
lean_dec_ref(v_opts_2312_);
v_val_2317_ = lean_ctor_get(v___x_2314_, 0);
lean_inc(v_val_2317_);
lean_dec_ref_known(v___x_2314_, 1);
v___x_2318_ = lean_unbox(v_val_2317_);
lean_dec(v_val_2317_);
v___y_2246_ = v___x_2318_;
goto v___jp_2245_;
}
v___jp_2245_:
{
if (v___y_2246_ == 0)
{
lean_object* v___x_2247_; lean_object* v_messages_2248_; uint8_t v___x_2249_; 
v___x_2247_ = lean_st_ref_get(v___y_2222_);
v_messages_2248_ = lean_ctor_get(v___x_2247_, 1);
lean_inc_ref(v_messages_2248_);
lean_dec(v___x_2247_);
v___x_2249_ = l_Lean_MessageLog_hasErrors(v_messages_2248_);
lean_dec_ref(v_messages_2248_);
if (v___x_2249_ == 0)
{
lean_del_object(v___x_2236_);
if (lean_obj_tag(v_stx_2220_) == 1)
{
lean_object* v_kind_2250_; 
v_kind_2250_ = lean_ctor_get(v_stx_2220_, 1);
lean_inc(v_kind_2250_);
if (lean_obj_tag(v_kind_2250_) == 1)
{
lean_object* v_pre_2251_; 
v_pre_2251_ = lean_ctor_get(v_kind_2250_, 0);
lean_inc(v_pre_2251_);
if (lean_obj_tag(v_pre_2251_) == 1)
{
lean_object* v_pre_2252_; 
v_pre_2252_ = lean_ctor_get(v_pre_2251_, 0);
lean_inc(v_pre_2252_);
if (lean_obj_tag(v_pre_2252_) == 1)
{
lean_object* v_pre_2253_; 
v_pre_2253_ = lean_ctor_get(v_pre_2252_, 0);
lean_inc(v_pre_2253_);
if (lean_obj_tag(v_pre_2253_) == 1)
{
lean_object* v_pre_2254_; 
v_pre_2254_ = lean_ctor_get(v_pre_2253_, 0);
lean_inc(v_pre_2254_);
if (lean_obj_tag(v_pre_2254_) == 0)
{
lean_object* v_info_2255_; lean_object* v_args_2256_; lean_object* v___x_2258_; uint8_t v_isShared_2259_; uint8_t v_isSharedCheck_2299_; 
v_info_2255_ = lean_ctor_get(v_stx_2220_, 0);
v_args_2256_ = lean_ctor_get(v_stx_2220_, 2);
v_isSharedCheck_2299_ = !lean_is_exclusive(v_stx_2220_);
if (v_isSharedCheck_2299_ == 0)
{
lean_object* v_unused_2300_; 
v_unused_2300_ = lean_ctor_get(v_stx_2220_, 1);
lean_dec(v_unused_2300_);
v___x_2258_ = v_stx_2220_;
v_isShared_2259_ = v_isSharedCheck_2299_;
goto v_resetjp_2257_;
}
else
{
lean_inc(v_args_2256_);
lean_inc(v_info_2255_);
lean_dec(v_stx_2220_);
v___x_2258_ = lean_box(0);
v_isShared_2259_ = v_isSharedCheck_2299_;
goto v_resetjp_2257_;
}
v_resetjp_2257_:
{
lean_object* v_str_2260_; lean_object* v_str_2261_; lean_object* v_str_2262_; lean_object* v_str_2263_; lean_object* v___x_2264_; uint8_t v___x_2265_; 
v_str_2260_ = lean_ctor_get(v_kind_2250_, 1);
lean_inc_ref(v_str_2260_);
lean_dec_ref_known(v_kind_2250_, 2);
v_str_2261_ = lean_ctor_get(v_pre_2251_, 1);
lean_inc_ref(v_str_2261_);
lean_dec_ref_known(v_pre_2251_, 2);
v_str_2262_ = lean_ctor_get(v_pre_2252_, 1);
lean_inc_ref(v_str_2262_);
lean_dec_ref_known(v_pre_2252_, 2);
v_str_2263_ = lean_ctor_get(v_pre_2253_, 1);
lean_inc_ref(v_str_2263_);
lean_dec_ref_known(v_pre_2253_, 2);
v___x_2264_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__1));
v___x_2265_ = lean_string_dec_eq(v_str_2263_, v___x_2264_);
lean_dec_ref(v_str_2263_);
if (v___x_2265_ == 0)
{
lean_dec_ref(v_str_2262_);
lean_dec_ref(v_str_2261_);
lean_dec_ref(v_str_2260_);
lean_del_object(v___x_2258_);
lean_dec_ref(v_args_2256_);
lean_dec(v_info_2255_);
goto v___jp_2230_;
}
else
{
lean_object* v___x_2266_; uint8_t v___x_2267_; 
v___x_2266_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__2));
v___x_2267_ = lean_string_dec_eq(v_str_2262_, v___x_2266_);
lean_dec_ref(v_str_2262_);
if (v___x_2267_ == 0)
{
lean_dec_ref(v_str_2261_);
lean_dec_ref(v_str_2260_);
lean_del_object(v___x_2258_);
lean_dec_ref(v_args_2256_);
lean_dec(v_info_2255_);
goto v___jp_2230_;
}
else
{
lean_object* v___x_2268_; uint8_t v___x_2269_; 
v___x_2268_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getDeclModifiers___closed__3));
v___x_2269_ = lean_string_dec_eq(v_str_2261_, v___x_2268_);
lean_dec_ref(v_str_2261_);
if (v___x_2269_ == 0)
{
lean_dec_ref(v_str_2260_);
lean_del_object(v___x_2258_);
lean_dec_ref(v_args_2256_);
lean_dec(v_info_2255_);
goto v___jp_2230_;
}
else
{
lean_object* v___x_2270_; uint8_t v___x_2271_; 
v___x_2270_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___closed__0));
v___x_2271_ = lean_string_dec_eq(v_str_2260_, v___x_2270_);
lean_dec_ref(v_str_2260_);
if (v___x_2271_ == 0)
{
lean_del_object(v___x_2258_);
lean_dec_ref(v_args_2256_);
lean_dec(v_info_2255_);
goto v___jp_2230_;
}
else
{
lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2277_; 
v___x_2272_ = l_Lean_Name_str___override(v_pre_2254_, v___x_2264_);
v___x_2273_ = l_Lean_Name_str___override(v___x_2272_, v___x_2266_);
v___x_2274_ = l_Lean_Name_str___override(v___x_2273_, v___x_2268_);
v___x_2275_ = l_Lean_Name_str___override(v___x_2274_, v___x_2270_);
if (v_isShared_2259_ == 0)
{
lean_ctor_set(v___x_2258_, 1, v___x_2275_);
v___x_2277_ = v___x_2258_;
goto v_reusejp_2276_;
}
else
{
lean_object* v_reuseFailAlloc_2298_; 
v_reuseFailAlloc_2298_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2298_, 0, v_info_2255_);
lean_ctor_set(v_reuseFailAlloc_2298_, 1, v___x_2275_);
lean_ctor_set(v_reuseFailAlloc_2298_, 2, v_args_2256_);
v___x_2277_ = v_reuseFailAlloc_2298_;
goto v_reusejp_2276_;
}
v_reusejp_2276_:
{
lean_object* v___x_2278_; 
v___x_2278_ = lp_mathlib_Lean_getDocStringText___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_docStringLinter_spec__2(v___x_2277_, v___y_2221_, v___y_2222_);
if (lean_obj_tag(v___x_2278_) == 0)
{
lean_object* v_a_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; 
v_a_2279_ = lean_ctor_get(v___x_2278_, 0);
lean_inc(v_a_2279_);
lean_dec_ref_known(v___x_2278_, 1);
v___x_2280_ = lean_box(0);
v___x_2281_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_lintVersoSyntax(v_a_2279_, v___x_2280_, v___y_2221_, v___y_2222_);
if (lean_obj_tag(v___x_2281_) == 0)
{
lean_object* v_a_2282_; lean_object* v___x_2283_; size_t v_sz_2284_; size_t v___x_2285_; lean_object* v___x_2286_; 
v_a_2282_ = lean_ctor_get(v___x_2281_, 0);
lean_inc(v_a_2282_);
lean_dec_ref_known(v___x_2281_, 1);
v___x_2283_ = lean_box(0);
v_sz_2284_ = lean_array_size(v_a_2282_);
v___x_2285_ = ((size_t)0ULL);
v___x_2286_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter_spec__0(v_a_2282_, v_sz_2284_, v___x_2285_, v___x_2283_, v___y_2221_, v___y_2222_);
lean_dec(v_a_2282_);
if (lean_obj_tag(v___x_2286_) == 0)
{
lean_object* v___x_2288_; uint8_t v_isShared_2289_; uint8_t v_isSharedCheck_2293_; 
v_isSharedCheck_2293_ = !lean_is_exclusive(v___x_2286_);
if (v_isSharedCheck_2293_ == 0)
{
lean_object* v_unused_2294_; 
v_unused_2294_ = lean_ctor_get(v___x_2286_, 0);
lean_dec(v_unused_2294_);
v___x_2288_ = v___x_2286_;
v_isShared_2289_ = v_isSharedCheck_2293_;
goto v_resetjp_2287_;
}
else
{
lean_dec(v___x_2286_);
v___x_2288_ = lean_box(0);
v_isShared_2289_ = v_isSharedCheck_2293_;
goto v_resetjp_2287_;
}
v_resetjp_2287_:
{
lean_object* v___x_2291_; 
if (v_isShared_2289_ == 0)
{
lean_ctor_set(v___x_2288_, 0, v___x_2283_);
v___x_2291_ = v___x_2288_;
goto v_reusejp_2290_;
}
else
{
lean_object* v_reuseFailAlloc_2292_; 
v_reuseFailAlloc_2292_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2292_, 0, v___x_2283_);
v___x_2291_ = v_reuseFailAlloc_2292_;
goto v_reusejp_2290_;
}
v_reusejp_2290_:
{
return v___x_2291_;
}
}
}
else
{
lean_object* v_a_2295_; 
v_a_2295_ = lean_ctor_get(v___x_2286_, 0);
lean_inc(v_a_2295_);
lean_dec_ref_known(v___x_2286_, 1);
v_a_2225_ = v_a_2295_;
goto v___jp_2224_;
}
}
else
{
lean_object* v_a_2296_; 
v_a_2296_ = lean_ctor_get(v___x_2281_, 0);
lean_inc(v_a_2296_);
lean_dec_ref_known(v___x_2281_, 1);
v_a_2225_ = v_a_2296_;
goto v___jp_2224_;
}
}
else
{
lean_object* v_a_2297_; 
v_a_2297_ = lean_ctor_get(v___x_2278_, 0);
lean_inc(v_a_2297_);
lean_dec_ref_known(v___x_2278_, 1);
v_a_2225_ = v_a_2297_;
goto v___jp_2224_;
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
lean_dec_ref_known(v_pre_2253_, 2);
lean_dec(v_pre_2254_);
lean_dec_ref_known(v_pre_2252_, 2);
lean_dec_ref_known(v_pre_2251_, 2);
lean_dec_ref_known(v_kind_2250_, 2);
lean_dec_ref_known(v_stx_2220_, 3);
goto v___jp_2230_;
}
}
else
{
lean_dec(v_pre_2253_);
lean_dec_ref_known(v_pre_2252_, 2);
lean_dec_ref_known(v_pre_2251_, 2);
lean_dec_ref_known(v_kind_2250_, 2);
lean_dec_ref_known(v_stx_2220_, 3);
goto v___jp_2230_;
}
}
else
{
lean_dec_ref_known(v_pre_2251_, 2);
lean_dec(v_pre_2252_);
lean_dec_ref_known(v_kind_2250_, 2);
lean_dec_ref_known(v_stx_2220_, 3);
goto v___jp_2230_;
}
}
else
{
lean_dec(v_pre_2251_);
lean_dec_ref_known(v_kind_2250_, 2);
lean_dec_ref_known(v_stx_2220_, 3);
goto v___jp_2230_;
}
}
else
{
lean_dec(v_kind_2250_);
lean_dec_ref_known(v_stx_2220_, 3);
goto v___jp_2230_;
}
}
else
{
lean_dec(v_stx_2220_);
goto v___jp_2230_;
}
}
else
{
lean_object* v___x_2301_; lean_object* v___x_2303_; 
lean_dec(v_stx_2220_);
v___x_2301_ = lean_box(0);
if (v_isShared_2237_ == 0)
{
lean_ctor_set(v___x_2236_, 0, v___x_2301_);
v___x_2303_ = v___x_2236_;
goto v_reusejp_2302_;
}
else
{
lean_object* v_reuseFailAlloc_2304_; 
v_reuseFailAlloc_2304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2304_, 0, v___x_2301_);
v___x_2303_ = v_reuseFailAlloc_2304_;
goto v_reusejp_2302_;
}
v_reusejp_2302_:
{
return v___x_2303_;
}
}
}
else
{
lean_object* v___x_2305_; lean_object* v___x_2307_; 
lean_dec(v_stx_2220_);
v___x_2305_ = lean_box(0);
if (v_isShared_2237_ == 0)
{
lean_ctor_set(v___x_2236_, 0, v___x_2305_);
v___x_2307_ = v___x_2236_;
goto v_reusejp_2306_;
}
else
{
lean_object* v_reuseFailAlloc_2308_; 
v_reuseFailAlloc_2308_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2308_, 0, v___x_2305_);
v___x_2307_ = v_reuseFailAlloc_2308_;
goto v_reusejp_2306_;
}
v_reusejp_2306_:
{
return v___x_2307_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0___boxed(lean_object* v_stx_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_){
_start:
{
lean_object* v_res_2324_; 
v_res_2324_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter___lam__0(v_stx_2320_, v___y_2321_, v___y_2322_);
lean_dec(v___y_2322_);
lean_dec_ref(v___y_2321_);
return v_res_2324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3183647073____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2337_; lean_object* v___x_2338_; 
v___x_2337_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_moduleDocVersoLinter));
v___x_2338_ = l_Lean_Elab_Command_addLinter(v___x_2337_);
return v___x_2338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3183647073____hygCtx___hyg_2____boxed(lean_object* v_a_2339_){
_start:
{
lean_object* v_res_2340_; 
v_res_2340_ = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3183647073____hygCtx___hyg_2_();
return v_res_2340_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Command(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocString(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__1 = _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__1();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__1);
lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__2 = _init_lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__2();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_deindentString___boxed__const__2);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_Iterators_Combinators_Zip(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_Iterators_Producers_Range(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_DocString(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_Iterators_Combinators_Zip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_Iterators_Producers_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_48192442____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_docString = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_docString);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_4112775180____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_docString_empty = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_docString_empty);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3513071771____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_docStringVerso = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_docStringVerso);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3364454516____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_DocString_0__Mathlib_Linter_Style_initFn_00___x40_Mathlib_Tactic_Linter_DocString_3183647073____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Std_Data_Iterators_Combinators_Zip(uint8_t builtin);
lean_object* initialize_Lean_Parser_Command(uint8_t builtin);
lean_object* initialize_Std_Data_Iterators_Producers_Range(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DocString(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_Iterators_Combinators_Zip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_Iterators_Producers_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_DocString(builtin);
}
#ifdef __cplusplus
}
#endif
