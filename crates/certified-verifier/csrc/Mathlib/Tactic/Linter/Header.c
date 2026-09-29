// Lean compiler output
// Module: Mathlib.Tactic.Linter.Header
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Std.Sync.Mutex public import Lean.Parser.Module public import Mathlib.Tactic.Linter.DirectoryDependency public meta import Lean.Linter.Basic public import Std.Sync.Mutex
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
uint8_t l_Lean_instBEqImport_beq(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lean_io_basemutex_unlock(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_instHashableImport_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_getRoot(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_System_FilePath_addExtension(lean_object*, lean_object*);
uint8_t l_System_FilePath_pathExists(lean_object*);
lean_object* l_IO_FS_readFile(lean_object*);
lean_object* l_Lean_parseImports_x27(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Syntax_getAtomVal(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
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
lean_object* l_Lean_Environment_header(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_MessageData_hint_x27(lean_object*);
lean_object* l_Lean_NameSet_ofList(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_getD___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* l_Lean_mkAtomFrom(lean_object*, lean_object*, uint8_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_nextn(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* l_String_Slice_Pos_prev_x3f(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_prevn(lean_object*, lean_object*, lean_object*);
uint8_t l_String_Slice_beq(lean_object*, lean_object*);
lean_object* l_List_getLastD___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getHeadInfo(lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_ParseImports_whitespace(lean_object*, lean_object*);
lean_object* l_Lean_ParseImports_main(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkInputContext___redArg(lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Parser_parseHeader(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Linter_directoryDependencyCheck(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Linter_linter_directoryDependency;
lean_object* l_Std_Mutex_new___redArg(lean_object*);
lean_object* lean_io_basemutex_lock(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
uint8_t l_Lean_Parser_isTerminalCommand(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Authors:"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Please, add at least one author!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Please, do not end the authors' line with a period."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " and "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Please, do not use 'and'; use ',' instead."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "  "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "Double spaces are not allowed."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "The authors line should begin with 'Authors: '"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Authors: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__11;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Malformed or missing copyright header: `"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "` should be alone on its own line."};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4_spec__5(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ",\n"};
static const lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__5;
static const lean_ctor_object lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__6 = (const lean_object*)&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = ",\n  "};
static const lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0 = (const lean_object*)&lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__1;
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = ".\nAll rights reserved."};
static const lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__2;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__4;
static lean_once_cell_t lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = ". All rights reserved."};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\n-/"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "-/"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Copyright too short!"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Second copyright line should be \""};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\""};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 95, .m_capacity = 95, .m_length = 94, .m_data = "If an authors line spans multiple lines, each line but the last must end with a trailing comma"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Copyright line should end with '. All rights reserved.'"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 93, .m_capacity = 93, .m_length = 92, .m_data = "There should be at least one copyright author, separated from the year by exactly one space."};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__10_value;
static const lean_array_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 246}, .m_size = 3, .m_capacity = 3, .m_data = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__3_value),((lean_object*)&lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "'Copyright (c) YYYY' should be followed by a space"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Copyright line should start with 'Copyright (c) YYYY'"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Copyright (c) 20"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__18;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "/-"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__25;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__26;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_2273770963____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_2273770963____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_inLibraryRootMutex;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "header"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(38, 218, 47, 33, 118, 85, 90, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "enable the header style linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(233, 226, 206, 163, 69, 79, 102, 93)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_header;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "license"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(105, 62, 218, 153, 100, 142, 29, 251)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(38, 218, 47, 33, 118, 85, 90, 105)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(204, 231, 116, 84, 245, 170, 131, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "Released under Apache 2.0 license as described in the file LICENSE."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "The text required as the second line of the copyright header."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(98, 189, 128, 85, 154, 50, 252, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(233, 226, 206, 163, 69, 79, 102, 93)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(79, 148, 77, 184, 111, 229, 221, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_style_header_license;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "import"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 219, 158, 40, 50, 143, 61, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__5_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "all"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__7_value),LEAN_SCALAR_PTR_LITERAL(107, 73, 92, 3, 207, 252, 164, 131)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__9_value),LEAN_SCALAR_PTR_LITERAL(89, 228, 64, 55, 26, 167, 248, 235)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "public"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__11_value),LEAN_SCALAR_PTR_LITERAL(198, 166, 14, 39, 152, 190, 236, 172)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(40, 173, 92, 3, 94, 219, 131, 202)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "prelude"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__1_value),LEAN_SCALAR_PTR_LITERAL(182, 6, 18, 235, 50, 88, 101, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "moduleTk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__2_value),LEAN_SCALAR_PTR_LITERAL(239, 68, 245, 129, 233, 83, 45, 77)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__3_value),LEAN_SCALAR_PTR_LITERAL(198, 239, 28, 252, 21, 233, 71, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToModuleTk_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToPreludeTk_x3f(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "MathlibTest"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Header"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 64, 137, 69, 117, 8, 91, 199)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(81, 198, 59, 166, 32, 76, 99, 16)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value),LEAN_SCALAR_PTR_LITERAL(192, 194, 95, 189, 251, 160, 187, 110)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__2_value),LEAN_SCALAR_PTR_LITERAL(123, 106, 32, 169, 237, 145, 211, 96)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Fail"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 64, 137, 69, 117, 8, 91, 199)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(81, 198, 59, 166, 32, 76, 99, 16)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value),LEAN_SCALAR_PTR_LITERAL(192, 194, 95, 189, 251, 160, 187, 110)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__4_value),LEAN_SCALAR_PTR_LITERAL(107, 154, 0, 109, 232, 103, 97, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Verso"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 64, 137, 69, 117, 8, 91, 199)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(81, 198, 59, 166, 32, 76, 99, 16)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value),LEAN_SCALAR_PTR_LITERAL(192, 194, 95, 189, 251, 160, 187, 110)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__6_value),LEAN_SCALAR_PTR_LITERAL(252, 121, 167, 231, 18, 90, 166, 75)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "DirectoryDependencyLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Test"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 64, 137, 69, 117, 8, 91, 199)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__8_value),LEAN_SCALAR_PTR_LITERAL(165, 61, 69, 173, 118, 209, 96, 41)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__9_value),LEAN_SCALAR_PTR_LITERAL(209, 169, 159, 116, 168, 28, 68, 137)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__11_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__12_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__15;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Files in mathlib cannot import the whole `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "` folder. Doing so would cause imports to be unnecessarily slow."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lake"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(111, 69, 182, 10, 108, 181, 149, 180)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 291, .m_capacity = 291, .m_length = 290, .m_data = "In the past, importing `Lake` in mathlib has led to dramatic slow-downs of the linter (see e.g. https://github.com/leanprover-community/mathlib4/pull/13779). Please consider carefully if this import is useful and make sure to benchmark it. If this is fine, feel free to silence this linter."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__4_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__6;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__8_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Replace"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__10_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Have"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 40, 204, 200, 191, 22, 81, 236)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__14 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__14_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__15_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 103, .m_capacity = 103, .m_length = 102, .m_data = "`Mathlib.Tactic.Have` defines a deprecated form of the `have` tactic; please do not use it in mathlib."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__16 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__16_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__17 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__18;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 109, .m_capacity = 109, .m_length = 108, .m_data = "`Mathlib.Tactic.Replace` defines a deprecated form of the `replace` tactic; please do not use it in mathlib."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__19 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__19_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__20 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__21;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Std"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__22 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__22_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__23 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Duplicate imports: `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "` already imported"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "* `"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "`:\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__3;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Init"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(153, 180, 146, 189, 15, 221, 10, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "moduleDoc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(249, 71, 187, 113, 90, 175, 60, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 136, .m_capacity = 136, .m_length = 135, .m_data = "The module doc-string for a file should be the first command after the imports.\nPlease, add a module doc-string (`/-! ... -/`) before `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "Type `m(odule docstring) + [tab]` to insert a template via snippet."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__13;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "deprecated_module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(239, 157, 18, 102, 89, 88, 107, 255)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__4_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value),LEAN_SCALAR_PTR_LITERAL(196, 77, 77, 155, 236, 239, 153, 45)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(37, 92, 218, 157, 119, 152, 114, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(128, 94, 11, 194, 237, 90, 20, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__7_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(6, 154, 50, 157, 87, 187, 225, 29)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Style"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__11_value),LEAN_SCALAR_PTR_LITERAL(6, 203, 122, 143, 225, 205, 116, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__1_value),LEAN_SCALAR_PTR_LITERAL(227, 24, 169, 226, 105, 42, 150, 25)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "headerLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__14_value),LEAN_SCALAR_PTR_LITERAL(165, 190, 198, 241, 127, 9, 0, 231)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_initFn_00___x40_Mathlib_Tactic_Linter_Header_1276962649____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_initFn_00___x40_Mathlib_Tactic_Linter_Header_1276962649____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(lean_object* v_s_2_, lean_object* v_pattern_3_, lean_object* v_offset_4_){
_start:
{
lean_object* v___y_6_; lean_object* v___x_20_; uint8_t v___x_21_; 
v___x_20_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_21_ = lean_string_dec_eq(v_pattern_3_, v___x_20_);
if (v___x_21_ == 0)
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_22_ = lean_unsigned_to_nat(0u);
v___x_23_ = lean_box(0);
v___x_24_ = l_String_splitOnAux(v_s_2_, v_pattern_3_, v___x_22_, v___x_22_, v___x_22_, v___x_23_);
lean_dec_ref(v_s_2_);
v___y_6_ = v___x_24_;
goto v___jp_5_;
}
else
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = lean_box(0);
v___x_26_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_26_, 0, v_s_2_);
lean_ctor_set(v___x_26_, 1, v___x_25_);
v___y_6_ = v___x_26_;
goto v___jp_5_;
}
v___jp_5_:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v_beg_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v_fin_14_; lean_object* v___x_15_; uint8_t v___x_16_; lean_object* v___x_17_; uint8_t v___x_18_; lean_object* v___x_19_; 
v___x_7_ = lean_unsigned_to_nat(0u);
v___x_8_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_9_ = l_List_getD___redArg(v___y_6_, v___x_7_, v___x_8_);
lean_dec(v___y_6_);
v___x_10_ = lean_string_utf8_byte_size(v___x_9_);
v_beg_11_ = lean_nat_add(v_offset_4_, v___x_10_);
v___x_12_ = lean_string_append(v___x_9_, v_pattern_3_);
v___x_13_ = lean_string_utf8_byte_size(v___x_12_);
lean_dec_ref(v___x_12_);
v_fin_14_ = lean_nat_add(v_offset_4_, v___x_13_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_beg_11_);
lean_ctor_set(v___x_15_, 1, v_fin_14_);
v___x_16_ = 1;
v___x_17_ = l_Lean_Syntax_ofRange(v___x_15_, v___x_16_);
v___x_18_ = 0;
v___x_19_ = l_Lean_mkAtomFrom(v___x_17_, v_pattern_3_, v___x_18_);
lean_dec(v___x_17_);
return v___x_19_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___boxed(lean_object* v_s_27_, lean_object* v_pattern_28_, lean_object* v_offset_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_s_27_, v_pattern_28_, v_offset_29_);
lean_dec(v_offset_29_);
return v_res_30_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__11(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__10));
v___x_44_ = lean_string_utf8_byte_size(v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks(lean_object* v_line_45_, lean_object* v_offset_46_){
_start:
{
lean_object* v___x_47_; lean_object* v_stxs_48_; lean_object* v_stxs_50_; lean_object* v___y_71_; uint32_t v___y_72_; lean_object* v_stxs_81_; lean_object* v_stxs_92_; lean_object* v_stxs_104_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; uint8_t v___x_128_; 
v___x_47_ = lean_unsigned_to_nat(0u);
v_stxs_48_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__0));
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__10));
v___x_126_ = lean_string_utf8_byte_size(v_line_45_);
v___x_127_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__11, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__11);
v___x_128_ = lean_nat_dec_le(v___x_127_, v___x_126_);
if (v___x_128_ == 0)
{
goto v___jp_115_;
}
else
{
uint8_t v___x_129_; 
v___x_129_ = lean_string_memcmp(v_line_45_, v___x_125_, v___x_47_, v___x_47_, v___x_127_);
if (v___x_129_ == 0)
{
goto v___jp_115_;
}
else
{
v_stxs_104_ = v_stxs_48_;
goto v___jp_103_;
}
}
v___jp_49_:
{
lean_object* v___x_51_; uint8_t v___x_52_; 
v___x_51_ = lean_array_get_size(v_stxs_50_);
v___x_52_ = lean_nat_dec_eq(v___x_51_, v___x_47_);
if (v___x_52_ == 0)
{
lean_dec_ref(v_line_45_);
return v_stxs_50_;
}
else
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v_startInclusive_59_; lean_object* v_endExclusive_60_; lean_object* v___x_61_; uint8_t v___x_62_; 
lean_dec_ref(v_stxs_50_);
v___x_53_ = lean_unsigned_to_nat(8u);
v___x_54_ = lean_string_utf8_byte_size(v_line_45_);
lean_inc_ref_n(v_line_45_, 2);
v___x_55_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_55_, 0, v_line_45_);
lean_ctor_set(v___x_55_, 1, v___x_47_);
lean_ctor_set(v___x_55_, 2, v___x_54_);
v___x_56_ = l_String_Slice_Pos_nextn(v___x_55_, v___x_47_, v___x_53_);
lean_dec_ref_known(v___x_55_, 3);
v___x_57_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_57_, 0, v_line_45_);
lean_ctor_set(v___x_57_, 1, v___x_56_);
lean_ctor_set(v___x_57_, 2, v___x_54_);
v___x_58_ = l_String_Slice_trimAscii(v___x_57_);
v_startInclusive_59_ = lean_ctor_get(v___x_58_, 1);
lean_inc(v_startInclusive_59_);
v_endExclusive_60_ = lean_ctor_get(v___x_58_, 2);
lean_inc(v_endExclusive_60_);
lean_dec_ref(v___x_58_);
v___x_61_ = lean_nat_sub(v_endExclusive_60_, v_startInclusive_59_);
lean_dec(v_startInclusive_59_);
lean_dec(v_endExclusive_60_);
v___x_62_ = lean_nat_dec_eq(v___x_61_, v___x_47_);
lean_dec(v___x_61_);
if (v___x_62_ == 0)
{
lean_dec_ref(v_line_45_);
return v_stxs_48_;
}
else
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_63_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__1));
v___x_64_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_line_45_, v___x_63_, v_offset_46_);
v___x_65_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__2));
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_64_);
lean_ctor_set(v___x_66_, 1, v___x_65_);
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = lean_mk_empty_array_with_capacity(v___x_67_);
v___x_69_ = lean_array_push(v___x_68_, v___x_66_);
return v___x_69_;
}
}
}
v___jp_70_:
{
uint32_t v___x_73_; uint8_t v___x_74_; 
v___x_73_ = 46;
v___x_74_ = lean_uint32_dec_eq(v___y_72_, v___x_73_);
if (v___x_74_ == 0)
{
v_stxs_50_ = v___y_71_;
goto v___jp_49_;
}
else
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v_stxs_79_; 
v___x_75_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__3));
lean_inc_ref(v_line_45_);
v___x_76_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_line_45_, v___x_75_, v_offset_46_);
v___x_77_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__4));
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_76_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
v_stxs_79_ = lean_array_push(v___y_71_, v___x_78_);
v_stxs_50_ = v_stxs_79_;
goto v___jp_49_;
}
}
v___jp_80_:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = lean_string_utf8_byte_size(v_line_45_);
lean_inc_ref(v_line_45_);
v___x_83_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_83_, 0, v_line_45_);
lean_ctor_set(v___x_83_, 1, v___x_47_);
lean_ctor_set(v___x_83_, 2, v___x_82_);
v___x_84_ = l_String_Slice_Pos_prev_x3f(v___x_83_, v___x_82_);
if (lean_obj_tag(v___x_84_) == 0)
{
uint32_t v___x_85_; 
lean_dec_ref_known(v___x_83_, 3);
v___x_85_ = 65;
v___y_71_ = v_stxs_81_;
v___y_72_ = v___x_85_;
goto v___jp_70_;
}
else
{
lean_object* v_val_86_; lean_object* v___x_87_; 
v_val_86_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_val_86_);
lean_dec_ref_known(v___x_84_, 1);
v___x_87_ = l_String_Slice_Pos_get_x3f(v___x_83_, v_val_86_);
lean_dec(v_val_86_);
lean_dec_ref_known(v___x_83_, 3);
if (lean_obj_tag(v___x_87_) == 0)
{
uint32_t v___x_88_; 
v___x_88_ = 65;
v___y_71_ = v_stxs_81_;
v___y_72_ = v___x_88_;
goto v___jp_70_;
}
else
{
lean_object* v_val_89_; uint32_t v___x_90_; 
v_val_89_ = lean_ctor_get(v___x_87_, 0);
lean_inc(v_val_89_);
lean_dec_ref_known(v___x_87_, 1);
v___x_90_ = lean_unbox_uint32(v_val_89_);
lean_dec(v_val_89_);
v___y_71_ = v_stxs_81_;
v___y_72_ = v___x_90_;
goto v___jp_70_;
}
}
}
v___jp_91_:
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; uint8_t v___x_98_; 
v___x_93_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__5));
v___x_94_ = lean_box(0);
v___x_95_ = l_String_splitOnAux(v_line_45_, v___x_93_, v___x_47_, v___x_47_, v___x_47_, v___x_94_);
v___x_96_ = l_List_lengthTR___redArg(v___x_95_);
lean_dec(v___x_95_);
v___x_97_ = lean_unsigned_to_nat(1u);
v___x_98_ = lean_nat_dec_eq(v___x_96_, v___x_97_);
lean_dec(v___x_96_);
if (v___x_98_ == 0)
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v_stxs_102_; 
lean_inc_ref(v_line_45_);
v___x_99_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_line_45_, v___x_93_, v_offset_46_);
v___x_100_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__6));
v___x_101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_99_);
lean_ctor_set(v___x_101_, 1, v___x_100_);
v_stxs_102_ = lean_array_push(v_stxs_92_, v___x_101_);
v_stxs_81_ = v_stxs_102_;
goto v___jp_80_;
}
else
{
v_stxs_81_ = v_stxs_92_;
goto v___jp_80_;
}
}
v___jp_103_:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; uint8_t v___x_110_; 
v___x_105_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__7));
v___x_106_ = lean_box(0);
v___x_107_ = l_String_splitOnAux(v_line_45_, v___x_105_, v___x_47_, v___x_47_, v___x_47_, v___x_106_);
v___x_108_ = l_List_lengthTR___redArg(v___x_107_);
lean_dec(v___x_107_);
v___x_109_ = lean_unsigned_to_nat(1u);
v___x_110_ = lean_nat_dec_eq(v___x_108_, v___x_109_);
lean_dec(v___x_108_);
if (v___x_110_ == 0)
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v_stxs_114_; 
lean_inc_ref(v_line_45_);
v___x_111_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_line_45_, v___x_105_, v_offset_46_);
v___x_112_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__8));
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_111_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v_stxs_114_ = lean_array_push(v_stxs_104_, v___x_113_);
v_stxs_92_ = v_stxs_114_;
goto v___jp_91_;
}
else
{
v_stxs_92_ = v_stxs_104_;
goto v___jp_91_;
}
}
v___jp_115_:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v_stxs_124_; 
v___x_116_ = lean_unsigned_to_nat(9u);
v___x_117_ = lean_string_utf8_byte_size(v_line_45_);
lean_inc_ref_n(v_line_45_, 2);
v___x_118_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_118_, 0, v_line_45_);
lean_ctor_set(v___x_118_, 1, v___x_47_);
lean_ctor_set(v___x_118_, 2, v___x_117_);
v___x_119_ = l_String_Slice_Pos_nextn(v___x_118_, v___x_47_, v___x_116_);
lean_dec_ref_known(v___x_118_, 3);
v___x_120_ = lean_string_utf8_extract_fast(v_line_45_, v___x_47_, v___x_119_);
lean_dec(v___x_119_);
v___x_121_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_line_45_, v___x_120_, v_offset_46_);
v___x_122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__9));
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_121_);
lean_ctor_set(v___x_123_, 1, v___x_122_);
v_stxs_124_ = lean_array_push(v_stxs_48_, v___x_123_);
v_stxs_104_ = v_stxs_124_;
goto v___jp_103_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___boxed(lean_object* v_line_130_, lean_object* v_offset_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks(v_line_130_, v_offset_131_);
lean_dec(v_offset_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0(lean_object* v_s_135_){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_136_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__0));
v___x_137_ = lean_string_append(v___x_136_, v_s_135_);
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___closed__1));
v___x_139_ = lean_string_append(v___x_137_, v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0___boxed(lean_object* v_s_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0(v_s_140_);
lean_dec_ref(v_s_140_);
return v_res_141_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4_spec__5(lean_object* v_a_142_, lean_object* v_as_143_, size_t v_i_144_, size_t v_stop_145_){
_start:
{
uint8_t v___x_146_; 
v___x_146_ = lean_usize_dec_eq(v_i_144_, v_stop_145_);
if (v___x_146_ == 0)
{
lean_object* v___x_147_; uint8_t v___x_148_; 
v___x_147_ = lean_array_uget_borrowed(v_as_143_, v_i_144_);
v___x_148_ = lean_string_dec_eq(v_a_142_, v___x_147_);
if (v___x_148_ == 0)
{
size_t v___x_149_; size_t v___x_150_; 
v___x_149_ = ((size_t)1ULL);
v___x_150_ = lean_usize_add(v_i_144_, v___x_149_);
v_i_144_ = v___x_150_;
goto _start;
}
else
{
return v___x_148_;
}
}
else
{
uint8_t v___x_152_; 
v___x_152_ = 0;
return v___x_152_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4_spec__5___boxed(lean_object* v_a_153_, lean_object* v_as_154_, lean_object* v_i_155_, lean_object* v_stop_156_){
_start:
{
size_t v_i_boxed_157_; size_t v_stop_boxed_158_; uint8_t v_res_159_; lean_object* v_r_160_; 
v_i_boxed_157_ = lean_unbox_usize(v_i_155_);
lean_dec(v_i_155_);
v_stop_boxed_158_ = lean_unbox_usize(v_stop_156_);
lean_dec(v_stop_156_);
v_res_159_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4_spec__5(v_a_153_, v_as_154_, v_i_boxed_157_, v_stop_boxed_158_);
lean_dec_ref(v_as_154_);
lean_dec_ref(v_a_153_);
v_r_160_ = lean_box(v_res_159_);
return v_r_160_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4(lean_object* v_as_161_, lean_object* v_a_162_){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; uint8_t v___x_165_; 
v___x_163_ = lean_unsigned_to_nat(0u);
v___x_164_ = lean_array_get_size(v_as_161_);
v___x_165_ = lean_nat_dec_lt(v___x_163_, v___x_164_);
if (v___x_165_ == 0)
{
return v___x_165_;
}
else
{
if (v___x_165_ == 0)
{
return v___x_165_;
}
else
{
size_t v___x_166_; size_t v___x_167_; uint8_t v___x_168_; 
v___x_166_ = ((size_t)0ULL);
v___x_167_ = lean_usize_of_nat(v___x_164_);
v___x_168_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4_spec__5(v_a_162_, v_as_161_, v___x_166_, v___x_167_);
return v___x_168_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4___boxed(lean_object* v_as_169_, lean_object* v_a_170_){
_start:
{
uint8_t v_res_171_; lean_object* v_r_172_; 
v_res_171_ = lp_mathlib_Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4(v_as_169_, v_a_170_);
lean_dec_ref(v_a_170_);
lean_dec_ref(v_as_169_);
v_r_172_ = lean_box(v_res_171_);
return v_r_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(lean_object* v_s_173_, lean_object* v_replacement_174_, lean_object* v_a_175_, lean_object* v_b_176_){
_start:
{
lean_object* v_it_178_; lean_object* v_startPos_179_; lean_object* v_endPos_180_; lean_object* v_it_189_; 
switch(lean_obj_tag(v_a_175_))
{
case 0:
{
lean_object* v_pos_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_207_; 
v_pos_195_ = lean_ctor_get(v_a_175_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v_a_175_);
if (v_isSharedCheck_207_ == 0)
{
v___x_197_ = v_a_175_;
v_isShared_198_ = v_isSharedCheck_207_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_pos_195_);
lean_dec(v_a_175_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_207_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v_startInclusive_199_; lean_object* v_endExclusive_200_; lean_object* v___x_201_; uint8_t v___x_202_; 
v_startInclusive_199_ = lean_ctor_get(v_s_173_, 1);
v_endExclusive_200_ = lean_ctor_get(v_s_173_, 2);
v___x_201_ = lean_nat_sub(v_endExclusive_200_, v_startInclusive_199_);
v___x_202_ = lean_nat_dec_eq(v_pos_195_, v___x_201_);
lean_dec(v___x_201_);
if (v___x_202_ == 0)
{
lean_object* v___x_204_; 
if (v_isShared_198_ == 0)
{
lean_ctor_set_tag(v___x_197_, 1);
v___x_204_ = v___x_197_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v_pos_195_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
v_it_189_ = v___x_204_;
goto v___jp_188_;
}
}
else
{
lean_object* v___x_206_; 
lean_del_object(v___x_197_);
lean_dec(v_pos_195_);
v___x_206_ = lean_box(3);
v_it_189_ = v___x_206_;
goto v___jp_188_;
}
}
}
case 1:
{
lean_object* v_pos_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_220_; 
v_pos_208_ = lean_ctor_get(v_a_175_, 0);
v_isSharedCheck_220_ = !lean_is_exclusive(v_a_175_);
if (v_isSharedCheck_220_ == 0)
{
v___x_210_ = v_a_175_;
v_isShared_211_ = v_isSharedCheck_220_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_pos_208_);
lean_dec(v_a_175_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_220_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
lean_object* v_str_212_; lean_object* v_startInclusive_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_218_; 
v_str_212_ = lean_ctor_get(v_s_173_, 0);
v_startInclusive_213_ = lean_ctor_get(v_s_173_, 1);
v___x_214_ = lean_nat_add(v_startInclusive_213_, v_pos_208_);
v___x_215_ = lean_string_utf8_next_fast(v_str_212_, v___x_214_);
lean_dec(v___x_214_);
v___x_216_ = lean_nat_sub(v___x_215_, v_startInclusive_213_);
lean_inc(v___x_216_);
if (v_isShared_211_ == 0)
{
lean_ctor_set_tag(v___x_210_, 0);
lean_ctor_set(v___x_210_, 0, v___x_216_);
v___x_218_ = v___x_210_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v___x_216_);
v___x_218_ = v_reuseFailAlloc_219_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
v_it_178_ = v___x_218_;
v_startPos_179_ = v_pos_208_;
v_endPos_180_ = v___x_216_;
goto v___jp_177_;
}
}
}
case 2:
{
lean_object* v_needle_221_; lean_object* v_table_222_; lean_object* v_stackPos_223_; lean_object* v_needlePos_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_283_; 
v_needle_221_ = lean_ctor_get(v_a_175_, 0);
v_table_222_ = lean_ctor_get(v_a_175_, 1);
v_stackPos_223_ = lean_ctor_get(v_a_175_, 2);
v_needlePos_224_ = lean_ctor_get(v_a_175_, 3);
v_isSharedCheck_283_ = !lean_is_exclusive(v_a_175_);
if (v_isSharedCheck_283_ == 0)
{
v___x_226_ = v_a_175_;
v_isShared_227_ = v_isSharedCheck_283_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_needlePos_224_);
lean_inc(v_stackPos_223_);
lean_inc(v_table_222_);
lean_inc(v_needle_221_);
lean_dec(v_a_175_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_283_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
lean_object* v_str_228_; lean_object* v_startInclusive_229_; lean_object* v_endExclusive_230_; lean_object* v_str_231_; lean_object* v_startInclusive_232_; lean_object* v_endExclusive_233_; lean_object* v_basePos_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; uint8_t v___x_238_; 
v_str_228_ = lean_ctor_get(v_needle_221_, 0);
v_startInclusive_229_ = lean_ctor_get(v_needle_221_, 1);
v_endExclusive_230_ = lean_ctor_get(v_needle_221_, 2);
v_str_231_ = lean_ctor_get(v_s_173_, 0);
v_startInclusive_232_ = lean_ctor_get(v_s_173_, 1);
v_endExclusive_233_ = lean_ctor_get(v_s_173_, 2);
v_basePos_234_ = lean_nat_sub(v_stackPos_223_, v_needlePos_224_);
v___x_235_ = lean_nat_sub(v_endExclusive_230_, v_startInclusive_229_);
v___x_236_ = lean_nat_add(v_basePos_234_, v___x_235_);
v___x_237_ = lean_nat_sub(v_endExclusive_233_, v_startInclusive_232_);
v___x_238_ = lean_nat_dec_le(v___x_236_, v___x_237_);
lean_dec(v___x_236_);
if (v___x_238_ == 0)
{
uint8_t v___x_239_; 
lean_dec(v___x_235_);
lean_del_object(v___x_226_);
lean_dec(v_needlePos_224_);
lean_dec(v_stackPos_223_);
lean_dec_ref(v_table_222_);
lean_dec_ref(v_needle_221_);
v___x_239_ = lean_nat_dec_lt(v_basePos_234_, v___x_237_);
if (v___x_239_ == 0)
{
lean_dec(v___x_237_);
lean_dec(v_basePos_234_);
lean_dec_ref(v_s_173_);
return v_b_176_;
}
else
{
lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_240_ = l_String_Slice_pos_x21(v_s_173_, v_basePos_234_);
lean_dec(v_basePos_234_);
v___x_241_ = lean_box(3);
v_it_178_ = v___x_241_;
v_startPos_179_ = v___x_240_;
v_endPos_180_ = v___x_237_;
goto v___jp_177_;
}
}
else
{
lean_object* v___x_242_; uint8_t v_stackByte_243_; lean_object* v___x_244_; uint8_t v_patByte_245_; uint8_t v___x_246_; 
lean_dec(v___x_237_);
v___x_242_ = lean_nat_add(v_startInclusive_232_, v_stackPos_223_);
v_stackByte_243_ = lean_string_get_byte_fast(v_str_231_, v___x_242_);
v___x_244_ = lean_nat_add(v_startInclusive_229_, v_needlePos_224_);
v_patByte_245_ = lean_string_get_byte_fast(v_str_228_, v___x_244_);
v___x_246_ = lean_uint8_dec_eq(v_stackByte_243_, v_patByte_245_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; uint8_t v___x_248_; 
lean_dec(v___x_235_);
v___x_247_ = lean_unsigned_to_nat(0u);
v___x_248_ = lean_nat_dec_eq(v_needlePos_224_, v___x_247_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v_newNeedlePos_251_; uint8_t v___x_252_; 
v___x_249_ = lean_unsigned_to_nat(1u);
v___x_250_ = lean_nat_sub(v_needlePos_224_, v___x_249_);
lean_dec(v_needlePos_224_);
v_newNeedlePos_251_ = lean_array_fget_borrowed(v_table_222_, v___x_250_);
lean_dec(v___x_250_);
v___x_252_ = lean_nat_dec_eq(v_newNeedlePos_251_, v___x_247_);
if (v___x_252_ == 0)
{
lean_object* v_oldBasePos_253_; lean_object* v___x_254_; lean_object* v_newBasePos_255_; lean_object* v___x_257_; 
lean_inc(v_newNeedlePos_251_);
v_oldBasePos_253_ = l_String_Slice_pos_x21(v_s_173_, v_basePos_234_);
lean_dec(v_basePos_234_);
v___x_254_ = lean_nat_sub(v_stackPos_223_, v_newNeedlePos_251_);
v_newBasePos_255_ = l_String_Slice_pos_x21(v_s_173_, v___x_254_);
lean_dec(v___x_254_);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 3, v_newNeedlePos_251_);
v___x_257_ = v___x_226_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_needle_221_);
lean_ctor_set(v_reuseFailAlloc_258_, 1, v_table_222_);
lean_ctor_set(v_reuseFailAlloc_258_, 2, v_stackPos_223_);
lean_ctor_set(v_reuseFailAlloc_258_, 3, v_newNeedlePos_251_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
v_it_178_ = v___x_257_;
v_startPos_179_ = v_oldBasePos_253_;
v_endPos_180_ = v_newBasePos_255_;
goto v___jp_177_;
}
}
else
{
lean_object* v_basePos_259_; lean_object* v_nextStackPos_260_; lean_object* v___x_262_; 
v_basePos_259_ = l_String_Slice_pos_x21(v_s_173_, v_basePos_234_);
lean_dec(v_basePos_234_);
v_nextStackPos_260_ = l_String_Slice_posGE___redArg(v_s_173_, v_stackPos_223_);
lean_inc(v_nextStackPos_260_);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 3, v___x_247_);
lean_ctor_set(v___x_226_, 2, v_nextStackPos_260_);
v___x_262_ = v___x_226_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_needle_221_);
lean_ctor_set(v_reuseFailAlloc_263_, 1, v_table_222_);
lean_ctor_set(v_reuseFailAlloc_263_, 2, v_nextStackPos_260_);
lean_ctor_set(v_reuseFailAlloc_263_, 3, v___x_247_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
v_it_178_ = v___x_262_;
v_startPos_179_ = v_basePos_259_;
v_endPos_180_ = v_nextStackPos_260_;
goto v___jp_177_;
}
}
}
else
{
lean_object* v_basePos_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v_nextStackPos_267_; lean_object* v___x_269_; 
lean_dec(v_basePos_234_);
lean_dec(v_needlePos_224_);
v_basePos_264_ = l_String_Slice_pos_x21(v_s_173_, v_stackPos_223_);
v___x_265_ = lean_unsigned_to_nat(1u);
v___x_266_ = lean_nat_add(v_stackPos_223_, v___x_265_);
lean_dec(v_stackPos_223_);
v_nextStackPos_267_ = l_String_Slice_posGE___redArg(v_s_173_, v___x_266_);
lean_inc(v_nextStackPos_267_);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 3, v___x_247_);
lean_ctor_set(v___x_226_, 2, v_nextStackPos_267_);
v___x_269_ = v___x_226_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v_needle_221_);
lean_ctor_set(v_reuseFailAlloc_270_, 1, v_table_222_);
lean_ctor_set(v_reuseFailAlloc_270_, 2, v_nextStackPos_267_);
lean_ctor_set(v_reuseFailAlloc_270_, 3, v___x_247_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
v_it_178_ = v___x_269_;
v_startPos_179_ = v_basePos_264_;
v_endPos_180_ = v_nextStackPos_267_;
goto v___jp_177_;
}
}
}
else
{
lean_object* v___x_271_; lean_object* v_nextStackPos_272_; lean_object* v_nextNeedlePos_273_; uint8_t v___x_274_; 
lean_dec(v_basePos_234_);
v___x_271_ = lean_unsigned_to_nat(1u);
v_nextStackPos_272_ = lean_nat_add(v_stackPos_223_, v___x_271_);
lean_dec(v_stackPos_223_);
v_nextNeedlePos_273_ = lean_nat_add(v_needlePos_224_, v___x_271_);
lean_dec(v_needlePos_224_);
v___x_274_ = lean_nat_dec_eq(v_nextNeedlePos_273_, v___x_235_);
lean_dec(v___x_235_);
if (v___x_274_ == 0)
{
lean_object* v___x_276_; 
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 3, v_nextNeedlePos_273_);
lean_ctor_set(v___x_226_, 2, v_nextStackPos_272_);
v___x_276_ = v___x_226_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v_needle_221_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v_table_222_);
lean_ctor_set(v_reuseFailAlloc_278_, 2, v_nextStackPos_272_);
lean_ctor_set(v_reuseFailAlloc_278_, 3, v_nextNeedlePos_273_);
v___x_276_ = v_reuseFailAlloc_278_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
v_a_175_ = v___x_276_;
goto _start;
}
}
else
{
lean_object* v___x_279_; lean_object* v___x_281_; 
lean_dec(v_nextNeedlePos_273_);
v___x_279_ = lean_unsigned_to_nat(0u);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 3, v___x_279_);
lean_ctor_set(v___x_226_, 2, v_nextStackPos_272_);
v___x_281_ = v___x_226_;
goto v_reusejp_280_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v_needle_221_);
lean_ctor_set(v_reuseFailAlloc_282_, 1, v_table_222_);
lean_ctor_set(v_reuseFailAlloc_282_, 2, v_nextStackPos_272_);
lean_ctor_set(v_reuseFailAlloc_282_, 3, v___x_279_);
v___x_281_ = v_reuseFailAlloc_282_;
goto v_reusejp_280_;
}
v_reusejp_280_:
{
v_it_189_ = v___x_281_;
goto v___jp_188_;
}
}
}
}
}
}
default: 
{
lean_dec_ref(v_s_173_);
return v_b_176_;
}
}
v___jp_177_:
{
lean_object* v___x_181_; lean_object* v_str_182_; lean_object* v_startInclusive_183_; lean_object* v_endExclusive_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
lean_inc_ref(v_s_173_);
v___x_181_ = l_String_Slice_slice_x21(v_s_173_, v_startPos_179_, v_endPos_180_);
lean_dec(v_endPos_180_);
lean_dec(v_startPos_179_);
v_str_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc_ref(v_str_182_);
v_startInclusive_183_ = lean_ctor_get(v___x_181_, 1);
lean_inc(v_startInclusive_183_);
v_endExclusive_184_ = lean_ctor_get(v___x_181_, 2);
lean_inc(v_endExclusive_184_);
lean_dec_ref(v___x_181_);
v___x_185_ = lean_string_utf8_extract_fast(v_str_182_, v_startInclusive_183_, v_endExclusive_184_);
lean_dec(v_endExclusive_184_);
lean_dec(v_startInclusive_183_);
lean_dec_ref(v_str_182_);
v___x_186_ = lean_string_append(v_b_176_, v___x_185_);
lean_dec_ref(v___x_185_);
v_a_175_ = v_it_178_;
v_b_176_ = v___x_186_;
goto _start;
}
v___jp_188_:
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_190_ = lean_unsigned_to_nat(0u);
v___x_191_ = lean_string_utf8_byte_size(v_replacement_174_);
v___x_192_ = lean_string_utf8_extract_fast(v_replacement_174_, v___x_190_, v___x_191_);
v___x_193_ = lean_string_append(v_b_176_, v___x_192_);
lean_dec_ref(v___x_192_);
v_a_175_ = v_it_189_;
v_b_176_ = v___x_193_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg___boxed(lean_object* v_s_284_, lean_object* v_replacement_285_, lean_object* v_a_286_, lean_object* v_b_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_284_, v_replacement_285_, v_a_286_, v_b_287_);
lean_dec_ref(v_replacement_285_);
return v_res_288_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_290_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__0));
v___x_291_ = lean_string_utf8_byte_size(v___x_290_);
return v___x_291_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; uint8_t v___x_294_; 
v___x_292_ = lean_unsigned_to_nat(0u);
v___x_293_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1);
v___x_294_ = lean_nat_dec_eq(v___x_293_, v___x_292_);
return v___x_294_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_295_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__1);
v___x_296_ = lean_unsigned_to_nat(0u);
v___x_297_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__0));
v___x_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_298_, 0, v___x_297_);
lean_ctor_set(v___x_298_, 1, v___x_296_);
lean_ctor_set(v___x_298_, 2, v___x_295_);
return v___x_298_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__4(void){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_299_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3);
v___x_300_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_299_);
return v___x_300_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__5(void){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_301_ = lean_unsigned_to_nat(0u);
v___x_302_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__4, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__4_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__4);
v___x_303_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__3);
v___x_304_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_304_, 0, v___x_303_);
lean_ctor_set(v___x_304_, 1, v___x_302_);
lean_ctor_set(v___x_304_, 2, v___x_301_);
lean_ctor_set(v___x_304_, 3, v___x_301_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg(lean_object* v_s_307_, lean_object* v_replacement_308_){
_start:
{
lean_object* v___x_309_; uint8_t v___x_310_; 
v___x_309_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_310_ = lean_uint8_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__2, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__2_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__2);
if (v___x_310_ == 0)
{
lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_311_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__5, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__5_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__5);
v___x_312_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_307_, v_replacement_308_, v___x_311_, v___x_309_);
return v___x_312_;
}
else
{
lean_object* v___x_313_; lean_object* v___x_314_; 
v___x_313_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__6));
v___x_314_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_307_, v_replacement_308_, v___x_313_, v___x_309_);
return v___x_314_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___boxed(lean_object* v_s_315_, lean_object* v_replacement_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg(v_s_315_, v_replacement_316_);
lean_dec_ref(v_replacement_316_);
return v_res_317_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_319_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__0));
v___x_320_ = lean_string_utf8_byte_size(v___x_319_);
return v___x_320_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; uint8_t v___x_323_; 
v___x_321_ = lean_unsigned_to_nat(0u);
v___x_322_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1);
v___x_323_ = lean_nat_dec_eq(v___x_322_, v___x_321_);
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_324_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__1);
v___x_325_ = lean_unsigned_to_nat(0u);
v___x_326_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__0));
v___x_327_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_327_, 0, v___x_326_);
lean_ctor_set(v___x_327_, 1, v___x_325_);
lean_ctor_set(v___x_327_, 2, v___x_324_);
return v___x_327_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3);
v___x_329_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_328_);
return v___x_329_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_330_ = lean_unsigned_to_nat(0u);
v___x_331_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__4, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__4_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__4);
v___x_332_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__3);
v___x_333_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
lean_ctor_set(v___x_333_, 1, v___x_331_);
lean_ctor_set(v___x_333_, 2, v___x_330_);
lean_ctor_set(v___x_333_, 3, v___x_330_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg(lean_object* v_s_334_, lean_object* v_replacement_335_){
_start:
{
lean_object* v___x_336_; uint8_t v___x_337_; 
v___x_336_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_337_ = lean_uint8_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__2, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__2_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__2);
if (v___x_337_ == 0)
{
lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_338_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__5, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__5_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___closed__5);
v___x_339_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_334_, v_replacement_335_, v___x_338_, v___x_336_);
return v___x_339_;
}
else
{
lean_object* v___x_340_; lean_object* v___x_341_; 
v___x_340_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__6));
v___x_341_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_334_, v_replacement_335_, v___x_340_, v___x_336_);
return v___x_341_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg___boxed(lean_object* v_s_342_, lean_object* v_replacement_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg(v_s_342_, v_replacement_343_);
lean_dec_ref(v_replacement_343_);
return v_res_344_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__1(void){
_start:
{
lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_346_ = ((lean_object*)(lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0));
v___x_347_ = lean_string_utf8_byte_size(v___x_346_);
return v___x_347_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3(lean_object* v___x_348_, lean_object* v_x_349_){
_start:
{
if (lean_obj_tag(v_x_349_) == 0)
{
uint8_t v___x_350_; 
v___x_350_ = 1;
return v___x_350_;
}
else
{
lean_object* v_head_351_; lean_object* v_tail_352_; uint8_t v___y_354_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; uint8_t v___x_360_; 
v_head_351_ = lean_ctor_get(v_x_349_, 0);
v_tail_352_ = lean_ctor_get(v_x_349_, 1);
v___x_356_ = ((lean_object*)(lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0));
v___x_357_ = lean_unsigned_to_nat(0u);
v___x_358_ = lean_string_utf8_byte_size(v_head_351_);
v___x_359_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__1, &lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__1_once, _init_lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__1);
v___x_360_ = lean_nat_dec_le(v___x_359_, v___x_358_);
if (v___x_360_ == 0)
{
uint8_t v___x_361_; 
v___x_361_ = lean_nat_dec_eq(v___x_348_, v___x_357_);
v___y_354_ = v___x_361_;
goto v___jp_353_;
}
else
{
lean_object* v___x_362_; uint8_t v___x_363_; 
v___x_362_ = lean_nat_sub(v___x_358_, v___x_359_);
v___x_363_ = lean_string_memcmp(v_head_351_, v___x_356_, v___x_362_, v___x_357_, v___x_359_);
lean_dec(v___x_362_);
v___y_354_ = v___x_363_;
goto v___jp_353_;
}
v___jp_353_:
{
if (v___y_354_ == 0)
{
return v___y_354_;
}
else
{
v_x_349_ = v_tail_352_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___boxed(lean_object* v___x_364_, lean_object* v_x_365_){
_start:
{
uint8_t v_res_366_; lean_object* v_r_367_; 
v_res_366_ = lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3(v___x_364_, v_x_365_);
lean_dec(v_x_365_);
lean_dec(v___x_364_);
v_r_367_ = lean_box(v_res_366_);
return v_r_367_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_369_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__0));
v___x_370_ = lean_string_utf8_byte_size(v___x_369_);
return v___x_370_;
}
}
static uint8_t _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_371_ = lean_unsigned_to_nat(0u);
v___x_372_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1);
v___x_373_ = lean_nat_dec_eq(v___x_372_, v___x_371_);
return v___x_373_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_374_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__1);
v___x_375_ = lean_unsigned_to_nat(0u);
v___x_376_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__0));
v___x_377_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_377_, 0, v___x_376_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
lean_ctor_set(v___x_377_, 2, v___x_374_);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__4(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3);
v___x_379_ = l_String_Slice_Pattern_ForwardSliceSearcher_buildTable(v___x_378_);
return v___x_379_;
}
}
static lean_object* _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; 
v___x_380_ = lean_unsigned_to_nat(0u);
v___x_381_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__4, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__4_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__4);
v___x_382_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__3);
v___x_383_ = lean_alloc_ctor(2, 4, 0);
lean_ctor_set(v___x_383_, 0, v___x_382_);
lean_ctor_set(v___x_383_, 1, v___x_381_);
lean_ctor_set(v___x_383_, 2, v___x_380_);
lean_ctor_set(v___x_383_, 3, v___x_380_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg(lean_object* v_s_384_, lean_object* v_replacement_385_){
_start:
{
lean_object* v___x_386_; uint8_t v___x_387_; 
v___x_386_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_387_ = lean_uint8_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__2, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__2_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__2);
if (v___x_387_ == 0)
{
lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_388_ = lean_obj_once(&lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__5, &lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__5_once, _init_lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___closed__5);
v___x_389_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_384_, v_replacement_385_, v___x_388_, v___x_386_);
return v___x_389_;
}
else
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = ((lean_object*)(lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg___closed__6));
v___x_391_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_384_, v_replacement_385_, v___x_390_, v___x_386_);
return v___x_391_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg___boxed(lean_object* v_s_392_, lean_object* v_replacement_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg(v_s_392_, v_replacement_393_);
lean_dec_ref(v_replacement_393_);
return v_res_394_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__9(void){
_start:
{
lean_object* v_copStop_404_; lean_object* v___x_405_; 
v_copStop_404_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__1));
v___x_405_ = lean_string_utf8_byte_size(v_copStop_404_);
return v___x_405_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__14(void){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_417_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__13));
v___x_418_ = lean_string_utf8_byte_size(v___x_417_);
return v___x_418_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__15(void){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_419_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__14, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__14_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__14);
v___x_420_ = lean_unsigned_to_nat(0u);
v___x_421_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__13));
v___x_422_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v___x_420_);
lean_ctor_set(v___x_422_, 2, v___x_419_);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__18(void){
_start:
{
lean_object* v_copStart_425_; lean_object* v___x_426_; 
v_copStart_425_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__17));
v___x_426_ = lean_string_utf8_byte_size(v_copStart_425_);
return v___x_426_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__22(void){
_start:
{
uint32_t v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; 
v___x_430_ = 45;
v___x_431_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__21));
v___x_432_ = lean_string_push(v___x_431_, v___x_430_);
return v___x_432_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__23(void){
_start:
{
lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_433_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__22, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__22_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__22);
v___x_434_ = lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0(v___x_433_);
return v___x_434_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24(void){
_start:
{
lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_435_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3));
v___x_436_ = lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___lam__0(v___x_435_);
return v___x_436_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__25(void){
_start:
{
lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_437_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19));
v___x_438_ = lean_string_utf8_byte_size(v___x_437_);
return v___x_438_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__26(void){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_439_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__25, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__25_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__25);
v___x_440_ = lean_unsigned_to_nat(0u);
v___x_441_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19));
v___x_442_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
lean_ctor_set(v___x_442_, 1, v___x_440_);
lean_ctor_set(v___x_442_, 2, v___x_439_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks(lean_object* v_copyright_443_, lean_object* v_expectedLicense_444_){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v_copStop_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v_preprocessCopyright_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v_copyright_463_; lean_object* v___y_465_; lean_object* v___y_472_; lean_object* v___y_473_; uint8_t v___y_482_; lean_object* v___y_483_; lean_object* v___y_484_; lean_object* v___y_485_; lean_object* v_output_486_; uint8_t v___y_491_; lean_object* v___y_492_; lean_object* v___y_493_; lean_object* v___y_494_; lean_object* v___y_495_; uint8_t v___y_496_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v___y_504_; lean_object* v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v_output_508_; lean_object* v___y_533_; lean_object* v___y_534_; lean_object* v___y_535_; lean_object* v___y_536_; lean_object* v___y_537_; lean_object* v___y_538_; lean_object* v___y_539_; lean_object* v___y_550_; lean_object* v___y_551_; lean_object* v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v_output_556_; lean_object* v___y_563_; lean_object* v___y_564_; lean_object* v___y_565_; lean_object* v___y_566_; lean_object* v___y_567_; lean_object* v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; uint8_t v___y_572_; lean_object* v___y_581_; lean_object* v___y_582_; lean_object* v___y_583_; lean_object* v___y_584_; lean_object* v___y_585_; lean_object* v___y_586_; lean_object* v___y_587_; lean_object* v___y_588_; lean_object* v___y_589_; lean_object* v___y_590_; lean_object* v_output_591_; lean_object* v___y_609_; lean_object* v___y_610_; lean_object* v___y_611_; lean_object* v___y_612_; lean_object* v___y_613_; lean_object* v___y_614_; lean_object* v___y_615_; lean_object* v___y_616_; lean_object* v___y_617_; lean_object* v___y_618_; lean_object* v___y_619_; uint8_t v___y_620_; lean_object* v___y_627_; lean_object* v___y_628_; lean_object* v___y_629_; lean_object* v___y_630_; lean_object* v___y_631_; lean_object* v___y_632_; lean_object* v_output_633_; lean_object* v___y_648_; lean_object* v___y_649_; lean_object* v___y_650_; lean_object* v___y_651_; lean_object* v___y_652_; lean_object* v___y_653_; lean_object* v___y_654_; lean_object* v___y_665_; lean_object* v___y_666_; lean_object* v___y_667_; lean_object* v___y_668_; lean_object* v___y_669_; lean_object* v___y_670_; lean_object* v_output_671_; lean_object* v_output_678_; lean_object* v_output_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; uint8_t v___x_717_; 
v___x_445_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__0));
v___x_446_ = lean_unsigned_to_nat(0u);
v___x_447_ = lean_string_utf8_byte_size(v_copyright_443_);
v___x_448_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_448_, 0, v_copyright_443_);
lean_ctor_set(v___x_448_, 1, v___x_446_);
lean_ctor_set(v___x_448_, 2, v___x_447_);
v___x_449_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg(v___x_448_, v___x_445_);
v___x_450_ = ((lean_object*)(lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3___closed__0));
v___x_451_ = lean_string_utf8_byte_size(v___x_449_);
v___x_452_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_452_, 0, v___x_449_);
lean_ctor_set(v___x_452_, 1, v___x_446_);
lean_ctor_set(v___x_452_, 2, v___x_451_);
v___x_453_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg(v___x_452_, v___x_450_);
v_copStop_454_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__1));
v___x_455_ = lean_string_utf8_byte_size(v___x_453_);
v___x_456_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_456_, 0, v___x_453_);
lean_ctor_set(v___x_456_, 1, v___x_446_);
lean_ctor_set(v___x_456_, 2, v___x_455_);
v_preprocessCopyright_457_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg(v___x_456_, v_copStop_454_);
v___x_458_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__2));
v___x_459_ = lean_box(0);
v___x_460_ = l_String_splitOnAux(v_preprocessCopyright_457_, v___x_458_, v___x_446_, v___x_446_, v___x_446_, v___x_459_);
lean_dec_ref(v_preprocessCopyright_457_);
v___x_461_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_462_ = l_List_getD___redArg(v___x_460_, v___x_446_, v___x_461_);
v_copyright_463_ = lean_string_append(v___x_462_, v___x_458_);
v_output_708_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks___closed__0));
v___x_709_ = lean_unsigned_to_nat(1u);
v___x_710_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19));
v___x_711_ = l_List_getD___redArg(v___x_460_, v___x_709_, v___x_710_);
lean_dec(v___x_460_);
v___x_712_ = lean_string_utf8_byte_size(v___x_711_);
lean_inc(v___x_711_);
v___x_713_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_713_, 0, v___x_711_);
lean_ctor_set(v___x_713_, 1, v___x_446_);
lean_ctor_set(v___x_713_, 2, v___x_712_);
v___x_714_ = l_String_Slice_Pos_nextn(v___x_713_, v___x_446_, v___x_709_);
lean_dec_ref_known(v___x_713_, 3);
v___x_715_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_715_, 0, v___x_711_);
lean_ctor_set(v___x_715_, 1, v___x_446_);
lean_ctor_set(v___x_715_, 2, v___x_714_);
v___x_716_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__26, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__26_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__26);
v___x_717_ = l_String_Slice_beq(v___x_715_, v___x_716_);
lean_dec_ref_known(v___x_715_, 3);
if (v___x_717_ == 0)
{
lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v_output_722_; 
v___x_718_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3));
lean_inc_ref(v_copyright_463_);
v___x_719_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_718_, v___x_446_);
v___x_720_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24);
v___x_721_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_721_, 0, v___x_719_);
lean_ctor_set(v___x_721_, 1, v___x_720_);
v_output_722_ = lean_array_push(v_output_708_, v___x_721_);
v_output_678_ = v_output_722_;
goto v___jp_677_;
}
else
{
v_output_678_ = v_output_708_;
goto v___jp_677_;
}
v___jp_464_:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v_output_470_; 
v___x_466_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3));
v___x_467_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_466_, v___x_446_);
v___x_468_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__4));
v___x_469_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_469_, 0, v___x_467_);
lean_ctor_set(v___x_469_, 1, v___x_468_);
v_output_470_ = lean_array_push(v___y_465_, v___x_469_);
return v_output_470_;
}
v___jp_471_:
{
lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v_output_480_; 
v___x_474_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___y_473_, v___x_446_);
v___x_475_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__5));
v___x_476_ = lean_string_append(v___x_475_, v_expectedLicense_444_);
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__6));
v___x_478_ = lean_string_append(v___x_476_, v___x_477_);
v___x_479_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_479_, 0, v___x_474_);
lean_ctor_set(v___x_479_, 1, v___x_478_);
v_output_480_ = lean_array_push(v___y_472_, v___x_479_);
return v_output_480_;
}
v___jp_481_:
{
lean_object* v___x_487_; lean_object* v_output_488_; uint8_t v___x_489_; 
v___x_487_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_authorsLineChecks(v___y_484_, v___y_483_);
lean_dec(v___y_483_);
v_output_488_ = l_Array_append___redArg(v_output_486_, v___x_487_);
lean_dec_ref(v___x_487_);
v___x_489_ = lean_string_dec_eq(v___y_485_, v_expectedLicense_444_);
if (v___x_489_ == 0)
{
v___y_472_ = v_output_488_;
v___y_473_ = v___y_485_;
goto v___jp_471_;
}
else
{
if (v___y_482_ == 0)
{
lean_dec_ref(v___y_485_);
lean_dec_ref(v_copyright_463_);
return v_output_488_;
}
else
{
v___y_472_ = v_output_488_;
v___y_473_ = v___y_485_;
goto v___jp_471_;
}
}
}
v___jp_490_:
{
if (v___y_496_ == 0)
{
v___y_482_ = v___y_491_;
v___y_483_ = v___y_493_;
v___y_484_ = v___y_492_;
v___y_485_ = v___y_495_;
v_output_486_ = v___y_494_;
goto v___jp_481_;
}
else
{
lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v_output_500_; 
lean_inc_ref(v___y_492_);
lean_inc_ref(v_copyright_463_);
v___x_497_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___y_492_, v___x_446_);
v___x_498_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__7));
v___x_499_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_499_, 0, v___x_497_);
lean_ctor_set(v___x_499_, 1, v___x_498_);
v_output_500_ = lean_array_push(v___y_494_, v___x_499_);
v___y_482_ = v___y_491_;
v___y_483_ = v___y_493_;
v___y_484_ = v___y_492_;
v___y_485_ = v___y_495_;
v_output_486_ = v_output_500_;
goto v___jp_481_;
}
}
v___jp_501_:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v_authorsLines_511_; lean_object* v___x_512_; uint8_t v___x_513_; 
v___x_509_ = lean_array_mk(v___y_503_);
v___x_510_ = lean_array_pop(v___x_509_);
v_authorsLines_511_ = lean_array_to_list(v___x_510_);
v___x_512_ = l_List_lengthTR___redArg(v_authorsLines_511_);
v___x_513_ = lean_nat_dec_eq(v___x_512_, v___x_446_);
if (v___x_513_ == 0)
{
lean_object* v_authorsLine_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v_authorsStart_520_; lean_object* v___x_521_; uint8_t v___x_522_; 
lean_inc(v_authorsLines_511_);
v_authorsLine_514_ = l_String_intercalate(v___y_505_, v_authorsLines_511_);
v___x_515_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_515_, 0, v___x_461_);
lean_ctor_set(v___x_515_, 1, v___y_502_);
lean_inc_ref(v___y_507_);
v___x_516_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_516_, 0, v___y_507_);
lean_ctor_set(v___x_516_, 1, v___x_515_);
v___x_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_517_, 0, v___y_504_);
lean_ctor_set(v___x_517_, 1, v___x_516_);
v___x_518_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_518_, 0, v___y_506_);
lean_ctor_set(v___x_518_, 1, v___x_517_);
v___x_519_ = l_String_intercalate(v___y_505_, v___x_518_);
v_authorsStart_520_ = lean_string_utf8_byte_size(v___x_519_);
lean_dec_ref(v___x_519_);
v___x_521_ = lean_unsigned_to_nat(1u);
v___x_522_ = lean_nat_dec_lt(v___x_521_, v___x_512_);
if (v___x_522_ == 0)
{
lean_dec(v___x_512_);
lean_dec(v_authorsLines_511_);
v___y_491_ = v___x_513_;
v___y_492_ = v_authorsLine_514_;
v___y_493_ = v_authorsStart_520_;
v___y_494_ = v_output_508_;
v___y_495_ = v___y_507_;
v___y_496_ = v___x_522_;
goto v___jp_490_;
}
else
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; uint8_t v___x_526_; 
v___x_523_ = lean_array_mk(v_authorsLines_511_);
v___x_524_ = lean_array_pop(v___x_523_);
v___x_525_ = lean_array_to_list(v___x_524_);
v___x_526_ = lp_mathlib_List_all___at___00Mathlib_Linter_copyrightHeaderChecks_spec__3(v___x_512_, v___x_525_);
lean_dec(v___x_525_);
lean_dec(v___x_512_);
if (v___x_526_ == 0)
{
v___y_491_ = v___x_513_;
v___y_492_ = v_authorsLine_514_;
v___y_493_ = v_authorsStart_520_;
v___y_494_ = v_output_508_;
v___y_495_ = v___y_507_;
v___y_496_ = v___x_522_;
goto v___jp_490_;
}
else
{
v___y_491_ = v___x_513_;
v___y_492_ = v_authorsLine_514_;
v___y_493_ = v_authorsStart_520_;
v___y_494_ = v_output_508_;
v___y_495_ = v___y_507_;
v___y_496_ = v___x_513_;
goto v___jp_490_;
}
}
}
else
{
lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v_output_531_; 
lean_dec(v___x_512_);
lean_dec(v_authorsLines_511_);
lean_dec_ref(v___y_507_);
lean_dec_ref(v___y_506_);
lean_dec_ref(v___y_504_);
lean_dec(v___y_502_);
v___x_527_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3));
v___x_528_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_527_, v___x_446_);
v___x_529_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__4));
v___x_530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_530_, 0, v___x_528_);
lean_ctor_set(v___x_530_, 1, v___x_529_);
v_output_531_ = lean_array_push(v_output_508_, v___x_530_);
return v_output_531_;
}
}
v___jp_532_:
{
lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v_output_548_; 
v___x_540_ = lean_unsigned_to_nat(22u);
v___x_541_ = lean_string_utf8_byte_size(v___y_536_);
lean_inc_ref(v___y_536_);
v___x_542_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_542_, 0, v___y_536_);
lean_ctor_set(v___x_542_, 1, v___x_446_);
lean_ctor_set(v___x_542_, 2, v___x_541_);
v___x_543_ = l_String_Slice_Pos_prevn(v___x_542_, v___x_541_, v___x_540_);
lean_dec_ref_known(v___x_542_, 3);
v___x_544_ = lean_string_utf8_extract_fast(v___y_536_, v___x_543_, v___x_541_);
lean_dec(v___x_543_);
lean_inc_ref(v_copyright_463_);
v___x_545_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_544_, v___x_446_);
v___x_546_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__8));
v___x_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_547_, 0, v___x_545_);
lean_ctor_set(v___x_547_, 1, v___x_546_);
v_output_548_ = lean_array_push(v___y_535_, v___x_547_);
v___y_502_ = v___y_533_;
v___y_503_ = v___y_534_;
v___y_504_ = v___y_536_;
v___y_505_ = v___y_537_;
v___y_506_ = v___y_538_;
v___y_507_ = v___y_539_;
v_output_508_ = v_output_548_;
goto v___jp_501_;
}
v___jp_549_:
{
lean_object* v___x_557_; lean_object* v___x_558_; uint8_t v___x_559_; 
v___x_557_ = lean_string_utf8_byte_size(v___y_552_);
v___x_558_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__9, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__9_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__9);
v___x_559_ = lean_nat_dec_le(v___x_558_, v___x_557_);
if (v___x_559_ == 0)
{
v___y_533_ = v___y_550_;
v___y_534_ = v___y_551_;
v___y_535_ = v_output_556_;
v___y_536_ = v___y_552_;
v___y_537_ = v___y_553_;
v___y_538_ = v___y_554_;
v___y_539_ = v___y_555_;
goto v___jp_532_;
}
else
{
lean_object* v___x_560_; uint8_t v___x_561_; 
v___x_560_ = lean_nat_sub(v___x_557_, v___x_558_);
v___x_561_ = lean_string_memcmp(v___y_552_, v_copStop_454_, v___x_560_, v___x_446_, v___x_558_);
lean_dec(v___x_560_);
if (v___x_561_ == 0)
{
v___y_533_ = v___y_550_;
v___y_534_ = v___y_551_;
v___y_535_ = v_output_556_;
v___y_536_ = v___y_552_;
v___y_537_ = v___y_553_;
v___y_538_ = v___y_554_;
v___y_539_ = v___y_555_;
goto v___jp_532_;
}
else
{
v___y_502_ = v___y_550_;
v___y_503_ = v___y_551_;
v___y_504_ = v___y_552_;
v___y_505_ = v___y_553_;
v___y_506_ = v___y_554_;
v___y_507_ = v___y_555_;
v_output_508_ = v_output_556_;
goto v___jp_501_;
}
}
}
v___jp_562_:
{
if (v___y_572_ == 0)
{
lean_dec(v___y_567_);
lean_dec_ref(v___y_564_);
v___y_550_ = v___y_563_;
v___y_551_ = v___y_566_;
v___y_552_ = v___y_568_;
v___y_553_ = v___y_569_;
v___y_554_ = v___y_570_;
v___y_555_ = v___y_571_;
v_output_556_ = v___y_565_;
goto v___jp_549_;
}
else
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v_output_579_; 
v___x_573_ = lean_unsigned_to_nat(19u);
v___x_574_ = l_String_Slice_Pos_nextn(v___y_564_, v___x_446_, v___x_573_);
lean_dec_ref(v___y_564_);
v___x_575_ = lean_string_utf8_extract_fast(v___y_568_, v___x_574_, v___y_567_);
lean_dec(v___y_567_);
lean_dec(v___x_574_);
lean_inc_ref(v_copyright_463_);
v___x_576_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_575_, v___x_446_);
v___x_577_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__10));
v___x_578_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_576_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v_output_579_ = lean_array_push(v___y_565_, v___x_578_);
v___y_550_ = v___y_563_;
v___y_551_ = v___y_566_;
v___y_552_ = v___y_568_;
v___y_553_ = v___y_569_;
v___y_554_ = v___y_570_;
v___y_555_ = v___y_571_;
v_output_556_ = v_output_579_;
goto v___jp_549_;
}
}
v___jp_580_:
{
lean_object* v___x_592_; uint8_t v___x_593_; 
v___x_592_ = lean_array_get_size(v_output_591_);
v___x_593_ = lean_nat_dec_eq(v___x_592_, v___x_446_);
if (v___x_593_ == 0)
{
lean_dec(v___y_589_);
lean_dec_ref(v___y_587_);
v___y_563_ = v___y_581_;
v___y_564_ = v___y_582_;
v___y_565_ = v_output_591_;
v___y_566_ = v___y_583_;
v___y_567_ = v___y_584_;
v___y_568_ = v___y_585_;
v___y_569_ = v___y_586_;
v___y_570_ = v___y_588_;
v___y_571_ = v___y_590_;
v___y_572_ = v___x_593_;
goto v___jp_562_;
}
else
{
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v_str_602_; lean_object* v_startInclusive_603_; lean_object* v_endExclusive_604_; lean_object* v___x_605_; lean_object* v___x_606_; uint8_t v___x_607_; 
v___x_594_ = lean_unsigned_to_nat(1u);
v___x_595_ = l_String_Slice_Pos_nextn(v___y_587_, v___x_446_, v___x_594_);
lean_dec_ref(v___y_587_);
v___x_596_ = lean_nat_add(v___y_589_, v___x_595_);
lean_dec(v___x_595_);
lean_dec(v___y_589_);
lean_inc(v___y_584_);
lean_inc(v___x_596_);
lean_inc_ref_n(v___y_585_, 2);
v___x_597_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_597_, 0, v___y_585_);
lean_ctor_set(v___x_597_, 1, v___x_596_);
lean_ctor_set(v___x_597_, 2, v___y_584_);
v___x_598_ = l_String_Slice_Pos_nextn(v___x_597_, v___x_446_, v___x_594_);
lean_dec_ref_known(v___x_597_, 3);
v___x_599_ = lean_nat_add(v___x_596_, v___x_598_);
lean_dec(v___x_598_);
v___x_600_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_600_, 0, v___y_585_);
lean_ctor_set(v___x_600_, 1, v___x_596_);
lean_ctor_set(v___x_600_, 2, v___x_599_);
v___x_601_ = l_String_Slice_trimAscii(v___x_600_);
v_str_602_ = lean_ctor_get(v___x_601_, 0);
lean_inc_ref(v_str_602_);
v_startInclusive_603_ = lean_ctor_get(v___x_601_, 1);
lean_inc(v_startInclusive_603_);
v_endExclusive_604_ = lean_ctor_get(v___x_601_, 2);
lean_inc(v_endExclusive_604_);
lean_dec_ref(v___x_601_);
v___x_605_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__11));
v___x_606_ = lean_string_utf8_extract_fast(v_str_602_, v_startInclusive_603_, v_endExclusive_604_);
lean_dec(v_endExclusive_604_);
lean_dec(v_startInclusive_603_);
lean_dec_ref(v_str_602_);
v___x_607_ = lp_mathlib_Array_contains___at___00Mathlib_Linter_copyrightHeaderChecks_spec__4(v___x_605_, v___x_606_);
lean_dec_ref(v___x_606_);
v___y_563_ = v___y_581_;
v___y_564_ = v___y_582_;
v___y_565_ = v_output_591_;
v___y_566_ = v___y_583_;
v___y_567_ = v___y_584_;
v___y_568_ = v___y_585_;
v___y_569_ = v___y_586_;
v___y_570_ = v___y_588_;
v___y_571_ = v___y_590_;
v___y_572_ = v___x_607_;
goto v___jp_562_;
}
}
v___jp_608_:
{
if (v___y_620_ == 0)
{
v___y_581_ = v___y_609_;
v___y_582_ = v___y_610_;
v___y_583_ = v___y_611_;
v___y_584_ = v___y_612_;
v___y_585_ = v___y_613_;
v___y_586_ = v___y_614_;
v___y_587_ = v___y_615_;
v___y_588_ = v___y_616_;
v___y_589_ = v___y_618_;
v___y_590_ = v___y_619_;
v_output_591_ = v___y_617_;
goto v___jp_580_;
}
else
{
lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v_output_625_; 
v___x_621_ = lean_string_utf8_extract_fast(v___y_613_, v___y_618_, v___y_612_);
lean_inc_ref(v_copyright_463_);
v___x_622_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_621_, v___x_446_);
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__12));
v___x_624_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_624_, 0, v___x_622_);
lean_ctor_set(v___x_624_, 1, v___x_623_);
v_output_625_ = lean_array_push(v___y_617_, v___x_624_);
v___y_581_ = v___y_609_;
v___y_582_ = v___y_610_;
v___y_583_ = v___y_611_;
v___y_584_ = v___y_612_;
v___y_585_ = v___y_613_;
v___y_586_ = v___y_614_;
v___y_587_ = v___y_615_;
v___y_588_ = v___y_616_;
v___y_589_ = v___y_618_;
v___y_590_ = v___y_619_;
v_output_591_ = v_output_625_;
goto v___jp_580_;
}
}
v___jp_626_:
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v_author_638_; lean_object* v___x_639_; uint8_t v___x_640_; 
v___x_634_ = lean_unsigned_to_nat(18u);
v___x_635_ = lean_string_utf8_byte_size(v___y_629_);
lean_inc_ref_n(v___y_629_, 2);
v___x_636_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_636_, 0, v___y_629_);
lean_ctor_set(v___x_636_, 1, v___x_446_);
lean_ctor_set(v___x_636_, 2, v___x_635_);
v___x_637_ = l_String_Slice_Pos_nextn(v___x_636_, v___x_446_, v___x_634_);
lean_inc(v___x_637_);
v_author_638_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_author_638_, 0, v___y_629_);
lean_ctor_set(v_author_638_, 1, v___x_637_);
lean_ctor_set(v_author_638_, 2, v___x_635_);
v___x_639_ = lean_array_get_size(v_output_633_);
v___x_640_ = lean_nat_dec_eq(v___x_639_, v___x_446_);
if (v___x_640_ == 0)
{
v___y_609_ = v___y_627_;
v___y_610_ = v___x_636_;
v___y_611_ = v___y_628_;
v___y_612_ = v___x_635_;
v___y_613_ = v___y_629_;
v___y_614_ = v___y_630_;
v___y_615_ = v_author_638_;
v___y_616_ = v___y_631_;
v___y_617_ = v_output_633_;
v___y_618_ = v___x_637_;
v___y_619_ = v___y_632_;
v___y_620_ = v___x_640_;
goto v___jp_608_;
}
else
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; uint8_t v___x_646_; 
v___x_641_ = lean_unsigned_to_nat(1u);
v___x_642_ = l_String_Slice_Pos_nextn(v_author_638_, v___x_446_, v___x_641_);
v___x_643_ = lean_nat_add(v___x_637_, v___x_642_);
lean_dec(v___x_642_);
lean_inc(v___x_637_);
lean_inc_ref(v___y_629_);
v___x_644_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_644_, 0, v___y_629_);
lean_ctor_set(v___x_644_, 1, v___x_637_);
lean_ctor_set(v___x_644_, 2, v___x_643_);
v___x_645_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__15, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__15_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__15);
v___x_646_ = l_String_Slice_beq(v___x_644_, v___x_645_);
lean_dec_ref_known(v___x_644_, 3);
if (v___x_646_ == 0)
{
v___y_609_ = v___y_627_;
v___y_610_ = v___x_636_;
v___y_611_ = v___y_628_;
v___y_612_ = v___x_635_;
v___y_613_ = v___y_629_;
v___y_614_ = v___y_630_;
v___y_615_ = v_author_638_;
v___y_616_ = v___y_631_;
v___y_617_ = v_output_633_;
v___y_618_ = v___x_637_;
v___y_619_ = v___y_632_;
v___y_620_ = v___x_640_;
goto v___jp_608_;
}
else
{
v___y_581_ = v___y_627_;
v___y_582_ = v___x_636_;
v___y_583_ = v___y_628_;
v___y_584_ = v___x_635_;
v___y_585_ = v___y_629_;
v___y_586_ = v___y_630_;
v___y_587_ = v_author_638_;
v___y_588_ = v___y_631_;
v___y_589_ = v___x_637_;
v___y_590_ = v___y_632_;
v_output_591_ = v_output_633_;
goto v___jp_580_;
}
}
}
v___jp_647_:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v_output_663_; 
v___x_655_ = lean_unsigned_to_nat(16u);
v___x_656_ = lean_string_utf8_byte_size(v___y_650_);
lean_inc_ref(v___y_650_);
v___x_657_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_657_, 0, v___y_650_);
lean_ctor_set(v___x_657_, 1, v___x_446_);
lean_ctor_set(v___x_657_, 2, v___x_656_);
v___x_658_ = l_String_Slice_Pos_nextn(v___x_657_, v___x_446_, v___x_655_);
lean_dec_ref_known(v___x_657_, 3);
v___x_659_ = lean_string_utf8_extract_fast(v___y_650_, v___x_446_, v___x_658_);
lean_dec(v___x_658_);
lean_inc_ref(v_copyright_463_);
v___x_660_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v___x_659_, v___x_446_);
v___x_661_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__16));
v___x_662_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_662_, 0, v___x_660_);
lean_ctor_set(v___x_662_, 1, v___x_661_);
v_output_663_ = lean_array_push(v___y_653_, v___x_662_);
v___y_627_ = v___y_648_;
v___y_628_ = v___y_649_;
v___y_629_ = v___y_650_;
v___y_630_ = v___y_651_;
v___y_631_ = v___y_652_;
v___y_632_ = v___y_654_;
v_output_633_ = v_output_663_;
goto v___jp_626_;
}
v___jp_664_:
{
lean_object* v_copStart_672_; lean_object* v___x_673_; lean_object* v___x_674_; uint8_t v___x_675_; 
v_copStart_672_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__17));
v___x_673_ = lean_string_utf8_byte_size(v___y_667_);
v___x_674_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__18, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__18_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__18);
v___x_675_ = lean_nat_dec_le(v___x_674_, v___x_673_);
if (v___x_675_ == 0)
{
v___y_648_ = v___y_665_;
v___y_649_ = v___y_666_;
v___y_650_ = v___y_667_;
v___y_651_ = v___y_668_;
v___y_652_ = v___y_669_;
v___y_653_ = v_output_671_;
v___y_654_ = v___y_670_;
goto v___jp_647_;
}
else
{
uint8_t v___x_676_; 
v___x_676_ = lean_string_memcmp(v___y_667_, v_copStart_672_, v___x_446_, v___x_446_, v___x_674_);
if (v___x_676_ == 0)
{
v___y_648_ = v___y_665_;
v___y_649_ = v___y_666_;
v___y_650_ = v___y_667_;
v___y_651_ = v___y_668_;
v___y_652_ = v___y_669_;
v___y_653_ = v_output_671_;
v___y_654_ = v___y_670_;
goto v___jp_647_;
}
else
{
v___y_627_ = v___y_665_;
v___y_628_ = v___y_666_;
v___y_629_ = v___y_667_;
v___y_630_ = v___y_668_;
v___y_631_ = v___y_669_;
v___y_632_ = v___y_670_;
v_output_633_ = v_output_671_;
goto v___jp_626_;
}
}
}
v___jp_677_:
{
lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_679_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19));
v___x_680_ = l_String_splitOnAux(v_copyright_463_, v___x_679_, v___x_446_, v___x_446_, v___x_446_, v___x_459_);
if (lean_obj_tag(v___x_680_) == 1)
{
lean_object* v_tail_681_; 
v_tail_681_ = lean_ctor_get(v___x_680_, 1);
lean_inc(v_tail_681_);
if (lean_obj_tag(v_tail_681_) == 1)
{
lean_object* v_tail_682_; 
v_tail_682_ = lean_ctor_get(v_tail_681_, 1);
lean_inc(v_tail_682_);
if (lean_obj_tag(v_tail_682_) == 1)
{
lean_object* v_head_683_; lean_object* v_head_684_; lean_object* v_head_685_; lean_object* v_tail_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_707_; 
v_head_683_ = lean_ctor_get(v___x_680_, 0);
lean_inc(v_head_683_);
v_head_684_ = lean_ctor_get(v_tail_681_, 0);
lean_inc(v_head_684_);
lean_dec_ref_known(v_tail_681_, 2);
v_head_685_ = lean_ctor_get(v_tail_682_, 0);
v_tail_686_ = lean_ctor_get(v_tail_682_, 1);
v_isSharedCheck_707_ = !lean_is_exclusive(v_tail_682_);
if (v_isSharedCheck_707_ == 0)
{
v___x_688_ = v_tail_682_;
v_isShared_689_ = v_isSharedCheck_707_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_tail_686_);
lean_inc(v_head_685_);
lean_dec(v_tail_682_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_707_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
lean_object* v___x_690_; uint8_t v___x_691_; 
v___x_690_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__20));
v___x_691_ = lean_string_dec_eq(v_head_683_, v___x_690_);
if (v___x_691_ == 0)
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_695_; 
lean_dec_ref_known(v___x_680_, 2);
lean_inc(v_head_683_);
lean_inc_ref(v_copyright_463_);
v___x_692_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v_head_683_, v___x_446_);
v___x_693_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__23, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__23_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__23);
if (v_isShared_689_ == 0)
{
lean_ctor_set_tag(v___x_688_, 0);
lean_ctor_set(v___x_688_, 1, v___x_693_);
lean_ctor_set(v___x_688_, 0, v___x_692_);
v___x_695_ = v___x_688_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_697_; 
v_reuseFailAlloc_697_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_697_, 0, v___x_692_);
lean_ctor_set(v_reuseFailAlloc_697_, 1, v___x_693_);
v___x_695_ = v_reuseFailAlloc_697_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
lean_object* v_output_696_; 
v_output_696_ = lean_array_push(v_output_678_, v___x_695_);
v___y_665_ = v___x_459_;
v___y_666_ = v_tail_686_;
v___y_667_ = v_head_684_;
v___y_668_ = v___x_679_;
v___y_669_ = v_head_683_;
v___y_670_ = v_head_685_;
v_output_671_ = v_output_696_;
goto v___jp_664_;
}
}
else
{
lean_object* v_closeComment_698_; lean_object* v___x_699_; uint8_t v___x_700_; 
v_closeComment_698_ = l_List_getLastD___redArg(v___x_680_, v___x_461_);
lean_dec_ref_known(v___x_680_, 2);
v___x_699_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__3));
v___x_700_ = lean_string_dec_eq(v_closeComment_698_, v___x_699_);
if (v___x_700_ == 0)
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_704_; 
lean_inc_ref(v_copyright_463_);
v___x_701_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax(v_copyright_463_, v_closeComment_698_, v___x_446_);
v___x_702_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24, &lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24_once, _init_lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__24);
if (v_isShared_689_ == 0)
{
lean_ctor_set_tag(v___x_688_, 0);
lean_ctor_set(v___x_688_, 1, v___x_702_);
lean_ctor_set(v___x_688_, 0, v___x_701_);
v___x_704_ = v___x_688_;
goto v_reusejp_703_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v___x_701_);
lean_ctor_set(v_reuseFailAlloc_706_, 1, v___x_702_);
v___x_704_ = v_reuseFailAlloc_706_;
goto v_reusejp_703_;
}
v_reusejp_703_:
{
lean_object* v_output_705_; 
v_output_705_ = lean_array_push(v_output_678_, v___x_704_);
v___y_665_ = v___x_459_;
v___y_666_ = v_tail_686_;
v___y_667_ = v_head_684_;
v___y_668_ = v___x_679_;
v___y_669_ = v_head_683_;
v___y_670_ = v_head_685_;
v_output_671_ = v_output_705_;
goto v___jp_664_;
}
}
else
{
lean_dec(v_closeComment_698_);
lean_del_object(v___x_688_);
v___y_665_ = v___x_459_;
v___y_666_ = v_tail_686_;
v___y_667_ = v_head_684_;
v___y_668_ = v___x_679_;
v___y_669_ = v_head_683_;
v___y_670_ = v_head_685_;
v_output_671_ = v_output_678_;
goto v___jp_664_;
}
}
}
}
else
{
lean_dec_ref_known(v_tail_681_, 2);
lean_dec(v_tail_682_);
lean_dec_ref_known(v___x_680_, 2);
v___y_465_ = v_output_678_;
goto v___jp_464_;
}
}
else
{
lean_dec(v_tail_681_);
lean_dec_ref_known(v___x_680_, 2);
v___y_465_ = v_output_678_;
goto v___jp_464_;
}
}
else
{
lean_dec(v___x_680_);
v___y_465_ = v_output_678_;
goto v___jp_464_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___boxed(lean_object* v_copyright_723_, lean_object* v_expectedLicense_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_mathlib_Mathlib_Linter_copyrightHeaderChecks(v_copyright_723_, v_expectedLicense_724_);
lean_dec_ref(v_expectedLicense_724_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0(lean_object* v_s_726_, lean_object* v_pattern_727_, lean_object* v_replacement_728_){
_start:
{
lean_object* v___x_729_; 
v___x_729_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___redArg(v_s_726_, v_replacement_728_);
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0___boxed(lean_object* v_s_730_, lean_object* v_pattern_731_, lean_object* v_replacement_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0(v_s_730_, v_pattern_731_, v_replacement_732_);
lean_dec_ref(v_replacement_732_);
lean_dec_ref(v_pattern_731_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1(lean_object* v_s_734_, lean_object* v_pattern_735_, lean_object* v_replacement_736_){
_start:
{
lean_object* v___x_737_; 
v___x_737_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___redArg(v_s_734_, v_replacement_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1___boxed(lean_object* v_s_738_, lean_object* v_pattern_739_, lean_object* v_replacement_740_){
_start:
{
lean_object* v_res_741_; 
v_res_741_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__1(v_s_738_, v_pattern_739_, v_replacement_740_);
lean_dec_ref(v_replacement_740_);
lean_dec_ref(v_pattern_739_);
return v_res_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2(lean_object* v_s_742_, lean_object* v_pattern_743_, lean_object* v_replacement_744_){
_start:
{
lean_object* v___x_745_; 
v___x_745_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___redArg(v_s_742_, v_replacement_744_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2___boxed(lean_object* v_s_746_, lean_object* v_pattern_747_, lean_object* v_replacement_748_){
_start:
{
lean_object* v_res_749_; 
v_res_749_ = lp_mathlib_String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__2(v_s_746_, v_pattern_747_, v_replacement_748_);
lean_dec_ref(v_replacement_748_);
lean_dec_ref(v_pattern_747_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0(lean_object* v_s_750_, lean_object* v_replacement_751_, lean_object* v_inst_752_, lean_object* v_R_753_, lean_object* v_a_754_, lean_object* v_b_755_, lean_object* v_c_756_){
_start:
{
lean_object* v___x_757_; 
v___x_757_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___redArg(v_s_750_, v_replacement_751_, v_a_754_, v_b_755_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0___boxed(lean_object* v_s_758_, lean_object* v_replacement_759_, lean_object* v_inst_760_, lean_object* v_R_761_, lean_object* v_a_762_, lean_object* v_b_763_, lean_object* v_c_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00String_Slice_replace___at___00Mathlib_Linter_copyrightHeaderChecks_spec__0_spec__0(v_s_758_, v_replacement_759_, v_inst_760_, v_R_761_, v_a_762_, v_b_763_, v_c_764_);
lean_dec_ref(v_replacement_759_);
return v_res_765_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot_spec__0(lean_object* v_modName_766_, lean_object* v_as_767_, size_t v_i_768_, size_t v_stop_769_){
_start:
{
uint8_t v___x_770_; 
v___x_770_ = lean_usize_dec_eq(v_i_768_, v_stop_769_);
if (v___x_770_ == 0)
{
lean_object* v___x_771_; lean_object* v_module_772_; uint8_t v___x_773_; 
v___x_771_ = lean_array_uget_borrowed(v_as_767_, v_i_768_);
v_module_772_ = lean_ctor_get(v___x_771_, 0);
v___x_773_ = lean_name_eq(v_module_772_, v_modName_766_);
if (v___x_773_ == 0)
{
size_t v___x_774_; size_t v___x_775_; 
v___x_774_ = ((size_t)1ULL);
v___x_775_ = lean_usize_add(v_i_768_, v___x_774_);
v_i_768_ = v___x_775_;
goto _start;
}
else
{
return v___x_773_;
}
}
else
{
uint8_t v___x_777_; 
v___x_777_ = 0;
return v___x_777_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot_spec__0___boxed(lean_object* v_modName_778_, lean_object* v_as_779_, lean_object* v_i_780_, lean_object* v_stop_781_){
_start:
{
size_t v_i_boxed_782_; size_t v_stop_boxed_783_; uint8_t v_res_784_; lean_object* v_r_785_; 
v_i_boxed_782_ = lean_unbox_usize(v_i_780_);
lean_dec(v_i_780_);
v_stop_boxed_783_ = lean_unbox_usize(v_stop_781_);
lean_dec(v_stop_781_);
v_res_784_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot_spec__0(v_modName_778_, v_as_779_, v_i_boxed_782_, v_stop_boxed_783_);
lean_dec_ref(v_as_779_);
lean_dec(v_modName_778_);
v_r_785_ = lean_box(v_res_784_);
return v_r_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot(lean_object* v_modName_787_){
_start:
{
lean_object* v___x_789_; uint8_t v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v_rootPath_793_; uint8_t v___x_794_; 
v___x_789_ = l_Lean_Name_getRoot(v_modName_787_);
v___x_790_ = 1;
v___x_791_ = l_Lean_Name_toString(v___x_789_, v___x_790_);
v___x_792_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot___closed__0));
v_rootPath_793_ = l_System_FilePath_addExtension(v___x_791_, v___x_792_);
v___x_794_ = l_System_FilePath_pathExists(v_rootPath_793_);
if (v___x_794_ == 0)
{
lean_object* v___x_795_; lean_object* v___x_796_; 
lean_dec_ref(v_rootPath_793_);
v___x_795_ = lean_box(v___x_794_);
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v___x_795_);
return v___x_796_;
}
else
{
lean_object* v___x_797_; 
v___x_797_ = l_IO_FS_readFile(v_rootPath_793_);
lean_dec_ref(v_rootPath_793_);
if (lean_obj_tag(v___x_797_) == 0)
{
lean_object* v_a_798_; lean_object* v___x_799_; lean_object* v___x_800_; 
v_a_798_ = lean_ctor_get(v___x_797_, 0);
lean_inc(v_a_798_);
lean_dec_ref_known(v___x_797_, 1);
v___x_799_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___x_800_ = l_Lean_parseImports_x27(v_a_798_, v___x_799_);
if (lean_obj_tag(v___x_800_) == 0)
{
lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_824_; 
v_a_801_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_824_ == 0)
{
v___x_803_ = v___x_800_;
v_isShared_804_ = v_isSharedCheck_824_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_800_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_824_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v_imports_805_; lean_object* v___x_806_; lean_object* v___x_807_; uint8_t v___x_808_; 
v_imports_805_ = lean_ctor_get(v_a_801_, 0);
lean_inc_ref(v_imports_805_);
lean_dec(v_a_801_);
v___x_806_ = lean_unsigned_to_nat(0u);
v___x_807_ = lean_array_get_size(v_imports_805_);
v___x_808_ = lean_nat_dec_lt(v___x_806_, v___x_807_);
if (v___x_808_ == 0)
{
lean_object* v___x_809_; lean_object* v___x_811_; 
lean_dec_ref(v_imports_805_);
v___x_809_ = lean_box(v___x_808_);
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 0, v___x_809_);
v___x_811_ = v___x_803_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v___x_809_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
return v___x_811_;
}
}
else
{
if (v___x_808_ == 0)
{
lean_object* v___x_813_; lean_object* v___x_815_; 
lean_dec_ref(v_imports_805_);
v___x_813_ = lean_box(v___x_808_);
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 0, v___x_813_);
v___x_815_ = v___x_803_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_816_; 
v_reuseFailAlloc_816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_816_, 0, v___x_813_);
v___x_815_ = v_reuseFailAlloc_816_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
return v___x_815_;
}
}
else
{
size_t v___x_817_; size_t v___x_818_; uint8_t v___x_819_; lean_object* v___x_820_; lean_object* v___x_822_; 
v___x_817_ = ((size_t)0ULL);
v___x_818_ = lean_usize_of_nat(v___x_807_);
v___x_819_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot_spec__0(v_modName_787_, v_imports_805_, v___x_817_, v___x_818_);
lean_dec_ref(v_imports_805_);
v___x_820_ = lean_box(v___x_819_);
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 0, v___x_820_);
v___x_822_ = v___x_803_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_820_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
v_a_825_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_800_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_800_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_830_; 
if (v_isShared_828_ == 0)
{
v___x_830_ = v___x_827_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_825_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
else
{
lean_object* v_a_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_840_; 
v_a_833_ = lean_ctor_get(v___x_797_, 0);
v_isSharedCheck_840_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_840_ == 0)
{
v___x_835_ = v___x_797_;
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_a_833_);
lean_dec(v___x_797_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_840_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
lean_object* v___x_838_; 
if (v_isShared_836_ == 0)
{
v___x_838_ = v___x_835_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_a_833_);
v___x_838_ = v_reuseFailAlloc_839_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
return v___x_838_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot___boxed(lean_object* v_modName_841_, lean_object* v_a_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot(v_modName_841_);
lean_dec(v_modName_841_);
return v_res_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_2273770963____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v___x_845_ = lean_box(0);
v___x_846_ = l_Std_Mutex_new___redArg(v___x_845_);
v___x_847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_847_, 0, v___x_846_);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_2273770963____hygCtx___hyg_2____boxed(lean_object* v_a_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_2273770963____hygCtx___hyg_2_();
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__spec__0(lean_object* v_name_850_, lean_object* v_decl_851_, lean_object* v_ref_852_){
_start:
{
lean_object* v_defValue_854_; lean_object* v_descr_855_; lean_object* v_deprecation_x3f_856_; lean_object* v___x_857_; uint8_t v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
v_defValue_854_ = lean_ctor_get(v_decl_851_, 0);
v_descr_855_ = lean_ctor_get(v_decl_851_, 1);
v_deprecation_x3f_856_ = lean_ctor_get(v_decl_851_, 2);
v___x_857_ = lean_alloc_ctor(1, 0, 1);
v___x_858_ = lean_unbox(v_defValue_854_);
lean_ctor_set_uint8(v___x_857_, 0, v___x_858_);
lean_inc(v_deprecation_x3f_856_);
lean_inc_ref(v_descr_855_);
lean_inc_n(v_name_850_, 2);
v___x_859_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_859_, 0, v_name_850_);
lean_ctor_set(v___x_859_, 1, v_ref_852_);
lean_ctor_set(v___x_859_, 2, v___x_857_);
lean_ctor_set(v___x_859_, 3, v_descr_855_);
lean_ctor_set(v___x_859_, 4, v_deprecation_x3f_856_);
v___x_860_ = lean_register_option(v_name_850_, v___x_859_);
if (lean_obj_tag(v___x_860_) == 0)
{
lean_object* v___x_862_; uint8_t v_isShared_863_; uint8_t v_isSharedCheck_868_; 
v_isSharedCheck_868_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_868_ == 0)
{
lean_object* v_unused_869_; 
v_unused_869_ = lean_ctor_get(v___x_860_, 0);
lean_dec(v_unused_869_);
v___x_862_ = v___x_860_;
v_isShared_863_ = v_isSharedCheck_868_;
goto v_resetjp_861_;
}
else
{
lean_dec(v___x_860_);
v___x_862_ = lean_box(0);
v_isShared_863_ = v_isSharedCheck_868_;
goto v_resetjp_861_;
}
v_resetjp_861_:
{
lean_object* v___x_864_; lean_object* v___x_866_; 
lean_inc(v_defValue_854_);
v___x_864_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_864_, 0, v_name_850_);
lean_ctor_set(v___x_864_, 1, v_defValue_854_);
if (v_isShared_863_ == 0)
{
lean_ctor_set(v___x_862_, 0, v___x_864_);
v___x_866_ = v___x_862_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_867_; 
v_reuseFailAlloc_867_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_867_, 0, v___x_864_);
v___x_866_ = v_reuseFailAlloc_867_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
return v___x_866_;
}
}
}
else
{
lean_object* v_a_870_; lean_object* v___x_872_; uint8_t v_isShared_873_; uint8_t v_isSharedCheck_877_; 
lean_dec(v_name_850_);
v_a_870_ = lean_ctor_get(v___x_860_, 0);
v_isSharedCheck_877_ = !lean_is_exclusive(v___x_860_);
if (v_isSharedCheck_877_ == 0)
{
v___x_872_ = v___x_860_;
v_isShared_873_ = v_isSharedCheck_877_;
goto v_resetjp_871_;
}
else
{
lean_inc(v_a_870_);
lean_dec(v___x_860_);
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
v_reuseFailAlloc_876_ = lean_alloc_ctor(1, 1, 0);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_878_, lean_object* v_decl_879_, lean_object* v_ref_880_, lean_object* v_a_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__spec__0(v_name_878_, v_decl_879_, v_ref_880_);
lean_dec_ref(v_decl_879_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_905_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_));
v___x_906_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_));
v___x_907_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__8_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_));
v___x_908_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4__spec__0(v___x_905_, v___x_906_, v___x_907_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4____boxed(lean_object* v_a_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_();
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__spec__0(lean_object* v_name_911_, lean_object* v_decl_912_, lean_object* v_ref_913_){
_start:
{
lean_object* v_defValue_915_; lean_object* v_descr_916_; lean_object* v_deprecation_x3f_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; 
v_defValue_915_ = lean_ctor_get(v_decl_912_, 0);
v_descr_916_ = lean_ctor_get(v_decl_912_, 1);
v_deprecation_x3f_917_ = lean_ctor_get(v_decl_912_, 2);
lean_inc(v_defValue_915_);
v___x_918_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_918_, 0, v_defValue_915_);
lean_inc(v_deprecation_x3f_917_);
lean_inc_ref(v_descr_916_);
lean_inc_n(v_name_911_, 2);
v___x_919_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_919_, 0, v_name_911_);
lean_ctor_set(v___x_919_, 1, v_ref_913_);
lean_ctor_set(v___x_919_, 2, v___x_918_);
lean_ctor_set(v___x_919_, 3, v_descr_916_);
lean_ctor_set(v___x_919_, 4, v_deprecation_x3f_917_);
v___x_920_ = lean_register_option(v_name_911_, v___x_919_);
if (lean_obj_tag(v___x_920_) == 0)
{
lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_928_; 
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_928_ == 0)
{
lean_object* v_unused_929_; 
v_unused_929_ = lean_ctor_get(v___x_920_, 0);
lean_dec(v_unused_929_);
v___x_922_ = v___x_920_;
v_isShared_923_ = v_isSharedCheck_928_;
goto v_resetjp_921_;
}
else
{
lean_dec(v___x_920_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_928_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_924_; lean_object* v___x_926_; 
lean_inc(v_defValue_915_);
v___x_924_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_924_, 0, v_name_911_);
lean_ctor_set(v___x_924_, 1, v_defValue_915_);
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 0, v___x_924_);
v___x_926_ = v___x_922_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v___x_924_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
else
{
lean_object* v_a_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_937_; 
lean_dec(v_name_911_);
v_a_930_ = lean_ctor_get(v___x_920_, 0);
v_isSharedCheck_937_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_937_ == 0)
{
v___x_932_ = v___x_920_;
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_a_930_);
lean_dec(v___x_920_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_935_; 
if (v_isShared_933_ == 0)
{
v___x_935_ = v___x_932_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v_a_930_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_938_, lean_object* v_decl_939_, lean_object* v_ref_940_, lean_object* v_a_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__spec__0(v_name_938_, v_decl_939_, v_ref_940_);
lean_dec_ref(v_decl_939_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; 
v___x_963_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_));
v___x_964_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_));
v___x_965_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_));
v___x_966_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4__spec__0(v___x_963_, v___x_964_, v___x_965_);
return v___x_966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4____boxed(lean_object* v_a_967_){
_start:
{
lean_object* v_res_968_; 
v_res_968_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_();
return v_res_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(lean_object* v_i_999_){
_start:
{
lean_object* v_stx_1000_; lean_object* v___x_1001_; uint8_t v___x_1002_; 
v_stx_1000_ = lean_ctor_get(v_i_999_, 1);
lean_inc_n(v_stx_1000_, 2);
lean_dec_ref(v_i_999_);
v___x_1001_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4));
v___x_1002_ = l_Lean_Syntax_isOfKind(v_stx_1000_, v___x_1001_);
if (v___x_1002_ == 0)
{
lean_object* v___x_1003_; 
lean_dec(v_stx_1000_);
v___x_1003_ = lean_box(0);
return v___x_1003_;
}
else
{
lean_object* v___x_1004_; lean_object* v___y_1016_; lean_object* v___x_1036_; uint8_t v___x_1037_; 
v___x_1004_ = lean_unsigned_to_nat(0u);
v___x_1036_ = l_Lean_Syntax_getArg(v_stx_1000_, v___x_1004_);
v___x_1037_ = l_Lean_Syntax_isNone(v___x_1036_);
if (v___x_1037_ == 0)
{
lean_object* v___x_1038_; uint8_t v___x_1039_; 
v___x_1038_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_1036_);
v___x_1039_ = l_Lean_Syntax_matchesNull(v___x_1036_, v___x_1038_);
if (v___x_1039_ == 0)
{
lean_object* v___x_1040_; 
lean_dec(v___x_1036_);
lean_dec(v_stx_1000_);
v___x_1040_ = lean_box(0);
return v___x_1040_;
}
else
{
lean_object* v___x_1041_; lean_object* v___x_1042_; uint8_t v___x_1043_; 
v___x_1041_ = l_Lean_Syntax_getArg(v___x_1036_, v___x_1004_);
lean_dec(v___x_1036_);
v___x_1042_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12));
v___x_1043_ = l_Lean_Syntax_isOfKind(v___x_1041_, v___x_1042_);
if (v___x_1043_ == 0)
{
lean_object* v___x_1044_; 
lean_dec(v_stx_1000_);
v___x_1044_ = lean_box(0);
return v___x_1044_;
}
else
{
goto v___jp_1026_;
}
}
}
else
{
lean_dec(v___x_1036_);
goto v___jp_1026_;
}
v___jp_1005_:
{
lean_object* v___x_1006_; lean_object* v_n_1007_; lean_object* v___x_1008_; uint8_t v___x_1009_; 
v___x_1006_ = lean_unsigned_to_nat(4u);
v_n_1007_ = l_Lean_Syntax_getArg(v_stx_1000_, v___x_1006_);
v___x_1008_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__6));
lean_inc(v_n_1007_);
v___x_1009_ = l_Lean_Syntax_isOfKind(v_n_1007_, v___x_1008_);
if (v___x_1009_ == 0)
{
lean_object* v___x_1010_; 
lean_dec(v_n_1007_);
lean_dec(v_stx_1000_);
v___x_1010_ = lean_box(0);
return v___x_1010_;
}
else
{
lean_object* v___x_1011_; lean_object* v___x_1012_; uint8_t v___x_1013_; 
v___x_1011_ = lean_unsigned_to_nat(5u);
v___x_1012_ = l_Lean_Syntax_getArg(v_stx_1000_, v___x_1011_);
lean_dec(v_stx_1000_);
v___x_1013_ = l_Lean_Syntax_matchesNull(v___x_1012_, v___x_1004_);
if (v___x_1013_ == 0)
{
lean_object* v___x_1014_; 
lean_dec(v_n_1007_);
v___x_1014_ = lean_box(0);
return v___x_1014_;
}
else
{
return v_n_1007_;
}
}
}
v___jp_1015_:
{
lean_object* v___x_1017_; lean_object* v___x_1018_; uint8_t v___x_1019_; 
v___x_1017_ = lean_unsigned_to_nat(3u);
v___x_1018_ = l_Lean_Syntax_getArg(v_stx_1000_, v___x_1017_);
v___x_1019_ = l_Lean_Syntax_isNone(v___x_1018_);
if (v___x_1019_ == 0)
{
uint8_t v___x_1020_; 
lean_inc(v___x_1018_);
v___x_1020_ = l_Lean_Syntax_matchesNull(v___x_1018_, v___y_1016_);
if (v___x_1020_ == 0)
{
lean_object* v___x_1021_; 
lean_dec(v___x_1018_);
lean_dec(v_stx_1000_);
v___x_1021_ = lean_box(0);
return v___x_1021_;
}
else
{
lean_object* v___x_1022_; lean_object* v___x_1023_; uint8_t v___x_1024_; 
v___x_1022_ = l_Lean_Syntax_getArg(v___x_1018_, v___x_1004_);
lean_dec(v___x_1018_);
v___x_1023_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8));
v___x_1024_ = l_Lean_Syntax_isOfKind(v___x_1022_, v___x_1023_);
if (v___x_1024_ == 0)
{
lean_object* v___x_1025_; 
lean_dec(v_stx_1000_);
v___x_1025_ = lean_box(0);
return v___x_1025_;
}
else
{
goto v___jp_1005_;
}
}
}
else
{
lean_dec(v___x_1018_);
goto v___jp_1005_;
}
}
v___jp_1026_:
{
lean_object* v___x_1027_; lean_object* v___x_1028_; uint8_t v___x_1029_; 
v___x_1027_ = lean_unsigned_to_nat(1u);
v___x_1028_ = l_Lean_Syntax_getArg(v_stx_1000_, v___x_1027_);
v___x_1029_ = l_Lean_Syntax_isNone(v___x_1028_);
if (v___x_1029_ == 0)
{
uint8_t v___x_1030_; 
lean_inc(v___x_1028_);
v___x_1030_ = l_Lean_Syntax_matchesNull(v___x_1028_, v___x_1027_);
if (v___x_1030_ == 0)
{
lean_object* v___x_1031_; 
lean_dec(v___x_1028_);
lean_dec(v_stx_1000_);
v___x_1031_ = lean_box(0);
return v___x_1031_;
}
else
{
lean_object* v___x_1032_; lean_object* v___x_1033_; uint8_t v___x_1034_; 
v___x_1032_ = l_Lean_Syntax_getArg(v___x_1028_, v___x_1004_);
lean_dec(v___x_1028_);
v___x_1033_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10));
v___x_1034_ = l_Lean_Syntax_isOfKind(v___x_1032_, v___x_1033_);
if (v___x_1034_ == 0)
{
lean_object* v___x_1035_; 
lean_dec(v_stx_1000_);
v___x_1035_ = lean_box(0);
return v___x_1035_;
}
else
{
v___y_1016_ = v___x_1027_;
goto v___jp_1015_;
}
}
}
else
{
lean_dec(v___x_1028_);
v___y_1016_ = v___x_1027_;
goto v___jp_1015_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0(lean_object* v_moduleTk_1045_, uint8_t v___x_1046_, lean_object* v_as_1047_, size_t v_i_1048_, size_t v_stop_1049_, lean_object* v_b_1050_){
_start:
{
lean_object* v___y_1052_; uint8_t v___x_1056_; 
v___x_1056_ = lean_usize_dec_eq(v_i_1048_, v_stop_1049_);
if (v___x_1056_ == 0)
{
lean_object* v___x_1057_; lean_object* v___x_1058_; uint8_t v___y_1060_; uint8_t v___y_1061_; lean_object* v___y_1062_; uint8_t v___y_1063_; uint8_t v___y_1068_; uint8_t v___y_1069_; lean_object* v___y_1070_; lean_object* v___y_1071_; uint8_t v___y_1072_; uint8_t v___y_1074_; uint8_t v___y_1075_; lean_object* v___y_1076_; lean_object* v___y_1077_; uint8_t v___y_1078_; lean_object* v___y_1080_; uint8_t v___y_1081_; lean_object* v___y_1082_; lean_object* v___y_1083_; uint8_t v___y_1084_; uint8_t v___x_1085_; 
v___x_1057_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__4));
v___x_1058_ = lean_array_uget_borrowed(v_as_1047_, v_i_1048_);
lean_inc(v___x_1058_);
v___x_1085_ = l_Lean_Syntax_isOfKind(v___x_1058_, v___x_1057_);
if (v___x_1085_ == 0)
{
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1086_; lean_object* v___y_1088_; lean_object* v___y_1089_; lean_object* v_allTk_1090_; lean_object* v___x_1099_; lean_object* v___y_1101_; lean_object* v_metaTk_1102_; lean_object* v_publicTk_1114_; lean_object* v___x_1124_; uint8_t v___x_1125_; 
v___x_1086_ = lean_unsigned_to_nat(0u);
v___x_1099_ = lean_unsigned_to_nat(1u);
v___x_1124_ = l_Lean_Syntax_getArg(v___x_1058_, v___x_1086_);
v___x_1125_ = l_Lean_Syntax_isNone(v___x_1124_);
if (v___x_1125_ == 0)
{
uint8_t v___x_1126_; 
lean_inc(v___x_1124_);
v___x_1126_ = l_Lean_Syntax_matchesNull(v___x_1124_, v___x_1099_);
if (v___x_1126_ == 0)
{
lean_dec(v___x_1124_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1127_; lean_object* v___x_1128_; uint8_t v___x_1129_; 
v___x_1127_ = l_Lean_Syntax_getArg(v___x_1124_, v___x_1086_);
lean_dec(v___x_1124_);
v___x_1128_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__12));
lean_inc(v___x_1127_);
v___x_1129_ = l_Lean_Syntax_isOfKind(v___x_1127_, v___x_1128_);
if (v___x_1129_ == 0)
{
lean_dec(v___x_1127_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v_publicTk_1130_; lean_object* v___x_1131_; 
v_publicTk_1130_ = l_Lean_Syntax_getArg(v___x_1127_, v___x_1086_);
lean_dec(v___x_1127_);
v___x_1131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1131_, 0, v_publicTk_1130_);
v_publicTk_1114_ = v___x_1131_;
goto v___jp_1113_;
}
}
}
else
{
lean_object* v___x_1132_; 
lean_dec(v___x_1124_);
v___x_1132_ = lean_box(0);
v_publicTk_1114_ = v___x_1132_;
goto v___jp_1113_;
}
v___jp_1087_:
{
lean_object* v___x_1091_; lean_object* v_n_1092_; lean_object* v___x_1093_; uint8_t v___x_1094_; 
v___x_1091_ = lean_unsigned_to_nat(4u);
v_n_1092_ = l_Lean_Syntax_getArg(v___x_1058_, v___x_1091_);
v___x_1093_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__6));
lean_inc(v_n_1092_);
v___x_1094_ = l_Lean_Syntax_isOfKind(v_n_1092_, v___x_1093_);
if (v___x_1094_ == 0)
{
lean_dec(v_n_1092_);
lean_dec(v_allTk_1090_);
lean_dec(v___y_1089_);
lean_dec(v___y_1088_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1095_; lean_object* v___x_1096_; uint8_t v___x_1097_; 
v___x_1095_ = lean_unsigned_to_nat(5u);
v___x_1096_ = l_Lean_Syntax_getArg(v___x_1058_, v___x_1095_);
v___x_1097_ = l_Lean_Syntax_matchesNull(v___x_1096_, v___x_1086_);
if (v___x_1097_ == 0)
{
lean_dec(v_n_1092_);
lean_dec(v_allTk_1090_);
lean_dec(v___y_1089_);
lean_dec(v___y_1088_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1098_; 
v___x_1098_ = l_Lean_TSyntax_getId(v_n_1092_);
lean_dec(v_n_1092_);
if (lean_obj_tag(v_allTk_1090_) == 0)
{
v___y_1080_ = v___y_1088_;
v___y_1081_ = v___x_1094_;
v___y_1082_ = v___x_1098_;
v___y_1083_ = v___y_1089_;
v___y_1084_ = v___x_1056_;
goto v___jp_1079_;
}
else
{
lean_dec_ref_known(v_allTk_1090_, 1);
v___y_1080_ = v___y_1088_;
v___y_1081_ = v___x_1094_;
v___y_1082_ = v___x_1098_;
v___y_1083_ = v___y_1089_;
v___y_1084_ = v___x_1094_;
goto v___jp_1079_;
}
}
}
}
v___jp_1100_:
{
lean_object* v___x_1103_; lean_object* v___x_1104_; uint8_t v___x_1105_; 
v___x_1103_ = lean_unsigned_to_nat(3u);
v___x_1104_ = l_Lean_Syntax_getArg(v___x_1058_, v___x_1103_);
v___x_1105_ = l_Lean_Syntax_isNone(v___x_1104_);
if (v___x_1105_ == 0)
{
uint8_t v___x_1106_; 
lean_inc(v___x_1104_);
v___x_1106_ = l_Lean_Syntax_matchesNull(v___x_1104_, v___x_1099_);
if (v___x_1106_ == 0)
{
lean_dec(v___x_1104_);
lean_dec(v_metaTk_1102_);
lean_dec(v___y_1101_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1107_; lean_object* v___x_1108_; uint8_t v___x_1109_; 
v___x_1107_ = l_Lean_Syntax_getArg(v___x_1104_, v___x_1086_);
lean_dec(v___x_1104_);
v___x_1108_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__8));
lean_inc(v___x_1107_);
v___x_1109_ = l_Lean_Syntax_isOfKind(v___x_1107_, v___x_1108_);
if (v___x_1109_ == 0)
{
lean_dec(v___x_1107_);
lean_dec(v_metaTk_1102_);
lean_dec(v___y_1101_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v_allTk_1110_; lean_object* v___x_1111_; 
v_allTk_1110_ = l_Lean_Syntax_getArg(v___x_1107_, v___x_1086_);
lean_dec(v___x_1107_);
v___x_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1111_, 0, v_allTk_1110_);
v___y_1088_ = v___y_1101_;
v___y_1089_ = v_metaTk_1102_;
v_allTk_1090_ = v___x_1111_;
goto v___jp_1087_;
}
}
}
else
{
lean_object* v___x_1112_; 
lean_dec(v___x_1104_);
v___x_1112_ = lean_box(0);
v___y_1088_ = v___y_1101_;
v___y_1089_ = v_metaTk_1102_;
v_allTk_1090_ = v___x_1112_;
goto v___jp_1087_;
}
}
v___jp_1113_:
{
lean_object* v___x_1115_; uint8_t v___x_1116_; 
v___x_1115_ = l_Lean_Syntax_getArg(v___x_1058_, v___x_1099_);
v___x_1116_ = l_Lean_Syntax_isNone(v___x_1115_);
if (v___x_1116_ == 0)
{
uint8_t v___x_1117_; 
lean_inc(v___x_1115_);
v___x_1117_ = l_Lean_Syntax_matchesNull(v___x_1115_, v___x_1099_);
if (v___x_1117_ == 0)
{
lean_dec(v___x_1115_);
lean_dec(v_publicTk_1114_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v___x_1118_; lean_object* v___x_1119_; uint8_t v___x_1120_; 
v___x_1118_ = l_Lean_Syntax_getArg(v___x_1115_, v___x_1086_);
lean_dec(v___x_1115_);
v___x_1119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__10));
lean_inc(v___x_1118_);
v___x_1120_ = l_Lean_Syntax_isOfKind(v___x_1118_, v___x_1119_);
if (v___x_1120_ == 0)
{
lean_dec(v___x_1118_);
lean_dec(v_publicTk_1114_);
v___y_1052_ = v_b_1050_;
goto v___jp_1051_;
}
else
{
lean_object* v_metaTk_1121_; lean_object* v___x_1122_; 
v_metaTk_1121_ = l_Lean_Syntax_getArg(v___x_1118_, v___x_1086_);
lean_dec(v___x_1118_);
v___x_1122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1122_, 0, v_metaTk_1121_);
v___y_1101_ = v_publicTk_1114_;
v_metaTk_1102_ = v___x_1122_;
goto v___jp_1100_;
}
}
}
else
{
lean_object* v___x_1123_; 
lean_dec(v___x_1115_);
v___x_1123_ = lean_box(0);
v___y_1101_ = v_publicTk_1114_;
v_metaTk_1102_ = v___x_1123_;
goto v___jp_1100_;
}
}
}
v___jp_1059_:
{
lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; 
v___x_1064_ = lean_alloc_ctor(0, 1, 3);
lean_ctor_set(v___x_1064_, 0, v___y_1062_);
lean_ctor_set_uint8(v___x_1064_, sizeof(void*)*1, v___y_1060_);
lean_ctor_set_uint8(v___x_1064_, sizeof(void*)*1 + 1, v___y_1061_);
lean_ctor_set_uint8(v___x_1064_, sizeof(void*)*1 + 2, v___y_1063_);
lean_inc(v___x_1058_);
v___x_1065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1064_);
lean_ctor_set(v___x_1065_, 1, v___x_1058_);
v___x_1066_ = lean_array_push(v_b_1050_, v___x_1065_);
v___y_1052_ = v___x_1066_;
goto v___jp_1051_;
}
v___jp_1067_:
{
if (lean_obj_tag(v___y_1071_) == 0)
{
v___y_1060_ = v___y_1068_;
v___y_1061_ = v___y_1072_;
v___y_1062_ = v___y_1070_;
v___y_1063_ = v___x_1056_;
goto v___jp_1059_;
}
else
{
lean_dec_ref_known(v___y_1071_, 1);
v___y_1060_ = v___y_1068_;
v___y_1061_ = v___y_1072_;
v___y_1062_ = v___y_1070_;
v___y_1063_ = v___y_1069_;
goto v___jp_1059_;
}
}
v___jp_1073_:
{
if (lean_obj_tag(v_moduleTk_1045_) == 0)
{
v___y_1068_ = v___y_1074_;
v___y_1069_ = v___y_1075_;
v___y_1070_ = v___y_1076_;
v___y_1071_ = v___y_1077_;
v___y_1072_ = v___y_1075_;
goto v___jp_1067_;
}
else
{
v___y_1068_ = v___y_1074_;
v___y_1069_ = v___y_1075_;
v___y_1070_ = v___y_1076_;
v___y_1071_ = v___y_1077_;
v___y_1072_ = v___y_1078_;
goto v___jp_1067_;
}
}
v___jp_1079_:
{
if (lean_obj_tag(v___y_1080_) == 0)
{
v___y_1074_ = v___y_1084_;
v___y_1075_ = v___y_1081_;
v___y_1076_ = v___y_1082_;
v___y_1077_ = v___y_1083_;
v___y_1078_ = v___x_1056_;
goto v___jp_1073_;
}
else
{
lean_dec_ref_known(v___y_1080_, 1);
if (v___y_1081_ == 0)
{
v___y_1074_ = v___y_1084_;
v___y_1075_ = v___y_1081_;
v___y_1076_ = v___y_1082_;
v___y_1077_ = v___y_1083_;
v___y_1078_ = v___y_1081_;
goto v___jp_1073_;
}
else
{
v___y_1068_ = v___y_1084_;
v___y_1069_ = v___y_1081_;
v___y_1070_ = v___y_1082_;
v___y_1071_ = v___y_1083_;
v___y_1072_ = v___x_1046_;
goto v___jp_1067_;
}
}
}
}
else
{
return v_b_1050_;
}
v___jp_1051_:
{
size_t v___x_1053_; size_t v___x_1054_; 
v___x_1053_ = ((size_t)1ULL);
v___x_1054_ = lean_usize_add(v_i_1048_, v___x_1053_);
v_i_1048_ = v___x_1054_;
v_b_1050_ = v___y_1052_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0___boxed(lean_object* v_moduleTk_1133_, lean_object* v___x_1134_, lean_object* v_as_1135_, lean_object* v_i_1136_, lean_object* v_stop_1137_, lean_object* v_b_1138_){
_start:
{
uint8_t v___x_3073__boxed_1139_; size_t v_i_boxed_1140_; size_t v_stop_boxed_1141_; lean_object* v_res_1142_; 
v___x_3073__boxed_1139_ = lean_unbox(v___x_1134_);
v_i_boxed_1140_ = lean_unbox_usize(v_i_1136_);
lean_dec(v_i_1136_);
v_stop_boxed_1141_ = lean_unbox_usize(v_stop_1137_);
lean_dec(v_stop_1137_);
v_res_1142_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0(v_moduleTk_1133_, v___x_3073__boxed_1139_, v_as_1135_, v_i_boxed_1140_, v_stop_boxed_1141_, v_b_1138_);
lean_dec_ref(v_as_1135_);
lean_dec(v_moduleTk_1133_);
return v_res_1142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0(lean_object* v_moduleTk_1145_, uint8_t v___x_1146_, lean_object* v_as_1147_, lean_object* v_start_1148_, lean_object* v_stop_1149_){
_start:
{
lean_object* v___x_1150_; uint8_t v___x_1151_; 
v___x_1150_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0));
v___x_1151_ = lean_nat_dec_lt(v_start_1148_, v_stop_1149_);
if (v___x_1151_ == 0)
{
return v___x_1150_;
}
else
{
lean_object* v___x_1152_; uint8_t v___x_1153_; 
v___x_1152_ = lean_array_get_size(v_as_1147_);
v___x_1153_ = lean_nat_dec_le(v_stop_1149_, v___x_1152_);
if (v___x_1153_ == 0)
{
uint8_t v___x_1154_; 
v___x_1154_ = lean_nat_dec_lt(v_start_1148_, v___x_1152_);
if (v___x_1154_ == 0)
{
return v___x_1150_;
}
else
{
size_t v___x_1155_; size_t v___x_1156_; lean_object* v___x_1157_; 
v___x_1155_ = lean_usize_of_nat(v_start_1148_);
v___x_1156_ = lean_usize_of_nat(v___x_1152_);
v___x_1157_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0(v_moduleTk_1145_, v___x_1146_, v_as_1147_, v___x_1155_, v___x_1156_, v___x_1150_);
return v___x_1157_;
}
}
else
{
size_t v___x_1158_; size_t v___x_1159_; lean_object* v___x_1160_; 
v___x_1158_ = lean_usize_of_nat(v_start_1148_);
v___x_1159_ = lean_usize_of_nat(v_stop_1149_);
v___x_1160_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0_spec__0(v_moduleTk_1145_, v___x_1146_, v_as_1147_, v___x_1158_, v___x_1159_, v___x_1150_);
return v___x_1160_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___boxed(lean_object* v_moduleTk_1161_, lean_object* v___x_1162_, lean_object* v_as_1163_, lean_object* v_start_1164_, lean_object* v_stop_1165_){
_start:
{
uint8_t v___x_3241__boxed_1166_; lean_object* v_res_1167_; 
v___x_3241__boxed_1166_ = lean_unbox(v___x_1162_);
v_res_1167_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0(v_moduleTk_1161_, v___x_3241__boxed_1166_, v_as_1163_, v_start_1164_, v_stop_1165_);
lean_dec(v_stop_1165_);
lean_dec(v_start_1164_);
lean_dec_ref(v_as_1163_);
lean_dec(v_moduleTk_1161_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs(lean_object* v_header_1185_){
_start:
{
lean_object* v___x_1186_; uint8_t v___x_1187_; 
v___x_1186_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0));
lean_inc(v_header_1185_);
v___x_1187_ = l_Lean_Syntax_isOfKind(v_header_1185_, v___x_1186_);
if (v___x_1187_ == 0)
{
lean_object* v___x_1188_; 
lean_dec(v_header_1185_);
v___x_1188_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0));
return v___x_1188_;
}
else
{
lean_object* v___x_1189_; lean_object* v___y_1191_; lean_object* v_moduleTk_1198_; lean_object* v___x_1208_; uint8_t v___x_1209_; 
v___x_1189_ = lean_unsigned_to_nat(0u);
v___x_1208_ = l_Lean_Syntax_getArg(v_header_1185_, v___x_1189_);
v___x_1209_ = l_Lean_Syntax_isNone(v___x_1208_);
if (v___x_1209_ == 0)
{
lean_object* v___x_1210_; uint8_t v___x_1211_; 
v___x_1210_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_1208_);
v___x_1211_ = l_Lean_Syntax_matchesNull(v___x_1208_, v___x_1210_);
if (v___x_1211_ == 0)
{
lean_object* v___x_1212_; 
lean_dec(v___x_1208_);
lean_dec(v_header_1185_);
v___x_1212_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0));
return v___x_1212_;
}
else
{
lean_object* v___x_1213_; lean_object* v___x_1214_; uint8_t v___x_1215_; 
v___x_1213_ = l_Lean_Syntax_getArg(v___x_1208_, v___x_1189_);
lean_dec(v___x_1208_);
v___x_1214_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4));
lean_inc(v___x_1213_);
v___x_1215_ = l_Lean_Syntax_isOfKind(v___x_1213_, v___x_1214_);
if (v___x_1215_ == 0)
{
lean_object* v___x_1216_; 
lean_dec(v___x_1213_);
lean_dec(v_header_1185_);
v___x_1216_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0));
return v___x_1216_;
}
else
{
lean_object* v_moduleTk_1217_; lean_object* v___x_1218_; 
v_moduleTk_1217_ = l_Lean_Syntax_getArg(v___x_1213_, v___x_1189_);
lean_dec(v___x_1213_);
v___x_1218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1218_, 0, v_moduleTk_1217_);
v_moduleTk_1198_ = v___x_1218_;
goto v___jp_1197_;
}
}
}
else
{
lean_object* v___x_1219_; 
lean_dec(v___x_1208_);
v___x_1219_ = lean_box(0);
v_moduleTk_1198_ = v___x_1219_;
goto v___jp_1197_;
}
v___jp_1190_:
{
lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v_imports_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; 
v___x_1192_ = lean_unsigned_to_nat(2u);
v___x_1193_ = l_Lean_Syntax_getArg(v_header_1185_, v___x_1192_);
lean_dec(v_header_1185_);
v_imports_1194_ = l_Lean_Syntax_getArgs(v___x_1193_);
lean_dec(v___x_1193_);
v___x_1195_ = lean_array_get_size(v_imports_1194_);
v___x_1196_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0(v___y_1191_, v___x_1187_, v_imports_1194_, v___x_1189_, v___x_1195_);
lean_dec_ref(v_imports_1194_);
lean_dec(v___y_1191_);
return v___x_1196_;
}
v___jp_1197_:
{
lean_object* v___x_1199_; lean_object* v___x_1200_; uint8_t v___x_1201_; 
v___x_1199_ = lean_unsigned_to_nat(1u);
v___x_1200_ = l_Lean_Syntax_getArg(v_header_1185_, v___x_1199_);
v___x_1201_ = l_Lean_Syntax_isNone(v___x_1200_);
if (v___x_1201_ == 0)
{
uint8_t v___x_1202_; 
lean_inc(v___x_1200_);
v___x_1202_ = l_Lean_Syntax_matchesNull(v___x_1200_, v___x_1199_);
if (v___x_1202_ == 0)
{
lean_object* v___x_1203_; 
lean_dec(v___x_1200_);
lean_dec(v_moduleTk_1198_);
lean_dec(v_header_1185_);
v___x_1203_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0));
return v___x_1203_;
}
else
{
lean_object* v___x_1204_; lean_object* v___x_1205_; uint8_t v___x_1206_; 
v___x_1204_ = l_Lean_Syntax_getArg(v___x_1200_, v___x_1189_);
lean_dec(v___x_1200_);
v___x_1205_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2));
v___x_1206_ = l_Lean_Syntax_isOfKind(v___x_1204_, v___x_1205_);
if (v___x_1206_ == 0)
{
lean_object* v___x_1207_; 
lean_dec(v_moduleTk_1198_);
lean_dec(v_header_1185_);
v___x_1207_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs_spec__0___closed__0));
return v___x_1207_;
}
else
{
v___y_1191_ = v_moduleTk_1198_;
goto v___jp_1190_;
}
}
}
else
{
lean_dec(v___x_1200_);
v___y_1191_ = v_moduleTk_1198_;
goto v___jp_1190_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToModuleTk_x3f(lean_object* v_header_1220_){
_start:
{
lean_object* v___x_1221_; uint8_t v___x_1222_; 
v___x_1221_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0));
lean_inc(v_header_1220_);
v___x_1222_ = l_Lean_Syntax_isOfKind(v_header_1220_, v___x_1221_);
if (v___x_1222_ == 0)
{
lean_object* v___x_1223_; 
lean_dec(v_header_1220_);
v___x_1223_ = lean_box(0);
return v___x_1223_;
}
else
{
lean_object* v___x_1224_; lean_object* v_tk_1226_; lean_object* v___x_1236_; uint8_t v___x_1237_; 
v___x_1224_ = lean_unsigned_to_nat(0u);
v___x_1236_ = l_Lean_Syntax_getArg(v_header_1220_, v___x_1224_);
v___x_1237_ = l_Lean_Syntax_isNone(v___x_1236_);
if (v___x_1237_ == 0)
{
lean_object* v___x_1238_; uint8_t v___x_1239_; 
v___x_1238_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_1236_);
v___x_1239_ = l_Lean_Syntax_matchesNull(v___x_1236_, v___x_1238_);
if (v___x_1239_ == 0)
{
lean_object* v___x_1240_; 
lean_dec(v___x_1236_);
lean_dec(v_header_1220_);
v___x_1240_ = lean_box(0);
return v___x_1240_;
}
else
{
lean_object* v___x_1241_; lean_object* v___x_1242_; uint8_t v___x_1243_; 
v___x_1241_ = l_Lean_Syntax_getArg(v___x_1236_, v___x_1224_);
lean_dec(v___x_1236_);
v___x_1242_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4));
lean_inc(v___x_1241_);
v___x_1243_ = l_Lean_Syntax_isOfKind(v___x_1241_, v___x_1242_);
if (v___x_1243_ == 0)
{
lean_object* v___x_1244_; 
lean_dec(v___x_1241_);
lean_dec(v_header_1220_);
v___x_1244_ = lean_box(0);
return v___x_1244_;
}
else
{
lean_object* v_tk_1245_; lean_object* v___x_1246_; 
v_tk_1245_ = l_Lean_Syntax_getArg(v___x_1241_, v___x_1224_);
lean_dec(v___x_1241_);
v___x_1246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1246_, 0, v_tk_1245_);
v_tk_1226_ = v___x_1246_;
goto v___jp_1225_;
}
}
}
else
{
lean_object* v___x_1247_; 
lean_dec(v___x_1236_);
v___x_1247_ = lean_box(0);
v_tk_1226_ = v___x_1247_;
goto v___jp_1225_;
}
v___jp_1225_:
{
lean_object* v___x_1227_; lean_object* v___x_1228_; uint8_t v___x_1229_; 
v___x_1227_ = lean_unsigned_to_nat(1u);
v___x_1228_ = l_Lean_Syntax_getArg(v_header_1220_, v___x_1227_);
lean_dec(v_header_1220_);
v___x_1229_ = l_Lean_Syntax_isNone(v___x_1228_);
if (v___x_1229_ == 0)
{
uint8_t v___x_1230_; 
lean_inc(v___x_1228_);
v___x_1230_ = l_Lean_Syntax_matchesNull(v___x_1228_, v___x_1227_);
if (v___x_1230_ == 0)
{
lean_object* v___x_1231_; 
lean_dec(v___x_1228_);
lean_dec(v_tk_1226_);
v___x_1231_ = lean_box(0);
return v___x_1231_;
}
else
{
lean_object* v___x_1232_; lean_object* v___x_1233_; uint8_t v___x_1234_; 
v___x_1232_ = l_Lean_Syntax_getArg(v___x_1228_, v___x_1224_);
lean_dec(v___x_1228_);
v___x_1233_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2));
v___x_1234_ = l_Lean_Syntax_isOfKind(v___x_1232_, v___x_1233_);
if (v___x_1234_ == 0)
{
lean_object* v___x_1235_; 
lean_dec(v_tk_1226_);
v___x_1235_ = lean_box(0);
return v___x_1235_;
}
else
{
return v_tk_1226_;
}
}
}
else
{
lean_dec(v___x_1228_);
return v_tk_1226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToPreludeTk_x3f(lean_object* v_header_1248_){
_start:
{
lean_object* v___x_1249_; uint8_t v___x_1250_; 
v___x_1249_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__0));
lean_inc(v_header_1248_);
v___x_1250_ = l_Lean_Syntax_isOfKind(v_header_1248_, v___x_1249_);
if (v___x_1250_ == 0)
{
lean_object* v___x_1251_; 
lean_dec(v_header_1248_);
v___x_1251_ = lean_box(0);
return v___x_1251_;
}
else
{
lean_object* v___x_1252_; lean_object* v___x_1266_; uint8_t v___x_1267_; 
v___x_1252_ = lean_unsigned_to_nat(0u);
v___x_1266_ = l_Lean_Syntax_getArg(v_header_1248_, v___x_1252_);
v___x_1267_ = l_Lean_Syntax_isNone(v___x_1266_);
if (v___x_1267_ == 0)
{
lean_object* v___x_1268_; uint8_t v___x_1269_; 
v___x_1268_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_1266_);
v___x_1269_ = l_Lean_Syntax_matchesNull(v___x_1266_, v___x_1268_);
if (v___x_1269_ == 0)
{
lean_object* v___x_1270_; 
lean_dec(v___x_1266_);
lean_dec(v_header_1248_);
v___x_1270_ = lean_box(0);
return v___x_1270_;
}
else
{
lean_object* v___x_1271_; lean_object* v___x_1272_; uint8_t v___x_1273_; 
v___x_1271_ = l_Lean_Syntax_getArg(v___x_1266_, v___x_1252_);
lean_dec(v___x_1266_);
v___x_1272_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__4));
v___x_1273_ = l_Lean_Syntax_isOfKind(v___x_1271_, v___x_1272_);
if (v___x_1273_ == 0)
{
lean_object* v___x_1274_; 
lean_dec(v_header_1248_);
v___x_1274_ = lean_box(0);
return v___x_1274_;
}
else
{
goto v___jp_1253_;
}
}
}
else
{
lean_dec(v___x_1266_);
goto v___jp_1253_;
}
v___jp_1253_:
{
lean_object* v___x_1254_; lean_object* v___x_1255_; uint8_t v___x_1256_; 
v___x_1254_ = lean_unsigned_to_nat(1u);
v___x_1255_ = l_Lean_Syntax_getArg(v_header_1248_, v___x_1254_);
lean_dec(v_header_1248_);
v___x_1256_ = l_Lean_Syntax_isNone(v___x_1255_);
if (v___x_1256_ == 0)
{
uint8_t v___x_1257_; 
lean_inc(v___x_1255_);
v___x_1257_ = l_Lean_Syntax_matchesNull(v___x_1255_, v___x_1254_);
if (v___x_1257_ == 0)
{
lean_object* v___x_1258_; 
lean_dec(v___x_1255_);
v___x_1258_ = lean_box(0);
return v___x_1258_;
}
else
{
lean_object* v___x_1259_; lean_object* v___x_1260_; uint8_t v___x_1261_; 
v___x_1259_ = l_Lean_Syntax_getArg(v___x_1255_, v___x_1252_);
lean_dec(v___x_1255_);
v___x_1260_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs___closed__2));
lean_inc(v___x_1259_);
v___x_1261_ = l_Lean_Syntax_isOfKind(v___x_1259_, v___x_1260_);
if (v___x_1261_ == 0)
{
lean_object* v___x_1262_; 
lean_dec(v___x_1259_);
v___x_1262_ = lean_box(0);
return v___x_1262_;
}
else
{
lean_object* v_tk_1263_; lean_object* v___x_1264_; 
v_tk_1263_ = l_Lean_Syntax_getArg(v___x_1259_, v___x_1252_);
lean_dec(v___x_1259_);
v___x_1264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1264_, 0, v_tk_1263_);
return v___x_1264_;
}
}
}
else
{
lean_object* v___x_1265_; 
lean_dec(v___x_1255_);
v___x_1265_ = lean_box(0);
return v___x_1265_;
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__15(void){
_start:
{
lean_object* v___x_1313_; lean_object* v___x_1314_; 
v___x_1313_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__14));
v___x_1314_ = l_Lean_NameSet_ofList(v___x_1313_);
return v___x_1314_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles(void){
_start:
{
lean_object* v___x_1315_; 
v___x_1315_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__15, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles___closed__15);
return v___x_1315_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_1316_; 
v___x_1316_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1316_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_1317_; lean_object* v___x_1318_; 
v___x_1317_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__0);
v___x_1318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1317_);
return v___x_1318_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; 
v___x_1319_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1);
v___x_1320_ = lean_unsigned_to_nat(0u);
v___x_1321_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1320_);
lean_ctor_set(v___x_1321_, 1, v___x_1320_);
lean_ctor_set(v___x_1321_, 2, v___x_1320_);
lean_ctor_set(v___x_1321_, 3, v___x_1320_);
lean_ctor_set(v___x_1321_, 4, v___x_1319_);
lean_ctor_set(v___x_1321_, 5, v___x_1319_);
lean_ctor_set(v___x_1321_, 6, v___x_1319_);
lean_ctor_set(v___x_1321_, 7, v___x_1319_);
lean_ctor_set(v___x_1321_, 8, v___x_1319_);
lean_ctor_set(v___x_1321_, 9, v___x_1319_);
return v___x_1321_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__3(void){
_start:
{
lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; 
v___x_1322_ = lean_unsigned_to_nat(32u);
v___x_1323_ = lean_mk_empty_array_with_capacity(v___x_1322_);
v___x_1324_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1323_);
return v___x_1324_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__4(void){
_start:
{
size_t v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; 
v___x_1325_ = ((size_t)5ULL);
v___x_1326_ = lean_unsigned_to_nat(0u);
v___x_1327_ = lean_unsigned_to_nat(32u);
v___x_1328_ = lean_mk_empty_array_with_capacity(v___x_1327_);
v___x_1329_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__3);
v___x_1330_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1330_, 0, v___x_1329_);
lean_ctor_set(v___x_1330_, 1, v___x_1328_);
lean_ctor_set(v___x_1330_, 2, v___x_1326_);
lean_ctor_set(v___x_1330_, 3, v___x_1326_);
lean_ctor_set_usize(v___x_1330_, 4, v___x_1325_);
return v___x_1330_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__5(void){
_start:
{
lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; 
v___x_1331_ = lean_box(1);
v___x_1332_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__4);
v___x_1333_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__1);
v___x_1334_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1334_, 0, v___x_1333_);
lean_ctor_set(v___x_1334_, 1, v___x_1332_);
lean_ctor_set(v___x_1334_, 2, v___x_1331_);
return v___x_1334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg(lean_object* v_msgData_1335_, lean_object* v___y_1336_){
_start:
{
lean_object* v___x_1338_; lean_object* v_env_1339_; lean_object* v___x_1340_; lean_object* v_scopes_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v_opts_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1338_ = lean_st_ref_get(v___y_1336_);
v_env_1339_ = lean_ctor_get(v___x_1338_, 0);
lean_inc_ref(v_env_1339_);
lean_dec(v___x_1338_);
v___x_1340_ = lean_st_ref_get(v___y_1336_);
v_scopes_1341_ = lean_ctor_get(v___x_1340_, 2);
lean_inc(v_scopes_1341_);
lean_dec(v___x_1340_);
v___x_1342_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1343_ = l_List_head_x21___redArg(v___x_1342_, v_scopes_1341_);
lean_dec(v_scopes_1341_);
v_opts_1344_ = lean_ctor_get(v___x_1343_, 1);
lean_inc_ref(v_opts_1344_);
lean_dec(v___x_1343_);
v___x_1345_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__2);
v___x_1346_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___closed__5);
v___x_1347_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1347_, 0, v_env_1339_);
lean_ctor_set(v___x_1347_, 1, v___x_1345_);
lean_ctor_set(v___x_1347_, 2, v___x_1346_);
lean_ctor_set(v___x_1347_, 3, v_opts_1344_);
v___x_1348_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1348_, 0, v___x_1347_);
lean_ctor_set(v___x_1348_, 1, v_msgData_1335_);
v___x_1349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1349_, 0, v___x_1348_);
return v___x_1349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_msgData_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_){
_start:
{
lean_object* v_res_1353_; 
v_res_1353_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg(v_msgData_1350_, v___y_1351_);
lean_dec(v___y_1351_);
return v_res_1353_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0(uint8_t v___y_1355_, uint8_t v_suppressElabErrors_1356_, lean_object* v_x_1357_){
_start:
{
if (lean_obj_tag(v_x_1357_) == 1)
{
lean_object* v_pre_1358_; 
v_pre_1358_ = lean_ctor_get(v_x_1357_, 0);
if (lean_obj_tag(v_pre_1358_) == 0)
{
lean_object* v_str_1359_; lean_object* v___x_1360_; uint8_t v___x_1361_; 
v_str_1359_ = lean_ctor_get(v_x_1357_, 1);
v___x_1360_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___closed__0));
v___x_1361_ = lean_string_dec_eq(v_str_1359_, v___x_1360_);
if (v___x_1361_ == 0)
{
return v___y_1355_;
}
else
{
return v_suppressElabErrors_1356_;
}
}
else
{
return v___y_1355_;
}
}
else
{
return v___y_1355_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___boxed(lean_object* v___y_1362_, lean_object* v_suppressElabErrors_1363_, lean_object* v_x_1364_){
_start:
{
uint8_t v___y_6068__boxed_1365_; uint8_t v_suppressElabErrors_boxed_1366_; uint8_t v_res_1367_; lean_object* v_r_1368_; 
v___y_6068__boxed_1365_ = lean_unbox(v___y_1362_);
v_suppressElabErrors_boxed_1366_ = lean_unbox(v_suppressElabErrors_1363_);
v_res_1367_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0(v___y_6068__boxed_1365_, v_suppressElabErrors_boxed_1366_, v_x_1364_);
lean_dec(v_x_1364_);
v_r_1368_ = lean_box(v_res_1367_);
return v_r_1368_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__5(lean_object* v_opts_1369_, lean_object* v_opt_1370_){
_start:
{
lean_object* v_name_1371_; lean_object* v_defValue_1372_; lean_object* v_map_1373_; lean_object* v___x_1374_; 
v_name_1371_ = lean_ctor_get(v_opt_1370_, 0);
v_defValue_1372_ = lean_ctor_get(v_opt_1370_, 1);
v_map_1373_ = lean_ctor_get(v_opts_1369_, 0);
v___x_1374_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1373_, v_name_1371_);
if (lean_obj_tag(v___x_1374_) == 0)
{
uint8_t v___x_1375_; 
v___x_1375_ = lean_unbox(v_defValue_1372_);
return v___x_1375_;
}
else
{
lean_object* v_val_1376_; 
v_val_1376_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_val_1376_);
lean_dec_ref_known(v___x_1374_, 1);
if (lean_obj_tag(v_val_1376_) == 1)
{
uint8_t v_v_1377_; 
v_v_1377_ = lean_ctor_get_uint8(v_val_1376_, 0);
lean_dec_ref_known(v_val_1376_, 0);
return v_v_1377_;
}
else
{
uint8_t v___x_1378_; 
lean_dec(v_val_1376_);
v___x_1378_ = lean_unbox(v_defValue_1372_);
return v___x_1378_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__5___boxed(lean_object* v_opts_1379_, lean_object* v_opt_1380_){
_start:
{
uint8_t v_res_1381_; lean_object* v_r_1382_; 
v_res_1381_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__5(v_opts_1379_, v_opt_1380_);
lean_dec_ref(v_opt_1380_);
lean_dec_ref(v_opts_1379_);
v_r_1382_ = lean_box(v_res_1381_);
return v_r_1382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1(lean_object* v_ref_1383_, lean_object* v_msgData_1384_, uint8_t v_severity_1385_, uint8_t v_isSilent_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_){
_start:
{
lean_object* v___y_1391_; lean_object* v___y_1392_; uint8_t v___y_1393_; lean_object* v___y_1394_; uint8_t v___y_1395_; lean_object* v___y_1396_; lean_object* v___y_1397_; lean_object* v___y_1398_; uint8_t v___y_1455_; uint8_t v___y_1456_; lean_object* v___y_1457_; uint8_t v___y_1458_; lean_object* v___y_1459_; uint8_t v___y_1483_; lean_object* v___y_1484_; uint8_t v___y_1485_; uint8_t v___y_1486_; lean_object* v___y_1487_; uint8_t v___y_1491_; uint8_t v___y_1492_; uint8_t v___y_1493_; uint8_t v___x_1508_; uint8_t v___y_1510_; uint8_t v___y_1511_; uint8_t v___y_1512_; uint8_t v___y_1514_; uint8_t v___x_1526_; 
v___x_1508_ = 2;
v___x_1526_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1385_, v___x_1508_);
if (v___x_1526_ == 0)
{
v___y_1514_ = v___x_1526_;
goto v___jp_1513_;
}
else
{
uint8_t v___x_1527_; 
lean_inc_ref(v_msgData_1384_);
v___x_1527_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1384_);
v___y_1514_ = v___x_1527_;
goto v___jp_1513_;
}
v___jp_1390_:
{
lean_object* v___x_1399_; 
v___x_1399_ = l_Lean_Elab_Command_getScope___redArg(v___y_1398_);
if (lean_obj_tag(v___x_1399_) == 0)
{
lean_object* v_a_1400_; lean_object* v___x_1401_; 
v_a_1400_ = lean_ctor_get(v___x_1399_, 0);
lean_inc(v_a_1400_);
lean_dec_ref_known(v___x_1399_, 1);
v___x_1401_ = l_Lean_Elab_Command_getScope___redArg(v___y_1398_);
if (lean_obj_tag(v___x_1401_) == 0)
{
lean_object* v_a_1402_; lean_object* v___x_1404_; uint8_t v_isShared_1405_; uint8_t v_isSharedCheck_1437_; 
v_a_1402_ = lean_ctor_get(v___x_1401_, 0);
v_isSharedCheck_1437_ = !lean_is_exclusive(v___x_1401_);
if (v_isSharedCheck_1437_ == 0)
{
v___x_1404_ = v___x_1401_;
v_isShared_1405_ = v_isSharedCheck_1437_;
goto v_resetjp_1403_;
}
else
{
lean_inc(v_a_1402_);
lean_dec(v___x_1401_);
v___x_1404_ = lean_box(0);
v_isShared_1405_ = v_isSharedCheck_1437_;
goto v_resetjp_1403_;
}
v_resetjp_1403_:
{
lean_object* v___x_1406_; lean_object* v_currNamespace_1407_; lean_object* v_openDecls_1408_; lean_object* v_env_1409_; lean_object* v_messages_1410_; lean_object* v_scopes_1411_; lean_object* v_usedQuotCtxts_1412_; lean_object* v_nextMacroScope_1413_; lean_object* v_maxRecDepth_1414_; lean_object* v_ngen_1415_; lean_object* v_auxDeclNGen_1416_; lean_object* v_infoState_1417_; lean_object* v_traceState_1418_; lean_object* v_snapshotTasks_1419_; lean_object* v_prevLinterStates_1420_; lean_object* v___x_1422_; uint8_t v_isShared_1423_; uint8_t v_isSharedCheck_1436_; 
v___x_1406_ = lean_st_ref_take(v___y_1398_);
v_currNamespace_1407_ = lean_ctor_get(v_a_1400_, 2);
lean_inc(v_currNamespace_1407_);
lean_dec(v_a_1400_);
v_openDecls_1408_ = lean_ctor_get(v_a_1402_, 3);
lean_inc(v_openDecls_1408_);
lean_dec(v_a_1402_);
v_env_1409_ = lean_ctor_get(v___x_1406_, 0);
v_messages_1410_ = lean_ctor_get(v___x_1406_, 1);
v_scopes_1411_ = lean_ctor_get(v___x_1406_, 2);
v_usedQuotCtxts_1412_ = lean_ctor_get(v___x_1406_, 3);
v_nextMacroScope_1413_ = lean_ctor_get(v___x_1406_, 4);
v_maxRecDepth_1414_ = lean_ctor_get(v___x_1406_, 5);
v_ngen_1415_ = lean_ctor_get(v___x_1406_, 6);
v_auxDeclNGen_1416_ = lean_ctor_get(v___x_1406_, 7);
v_infoState_1417_ = lean_ctor_get(v___x_1406_, 8);
v_traceState_1418_ = lean_ctor_get(v___x_1406_, 9);
v_snapshotTasks_1419_ = lean_ctor_get(v___x_1406_, 10);
v_prevLinterStates_1420_ = lean_ctor_get(v___x_1406_, 11);
v_isSharedCheck_1436_ = !lean_is_exclusive(v___x_1406_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1422_ = v___x_1406_;
v_isShared_1423_ = v_isSharedCheck_1436_;
goto v_resetjp_1421_;
}
else
{
lean_inc(v_prevLinterStates_1420_);
lean_inc(v_snapshotTasks_1419_);
lean_inc(v_traceState_1418_);
lean_inc(v_infoState_1417_);
lean_inc(v_auxDeclNGen_1416_);
lean_inc(v_ngen_1415_);
lean_inc(v_maxRecDepth_1414_);
lean_inc(v_nextMacroScope_1413_);
lean_inc(v_usedQuotCtxts_1412_);
lean_inc(v_scopes_1411_);
lean_inc(v_messages_1410_);
lean_inc(v_env_1409_);
lean_dec(v___x_1406_);
v___x_1422_ = lean_box(0);
v_isShared_1423_ = v_isSharedCheck_1436_;
goto v_resetjp_1421_;
}
v_resetjp_1421_:
{
lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1429_; 
v___x_1424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1424_, 0, v_currNamespace_1407_);
lean_ctor_set(v___x_1424_, 1, v_openDecls_1408_);
v___x_1425_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1424_);
lean_ctor_set(v___x_1425_, 1, v___y_1392_);
lean_inc_ref(v___y_1394_);
lean_inc_ref(v___y_1397_);
v___x_1426_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1426_, 0, v___y_1397_);
lean_ctor_set(v___x_1426_, 1, v___y_1396_);
lean_ctor_set(v___x_1426_, 2, v___y_1391_);
lean_ctor_set(v___x_1426_, 3, v___y_1394_);
lean_ctor_set(v___x_1426_, 4, v___x_1425_);
lean_ctor_set_uint8(v___x_1426_, sizeof(void*)*5, v___y_1393_);
lean_ctor_set_uint8(v___x_1426_, sizeof(void*)*5 + 1, v___y_1395_);
lean_ctor_set_uint8(v___x_1426_, sizeof(void*)*5 + 2, v_isSilent_1386_);
v___x_1427_ = l_Lean_MessageLog_add(v___x_1426_, v_messages_1410_);
if (v_isShared_1423_ == 0)
{
lean_ctor_set(v___x_1422_, 1, v___x_1427_);
v___x_1429_ = v___x_1422_;
goto v_reusejp_1428_;
}
else
{
lean_object* v_reuseFailAlloc_1435_; 
v_reuseFailAlloc_1435_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1435_, 0, v_env_1409_);
lean_ctor_set(v_reuseFailAlloc_1435_, 1, v___x_1427_);
lean_ctor_set(v_reuseFailAlloc_1435_, 2, v_scopes_1411_);
lean_ctor_set(v_reuseFailAlloc_1435_, 3, v_usedQuotCtxts_1412_);
lean_ctor_set(v_reuseFailAlloc_1435_, 4, v_nextMacroScope_1413_);
lean_ctor_set(v_reuseFailAlloc_1435_, 5, v_maxRecDepth_1414_);
lean_ctor_set(v_reuseFailAlloc_1435_, 6, v_ngen_1415_);
lean_ctor_set(v_reuseFailAlloc_1435_, 7, v_auxDeclNGen_1416_);
lean_ctor_set(v_reuseFailAlloc_1435_, 8, v_infoState_1417_);
lean_ctor_set(v_reuseFailAlloc_1435_, 9, v_traceState_1418_);
lean_ctor_set(v_reuseFailAlloc_1435_, 10, v_snapshotTasks_1419_);
lean_ctor_set(v_reuseFailAlloc_1435_, 11, v_prevLinterStates_1420_);
v___x_1429_ = v_reuseFailAlloc_1435_;
goto v_reusejp_1428_;
}
v_reusejp_1428_:
{
lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1433_; 
v___x_1430_ = lean_st_ref_set(v___y_1398_, v___x_1429_);
v___x_1431_ = lean_box(0);
if (v_isShared_1405_ == 0)
{
lean_ctor_set(v___x_1404_, 0, v___x_1431_);
v___x_1433_ = v___x_1404_;
goto v_reusejp_1432_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v___x_1431_);
v___x_1433_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1432_;
}
v_reusejp_1432_:
{
return v___x_1433_;
}
}
}
}
}
else
{
lean_object* v_a_1438_; lean_object* v___x_1440_; uint8_t v_isShared_1441_; uint8_t v_isSharedCheck_1445_; 
lean_dec(v_a_1400_);
lean_dec_ref(v___y_1396_);
lean_dec_ref(v___y_1392_);
lean_dec(v___y_1391_);
v_a_1438_ = lean_ctor_get(v___x_1401_, 0);
v_isSharedCheck_1445_ = !lean_is_exclusive(v___x_1401_);
if (v_isSharedCheck_1445_ == 0)
{
v___x_1440_ = v___x_1401_;
v_isShared_1441_ = v_isSharedCheck_1445_;
goto v_resetjp_1439_;
}
else
{
lean_inc(v_a_1438_);
lean_dec(v___x_1401_);
v___x_1440_ = lean_box(0);
v_isShared_1441_ = v_isSharedCheck_1445_;
goto v_resetjp_1439_;
}
v_resetjp_1439_:
{
lean_object* v___x_1443_; 
if (v_isShared_1441_ == 0)
{
v___x_1443_ = v___x_1440_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1444_; 
v_reuseFailAlloc_1444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1444_, 0, v_a_1438_);
v___x_1443_ = v_reuseFailAlloc_1444_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
return v___x_1443_;
}
}
}
}
else
{
lean_object* v_a_1446_; lean_object* v___x_1448_; uint8_t v_isShared_1449_; uint8_t v_isSharedCheck_1453_; 
lean_dec_ref(v___y_1396_);
lean_dec_ref(v___y_1392_);
lean_dec(v___y_1391_);
v_a_1446_ = lean_ctor_get(v___x_1399_, 0);
v_isSharedCheck_1453_ = !lean_is_exclusive(v___x_1399_);
if (v_isSharedCheck_1453_ == 0)
{
v___x_1448_ = v___x_1399_;
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
else
{
lean_inc(v_a_1446_);
lean_dec(v___x_1399_);
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
}
v___jp_1454_:
{
lean_object* v_fileName_1460_; lean_object* v_fileMap_1461_; uint8_t v_suppressElabErrors_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v_a_1465_; lean_object* v___x_1467_; uint8_t v_isShared_1468_; uint8_t v_isSharedCheck_1481_; 
v_fileName_1460_ = lean_ctor_get(v___y_1387_, 0);
v_fileMap_1461_ = lean_ctor_get(v___y_1387_, 1);
v_suppressElabErrors_1462_ = lean_ctor_get_uint8(v___y_1387_, sizeof(void*)*10);
v___x_1463_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1384_);
v___x_1464_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg(v___x_1463_, v___y_1388_);
v_a_1465_ = lean_ctor_get(v___x_1464_, 0);
v_isSharedCheck_1481_ = !lean_is_exclusive(v___x_1464_);
if (v_isSharedCheck_1481_ == 0)
{
v___x_1467_ = v___x_1464_;
v_isShared_1468_ = v_isSharedCheck_1481_;
goto v_resetjp_1466_;
}
else
{
lean_inc(v_a_1465_);
lean_dec(v___x_1464_);
v___x_1467_ = lean_box(0);
v_isShared_1468_ = v_isSharedCheck_1481_;
goto v_resetjp_1466_;
}
v_resetjp_1466_:
{
lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; 
lean_inc_ref_n(v_fileMap_1461_, 2);
v___x_1469_ = l_Lean_FileMap_toPosition(v_fileMap_1461_, v___y_1457_);
lean_dec(v___y_1457_);
v___x_1470_ = l_Lean_FileMap_toPosition(v_fileMap_1461_, v___y_1459_);
lean_dec(v___y_1459_);
v___x_1471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1471_, 0, v___x_1470_);
v___x_1472_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
if (v_suppressElabErrors_1462_ == 0)
{
lean_del_object(v___x_1467_);
v___y_1391_ = v___x_1471_;
v___y_1392_ = v_a_1465_;
v___y_1393_ = v___y_1456_;
v___y_1394_ = v___x_1472_;
v___y_1395_ = v___y_1458_;
v___y_1396_ = v___x_1469_;
v___y_1397_ = v_fileName_1460_;
v___y_1398_ = v___y_1388_;
goto v___jp_1390_;
}
else
{
lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___f_1475_; uint8_t v___x_1476_; 
v___x_1473_ = lean_box(v___y_1455_);
v___x_1474_ = lean_box(v_suppressElabErrors_1462_);
v___f_1475_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1475_, 0, v___x_1473_);
lean_closure_set(v___f_1475_, 1, v___x_1474_);
lean_inc(v_a_1465_);
v___x_1476_ = l_Lean_MessageData_hasTag(v___f_1475_, v_a_1465_);
if (v___x_1476_ == 0)
{
lean_object* v___x_1477_; lean_object* v___x_1479_; 
lean_dec_ref_known(v___x_1471_, 1);
lean_dec_ref(v___x_1469_);
lean_dec(v_a_1465_);
v___x_1477_ = lean_box(0);
if (v_isShared_1468_ == 0)
{
lean_ctor_set(v___x_1467_, 0, v___x_1477_);
v___x_1479_ = v___x_1467_;
goto v_reusejp_1478_;
}
else
{
lean_object* v_reuseFailAlloc_1480_; 
v_reuseFailAlloc_1480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1480_, 0, v___x_1477_);
v___x_1479_ = v_reuseFailAlloc_1480_;
goto v_reusejp_1478_;
}
v_reusejp_1478_:
{
return v___x_1479_;
}
}
else
{
lean_del_object(v___x_1467_);
v___y_1391_ = v___x_1471_;
v___y_1392_ = v_a_1465_;
v___y_1393_ = v___y_1456_;
v___y_1394_ = v___x_1472_;
v___y_1395_ = v___y_1458_;
v___y_1396_ = v___x_1469_;
v___y_1397_ = v_fileName_1460_;
v___y_1398_ = v___y_1388_;
goto v___jp_1390_;
}
}
}
}
v___jp_1482_:
{
lean_object* v___x_1488_; 
v___x_1488_ = l_Lean_Syntax_getTailPos_x3f(v___y_1484_, v___y_1485_);
lean_dec(v___y_1484_);
if (lean_obj_tag(v___x_1488_) == 0)
{
lean_inc(v___y_1487_);
v___y_1455_ = v___y_1483_;
v___y_1456_ = v___y_1485_;
v___y_1457_ = v___y_1487_;
v___y_1458_ = v___y_1486_;
v___y_1459_ = v___y_1487_;
goto v___jp_1454_;
}
else
{
lean_object* v_val_1489_; 
v_val_1489_ = lean_ctor_get(v___x_1488_, 0);
lean_inc(v_val_1489_);
lean_dec_ref_known(v___x_1488_, 1);
v___y_1455_ = v___y_1483_;
v___y_1456_ = v___y_1485_;
v___y_1457_ = v___y_1487_;
v___y_1458_ = v___y_1486_;
v___y_1459_ = v_val_1489_;
goto v___jp_1454_;
}
}
v___jp_1490_:
{
lean_object* v___x_1494_; 
v___x_1494_ = l_Lean_Elab_Command_getRef___redArg(v___y_1387_);
if (lean_obj_tag(v___x_1494_) == 0)
{
lean_object* v_a_1495_; lean_object* v_ref_1496_; lean_object* v___x_1497_; 
v_a_1495_ = lean_ctor_get(v___x_1494_, 0);
lean_inc(v_a_1495_);
lean_dec_ref_known(v___x_1494_, 1);
v_ref_1496_ = l_Lean_replaceRef(v_ref_1383_, v_a_1495_);
lean_dec(v_a_1495_);
v___x_1497_ = l_Lean_Syntax_getPos_x3f(v_ref_1496_, v___y_1492_);
if (lean_obj_tag(v___x_1497_) == 0)
{
lean_object* v___x_1498_; 
v___x_1498_ = lean_unsigned_to_nat(0u);
v___y_1483_ = v___y_1491_;
v___y_1484_ = v_ref_1496_;
v___y_1485_ = v___y_1492_;
v___y_1486_ = v___y_1493_;
v___y_1487_ = v___x_1498_;
goto v___jp_1482_;
}
else
{
lean_object* v_val_1499_; 
v_val_1499_ = lean_ctor_get(v___x_1497_, 0);
lean_inc(v_val_1499_);
lean_dec_ref_known(v___x_1497_, 1);
v___y_1483_ = v___y_1491_;
v___y_1484_ = v_ref_1496_;
v___y_1485_ = v___y_1492_;
v___y_1486_ = v___y_1493_;
v___y_1487_ = v_val_1499_;
goto v___jp_1482_;
}
}
else
{
lean_object* v_a_1500_; lean_object* v___x_1502_; uint8_t v_isShared_1503_; uint8_t v_isSharedCheck_1507_; 
lean_dec_ref(v_msgData_1384_);
v_a_1500_ = lean_ctor_get(v___x_1494_, 0);
v_isSharedCheck_1507_ = !lean_is_exclusive(v___x_1494_);
if (v_isSharedCheck_1507_ == 0)
{
v___x_1502_ = v___x_1494_;
v_isShared_1503_ = v_isSharedCheck_1507_;
goto v_resetjp_1501_;
}
else
{
lean_inc(v_a_1500_);
lean_dec(v___x_1494_);
v___x_1502_ = lean_box(0);
v_isShared_1503_ = v_isSharedCheck_1507_;
goto v_resetjp_1501_;
}
v_resetjp_1501_:
{
lean_object* v___x_1505_; 
if (v_isShared_1503_ == 0)
{
v___x_1505_ = v___x_1502_;
goto v_reusejp_1504_;
}
else
{
lean_object* v_reuseFailAlloc_1506_; 
v_reuseFailAlloc_1506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1506_, 0, v_a_1500_);
v___x_1505_ = v_reuseFailAlloc_1506_;
goto v_reusejp_1504_;
}
v_reusejp_1504_:
{
return v___x_1505_;
}
}
}
}
v___jp_1509_:
{
if (v___y_1512_ == 0)
{
v___y_1491_ = v___y_1510_;
v___y_1492_ = v___y_1511_;
v___y_1493_ = v_severity_1385_;
goto v___jp_1490_;
}
else
{
v___y_1491_ = v___y_1510_;
v___y_1492_ = v___y_1511_;
v___y_1493_ = v___x_1508_;
goto v___jp_1490_;
}
}
v___jp_1513_:
{
if (v___y_1514_ == 0)
{
lean_object* v___x_1515_; lean_object* v_scopes_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v_opts_1519_; uint8_t v___x_1520_; uint8_t v___x_1521_; 
v___x_1515_ = lean_st_ref_get(v___y_1388_);
v_scopes_1516_ = lean_ctor_get(v___x_1515_, 2);
lean_inc(v_scopes_1516_);
lean_dec(v___x_1515_);
v___x_1517_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1518_ = l_List_head_x21___redArg(v___x_1517_, v_scopes_1516_);
lean_dec(v_scopes_1516_);
v_opts_1519_ = lean_ctor_get(v___x_1518_, 1);
lean_inc_ref(v_opts_1519_);
lean_dec(v___x_1518_);
v___x_1520_ = 1;
v___x_1521_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1385_, v___x_1520_);
if (v___x_1521_ == 0)
{
lean_dec_ref(v_opts_1519_);
v___y_1510_ = v___y_1514_;
v___y_1511_ = v___y_1514_;
v___y_1512_ = v___x_1521_;
goto v___jp_1509_;
}
else
{
lean_object* v___x_1522_; uint8_t v___x_1523_; 
v___x_1522_ = l_Lean_warningAsError;
v___x_1523_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__5(v_opts_1519_, v___x_1522_);
lean_dec_ref(v_opts_1519_);
v___y_1510_ = v___y_1514_;
v___y_1511_ = v___y_1514_;
v___y_1512_ = v___x_1523_;
goto v___jp_1509_;
}
}
else
{
lean_object* v___x_1524_; lean_object* v___x_1525_; 
lean_dec_ref(v_msgData_1384_);
v___x_1524_ = lean_box(0);
v___x_1525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1525_, 0, v___x_1524_);
return v___x_1525_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1___boxed(lean_object* v_ref_1528_, lean_object* v_msgData_1529_, lean_object* v_severity_1530_, lean_object* v_isSilent_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
uint8_t v_severity_boxed_1535_; uint8_t v_isSilent_boxed_1536_; lean_object* v_res_1537_; 
v_severity_boxed_1535_ = lean_unbox(v_severity_1530_);
v_isSilent_boxed_1536_ = lean_unbox(v_isSilent_1531_);
v_res_1537_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1(v_ref_1528_, v_msgData_1529_, v_severity_boxed_1535_, v_isSilent_boxed_1536_, v___y_1532_, v___y_1533_);
lean_dec(v___y_1533_);
lean_dec_ref(v___y_1532_);
lean_dec(v_ref_1528_);
return v_res_1537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0(lean_object* v_ref_1538_, lean_object* v_msgData_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_){
_start:
{
uint8_t v___x_1543_; uint8_t v___x_1544_; lean_object* v___x_1545_; 
v___x_1543_ = 1;
v___x_1544_ = 0;
v___x_1545_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1(v_ref_1538_, v_msgData_1539_, v___x_1543_, v___x_1544_, v___y_1540_, v___y_1541_);
return v___x_1545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0___boxed(lean_object* v_ref_1546_, lean_object* v_msgData_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_){
_start:
{
lean_object* v_res_1551_; 
v_res_1551_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0(v_ref_1546_, v_msgData_1547_, v___y_1548_, v___y_1549_);
lean_dec(v___y_1549_);
lean_dec_ref(v___y_1548_);
lean_dec(v_ref_1546_);
return v_res_1551_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1553_; lean_object* v___x_1554_; 
v___x_1553_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__0));
v___x_1554_ = l_Lean_stringToMessageData(v___x_1553_);
return v___x_1554_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__3(void){
_start:
{
lean_object* v___x_1556_; lean_object* v___x_1557_; 
v___x_1556_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__2));
v___x_1557_ = l_Lean_stringToMessageData(v___x_1556_);
return v___x_1557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(lean_object* v_linterOption_1558_, lean_object* v_stx_1559_, lean_object* v_msg_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_){
_start:
{
lean_object* v_name_1564_; lean_object* v___x_1566_; uint8_t v_isShared_1567_; uint8_t v_isSharedCheck_1582_; 
v_name_1564_ = lean_ctor_get(v_linterOption_1558_, 0);
v_isSharedCheck_1582_ = !lean_is_exclusive(v_linterOption_1558_);
if (v_isSharedCheck_1582_ == 0)
{
lean_object* v_unused_1583_; 
v_unused_1583_ = lean_ctor_get(v_linterOption_1558_, 1);
lean_dec(v_unused_1583_);
v___x_1566_ = v_linterOption_1558_;
v_isShared_1567_ = v_isSharedCheck_1582_;
goto v_resetjp_1565_;
}
else
{
lean_inc(v_name_1564_);
lean_dec(v_linterOption_1558_);
v___x_1566_ = lean_box(0);
v_isShared_1567_ = v_isSharedCheck_1582_;
goto v_resetjp_1565_;
}
v_resetjp_1565_:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1571_; 
v___x_1568_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__1);
lean_inc(v_name_1564_);
v___x_1569_ = l_Lean_MessageData_ofName(v_name_1564_);
if (v_isShared_1567_ == 0)
{
lean_ctor_set_tag(v___x_1566_, 7);
lean_ctor_set(v___x_1566_, 1, v___x_1569_);
lean_ctor_set(v___x_1566_, 0, v___x_1568_);
v___x_1571_ = v___x_1566_;
goto v_reusejp_1570_;
}
else
{
lean_object* v_reuseFailAlloc_1581_; 
v_reuseFailAlloc_1581_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1581_, 0, v___x_1568_);
lean_ctor_set(v_reuseFailAlloc_1581_, 1, v___x_1569_);
v___x_1571_ = v_reuseFailAlloc_1581_;
goto v_reusejp_1570_;
}
v_reusejp_1570_:
{
lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v_disable_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; 
v___x_1572_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___closed__3);
v___x_1573_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1573_, 0, v___x_1571_);
lean_ctor_set(v___x_1573_, 1, v___x_1572_);
v_disable_1574_ = l_Lean_MessageData_note(v___x_1573_);
v___x_1575_ = l_Lean_Linter_linterMessageTag;
v___x_1576_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1576_, 0, v_msg_1560_);
lean_ctor_set(v___x_1576_, 1, v_disable_1574_);
v___x_1577_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1577_, 0, v___x_1575_);
lean_ctor_set(v___x_1577_, 1, v___x_1576_);
v___x_1578_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1578_, 0, v_name_1564_);
lean_ctor_set(v___x_1578_, 1, v___x_1577_);
lean_inc(v_stx_1559_);
v___x_1579_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1579_, 0, v_stx_1559_);
lean_ctor_set(v___x_1579_, 1, v___x_1578_);
v___x_1580_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0(v_stx_1559_, v___x_1579_, v___y_1561_, v___y_1562_);
lean_dec(v_stx_1559_);
return v___x_1580_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0___boxed(lean_object* v_linterOption_1584_, lean_object* v_stx_1585_, lean_object* v_msg_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_){
_start:
{
lean_object* v_res_1590_; 
v_res_1590_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v_linterOption_1584_, v_stx_1585_, v_msg_1586_, v___y_1587_, v___y_1588_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
return v_res_1590_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__1(lean_object* v_a_1591_, lean_object* v_x_1592_){
_start:
{
if (lean_obj_tag(v_x_1592_) == 0)
{
uint8_t v___x_1593_; 
v___x_1593_ = 0;
return v___x_1593_;
}
else
{
lean_object* v_head_1594_; lean_object* v_tail_1595_; uint8_t v___x_1596_; 
v_head_1594_ = lean_ctor_get(v_x_1592_, 0);
v_tail_1595_ = lean_ctor_get(v_x_1592_, 1);
v___x_1596_ = lean_name_eq(v_a_1591_, v_head_1594_);
if (v___x_1596_ == 0)
{
v_x_1592_ = v_tail_1595_;
goto _start;
}
else
{
return v___x_1596_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__1___boxed(lean_object* v_a_1598_, lean_object* v_x_1599_){
_start:
{
uint8_t v_res_1600_; lean_object* v_r_1601_; 
v_res_1600_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__1(v_a_1598_, v_x_1599_);
lean_dec(v_x_1599_);
lean_dec(v_a_1598_);
v_r_1601_ = lean_box(v_res_1600_);
return v_r_1601_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__6(void){
_start:
{
lean_object* v___x_1610_; lean_object* v___x_1611_; 
v___x_1610_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__5));
v___x_1611_ = l_Lean_MessageData_ofFormat(v___x_1610_);
return v___x_1611_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__18(void){
_start:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1633_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__17));
v___x_1634_ = l_Lean_MessageData_ofFormat(v___x_1633_);
return v___x_1634_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__21(void){
_start:
{
lean_object* v___x_1638_; lean_object* v___x_1639_; 
v___x_1638_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__20));
v___x_1639_ = l_Lean_MessageData_ofFormat(v___x_1638_);
return v___x_1639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2(uint8_t v___y_1642_, lean_object* v_mainModule_1643_, lean_object* v_as_1644_, size_t v_sz_1645_, size_t v_i_1646_, lean_object* v_b_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_){
_start:
{
lean_object* v_a_1652_; uint8_t v___x_1656_; 
v___x_1656_ = lean_usize_dec_lt(v_i_1646_, v_sz_1645_);
if (v___x_1656_ == 0)
{
lean_object* v___x_1657_; 
v___x_1657_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1657_, 0, v_b_1647_);
return v___x_1657_;
}
else
{
lean_object* v_a_1658_; lean_object* v_toImport_1659_; lean_object* v_module_1660_; lean_object* v___x_1661_; lean_object* v___y_1663_; lean_object* v___y_1664_; lean_object* v_modName_1678_; lean_object* v___y_1679_; lean_object* v___y_1680_; 
v_a_1658_ = lean_array_uget_borrowed(v_as_1644_, v_i_1646_);
v_toImport_1659_ = lean_ctor_get(v_a_1658_, 0);
v_module_1660_ = lean_ctor_get(v_toImport_1659_, 0);
v___x_1661_ = lean_box(0);
if (lean_obj_tag(v_module_1660_) == 1)
{
lean_object* v_pre_1688_; 
v_pre_1688_ = lean_ctor_get(v_module_1660_, 0);
switch(lean_obj_tag(v_pre_1688_))
{
case 1:
{
lean_object* v_str_1689_; lean_object* v_pre_1690_; lean_object* v_str_1691_; lean_object* v___x_1692_; 
v_str_1689_ = lean_ctor_get(v_module_1660_, 1);
v_pre_1690_ = lean_ctor_get(v_pre_1688_, 0);
v_str_1691_ = lean_ctor_get(v_pre_1688_, 1);
v___x_1692_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn___closed__6_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_));
switch(lean_obj_tag(v_pre_1690_))
{
case 0:
{
uint8_t v___x_1693_; 
v___x_1693_ = lean_string_dec_eq(v_str_1691_, v___x_1692_);
if (v___x_1693_ == 0)
{
lean_object* v___x_1694_; uint8_t v___x_1695_; 
v___x_1694_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0));
v___x_1695_ = lean_string_dec_eq(v_str_1691_, v___x_1694_);
if (v___x_1695_ == 0)
{
lean_inc_ref(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
lean_object* v___x_1696_; uint8_t v___x_1697_; 
v___x_1696_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__7));
v___x_1697_ = lean_string_dec_eq(v_str_1689_, v___x_1696_);
if (v___x_1697_ == 0)
{
lean_object* v___x_1698_; uint8_t v___x_1699_; 
v___x_1698_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__8));
v___x_1699_ = lean_string_dec_eq(v_str_1689_, v___x_1698_);
if (v___x_1699_ == 0)
{
lean_object* v___x_1700_; lean_object* v___x_1701_; 
v___x_1700_ = l_Lean_Name_str___override(v_pre_1690_, v___x_1694_);
lean_inc_ref(v_str_1689_);
v___x_1701_ = l_Lean_Name_str___override(v___x_1700_, v_str_1689_);
v_modName_1678_ = v___x_1701_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
}
else
{
lean_object* v___x_1702_; uint8_t v___x_1703_; 
v___x_1702_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9));
v___x_1703_ = lean_string_dec_eq(v_str_1689_, v___x_1702_);
if (v___x_1703_ == 0)
{
lean_object* v___x_1704_; lean_object* v___x_1705_; 
v___x_1704_ = l_Lean_Name_str___override(v_pre_1690_, v___x_1692_);
lean_inc_ref(v_str_1689_);
v___x_1705_ = l_Lean_Name_str___override(v___x_1704_, v_str_1689_);
v_modName_1678_ = v___x_1705_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
}
case 1:
{
lean_object* v_pre_1706_; 
v_pre_1706_ = lean_ctor_get(v_pre_1690_, 0);
if (lean_obj_tag(v_pre_1706_) == 0)
{
lean_object* v_str_1707_; lean_object* v___x_1708_; uint8_t v___x_1709_; 
v_str_1707_ = lean_ctor_get(v_pre_1690_, 1);
v___x_1708_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0));
v___x_1709_ = lean_string_dec_eq(v_str_1707_, v___x_1708_);
if (v___x_1709_ == 0)
{
uint8_t v___x_1710_; 
v___x_1710_ = lean_string_dec_eq(v_str_1707_, v___x_1692_);
if (v___x_1710_ == 0)
{
lean_inc_ref(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
lean_object* v___x_1711_; uint8_t v___x_1712_; 
v___x_1711_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9));
v___x_1712_ = lean_string_dec_eq(v_str_1691_, v___x_1711_);
if (v___x_1712_ == 0)
{
lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; 
v___x_1713_ = l_Lean_Name_str___override(v_pre_1706_, v___x_1692_);
lean_inc_ref(v_str_1691_);
v___x_1714_ = l_Lean_Name_str___override(v___x_1713_, v_str_1691_);
lean_inc_ref(v_str_1689_);
v___x_1715_ = l_Lean_Name_str___override(v___x_1714_, v_str_1689_);
v_modName_1678_ = v___x_1715_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
lean_object* v___x_1716_; uint8_t v___x_1717_; 
v___x_1716_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__10));
v___x_1717_ = lean_string_dec_eq(v_str_1689_, v___x_1716_);
if (v___x_1717_ == 0)
{
lean_object* v___x_1718_; uint8_t v___x_1719_; 
v___x_1718_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__11));
v___x_1719_ = lean_string_dec_eq(v_str_1689_, v___x_1718_);
if (v___x_1719_ == 0)
{
lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; 
v___x_1720_ = l_Lean_Name_str___override(v_pre_1706_, v___x_1692_);
v___x_1721_ = l_Lean_Name_str___override(v___x_1720_, v___x_1711_);
lean_inc_ref(v_str_1689_);
v___x_1722_ = l_Lean_Name_str___override(v___x_1721_, v_str_1689_);
v_modName_1678_ = v___x_1722_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
lean_object* v___x_1723_; uint8_t v___x_1724_; 
v___x_1723_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__15));
v___x_1724_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__1(v_mainModule_1643_, v___x_1723_);
if (v___x_1724_ == 0)
{
lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; 
v___x_1725_ = lp_mathlib_Mathlib_Linter_linter_style_header;
lean_inc(v_a_1658_);
v___x_1726_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(v_a_1658_);
v___x_1727_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__18, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__18_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__18);
v___x_1728_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_1725_, v___x_1726_, v___x_1727_, v___y_1648_, v___y_1649_);
if (lean_obj_tag(v___x_1728_) == 0)
{
lean_dec_ref_known(v___x_1728_, 1);
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
else
{
return v___x_1728_;
}
}
else
{
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
}
}
else
{
lean_object* v___x_1729_; uint8_t v___x_1730_; 
v___x_1729_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12));
v___x_1730_ = lean_name_eq(v_mainModule_1643_, v___x_1729_);
if (v___x_1730_ == 0)
{
lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; 
v___x_1731_ = lp_mathlib_Mathlib_Linter_linter_style_header;
lean_inc(v_a_1658_);
v___x_1732_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(v_a_1658_);
v___x_1733_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__21, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__21_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__21);
v___x_1734_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_1731_, v___x_1732_, v___x_1733_, v___y_1648_, v___y_1649_);
if (lean_obj_tag(v___x_1734_) == 0)
{
lean_dec_ref_known(v___x_1734_, 1);
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
else
{
return v___x_1734_;
}
}
else
{
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
}
}
}
}
else
{
lean_object* v___x_1735_; uint8_t v___x_1736_; 
v___x_1735_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__8));
v___x_1736_ = lean_string_dec_eq(v_str_1691_, v___x_1735_);
if (v___x_1736_ == 0)
{
lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; 
v___x_1737_ = l_Lean_Name_str___override(v_pre_1706_, v___x_1708_);
lean_inc_ref(v_str_1691_);
v___x_1738_ = l_Lean_Name_str___override(v___x_1737_, v_str_1691_);
lean_inc_ref(v_str_1689_);
v___x_1739_ = l_Lean_Name_str___override(v___x_1738_, v_str_1689_);
v_modName_1678_ = v___x_1739_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
lean_object* v___x_1740_; uint8_t v___x_1741_; 
v___x_1740_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__9));
v___x_1741_ = lean_string_dec_eq(v_str_1689_, v___x_1740_);
if (v___x_1741_ == 0)
{
lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; 
v___x_1742_ = l_Lean_Name_str___override(v_pre_1706_, v___x_1708_);
v___x_1743_ = l_Lean_Name_str___override(v___x_1742_, v___x_1735_);
lean_inc_ref(v_str_1689_);
v___x_1744_ = l_Lean_Name_str___override(v___x_1743_, v_str_1689_);
v_modName_1678_ = v___x_1744_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
}
}
else
{
lean_inc_ref(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
}
default: 
{
lean_inc_ref(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
}
}
case 0:
{
lean_object* v_str_1745_; lean_object* v___x_1746_; uint8_t v___x_1747_; 
v_str_1745_ = lean_ctor_get(v_module_1660_, 1);
v___x_1746_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent___closed__0));
v___x_1747_ = lean_string_dec_eq(v_str_1745_, v___x_1746_);
if (v___x_1747_ == 0)
{
lean_object* v___x_1748_; uint8_t v___x_1749_; 
v___x_1748_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__22));
v___x_1749_ = lean_string_dec_eq(v_str_1745_, v___x_1748_);
if (v___x_1749_ == 0)
{
lean_object* v___x_1750_; uint8_t v___x_1751_; 
v___x_1750_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__23));
v___x_1751_ = lean_string_dec_eq(v_str_1745_, v___x_1750_);
if (v___x_1751_ == 0)
{
lean_inc_ref(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
else
{
v___y_1663_ = v___y_1648_;
v___y_1664_ = v___y_1649_;
goto v___jp_1662_;
}
}
default: 
{
lean_inc_ref(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
}
}
else
{
lean_inc(v_module_1660_);
v_modName_1678_ = v_module_1660_;
v___y_1679_ = v___y_1648_;
v___y_1680_ = v___y_1649_;
goto v___jp_1677_;
}
v___jp_1662_:
{
lean_object* v_toImport_1665_; lean_object* v_module_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; 
v_toImport_1665_ = lean_ctor_get(v_a_1658_, 0);
v_module_1666_ = lean_ctor_get(v_toImport_1665_, 0);
v___x_1667_ = lp_mathlib_Mathlib_Linter_linter_style_header;
lean_inc(v_a_1658_);
v___x_1668_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(v_a_1658_);
v___x_1669_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__0));
lean_inc(v_module_1666_);
v___x_1670_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_module_1666_, v___y_1642_);
v___x_1671_ = lean_string_append(v___x_1669_, v___x_1670_);
lean_dec_ref(v___x_1670_);
v___x_1672_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__1));
v___x_1673_ = lean_string_append(v___x_1671_, v___x_1672_);
v___x_1674_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1674_, 0, v___x_1673_);
v___x_1675_ = l_Lean_MessageData_ofFormat(v___x_1674_);
v___x_1676_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_1667_, v___x_1668_, v___x_1675_, v___y_1663_, v___y_1664_);
if (lean_obj_tag(v___x_1676_) == 0)
{
lean_dec_ref_known(v___x_1676_, 1);
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
else
{
return v___x_1676_;
}
}
v___jp_1677_:
{
lean_object* v___x_1681_; lean_object* v___x_1682_; uint8_t v___x_1683_; 
v___x_1681_ = l_Lean_Name_getRoot(v_modName_1678_);
lean_dec(v_modName_1678_);
v___x_1682_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__3));
v___x_1683_ = lean_name_eq(v___x_1681_, v___x_1682_);
lean_dec(v___x_1681_);
if (v___x_1683_ == 0)
{
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
else
{
lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; 
v___x_1684_ = lp_mathlib_Mathlib_Linter_linter_style_header;
lean_inc(v_a_1658_);
v___x_1685_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(v_a_1658_);
v___x_1686_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__6, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__6_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__6);
v___x_1687_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_1684_, v___x_1685_, v___x_1686_, v___y_1679_, v___y_1680_);
if (lean_obj_tag(v___x_1687_) == 0)
{
lean_dec_ref_known(v___x_1687_, 1);
v_a_1652_ = v___x_1661_;
goto v___jp_1651_;
}
else
{
return v___x_1687_;
}
}
}
}
v___jp_1651_:
{
size_t v___x_1653_; size_t v___x_1654_; 
v___x_1653_ = ((size_t)1ULL);
v___x_1654_ = lean_usize_add(v_i_1646_, v___x_1653_);
v_i_1646_ = v___x_1654_;
v_b_1647_ = v_a_1652_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___boxed(lean_object* v___y_1752_, lean_object* v_mainModule_1753_, lean_object* v_as_1754_, lean_object* v_sz_1755_, lean_object* v_i_1756_, lean_object* v_b_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_){
_start:
{
uint8_t v___y_6525__boxed_1761_; size_t v_sz_boxed_1762_; size_t v_i_boxed_1763_; lean_object* v_res_1764_; 
v___y_6525__boxed_1761_ = lean_unbox(v___y_1752_);
v_sz_boxed_1762_ = lean_unbox_usize(v_sz_1755_);
lean_dec(v_sz_1755_);
v_i_boxed_1763_ = lean_unbox_usize(v_i_1756_);
lean_dec(v_i_1756_);
v_res_1764_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2(v___y_6525__boxed_1761_, v_mainModule_1753_, v_as_1754_, v_sz_boxed_1762_, v_i_boxed_1763_, v_b_1757_, v___y_1758_, v___y_1759_);
lean_dec(v___y_1759_);
lean_dec_ref(v___y_1758_);
lean_dec_ref(v_as_1754_);
lean_dec(v_mainModule_1753_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck(lean_object* v_imports_1767_, lean_object* v_mainModule_1768_, lean_object* v_a_1769_, lean_object* v_a_1770_){
_start:
{
uint8_t v___y_1773_; lean_object* v___x_1788_; lean_object* v___x_1789_; uint8_t v___x_1790_; 
v___x_1788_ = l_Lean_Name_getRoot(v_mainModule_1768_);
v___x_1789_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck___closed__0));
v___x_1790_ = lean_name_eq(v___x_1788_, v___x_1789_);
lean_dec(v___x_1788_);
if (v___x_1790_ == 0)
{
lean_object* v___x_1791_; uint8_t v___x_1792_; 
v___x_1791_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles;
v___x_1792_ = l_Lean_NameSet_contains(v___x_1791_, v_mainModule_1768_);
v___y_1773_ = v___x_1792_;
goto v___jp_1772_;
}
else
{
v___y_1773_ = v___x_1790_;
goto v___jp_1772_;
}
v___jp_1772_:
{
if (v___y_1773_ == 0)
{
lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1774_ = lean_box(0);
v___x_1775_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1775_, 0, v___x_1774_);
return v___x_1775_;
}
else
{
lean_object* v___x_1776_; size_t v_sz_1777_; size_t v___x_1778_; lean_object* v___x_1779_; 
v___x_1776_ = lean_box(0);
v_sz_1777_ = lean_array_size(v_imports_1767_);
v___x_1778_ = ((size_t)0ULL);
v___x_1779_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2(v___y_1773_, v_mainModule_1768_, v_imports_1767_, v_sz_1777_, v___x_1778_, v___x_1776_, v_a_1769_, v_a_1770_);
if (lean_obj_tag(v___x_1779_) == 0)
{
lean_object* v___x_1781_; uint8_t v_isShared_1782_; uint8_t v_isSharedCheck_1786_; 
v_isSharedCheck_1786_ = !lean_is_exclusive(v___x_1779_);
if (v_isSharedCheck_1786_ == 0)
{
lean_object* v_unused_1787_; 
v_unused_1787_ = lean_ctor_get(v___x_1779_, 0);
lean_dec(v_unused_1787_);
v___x_1781_ = v___x_1779_;
v_isShared_1782_ = v_isSharedCheck_1786_;
goto v_resetjp_1780_;
}
else
{
lean_dec(v___x_1779_);
v___x_1781_ = lean_box(0);
v_isShared_1782_ = v_isSharedCheck_1786_;
goto v_resetjp_1780_;
}
v_resetjp_1780_:
{
lean_object* v___x_1784_; 
if (v_isShared_1782_ == 0)
{
lean_ctor_set(v___x_1781_, 0, v___x_1776_);
v___x_1784_ = v___x_1781_;
goto v_reusejp_1783_;
}
else
{
lean_object* v_reuseFailAlloc_1785_; 
v_reuseFailAlloc_1785_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1785_, 0, v___x_1776_);
v___x_1784_ = v_reuseFailAlloc_1785_;
goto v_reusejp_1783_;
}
v_reusejp_1783_:
{
return v___x_1784_;
}
}
}
else
{
return v___x_1779_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck___boxed(lean_object* v_imports_1793_, lean_object* v_mainModule_1794_, lean_object* v_a_1795_, lean_object* v_a_1796_, lean_object* v_a_1797_){
_start:
{
lean_object* v_res_1798_; 
v_res_1798_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck(v_imports_1793_, v_mainModule_1794_, v_a_1795_, v_a_1796_);
lean_dec(v_a_1796_);
lean_dec_ref(v_a_1795_);
lean_dec(v_mainModule_1794_);
lean_dec_ref(v_imports_1793_);
return v_res_1798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4(lean_object* v_msgData_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_){
_start:
{
lean_object* v___x_1803_; 
v___x_1803_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___redArg(v_msgData_1799_, v___y_1801_);
return v___x_1803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_msgData_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_){
_start:
{
lean_object* v_res_1808_; 
v_res_1808_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0_spec__0_spec__1_spec__4(v_msgData_1804_, v___y_1805_, v___y_1806_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
return v_res_1808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3_spec__5___redArg(lean_object* v_x_1809_, lean_object* v_x_1810_){
_start:
{
if (lean_obj_tag(v_x_1810_) == 0)
{
return v_x_1809_;
}
else
{
lean_object* v_key_1811_; lean_object* v_value_1812_; lean_object* v_tail_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1836_; 
v_key_1811_ = lean_ctor_get(v_x_1810_, 0);
v_value_1812_ = lean_ctor_get(v_x_1810_, 1);
v_tail_1813_ = lean_ctor_get(v_x_1810_, 2);
v_isSharedCheck_1836_ = !lean_is_exclusive(v_x_1810_);
if (v_isSharedCheck_1836_ == 0)
{
v___x_1815_ = v_x_1810_;
v_isShared_1816_ = v_isSharedCheck_1836_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_tail_1813_);
lean_inc(v_value_1812_);
lean_inc(v_key_1811_);
lean_dec(v_x_1810_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1836_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1817_; uint64_t v___x_1818_; uint64_t v___x_1819_; uint64_t v___x_1820_; uint64_t v_fold_1821_; uint64_t v___x_1822_; uint64_t v___x_1823_; uint64_t v___x_1824_; size_t v___x_1825_; size_t v___x_1826_; size_t v___x_1827_; size_t v___x_1828_; size_t v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1832_; 
v___x_1817_ = lean_array_get_size(v_x_1809_);
v___x_1818_ = l_Lean_instHashableImport_hash(v_key_1811_);
v___x_1819_ = 32ULL;
v___x_1820_ = lean_uint64_shift_right(v___x_1818_, v___x_1819_);
v_fold_1821_ = lean_uint64_xor(v___x_1818_, v___x_1820_);
v___x_1822_ = 16ULL;
v___x_1823_ = lean_uint64_shift_right(v_fold_1821_, v___x_1822_);
v___x_1824_ = lean_uint64_xor(v_fold_1821_, v___x_1823_);
v___x_1825_ = lean_uint64_to_usize(v___x_1824_);
v___x_1826_ = lean_usize_of_nat(v___x_1817_);
v___x_1827_ = ((size_t)1ULL);
v___x_1828_ = lean_usize_sub(v___x_1826_, v___x_1827_);
v___x_1829_ = lean_usize_land(v___x_1825_, v___x_1828_);
v___x_1830_ = lean_array_uget_borrowed(v_x_1809_, v___x_1829_);
lean_inc(v___x_1830_);
if (v_isShared_1816_ == 0)
{
lean_ctor_set(v___x_1815_, 2, v___x_1830_);
v___x_1832_ = v___x_1815_;
goto v_reusejp_1831_;
}
else
{
lean_object* v_reuseFailAlloc_1835_; 
v_reuseFailAlloc_1835_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1835_, 0, v_key_1811_);
lean_ctor_set(v_reuseFailAlloc_1835_, 1, v_value_1812_);
lean_ctor_set(v_reuseFailAlloc_1835_, 2, v___x_1830_);
v___x_1832_ = v_reuseFailAlloc_1835_;
goto v_reusejp_1831_;
}
v_reusejp_1831_:
{
lean_object* v___x_1833_; 
v___x_1833_ = lean_array_uset(v_x_1809_, v___x_1829_, v___x_1832_);
v_x_1809_ = v___x_1833_;
v_x_1810_ = v_tail_1813_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3___redArg(lean_object* v_i_1837_, lean_object* v_source_1838_, lean_object* v_target_1839_){
_start:
{
lean_object* v___x_1840_; uint8_t v___x_1841_; 
v___x_1840_ = lean_array_get_size(v_source_1838_);
v___x_1841_ = lean_nat_dec_lt(v_i_1837_, v___x_1840_);
if (v___x_1841_ == 0)
{
lean_dec_ref(v_source_1838_);
lean_dec(v_i_1837_);
return v_target_1839_;
}
else
{
lean_object* v_es_1842_; lean_object* v___x_1843_; lean_object* v_source_1844_; lean_object* v_target_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; 
v_es_1842_ = lean_array_fget(v_source_1838_, v_i_1837_);
v___x_1843_ = lean_box(0);
v_source_1844_ = lean_array_fset(v_source_1838_, v_i_1837_, v___x_1843_);
v_target_1845_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3_spec__5___redArg(v_target_1839_, v_es_1842_);
v___x_1846_ = lean_unsigned_to_nat(1u);
v___x_1847_ = lean_nat_add(v_i_1837_, v___x_1846_);
lean_dec(v_i_1837_);
v_i_1837_ = v___x_1847_;
v_source_1838_ = v_source_1844_;
v_target_1839_ = v_target_1845_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2___redArg(lean_object* v_data_1849_){
_start:
{
lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v_nbuckets_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; 
v___x_1850_ = lean_array_get_size(v_data_1849_);
v___x_1851_ = lean_unsigned_to_nat(2u);
v_nbuckets_1852_ = lean_nat_mul(v___x_1850_, v___x_1851_);
v___x_1853_ = lean_unsigned_to_nat(0u);
v___x_1854_ = lean_box(0);
v___x_1855_ = lean_mk_array(v_nbuckets_1852_, v___x_1854_);
v___x_1856_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3___redArg(v___x_1853_, v_data_1849_, v___x_1855_);
return v___x_1856_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg(lean_object* v_a_1857_, lean_object* v_x_1858_){
_start:
{
if (lean_obj_tag(v_x_1858_) == 0)
{
uint8_t v___x_1859_; 
v___x_1859_ = 0;
return v___x_1859_;
}
else
{
lean_object* v_key_1860_; lean_object* v_tail_1861_; uint8_t v___x_1862_; 
v_key_1860_ = lean_ctor_get(v_x_1858_, 0);
v_tail_1861_ = lean_ctor_get(v_x_1858_, 2);
v___x_1862_ = l_Lean_instBEqImport_beq(v_key_1860_, v_a_1857_);
if (v___x_1862_ == 0)
{
v_x_1858_ = v_tail_1861_;
goto _start;
}
else
{
return v___x_1862_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg___boxed(lean_object* v_a_1864_, lean_object* v_x_1865_){
_start:
{
uint8_t v_res_1866_; lean_object* v_r_1867_; 
v_res_1866_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg(v_a_1864_, v_x_1865_);
lean_dec(v_x_1865_);
lean_dec_ref(v_a_1864_);
v_r_1867_ = lean_box(v_res_1866_);
return v_r_1867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1___redArg(lean_object* v_m_1868_, lean_object* v_a_1869_, lean_object* v_b_1870_){
_start:
{
lean_object* v_size_1871_; lean_object* v_buckets_1872_; lean_object* v___x_1873_; uint64_t v___x_1874_; uint64_t v___x_1875_; uint64_t v___x_1876_; uint64_t v_fold_1877_; uint64_t v___x_1878_; uint64_t v___x_1879_; uint64_t v___x_1880_; size_t v___x_1881_; size_t v___x_1882_; size_t v___x_1883_; size_t v___x_1884_; size_t v___x_1885_; lean_object* v_bkt_1886_; uint8_t v___x_1887_; 
v_size_1871_ = lean_ctor_get(v_m_1868_, 0);
v_buckets_1872_ = lean_ctor_get(v_m_1868_, 1);
v___x_1873_ = lean_array_get_size(v_buckets_1872_);
v___x_1874_ = l_Lean_instHashableImport_hash(v_a_1869_);
v___x_1875_ = 32ULL;
v___x_1876_ = lean_uint64_shift_right(v___x_1874_, v___x_1875_);
v_fold_1877_ = lean_uint64_xor(v___x_1874_, v___x_1876_);
v___x_1878_ = 16ULL;
v___x_1879_ = lean_uint64_shift_right(v_fold_1877_, v___x_1878_);
v___x_1880_ = lean_uint64_xor(v_fold_1877_, v___x_1879_);
v___x_1881_ = lean_uint64_to_usize(v___x_1880_);
v___x_1882_ = lean_usize_of_nat(v___x_1873_);
v___x_1883_ = ((size_t)1ULL);
v___x_1884_ = lean_usize_sub(v___x_1882_, v___x_1883_);
v___x_1885_ = lean_usize_land(v___x_1881_, v___x_1884_);
v_bkt_1886_ = lean_array_uget_borrowed(v_buckets_1872_, v___x_1885_);
v___x_1887_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg(v_a_1869_, v_bkt_1886_);
if (v___x_1887_ == 0)
{
lean_object* v___x_1889_; uint8_t v_isShared_1890_; uint8_t v_isSharedCheck_1908_; 
lean_inc_ref(v_buckets_1872_);
lean_inc(v_size_1871_);
v_isSharedCheck_1908_ = !lean_is_exclusive(v_m_1868_);
if (v_isSharedCheck_1908_ == 0)
{
lean_object* v_unused_1909_; lean_object* v_unused_1910_; 
v_unused_1909_ = lean_ctor_get(v_m_1868_, 1);
lean_dec(v_unused_1909_);
v_unused_1910_ = lean_ctor_get(v_m_1868_, 0);
lean_dec(v_unused_1910_);
v___x_1889_ = v_m_1868_;
v_isShared_1890_ = v_isSharedCheck_1908_;
goto v_resetjp_1888_;
}
else
{
lean_dec(v_m_1868_);
v___x_1889_ = lean_box(0);
v_isShared_1890_ = v_isSharedCheck_1908_;
goto v_resetjp_1888_;
}
v_resetjp_1888_:
{
lean_object* v___x_1891_; lean_object* v_size_x27_1892_; lean_object* v___x_1893_; lean_object* v_buckets_x27_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; uint8_t v___x_1900_; 
v___x_1891_ = lean_unsigned_to_nat(1u);
v_size_x27_1892_ = lean_nat_add(v_size_1871_, v___x_1891_);
lean_dec(v_size_1871_);
lean_inc(v_bkt_1886_);
v___x_1893_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1893_, 0, v_a_1869_);
lean_ctor_set(v___x_1893_, 1, v_b_1870_);
lean_ctor_set(v___x_1893_, 2, v_bkt_1886_);
v_buckets_x27_1894_ = lean_array_uset(v_buckets_1872_, v___x_1885_, v___x_1893_);
v___x_1895_ = lean_unsigned_to_nat(4u);
v___x_1896_ = lean_nat_mul(v_size_x27_1892_, v___x_1895_);
v___x_1897_ = lean_unsigned_to_nat(3u);
v___x_1898_ = lean_nat_div(v___x_1896_, v___x_1897_);
lean_dec(v___x_1896_);
v___x_1899_ = lean_array_get_size(v_buckets_x27_1894_);
v___x_1900_ = lean_nat_dec_le(v___x_1898_, v___x_1899_);
lean_dec(v___x_1898_);
if (v___x_1900_ == 0)
{
lean_object* v_val_1901_; lean_object* v___x_1903_; 
v_val_1901_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2___redArg(v_buckets_x27_1894_);
if (v_isShared_1890_ == 0)
{
lean_ctor_set(v___x_1889_, 1, v_val_1901_);
lean_ctor_set(v___x_1889_, 0, v_size_x27_1892_);
v___x_1903_ = v___x_1889_;
goto v_reusejp_1902_;
}
else
{
lean_object* v_reuseFailAlloc_1904_; 
v_reuseFailAlloc_1904_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1904_, 0, v_size_x27_1892_);
lean_ctor_set(v_reuseFailAlloc_1904_, 1, v_val_1901_);
v___x_1903_ = v_reuseFailAlloc_1904_;
goto v_reusejp_1902_;
}
v_reusejp_1902_:
{
return v___x_1903_;
}
}
else
{
lean_object* v___x_1906_; 
if (v_isShared_1890_ == 0)
{
lean_ctor_set(v___x_1889_, 1, v_buckets_x27_1894_);
lean_ctor_set(v___x_1889_, 0, v_size_x27_1892_);
v___x_1906_ = v___x_1889_;
goto v_reusejp_1905_;
}
else
{
lean_object* v_reuseFailAlloc_1907_; 
v_reuseFailAlloc_1907_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1907_, 0, v_size_x27_1892_);
lean_ctor_set(v_reuseFailAlloc_1907_, 1, v_buckets_x27_1894_);
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
else
{
lean_dec(v_b_1870_);
lean_dec_ref(v_a_1869_);
return v_m_1868_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg(lean_object* v_m_1911_, lean_object* v_a_1912_){
_start:
{
lean_object* v_buckets_1913_; lean_object* v___x_1914_; uint64_t v___x_1915_; uint64_t v___x_1916_; uint64_t v___x_1917_; uint64_t v_fold_1918_; uint64_t v___x_1919_; uint64_t v___x_1920_; uint64_t v___x_1921_; size_t v___x_1922_; size_t v___x_1923_; size_t v___x_1924_; size_t v___x_1925_; size_t v___x_1926_; lean_object* v___x_1927_; uint8_t v___x_1928_; 
v_buckets_1913_ = lean_ctor_get(v_m_1911_, 1);
v___x_1914_ = lean_array_get_size(v_buckets_1913_);
v___x_1915_ = l_Lean_instHashableImport_hash(v_a_1912_);
v___x_1916_ = 32ULL;
v___x_1917_ = lean_uint64_shift_right(v___x_1915_, v___x_1916_);
v_fold_1918_ = lean_uint64_xor(v___x_1915_, v___x_1917_);
v___x_1919_ = 16ULL;
v___x_1920_ = lean_uint64_shift_right(v_fold_1918_, v___x_1919_);
v___x_1921_ = lean_uint64_xor(v_fold_1918_, v___x_1920_);
v___x_1922_ = lean_uint64_to_usize(v___x_1921_);
v___x_1923_ = lean_usize_of_nat(v___x_1914_);
v___x_1924_ = ((size_t)1ULL);
v___x_1925_ = lean_usize_sub(v___x_1923_, v___x_1924_);
v___x_1926_ = lean_usize_land(v___x_1922_, v___x_1925_);
v___x_1927_ = lean_array_uget_borrowed(v_buckets_1913_, v___x_1926_);
v___x_1928_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg(v_a_1912_, v___x_1927_);
return v___x_1928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg___boxed(lean_object* v_m_1929_, lean_object* v_a_1930_){
_start:
{
uint8_t v_res_1931_; lean_object* v_r_1932_; 
v_res_1931_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg(v_m_1929_, v_a_1930_);
lean_dec_ref(v_a_1930_);
lean_dec_ref(v_m_1929_);
v_r_1932_ = lean_box(v_res_1931_);
return v_r_1932_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1934_; lean_object* v___x_1935_; 
v___x_1934_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__0));
v___x_1935_ = l_Lean_stringToMessageData(v___x_1934_);
return v___x_1935_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__3(void){
_start:
{
lean_object* v___x_1937_; lean_object* v___x_1938_; 
v___x_1937_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__2));
v___x_1938_ = l_Lean_stringToMessageData(v___x_1937_);
return v___x_1938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2(lean_object* v_as_1939_, size_t v_sz_1940_, size_t v_i_1941_, lean_object* v_b_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_){
_start:
{
lean_object* v_a_1947_; uint8_t v___x_1951_; 
v___x_1951_ = lean_usize_dec_lt(v_i_1941_, v_sz_1940_);
if (v___x_1951_ == 0)
{
lean_object* v___x_1952_; 
v___x_1952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1952_, 0, v_b_1942_);
return v___x_1952_;
}
else
{
lean_object* v_a_1953_; lean_object* v_toImport_1954_; uint8_t v___x_1955_; 
v_a_1953_ = lean_array_uget_borrowed(v_as_1939_, v_i_1941_);
v_toImport_1954_ = lean_ctor_get(v_a_1953_, 0);
v___x_1955_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg(v_b_1942_, v_toImport_1954_);
if (v___x_1955_ == 0)
{
lean_object* v___x_1956_; lean_object* v___x_1957_; 
v___x_1956_ = lean_box(0);
lean_inc_ref(v_toImport_1954_);
v___x_1957_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1___redArg(v_b_1942_, v_toImport_1954_, v___x_1956_);
v_a_1947_ = v___x_1957_;
goto v___jp_1946_;
}
else
{
lean_object* v_module_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; 
v_module_1958_ = lean_ctor_get(v_toImport_1954_, 0);
v___x_1959_ = lp_mathlib_Mathlib_Linter_linter_style_header;
lean_inc(v_a_1953_);
v___x_1960_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_ImportRef_getIdent(v_a_1953_);
v___x_1961_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__1);
lean_inc(v_module_1958_);
v___x_1962_ = l_Lean_MessageData_ofName(v_module_1958_);
v___x_1963_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1963_, 0, v___x_1961_);
lean_ctor_set(v___x_1963_, 1, v___x_1962_);
v___x_1964_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___closed__3);
v___x_1965_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1965_, 0, v___x_1963_);
lean_ctor_set(v___x_1965_, 1, v___x_1964_);
v___x_1966_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_1959_, v___x_1960_, v___x_1965_, v___y_1943_, v___y_1944_);
if (lean_obj_tag(v___x_1966_) == 0)
{
lean_dec_ref_known(v___x_1966_, 1);
v_a_1947_ = v_b_1942_;
goto v___jp_1946_;
}
else
{
lean_object* v_a_1967_; lean_object* v___x_1969_; uint8_t v_isShared_1970_; uint8_t v_isSharedCheck_1974_; 
lean_dec_ref(v_b_1942_);
v_a_1967_ = lean_ctor_get(v___x_1966_, 0);
v_isSharedCheck_1974_ = !lean_is_exclusive(v___x_1966_);
if (v_isSharedCheck_1974_ == 0)
{
v___x_1969_ = v___x_1966_;
v_isShared_1970_ = v_isSharedCheck_1974_;
goto v_resetjp_1968_;
}
else
{
lean_inc(v_a_1967_);
lean_dec(v___x_1966_);
v___x_1969_ = lean_box(0);
v_isShared_1970_ = v_isSharedCheck_1974_;
goto v_resetjp_1968_;
}
v_resetjp_1968_:
{
lean_object* v___x_1972_; 
if (v_isShared_1970_ == 0)
{
v___x_1972_ = v___x_1969_;
goto v_reusejp_1971_;
}
else
{
lean_object* v_reuseFailAlloc_1973_; 
v_reuseFailAlloc_1973_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1973_, 0, v_a_1967_);
v___x_1972_ = v_reuseFailAlloc_1973_;
goto v_reusejp_1971_;
}
v_reusejp_1971_:
{
return v___x_1972_;
}
}
}
}
}
v___jp_1946_:
{
size_t v___x_1948_; size_t v___x_1949_; 
v___x_1948_ = ((size_t)1ULL);
v___x_1949_ = lean_usize_add(v_i_1941_, v___x_1948_);
v_i_1941_ = v___x_1949_;
v_b_1942_ = v_a_1947_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2___boxed(lean_object* v_as_1975_, lean_object* v_sz_1976_, lean_object* v_i_1977_, lean_object* v_b_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
size_t v_sz_boxed_1982_; size_t v_i_boxed_1983_; lean_object* v_res_1984_; 
v_sz_boxed_1982_ = lean_unbox_usize(v_sz_1976_);
lean_dec(v_sz_1976_);
v_i_boxed_1983_ = lean_unbox_usize(v_i_1977_);
lean_dec(v_i_1977_);
v_res_1984_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2(v_as_1975_, v_sz_boxed_1982_, v_i_boxed_1983_, v_b_1978_, v___y_1979_, v___y_1980_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
lean_dec_ref(v_as_1975_);
return v_res_1984_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__0(void){
_start:
{
lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; 
v___x_1985_ = lean_box(0);
v___x_1986_ = lean_unsigned_to_nat(16u);
v___x_1987_ = lean_mk_array(v___x_1986_, v___x_1985_);
return v___x_1987_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__1(void){
_start:
{
lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v_importsSoFar_1990_; 
v___x_1988_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__0, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__0);
v___x_1989_ = lean_unsigned_to_nat(0u);
v_importsSoFar_1990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_importsSoFar_1990_, 0, v___x_1989_);
lean_ctor_set(v_importsSoFar_1990_, 1, v___x_1988_);
return v_importsSoFar_1990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck(lean_object* v_imports_1991_, lean_object* v_a_1992_, lean_object* v_a_1993_){
_start:
{
lean_object* v_importsSoFar_1995_; size_t v_sz_1996_; size_t v___x_1997_; lean_object* v___x_1998_; 
v_importsSoFar_1995_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___closed__1);
v_sz_1996_ = lean_array_size(v_imports_1991_);
v___x_1997_ = ((size_t)0ULL);
v___x_1998_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__2(v_imports_1991_, v_sz_1996_, v___x_1997_, v_importsSoFar_1995_, v_a_1992_, v_a_1993_);
if (lean_obj_tag(v___x_1998_) == 0)
{
lean_object* v___x_2000_; uint8_t v_isShared_2001_; uint8_t v_isSharedCheck_2006_; 
v_isSharedCheck_2006_ = !lean_is_exclusive(v___x_1998_);
if (v_isSharedCheck_2006_ == 0)
{
lean_object* v_unused_2007_; 
v_unused_2007_ = lean_ctor_get(v___x_1998_, 0);
lean_dec(v_unused_2007_);
v___x_2000_ = v___x_1998_;
v_isShared_2001_ = v_isSharedCheck_2006_;
goto v_resetjp_1999_;
}
else
{
lean_dec(v___x_1998_);
v___x_2000_ = lean_box(0);
v_isShared_2001_ = v_isSharedCheck_2006_;
goto v_resetjp_1999_;
}
v_resetjp_1999_:
{
lean_object* v___x_2002_; lean_object* v___x_2004_; 
v___x_2002_ = lean_box(0);
if (v_isShared_2001_ == 0)
{
lean_ctor_set(v___x_2000_, 0, v___x_2002_);
v___x_2004_ = v___x_2000_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2005_; 
v_reuseFailAlloc_2005_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2005_, 0, v___x_2002_);
v___x_2004_ = v_reuseFailAlloc_2005_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
return v___x_2004_;
}
}
}
else
{
lean_object* v_a_2008_; lean_object* v___x_2010_; uint8_t v_isShared_2011_; uint8_t v_isSharedCheck_2015_; 
v_a_2008_ = lean_ctor_get(v___x_1998_, 0);
v_isSharedCheck_2015_ = !lean_is_exclusive(v___x_1998_);
if (v_isSharedCheck_2015_ == 0)
{
v___x_2010_ = v___x_1998_;
v_isShared_2011_ = v_isSharedCheck_2015_;
goto v_resetjp_2009_;
}
else
{
lean_inc(v_a_2008_);
lean_dec(v___x_1998_);
v___x_2010_ = lean_box(0);
v_isShared_2011_ = v_isSharedCheck_2015_;
goto v_resetjp_2009_;
}
v_resetjp_2009_:
{
lean_object* v___x_2013_; 
if (v_isShared_2011_ == 0)
{
v___x_2013_ = v___x_2010_;
goto v_reusejp_2012_;
}
else
{
lean_object* v_reuseFailAlloc_2014_; 
v_reuseFailAlloc_2014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2014_, 0, v_a_2008_);
v___x_2013_ = v_reuseFailAlloc_2014_;
goto v_reusejp_2012_;
}
v_reusejp_2012_:
{
return v___x_2013_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck___boxed(lean_object* v_imports_2016_, lean_object* v_a_2017_, lean_object* v_a_2018_, lean_object* v_a_2019_){
_start:
{
lean_object* v_res_2020_; 
v_res_2020_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck(v_imports_2016_, v_a_2017_, v_a_2018_);
lean_dec(v_a_2018_);
lean_dec_ref(v_a_2017_);
lean_dec_ref(v_imports_2016_);
return v_res_2020_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0(lean_object* v_00_u03b2_2021_, lean_object* v_m_2022_, lean_object* v_a_2023_){
_start:
{
uint8_t v___x_2024_; 
v___x_2024_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___redArg(v_m_2022_, v_a_2023_);
return v___x_2024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0___boxed(lean_object* v_00_u03b2_2025_, lean_object* v_m_2026_, lean_object* v_a_2027_){
_start:
{
uint8_t v_res_2028_; lean_object* v_r_2029_; 
v_res_2028_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0(v_00_u03b2_2025_, v_m_2026_, v_a_2027_);
lean_dec_ref(v_a_2027_);
lean_dec_ref(v_m_2026_);
v_r_2029_ = lean_box(v_res_2028_);
return v_r_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1(lean_object* v_00_u03b2_2030_, lean_object* v_m_2031_, lean_object* v_a_2032_, lean_object* v_b_2033_){
_start:
{
lean_object* v___x_2034_; 
v___x_2034_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1___redArg(v_m_2031_, v_a_2032_, v_b_2033_);
return v___x_2034_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0(lean_object* v_00_u03b2_2035_, lean_object* v_a_2036_, lean_object* v_x_2037_){
_start:
{
uint8_t v___x_2038_; 
v___x_2038_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___redArg(v_a_2036_, v_x_2037_);
return v___x_2038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0___boxed(lean_object* v_00_u03b2_2039_, lean_object* v_a_2040_, lean_object* v_x_2041_){
_start:
{
uint8_t v_res_2042_; lean_object* v_r_2043_; 
v_res_2042_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__0_spec__0(v_00_u03b2_2039_, v_a_2040_, v_x_2041_);
lean_dec(v_x_2041_);
lean_dec_ref(v_a_2040_);
v_r_2043_ = lean_box(v_res_2042_);
return v_r_2043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2(lean_object* v_00_u03b2_2044_, lean_object* v_data_2045_){
_start:
{
lean_object* v___x_2046_; 
v___x_2046_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2___redArg(v_data_2045_);
return v___x_2046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_2047_, lean_object* v_i_2048_, lean_object* v_source_2049_, lean_object* v_target_2050_){
_start:
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3___redArg(v_i_2048_, v_source_2049_, v_target_2050_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_2052_, lean_object* v_x_2053_, lean_object* v_x_2054_){
_start:
{
lean_object* v___x_2055_; 
v___x_2055_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck_spec__1_spec__2_spec__3_spec__5___redArg(v_x_2053_, v_x_2054_);
return v___x_2055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg(lean_object* v___y_2056_){
_start:
{
lean_object* v___x_2058_; lean_object* v_env_2059_; lean_object* v___x_2060_; lean_object* v_mainModule_2061_; lean_object* v___x_2062_; 
v___x_2058_ = lean_st_ref_get(v___y_2056_);
v_env_2059_ = lean_ctor_get(v___x_2058_, 0);
lean_inc_ref(v_env_2059_);
lean_dec(v___x_2058_);
v___x_2060_ = l_Lean_Environment_header(v_env_2059_);
lean_dec_ref(v_env_2059_);
v_mainModule_2061_ = lean_ctor_get(v___x_2060_, 0);
lean_inc(v_mainModule_2061_);
lean_dec_ref(v___x_2060_);
v___x_2062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2062_, 0, v_mainModule_2061_);
return v___x_2062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg___boxed(lean_object* v___y_2063_, lean_object* v___y_2064_){
_start:
{
lean_object* v_res_2065_; 
v_res_2065_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg(v___y_2063_);
lean_dec(v___y_2063_);
return v_res_2065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1(lean_object* v___y_2066_, lean_object* v___y_2067_){
_start:
{
lean_object* v___x_2069_; 
v___x_2069_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg(v___y_2067_);
return v___x_2069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___boxed(lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_){
_start:
{
lean_object* v_res_2073_; 
v_res_2073_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1(v___y_2070_, v___y_2071_);
lean_dec(v___y_2071_);
lean_dec_ref(v___y_2070_);
return v_res_2073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0(lean_object* v_mutex_2074_, lean_object* v_a_x3f_2075_){
_start:
{
lean_object* v___x_2077_; lean_object* v___x_2078_; 
v___x_2077_ = lean_io_basemutex_unlock(v_mutex_2074_);
v___x_2078_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2078_, 0, v___x_2077_);
return v___x_2078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0___boxed(lean_object* v_mutex_2079_, lean_object* v_a_x3f_2080_, lean_object* v___y_2081_){
_start:
{
lean_object* v_res_2082_; 
v_res_2082_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0(v_mutex_2079_, v_a_x3f_2080_);
lean_dec(v_a_x3f_2080_);
lean_dec(v_mutex_2079_);
return v_res_2082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg(lean_object* v_mutex_2083_, lean_object* v_k_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_){
_start:
{
lean_object* v_ref_2088_; lean_object* v_mutex_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; 
v_ref_2088_ = lean_ctor_get(v_mutex_2083_, 0);
lean_inc(v_ref_2088_);
v_mutex_2089_ = lean_ctor_get(v_mutex_2083_, 1);
lean_inc(v_mutex_2089_);
lean_dec_ref(v_mutex_2083_);
v___x_2090_ = lean_io_basemutex_lock(v_mutex_2089_);
lean_inc(v___y_2086_);
lean_inc_ref(v___y_2085_);
v___x_2091_ = lean_apply_4(v_k_2084_, v_ref_2088_, v___y_2085_, v___y_2086_, lean_box(0));
if (lean_obj_tag(v___x_2091_) == 0)
{
lean_object* v_a_2092_; lean_object* v___x_2094_; uint8_t v_isShared_2095_; uint8_t v_isSharedCheck_2108_; 
v_a_2092_ = lean_ctor_get(v___x_2091_, 0);
v_isSharedCheck_2108_ = !lean_is_exclusive(v___x_2091_);
if (v_isSharedCheck_2108_ == 0)
{
v___x_2094_ = v___x_2091_;
v_isShared_2095_ = v_isSharedCheck_2108_;
goto v_resetjp_2093_;
}
else
{
lean_inc(v_a_2092_);
lean_dec(v___x_2091_);
v___x_2094_ = lean_box(0);
v_isShared_2095_ = v_isSharedCheck_2108_;
goto v_resetjp_2093_;
}
v_resetjp_2093_:
{
lean_object* v___x_2097_; 
lean_inc(v_a_2092_);
if (v_isShared_2095_ == 0)
{
lean_ctor_set_tag(v___x_2094_, 1);
v___x_2097_ = v___x_2094_;
goto v_reusejp_2096_;
}
else
{
lean_object* v_reuseFailAlloc_2107_; 
v_reuseFailAlloc_2107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2107_, 0, v_a_2092_);
v___x_2097_ = v_reuseFailAlloc_2107_;
goto v_reusejp_2096_;
}
v_reusejp_2096_:
{
lean_object* v___x_2098_; lean_object* v___x_2100_; uint8_t v_isShared_2101_; uint8_t v_isSharedCheck_2105_; 
v___x_2098_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0(v_mutex_2089_, v___x_2097_);
lean_dec_ref(v___x_2097_);
lean_dec(v_mutex_2089_);
v_isSharedCheck_2105_ = !lean_is_exclusive(v___x_2098_);
if (v_isSharedCheck_2105_ == 0)
{
lean_object* v_unused_2106_; 
v_unused_2106_ = lean_ctor_get(v___x_2098_, 0);
lean_dec(v_unused_2106_);
v___x_2100_ = v___x_2098_;
v_isShared_2101_ = v_isSharedCheck_2105_;
goto v_resetjp_2099_;
}
else
{
lean_dec(v___x_2098_);
v___x_2100_ = lean_box(0);
v_isShared_2101_ = v_isSharedCheck_2105_;
goto v_resetjp_2099_;
}
v_resetjp_2099_:
{
lean_object* v___x_2103_; 
if (v_isShared_2101_ == 0)
{
lean_ctor_set(v___x_2100_, 0, v_a_2092_);
v___x_2103_ = v___x_2100_;
goto v_reusejp_2102_;
}
else
{
lean_object* v_reuseFailAlloc_2104_; 
v_reuseFailAlloc_2104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2104_, 0, v_a_2092_);
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
}
else
{
lean_object* v_a_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2113_; uint8_t v_isShared_2114_; uint8_t v_isSharedCheck_2118_; 
v_a_2109_ = lean_ctor_get(v___x_2091_, 0);
lean_inc(v_a_2109_);
lean_dec_ref_known(v___x_2091_, 1);
v___x_2110_ = lean_box(0);
v___x_2111_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___lam__0(v_mutex_2089_, v___x_2110_);
lean_dec(v_mutex_2089_);
v_isSharedCheck_2118_ = !lean_is_exclusive(v___x_2111_);
if (v_isSharedCheck_2118_ == 0)
{
lean_object* v_unused_2119_; 
v_unused_2119_ = lean_ctor_get(v___x_2111_, 0);
lean_dec(v_unused_2119_);
v___x_2113_ = v___x_2111_;
v_isShared_2114_ = v_isSharedCheck_2118_;
goto v_resetjp_2112_;
}
else
{
lean_dec(v___x_2111_);
v___x_2113_ = lean_box(0);
v_isShared_2114_ = v_isSharedCheck_2118_;
goto v_resetjp_2112_;
}
v_resetjp_2112_:
{
lean_object* v___x_2116_; 
if (v_isShared_2114_ == 0)
{
lean_ctor_set_tag(v___x_2113_, 1);
lean_ctor_set(v___x_2113_, 0, v_a_2109_);
v___x_2116_ = v___x_2113_;
goto v_reusejp_2115_;
}
else
{
lean_object* v_reuseFailAlloc_2117_; 
v_reuseFailAlloc_2117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2117_, 0, v_a_2109_);
v___x_2116_ = v_reuseFailAlloc_2117_;
goto v_reusejp_2115_;
}
v_reusejp_2115_:
{
return v___x_2116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg___boxed(lean_object* v_mutex_2120_, lean_object* v_k_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_){
_start:
{
lean_object* v_res_2125_; 
v_res_2125_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg(v_mutex_2120_, v_k_2121_, v___y_2122_, v___y_2123_);
lean_dec(v___y_2123_);
lean_dec_ref(v___y_2122_);
return v_res_2125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2(lean_object* v_00_u03b1_2126_, lean_object* v_00_u03b2_2127_, lean_object* v_mutex_2128_, lean_object* v_k_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_){
_start:
{
lean_object* v___x_2133_; 
v___x_2133_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg(v_mutex_2128_, v_k_2129_, v___y_2130_, v___y_2131_);
return v___x_2133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___boxed(lean_object* v_00_u03b1_2134_, lean_object* v_00_u03b2_2135_, lean_object* v_mutex_2136_, lean_object* v_k_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_){
_start:
{
lean_object* v_res_2141_; 
v_res_2141_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2(v_00_u03b1_2134_, v_00_u03b2_2135_, v_mutex_2136_, v_k_2137_, v___y_2138_, v___y_2139_);
lean_dec(v___y_2139_);
lean_dec_ref(v___y_2138_);
return v_res_2141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__3(lean_object* v_opts_2142_, lean_object* v_opt_2143_){
_start:
{
lean_object* v_name_2144_; lean_object* v_defValue_2145_; lean_object* v_map_2146_; lean_object* v___x_2147_; 
v_name_2144_ = lean_ctor_get(v_opt_2143_, 0);
v_defValue_2145_ = lean_ctor_get(v_opt_2143_, 1);
v_map_2146_ = lean_ctor_get(v_opts_2142_, 0);
v___x_2147_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2146_, v_name_2144_);
if (lean_obj_tag(v___x_2147_) == 0)
{
lean_inc(v_defValue_2145_);
return v_defValue_2145_;
}
else
{
lean_object* v_val_2148_; 
v_val_2148_ = lean_ctor_get(v___x_2147_, 0);
lean_inc(v_val_2148_);
lean_dec_ref_known(v___x_2147_, 1);
if (lean_obj_tag(v_val_2148_) == 0)
{
lean_object* v_v_2149_; 
v_v_2149_ = lean_ctor_get(v_val_2148_, 0);
lean_inc_ref(v_v_2149_);
lean_dec_ref_known(v_val_2148_, 1);
return v_v_2149_;
}
else
{
lean_dec(v_val_2148_);
lean_inc(v_defValue_2145_);
return v_defValue_2145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__3___boxed(lean_object* v_opts_2150_, lean_object* v_opt_2151_){
_start:
{
lean_object* v_res_2152_; 
v_res_2152_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__3(v_opts_2150_, v_opt_2151_);
lean_dec_ref(v_opt_2151_);
lean_dec_ref(v_opts_2150_);
return v_res_2152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__0(lean_object* v_a_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_){
_start:
{
lean_object* v___x_2158_; 
v___x_2158_ = lean_st_ref_get(v___y_2154_);
if (lean_obj_tag(v___x_2158_) == 0)
{
lean_object* v___x_2159_; 
v___x_2159_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_isInLibraryRoot(v_a_2153_);
if (lean_obj_tag(v___x_2159_) == 0)
{
lean_object* v_a_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2169_; 
v_a_2160_ = lean_ctor_get(v___x_2159_, 0);
v_isSharedCheck_2169_ = !lean_is_exclusive(v___x_2159_);
if (v_isSharedCheck_2169_ == 0)
{
v___x_2162_ = v___x_2159_;
v_isShared_2163_ = v_isSharedCheck_2169_;
goto v_resetjp_2161_;
}
else
{
lean_inc(v_a_2160_);
lean_dec(v___x_2159_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2169_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2167_; 
lean_inc(v_a_2160_);
v___x_2164_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2164_, 0, v_a_2160_);
v___x_2165_ = lean_st_ref_set(v___y_2154_, v___x_2164_);
if (v_isShared_2163_ == 0)
{
v___x_2167_ = v___x_2162_;
goto v_reusejp_2166_;
}
else
{
lean_object* v_reuseFailAlloc_2168_; 
v_reuseFailAlloc_2168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2168_, 0, v_a_2160_);
v___x_2167_ = v_reuseFailAlloc_2168_;
goto v_reusejp_2166_;
}
v_reusejp_2166_:
{
return v___x_2167_;
}
}
}
else
{
lean_object* v_a_2170_; lean_object* v___x_2172_; uint8_t v_isShared_2173_; uint8_t v_isSharedCheck_2182_; 
v_a_2170_ = lean_ctor_get(v___x_2159_, 0);
v_isSharedCheck_2182_ = !lean_is_exclusive(v___x_2159_);
if (v_isSharedCheck_2182_ == 0)
{
v___x_2172_ = v___x_2159_;
v_isShared_2173_ = v_isSharedCheck_2182_;
goto v_resetjp_2171_;
}
else
{
lean_inc(v_a_2170_);
lean_dec(v___x_2159_);
v___x_2172_ = lean_box(0);
v_isShared_2173_ = v_isSharedCheck_2182_;
goto v_resetjp_2171_;
}
v_resetjp_2171_:
{
lean_object* v_ref_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2180_; 
v_ref_2174_ = lean_ctor_get(v___y_2155_, 7);
v___x_2175_ = lean_io_error_to_string(v_a_2170_);
v___x_2176_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2176_, 0, v___x_2175_);
v___x_2177_ = l_Lean_MessageData_ofFormat(v___x_2176_);
lean_inc(v_ref_2174_);
v___x_2178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2178_, 0, v_ref_2174_);
lean_ctor_set(v___x_2178_, 1, v___x_2177_);
if (v_isShared_2173_ == 0)
{
lean_ctor_set(v___x_2172_, 0, v___x_2178_);
v___x_2180_ = v___x_2172_;
goto v_reusejp_2179_;
}
else
{
lean_object* v_reuseFailAlloc_2181_; 
v_reuseFailAlloc_2181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2181_, 0, v___x_2178_);
v___x_2180_ = v_reuseFailAlloc_2181_;
goto v_reusejp_2179_;
}
v_reusejp_2179_:
{
return v___x_2180_;
}
}
}
}
else
{
lean_object* v_val_2183_; lean_object* v___x_2185_; uint8_t v_isShared_2186_; uint8_t v_isSharedCheck_2190_; 
v_val_2183_ = lean_ctor_get(v___x_2158_, 0);
v_isSharedCheck_2190_ = !lean_is_exclusive(v___x_2158_);
if (v_isSharedCheck_2190_ == 0)
{
v___x_2185_ = v___x_2158_;
v_isShared_2186_ = v_isSharedCheck_2190_;
goto v_resetjp_2184_;
}
else
{
lean_inc(v_val_2183_);
lean_dec(v___x_2158_);
v___x_2185_ = lean_box(0);
v_isShared_2186_ = v_isSharedCheck_2190_;
goto v_resetjp_2184_;
}
v_resetjp_2184_:
{
lean_object* v___x_2188_; 
if (v_isShared_2186_ == 0)
{
lean_ctor_set_tag(v___x_2185_, 0);
v___x_2188_ = v___x_2185_;
goto v_reusejp_2187_;
}
else
{
lean_object* v_reuseFailAlloc_2189_; 
v_reuseFailAlloc_2189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2189_, 0, v_val_2183_);
v___x_2188_ = v_reuseFailAlloc_2189_;
goto v_reusejp_2187_;
}
v_reusejp_2187_:
{
return v___x_2188_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__0___boxed(lean_object* v_a_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_){
_start:
{
lean_object* v_res_2196_; 
v_res_2196_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__0(v_a_2191_, v___y_2192_, v___y_2193_, v___y_2194_);
lean_dec(v___y_2194_);
lean_dec_ref(v___y_2193_);
lean_dec(v___y_2192_);
lean_dec(v_a_2191_);
return v_res_2196_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__1(void){
_start:
{
lean_object* v___x_2198_; lean_object* v___x_2199_; 
v___x_2198_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__0));
v___x_2199_ = l_Lean_stringToMessageData(v___x_2198_);
return v___x_2199_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__3(void){
_start:
{
lean_object* v___x_2201_; lean_object* v___x_2202_; 
v___x_2201_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__2));
v___x_2202_ = l_Lean_stringToMessageData(v___x_2201_);
return v___x_2202_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__4(void){
_start:
{
lean_object* v___x_2203_; lean_object* v___x_2204_; 
v___x_2203_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_copyrightHeaderChecks___closed__19));
v___x_2204_ = l_Lean_stringToMessageData(v___x_2203_);
return v___x_2204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4(lean_object* v_as_2205_, size_t v_sz_2206_, size_t v_i_2207_, lean_object* v_b_2208_, lean_object* v___y_2209_, lean_object* v___y_2210_){
_start:
{
uint8_t v___x_2212_; 
v___x_2212_ = lean_usize_dec_lt(v_i_2207_, v_sz_2206_);
if (v___x_2212_ == 0)
{
lean_object* v___x_2213_; 
v___x_2213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2213_, 0, v_b_2208_);
return v___x_2213_;
}
else
{
lean_object* v_a_2214_; lean_object* v_fst_2215_; lean_object* v_snd_2216_; lean_object* v___x_2218_; uint8_t v_isShared_2219_; uint8_t v_isSharedCheck_2238_; 
v_a_2214_ = lean_array_uget(v_as_2205_, v_i_2207_);
v_fst_2215_ = lean_ctor_get(v_a_2214_, 0);
v_snd_2216_ = lean_ctor_get(v_a_2214_, 1);
v_isSharedCheck_2238_ = !lean_is_exclusive(v_a_2214_);
if (v_isSharedCheck_2238_ == 0)
{
v___x_2218_ = v_a_2214_;
v_isShared_2219_ = v_isSharedCheck_2238_;
goto v_resetjp_2217_;
}
else
{
lean_inc(v_snd_2216_);
lean_inc(v_fst_2215_);
lean_dec(v_a_2214_);
v___x_2218_ = lean_box(0);
v_isShared_2219_ = v_isSharedCheck_2238_;
goto v_resetjp_2217_;
}
v_resetjp_2217_:
{
lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2225_; 
v___x_2220_ = lp_mathlib_Mathlib_Linter_linter_style_header;
v___x_2221_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__1);
v___x_2222_ = l_Lean_Syntax_getAtomVal(v_fst_2215_);
v___x_2223_ = l_Lean_stringToMessageData(v___x_2222_);
if (v_isShared_2219_ == 0)
{
lean_ctor_set_tag(v___x_2218_, 7);
lean_ctor_set(v___x_2218_, 1, v___x_2223_);
lean_ctor_set(v___x_2218_, 0, v___x_2221_);
v___x_2225_ = v___x_2218_;
goto v_reusejp_2224_;
}
else
{
lean_object* v_reuseFailAlloc_2237_; 
v_reuseFailAlloc_2237_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2237_, 0, v___x_2221_);
lean_ctor_set(v_reuseFailAlloc_2237_, 1, v___x_2223_);
v___x_2225_ = v_reuseFailAlloc_2237_;
goto v_reusejp_2224_;
}
v_reusejp_2224_:
{
lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; 
v___x_2226_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__3);
v___x_2227_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2227_, 0, v___x_2225_);
lean_ctor_set(v___x_2227_, 1, v___x_2226_);
v___x_2228_ = l_Lean_stringToMessageData(v_snd_2216_);
v___x_2229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2229_, 0, v___x_2227_);
lean_ctor_set(v___x_2229_, 1, v___x_2228_);
v___x_2230_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___closed__4);
v___x_2231_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2231_, 0, v___x_2229_);
lean_ctor_set(v___x_2231_, 1, v___x_2230_);
v___x_2232_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_2220_, v_fst_2215_, v___x_2231_, v___y_2209_, v___y_2210_);
if (lean_obj_tag(v___x_2232_) == 0)
{
lean_object* v___x_2233_; size_t v___x_2234_; size_t v___x_2235_; 
lean_dec_ref_known(v___x_2232_, 1);
v___x_2233_ = lean_box(0);
v___x_2234_ = ((size_t)1ULL);
v___x_2235_ = lean_usize_add(v_i_2207_, v___x_2234_);
v_i_2207_ = v___x_2235_;
v_b_2208_ = v___x_2233_;
goto _start;
}
else
{
return v___x_2232_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4___boxed(lean_object* v_as_2239_, lean_object* v_sz_2240_, lean_object* v_i_2241_, lean_object* v_b_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_){
_start:
{
size_t v_sz_boxed_2246_; size_t v_i_boxed_2247_; lean_object* v_res_2248_; 
v_sz_boxed_2246_ = lean_unbox_usize(v_sz_2240_);
lean_dec(v_sz_2240_);
v_i_boxed_2247_ = lean_unbox_usize(v_i_2241_);
lean_dec(v_i_2241_);
v_res_2248_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4(v_as_2239_, v_sz_boxed_2246_, v_i_boxed_2247_, v_b_2242_, v___y_2243_, v___y_2244_);
lean_dec(v___y_2244_);
lean_dec_ref(v___y_2243_);
lean_dec_ref(v_as_2239_);
return v_res_2248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg(lean_object* v_o_2249_, lean_object* v___y_2250_){
_start:
{
lean_object* v___x_2252_; lean_object* v_env_2253_; lean_object* v___x_2254_; lean_object* v_toEnvExtension_2255_; lean_object* v_asyncMode_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v_merged_2260_; lean_object* v___x_2262_; uint8_t v_isShared_2263_; uint8_t v_isSharedCheck_2268_; 
v___x_2252_ = lean_st_ref_get(v___y_2250_);
v_env_2253_ = lean_ctor_get(v___x_2252_, 0);
lean_inc_ref(v_env_2253_);
lean_dec(v___x_2252_);
v___x_2254_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_2255_ = lean_ctor_get(v___x_2254_, 0);
v_asyncMode_2256_ = lean_ctor_get(v_toEnvExtension_2255_, 2);
v___x_2257_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_2258_ = lean_box(0);
v___x_2259_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_2257_, v___x_2254_, v_env_2253_, v_asyncMode_2256_, v___x_2258_);
v_merged_2260_ = lean_ctor_get(v___x_2259_, 0);
v_isSharedCheck_2268_ = !lean_is_exclusive(v___x_2259_);
if (v_isSharedCheck_2268_ == 0)
{
lean_object* v_unused_2269_; 
v_unused_2269_ = lean_ctor_get(v___x_2259_, 1);
lean_dec(v_unused_2269_);
v___x_2262_ = v___x_2259_;
v_isShared_2263_ = v_isSharedCheck_2268_;
goto v_resetjp_2261_;
}
else
{
lean_inc(v_merged_2260_);
lean_dec(v___x_2259_);
v___x_2262_ = lean_box(0);
v_isShared_2263_ = v_isSharedCheck_2268_;
goto v_resetjp_2261_;
}
v_resetjp_2261_:
{
lean_object* v___x_2265_; 
if (v_isShared_2263_ == 0)
{
lean_ctor_set(v___x_2262_, 1, v_merged_2260_);
lean_ctor_set(v___x_2262_, 0, v_o_2249_);
v___x_2265_ = v___x_2262_;
goto v_reusejp_2264_;
}
else
{
lean_object* v_reuseFailAlloc_2267_; 
v_reuseFailAlloc_2267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2267_, 0, v_o_2249_);
lean_ctor_set(v_reuseFailAlloc_2267_, 1, v_merged_2260_);
v___x_2265_ = v_reuseFailAlloc_2267_;
goto v_reusejp_2264_;
}
v_reusejp_2264_:
{
lean_object* v___x_2266_; 
v___x_2266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2266_, 0, v___x_2265_);
return v___x_2266_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_){
_start:
{
lean_object* v_res_2273_; 
v_res_2273_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg(v_o_2270_, v___y_2271_);
lean_dec(v___y_2271_);
return v_res_2273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0(lean_object* v___y_2274_, lean_object* v___y_2275_){
_start:
{
lean_object* v___x_2277_; lean_object* v_scopes_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v_opts_2281_; lean_object* v___x_2282_; 
v___x_2277_ = lean_st_ref_get(v___y_2275_);
v_scopes_2278_ = lean_ctor_get(v___x_2277_, 2);
lean_inc(v_scopes_2278_);
lean_dec(v___x_2277_);
v___x_2279_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2280_ = l_List_head_x21___redArg(v___x_2279_, v_scopes_2278_);
lean_dec(v_scopes_2278_);
v_opts_2281_ = lean_ctor_get(v___x_2280_, 1);
lean_inc_ref(v_opts_2281_);
lean_dec(v___x_2280_);
v___x_2282_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg(v_opts_2281_, v___y_2275_);
return v___x_2282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0___boxed(lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_){
_start:
{
lean_object* v_res_2286_; 
v_res_2286_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0(v___y_2283_, v___y_2284_);
lean_dec(v___y_2284_);
lean_dec_ref(v___y_2283_);
return v_res_2286_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__4(void){
_start:
{
lean_object* v___x_2294_; lean_object* v___x_2295_; 
v___x_2294_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__3));
v___x_2295_ = l_Lean_stringToMessageData(v___x_2294_);
return v___x_2295_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__9(void){
_start:
{
lean_object* v___x_2304_; lean_object* v___x_2305_; 
v___x_2304_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__8));
v___x_2305_ = l_Lean_stringToMessageData(v___x_2304_);
return v___x_2305_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__11(void){
_start:
{
lean_object* v___x_2307_; lean_object* v___x_2308_; 
v___x_2307_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__10));
v___x_2308_ = l_Lean_stringToMessageData(v___x_2307_);
return v___x_2308_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__13(void){
_start:
{
lean_object* v___x_2310_; lean_object* v___x_2311_; 
v___x_2310_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__12));
v___x_2311_ = l_Lean_stringToMessageData(v___x_2310_);
return v___x_2311_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__14(void){
_start:
{
lean_object* v___x_2312_; lean_object* v___x_2313_; 
v___x_2312_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__13);
v___x_2313_ = l_Lean_MessageData_hint_x27(v___x_2312_);
return v___x_2313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1(lean_object* v_stx_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_){
_start:
{
lean_object* v___y_2325_; lean_object* v___y_2326_; lean_object* v___y_2327_; lean_object* v___y_2328_; lean_object* v___y_2343_; lean_object* v___y_2344_; lean_object* v___y_2345_; lean_object* v___y_2361_; lean_object* v___y_2362_; lean_object* v___y_2363_; uint8_t v___y_2364_; lean_object* v___x_2367_; lean_object* v_a_2368_; lean_object* v___x_2370_; uint8_t v_isShared_2371_; uint8_t v_isSharedCheck_2540_; 
v___x_2367_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0(v___y_2321_, v___y_2322_);
v_a_2368_ = lean_ctor_get(v___x_2367_, 0);
v_isSharedCheck_2540_ = !lean_is_exclusive(v___x_2367_);
if (v_isSharedCheck_2540_ == 0)
{
v___x_2370_ = v___x_2367_;
v_isShared_2371_ = v_isSharedCheck_2540_;
goto v_resetjp_2369_;
}
else
{
lean_inc(v_a_2368_);
lean_dec(v___x_2367_);
v___x_2370_ = lean_box(0);
v_isShared_2371_ = v_isSharedCheck_2540_;
goto v_resetjp_2369_;
}
v___jp_2324_:
{
lean_object* v___x_2329_; lean_object* v___x_2330_; size_t v_sz_2331_; size_t v___x_2332_; lean_object* v___x_2333_; 
v___x_2329_ = lp_mathlib_Mathlib_Linter_copyrightHeaderChecks(v___y_2328_, v___y_2325_);
lean_dec_ref(v___y_2325_);
v___x_2330_ = lean_box(0);
v_sz_2331_ = lean_array_size(v___x_2329_);
v___x_2332_ = ((size_t)0ULL);
v___x_2333_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__4(v___x_2329_, v_sz_2331_, v___x_2332_, v___x_2330_, v___y_2327_, v___y_2326_);
lean_dec_ref(v___x_2329_);
if (lean_obj_tag(v___x_2333_) == 0)
{
lean_object* v___x_2335_; uint8_t v_isShared_2336_; uint8_t v_isSharedCheck_2340_; 
v_isSharedCheck_2340_ = !lean_is_exclusive(v___x_2333_);
if (v_isSharedCheck_2340_ == 0)
{
lean_object* v_unused_2341_; 
v_unused_2341_ = lean_ctor_get(v___x_2333_, 0);
lean_dec(v_unused_2341_);
v___x_2335_ = v___x_2333_;
v_isShared_2336_ = v_isSharedCheck_2340_;
goto v_resetjp_2334_;
}
else
{
lean_dec(v___x_2333_);
v___x_2335_ = lean_box(0);
v_isShared_2336_ = v_isSharedCheck_2340_;
goto v_resetjp_2334_;
}
v_resetjp_2334_:
{
lean_object* v___x_2338_; 
if (v_isShared_2336_ == 0)
{
lean_ctor_set(v___x_2335_, 0, v___x_2330_);
v___x_2338_ = v___x_2335_;
goto v_reusejp_2337_;
}
else
{
lean_object* v_reuseFailAlloc_2339_; 
v_reuseFailAlloc_2339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2339_, 0, v___x_2330_);
v___x_2338_ = v_reuseFailAlloc_2339_;
goto v_reusejp_2337_;
}
v_reusejp_2337_:
{
return v___x_2338_;
}
}
}
else
{
return v___x_2333_;
}
}
v___jp_2342_:
{
lean_object* v___x_2346_; lean_object* v_scopes_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v_opts_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; 
v___x_2346_ = lean_st_ref_get(v___y_2343_);
v_scopes_2347_ = lean_ctor_get(v___x_2346_, 2);
lean_inc(v_scopes_2347_);
lean_dec(v___x_2346_);
v___x_2348_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2349_ = l_List_head_x21___redArg(v___x_2348_, v_scopes_2347_);
lean_dec(v_scopes_2347_);
v_opts_2350_ = lean_ctor_get(v___x_2349_, 1);
lean_inc_ref(v_opts_2350_);
lean_dec(v___x_2349_);
v___x_2351_ = lp_mathlib_Mathlib_Linter_linter_style_header_license;
v___x_2352_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__3(v_opts_2350_, v___x_2351_);
lean_dec_ref(v_opts_2350_);
v___x_2353_ = l_Lean_Syntax_getHeadInfo(v___y_2345_);
lean_dec(v___y_2345_);
if (lean_obj_tag(v___x_2353_) == 0)
{
lean_object* v_leading_2354_; lean_object* v_str_2355_; lean_object* v_startPos_2356_; lean_object* v_stopPos_2357_; lean_object* v___x_2358_; 
v_leading_2354_ = lean_ctor_get(v___x_2353_, 0);
lean_inc_ref(v_leading_2354_);
lean_dec_ref_known(v___x_2353_, 4);
v_str_2355_ = lean_ctor_get(v_leading_2354_, 0);
lean_inc_ref(v_str_2355_);
v_startPos_2356_ = lean_ctor_get(v_leading_2354_, 1);
lean_inc(v_startPos_2356_);
v_stopPos_2357_ = lean_ctor_get(v_leading_2354_, 2);
lean_inc(v_stopPos_2357_);
lean_dec_ref(v_leading_2354_);
v___x_2358_ = lean_string_utf8_extract(v_str_2355_, v_startPos_2356_, v_stopPos_2357_);
lean_dec(v_stopPos_2357_);
lean_dec(v_startPos_2356_);
lean_dec_ref(v_str_2355_);
v___y_2325_ = v___x_2352_;
v___y_2326_ = v___y_2343_;
v___y_2327_ = v___y_2344_;
v___y_2328_ = v___x_2358_;
goto v___jp_2324_;
}
else
{
lean_object* v___x_2359_; 
lean_dec(v___x_2353_);
v___x_2359_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_toSyntax___closed__0));
v___y_2325_ = v___x_2352_;
v___y_2326_ = v___y_2343_;
v___y_2327_ = v___y_2344_;
v___y_2328_ = v___x_2359_;
goto v___jp_2324_;
}
}
v___jp_2360_:
{
if (v___y_2364_ == 0)
{
lean_object* v___x_2365_; lean_object* v___x_2366_; 
lean_dec(v___y_2363_);
v___x_2365_ = lean_box(0);
v___x_2366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2366_, 0, v___x_2365_);
return v___x_2366_;
}
else
{
v___y_2343_ = v___y_2361_;
v___y_2344_ = v___y_2362_;
v___y_2345_ = v___y_2363_;
goto v___jp_2342_;
}
}
v_resetjp_2369_:
{
lean_object* v___x_2372_; uint8_t v___x_2373_; 
v___x_2372_ = lp_mathlib_Mathlib_Linter_linter_style_header;
v___x_2373_ = l_Lean_Linter_getLinterValue(v___x_2372_, v_a_2368_);
lean_dec(v_a_2368_);
if (v___x_2373_ == 0)
{
lean_object* v___x_2374_; lean_object* v___x_2376_; 
lean_dec(v_stx_2320_);
v___x_2374_ = lean_box(0);
if (v_isShared_2371_ == 0)
{
lean_ctor_set(v___x_2370_, 0, v___x_2374_);
v___x_2376_ = v___x_2370_;
goto v_reusejp_2375_;
}
else
{
lean_object* v_reuseFailAlloc_2377_; 
v_reuseFailAlloc_2377_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2377_, 0, v___x_2374_);
v___x_2376_ = v_reuseFailAlloc_2377_;
goto v_reusejp_2375_;
}
v_reusejp_2375_:
{
return v___x_2376_;
}
}
else
{
lean_object* v___x_2378_; lean_object* v_messages_2379_; uint8_t v___x_2380_; 
v___x_2378_ = lean_st_ref_get(v___y_2322_);
v_messages_2379_ = lean_ctor_get(v___x_2378_, 1);
lean_inc_ref(v_messages_2379_);
lean_dec(v___x_2378_);
v___x_2380_ = l_Lean_MessageLog_hasErrors(v_messages_2379_);
lean_dec_ref(v_messages_2379_);
if (v___x_2380_ == 0)
{
lean_object* v_fileMap_2381_; lean_object* v_cmdPos_2382_; lean_object* v_source_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v_pos_2390_; uint8_t v___x_2391_; 
v_fileMap_2381_ = lean_ctor_get(v___y_2321_, 1);
v_cmdPos_2382_ = lean_ctor_get(v___y_2321_, 3);
v_source_2383_ = lean_ctor_get(v_fileMap_2381_, 0);
v___x_2384_ = lean_unsigned_to_nat(0u);
v___x_2385_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__0));
v___x_2386_ = lean_box(0);
v___x_2387_ = lean_alloc_ctor(0, 3, 5);
lean_ctor_set(v___x_2387_, 0, v___x_2385_);
lean_ctor_set(v___x_2387_, 1, v___x_2384_);
lean_ctor_set(v___x_2387_, 2, v___x_2386_);
lean_ctor_set_uint8(v___x_2387_, sizeof(void*)*3, v___x_2380_);
lean_ctor_set_uint8(v___x_2387_, sizeof(void*)*3 + 1, v___x_2380_);
lean_ctor_set_uint8(v___x_2387_, sizeof(void*)*3 + 2, v___x_2380_);
lean_ctor_set_uint8(v___x_2387_, sizeof(void*)*3 + 3, v___x_2380_);
lean_ctor_set_uint8(v___x_2387_, sizeof(void*)*3 + 4, v___x_2380_);
v___x_2388_ = l_Lean_ParseImports_whitespace(v_source_2383_, v___x_2387_);
lean_inc_ref(v_source_2383_);
v___x_2389_ = l_Lean_ParseImports_main(v_source_2383_, v___x_2388_);
v_pos_2390_ = lean_ctor_get(v___x_2389_, 1);
lean_inc(v_pos_2390_);
lean_dec_ref(v___x_2389_);
v___x_2391_ = lean_nat_dec_eq(v_cmdPos_2382_, v_pos_2390_);
lean_dec(v_pos_2390_);
if (v___x_2391_ == 0)
{
lean_object* v___x_2392_; lean_object* v___x_2394_; 
lean_dec(v_stx_2320_);
v___x_2392_ = lean_box(0);
if (v_isShared_2371_ == 0)
{
lean_ctor_set(v___x_2370_, 0, v___x_2392_);
v___x_2394_ = v___x_2370_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v___x_2392_);
v___x_2394_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2393_;
}
v_reusejp_2393_:
{
return v___x_2394_;
}
}
else
{
lean_object* v___x_2396_; lean_object* v_a_2397_; lean_object* v___x_2399_; uint8_t v_isShared_2400_; uint8_t v_isSharedCheck_2535_; 
lean_del_object(v___x_2370_);
v___x_2396_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__1___redArg(v___y_2322_);
v_a_2397_ = lean_ctor_get(v___x_2396_, 0);
v_isSharedCheck_2535_ = !lean_is_exclusive(v___x_2396_);
if (v_isSharedCheck_2535_ == 0)
{
v___x_2399_ = v___x_2396_;
v_isShared_2400_ = v_isSharedCheck_2535_;
goto v_resetjp_2398_;
}
else
{
lean_inc(v_a_2397_);
lean_dec(v___x_2396_);
v___x_2399_ = lean_box(0);
v_isShared_2400_ = v_isSharedCheck_2535_;
goto v_resetjp_2398_;
}
v_resetjp_2398_:
{
lean_object* v___y_2407_; uint8_t v___y_2408_; lean_object* v___y_2409_; lean_object* v___y_2410_; uint8_t v___y_2411_; uint8_t v___y_2415_; uint8_t v___y_2416_; lean_object* v___y_2417_; lean_object* v___y_2418_; lean_object* v___y_2419_; lean_object* v___y_2423_; uint8_t v___y_2424_; lean_object* v___y_2425_; lean_object* v___y_2426_; uint8_t v___y_2427_; lean_object* v___y_2428_; lean_object* v___y_2429_; lean_object* v___y_2430_; lean_object* v___y_2436_; lean_object* v___y_2437_; uint8_t v___y_2438_; lean_object* v___f_2491_; lean_object* v___y_2493_; lean_object* v___y_2494_; uint8_t v___y_2518_; uint8_t v___y_2530_; uint8_t v___x_2532_; 
lean_inc(v_a_2397_);
v___f_2491_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2491_, 0, v_a_2397_);
lean_inc(v_stx_2320_);
v___x_2532_ = l_Lean_Parser_isTerminalCommand(v_stx_2320_);
if (v___x_2532_ == 0)
{
lean_object* v___x_2533_; uint8_t v___x_2534_; 
v___x_2533_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__16));
lean_inc(v_stx_2320_);
v___x_2534_ = l_Lean_Syntax_isOfKind(v_stx_2320_, v___x_2533_);
v___y_2530_ = v___x_2534_;
goto v___jp_2529_;
}
else
{
v___y_2530_ = v___x_2391_;
goto v___jp_2529_;
}
v___jp_2401_:
{
lean_object* v___x_2402_; lean_object* v___x_2404_; 
v___x_2402_ = lean_box(0);
if (v_isShared_2400_ == 0)
{
lean_ctor_set(v___x_2399_, 0, v___x_2402_);
v___x_2404_ = v___x_2399_;
goto v_reusejp_2403_;
}
else
{
lean_object* v_reuseFailAlloc_2405_; 
v_reuseFailAlloc_2405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2405_, 0, v___x_2402_);
v___x_2404_ = v_reuseFailAlloc_2405_;
goto v_reusejp_2403_;
}
v_reusejp_2403_:
{
return v___x_2404_;
}
}
v___jp_2406_:
{
if (v___y_2411_ == 0)
{
lean_dec(v_a_2397_);
v___y_2361_ = v___y_2407_;
v___y_2362_ = v___y_2409_;
v___y_2363_ = v___y_2410_;
v___y_2364_ = v___x_2380_;
goto v___jp_2360_;
}
else
{
lean_object* v___x_2412_; uint8_t v___x_2413_; 
v___x_2412_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__2___closed__12));
v___x_2413_ = lean_name_eq(v_a_2397_, v___x_2412_);
lean_dec(v_a_2397_);
if (v___x_2413_ == 0)
{
v___y_2343_ = v___y_2407_;
v___y_2344_ = v___y_2409_;
v___y_2345_ = v___y_2410_;
goto v___jp_2342_;
}
else
{
v___y_2361_ = v___y_2407_;
v___y_2362_ = v___y_2409_;
v___y_2363_ = v___y_2410_;
v___y_2364_ = v___y_2408_;
goto v___jp_2360_;
}
}
}
v___jp_2414_:
{
lean_object* v___x_2420_; uint8_t v___x_2421_; 
v___x_2420_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__2));
v___x_2421_ = lean_name_eq(v_a_2397_, v___x_2420_);
if (v___x_2421_ == 0)
{
v___y_2407_ = v___y_2419_;
v___y_2408_ = v___y_2415_;
v___y_2409_ = v___y_2418_;
v___y_2410_ = v___y_2417_;
v___y_2411_ = v___y_2416_;
goto v___jp_2406_;
}
else
{
v___y_2407_ = v___y_2419_;
v___y_2408_ = v___y_2415_;
v___y_2409_ = v___y_2418_;
v___y_2410_ = v___y_2417_;
v___y_2411_ = v___y_2415_;
goto v___jp_2406_;
}
}
v___jp_2422_:
{
lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; 
v___x_2431_ = lean_array_to_list(v___y_2423_);
v___x_2432_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__4, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__4);
v___x_2433_ = l_Lean_MessageData_joinSep(v___x_2431_, v___x_2432_);
lean_inc_ref(v___y_2426_);
v___x_2434_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___y_2426_, v___y_2430_, v___x_2433_, v___y_2425_, v___y_2428_);
if (lean_obj_tag(v___x_2434_) == 0)
{
lean_dec_ref_known(v___x_2434_, 1);
v___y_2415_ = v___y_2424_;
v___y_2416_ = v___y_2427_;
v___y_2417_ = v___y_2429_;
v___y_2418_ = v___y_2425_;
v___y_2419_ = v___y_2428_;
goto v___jp_2414_;
}
else
{
lean_dec(v___y_2429_);
lean_dec(v_a_2397_);
return v___x_2434_;
}
}
v___jp_2435_:
{
lean_object* v_fileName_2439_; lean_object* v_ref_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; 
v_fileName_2439_ = lean_ctor_get(v___y_2436_, 0);
v_ref_2440_ = lean_ctor_get(v___y_2436_, 7);
v___x_2441_ = lean_string_utf8_byte_size(v_source_2383_);
lean_inc_ref(v_fileName_2439_);
lean_inc_ref(v_source_2383_);
v___x_2442_ = l_Lean_Parser_mkInputContext___redArg(v_source_2383_, v_fileName_2439_, v___x_2391_, v___x_2441_);
v___x_2443_ = l_Lean_Parser_parseHeader(v___x_2442_);
if (lean_obj_tag(v___x_2443_) == 0)
{
lean_object* v_a_2444_; lean_object* v___x_2446_; uint8_t v_isShared_2447_; uint8_t v_isSharedCheck_2478_; 
v_a_2444_ = lean_ctor_get(v___x_2443_, 0);
v_isSharedCheck_2478_ = !lean_is_exclusive(v___x_2443_);
if (v_isSharedCheck_2478_ == 0)
{
v___x_2446_ = v___x_2443_;
v_isShared_2447_ = v_isSharedCheck_2478_;
goto v_resetjp_2445_;
}
else
{
lean_inc(v_a_2444_);
lean_dec(v___x_2443_);
v___x_2446_ = lean_box(0);
v_isShared_2447_ = v_isSharedCheck_2478_;
goto v_resetjp_2445_;
}
v_resetjp_2445_:
{
lean_object* v_snd_2448_; lean_object* v_fst_2449_; lean_object* v_snd_2450_; uint8_t v___x_2451_; 
v_snd_2448_ = lean_ctor_get(v_a_2444_, 1);
lean_inc(v_snd_2448_);
v_fst_2449_ = lean_ctor_get(v_a_2444_, 0);
lean_inc(v_fst_2449_);
lean_dec(v_a_2444_);
v_snd_2450_ = lean_ctor_get(v_snd_2448_, 1);
lean_inc(v_snd_2450_);
lean_dec(v_snd_2448_);
v___x_2451_ = l_Lean_MessageLog_hasErrors(v_snd_2450_);
lean_dec(v_snd_2450_);
if (v___x_2451_ == 0)
{
lean_object* v___x_2452_; lean_object* v___x_2453_; 
lean_del_object(v___x_2446_);
lean_inc(v_fst_2449_);
v___x_2452_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_headerToImportRefs(v_fst_2449_);
v___x_2453_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck(v___x_2452_, v_a_2397_, v___y_2436_, v___y_2437_);
if (lean_obj_tag(v___x_2453_) == 0)
{
lean_object* v___x_2454_; 
lean_dec_ref_known(v___x_2453_, 1);
v___x_2454_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_duplicateImportsCheck(v___x_2452_, v___y_2436_, v___y_2437_);
if (lean_obj_tag(v___x_2454_) == 0)
{
lean_object* v___x_2455_; 
lean_dec_ref_known(v___x_2454_, 1);
lean_inc(v_a_2397_);
v___x_2455_ = lp_mathlib_Mathlib_Linter_directoryDependencyCheck(v_a_2397_, v___y_2436_, v___y_2437_);
if (lean_obj_tag(v___x_2455_) == 0)
{
lean_object* v_a_2456_; lean_object* v___x_2457_; uint8_t v___x_2458_; 
v_a_2456_ = lean_ctor_get(v___x_2455_, 0);
lean_inc(v_a_2456_);
lean_dec_ref_known(v___x_2455_, 1);
v___x_2457_ = lean_array_get_size(v_a_2456_);
v___x_2458_ = lean_nat_dec_lt(v___x_2384_, v___x_2457_);
if (v___x_2458_ == 0)
{
lean_dec(v_a_2456_);
lean_dec_ref(v___x_2452_);
v___y_2415_ = v___x_2451_;
v___y_2416_ = v___y_2438_;
v___y_2417_ = v_fst_2449_;
v___y_2418_ = v___y_2436_;
v___y_2419_ = v___y_2437_;
goto v___jp_2414_;
}
else
{
lean_object* v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; uint8_t v___x_2463_; 
v___x_2459_ = lp_mathlib_Mathlib_Linter_linter_directoryDependency;
v___x_2460_ = lean_array_get_size(v___x_2452_);
v___x_2461_ = lean_unsigned_to_nat(1u);
v___x_2462_ = lean_nat_sub(v___x_2460_, v___x_2461_);
v___x_2463_ = lean_nat_dec_lt(v___x_2462_, v___x_2460_);
if (v___x_2463_ == 0)
{
lean_dec(v___x_2462_);
lean_dec_ref(v___x_2452_);
lean_inc(v_fst_2449_);
v___y_2423_ = v_a_2456_;
v___y_2424_ = v___x_2451_;
v___y_2425_ = v___y_2436_;
v___y_2426_ = v___x_2459_;
v___y_2427_ = v___y_2438_;
v___y_2428_ = v___y_2437_;
v___y_2429_ = v_fst_2449_;
v___y_2430_ = v_fst_2449_;
goto v___jp_2422_;
}
else
{
lean_object* v___x_2464_; lean_object* v_stx_2465_; 
v___x_2464_ = lean_array_fget(v___x_2452_, v___x_2462_);
lean_dec(v___x_2462_);
lean_dec_ref(v___x_2452_);
v_stx_2465_ = lean_ctor_get(v___x_2464_, 1);
lean_inc(v_stx_2465_);
lean_dec(v___x_2464_);
v___y_2423_ = v_a_2456_;
v___y_2424_ = v___x_2451_;
v___y_2425_ = v___y_2436_;
v___y_2426_ = v___x_2459_;
v___y_2427_ = v___y_2438_;
v___y_2428_ = v___y_2437_;
v___y_2429_ = v_fst_2449_;
v___y_2430_ = v_stx_2465_;
goto v___jp_2422_;
}
}
}
else
{
lean_object* v_a_2466_; lean_object* v___x_2468_; uint8_t v_isShared_2469_; uint8_t v_isSharedCheck_2473_; 
lean_dec_ref(v___x_2452_);
lean_dec(v_fst_2449_);
lean_dec(v_a_2397_);
v_a_2466_ = lean_ctor_get(v___x_2455_, 0);
v_isSharedCheck_2473_ = !lean_is_exclusive(v___x_2455_);
if (v_isSharedCheck_2473_ == 0)
{
v___x_2468_ = v___x_2455_;
v_isShared_2469_ = v_isSharedCheck_2473_;
goto v_resetjp_2467_;
}
else
{
lean_inc(v_a_2466_);
lean_dec(v___x_2455_);
v___x_2468_ = lean_box(0);
v_isShared_2469_ = v_isSharedCheck_2473_;
goto v_resetjp_2467_;
}
v_resetjp_2467_:
{
lean_object* v___x_2471_; 
if (v_isShared_2469_ == 0)
{
v___x_2471_ = v___x_2468_;
goto v_reusejp_2470_;
}
else
{
lean_object* v_reuseFailAlloc_2472_; 
v_reuseFailAlloc_2472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2472_, 0, v_a_2466_);
v___x_2471_ = v_reuseFailAlloc_2472_;
goto v_reusejp_2470_;
}
v_reusejp_2470_:
{
return v___x_2471_;
}
}
}
}
else
{
lean_dec_ref(v___x_2452_);
lean_dec(v_fst_2449_);
lean_dec(v_a_2397_);
return v___x_2454_;
}
}
else
{
lean_dec_ref(v___x_2452_);
lean_dec(v_fst_2449_);
lean_dec(v_a_2397_);
return v___x_2453_;
}
}
else
{
lean_object* v___x_2474_; lean_object* v___x_2476_; 
lean_dec(v_fst_2449_);
lean_dec(v_a_2397_);
v___x_2474_ = lean_box(0);
if (v_isShared_2447_ == 0)
{
lean_ctor_set(v___x_2446_, 0, v___x_2474_);
v___x_2476_ = v___x_2446_;
goto v_reusejp_2475_;
}
else
{
lean_object* v_reuseFailAlloc_2477_; 
v_reuseFailAlloc_2477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2477_, 0, v___x_2474_);
v___x_2476_ = v_reuseFailAlloc_2477_;
goto v_reusejp_2475_;
}
v_reusejp_2475_:
{
return v___x_2476_;
}
}
}
}
else
{
lean_object* v_a_2479_; lean_object* v___x_2481_; uint8_t v_isShared_2482_; uint8_t v_isSharedCheck_2490_; 
lean_dec(v_a_2397_);
v_a_2479_ = lean_ctor_get(v___x_2443_, 0);
v_isSharedCheck_2490_ = !lean_is_exclusive(v___x_2443_);
if (v_isSharedCheck_2490_ == 0)
{
v___x_2481_ = v___x_2443_;
v_isShared_2482_ = v_isSharedCheck_2490_;
goto v_resetjp_2480_;
}
else
{
lean_inc(v_a_2479_);
lean_dec(v___x_2443_);
v___x_2481_ = lean_box(0);
v_isShared_2482_ = v_isSharedCheck_2490_;
goto v_resetjp_2480_;
}
v_resetjp_2480_:
{
lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2488_; 
v___x_2483_ = lean_io_error_to_string(v_a_2479_);
v___x_2484_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2484_, 0, v___x_2483_);
v___x_2485_ = l_Lean_MessageData_ofFormat(v___x_2484_);
lean_inc(v_ref_2440_);
v___x_2486_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2486_, 0, v_ref_2440_);
lean_ctor_set(v___x_2486_, 1, v___x_2485_);
if (v_isShared_2482_ == 0)
{
lean_ctor_set(v___x_2481_, 0, v___x_2486_);
v___x_2488_ = v___x_2481_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2489_; 
v_reuseFailAlloc_2489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2489_, 0, v___x_2486_);
v___x_2488_ = v_reuseFailAlloc_2489_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
return v___x_2488_;
}
}
}
}
v___jp_2492_:
{
lean_object* v___x_2495_; lean_object* v___x_2496_; 
v___x_2495_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_inLibraryRootMutex;
v___x_2496_ = lp_mathlib_Std_Mutex_atomically___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__2___redArg(v___x_2495_, v___f_2491_, v___y_2493_, v___y_2494_);
if (lean_obj_tag(v___x_2496_) == 0)
{
lean_object* v_a_2497_; lean_object* v___x_2499_; uint8_t v_isShared_2500_; uint8_t v_isSharedCheck_2508_; 
v_a_2497_ = lean_ctor_get(v___x_2496_, 0);
v_isSharedCheck_2508_ = !lean_is_exclusive(v___x_2496_);
if (v_isSharedCheck_2508_ == 0)
{
v___x_2499_ = v___x_2496_;
v_isShared_2500_ = v_isSharedCheck_2508_;
goto v_resetjp_2498_;
}
else
{
lean_inc(v_a_2497_);
lean_dec(v___x_2496_);
v___x_2499_ = lean_box(0);
v_isShared_2500_ = v_isSharedCheck_2508_;
goto v_resetjp_2498_;
}
v_resetjp_2498_:
{
uint8_t v___x_2501_; 
v___x_2501_ = lean_unbox(v_a_2497_);
lean_dec(v_a_2497_);
if (v___x_2501_ == 0)
{
lean_object* v___x_2502_; uint8_t v___x_2503_; 
v___x_2502_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles;
v___x_2503_ = l_Lean_NameSet_contains(v___x_2502_, v_a_2397_);
if (v___x_2503_ == 0)
{
lean_object* v___x_2504_; lean_object* v___x_2506_; 
lean_dec(v_a_2397_);
v___x_2504_ = lean_box(0);
if (v_isShared_2500_ == 0)
{
lean_ctor_set(v___x_2499_, 0, v___x_2504_);
v___x_2506_ = v___x_2499_;
goto v_reusejp_2505_;
}
else
{
lean_object* v_reuseFailAlloc_2507_; 
v_reuseFailAlloc_2507_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2507_, 0, v___x_2504_);
v___x_2506_ = v_reuseFailAlloc_2507_;
goto v_reusejp_2505_;
}
v_reusejp_2505_:
{
return v___x_2506_;
}
}
else
{
lean_del_object(v___x_2499_);
v___y_2436_ = v___y_2493_;
v___y_2437_ = v___y_2494_;
v___y_2438_ = v___x_2503_;
goto v___jp_2435_;
}
}
else
{
lean_del_object(v___x_2499_);
v___y_2436_ = v___y_2493_;
v___y_2437_ = v___y_2494_;
v___y_2438_ = v___x_2391_;
goto v___jp_2435_;
}
}
}
else
{
lean_object* v_a_2509_; lean_object* v___x_2511_; uint8_t v_isShared_2512_; uint8_t v_isSharedCheck_2516_; 
lean_dec(v_a_2397_);
v_a_2509_ = lean_ctor_get(v___x_2496_, 0);
v_isSharedCheck_2516_ = !lean_is_exclusive(v___x_2496_);
if (v_isSharedCheck_2516_ == 0)
{
v___x_2511_ = v___x_2496_;
v_isShared_2512_ = v_isSharedCheck_2516_;
goto v_resetjp_2510_;
}
else
{
lean_inc(v_a_2509_);
lean_dec(v___x_2496_);
v___x_2511_ = lean_box(0);
v_isShared_2512_ = v_isSharedCheck_2516_;
goto v_resetjp_2510_;
}
v_resetjp_2510_:
{
lean_object* v___x_2514_; 
if (v_isShared_2512_ == 0)
{
v___x_2514_ = v___x_2511_;
goto v_reusejp_2513_;
}
else
{
lean_object* v_reuseFailAlloc_2515_; 
v_reuseFailAlloc_2515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2515_, 0, v_a_2509_);
v___x_2514_ = v_reuseFailAlloc_2515_;
goto v_reusejp_2513_;
}
v_reusejp_2513_:
{
return v___x_2514_;
}
}
}
}
v___jp_2517_:
{
if (v___y_2518_ == 0)
{
lean_object* v___x_2519_; uint8_t v___x_2520_; 
lean_del_object(v___x_2399_);
v___x_2519_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__7));
lean_inc(v_stx_2320_);
v___x_2520_ = l_Lean_Syntax_isOfKind(v_stx_2320_, v___x_2519_);
if (v___x_2520_ == 0)
{
lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; 
v___x_2521_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__9);
lean_inc(v_stx_2320_);
v___x_2522_ = l_Lean_MessageData_ofSyntax(v_stx_2320_);
v___x_2523_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2523_, 0, v___x_2521_);
lean_ctor_set(v___x_2523_, 1, v___x_2522_);
v___x_2524_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__11, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__11);
v___x_2525_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2525_, 0, v___x_2523_);
lean_ctor_set(v___x_2525_, 1, v___x_2524_);
v___x_2526_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__14, &lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___closed__14);
v___x_2527_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2527_, 0, v___x_2525_);
lean_ctor_set(v___x_2527_, 1, v___x_2526_);
v___x_2528_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_broadImportsCheck_spec__0(v___x_2372_, v_stx_2320_, v___x_2527_, v___y_2321_, v___y_2322_);
if (lean_obj_tag(v___x_2528_) == 0)
{
lean_dec_ref_known(v___x_2528_, 1);
v___y_2493_ = v___y_2321_;
v___y_2494_ = v___y_2322_;
goto v___jp_2492_;
}
else
{
lean_dec_ref(v___f_2491_);
lean_dec(v_a_2397_);
return v___x_2528_;
}
}
else
{
lean_dec(v_stx_2320_);
v___y_2493_ = v___y_2321_;
v___y_2494_ = v___y_2322_;
goto v___jp_2492_;
}
}
else
{
lean_dec_ref(v___f_2491_);
lean_dec(v_a_2397_);
lean_dec(v_stx_2320_);
goto v___jp_2401_;
}
}
v___jp_2529_:
{
if (v___y_2530_ == 0)
{
if (lean_obj_tag(v_a_2397_) == 1)
{
lean_object* v_pre_2531_; 
v_pre_2531_ = lean_ctor_get(v_a_2397_, 0);
if (lean_obj_tag(v_pre_2531_) == 0)
{
lean_dec_ref_known(v_a_2397_, 2);
lean_dec_ref(v___f_2491_);
lean_dec(v_stx_2320_);
goto v___jp_2401_;
}
else
{
v___y_2518_ = v___x_2380_;
goto v___jp_2517_;
}
}
else
{
v___y_2518_ = v___x_2380_;
goto v___jp_2517_;
}
}
else
{
lean_dec_ref(v___f_2491_);
lean_dec(v_a_2397_);
lean_dec(v_stx_2320_);
goto v___jp_2401_;
}
}
}
}
}
else
{
lean_object* v___x_2536_; lean_object* v___x_2538_; 
lean_dec(v_stx_2320_);
v___x_2536_ = lean_box(0);
if (v_isShared_2371_ == 0)
{
lean_ctor_set(v___x_2370_, 0, v___x_2536_);
v___x_2538_ = v___x_2370_;
goto v_reusejp_2537_;
}
else
{
lean_object* v_reuseFailAlloc_2539_; 
v_reuseFailAlloc_2539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2539_, 0, v___x_2536_);
v___x_2538_ = v_reuseFailAlloc_2539_;
goto v_reusejp_2537_;
}
v_reusejp_2537_:
{
return v___x_2538_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1___boxed(lean_object* v_stx_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_){
_start:
{
lean_object* v_res_2545_; 
v_res_2545_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter___lam__1(v_stx_2541_, v___y_2542_, v___y_2543_);
lean_dec(v___y_2543_);
lean_dec_ref(v___y_2542_);
return v_res_2545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0(lean_object* v_o_2589_, lean_object* v___y_2590_, lean_object* v___y_2591_){
_start:
{
lean_object* v___x_2593_; 
v___x_2593_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___redArg(v_o_2589_, v___y_2591_);
return v___x_2593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0___boxed(lean_object* v_o_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_){
_start:
{
lean_object* v_res_2598_; 
v_res_2598_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter_spec__0_spec__0(v_o_2594_, v___y_2595_, v___y_2596_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2595_);
return v_res_2598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_initFn_00___x40_Mathlib_Tactic_Linter_Header_1276962649____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2600_; lean_object* v___x_2601_; 
v___x_2600_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerLinter));
v___x_2601_ = l_Lean_Elab_Command_addLinter(v___x_2600_);
return v___x_2601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_initFn_00___x40_Mathlib_Tactic_Linter_Header_1276962649____hygCtx___hyg_2____boxed(lean_object* v_a_2602_){
_start:
{
lean_object* v_res_2603_; 
v_res_2603_ = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_initFn_00___x40_Mathlib_Tactic_Linter_Header_1276962649____hygCtx___hyg_2_();
return v_res_2603_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(uint8_t builtin);
lean_object* runtime_initialize_Std_Sync_Mutex(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Sync_Mutex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Std_Sync_Mutex(uint8_t builtin);
lean_object* runtime_initialize_Lean_Linter_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Sync_Mutex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Linter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_2273770963____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_inLibraryRootMutex = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_inLibraryRootMutex);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_937568399____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_header = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_header);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_Header_677549960____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_style_header_license = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_style_header_license);
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles = _init_lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_headerTestFiles);
res = lp_mathlib___private_Mathlib_Tactic_Linter_Header_0__Mathlib_Linter_Style_Header_initFn_00___x40_Mathlib_Tactic_Linter_Header_1276962649____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Std_Sync_Mutex(uint8_t builtin);
lean_object* initialize_Lean_Parser_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(uint8_t builtin);
lean_object* initialize_Lean_Linter_Basic(uint8_t builtin);
lean_object* initialize_Std_Sync_Mutex(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Sync_Mutex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Linter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Sync_Mutex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
}
#ifdef __cplusplus
}
#endif
