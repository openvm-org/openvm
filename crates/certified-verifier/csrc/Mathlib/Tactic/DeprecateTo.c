// Lean compiler output
// Module: Mathlib.Tactic.DeprecateTo
// Imports: public import Init public meta import Init public meta import Std.Time.Format public import Batteries.Tactic.Alias public import Mathlib.Init
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
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_formatStx(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Meta_isMatcherCore(lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
uint8_t l_Lean_isRecCore(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
extern lean_object* l_Lean_ResolveName_backward_privateInPublic_warn;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Meta_Tactic_TryThis_SuggestionText_prettyExtra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_constants(lean_object*);
uint8_t l_Lean_Name_isInternalDetail(lean_object*);
uint8_t l_Lean_isAuxRecursor(lean_object*, lean_object*);
uint8_t l_Lean_isNoConfusion(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Rat_ofInt(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Array_contains___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* lean_get_current_time();
lean_object* l_Std_Time_Database_defaultGetLocalZoneRules();
lean_object* l_Std_Time_TimeZone_ZoneRules_findLocalTimeTypeForTimestamp(lean_object*, lean_object*);
lean_object* l_Std_Time_TimeZone_LocalTimeType_getTimeZone(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* l_Std_Time_Duration_ofNanoseconds(lean_object*);
lean_object* l_Std_Time_PlainDateTime_ofWallTime(lean_object*);
lean_object* l_Std_Time_PlainDate_toLeanDateString(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_TSyntax_getString(lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_ResolveName_resolveGlobalName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Alias"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "alias"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(222, 148, 33, 118, 168, 84, 84, 125)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__3_value),LEAN_SCALAR_PTR_LITERAL(99, 238, 36, 228, 208, 55, 25, 95)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "attributes"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__13_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__14_value),LEAN_SCALAR_PTR_LITERAL(66, 184, 196, 169, 25, 125, 40, 35)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "@["};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "attrInstance"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__13_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__17_value),LEAN_SCALAR_PTR_LITERAL(241, 75, 242, 110, 47, 5, 20, 104)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "attrKind"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__13_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__19_value),LEAN_SCALAR_PTR_LITERAL(32, 164, 20, 104, 12, 221, 204, 110)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "deprecated"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__21_value),LEAN_SCALAR_PTR_LITERAL(71, 123, 37, 172, 84, 157, 83, 143)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "since"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__28_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\""};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__2_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__2(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__0_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__2_value),LEAN_SCALAR_PTR_LITERAL(117, 34, 246, 137, 114, 183, 220, 217)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__4_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__6_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__8_value),LEAN_SCALAR_PTR_LITERAL(22, 101, 130, 251, 183, 19, 113, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "theorem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__12_value),LEAN_SCALAR_PTR_LITERAL(238, 116, 137, 74, 194, 103, 58, 54)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "DeprecateTo"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "commandDeprecateTo______"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 42, 169, 7, 210, 22, 184, 107)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(177, 99, 236, 115, 20, 68, 175, 180)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "deprecate"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "to"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__20_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ppLine"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__28_value),LEAN_SCALAR_PTR_LITERAL(117, 61, 38, 245, 158, 59, 171, 58)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__27_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__33_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__32_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__37_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo____________ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__37_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "\n\nTry this:\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19_spec__24(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19_spec__24___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__0 = (const lean_object*)&lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__0_value;
static const lean_string_object lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1 = (const lean_object*)&lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__17_spec__26(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__17(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__21(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Private declaration `"};
static const lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__0 = (const lean_object*)&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__1;
static const lean_string_object lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 167, .m_capacity = 167, .m_length = 166, .m_data = "` accessed publicly; this is allowed only because the `backward.privateInPublic` option is enabled. \n\nDisable `backward.privateInPublic.warn` to silence this warning."};
static const lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__2 = (const lean_object*)&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__16(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__16___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__15(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "sorryAx"};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(196, 190, 164, 146, 38, 179, 69, 72)}};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "noConfusionType"};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inj"};
static const lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__3 = (const lean_object*)&lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Warnings:\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "New declaration uses the old name "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "* Pairings:\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "\n\n* Ignoring: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Expected to find one declaration called "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", found "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Unused names: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Un-deprecated declarations: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg(lean_object* v___y_1_){
_start:
{
lean_object* v___x_3_; lean_object* v_env_4_; lean_object* v___x_5_; lean_object* v_mainModule_6_; lean_object* v___x_7_; 
v___x_3_ = lean_st_ref_get(v___y_1_);
v_env_4_ = lean_ctor_get(v___x_3_, 0);
lean_inc_ref(v_env_4_);
lean_dec(v___x_3_);
v___x_5_ = l_Lean_Environment_header(v_env_4_);
lean_dec_ref(v_env_4_);
v_mainModule_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_mainModule_6_);
lean_dec_ref(v___x_5_);
v___x_7_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_7_, 0, v_mainModule_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg___boxed(lean_object* v___y_8_, lean_object* v___y_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg(v___y_8_);
lean_dec(v___y_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1(lean_object* v___y_11_, lean_object* v___y_12_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg(v___y_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___boxed(lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1(v___y_15_, v___y_16_);
lean_dec(v___y_16_);
lean_dec_ref(v___y_15_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__0(lean_object* v_s_19_, lean_object* v_pos_20_){
_start:
{
lean_object* v_str_21_; lean_object* v_startInclusive_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; uint8_t v___x_26_; 
v_str_21_ = lean_ctor_get(v_s_19_, 0);
v_startInclusive_22_ = lean_ctor_get(v_s_19_, 1);
v___x_23_ = lean_nat_add(v_startInclusive_22_, v_pos_20_);
v___x_24_ = lean_nat_sub(v___x_23_, v_startInclusive_22_);
v___x_25_ = lean_unsigned_to_nat(0u);
v___x_26_ = lean_nat_dec_eq(v___x_24_, v___x_25_);
if (v___x_26_ == 0)
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; uint8_t v___y_35_; lean_object* v___x_36_; uint32_t v___x_37_; uint8_t v___y_39_; uint32_t v___x_44_; uint8_t v___x_45_; 
lean_inc(v_startInclusive_22_);
lean_inc_ref(v_str_21_);
v___x_27_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_27_, 0, v_str_21_);
lean_ctor_set(v___x_27_, 1, v_startInclusive_22_);
lean_ctor_set(v___x_27_, 2, v___x_23_);
v___x_28_ = lean_unsigned_to_nat(1u);
v___x_29_ = lean_nat_sub(v___x_24_, v___x_28_);
lean_dec(v___x_24_);
v___x_30_ = l_String_Slice_posLE(v___x_27_, v___x_29_);
lean_dec_ref_known(v___x_27_, 3);
v___x_36_ = lean_nat_add(v_startInclusive_22_, v___x_30_);
v___x_37_ = lean_string_utf8_get_fast(v_str_21_, v___x_36_);
lean_dec(v___x_36_);
v___x_44_ = 32;
v___x_45_ = lean_uint32_dec_eq(v___x_37_, v___x_44_);
if (v___x_45_ == 0)
{
uint32_t v___x_46_; uint8_t v___x_47_; 
v___x_46_ = 9;
v___x_47_ = lean_uint32_dec_eq(v___x_37_, v___x_46_);
v___y_39_ = v___x_47_;
goto v___jp_38_;
}
else
{
v___y_39_ = v___x_45_;
goto v___jp_38_;
}
v___jp_31_:
{
uint8_t v___x_32_; 
v___x_32_ = lean_nat_dec_lt(v___x_30_, v_pos_20_);
if (v___x_32_ == 0)
{
lean_dec(v___x_30_);
return v_pos_20_;
}
else
{
lean_dec(v_pos_20_);
v_pos_20_ = v___x_30_;
goto _start;
}
}
v___jp_34_:
{
if (v___y_35_ == 0)
{
lean_dec(v___x_30_);
return v_pos_20_;
}
else
{
goto v___jp_31_;
}
}
v___jp_38_:
{
if (v___y_39_ == 0)
{
uint32_t v___x_40_; uint8_t v___x_41_; 
v___x_40_ = 13;
v___x_41_ = lean_uint32_dec_eq(v___x_37_, v___x_40_);
if (v___x_41_ == 0)
{
uint32_t v___x_42_; uint8_t v___x_43_; 
v___x_42_ = 10;
v___x_43_ = lean_uint32_dec_eq(v___x_37_, v___x_42_);
v___y_35_ = v___x_43_;
goto v___jp_34_;
}
else
{
v___y_35_ = v___x_41_;
goto v___jp_34_;
}
}
else
{
goto v___jp_31_;
}
}
}
else
{
lean_dec(v___x_24_);
lean_dec(v___x_23_);
return v_pos_20_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__0___boxed(lean_object* v_s_48_, lean_object* v_pos_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__0(v_s_48_, v_pos_49_);
lean_dec_ref(v_s_48_);
return v_res_50_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__12(void){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = l_Array_mkArray0(lean_box(0));
return v___x_72_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__31(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lean_unsigned_to_nat(0u);
v___x_107_ = lean_nat_to_int(v___x_106_);
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__32(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = lean_unsigned_to_nat(1000000000u);
v___x_109_ = lean_nat_to_int(v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx(lean_object* v_id_110_, lean_object* v_n_111_, lean_object* v_dat_112_, lean_object* v_a_113_, lean_object* v_a_114_){
_start:
{
lean_object* v___y_117_; lean_object* v___y_118_; lean_object* v_dat_156_; lean_object* v___y_157_; lean_object* v___y_158_; lean_object* v_a_198_; 
if (lean_obj_tag(v_dat_112_) == 0)
{
lean_object* v___x_205_; 
v___x_205_ = lean_get_current_time();
if (lean_obj_tag(v___x_205_) == 0)
{
lean_object* v_a_206_; lean_object* v___x_207_; 
v_a_206_ = lean_ctor_get(v___x_205_, 0);
lean_inc(v_a_206_);
lean_dec_ref_known(v___x_205_, 1);
v___x_207_ = l_Std_Time_Database_defaultGetLocalZoneRules();
if (lean_obj_tag(v___x_207_) == 0)
{
lean_object* v_a_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v_offset_211_; lean_object* v_second_212_; lean_object* v_nano_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v_date_223_; lean_object* v___x_224_; 
v_a_208_ = lean_ctor_get(v___x_207_, 0);
lean_inc(v_a_208_);
lean_dec_ref_known(v___x_207_, 1);
v___x_209_ = l_Std_Time_TimeZone_ZoneRules_findLocalTimeTypeForTimestamp(v_a_208_, v_a_206_);
v___x_210_ = l_Std_Time_TimeZone_LocalTimeType_getTimeZone(v___x_209_);
lean_dec_ref(v___x_209_);
v_offset_211_ = lean_ctor_get(v___x_210_, 0);
lean_inc(v_offset_211_);
lean_dec_ref(v___x_210_);
v_second_212_ = lean_ctor_get(v_a_206_, 0);
lean_inc(v_second_212_);
v_nano_213_ = lean_ctor_get(v_a_206_, 1);
lean_inc(v_nano_213_);
lean_dec(v_a_206_);
v___x_214_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__31, &lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__31);
v___x_215_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__32, &lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__32);
v___x_216_ = lean_int_mul(v_second_212_, v___x_215_);
lean_dec(v_second_212_);
v___x_217_ = lean_int_add(v___x_216_, v_nano_213_);
lean_dec(v_nano_213_);
lean_dec(v___x_216_);
v___x_218_ = lean_int_mul(v_offset_211_, v___x_215_);
lean_dec(v_offset_211_);
v___x_219_ = lean_int_add(v___x_218_, v___x_214_);
lean_dec(v___x_218_);
v___x_220_ = lean_int_add(v___x_217_, v___x_219_);
lean_dec(v___x_219_);
lean_dec(v___x_217_);
v___x_221_ = l_Std_Time_Duration_ofNanoseconds(v___x_220_);
lean_dec(v___x_220_);
v___x_222_ = l_Std_Time_PlainDateTime_ofWallTime(v___x_221_);
v_date_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc_ref(v_date_223_);
lean_dec_ref(v___x_222_);
v___x_224_ = l_Std_Time_PlainDate_toLeanDateString(v_date_223_);
v_dat_156_ = v___x_224_;
v___y_157_ = v_a_113_;
v___y_158_ = v_a_114_;
goto v___jp_155_;
}
else
{
lean_object* v_a_225_; 
lean_dec(v_a_206_);
lean_dec(v_n_111_);
lean_dec(v_id_110_);
v_a_225_ = lean_ctor_get(v___x_207_, 0);
lean_inc(v_a_225_);
lean_dec_ref_known(v___x_207_, 1);
v_a_198_ = v_a_225_;
goto v___jp_197_;
}
}
else
{
lean_object* v_a_226_; 
lean_dec(v_n_111_);
lean_dec(v_id_110_);
v_a_226_ = lean_ctor_get(v___x_205_, 0);
lean_inc(v_a_226_);
lean_dec_ref_known(v___x_205_, 1);
v_a_198_ = v_a_226_;
goto v___jp_197_;
}
}
else
{
lean_object* v_val_227_; 
v_val_227_ = lean_ctor_get(v_dat_112_, 0);
lean_inc(v_val_227_);
lean_dec_ref_known(v_dat_112_, 1);
v_dat_156_ = v_val_227_;
v___y_157_ = v_a_113_;
v___y_158_ = v_a_114_;
goto v___jp_155_;
}
v___jp_116_:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__3));
v___x_120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__4));
v___x_121_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9));
v___x_122_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__11));
v___x_123_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__12, &lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__12);
lean_inc_n(v___y_118_, 17);
v___x_124_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_124_, 0, v___y_118_);
lean_ctor_set(v___x_124_, 1, v___x_122_);
lean_ctor_set(v___x_124_, 2, v___x_123_);
v___x_125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__15));
v___x_126_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__16));
v___x_127_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_127_, 0, v___y_118_);
lean_ctor_set(v___x_127_, 1, v___x_126_);
v___x_128_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__18));
v___x_129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__20));
lean_inc_ref_n(v___x_124_, 9);
v___x_130_ = l_Lean_Syntax_node1(v___y_118_, v___x_129_, v___x_124_);
v___x_131_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__21));
v___x_132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__22));
v___x_133_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_133_, 0, v___y_118_);
lean_ctor_set(v___x_133_, 1, v___x_131_);
v___x_134_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23));
v___x_135_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_135_, 0, v___y_118_);
lean_ctor_set(v___x_135_, 1, v___x_134_);
v___x_136_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__24));
v___x_137_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_137_, 0, v___y_118_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__25));
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v___y_118_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26));
v___x_141_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_141_, 0, v___y_118_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
lean_inc_ref(v___x_139_);
v___x_142_ = l_Lean_Syntax_node5(v___y_118_, v___x_122_, v___x_135_, v___x_137_, v___x_139_, v___y_117_, v___x_141_);
v___x_143_ = l_Lean_Syntax_node5(v___y_118_, v___x_132_, v___x_133_, v___x_124_, v___x_124_, v___x_124_, v___x_142_);
v___x_144_ = l_Lean_Syntax_node2(v___y_118_, v___x_128_, v___x_130_, v___x_143_);
v___x_145_ = l_Lean_Syntax_node1(v___y_118_, v___x_122_, v___x_144_);
v___x_146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27));
v___x_147_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_147_, 0, v___y_118_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
v___x_148_ = l_Lean_Syntax_node3(v___y_118_, v___x_125_, v___x_127_, v___x_145_, v___x_147_);
v___x_149_ = l_Lean_Syntax_node1(v___y_118_, v___x_122_, v___x_148_);
v___x_150_ = l_Lean_Syntax_node7(v___y_118_, v___x_121_, v___x_124_, v___x_149_, v___x_124_, v___x_124_, v___x_124_, v___x_124_, v___x_124_);
v___x_151_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_151_, 0, v___y_118_);
lean_ctor_set(v___x_151_, 1, v___x_119_);
v___x_152_ = l_Lean_mkIdent(v_n_111_);
v___x_153_ = l_Lean_Syntax_node5(v___y_118_, v___x_120_, v___x_150_, v___x_151_, v___x_152_, v___x_139_, v_id_110_);
v___x_154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
return v___x_154_;
}
v___jp_155_:
{
lean_object* v___x_159_; 
v___x_159_ = l_Lean_Elab_Command_getRef___redArg(v___y_157_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_a_160_; lean_object* v___x_161_; 
v_a_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc(v_a_160_);
lean_dec_ref_known(v___x_159_, 1);
v___x_161_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_157_);
if (lean_obj_tag(v___x_161_) == 0)
{
lean_object* v___x_162_; lean_object* v_quotContext_x3f_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; uint8_t v___x_178_; lean_object* v___x_179_; 
lean_dec_ref_known(v___x_161_, 1);
v___x_162_ = lean_string_utf8_byte_size(v_dat_156_);
v_quotContext_x3f_163_ = lean_ctor_get(v___y_157_, 5);
v___x_164_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_dat_156_);
v___x_165_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_165_, 0, v_dat_156_);
lean_ctor_set(v___x_165_, 1, v___x_164_);
lean_ctor_set(v___x_165_, 2, v___x_162_);
v___x_166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__29));
v___x_167_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__30));
v___x_168_ = lp_mathlib_String_Slice_Pos_revSkipWhile___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__0(v___x_165_, v___x_162_);
lean_dec_ref_known(v___x_165_, 3);
v___x_169_ = lean_string_utf8_extract_fast(v_dat_156_, v___x_164_, v___x_168_);
lean_dec(v___x_168_);
lean_dec_ref(v_dat_156_);
v___x_170_ = lean_string_append(v___x_167_, v___x_169_);
lean_dec_ref(v___x_169_);
v___x_171_ = lean_string_append(v___x_170_, v___x_167_);
v___x_172_ = l_Lean_mkAtom(v___x_171_);
v___x_173_ = lean_unsigned_to_nat(1u);
v___x_174_ = lean_mk_empty_array_with_capacity(v___x_173_);
v___x_175_ = lean_array_push(v___x_174_, v___x_172_);
v___x_176_ = lean_box(2);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v___x_166_);
lean_ctor_set(v___x_177_, 2, v___x_175_);
v___x_178_ = 0;
v___x_179_ = l_Lean_SourceInfo_fromRef(v_a_160_, v___x_178_);
lean_dec(v_a_160_);
if (lean_obj_tag(v_quotContext_x3f_163_) == 0)
{
lean_object* v___x_180_; 
v___x_180_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__1___redArg(v___y_158_);
lean_dec_ref(v___x_180_);
v___y_117_ = v___x_177_;
v___y_118_ = v___x_179_;
goto v___jp_116_;
}
else
{
v___y_117_ = v___x_177_;
v___y_118_ = v___x_179_;
goto v___jp_116_;
}
}
else
{
lean_object* v_a_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_188_; 
lean_dec(v_a_160_);
lean_dec_ref(v_dat_156_);
lean_dec(v_n_111_);
lean_dec(v_id_110_);
v_a_181_ = lean_ctor_get(v___x_161_, 0);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_161_);
if (v_isSharedCheck_188_ == 0)
{
v___x_183_ = v___x_161_;
v_isShared_184_ = v_isSharedCheck_188_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_a_181_);
lean_dec(v___x_161_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_188_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v___x_186_; 
if (v_isShared_184_ == 0)
{
v___x_186_ = v___x_183_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_a_181_);
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
else
{
lean_object* v_a_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_196_; 
lean_dec_ref(v_dat_156_);
lean_dec(v_n_111_);
lean_dec(v_id_110_);
v_a_189_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_196_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_196_ == 0)
{
v___x_191_ = v___x_159_;
v_isShared_192_ = v_isSharedCheck_196_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_a_189_);
lean_dec(v___x_159_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_196_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v___x_194_; 
if (v_isShared_192_ == 0)
{
v___x_194_ = v___x_191_;
goto v_reusejp_193_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v_a_189_);
v___x_194_ = v_reuseFailAlloc_195_;
goto v_reusejp_193_;
}
v_reusejp_193_:
{
return v___x_194_;
}
}
}
}
v___jp_197_:
{
lean_object* v_ref_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v_ref_199_ = lean_ctor_get(v_a_113_, 7);
v___x_200_ = lean_io_error_to_string(v_a_198_);
v___x_201_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
v___x_202_ = l_Lean_MessageData_ofFormat(v___x_201_);
lean_inc(v_ref_199_);
v___x_203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_203_, 0, v_ref_199_);
lean_ctor_set(v___x_203_, 1, v___x_202_);
v___x_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
return v___x_204_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___boxed(lean_object* v_id_228_, lean_object* v_n_229_, lean_object* v_dat_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx(v_id_228_, v_n_229_, v_dat_230_, v_a_231_, v_a_232_);
lean_dec(v_a_232_);
lean_dec_ref(v_a_231_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__2_spec__2(lean_object* v_a_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lean_nat_to_int(v_a_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_DeprecateTo_mkDeprecationStx_spec__2(lean_object* v_a_237_){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_nat_to_int(v_a_237_);
v___x_239_ = l_Rat_ofInt(v___x_238_);
return v___x_239_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_240_, lean_object* v_i_241_, lean_object* v_k_242_){
_start:
{
lean_object* v___x_243_; uint8_t v___x_244_; 
v___x_243_ = lean_array_get_size(v_keys_240_);
v___x_244_ = lean_nat_dec_lt(v_i_241_, v___x_243_);
if (v___x_244_ == 0)
{
lean_dec(v_i_241_);
return v___x_244_;
}
else
{
lean_object* v_k_x27_245_; uint8_t v___x_246_; 
v_k_x27_245_ = lean_array_fget_borrowed(v_keys_240_, v_i_241_);
v___x_246_ = lean_name_eq(v_k_242_, v_k_x27_245_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_247_ = lean_unsigned_to_nat(1u);
v___x_248_ = lean_nat_add(v_i_241_, v___x_247_);
lean_dec(v_i_241_);
v_i_241_ = v___x_248_;
goto _start;
}
else
{
lean_dec(v_i_241_);
return v___x_246_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_250_, lean_object* v_i_251_, lean_object* v_k_252_){
_start:
{
uint8_t v_res_253_; lean_object* v_r_254_; 
v_res_253_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg(v_keys_250_, v_i_251_, v_k_252_);
lean_dec(v_k_252_);
lean_dec_ref(v_keys_250_);
v_r_254_ = lean_box(v_res_253_);
return v_r_254_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg(lean_object* v_x_255_, size_t v_x_256_, lean_object* v_x_257_){
_start:
{
if (lean_obj_tag(v_x_255_) == 0)
{
lean_object* v_es_258_; lean_object* v___x_259_; size_t v___x_260_; size_t v___x_261_; lean_object* v_j_262_; lean_object* v___x_263_; 
v_es_258_ = lean_ctor_get(v_x_255_, 0);
v___x_259_ = lean_box(2);
v___x_260_ = ((size_t)31ULL);
v___x_261_ = lean_usize_land(v_x_256_, v___x_260_);
v_j_262_ = lean_usize_to_nat(v___x_261_);
v___x_263_ = lean_array_get_borrowed(v___x_259_, v_es_258_, v_j_262_);
lean_dec(v_j_262_);
switch(lean_obj_tag(v___x_263_))
{
case 0:
{
lean_object* v_key_264_; uint8_t v___x_265_; 
v_key_264_ = lean_ctor_get(v___x_263_, 0);
v___x_265_ = lean_name_eq(v_x_257_, v_key_264_);
return v___x_265_;
}
case 1:
{
lean_object* v_node_266_; size_t v___x_267_; size_t v___x_268_; 
v_node_266_ = lean_ctor_get(v___x_263_, 0);
v___x_267_ = ((size_t)5ULL);
v___x_268_ = lean_usize_shift_right(v_x_256_, v___x_267_);
v_x_255_ = v_node_266_;
v_x_256_ = v___x_268_;
goto _start;
}
default: 
{
uint8_t v___x_270_; 
v___x_270_ = 0;
return v___x_270_;
}
}
}
else
{
lean_object* v_ks_271_; lean_object* v___x_272_; uint8_t v___x_273_; 
v_ks_271_ = lean_ctor_get(v_x_255_, 0);
v___x_272_ = lean_unsigned_to_nat(0u);
v___x_273_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg(v_ks_271_, v___x_272_, v_x_257_);
return v___x_273_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg___boxed(lean_object* v_x_274_, lean_object* v_x_275_, lean_object* v_x_276_){
_start:
{
size_t v_x_1312__boxed_277_; uint8_t v_res_278_; lean_object* v_r_279_; 
v_x_1312__boxed_277_ = lean_unbox_usize(v_x_275_);
lean_dec(v_x_275_);
v_res_278_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg(v_x_274_, v_x_1312__boxed_277_, v_x_276_);
lean_dec(v_x_276_);
lean_dec_ref(v_x_274_);
v_r_279_ = lean_box(v_res_278_);
return v_r_279_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg(lean_object* v_x_280_, lean_object* v_x_281_){
_start:
{
uint64_t v___y_283_; 
if (lean_obj_tag(v_x_281_) == 0)
{
uint64_t v___x_286_; 
v___x_286_ = 1723ULL;
v___y_283_ = v___x_286_;
goto v___jp_282_;
}
else
{
uint64_t v_hash_287_; 
v_hash_287_ = lean_ctor_get_uint64(v_x_281_, sizeof(void*)*2);
v___y_283_ = v_hash_287_;
goto v___jp_282_;
}
v___jp_282_:
{
size_t v___x_284_; uint8_t v___x_285_; 
v___x_284_ = lean_uint64_to_usize(v___y_283_);
v___x_285_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg(v_x_280_, v___x_284_, v_x_281_);
return v___x_285_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg___boxed(lean_object* v_x_288_, lean_object* v_x_289_){
_start:
{
uint8_t v_res_290_; lean_object* v_r_291_; 
v_res_290_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg(v_x_288_, v_x_289_);
lean_dec(v_x_289_);
lean_dec_ref(v_x_288_);
v_r_291_ = lean_box(v_res_290_);
return v_r_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg(lean_object* v_old_292_, lean_object* v_as_x27_293_, lean_object* v_b_294_){
_start:
{
if (lean_obj_tag(v_as_x27_293_) == 0)
{
lean_dec_ref(v_old_292_);
return v_b_294_;
}
else
{
lean_object* v_head_295_; lean_object* v_tail_296_; lean_object* v_fst_297_; lean_object* v___x_298_; lean_object* v_map_u2082_299_; uint8_t v___x_300_; 
v_head_295_ = lean_ctor_get(v_as_x27_293_, 0);
v_tail_296_ = lean_ctor_get(v_as_x27_293_, 1);
v_fst_297_ = lean_ctor_get(v_head_295_, 0);
lean_inc_ref(v_old_292_);
v___x_298_ = l_Lean_Environment_constants(v_old_292_);
v_map_u2082_299_ = lean_ctor_get(v___x_298_, 1);
lean_inc_ref(v_map_u2082_299_);
lean_dec_ref(v___x_298_);
v___x_300_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg(v_map_u2082_299_, v_fst_297_);
lean_dec_ref(v_map_u2082_299_);
if (v___x_300_ == 0)
{
lean_object* v___x_301_; 
lean_inc(v_fst_297_);
v___x_301_ = lean_array_push(v_b_294_, v_fst_297_);
v_as_x27_293_ = v_tail_296_;
v_b_294_ = v___x_301_;
goto _start;
}
else
{
v_as_x27_293_ = v_tail_296_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg___boxed(lean_object* v_old_304_, lean_object* v_as_x27_305_, lean_object* v_b_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg(v_old_304_, v_as_x27_305_, v_b_306_);
lean_dec(v_as_x27_305_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___lam__0(lean_object* v_ps_308_, lean_object* v_k_309_, lean_object* v_v_310_){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_311_, 0, v_k_309_);
lean_ctor_set(v___x_311_, 1, v_v_310_);
v___x_312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
lean_ctor_set(v___x_312_, 1, v_ps_308_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg___lam__0(lean_object* v_f_313_, lean_object* v_x1_314_, lean_object* v_x2_315_, lean_object* v_x3_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lean_apply_3(v_f_313_, v_x1_314_, v_x2_315_, v_x3_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg(lean_object* v_f_318_, lean_object* v_keys_319_, lean_object* v_vals_320_, lean_object* v_i_321_, lean_object* v_acc_322_){
_start:
{
lean_object* v___x_323_; uint8_t v___x_324_; 
v___x_323_ = lean_array_get_size(v_keys_319_);
v___x_324_ = lean_nat_dec_lt(v_i_321_, v___x_323_);
if (v___x_324_ == 0)
{
lean_dec(v_i_321_);
lean_dec(v_f_318_);
return v_acc_322_;
}
else
{
lean_object* v_k_325_; lean_object* v_v_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v_k_325_ = lean_array_fget_borrowed(v_keys_319_, v_i_321_);
v_v_326_ = lean_array_fget_borrowed(v_vals_320_, v_i_321_);
lean_inc(v_f_318_);
lean_inc(v_v_326_);
lean_inc(v_k_325_);
v___x_327_ = lean_apply_3(v_f_318_, v_acc_322_, v_k_325_, v_v_326_);
v___x_328_ = lean_unsigned_to_nat(1u);
v___x_329_ = lean_nat_add(v_i_321_, v___x_328_);
lean_dec(v_i_321_);
v_i_321_ = v___x_329_;
v_acc_322_ = v___x_327_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg___boxed(lean_object* v_f_331_, lean_object* v_keys_332_, lean_object* v_vals_333_, lean_object* v_i_334_, lean_object* v_acc_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg(v_f_331_, v_keys_332_, v_vals_333_, v_i_334_, v_acc_335_);
lean_dec_ref(v_vals_333_);
lean_dec_ref(v_keys_332_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(lean_object* v_f_337_, lean_object* v_x_338_, lean_object* v_x_339_){
_start:
{
if (lean_obj_tag(v_x_338_) == 0)
{
lean_object* v_es_340_; lean_object* v___x_341_; lean_object* v___x_342_; uint8_t v___x_343_; 
v_es_340_ = lean_ctor_get(v_x_338_, 0);
v___x_341_ = lean_unsigned_to_nat(0u);
v___x_342_ = lean_array_get_size(v_es_340_);
v___x_343_ = lean_nat_dec_lt(v___x_341_, v___x_342_);
if (v___x_343_ == 0)
{
lean_dec(v_f_337_);
return v_x_339_;
}
else
{
uint8_t v___x_344_; 
v___x_344_ = lean_nat_dec_le(v___x_342_, v___x_342_);
if (v___x_344_ == 0)
{
if (v___x_343_ == 0)
{
lean_dec(v_f_337_);
return v_x_339_;
}
else
{
size_t v___x_345_; size_t v___x_346_; lean_object* v___x_347_; 
v___x_345_ = ((size_t)0ULL);
v___x_346_ = lean_usize_of_nat(v___x_342_);
v___x_347_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_337_, v_es_340_, v___x_345_, v___x_346_, v_x_339_);
return v___x_347_;
}
}
else
{
size_t v___x_348_; size_t v___x_349_; lean_object* v___x_350_; 
v___x_348_ = ((size_t)0ULL);
v___x_349_ = lean_usize_of_nat(v___x_342_);
v___x_350_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_337_, v_es_340_, v___x_348_, v___x_349_, v_x_339_);
return v___x_350_;
}
}
}
else
{
lean_object* v_ks_351_; lean_object* v_vs_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v_ks_351_ = lean_ctor_get(v_x_338_, 0);
v_vs_352_ = lean_ctor_get(v_x_338_, 1);
v___x_353_ = lean_unsigned_to_nat(0u);
v___x_354_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg(v_f_337_, v_ks_351_, v_vs_352_, v___x_353_, v_x_339_);
return v___x_354_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(lean_object* v_f_355_, lean_object* v_as_356_, size_t v_i_357_, size_t v_stop_358_, lean_object* v_b_359_){
_start:
{
lean_object* v___y_361_; uint8_t v___x_365_; 
v___x_365_ = lean_usize_dec_eq(v_i_357_, v_stop_358_);
if (v___x_365_ == 0)
{
lean_object* v___x_366_; 
v___x_366_ = lean_array_uget_borrowed(v_as_356_, v_i_357_);
switch(lean_obj_tag(v___x_366_))
{
case 0:
{
lean_object* v_key_367_; lean_object* v_val_368_; lean_object* v___x_369_; 
v_key_367_ = lean_ctor_get(v___x_366_, 0);
v_val_368_ = lean_ctor_get(v___x_366_, 1);
lean_inc(v_f_355_);
lean_inc(v_val_368_);
lean_inc(v_key_367_);
v___x_369_ = lean_apply_3(v_f_355_, v_b_359_, v_key_367_, v_val_368_);
v___y_361_ = v___x_369_;
goto v___jp_360_;
}
case 1:
{
lean_object* v_node_370_; lean_object* v___x_371_; 
v_node_370_ = lean_ctor_get(v___x_366_, 0);
lean_inc(v_f_355_);
v___x_371_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(v_f_355_, v_node_370_, v_b_359_);
v___y_361_ = v___x_371_;
goto v___jp_360_;
}
default: 
{
v___y_361_ = v_b_359_;
goto v___jp_360_;
}
}
}
else
{
lean_dec(v_f_355_);
return v_b_359_;
}
v___jp_360_:
{
size_t v___x_362_; size_t v___x_363_; 
v___x_362_ = ((size_t)1ULL);
v___x_363_ = lean_usize_add(v_i_357_, v___x_362_);
v_i_357_ = v___x_363_;
v_b_359_ = v___y_361_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg___boxed(lean_object* v_f_372_, lean_object* v_as_373_, lean_object* v_i_374_, lean_object* v_stop_375_, lean_object* v_b_376_){
_start:
{
size_t v_i_boxed_377_; size_t v_stop_boxed_378_; lean_object* v_res_379_; 
v_i_boxed_377_ = lean_unbox_usize(v_i_374_);
lean_dec(v_i_374_);
v_stop_boxed_378_ = lean_unbox_usize(v_stop_375_);
lean_dec(v_stop_375_);
v_res_379_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_372_, v_as_373_, v_i_boxed_377_, v_stop_boxed_378_, v_b_376_);
lean_dec_ref(v_as_373_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg___boxed(lean_object* v_f_380_, lean_object* v_x_381_, lean_object* v_x_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(v_f_380_, v_x_381_, v_x_382_);
lean_dec_ref(v_x_381_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg(lean_object* v_map_384_, lean_object* v_f_385_, lean_object* v_init_386_){
_start:
{
lean_object* v___f_387_; lean_object* v___x_388_; 
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg___lam__0), 4, 1);
lean_closure_set(v___f_387_, 0, v_f_385_);
v___x_388_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(v___f_387_, v_map_384_, v_init_386_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg___boxed(lean_object* v_map_389_, lean_object* v_f_390_, lean_object* v_init_391_){
_start:
{
lean_object* v_res_392_; 
v_res_392_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg(v_map_389_, v_f_390_, v_init_391_);
lean_dec_ref(v_map_389_);
return v_res_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg(lean_object* v_m_394_){
_start:
{
lean_object* v___f_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___f_395_ = ((lean_object*)(lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___closed__0));
v___x_396_ = lean_box(0);
v___x_397_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg(v_m_394_, v___f_395_, v___x_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg___boxed(lean_object* v_m_398_){
_start:
{
lean_object* v_res_399_; 
v_res_399_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg(v_m_398_);
lean_dec_ref(v_m_398_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg(lean_object* v_hi_400_, lean_object* v_pivot_401_, lean_object* v_as_402_, lean_object* v_i_403_, lean_object* v_k_404_){
_start:
{
uint8_t v___x_405_; 
v___x_405_ = lean_nat_dec_lt(v_k_404_, v_hi_400_);
if (v___x_405_ == 0)
{
lean_object* v___x_406_; lean_object* v___x_407_; 
lean_dec(v_k_404_);
lean_dec(v_pivot_401_);
v___x_406_ = lean_array_fswap(v_as_402_, v_i_403_, v_hi_400_);
v___x_407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_407_, 0, v_i_403_);
lean_ctor_set(v___x_407_, 1, v___x_406_);
return v___x_407_;
}
else
{
lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; uint8_t v___x_411_; 
v___x_408_ = lean_array_fget_borrowed(v_as_402_, v_k_404_);
lean_inc(v___x_408_);
v___x_409_ = l_Lean_Name_toString(v___x_408_, v___x_405_);
lean_inc(v_pivot_401_);
v___x_410_ = l_Lean_Name_toString(v_pivot_401_, v___x_405_);
v___x_411_ = lean_string_dec_lt(v___x_409_, v___x_410_);
lean_dec_ref(v___x_410_);
lean_dec_ref(v___x_409_);
if (v___x_411_ == 0)
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = lean_unsigned_to_nat(1u);
v___x_413_ = lean_nat_add(v_k_404_, v___x_412_);
lean_dec(v_k_404_);
v_k_404_ = v___x_413_;
goto _start;
}
else
{
lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_415_ = lean_array_fswap(v_as_402_, v_i_403_, v_k_404_);
v___x_416_ = lean_unsigned_to_nat(1u);
v___x_417_ = lean_nat_add(v_i_403_, v___x_416_);
lean_dec(v_i_403_);
v___x_418_ = lean_nat_add(v_k_404_, v___x_416_);
lean_dec(v_k_404_);
v_as_402_ = v___x_415_;
v_i_403_ = v___x_417_;
v_k_404_ = v___x_418_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg___boxed(lean_object* v_hi_420_, lean_object* v_pivot_421_, lean_object* v_as_422_, lean_object* v_i_423_, lean_object* v_k_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg(v_hi_420_, v_pivot_421_, v_as_422_, v_i_423_, v_k_424_);
lean_dec(v_hi_420_);
return v_res_425_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0(uint8_t v___x_426_, lean_object* v_x1_427_, lean_object* v_x2_428_){
_start:
{
lean_object* v___x_429_; lean_object* v___x_430_; uint8_t v___x_431_; 
v___x_429_ = l_Lean_Name_toString(v_x1_427_, v___x_426_);
v___x_430_ = l_Lean_Name_toString(v_x2_428_, v___x_426_);
v___x_431_ = lean_string_dec_lt(v___x_429_, v___x_430_);
lean_dec_ref(v___x_430_);
lean_dec_ref(v___x_429_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0___boxed(lean_object* v___x_432_, lean_object* v_x1_433_, lean_object* v_x2_434_){
_start:
{
uint8_t v___x_1512__boxed_435_; uint8_t v_res_436_; lean_object* v_r_437_; 
v___x_1512__boxed_435_ = lean_unbox(v___x_432_);
v_res_436_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0(v___x_1512__boxed_435_, v_x1_433_, v_x2_434_);
v_r_437_ = lean_box(v_res_436_);
return v_r_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(lean_object* v_n_438_, lean_object* v_as_439_, lean_object* v_lo_440_, lean_object* v_hi_441_){
_start:
{
lean_object* v___y_443_; uint8_t v___x_453_; 
v___x_453_ = lean_nat_dec_lt(v_lo_440_, v_hi_441_);
if (v___x_453_ == 0)
{
lean_dec(v_lo_440_);
return v_as_439_;
}
else
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v_mid_456_; lean_object* v___y_458_; lean_object* v___y_464_; lean_object* v___x_469_; lean_object* v___x_470_; uint8_t v___x_471_; 
v___x_454_ = lean_nat_add(v_lo_440_, v_hi_441_);
v___x_455_ = lean_unsigned_to_nat(1u);
v_mid_456_ = lean_nat_shiftr(v___x_454_, v___x_455_);
lean_dec(v___x_454_);
v___x_469_ = lean_array_fget_borrowed(v_as_439_, v_mid_456_);
v___x_470_ = lean_array_fget_borrowed(v_as_439_, v_lo_440_);
lean_inc(v___x_470_);
lean_inc(v___x_469_);
v___x_471_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0(v___x_453_, v___x_469_, v___x_470_);
if (v___x_471_ == 0)
{
v___y_464_ = v_as_439_;
goto v___jp_463_;
}
else
{
lean_object* v___x_472_; 
v___x_472_ = lean_array_fswap(v_as_439_, v_lo_440_, v_mid_456_);
v___y_464_ = v___x_472_;
goto v___jp_463_;
}
v___jp_457_:
{
lean_object* v___x_459_; lean_object* v___x_460_; uint8_t v___x_461_; 
v___x_459_ = lean_array_fget_borrowed(v___y_458_, v_mid_456_);
v___x_460_ = lean_array_fget_borrowed(v___y_458_, v_hi_441_);
lean_inc(v___x_460_);
lean_inc(v___x_459_);
v___x_461_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0(v___x_453_, v___x_459_, v___x_460_);
if (v___x_461_ == 0)
{
lean_dec(v_mid_456_);
v___y_443_ = v___y_458_;
goto v___jp_442_;
}
else
{
lean_object* v___x_462_; 
v___x_462_ = lean_array_fswap(v___y_458_, v_mid_456_, v_hi_441_);
lean_dec(v_mid_456_);
v___y_443_ = v___x_462_;
goto v___jp_442_;
}
}
v___jp_463_:
{
lean_object* v___x_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v___x_465_ = lean_array_fget_borrowed(v___y_464_, v_hi_441_);
v___x_466_ = lean_array_fget_borrowed(v___y_464_, v_lo_440_);
lean_inc(v___x_466_);
lean_inc(v___x_465_);
v___x_467_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___lam__0(v___x_453_, v___x_465_, v___x_466_);
if (v___x_467_ == 0)
{
v___y_458_ = v___y_464_;
goto v___jp_457_;
}
else
{
lean_object* v___x_468_; 
v___x_468_ = lean_array_fswap(v___y_464_, v_lo_440_, v_hi_441_);
v___y_458_ = v___x_468_;
goto v___jp_457_;
}
}
}
v___jp_442_:
{
lean_object* v_pivot_444_; lean_object* v___x_445_; lean_object* v_fst_446_; lean_object* v_snd_447_; uint8_t v___x_448_; 
v_pivot_444_ = lean_array_fget(v___y_443_, v_hi_441_);
lean_inc_n(v_lo_440_, 2);
v___x_445_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg(v_hi_441_, v_pivot_444_, v___y_443_, v_lo_440_, v_lo_440_);
v_fst_446_ = lean_ctor_get(v___x_445_, 0);
lean_inc(v_fst_446_);
v_snd_447_ = lean_ctor_get(v___x_445_, 1);
lean_inc(v_snd_447_);
lean_dec_ref(v___x_445_);
v___x_448_ = lean_nat_dec_le(v_hi_441_, v_fst_446_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_449_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(v_n_438_, v_snd_447_, v_lo_440_, v_fst_446_);
v___x_450_ = lean_unsigned_to_nat(1u);
v___x_451_ = lean_nat_add(v_fst_446_, v___x_450_);
lean_dec(v_fst_446_);
v_as_439_ = v___x_449_;
v_lo_440_ = v___x_451_;
goto _start;
}
else
{
lean_dec(v_fst_446_);
lean_dec(v_lo_440_);
return v_snd_447_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg___boxed(lean_object* v_n_473_, lean_object* v_as_474_, lean_object* v_lo_475_, lean_object* v_hi_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(v_n_473_, v_as_474_, v_lo_475_, v_hi_476_);
lean_dec(v_hi_476_);
lean_dec(v_n_473_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames(lean_object* v_old_480_, lean_object* v_new_481_){
_start:
{
lean_object* v___x_482_; lean_object* v_map_u2082_483_; lean_object* v___x_484_; lean_object* v_diffs_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; uint8_t v___x_489_; 
v___x_482_ = l_Lean_Environment_constants(v_new_481_);
v_map_u2082_483_ = lean_ctor_get(v___x_482_, 1);
lean_inc_ref(v_map_u2082_483_);
lean_dec_ref(v___x_482_);
v___x_484_ = lean_unsigned_to_nat(0u);
v_diffs_485_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames___closed__0));
v___x_486_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg(v_map_u2082_483_);
lean_dec_ref(v_map_u2082_483_);
v___x_487_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg(v_old_480_, v___x_486_, v_diffs_485_);
lean_dec(v___x_486_);
v___x_488_ = lean_array_get_size(v___x_487_);
v___x_489_ = lean_nat_dec_eq(v___x_488_, v___x_484_);
if (v___x_489_ == 0)
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___y_493_; uint8_t v___x_497_; 
v___x_490_ = lean_unsigned_to_nat(1u);
v___x_491_ = lean_nat_sub(v___x_488_, v___x_490_);
v___x_497_ = lean_nat_dec_le(v___x_484_, v___x_491_);
if (v___x_497_ == 0)
{
lean_inc(v___x_491_);
v___y_493_ = v___x_491_;
goto v___jp_492_;
}
else
{
v___y_493_ = v___x_484_;
goto v___jp_492_;
}
v___jp_492_:
{
uint8_t v___x_494_; 
v___x_494_ = lean_nat_dec_le(v___y_493_, v___x_491_);
if (v___x_494_ == 0)
{
lean_object* v___x_495_; 
lean_dec(v___x_491_);
lean_inc(v___y_493_);
v___x_495_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(v___x_488_, v___x_487_, v___y_493_, v___y_493_);
lean_dec(v___y_493_);
return v___x_495_;
}
else
{
lean_object* v___x_496_; 
v___x_496_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(v___x_488_, v___x_487_, v___y_493_, v___x_491_);
lean_dec(v___x_491_);
return v___x_496_;
}
}
}
else
{
return v___x_487_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0(lean_object* v_00_u03b2_498_, lean_object* v_x_499_, lean_object* v_x_500_){
_start:
{
uint8_t v___x_501_; 
v___x_501_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___redArg(v_x_499_, v_x_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0___boxed(lean_object* v_00_u03b2_502_, lean_object* v_x_503_, lean_object* v_x_504_){
_start:
{
uint8_t v_res_505_; lean_object* v_r_506_; 
v_res_505_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0(v_00_u03b2_502_, v_x_503_, v_x_504_);
lean_dec(v_x_504_);
lean_dec_ref(v_x_503_);
v_r_506_ = lean_box(v_res_505_);
return v_r_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1(lean_object* v_00_u03b2_507_, lean_object* v_m_508_){
_start:
{
lean_object* v___x_509_; 
v___x_509_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___redArg(v_m_508_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1___boxed(lean_object* v_00_u03b2_510_, lean_object* v_m_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1(v_00_u03b2_510_, v_m_511_);
lean_dec_ref(v_m_511_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2(lean_object* v_old_513_, lean_object* v_as_514_, lean_object* v_as_x27_515_, lean_object* v_b_516_, lean_object* v_a_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___redArg(v_old_513_, v_as_x27_515_, v_b_516_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2___boxed(lean_object* v_old_519_, lean_object* v_as_520_, lean_object* v_as_x27_521_, lean_object* v_b_522_, lean_object* v_a_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__2(v_old_519_, v_as_520_, v_as_x27_521_, v_b_522_, v_a_523_);
lean_dec(v_as_x27_521_);
lean_dec(v_as_520_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3(lean_object* v_n_525_, lean_object* v_as_526_, lean_object* v_lo_527_, lean_object* v_hi_528_, lean_object* v_w_529_, lean_object* v_hlo_530_, lean_object* v_hhi_531_){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___redArg(v_n_525_, v_as_526_, v_lo_527_, v_hi_528_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3___boxed(lean_object* v_n_533_, lean_object* v_as_534_, lean_object* v_lo_535_, lean_object* v_hi_536_, lean_object* v_w_537_, lean_object* v_hlo_538_, lean_object* v_hhi_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3(v_n_533_, v_as_534_, v_lo_535_, v_hi_536_, v_w_537_, v_hlo_538_, v_hhi_539_);
lean_dec(v_hi_536_);
lean_dec(v_n_533_);
return v_res_540_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0(lean_object* v_00_u03b2_541_, lean_object* v_x_542_, size_t v_x_543_, lean_object* v_x_544_){
_start:
{
uint8_t v___x_545_; 
v___x_545_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___redArg(v_x_542_, v_x_543_, v_x_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0___boxed(lean_object* v_00_u03b2_546_, lean_object* v_x_547_, lean_object* v_x_548_, lean_object* v_x_549_){
_start:
{
size_t v_x_1634__boxed_550_; uint8_t v_res_551_; lean_object* v_r_552_; 
v_x_1634__boxed_550_ = lean_unbox_usize(v_x_548_);
lean_dec(v_x_548_);
v_res_551_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0(v_00_u03b2_546_, v_x_547_, v_x_1634__boxed_550_, v_x_549_);
lean_dec(v_x_549_);
lean_dec_ref(v_x_547_);
v_r_552_ = lean_box(v_res_551_);
return v_r_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2(lean_object* v_00_u03c3_553_, lean_object* v_00_u03b2_554_, lean_object* v_map_555_, lean_object* v_f_556_, lean_object* v_init_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___redArg(v_map_555_, v_f_556_, v_init_557_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2___boxed(lean_object* v_00_u03c3_559_, lean_object* v_00_u03b2_560_, lean_object* v_map_561_, lean_object* v_f_562_, lean_object* v_init_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2(v_00_u03c3_559_, v_00_u03b2_560_, v_map_561_, v_f_562_, v_init_563_);
lean_dec_ref(v_map_561_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5(lean_object* v_n_565_, lean_object* v_lo_566_, lean_object* v_hi_567_, lean_object* v_hhi_568_, lean_object* v_pivot_569_, lean_object* v_as_570_, lean_object* v_i_571_, lean_object* v_k_572_, lean_object* v_ilo_573_, lean_object* v_ik_574_, lean_object* v_w_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___redArg(v_hi_567_, v_pivot_569_, v_as_570_, v_i_571_, v_k_572_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5___boxed(lean_object* v_n_577_, lean_object* v_lo_578_, lean_object* v_hi_579_, lean_object* v_hhi_580_, lean_object* v_pivot_581_, lean_object* v_as_582_, lean_object* v_i_583_, lean_object* v_k_584_, lean_object* v_ilo_585_, lean_object* v_ik_586_, lean_object* v_w_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__3_spec__5(v_n_577_, v_lo_578_, v_hi_579_, v_hhi_580_, v_pivot_581_, v_as_582_, v_i_583_, v_k_584_, v_ilo_585_, v_ik_586_, v_w_587_);
lean_dec(v_hi_579_);
lean_dec(v_lo_578_);
lean_dec(v_n_577_);
return v_res_588_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_589_, lean_object* v_keys_590_, lean_object* v_vals_591_, lean_object* v_heq_592_, lean_object* v_i_593_, lean_object* v_k_594_){
_start:
{
uint8_t v___x_595_; 
v___x_595_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___redArg(v_keys_590_, v_i_593_, v_k_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_596_, lean_object* v_keys_597_, lean_object* v_vals_598_, lean_object* v_heq_599_, lean_object* v_i_600_, lean_object* v_k_601_){
_start:
{
uint8_t v_res_602_; lean_object* v_r_603_; 
v_res_602_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__0_spec__0_spec__1(v_00_u03b2_596_, v_keys_597_, v_vals_598_, v_heq_599_, v_i_600_, v_k_601_);
lean_dec(v_k_601_);
lean_dec_ref(v_vals_598_);
lean_dec_ref(v_keys_597_);
v_r_603_ = lean_box(v_res_602_);
return v_r_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___redArg(lean_object* v_map_604_, lean_object* v_f_605_, lean_object* v_init_606_){
_start:
{
lean_object* v___x_607_; 
v___x_607_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(v_f_605_, v_map_604_, v_init_606_);
return v___x_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_map_608_, lean_object* v_f_609_, lean_object* v_init_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___redArg(v_map_608_, v_f_609_, v_init_610_);
lean_dec_ref(v_map_608_);
return v_res_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4(lean_object* v_00_u03c3_612_, lean_object* v_00_u03b2_613_, lean_object* v_map_614_, lean_object* v_f_615_, lean_object* v_init_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(v_f_615_, v_map_614_, v_init_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03c3_618_, lean_object* v_00_u03b2_619_, lean_object* v_map_620_, lean_object* v_f_621_, lean_object* v_init_622_){
_start:
{
lean_object* v_res_623_; 
v_res_623_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4(v_00_u03c3_618_, v_00_u03b2_619_, v_map_620_, v_f_621_, v_init_622_);
lean_dec_ref(v_map_620_);
return v_res_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7(lean_object* v_00_u03c3_624_, lean_object* v_00_u03b1_625_, lean_object* v_00_u03b2_626_, lean_object* v_f_627_, lean_object* v_x_628_, lean_object* v_x_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___redArg(v_f_627_, v_x_628_, v_x_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7___boxed(lean_object* v_00_u03c3_631_, lean_object* v_00_u03b1_632_, lean_object* v_00_u03b2_633_, lean_object* v_f_634_, lean_object* v_x_635_, lean_object* v_x_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7(v_00_u03c3_631_, v_00_u03b1_632_, v_00_u03b2_633_, v_f_634_, v_x_635_, v_x_636_);
lean_dec_ref(v_x_635_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9(lean_object* v_00_u03b1_638_, lean_object* v_00_u03b2_639_, lean_object* v_00_u03c3_640_, lean_object* v_f_641_, lean_object* v_as_642_, size_t v_i_643_, size_t v_stop_644_, lean_object* v_b_645_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_641_, v_as_642_, v_i_643_, v_stop_644_, v_b_645_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9___boxed(lean_object* v_00_u03b1_647_, lean_object* v_00_u03b2_648_, lean_object* v_00_u03c3_649_, lean_object* v_f_650_, lean_object* v_as_651_, lean_object* v_i_652_, lean_object* v_stop_653_, lean_object* v_b_654_){
_start:
{
size_t v_i_boxed_655_; size_t v_stop_boxed_656_; lean_object* v_res_657_; 
v_i_boxed_655_ = lean_unbox_usize(v_i_652_);
lean_dec(v_i_652_);
v_stop_boxed_656_ = lean_unbox_usize(v_stop_653_);
lean_dec(v_stop_653_);
v_res_657_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__9(v_00_u03b1_647_, v_00_u03b2_648_, v_00_u03c3_649_, v_f_650_, v_as_651_, v_i_boxed_655_, v_stop_boxed_656_, v_b_654_);
lean_dec_ref(v_as_651_);
return v_res_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10(lean_object* v_00_u03c3_658_, lean_object* v_00_u03b1_659_, lean_object* v_00_u03b2_660_, lean_object* v_f_661_, lean_object* v_keys_662_, lean_object* v_vals_663_, lean_object* v_heq_664_, lean_object* v_i_665_, lean_object* v_acc_666_){
_start:
{
lean_object* v___x_667_; 
v___x_667_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___redArg(v_f_661_, v_keys_662_, v_vals_663_, v_i_665_, v_acc_666_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10___boxed(lean_object* v_00_u03c3_668_, lean_object* v_00_u03b1_669_, lean_object* v_00_u03b2_670_, lean_object* v_f_671_, lean_object* v_keys_672_, lean_object* v_vals_673_, lean_object* v_heq_674_, lean_object* v_i_675_, lean_object* v_acc_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_Tactic_DeprecateTo_newNames_spec__1_spec__2_spec__4_spec__7_spec__10(v_00_u03c3_668_, v_00_u03b1_669_, v_00_u03b2_670_, v_f_671_, v_keys_672_, v_vals_673_, v_heq_674_, v_i_675_, v_acc_676_);
lean_dec_ref(v_vals_673_);
lean_dec_ref(v_keys_672_);
return v_res_677_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14(void){
_start:
{
uint8_t v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_714_ = 0;
v___x_715_ = lean_box(0);
v___x_716_ = l_Lean_SourceInfo_fromRef(v___x_715_, v___x_714_);
return v___x_716_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__15(void){
_start:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_717_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__12));
v___x_718_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14, &lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14);
v___x_719_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
lean_ctor_set(v___x_719_, 1, v___x_717_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem(lean_object* v_newName_720_, lean_object* v_x_721_){
_start:
{
lean_object* v___x_722_; uint8_t v___x_723_; 
v___x_722_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__1));
lean_inc(v_x_721_);
v___x_723_ = l_Lean_Syntax_isOfKind(v_x_721_, v___x_722_);
if (v___x_723_ == 0)
{
lean_object* v___x_724_; lean_object* v___x_725_; uint8_t v___x_726_; 
v___x_724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__2));
v___x_725_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__3));
lean_inc(v_x_721_);
v___x_726_ = l_Lean_Syntax_isOfKind(v_x_721_, v___x_725_);
if (v___x_726_ == 0)
{
lean_object* v___x_727_; lean_object* v___x_728_; 
lean_dec(v_newName_720_);
v___x_727_ = lean_box(0);
v___x_728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_727_);
lean_ctor_set(v___x_728_, 1, v_x_721_);
return v___x_728_;
}
else
{
lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; uint8_t v___x_732_; 
v___x_729_ = lean_unsigned_to_nat(0u);
v___x_730_ = l_Lean_Syntax_getArg(v_x_721_, v___x_729_);
v___x_731_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9));
lean_inc(v___x_730_);
v___x_732_ = l_Lean_Syntax_isOfKind(v___x_730_, v___x_731_);
if (v___x_732_ == 0)
{
lean_object* v___x_733_; lean_object* v___x_734_; 
lean_dec(v___x_730_);
lean_dec(v_newName_720_);
v___x_733_ = lean_box(0);
v___x_734_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_733_);
lean_ctor_set(v___x_734_, 1, v_x_721_);
return v___x_734_;
}
else
{
lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; uint8_t v___x_738_; 
v___x_735_ = lean_unsigned_to_nat(1u);
v___x_736_ = l_Lean_Syntax_getArg(v_x_721_, v___x_735_);
v___x_737_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__5));
lean_inc(v___x_736_);
v___x_738_ = l_Lean_Syntax_isOfKind(v___x_736_, v___x_737_);
if (v___x_738_ == 0)
{
lean_object* v___x_739_; lean_object* v___x_740_; 
lean_dec(v___x_736_);
lean_dec(v___x_730_);
lean_dec(v_newName_720_);
v___x_739_ = lean_box(0);
v___x_740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_739_);
lean_ctor_set(v___x_740_, 1, v_x_721_);
return v___x_740_;
}
else
{
lean_object* v_id_741_; lean_object* v___x_742_; uint8_t v___x_743_; 
v_id_741_ = l_Lean_Syntax_getArg(v___x_736_, v___x_735_);
v___x_742_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7));
lean_inc(v_id_741_);
v___x_743_ = l_Lean_Syntax_isOfKind(v_id_741_, v___x_742_);
if (v___x_743_ == 0)
{
lean_object* v___x_744_; lean_object* v___x_745_; 
lean_dec(v_id_741_);
lean_dec(v___x_736_);
lean_dec(v___x_730_);
lean_dec(v_newName_720_);
v___x_744_ = lean_box(0);
v___x_745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
lean_ctor_set(v___x_745_, 1, v_x_721_);
return v___x_745_;
}
else
{
lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; uint8_t v___x_749_; 
v___x_746_ = lean_unsigned_to_nat(2u);
v___x_747_ = l_Lean_Syntax_getArg(v___x_736_, v___x_746_);
v___x_748_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9));
lean_inc(v___x_747_);
v___x_749_ = l_Lean_Syntax_isOfKind(v___x_747_, v___x_748_);
if (v___x_749_ == 0)
{
lean_object* v___x_750_; lean_object* v___x_751_; 
lean_dec(v___x_747_);
lean_dec(v_id_741_);
lean_dec(v___x_736_);
lean_dec(v___x_730_);
lean_dec(v_newName_720_);
v___x_750_ = lean_box(0);
v___x_751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_751_, 0, v___x_750_);
lean_ctor_set(v___x_751_, 1, v_x_721_);
return v___x_751_;
}
else
{
lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; 
lean_dec(v_x_721_);
v___x_752_ = lean_unsigned_to_nat(3u);
v___x_753_ = l_Lean_Syntax_getArg(v___x_736_, v___x_752_);
lean_dec(v___x_736_);
v___x_754_ = lean_box(0);
v___x_755_ = l_Lean_SourceInfo_fromRef(v___x_754_, v___x_723_);
lean_inc_n(v___x_755_, 2);
v___x_756_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_756_, 0, v___x_755_);
lean_ctor_set(v___x_756_, 1, v___x_724_);
v___x_757_ = lean_box(2);
v___x_758_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__11));
v___x_759_ = lean_mk_empty_array_with_capacity(v___x_746_);
v___x_760_ = lean_array_push(v___x_759_, v_newName_720_);
v___x_761_ = lean_array_push(v___x_760_, v___x_758_);
v___x_762_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_762_, 0, v___x_757_);
lean_ctor_set(v___x_762_, 1, v___x_742_);
lean_ctor_set(v___x_762_, 2, v___x_761_);
v___x_763_ = l_Lean_Syntax_node4(v___x_755_, v___x_737_, v___x_756_, v___x_762_, v___x_747_, v___x_753_);
v___x_764_ = l_Lean_Syntax_node2(v___x_755_, v___x_725_, v___x_730_, v___x_763_);
v___x_765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_765_, 0, v_id_741_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
return v___x_765_;
}
}
}
}
}
}
else
{
lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; uint8_t v___x_769_; 
v___x_766_ = lean_unsigned_to_nat(0u);
v___x_767_ = l_Lean_Syntax_getArg(v_x_721_, v___x_766_);
v___x_768_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__9));
lean_inc(v___x_767_);
v___x_769_ = l_Lean_Syntax_isOfKind(v___x_767_, v___x_768_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; lean_object* v___x_771_; 
lean_dec(v___x_767_);
lean_dec(v_newName_720_);
v___x_770_ = lean_box(0);
v___x_771_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_771_, 0, v___x_770_);
lean_ctor_set(v___x_771_, 1, v_x_721_);
return v___x_771_;
}
else
{
lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; uint8_t v___x_775_; 
v___x_772_ = lean_unsigned_to_nat(1u);
v___x_773_ = l_Lean_Syntax_getArg(v_x_721_, v___x_772_);
v___x_774_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__13));
lean_inc(v___x_773_);
v___x_775_ = l_Lean_Syntax_isOfKind(v___x_773_, v___x_774_);
if (v___x_775_ == 0)
{
lean_object* v___x_776_; lean_object* v___x_777_; 
lean_dec(v___x_773_);
lean_dec(v___x_767_);
lean_dec(v_newName_720_);
v___x_776_ = lean_box(0);
v___x_777_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_777_, 0, v___x_776_);
lean_ctor_set(v___x_777_, 1, v_x_721_);
return v___x_777_;
}
else
{
lean_object* v_id_778_; lean_object* v___x_779_; uint8_t v___x_780_; 
v_id_778_ = l_Lean_Syntax_getArg(v___x_773_, v___x_772_);
v___x_779_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__7));
lean_inc(v_id_778_);
v___x_780_ = l_Lean_Syntax_isOfKind(v_id_778_, v___x_779_);
if (v___x_780_ == 0)
{
lean_object* v___x_781_; lean_object* v___x_782_; 
lean_dec(v_id_778_);
lean_dec(v___x_773_);
lean_dec(v___x_767_);
lean_dec(v_newName_720_);
v___x_781_ = lean_box(0);
v___x_782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_782_, 0, v___x_781_);
lean_ctor_set(v___x_782_, 1, v_x_721_);
return v___x_782_;
}
else
{
lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; uint8_t v___x_786_; 
v___x_783_ = lean_unsigned_to_nat(2u);
v___x_784_ = l_Lean_Syntax_getArg(v___x_773_, v___x_783_);
v___x_785_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__9));
lean_inc(v___x_784_);
v___x_786_ = l_Lean_Syntax_isOfKind(v___x_784_, v___x_785_);
if (v___x_786_ == 0)
{
lean_object* v___x_787_; lean_object* v___x_788_; 
lean_dec(v___x_784_);
lean_dec(v_id_778_);
lean_dec(v___x_773_);
lean_dec(v___x_767_);
lean_dec(v_newName_720_);
v___x_787_ = lean_box(0);
v___x_788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_788_, 0, v___x_787_);
lean_ctor_set(v___x_788_, 1, v_x_721_);
return v___x_788_;
}
else
{
lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; 
lean_dec(v_x_721_);
v___x_789_ = lean_unsigned_to_nat(3u);
v___x_790_ = l_Lean_Syntax_getArg(v___x_773_, v___x_789_);
lean_dec(v___x_773_);
v___x_791_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14, &lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__14);
v___x_792_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__15, &lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__15);
v___x_793_ = lean_box(2);
v___x_794_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem___closed__11));
v___x_795_ = lean_mk_empty_array_with_capacity(v___x_783_);
v___x_796_ = lean_array_push(v___x_795_, v_newName_720_);
v___x_797_ = lean_array_push(v___x_796_, v___x_794_);
v___x_798_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_798_, 0, v___x_793_);
lean_ctor_set(v___x_798_, 1, v___x_779_);
lean_ctor_set(v___x_798_, 2, v___x_797_);
v___x_799_ = l_Lean_Syntax_node4(v___x_791_, v___x_774_, v___x_792_, v___x_798_, v___x_784_, v___x_790_);
v___x_800_ = l_Lean_Syntax_node2(v___x_791_, v___x_722_, v___x_767_, v___x_799_);
v___x_801_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_801_, 0, v_id_778_);
lean_ctor_set(v___x_801_, 1, v___x_800_);
return v___x_801_;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_890_ = lean_box(0);
v___x_891_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_892_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_892_, 0, v___x_891_);
lean_ctor_set(v___x_892_, 1, v___x_890_);
return v___x_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg(){
_start:
{
lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_894_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___closed__0);
v___x_895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_895_, 0, v___x_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg___boxed(lean_object* v___y_896_){
_start:
{
lean_object* v_res_897_; 
v_res_897_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg();
return v_res_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0(lean_object* v_00_u03b1_898_, lean_object* v___y_899_, lean_object* v___y_900_){
_start:
{
lean_object* v___x_902_; 
v___x_902_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg();
return v___x_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___boxed(lean_object* v_00_u03b1_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_){
_start:
{
lean_object* v_res_907_; 
v_res_907_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0(v_00_u03b1_903_, v___y_904_, v___y_905_);
lean_dec(v___y_905_);
lean_dec_ref(v___y_904_);
return v_res_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6(lean_object* v_x_909_, lean_object* v_x_910_){
_start:
{
if (lean_obj_tag(v_x_910_) == 0)
{
return v_x_909_;
}
else
{
lean_object* v_head_911_; lean_object* v_tail_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; 
v_head_911_ = lean_ctor_get(v_x_910_, 0);
v_tail_912_ = lean_ctor_get(v_x_910_, 1);
v___x_913_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6___closed__0));
v___x_914_ = lean_string_append(v_x_909_, v___x_913_);
v___x_915_ = lean_string_append(v___x_914_, v_head_911_);
v_x_909_ = v___x_915_;
v_x_910_ = v_tail_912_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6___boxed(lean_object* v_x_917_, lean_object* v_x_918_){
_start:
{
lean_object* v_res_919_; 
v_res_919_ = lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6(v_x_917_, v_x_918_);
lean_dec(v_x_918_);
return v_res_919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg(size_t v_sz_920_, size_t v_i_921_, lean_object* v_bs_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
uint8_t v___x_926_; 
v___x_926_ = lean_usize_dec_lt(v_i_921_, v_sz_920_);
if (v___x_926_ == 0)
{
lean_object* v___x_927_; 
v___x_927_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_927_, 0, v_bs_922_);
return v___x_927_;
}
else
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v_v_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v___x_928_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__34));
v___x_929_ = lean_unsigned_to_nat(0u);
v_v_930_ = lean_array_uget_borrowed(v_bs_922_, v_i_921_);
lean_inc(v_v_930_);
v___x_931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_931_, 0, v___x_928_);
lean_ctor_set(v___x_931_, 1, v_v_930_);
v___x_932_ = lean_box(0);
v___x_933_ = l_Lean_Meta_Tactic_TryThis_SuggestionText_prettyExtra(v___x_931_, v___x_932_, v___x_929_, v___x_929_, v___y_923_, v___y_924_);
if (lean_obj_tag(v___x_933_) == 0)
{
lean_object* v_a_934_; lean_object* v_bs_x27_935_; size_t v___x_936_; size_t v___x_937_; lean_object* v___x_938_; 
v_a_934_ = lean_ctor_get(v___x_933_, 0);
lean_inc(v_a_934_);
lean_dec_ref_known(v___x_933_, 1);
v_bs_x27_935_ = lean_array_uset(v_bs_922_, v_i_921_, v___x_929_);
v___x_936_ = ((size_t)1ULL);
v___x_937_ = lean_usize_add(v_i_921_, v___x_936_);
v___x_938_ = lean_array_uset(v_bs_x27_935_, v_i_921_, v_a_934_);
v_i_921_ = v___x_937_;
v_bs_922_ = v___x_938_;
goto _start;
}
else
{
lean_object* v_a_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_947_; 
lean_dec_ref(v_bs_922_);
v_a_940_ = lean_ctor_get(v___x_933_, 0);
v_isSharedCheck_947_ = !lean_is_exclusive(v___x_933_);
if (v_isSharedCheck_947_ == 0)
{
v___x_942_ = v___x_933_;
v_isShared_943_ = v_isSharedCheck_947_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_a_940_);
lean_dec(v___x_933_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_947_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v___x_945_; 
if (v_isShared_943_ == 0)
{
v___x_945_ = v___x_942_;
goto v_reusejp_944_;
}
else
{
lean_object* v_reuseFailAlloc_946_; 
v_reuseFailAlloc_946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_946_, 0, v_a_940_);
v___x_945_ = v_reuseFailAlloc_946_;
goto v_reusejp_944_;
}
v_reusejp_944_:
{
return v___x_945_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg___boxed(lean_object* v_sz_948_, lean_object* v_i_949_, lean_object* v_bs_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
size_t v_sz_boxed_954_; size_t v_i_boxed_955_; lean_object* v_res_956_; 
v_sz_boxed_954_ = lean_unbox_usize(v_sz_948_);
lean_dec(v_sz_948_);
v_i_boxed_955_ = lean_unbox_usize(v_i_949_);
lean_dec(v_i_949_);
v_res_956_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg(v_sz_boxed_954_, v_i_boxed_955_, v_bs_950_, v___y_951_, v___y_952_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
return v_res_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0(lean_object* v___x_959_, size_t v___x_960_, lean_object* v___x_961_, lean_object* v___x_962_, lean_object* v___x_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_){
_start:
{
size_t v_sz_971_; lean_object* v___x_972_; 
v_sz_971_ = lean_array_size(v___x_959_);
v___x_972_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg(v_sz_971_, v___x_960_, v___x_959_, v___y_968_, v___y_969_);
if (lean_obj_tag(v___x_972_) == 0)
{
lean_object* v_a_973_; lean_object* v_ref_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; uint8_t v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; 
v_a_973_ = lean_ctor_get(v___x_972_, 0);
lean_inc(v_a_973_);
lean_dec_ref_known(v___x_972_, 1);
v_ref_974_ = lean_ctor_get(v___y_968_, 5);
lean_inc(v_ref_974_);
v___x_975_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0));
v___x_976_ = lean_array_get(v___x_975_, v_a_973_, v___x_961_);
v___x_977_ = lean_array_to_list(v_a_973_);
v___x_978_ = l_List_drop___redArg(v___x_962_, v___x_977_);
lean_dec(v___x_977_);
v___x_979_ = lp_mathlib_List_foldl___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__6(v___x_976_, v___x_978_);
lean_dec(v___x_978_);
v___x_980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_980_, 0, v___x_979_);
v___x_981_ = lean_box(0);
v___x_982_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_982_, 0, v___x_980_);
lean_ctor_set(v___x_982_, 1, v___x_981_);
lean_ctor_set(v___x_982_, 2, v___x_981_);
lean_ctor_set(v___x_982_, 3, v___x_981_);
lean_ctor_set(v___x_982_, 4, v___x_981_);
lean_ctor_set(v___x_982_, 5, v___x_981_);
v___x_983_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__1));
v___x_984_ = lean_string_append(v___x_963_, v___x_983_);
v___x_985_ = 4;
v___x_986_ = l_Lean_MessageData_nil;
v___x_987_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_ref_974_, v___x_982_, v___x_981_, v___x_984_, v___x_981_, v___x_985_, v___x_986_, v___y_968_, v___y_969_);
lean_dec_ref(v___y_968_);
return v___x_987_;
}
else
{
lean_object* v_a_988_; lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_995_; 
lean_dec_ref(v___y_968_);
lean_dec_ref(v___x_963_);
lean_dec(v___x_962_);
v_a_988_ = lean_ctor_get(v___x_972_, 0);
v_isSharedCheck_995_ = !lean_is_exclusive(v___x_972_);
if (v_isSharedCheck_995_ == 0)
{
v___x_990_ = v___x_972_;
v_isShared_991_ = v_isSharedCheck_995_;
goto v_resetjp_989_;
}
else
{
lean_inc(v_a_988_);
lean_dec(v___x_972_);
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
return v___x_993_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___boxed(lean_object* v___x_996_, lean_object* v___x_997_, lean_object* v___x_998_, lean_object* v___x_999_, lean_object* v___x_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_){
_start:
{
size_t v___x_22608__boxed_1008_; lean_object* v_res_1009_; 
v___x_22608__boxed_1008_ = lean_unbox_usize(v___x_997_);
lean_dec(v___x_997_);
v_res_1009_ = lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0(v___x_996_, v___x_22608__boxed_1008_, v___x_998_, v___x_999_, v___x_1000_, v___y_1001_, v___y_1002_, v___y_1003_, v___y_1004_, v___y_1005_, v___y_1006_);
lean_dec(v___y_1006_);
lean_dec(v___y_1004_);
lean_dec_ref(v___y_1003_);
lean_dec(v___y_1002_);
lean_dec_ref(v___y_1001_);
lean_dec(v___x_998_);
return v_res_1009_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__0(void){
_start:
{
lean_object* v___x_1010_; 
v___x_1010_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1010_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1(void){
_start:
{
lean_object* v___x_1011_; lean_object* v___x_1012_; 
v___x_1011_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__0);
v___x_1012_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1012_, 0, v___x_1011_);
return v___x_1012_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__2(void){
_start:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; 
v___x_1013_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1);
v___x_1014_ = lean_unsigned_to_nat(0u);
v___x_1015_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1014_);
lean_ctor_set(v___x_1015_, 1, v___x_1014_);
lean_ctor_set(v___x_1015_, 2, v___x_1014_);
lean_ctor_set(v___x_1015_, 3, v___x_1014_);
lean_ctor_set(v___x_1015_, 4, v___x_1013_);
lean_ctor_set(v___x_1015_, 5, v___x_1013_);
lean_ctor_set(v___x_1015_, 6, v___x_1013_);
lean_ctor_set(v___x_1015_, 7, v___x_1013_);
lean_ctor_set(v___x_1015_, 8, v___x_1013_);
lean_ctor_set(v___x_1015_, 9, v___x_1013_);
return v___x_1015_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; 
v___x_1016_ = lean_unsigned_to_nat(32u);
v___x_1017_ = lean_mk_empty_array_with_capacity(v___x_1016_);
v___x_1018_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1018_, 0, v___x_1017_);
return v___x_1018_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__4(void){
_start:
{
size_t v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; 
v___x_1019_ = ((size_t)5ULL);
v___x_1020_ = lean_unsigned_to_nat(0u);
v___x_1021_ = lean_unsigned_to_nat(32u);
v___x_1022_ = lean_mk_empty_array_with_capacity(v___x_1021_);
v___x_1023_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__3);
v___x_1024_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1024_, 0, v___x_1023_);
lean_ctor_set(v___x_1024_, 1, v___x_1022_);
lean_ctor_set(v___x_1024_, 2, v___x_1020_);
lean_ctor_set(v___x_1024_, 3, v___x_1020_);
lean_ctor_set_usize(v___x_1024_, 4, v___x_1019_);
return v___x_1024_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__5(void){
_start:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; 
v___x_1025_ = lean_box(1);
v___x_1026_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__4);
v___x_1027_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__1);
v___x_1028_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1027_);
lean_ctor_set(v___x_1028_, 1, v___x_1026_);
lean_ctor_set(v___x_1028_, 2, v___x_1025_);
return v___x_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg(lean_object* v_msgData_1029_, lean_object* v___y_1030_){
_start:
{
lean_object* v___x_1032_; lean_object* v_env_1033_; lean_object* v___x_1034_; lean_object* v_scopes_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v_opts_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; 
v___x_1032_ = lean_st_ref_get(v___y_1030_);
v_env_1033_ = lean_ctor_get(v___x_1032_, 0);
lean_inc_ref(v_env_1033_);
lean_dec(v___x_1032_);
v___x_1034_ = lean_st_ref_get(v___y_1030_);
v_scopes_1035_ = lean_ctor_get(v___x_1034_, 2);
lean_inc(v_scopes_1035_);
lean_dec(v___x_1034_);
v___x_1036_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1037_ = l_List_head_x21___redArg(v___x_1036_, v_scopes_1035_);
lean_dec(v_scopes_1035_);
v_opts_1038_ = lean_ctor_get(v___x_1037_, 1);
lean_inc_ref(v_opts_1038_);
lean_dec(v___x_1037_);
v___x_1039_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__2);
v___x_1040_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___closed__5);
v___x_1041_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1041_, 0, v_env_1033_);
lean_ctor_set(v___x_1041_, 1, v___x_1039_);
lean_ctor_set(v___x_1041_, 2, v___x_1040_);
lean_ctor_set(v___x_1041_, 3, v_opts_1038_);
v___x_1042_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1041_);
lean_ctor_set(v___x_1042_, 1, v_msgData_1029_);
v___x_1043_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1043_, 0, v___x_1042_);
return v___x_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg___boxed(lean_object* v_msgData_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v_res_1047_; 
v_res_1047_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg(v_msgData_1044_, v___y_1045_);
lean_dec(v___y_1045_);
return v_res_1047_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12(lean_object* v_opts_1048_, lean_object* v_opt_1049_){
_start:
{
lean_object* v_name_1050_; lean_object* v_defValue_1051_; lean_object* v_map_1052_; lean_object* v___x_1053_; 
v_name_1050_ = lean_ctor_get(v_opt_1049_, 0);
v_defValue_1051_ = lean_ctor_get(v_opt_1049_, 1);
v_map_1052_ = lean_ctor_get(v_opts_1048_, 0);
v___x_1053_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1052_, v_name_1050_);
if (lean_obj_tag(v___x_1053_) == 0)
{
uint8_t v___x_1054_; 
v___x_1054_ = lean_unbox(v_defValue_1051_);
return v___x_1054_;
}
else
{
lean_object* v_val_1055_; 
v_val_1055_ = lean_ctor_get(v___x_1053_, 0);
lean_inc(v_val_1055_);
lean_dec_ref_known(v___x_1053_, 1);
if (lean_obj_tag(v_val_1055_) == 1)
{
uint8_t v_v_1056_; 
v_v_1056_ = lean_ctor_get_uint8(v_val_1055_, 0);
lean_dec_ref_known(v_val_1055_, 0);
return v_v_1056_;
}
else
{
uint8_t v___x_1057_; 
lean_dec(v_val_1055_);
v___x_1057_ = lean_unbox(v_defValue_1051_);
return v___x_1057_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12___boxed(lean_object* v_opts_1058_, lean_object* v_opt_1059_){
_start:
{
uint8_t v_res_1060_; lean_object* v_r_1061_; 
v_res_1060_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12(v_opts_1058_, v_opt_1059_);
lean_dec_ref(v_opt_1059_);
lean_dec_ref(v_opts_1058_);
v_r_1061_ = lean_box(v_res_1060_);
return v_r_1061_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0(uint8_t v___y_1063_, uint8_t v_suppressElabErrors_1064_, lean_object* v_x_1065_){
_start:
{
if (lean_obj_tag(v_x_1065_) == 1)
{
lean_object* v_pre_1066_; 
v_pre_1066_ = lean_ctor_get(v_x_1065_, 0);
if (lean_obj_tag(v_pre_1066_) == 0)
{
lean_object* v_str_1067_; lean_object* v___x_1068_; uint8_t v___x_1069_; 
v_str_1067_ = lean_ctor_get(v_x_1065_, 1);
v___x_1068_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___closed__0));
v___x_1069_ = lean_string_dec_eq(v_str_1067_, v___x_1068_);
if (v___x_1069_ == 0)
{
return v___y_1063_;
}
else
{
return v_suppressElabErrors_1064_;
}
}
else
{
return v___y_1063_;
}
}
else
{
return v___y_1063_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___boxed(lean_object* v___y_1070_, lean_object* v_suppressElabErrors_1071_, lean_object* v_x_1072_){
_start:
{
uint8_t v___y_22802__boxed_1073_; uint8_t v_suppressElabErrors_boxed_1074_; uint8_t v_res_1075_; lean_object* v_r_1076_; 
v___y_22802__boxed_1073_ = lean_unbox(v___y_1070_);
v_suppressElabErrors_boxed_1074_ = lean_unbox(v_suppressElabErrors_1071_);
v_res_1075_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0(v___y_22802__boxed_1073_, v_suppressElabErrors_boxed_1074_, v_x_1072_);
lean_dec(v_x_1072_);
v_r_1076_ = lean_box(v_res_1075_);
return v_r_1076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10(lean_object* v_ref_1077_, lean_object* v_msgData_1078_, uint8_t v_severity_1079_, uint8_t v_isSilent_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_){
_start:
{
lean_object* v___y_1085_; lean_object* v___y_1086_; lean_object* v___y_1087_; lean_object* v___y_1088_; uint8_t v___y_1089_; uint8_t v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1092_; uint8_t v___y_1149_; uint8_t v___y_1150_; uint8_t v___y_1151_; lean_object* v___y_1152_; lean_object* v___y_1153_; uint8_t v___y_1177_; uint8_t v___y_1178_; uint8_t v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; uint8_t v___y_1185_; uint8_t v___y_1186_; uint8_t v___y_1187_; uint8_t v___x_1202_; uint8_t v___y_1204_; uint8_t v___y_1205_; uint8_t v___y_1206_; uint8_t v___y_1208_; uint8_t v___x_1220_; 
v___x_1202_ = 2;
v___x_1220_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1079_, v___x_1202_);
if (v___x_1220_ == 0)
{
v___y_1208_ = v___x_1220_;
goto v___jp_1207_;
}
else
{
uint8_t v___x_1221_; 
lean_inc_ref(v_msgData_1078_);
v___x_1221_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1078_);
v___y_1208_ = v___x_1221_;
goto v___jp_1207_;
}
v___jp_1084_:
{
lean_object* v___x_1093_; 
v___x_1093_ = l_Lean_Elab_Command_getScope___redArg(v___y_1092_);
if (lean_obj_tag(v___x_1093_) == 0)
{
lean_object* v_a_1094_; lean_object* v___x_1095_; 
v_a_1094_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_a_1094_);
lean_dec_ref_known(v___x_1093_, 1);
v___x_1095_ = l_Lean_Elab_Command_getScope___redArg(v___y_1092_);
if (lean_obj_tag(v___x_1095_) == 0)
{
lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1131_; 
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1098_ = v___x_1095_;
v_isShared_1099_ = v_isSharedCheck_1131_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v___x_1095_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1131_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1100_; lean_object* v_currNamespace_1101_; lean_object* v_openDecls_1102_; lean_object* v_env_1103_; lean_object* v_messages_1104_; lean_object* v_scopes_1105_; lean_object* v_usedQuotCtxts_1106_; lean_object* v_nextMacroScope_1107_; lean_object* v_maxRecDepth_1108_; lean_object* v_ngen_1109_; lean_object* v_auxDeclNGen_1110_; lean_object* v_infoState_1111_; lean_object* v_traceState_1112_; lean_object* v_snapshotTasks_1113_; lean_object* v_prevLinterStates_1114_; lean_object* v___x_1116_; uint8_t v_isShared_1117_; uint8_t v_isSharedCheck_1130_; 
v___x_1100_ = lean_st_ref_take(v___y_1092_);
v_currNamespace_1101_ = lean_ctor_get(v_a_1094_, 2);
lean_inc(v_currNamespace_1101_);
lean_dec(v_a_1094_);
v_openDecls_1102_ = lean_ctor_get(v_a_1096_, 3);
lean_inc(v_openDecls_1102_);
lean_dec(v_a_1096_);
v_env_1103_ = lean_ctor_get(v___x_1100_, 0);
v_messages_1104_ = lean_ctor_get(v___x_1100_, 1);
v_scopes_1105_ = lean_ctor_get(v___x_1100_, 2);
v_usedQuotCtxts_1106_ = lean_ctor_get(v___x_1100_, 3);
v_nextMacroScope_1107_ = lean_ctor_get(v___x_1100_, 4);
v_maxRecDepth_1108_ = lean_ctor_get(v___x_1100_, 5);
v_ngen_1109_ = lean_ctor_get(v___x_1100_, 6);
v_auxDeclNGen_1110_ = lean_ctor_get(v___x_1100_, 7);
v_infoState_1111_ = lean_ctor_get(v___x_1100_, 8);
v_traceState_1112_ = lean_ctor_get(v___x_1100_, 9);
v_snapshotTasks_1113_ = lean_ctor_get(v___x_1100_, 10);
v_prevLinterStates_1114_ = lean_ctor_get(v___x_1100_, 11);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1116_ = v___x_1100_;
v_isShared_1117_ = v_isSharedCheck_1130_;
goto v_resetjp_1115_;
}
else
{
lean_inc(v_prevLinterStates_1114_);
lean_inc(v_snapshotTasks_1113_);
lean_inc(v_traceState_1112_);
lean_inc(v_infoState_1111_);
lean_inc(v_auxDeclNGen_1110_);
lean_inc(v_ngen_1109_);
lean_inc(v_maxRecDepth_1108_);
lean_inc(v_nextMacroScope_1107_);
lean_inc(v_usedQuotCtxts_1106_);
lean_inc(v_scopes_1105_);
lean_inc(v_messages_1104_);
lean_inc(v_env_1103_);
lean_dec(v___x_1100_);
v___x_1116_ = lean_box(0);
v_isShared_1117_ = v_isSharedCheck_1130_;
goto v_resetjp_1115_;
}
v_resetjp_1115_:
{
lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1123_; 
v___x_1118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1118_, 0, v_currNamespace_1101_);
lean_ctor_set(v___x_1118_, 1, v_openDecls_1102_);
v___x_1119_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1119_, 0, v___x_1118_);
lean_ctor_set(v___x_1119_, 1, v___y_1091_);
lean_inc_ref(v___y_1086_);
lean_inc_ref(v___y_1085_);
v___x_1120_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1120_, 0, v___y_1085_);
lean_ctor_set(v___x_1120_, 1, v___y_1088_);
lean_ctor_set(v___x_1120_, 2, v___y_1087_);
lean_ctor_set(v___x_1120_, 3, v___y_1086_);
lean_ctor_set(v___x_1120_, 4, v___x_1119_);
lean_ctor_set_uint8(v___x_1120_, sizeof(void*)*5, v___y_1090_);
lean_ctor_set_uint8(v___x_1120_, sizeof(void*)*5 + 1, v___y_1089_);
lean_ctor_set_uint8(v___x_1120_, sizeof(void*)*5 + 2, v_isSilent_1080_);
v___x_1121_ = l_Lean_MessageLog_add(v___x_1120_, v_messages_1104_);
if (v_isShared_1117_ == 0)
{
lean_ctor_set(v___x_1116_, 1, v___x_1121_);
v___x_1123_ = v___x_1116_;
goto v_reusejp_1122_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_env_1103_);
lean_ctor_set(v_reuseFailAlloc_1129_, 1, v___x_1121_);
lean_ctor_set(v_reuseFailAlloc_1129_, 2, v_scopes_1105_);
lean_ctor_set(v_reuseFailAlloc_1129_, 3, v_usedQuotCtxts_1106_);
lean_ctor_set(v_reuseFailAlloc_1129_, 4, v_nextMacroScope_1107_);
lean_ctor_set(v_reuseFailAlloc_1129_, 5, v_maxRecDepth_1108_);
lean_ctor_set(v_reuseFailAlloc_1129_, 6, v_ngen_1109_);
lean_ctor_set(v_reuseFailAlloc_1129_, 7, v_auxDeclNGen_1110_);
lean_ctor_set(v_reuseFailAlloc_1129_, 8, v_infoState_1111_);
lean_ctor_set(v_reuseFailAlloc_1129_, 9, v_traceState_1112_);
lean_ctor_set(v_reuseFailAlloc_1129_, 10, v_snapshotTasks_1113_);
lean_ctor_set(v_reuseFailAlloc_1129_, 11, v_prevLinterStates_1114_);
v___x_1123_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1122_;
}
v_reusejp_1122_:
{
lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1127_; 
v___x_1124_ = lean_st_ref_set(v___y_1092_, v___x_1123_);
v___x_1125_ = lean_box(0);
if (v_isShared_1099_ == 0)
{
lean_ctor_set(v___x_1098_, 0, v___x_1125_);
v___x_1127_ = v___x_1098_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1139_; 
lean_dec(v_a_1094_);
lean_dec_ref(v___y_1091_);
lean_dec_ref(v___y_1088_);
lean_dec(v___y_1087_);
v_a_1132_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1139_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1139_ == 0)
{
v___x_1134_ = v___x_1095_;
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_a_1132_);
lean_dec(v___x_1095_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1139_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1137_; 
if (v_isShared_1135_ == 0)
{
v___x_1137_ = v___x_1134_;
goto v_reusejp_1136_;
}
else
{
lean_object* v_reuseFailAlloc_1138_; 
v_reuseFailAlloc_1138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1138_, 0, v_a_1132_);
v___x_1137_ = v_reuseFailAlloc_1138_;
goto v_reusejp_1136_;
}
v_reusejp_1136_:
{
return v___x_1137_;
}
}
}
}
else
{
lean_object* v_a_1140_; lean_object* v___x_1142_; uint8_t v_isShared_1143_; uint8_t v_isSharedCheck_1147_; 
lean_dec_ref(v___y_1091_);
lean_dec_ref(v___y_1088_);
lean_dec(v___y_1087_);
v_a_1140_ = lean_ctor_get(v___x_1093_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v___x_1093_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1142_ = v___x_1093_;
v_isShared_1143_ = v_isSharedCheck_1147_;
goto v_resetjp_1141_;
}
else
{
lean_inc(v_a_1140_);
lean_dec(v___x_1093_);
v___x_1142_ = lean_box(0);
v_isShared_1143_ = v_isSharedCheck_1147_;
goto v_resetjp_1141_;
}
v_resetjp_1141_:
{
lean_object* v___x_1145_; 
if (v_isShared_1143_ == 0)
{
v___x_1145_ = v___x_1142_;
goto v_reusejp_1144_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v_a_1140_);
v___x_1145_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1144_;
}
v_reusejp_1144_:
{
return v___x_1145_;
}
}
}
}
v___jp_1148_:
{
lean_object* v_fileName_1154_; lean_object* v_fileMap_1155_; uint8_t v_suppressElabErrors_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v_a_1159_; lean_object* v___x_1161_; uint8_t v_isShared_1162_; uint8_t v_isSharedCheck_1175_; 
v_fileName_1154_ = lean_ctor_get(v___y_1081_, 0);
v_fileMap_1155_ = lean_ctor_get(v___y_1081_, 1);
v_suppressElabErrors_1156_ = lean_ctor_get_uint8(v___y_1081_, sizeof(void*)*10);
v___x_1157_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1078_);
v___x_1158_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg(v___x_1157_, v___y_1082_);
v_a_1159_ = lean_ctor_get(v___x_1158_, 0);
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1158_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1161_ = v___x_1158_;
v_isShared_1162_ = v_isSharedCheck_1175_;
goto v_resetjp_1160_;
}
else
{
lean_inc(v_a_1159_);
lean_dec(v___x_1158_);
v___x_1161_ = lean_box(0);
v_isShared_1162_ = v_isSharedCheck_1175_;
goto v_resetjp_1160_;
}
v_resetjp_1160_:
{
lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; 
lean_inc_ref_n(v_fileMap_1155_, 2);
v___x_1163_ = l_Lean_FileMap_toPosition(v_fileMap_1155_, v___y_1152_);
lean_dec(v___y_1152_);
v___x_1164_ = l_Lean_FileMap_toPosition(v_fileMap_1155_, v___y_1153_);
lean_dec(v___y_1153_);
v___x_1165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1165_, 0, v___x_1164_);
v___x_1166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0));
if (v_suppressElabErrors_1156_ == 0)
{
lean_del_object(v___x_1161_);
v___y_1085_ = v_fileName_1154_;
v___y_1086_ = v___x_1166_;
v___y_1087_ = v___x_1165_;
v___y_1088_ = v___x_1163_;
v___y_1089_ = v___y_1151_;
v___y_1090_ = v___y_1150_;
v___y_1091_ = v_a_1159_;
v___y_1092_ = v___y_1082_;
goto v___jp_1084_;
}
else
{
lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___f_1169_; uint8_t v___x_1170_; 
v___x_1167_ = lean_box(v___y_1149_);
v___x_1168_ = lean_box(v_suppressElabErrors_1156_);
v___f_1169_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1169_, 0, v___x_1167_);
lean_closure_set(v___f_1169_, 1, v___x_1168_);
lean_inc(v_a_1159_);
v___x_1170_ = l_Lean_MessageData_hasTag(v___f_1169_, v_a_1159_);
if (v___x_1170_ == 0)
{
lean_object* v___x_1171_; lean_object* v___x_1173_; 
lean_dec_ref_known(v___x_1165_, 1);
lean_dec_ref(v___x_1163_);
lean_dec(v_a_1159_);
v___x_1171_ = lean_box(0);
if (v_isShared_1162_ == 0)
{
lean_ctor_set(v___x_1161_, 0, v___x_1171_);
v___x_1173_ = v___x_1161_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1174_; 
v_reuseFailAlloc_1174_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1174_, 0, v___x_1171_);
v___x_1173_ = v_reuseFailAlloc_1174_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
return v___x_1173_;
}
}
else
{
lean_del_object(v___x_1161_);
v___y_1085_ = v_fileName_1154_;
v___y_1086_ = v___x_1166_;
v___y_1087_ = v___x_1165_;
v___y_1088_ = v___x_1163_;
v___y_1089_ = v___y_1151_;
v___y_1090_ = v___y_1150_;
v___y_1091_ = v_a_1159_;
v___y_1092_ = v___y_1082_;
goto v___jp_1084_;
}
}
}
}
v___jp_1176_:
{
lean_object* v___x_1182_; 
v___x_1182_ = l_Lean_Syntax_getTailPos_x3f(v___y_1180_, v___y_1179_);
lean_dec(v___y_1180_);
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_inc(v___y_1181_);
v___y_1149_ = v___y_1177_;
v___y_1150_ = v___y_1179_;
v___y_1151_ = v___y_1178_;
v___y_1152_ = v___y_1181_;
v___y_1153_ = v___y_1181_;
goto v___jp_1148_;
}
else
{
lean_object* v_val_1183_; 
v_val_1183_ = lean_ctor_get(v___x_1182_, 0);
lean_inc(v_val_1183_);
lean_dec_ref_known(v___x_1182_, 1);
v___y_1149_ = v___y_1177_;
v___y_1150_ = v___y_1179_;
v___y_1151_ = v___y_1178_;
v___y_1152_ = v___y_1181_;
v___y_1153_ = v_val_1183_;
goto v___jp_1148_;
}
}
v___jp_1184_:
{
lean_object* v___x_1188_; 
v___x_1188_ = l_Lean_Elab_Command_getRef___redArg(v___y_1081_);
if (lean_obj_tag(v___x_1188_) == 0)
{
lean_object* v_a_1189_; lean_object* v_ref_1190_; lean_object* v___x_1191_; 
v_a_1189_ = lean_ctor_get(v___x_1188_, 0);
lean_inc(v_a_1189_);
lean_dec_ref_known(v___x_1188_, 1);
v_ref_1190_ = l_Lean_replaceRef(v_ref_1077_, v_a_1189_);
lean_dec(v_a_1189_);
v___x_1191_ = l_Lean_Syntax_getPos_x3f(v_ref_1190_, v___y_1186_);
if (lean_obj_tag(v___x_1191_) == 0)
{
lean_object* v___x_1192_; 
v___x_1192_ = lean_unsigned_to_nat(0u);
v___y_1177_ = v___y_1185_;
v___y_1178_ = v___y_1187_;
v___y_1179_ = v___y_1186_;
v___y_1180_ = v_ref_1190_;
v___y_1181_ = v___x_1192_;
goto v___jp_1176_;
}
else
{
lean_object* v_val_1193_; 
v_val_1193_ = lean_ctor_get(v___x_1191_, 0);
lean_inc(v_val_1193_);
lean_dec_ref_known(v___x_1191_, 1);
v___y_1177_ = v___y_1185_;
v___y_1178_ = v___y_1187_;
v___y_1179_ = v___y_1186_;
v___y_1180_ = v_ref_1190_;
v___y_1181_ = v_val_1193_;
goto v___jp_1176_;
}
}
else
{
lean_object* v_a_1194_; lean_object* v___x_1196_; uint8_t v_isShared_1197_; uint8_t v_isSharedCheck_1201_; 
lean_dec_ref(v_msgData_1078_);
v_a_1194_ = lean_ctor_get(v___x_1188_, 0);
v_isSharedCheck_1201_ = !lean_is_exclusive(v___x_1188_);
if (v_isSharedCheck_1201_ == 0)
{
v___x_1196_ = v___x_1188_;
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
else
{
lean_inc(v_a_1194_);
lean_dec(v___x_1188_);
v___x_1196_ = lean_box(0);
v_isShared_1197_ = v_isSharedCheck_1201_;
goto v_resetjp_1195_;
}
v_resetjp_1195_:
{
lean_object* v___x_1199_; 
if (v_isShared_1197_ == 0)
{
v___x_1199_ = v___x_1196_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v_a_1194_);
v___x_1199_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
return v___x_1199_;
}
}
}
}
v___jp_1203_:
{
if (v___y_1206_ == 0)
{
v___y_1185_ = v___y_1204_;
v___y_1186_ = v___y_1205_;
v___y_1187_ = v_severity_1079_;
goto v___jp_1184_;
}
else
{
v___y_1185_ = v___y_1204_;
v___y_1186_ = v___y_1205_;
v___y_1187_ = v___x_1202_;
goto v___jp_1184_;
}
}
v___jp_1207_:
{
if (v___y_1208_ == 0)
{
lean_object* v___x_1209_; lean_object* v_scopes_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v_opts_1213_; uint8_t v___x_1214_; uint8_t v___x_1215_; 
v___x_1209_ = lean_st_ref_get(v___y_1082_);
v_scopes_1210_ = lean_ctor_get(v___x_1209_, 2);
lean_inc(v_scopes_1210_);
lean_dec(v___x_1209_);
v___x_1211_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1212_ = l_List_head_x21___redArg(v___x_1211_, v_scopes_1210_);
lean_dec(v_scopes_1210_);
v_opts_1213_ = lean_ctor_get(v___x_1212_, 1);
lean_inc_ref(v_opts_1213_);
lean_dec(v___x_1212_);
v___x_1214_ = 1;
v___x_1215_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1079_, v___x_1214_);
if (v___x_1215_ == 0)
{
lean_dec_ref(v_opts_1213_);
v___y_1204_ = v___y_1208_;
v___y_1205_ = v___y_1208_;
v___y_1206_ = v___x_1215_;
goto v___jp_1203_;
}
else
{
lean_object* v___x_1216_; uint8_t v___x_1217_; 
v___x_1216_ = l_Lean_warningAsError;
v___x_1217_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12(v_opts_1213_, v___x_1216_);
lean_dec_ref(v_opts_1213_);
v___y_1204_ = v___y_1208_;
v___y_1205_ = v___y_1208_;
v___y_1206_ = v___x_1217_;
goto v___jp_1203_;
}
}
else
{
lean_object* v___x_1218_; lean_object* v___x_1219_; 
lean_dec_ref(v_msgData_1078_);
v___x_1218_ = lean_box(0);
v___x_1219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1218_);
return v___x_1219_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10___boxed(lean_object* v_ref_1222_, lean_object* v_msgData_1223_, lean_object* v_severity_1224_, lean_object* v_isSilent_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_){
_start:
{
uint8_t v_severity_boxed_1229_; uint8_t v_isSilent_boxed_1230_; lean_object* v_res_1231_; 
v_severity_boxed_1229_ = lean_unbox(v_severity_1224_);
v_isSilent_boxed_1230_ = lean_unbox(v_isSilent_1225_);
v_res_1231_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10(v_ref_1222_, v_msgData_1223_, v_severity_boxed_1229_, v_isSilent_boxed_1230_, v___y_1226_, v___y_1227_);
lean_dec(v___y_1227_);
lean_dec_ref(v___y_1226_);
lean_dec(v_ref_1222_);
return v_res_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7(lean_object* v_ref_1232_, lean_object* v_msgData_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_){
_start:
{
uint8_t v___x_1237_; uint8_t v___x_1238_; lean_object* v___x_1239_; 
v___x_1237_ = 1;
v___x_1238_ = 0;
v___x_1239_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10(v_ref_1232_, v_msgData_1233_, v___x_1237_, v___x_1238_, v___y_1234_, v___y_1235_);
return v___x_1239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7___boxed(lean_object* v_ref_1240_, lean_object* v_msgData_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7(v_ref_1240_, v_msgData_1241_, v___y_1242_, v___y_1243_);
lean_dec(v___y_1243_);
lean_dec_ref(v___y_1242_);
lean_dec(v_ref_1240_);
return v_res_1245_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg(lean_object* v_xs_1246_, lean_object* v_ys_1247_, lean_object* v_x_1248_){
_start:
{
lean_object* v_zero_1249_; uint8_t v_isZero_1250_; 
v_zero_1249_ = lean_unsigned_to_nat(0u);
v_isZero_1250_ = lean_nat_dec_eq(v_x_1248_, v_zero_1249_);
if (v_isZero_1250_ == 1)
{
lean_dec(v_x_1248_);
return v_isZero_1250_;
}
else
{
lean_object* v_one_1251_; lean_object* v_n_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; uint8_t v___x_1255_; 
v_one_1251_ = lean_unsigned_to_nat(1u);
v_n_1252_ = lean_nat_sub(v_x_1248_, v_one_1251_);
lean_dec(v_x_1248_);
v___x_1253_ = lean_array_fget_borrowed(v_xs_1246_, v_n_1252_);
v___x_1254_ = lean_array_fget_borrowed(v_ys_1247_, v_n_1252_);
v___x_1255_ = lean_string_dec_eq(v___x_1253_, v___x_1254_);
if (v___x_1255_ == 0)
{
lean_dec(v_n_1252_);
return v___x_1255_;
}
else
{
v_x_1248_ = v_n_1252_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg___boxed(lean_object* v_xs_1257_, lean_object* v_ys_1258_, lean_object* v_x_1259_){
_start:
{
uint8_t v_res_1260_; lean_object* v_r_1261_; 
v_res_1260_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg(v_xs_1257_, v_ys_1258_, v_x_1259_);
lean_dec_ref(v_ys_1258_);
lean_dec_ref(v_xs_1257_);
v_r_1261_ = lean_box(v_res_1260_);
return v_r_1261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19_spec__24(lean_object* v_xs_1262_, lean_object* v_v_1263_, lean_object* v_i_1264_){
_start:
{
lean_object* v___x_1265_; uint8_t v___x_1266_; 
v___x_1265_ = lean_array_get_size(v_xs_1262_);
v___x_1266_ = lean_nat_dec_lt(v_i_1264_, v___x_1265_);
if (v___x_1266_ == 0)
{
lean_object* v___x_1267_; 
lean_dec(v_i_1264_);
v___x_1267_ = lean_box(0);
return v___x_1267_;
}
else
{
lean_object* v___x_1268_; uint8_t v___x_1269_; 
v___x_1268_ = lean_array_fget_borrowed(v_xs_1262_, v_i_1264_);
v___x_1269_ = lean_name_eq(v___x_1268_, v_v_1263_);
if (v___x_1269_ == 0)
{
lean_object* v___x_1270_; lean_object* v___x_1271_; 
v___x_1270_ = lean_unsigned_to_nat(1u);
v___x_1271_ = lean_nat_add(v_i_1264_, v___x_1270_);
lean_dec(v_i_1264_);
v_i_1264_ = v___x_1271_;
goto _start;
}
else
{
lean_object* v___x_1273_; 
v___x_1273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1273_, 0, v_i_1264_);
return v___x_1273_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19_spec__24___boxed(lean_object* v_xs_1274_, lean_object* v_v_1275_, lean_object* v_i_1276_){
_start:
{
lean_object* v_res_1277_; 
v_res_1277_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19_spec__24(v_xs_1274_, v_v_1275_, v_i_1276_);
lean_dec(v_v_1275_);
lean_dec_ref(v_xs_1274_);
return v_res_1277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19(lean_object* v_xs_1278_, lean_object* v_v_1279_){
_start:
{
lean_object* v___x_1280_; lean_object* v___x_1281_; 
v___x_1280_ = lean_unsigned_to_nat(0u);
v___x_1281_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19_spec__24(v_xs_1278_, v_v_1279_, v___x_1280_);
return v___x_1281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19___boxed(lean_object* v_xs_1282_, lean_object* v_v_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19(v_xs_1282_, v_v_1283_);
lean_dec(v_v_1283_);
lean_dec_ref(v_xs_1282_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12(lean_object* v_as_1285_, lean_object* v_a_1286_){
_start:
{
lean_object* v___x_1287_; 
v___x_1287_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12_spec__19(v_as_1285_, v_a_1286_);
if (lean_obj_tag(v___x_1287_) == 0)
{
return v_as_1285_;
}
else
{
lean_object* v_val_1288_; lean_object* v___x_1289_; 
v_val_1288_ = lean_ctor_get(v___x_1287_, 0);
lean_inc(v_val_1288_);
lean_dec_ref_known(v___x_1287_, 1);
v___x_1289_ = l_Array_eraseIdx___redArg(v_as_1285_, v_val_1288_);
return v___x_1289_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12___boxed(lean_object* v_as_1290_, lean_object* v_a_1291_){
_start:
{
lean_object* v_res_1292_; 
v_res_1292_ = lp_mathlib_Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12(v_as_1290_, v_a_1291_);
lean_dec(v_a_1291_);
return v_res_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14(lean_object* v_x_1294_, lean_object* v_x_1295_){
_start:
{
if (lean_obj_tag(v_x_1295_) == 0)
{
return v_x_1294_;
}
else
{
lean_object* v_head_1296_; lean_object* v_tail_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; uint8_t v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; 
v_head_1296_ = lean_ctor_get(v_x_1295_, 0);
lean_inc(v_head_1296_);
v_tail_1297_ = lean_ctor_get(v_x_1295_, 1);
lean_inc(v_tail_1297_);
lean_dec_ref_known(v_x_1295_, 2);
v___x_1298_ = ((lean_object*)(lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0));
v___x_1299_ = lean_string_append(v_x_1294_, v___x_1298_);
v___x_1300_ = 1;
v___x_1301_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_head_1296_, v___x_1300_);
v___x_1302_ = lean_string_append(v___x_1299_, v___x_1301_);
lean_dec_ref(v___x_1301_);
v_x_1294_ = v___x_1302_;
v_x_1295_ = v_tail_1297_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10(lean_object* v_x_1306_){
_start:
{
if (lean_obj_tag(v_x_1306_) == 0)
{
lean_object* v___x_1307_; 
v___x_1307_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__0));
return v___x_1307_;
}
else
{
lean_object* v_tail_1308_; 
v_tail_1308_ = lean_ctor_get(v_x_1306_, 1);
if (lean_obj_tag(v_tail_1308_) == 0)
{
lean_object* v_head_1309_; lean_object* v___x_1310_; uint8_t v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; 
v_head_1309_ = lean_ctor_get(v_x_1306_, 0);
lean_inc(v_head_1309_);
lean_dec_ref_known(v_x_1306_, 2);
v___x_1310_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1));
v___x_1311_ = 1;
v___x_1312_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_head_1309_, v___x_1311_);
v___x_1313_ = lean_string_append(v___x_1310_, v___x_1312_);
lean_dec_ref(v___x_1312_);
v___x_1314_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27));
v___x_1315_ = lean_string_append(v___x_1313_, v___x_1314_);
return v___x_1315_;
}
else
{
lean_object* v_head_1316_; lean_object* v___x_1317_; uint8_t v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; uint32_t v___x_1322_; lean_object* v___x_1323_; 
lean_inc(v_tail_1308_);
v_head_1316_ = lean_ctor_get(v_x_1306_, 0);
lean_inc(v_head_1316_);
lean_dec_ref_known(v_x_1306_, 2);
v___x_1317_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1));
v___x_1318_ = 1;
v___x_1319_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_head_1316_, v___x_1318_);
v___x_1320_ = lean_string_append(v___x_1317_, v___x_1319_);
lean_dec_ref(v___x_1319_);
v___x_1321_ = lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14(v___x_1320_, v_tail_1308_);
v___x_1322_ = 93;
v___x_1323_ = lean_string_push(v___x_1321_, v___x_1322_);
return v___x_1323_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__17_spec__26(lean_object* v_x_1324_, lean_object* v_x_1325_){
_start:
{
if (lean_obj_tag(v_x_1325_) == 0)
{
return v_x_1324_;
}
else
{
lean_object* v_head_1326_; lean_object* v_tail_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; uint8_t v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; 
v_head_1326_ = lean_ctor_get(v_x_1325_, 0);
lean_inc(v_head_1326_);
v_tail_1327_ = lean_ctor_get(v_x_1325_, 1);
lean_inc(v_tail_1327_);
lean_dec_ref_known(v_x_1325_, 2);
v___x_1328_ = ((lean_object*)(lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0));
v___x_1329_ = lean_string_append(v_x_1324_, v___x_1328_);
v___x_1330_ = lean_box(0);
v___x_1331_ = 0;
v___x_1332_ = l_Lean_Syntax_formatStx(v_head_1326_, v___x_1330_, v___x_1331_);
v___x_1333_ = l_Std_Format_defWidth;
v___x_1334_ = lean_unsigned_to_nat(0u);
v___x_1335_ = l_Std_Format_pretty(v___x_1332_, v___x_1333_, v___x_1334_, v___x_1334_);
v___x_1336_ = lean_string_append(v___x_1329_, v___x_1335_);
lean_dec_ref(v___x_1335_);
v_x_1324_ = v___x_1336_;
v_x_1325_ = v_tail_1327_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__17(lean_object* v_x_1338_){
_start:
{
if (lean_obj_tag(v_x_1338_) == 0)
{
lean_object* v___x_1339_; 
v___x_1339_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__0));
return v___x_1339_;
}
else
{
lean_object* v_tail_1340_; 
v_tail_1340_ = lean_ctor_get(v_x_1338_, 1);
if (lean_obj_tag(v_tail_1340_) == 0)
{
lean_object* v_head_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; uint8_t v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; 
v_head_1341_ = lean_ctor_get(v_x_1338_, 0);
lean_inc(v_head_1341_);
lean_dec_ref_known(v_x_1338_, 2);
v___x_1342_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1));
v___x_1343_ = lean_box(0);
v___x_1344_ = 0;
v___x_1345_ = l_Lean_Syntax_formatStx(v_head_1341_, v___x_1343_, v___x_1344_);
v___x_1346_ = l_Std_Format_defWidth;
v___x_1347_ = lean_unsigned_to_nat(0u);
v___x_1348_ = l_Std_Format_pretty(v___x_1345_, v___x_1346_, v___x_1347_, v___x_1347_);
v___x_1349_ = lean_string_append(v___x_1342_, v___x_1348_);
lean_dec_ref(v___x_1348_);
v___x_1350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27));
v___x_1351_ = lean_string_append(v___x_1349_, v___x_1350_);
return v___x_1351_;
}
else
{
lean_object* v_head_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; uint8_t v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; uint32_t v___x_1362_; lean_object* v___x_1363_; 
lean_inc(v_tail_1340_);
v_head_1352_ = lean_ctor_get(v_x_1338_, 0);
lean_inc(v_head_1352_);
lean_dec_ref_known(v_x_1338_, 2);
v___x_1353_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1));
v___x_1354_ = lean_box(0);
v___x_1355_ = 0;
v___x_1356_ = l_Lean_Syntax_formatStx(v_head_1352_, v___x_1354_, v___x_1355_);
v___x_1357_ = l_Std_Format_defWidth;
v___x_1358_ = lean_unsigned_to_nat(0u);
v___x_1359_ = l_Std_Format_pretty(v___x_1356_, v___x_1357_, v___x_1358_, v___x_1358_);
v___x_1360_ = lean_string_append(v___x_1353_, v___x_1359_);
lean_dec_ref(v___x_1359_);
v___x_1361_ = lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__17_spec__26(v___x_1360_, v_tail_1340_);
v___x_1362_ = 93;
v___x_1363_ = lean_string_push(v___x_1361_, v___x_1362_);
return v___x_1363_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__3_spec__5(lean_object* v_x_1364_, lean_object* v_x_1365_){
_start:
{
if (lean_obj_tag(v_x_1365_) == 0)
{
return v_x_1364_;
}
else
{
lean_object* v_head_1366_; lean_object* v_tail_1367_; lean_object* v_fst_1368_; lean_object* v_snd_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; uint8_t v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v_head_1366_ = lean_ctor_get(v_x_1365_, 0);
lean_inc(v_head_1366_);
v_tail_1367_ = lean_ctor_get(v_x_1365_, 1);
lean_inc(v_tail_1367_);
lean_dec_ref_known(v_x_1365_, 2);
v_fst_1368_ = lean_ctor_get(v_head_1366_, 0);
lean_inc(v_fst_1368_);
v_snd_1369_ = lean_ctor_get(v_head_1366_, 1);
lean_inc(v_snd_1369_);
lean_dec(v_head_1366_);
v___x_1370_ = ((lean_object*)(lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0));
v___x_1371_ = lean_string_append(v_x_1364_, v___x_1370_);
v___x_1372_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23));
v___x_1373_ = 1;
v___x_1374_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_1368_, v___x_1373_);
v___x_1375_ = lean_string_append(v___x_1372_, v___x_1374_);
lean_dec_ref(v___x_1374_);
v___x_1376_ = lean_string_append(v___x_1375_, v___x_1370_);
v___x_1377_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_snd_1369_, v___x_1373_);
v___x_1378_ = lean_string_append(v___x_1376_, v___x_1377_);
lean_dec_ref(v___x_1377_);
v___x_1379_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26));
v___x_1380_ = lean_string_append(v___x_1378_, v___x_1379_);
v___x_1381_ = lean_string_append(v___x_1371_, v___x_1380_);
lean_dec_ref(v___x_1380_);
v_x_1364_ = v___x_1381_;
v_x_1365_ = v_tail_1367_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__3(lean_object* v_x_1383_){
_start:
{
if (lean_obj_tag(v_x_1383_) == 0)
{
lean_object* v___x_1384_; 
v___x_1384_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__0));
return v___x_1384_;
}
else
{
lean_object* v_tail_1385_; 
v_tail_1385_ = lean_ctor_get(v_x_1383_, 1);
if (lean_obj_tag(v_tail_1385_) == 0)
{
lean_object* v_head_1386_; lean_object* v_fst_1387_; lean_object* v_snd_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; uint8_t v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; 
v_head_1386_ = lean_ctor_get(v_x_1383_, 0);
lean_inc(v_head_1386_);
lean_dec_ref_known(v_x_1383_, 2);
v_fst_1387_ = lean_ctor_get(v_head_1386_, 0);
lean_inc(v_fst_1387_);
v_snd_1388_ = lean_ctor_get(v_head_1386_, 1);
lean_inc(v_snd_1388_);
lean_dec(v_head_1386_);
v___x_1389_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1));
v___x_1390_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23));
v___x_1391_ = 1;
v___x_1392_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_1387_, v___x_1391_);
v___x_1393_ = lean_string_append(v___x_1390_, v___x_1392_);
lean_dec_ref(v___x_1392_);
v___x_1394_ = ((lean_object*)(lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0));
v___x_1395_ = lean_string_append(v___x_1393_, v___x_1394_);
v___x_1396_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_snd_1388_, v___x_1391_);
v___x_1397_ = lean_string_append(v___x_1395_, v___x_1396_);
lean_dec_ref(v___x_1396_);
v___x_1398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26));
v___x_1399_ = lean_string_append(v___x_1397_, v___x_1398_);
v___x_1400_ = lean_string_append(v___x_1389_, v___x_1399_);
lean_dec_ref(v___x_1399_);
v___x_1401_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__27));
v___x_1402_ = lean_string_append(v___x_1400_, v___x_1401_);
return v___x_1402_;
}
else
{
lean_object* v_head_1403_; lean_object* v_fst_1404_; lean_object* v_snd_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; uint8_t v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; uint32_t v___x_1419_; lean_object* v___x_1420_; 
lean_inc(v_tail_1385_);
v_head_1403_ = lean_ctor_get(v_x_1383_, 0);
lean_inc(v_head_1403_);
lean_dec_ref_known(v_x_1383_, 2);
v_fst_1404_ = lean_ctor_get(v_head_1403_, 0);
lean_inc(v_fst_1404_);
v_snd_1405_ = lean_ctor_get(v_head_1403_, 1);
lean_inc(v_snd_1405_);
lean_dec(v_head_1403_);
v___x_1406_ = ((lean_object*)(lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10___closed__1));
v___x_1407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__23));
v___x_1408_ = 1;
v___x_1409_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_1404_, v___x_1408_);
v___x_1410_ = lean_string_append(v___x_1407_, v___x_1409_);
lean_dec_ref(v___x_1409_);
v___x_1411_ = ((lean_object*)(lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10_spec__14___closed__0));
v___x_1412_ = lean_string_append(v___x_1410_, v___x_1411_);
v___x_1413_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_snd_1405_, v___x_1408_);
v___x_1414_ = lean_string_append(v___x_1412_, v___x_1413_);
lean_dec_ref(v___x_1413_);
v___x_1415_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx___closed__26));
v___x_1416_ = lean_string_append(v___x_1414_, v___x_1415_);
v___x_1417_ = lean_string_append(v___x_1406_, v___x_1416_);
lean_dec_ref(v___x_1416_);
v___x_1418_ = lp_mathlib_List_foldl___at___00List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__3_spec__5(v___x_1417_, v_tail_1385_);
v___x_1419_ = 93;
v___x_1420_ = lean_string_push(v___x_1418_, v___x_1419_);
return v___x_1420_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21(lean_object* v_msgData_1421_, uint8_t v_severity_1422_, uint8_t v_isSilent_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_){
_start:
{
lean_object* v___x_1427_; 
v___x_1427_ = l_Lean_Elab_Command_getRef___redArg(v___y_1424_);
if (lean_obj_tag(v___x_1427_) == 0)
{
lean_object* v_a_1428_; lean_object* v___x_1429_; 
v_a_1428_ = lean_ctor_get(v___x_1427_, 0);
lean_inc(v_a_1428_);
lean_dec_ref_known(v___x_1427_, 1);
v___x_1429_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10(v_a_1428_, v_msgData_1421_, v_severity_1422_, v_isSilent_1423_, v___y_1424_, v___y_1425_);
lean_dec(v_a_1428_);
return v___x_1429_;
}
else
{
lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1437_; 
lean_dec_ref(v_msgData_1421_);
v_a_1430_ = lean_ctor_get(v___x_1427_, 0);
v_isSharedCheck_1437_ = !lean_is_exclusive(v___x_1427_);
if (v_isSharedCheck_1437_ == 0)
{
v___x_1432_ = v___x_1427_;
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1427_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1435_; 
if (v_isShared_1433_ == 0)
{
v___x_1435_ = v___x_1432_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1436_; 
v_reuseFailAlloc_1436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1436_, 0, v_a_1430_);
v___x_1435_ = v_reuseFailAlloc_1436_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
return v___x_1435_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21___boxed(lean_object* v_msgData_1438_, lean_object* v_severity_1439_, lean_object* v_isSilent_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_){
_start:
{
uint8_t v_severity_boxed_1444_; uint8_t v_isSilent_boxed_1445_; lean_object* v_res_1446_; 
v_severity_boxed_1444_ = lean_unbox(v_severity_1439_);
v_isSilent_boxed_1445_ = lean_unbox(v_isSilent_1440_);
v_res_1446_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21(v_msgData_1438_, v_severity_boxed_1444_, v_isSilent_boxed_1445_, v___y_1441_, v___y_1442_);
lean_dec(v___y_1442_);
lean_dec_ref(v___y_1441_);
return v_res_1446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13(lean_object* v_msgData_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_){
_start:
{
uint8_t v___x_1451_; uint8_t v___x_1452_; lean_object* v___x_1453_; 
v___x_1451_ = 2;
v___x_1452_ = 0;
v___x_1453_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21(v_msgData_1447_, v___x_1451_, v___x_1452_, v___y_1448_, v___y_1449_);
return v___x_1453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13___boxed(lean_object* v_msgData_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_){
_start:
{
lean_object* v_res_1458_; 
v_res_1458_ = lp_mathlib_Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13(v_msgData_1454_, v___y_1455_, v___y_1456_);
lean_dec(v___y_1456_);
lean_dec_ref(v___y_1455_);
return v_res_1458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18(lean_object* v_a_1460_, lean_object* v_as_1461_, size_t v_i_1462_, size_t v_stop_1463_, lean_object* v_b_1464_){
_start:
{
lean_object* v___y_1466_; uint8_t v___x_1470_; 
v___x_1470_ = lean_usize_dec_eq(v_i_1462_, v_stop_1463_);
if (v___x_1470_ == 0)
{
lean_object* v___x_1471_; lean_object* v___x_1472_; uint8_t v___x_1473_; 
v___x_1471_ = lean_array_uget_borrowed(v_as_1461_, v_i_1462_);
v___x_1472_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18___closed__0));
lean_inc(v___x_1471_);
lean_inc_ref(v_a_1460_);
v___x_1473_ = l_Array_contains___redArg(v___x_1472_, v_a_1460_, v___x_1471_);
if (v___x_1473_ == 0)
{
lean_object* v___x_1474_; 
lean_inc(v___x_1471_);
v___x_1474_ = lean_array_push(v_b_1464_, v___x_1471_);
v___y_1466_ = v___x_1474_;
goto v___jp_1465_;
}
else
{
v___y_1466_ = v_b_1464_;
goto v___jp_1465_;
}
}
else
{
lean_dec_ref(v_a_1460_);
return v_b_1464_;
}
v___jp_1465_:
{
size_t v___x_1467_; size_t v___x_1468_; 
v___x_1467_ = ((size_t)1ULL);
v___x_1468_ = lean_usize_add(v_i_1462_, v___x_1467_);
v_i_1462_ = v___x_1468_;
v_b_1464_ = v___y_1466_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18___boxed(lean_object* v_a_1475_, lean_object* v_as_1476_, lean_object* v_i_1477_, lean_object* v_stop_1478_, lean_object* v_b_1479_){
_start:
{
size_t v_i_boxed_1480_; size_t v_stop_boxed_1481_; lean_object* v_res_1482_; 
v_i_boxed_1480_ = lean_unbox_usize(v_i_1477_);
lean_dec(v_i_1477_);
v_stop_boxed_1481_ = lean_unbox_usize(v_stop_1478_);
lean_dec(v_stop_1478_);
v_res_1482_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18(v_a_1475_, v_as_1476_, v_i_boxed_1480_, v_stop_boxed_1481_, v_b_1479_);
lean_dec_ref(v_as_1476_);
return v_res_1482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__21(lean_object* v_msgData_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_){
_start:
{
uint8_t v___x_1487_; uint8_t v___x_1488_; lean_object* v___x_1489_; 
v___x_1487_ = 1;
v___x_1488_ = 0;
v___x_1489_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13_spec__21(v_msgData_1483_, v___x_1487_, v___x_1488_, v___y_1484_, v___y_1485_);
return v___x_1489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__21___boxed(lean_object* v_msgData_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_){
_start:
{
lean_object* v_res_1494_; 
v_res_1494_ = lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__21(v_msgData_1490_, v___y_1491_, v___y_1492_);
lean_dec(v___y_1492_);
lean_dec_ref(v___y_1491_);
return v_res_1494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg(lean_object* v_opt_1495_, lean_object* v___y_1496_){
_start:
{
lean_object* v___x_1498_; lean_object* v_scopes_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v_opts_1502_; uint8_t v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; 
v___x_1498_ = lean_st_ref_get(v___y_1496_);
v_scopes_1499_ = lean_ctor_get(v___x_1498_, 2);
lean_inc(v_scopes_1499_);
lean_dec(v___x_1498_);
v___x_1500_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1501_ = l_List_head_x21___redArg(v___x_1500_, v_scopes_1499_);
lean_dec(v_scopes_1499_);
v_opts_1502_ = lean_ctor_get(v___x_1501_, 1);
lean_inc_ref(v_opts_1502_);
lean_dec(v___x_1501_);
v___x_1503_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__12(v_opts_1502_, v_opt_1495_);
lean_dec_ref(v_opts_1502_);
v___x_1504_ = lean_box(v___x_1503_);
v___x_1505_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1505_, 0, v___x_1504_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg___boxed(lean_object* v_opt_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_){
_start:
{
lean_object* v_res_1509_; 
v_res_1509_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg(v_opt_1506_, v___y_1507_);
lean_dec(v___y_1507_);
lean_dec_ref(v_opt_1506_);
return v_res_1509_;
}
}
static lean_object* _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__1(void){
_start:
{
lean_object* v___x_1511_; lean_object* v___x_1512_; 
v___x_1511_ = ((lean_object*)(lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__0));
v___x_1512_ = l_Lean_stringToMessageData(v___x_1511_);
return v___x_1512_;
}
}
static lean_object* _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__3(void){
_start:
{
lean_object* v___x_1514_; lean_object* v___x_1515_; 
v___x_1514_ = ((lean_object*)(lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__2));
v___x_1515_ = l_Lean_stringToMessageData(v___x_1514_);
return v___x_1515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17(lean_object* v_id_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_){
_start:
{
lean_object* v___x_1520_; lean_object* v_env_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; lean_object* v_a_1524_; lean_object* v___x_1526_; uint8_t v_isShared_1527_; uint8_t v_isSharedCheck_1543_; 
v___x_1520_ = lean_st_ref_get(v___y_1518_);
v_env_1521_ = lean_ctor_get(v___x_1520_, 0);
lean_inc_ref(v_env_1521_);
lean_dec(v___x_1520_);
v___x_1522_ = l_Lean_ResolveName_backward_privateInPublic_warn;
v___x_1523_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg(v___x_1522_, v___y_1518_);
v_a_1524_ = lean_ctor_get(v___x_1523_, 0);
v_isSharedCheck_1543_ = !lean_is_exclusive(v___x_1523_);
if (v_isSharedCheck_1543_ == 0)
{
v___x_1526_ = v___x_1523_;
v_isShared_1527_ = v_isSharedCheck_1543_;
goto v_resetjp_1525_;
}
else
{
lean_inc(v_a_1524_);
lean_dec(v___x_1523_);
v___x_1526_ = lean_box(0);
v_isShared_1527_ = v_isSharedCheck_1543_;
goto v_resetjp_1525_;
}
v_resetjp_1525_:
{
uint8_t v_isExporting_1533_; 
v_isExporting_1533_ = lean_ctor_get_uint8(v_env_1521_, sizeof(void*)*8);
lean_dec_ref(v_env_1521_);
if (v_isExporting_1533_ == 0)
{
lean_dec(v_a_1524_);
lean_dec(v_id_1516_);
goto v___jp_1528_;
}
else
{
uint8_t v___x_1534_; 
v___x_1534_ = l_Lean_isPrivateName(v_id_1516_);
if (v___x_1534_ == 0)
{
lean_dec(v_a_1524_);
lean_dec(v_id_1516_);
goto v___jp_1528_;
}
else
{
uint8_t v___x_1535_; 
v___x_1535_ = lean_unbox(v_a_1524_);
lean_dec(v_a_1524_);
if (v___x_1535_ == 0)
{
lean_dec(v_id_1516_);
goto v___jp_1528_;
}
else
{
lean_object* v___x_1536_; uint8_t v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; 
lean_del_object(v___x_1526_);
v___x_1536_ = lean_obj_once(&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__1, &lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__1_once, _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__1);
v___x_1537_ = 0;
v___x_1538_ = l_Lean_MessageData_ofConstName(v_id_1516_, v___x_1537_);
v___x_1539_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1536_);
lean_ctor_set(v___x_1539_, 1, v___x_1538_);
v___x_1540_ = lean_obj_once(&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__3, &lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__3_once, _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___closed__3);
v___x_1541_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1541_, 0, v___x_1539_);
lean_ctor_set(v___x_1541_, 1, v___x_1540_);
v___x_1542_ = lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__21(v___x_1541_, v___y_1517_, v___y_1518_);
return v___x_1542_;
}
}
}
v___jp_1528_:
{
lean_object* v___x_1529_; lean_object* v___x_1531_; 
v___x_1529_ = lean_box(0);
if (v_isShared_1527_ == 0)
{
lean_ctor_set(v___x_1526_, 0, v___x_1529_);
v___x_1531_ = v___x_1526_;
goto v_reusejp_1530_;
}
else
{
lean_object* v_reuseFailAlloc_1532_; 
v_reuseFailAlloc_1532_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1532_, 0, v___x_1529_);
v___x_1531_ = v_reuseFailAlloc_1532_;
goto v_reusejp_1530_;
}
v_reusejp_1530_:
{
return v___x_1531_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17___boxed(lean_object* v_id_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v_res_1548_; 
v_res_1548_ = lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17(v_id_1544_, v___y_1545_, v___y_1546_);
lean_dec(v___y_1546_);
lean_dec_ref(v___y_1545_);
return v_res_1548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__16(lean_object* v_x_1549_){
_start:
{
if (lean_obj_tag(v_x_1549_) == 0)
{
lean_object* v___x_1550_; 
v___x_1550_ = lean_box(0);
return v___x_1550_;
}
else
{
lean_object* v_head_1551_; lean_object* v_tail_1552_; lean_object* v_fst_1553_; uint8_t v___x_1554_; 
v_head_1551_ = lean_ctor_get(v_x_1549_, 0);
v_tail_1552_ = lean_ctor_get(v_x_1549_, 1);
v_fst_1553_ = lean_ctor_get(v_head_1551_, 0);
v___x_1554_ = l_Lean_isPrivateName(v_fst_1553_);
if (v___x_1554_ == 0)
{
v_x_1549_ = v_tail_1552_;
goto _start;
}
else
{
lean_object* v___x_1556_; 
lean_inc(v_head_1551_);
v___x_1556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1556_, 0, v_head_1551_);
return v___x_1556_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__16___boxed(lean_object* v_x_1557_){
_start:
{
lean_object* v_res_1558_; 
v_res_1558_ = lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__16(v_x_1557_);
lean_dec(v_x_1557_);
return v_res_1558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11(lean_object* v_id_1559_, uint8_t v_enableLog_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_){
_start:
{
lean_object* v___x_1564_; lean_object* v_env_1565_; lean_object* v___x_1566_; lean_object* v_scopes_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v_opts_1570_; lean_object* v___x_1571_; 
v___x_1564_ = lean_st_ref_get(v___y_1562_);
v_env_1565_ = lean_ctor_get(v___x_1564_, 0);
lean_inc_ref(v_env_1565_);
lean_dec(v___x_1564_);
v___x_1566_ = lean_st_ref_get(v___y_1562_);
v_scopes_1567_ = lean_ctor_get(v___x_1566_, 2);
lean_inc(v_scopes_1567_);
lean_dec(v___x_1566_);
v___x_1568_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1569_ = l_List_head_x21___redArg(v___x_1568_, v_scopes_1567_);
lean_dec(v_scopes_1567_);
v_opts_1570_ = lean_ctor_get(v___x_1569_, 1);
lean_inc_ref(v_opts_1570_);
lean_dec(v___x_1569_);
v___x_1571_ = l_Lean_Elab_Command_getScope___redArg(v___y_1562_);
if (lean_obj_tag(v___x_1571_) == 0)
{
lean_object* v_a_1572_; lean_object* v_currNamespace_1573_; lean_object* v___x_1574_; 
v_a_1572_ = lean_ctor_get(v___x_1571_, 0);
lean_inc(v_a_1572_);
lean_dec_ref_known(v___x_1571_, 1);
v_currNamespace_1573_ = lean_ctor_get(v_a_1572_, 2);
lean_inc(v_currNamespace_1573_);
lean_dec(v_a_1572_);
v___x_1574_ = l_Lean_Elab_Command_getScope___redArg(v___y_1562_);
if (lean_obj_tag(v___x_1574_) == 0)
{
lean_object* v_a_1575_; lean_object* v___x_1577_; uint8_t v_isShared_1578_; uint8_t v_isSharedCheck_1613_; 
v_a_1575_ = lean_ctor_get(v___x_1574_, 0);
v_isSharedCheck_1613_ = !lean_is_exclusive(v___x_1574_);
if (v_isSharedCheck_1613_ == 0)
{
v___x_1577_ = v___x_1574_;
v_isShared_1578_ = v_isSharedCheck_1613_;
goto v_resetjp_1576_;
}
else
{
lean_inc(v_a_1575_);
lean_dec(v___x_1574_);
v___x_1577_ = lean_box(0);
v_isShared_1578_ = v_isSharedCheck_1613_;
goto v_resetjp_1576_;
}
v_resetjp_1576_:
{
lean_object* v_openDecls_1579_; lean_object* v___x_1580_; lean_object* v_env_1581_; lean_object* v_res_1582_; 
v_openDecls_1579_ = lean_ctor_get(v_a_1575_, 3);
lean_inc(v_openDecls_1579_);
lean_dec(v_a_1575_);
v___x_1580_ = lean_st_ref_get(v___y_1562_);
v_env_1581_ = lean_ctor_get(v___x_1580_, 0);
lean_inc_ref(v_env_1581_);
lean_dec(v___x_1580_);
v_res_1582_ = l_Lean_ResolveName_resolveGlobalName(v_env_1565_, v_opts_1570_, v_currNamespace_1573_, v_openDecls_1579_, v_id_1559_);
lean_dec_ref(v_opts_1570_);
if (v_enableLog_1560_ == 0)
{
lean_object* v___x_1584_; 
lean_dec_ref(v_env_1581_);
if (v_isShared_1578_ == 0)
{
lean_ctor_set(v___x_1577_, 0, v_res_1582_);
v___x_1584_ = v___x_1577_;
goto v_reusejp_1583_;
}
else
{
lean_object* v_reuseFailAlloc_1585_; 
v_reuseFailAlloc_1585_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1585_, 0, v_res_1582_);
v___x_1584_ = v_reuseFailAlloc_1585_;
goto v_reusejp_1583_;
}
v_reusejp_1583_:
{
return v___x_1584_;
}
}
else
{
uint8_t v_isExporting_1586_; 
v_isExporting_1586_ = lean_ctor_get_uint8(v_env_1581_, sizeof(void*)*8);
lean_dec_ref(v_env_1581_);
if (v_isExporting_1586_ == 0)
{
lean_object* v___x_1588_; 
if (v_isShared_1578_ == 0)
{
lean_ctor_set(v___x_1577_, 0, v_res_1582_);
v___x_1588_ = v___x_1577_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v_res_1582_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
else
{
lean_object* v___x_1590_; 
v___x_1590_ = lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__16(v_res_1582_);
if (lean_obj_tag(v___x_1590_) == 1)
{
lean_object* v_val_1591_; lean_object* v_fst_1592_; lean_object* v___x_1593_; 
lean_del_object(v___x_1577_);
v_val_1591_ = lean_ctor_get(v___x_1590_, 0);
lean_inc(v_val_1591_);
lean_dec_ref_known(v___x_1590_, 1);
v_fst_1592_ = lean_ctor_get(v_val_1591_, 0);
lean_inc(v_fst_1592_);
lean_dec(v_val_1591_);
v___x_1593_ = lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17(v_fst_1592_, v___y_1561_, v___y_1562_);
if (lean_obj_tag(v___x_1593_) == 0)
{
lean_object* v___x_1595_; uint8_t v_isShared_1596_; uint8_t v_isSharedCheck_1600_; 
v_isSharedCheck_1600_ = !lean_is_exclusive(v___x_1593_);
if (v_isSharedCheck_1600_ == 0)
{
lean_object* v_unused_1601_; 
v_unused_1601_ = lean_ctor_get(v___x_1593_, 0);
lean_dec(v_unused_1601_);
v___x_1595_ = v___x_1593_;
v_isShared_1596_ = v_isSharedCheck_1600_;
goto v_resetjp_1594_;
}
else
{
lean_dec(v___x_1593_);
v___x_1595_ = lean_box(0);
v_isShared_1596_ = v_isSharedCheck_1600_;
goto v_resetjp_1594_;
}
v_resetjp_1594_:
{
lean_object* v___x_1598_; 
if (v_isShared_1596_ == 0)
{
lean_ctor_set(v___x_1595_, 0, v_res_1582_);
v___x_1598_ = v___x_1595_;
goto v_reusejp_1597_;
}
else
{
lean_object* v_reuseFailAlloc_1599_; 
v_reuseFailAlloc_1599_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1599_, 0, v_res_1582_);
v___x_1598_ = v_reuseFailAlloc_1599_;
goto v_reusejp_1597_;
}
v_reusejp_1597_:
{
return v___x_1598_;
}
}
}
else
{
lean_object* v_a_1602_; lean_object* v___x_1604_; uint8_t v_isShared_1605_; uint8_t v_isSharedCheck_1609_; 
lean_dec(v_res_1582_);
v_a_1602_ = lean_ctor_get(v___x_1593_, 0);
v_isSharedCheck_1609_ = !lean_is_exclusive(v___x_1593_);
if (v_isSharedCheck_1609_ == 0)
{
v___x_1604_ = v___x_1593_;
v_isShared_1605_ = v_isSharedCheck_1609_;
goto v_resetjp_1603_;
}
else
{
lean_inc(v_a_1602_);
lean_dec(v___x_1593_);
v___x_1604_ = lean_box(0);
v_isShared_1605_ = v_isSharedCheck_1609_;
goto v_resetjp_1603_;
}
v_resetjp_1603_:
{
lean_object* v___x_1607_; 
if (v_isShared_1605_ == 0)
{
v___x_1607_ = v___x_1604_;
goto v_reusejp_1606_;
}
else
{
lean_object* v_reuseFailAlloc_1608_; 
v_reuseFailAlloc_1608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1608_, 0, v_a_1602_);
v___x_1607_ = v_reuseFailAlloc_1608_;
goto v_reusejp_1606_;
}
v_reusejp_1606_:
{
return v___x_1607_;
}
}
}
}
else
{
lean_object* v___x_1611_; 
lean_dec(v___x_1590_);
if (v_isShared_1578_ == 0)
{
lean_ctor_set(v___x_1577_, 0, v_res_1582_);
v___x_1611_ = v___x_1577_;
goto v_reusejp_1610_;
}
else
{
lean_object* v_reuseFailAlloc_1612_; 
v_reuseFailAlloc_1612_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1612_, 0, v_res_1582_);
v___x_1611_ = v_reuseFailAlloc_1612_;
goto v_reusejp_1610_;
}
v_reusejp_1610_:
{
return v___x_1611_;
}
}
}
}
}
}
else
{
lean_object* v_a_1614_; lean_object* v___x_1616_; uint8_t v_isShared_1617_; uint8_t v_isSharedCheck_1621_; 
lean_dec(v_currNamespace_1573_);
lean_dec_ref(v_opts_1570_);
lean_dec_ref(v_env_1565_);
lean_dec(v_id_1559_);
v_a_1614_ = lean_ctor_get(v___x_1574_, 0);
v_isSharedCheck_1621_ = !lean_is_exclusive(v___x_1574_);
if (v_isSharedCheck_1621_ == 0)
{
v___x_1616_ = v___x_1574_;
v_isShared_1617_ = v_isSharedCheck_1621_;
goto v_resetjp_1615_;
}
else
{
lean_inc(v_a_1614_);
lean_dec(v___x_1574_);
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
else
{
lean_object* v_a_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1629_; 
lean_dec_ref(v_opts_1570_);
lean_dec_ref(v_env_1565_);
lean_dec(v_id_1559_);
v_a_1622_ = lean_ctor_get(v___x_1571_, 0);
v_isSharedCheck_1629_ = !lean_is_exclusive(v___x_1571_);
if (v_isSharedCheck_1629_ == 0)
{
v___x_1624_ = v___x_1571_;
v_isShared_1625_ = v_isSharedCheck_1629_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_a_1622_);
lean_dec(v___x_1571_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1629_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v___x_1627_; 
if (v_isShared_1625_ == 0)
{
v___x_1627_ = v___x_1624_;
goto v_reusejp_1626_;
}
else
{
lean_object* v_reuseFailAlloc_1628_; 
v_reuseFailAlloc_1628_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1628_, 0, v_a_1622_);
v___x_1627_ = v_reuseFailAlloc_1628_;
goto v_reusejp_1626_;
}
v_reusejp_1626_:
{
return v___x_1627_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11___boxed(lean_object* v_id_1630_, lean_object* v_enableLog_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_){
_start:
{
uint8_t v_enableLog_boxed_1635_; lean_object* v_res_1636_; 
v_enableLog_boxed_1635_ = lean_unbox(v_enableLog_1631_);
v_res_1636_ = lp_mathlib_Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11(v_id_1630_, v_enableLog_boxed_1635_, v___y_1632_, v___y_1633_);
lean_dec(v___y_1633_);
lean_dec_ref(v___y_1632_);
return v_res_1636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8(lean_object* v_as_1638_, size_t v_i_1639_, size_t v_stop_1640_, lean_object* v_b_1641_){
_start:
{
uint8_t v___x_1642_; 
v___x_1642_ = lean_usize_dec_eq(v_i_1639_, v_stop_1640_);
if (v___x_1642_ == 0)
{
lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; size_t v___x_1647_; size_t v___x_1648_; 
v___x_1643_ = lean_array_uget_borrowed(v_as_1638_, v_i_1639_);
v___x_1644_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8___closed__0));
v___x_1645_ = lean_string_append(v_b_1641_, v___x_1644_);
v___x_1646_ = lean_string_append(v___x_1645_, v___x_1643_);
v___x_1647_ = ((size_t)1ULL);
v___x_1648_ = lean_usize_add(v_i_1639_, v___x_1647_);
v_i_1639_ = v___x_1648_;
v_b_1641_ = v___x_1646_;
goto _start;
}
else
{
return v_b_1641_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8___boxed(lean_object* v_as_1650_, lean_object* v_i_1651_, lean_object* v_stop_1652_, lean_object* v_b_1653_){
_start:
{
size_t v_i_boxed_1654_; size_t v_stop_boxed_1655_; lean_object* v_res_1656_; 
v_i_boxed_1654_ = lean_unbox_usize(v_i_1651_);
lean_dec(v_i_1651_);
v_stop_boxed_1655_ = lean_unbox_usize(v_stop_1652_);
lean_dec(v_stop_1652_);
v_res_1656_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8(v_as_1650_, v_i_boxed_1654_, v_stop_boxed_1655_, v_b_1653_);
lean_dec_ref(v_as_1650_);
return v_res_1656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__2(size_t v_sz_1657_, size_t v_i_1658_, lean_object* v_bs_1659_){
_start:
{
uint8_t v___x_1660_; 
v___x_1660_ = lean_usize_dec_lt(v_i_1658_, v_sz_1657_);
if (v___x_1660_ == 0)
{
return v_bs_1659_;
}
else
{
lean_object* v_v_1661_; lean_object* v_fst_1662_; lean_object* v_snd_1663_; lean_object* v___x_1665_; uint8_t v_isShared_1666_; uint8_t v_isSharedCheck_1677_; 
v_v_1661_ = lean_array_uget(v_bs_1659_, v_i_1658_);
v_fst_1662_ = lean_ctor_get(v_v_1661_, 0);
v_snd_1663_ = lean_ctor_get(v_v_1661_, 1);
v_isSharedCheck_1677_ = !lean_is_exclusive(v_v_1661_);
if (v_isSharedCheck_1677_ == 0)
{
v___x_1665_ = v_v_1661_;
v_isShared_1666_ = v_isSharedCheck_1677_;
goto v_resetjp_1664_;
}
else
{
lean_inc(v_snd_1663_);
lean_inc(v_fst_1662_);
lean_dec(v_v_1661_);
v___x_1665_ = lean_box(0);
v_isShared_1666_ = v_isSharedCheck_1677_;
goto v_resetjp_1664_;
}
v_resetjp_1664_:
{
lean_object* v___x_1667_; lean_object* v_bs_x27_1668_; lean_object* v___x_1669_; lean_object* v___x_1671_; 
v___x_1667_ = lean_unsigned_to_nat(0u);
v_bs_x27_1668_ = lean_array_uset(v_bs_1659_, v_i_1658_, v___x_1667_);
v___x_1669_ = l_Lean_TSyntax_getId(v_fst_1662_);
lean_dec(v_fst_1662_);
if (v_isShared_1666_ == 0)
{
lean_ctor_set(v___x_1665_, 0, v___x_1669_);
v___x_1671_ = v___x_1665_;
goto v_reusejp_1670_;
}
else
{
lean_object* v_reuseFailAlloc_1676_; 
v_reuseFailAlloc_1676_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1676_, 0, v___x_1669_);
lean_ctor_set(v_reuseFailAlloc_1676_, 1, v_snd_1663_);
v___x_1671_ = v_reuseFailAlloc_1676_;
goto v_reusejp_1670_;
}
v_reusejp_1670_:
{
size_t v___x_1672_; size_t v___x_1673_; lean_object* v___x_1674_; 
v___x_1672_ = ((size_t)1ULL);
v___x_1673_ = lean_usize_add(v_i_1658_, v___x_1672_);
v___x_1674_ = lean_array_uset(v_bs_x27_1668_, v_i_1658_, v___x_1671_);
v_i_1658_ = v___x_1673_;
v_bs_1659_ = v___x_1674_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__2___boxed(lean_object* v_sz_1678_, lean_object* v_i_1679_, lean_object* v_bs_1680_){
_start:
{
size_t v_sz_boxed_1681_; size_t v_i_boxed_1682_; lean_object* v_res_1683_; 
v_sz_boxed_1681_ = lean_unbox_usize(v_sz_1678_);
lean_dec(v_sz_1678_);
v_i_boxed_1682_ = lean_unbox_usize(v_i_1679_);
lean_dec(v_i_1679_);
v_res_1683_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__2(v_sz_boxed_1681_, v_i_boxed_1682_, v_bs_1680_);
return v_res_1683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__15(lean_object* v_ref_1684_, lean_object* v_msgData_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_){
_start:
{
uint8_t v___x_1689_; uint8_t v___x_1690_; lean_object* v___x_1691_; 
v___x_1689_ = 2;
v___x_1690_ = 0;
v___x_1691_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10(v_ref_1684_, v_msgData_1685_, v___x_1689_, v___x_1690_, v___y_1686_, v___y_1687_);
return v___x_1691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__15___boxed(lean_object* v_ref_1692_, lean_object* v_msgData_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_){
_start:
{
lean_object* v_res_1697_; 
v_res_1697_ = lp_mathlib_Lean_logErrorAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__15(v_ref_1692_, v_msgData_1693_, v___y_1694_, v___y_1695_);
lean_dec(v___y_1695_);
lean_dec_ref(v___y_1694_);
lean_dec(v_ref_1692_);
return v_res_1697_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__1(void){
_start:
{
lean_object* v___x_1700_; lean_object* v___x_1701_; 
v___x_1700_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__0));
v___x_1701_ = l_Lean_MessageData_ofFormat(v___x_1700_);
return v___x_1701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg(lean_object* v_as_x27_1702_, lean_object* v_b_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_){
_start:
{
if (lean_obj_tag(v_as_x27_1702_) == 0)
{
lean_object* v___x_1707_; 
v___x_1707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1707_, 0, v_b_1703_);
return v___x_1707_;
}
else
{
lean_object* v_head_1708_; lean_object* v_tail_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; 
v_head_1708_ = lean_ctor_get(v_as_x27_1702_, 0);
v_tail_1709_ = lean_ctor_get(v_as_x27_1702_, 1);
v___x_1710_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__1, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__1_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___closed__1);
v___x_1711_ = lp_mathlib_Lean_logErrorAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__15(v_head_1708_, v___x_1710_, v___y_1704_, v___y_1705_);
if (lean_obj_tag(v___x_1711_) == 0)
{
lean_object* v___x_1712_; 
lean_dec_ref_known(v___x_1711_, 1);
v___x_1712_ = lean_box(0);
v_as_x27_1702_ = v_tail_1709_;
v_b_1703_ = v___x_1712_;
goto _start;
}
else
{
return v___x_1711_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg___boxed(lean_object* v_as_x27_1714_, lean_object* v_b_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_){
_start:
{
lean_object* v_res_1719_; 
v_res_1719_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg(v_as_x27_1714_, v_b_1715_, v___y_1716_, v___y_1717_);
lean_dec(v___y_1717_);
lean_dec_ref(v___y_1716_);
lean_dec(v_as_x27_1714_);
return v_res_1719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg(lean_object* v_declName_1720_, lean_object* v___y_1721_){
_start:
{
lean_object* v___x_1723_; lean_object* v_env_1724_; uint8_t v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1723_ = lean_st_ref_get(v___y_1721_);
v_env_1724_ = lean_ctor_get(v___x_1723_, 0);
lean_inc_ref(v_env_1724_);
lean_dec(v___x_1723_);
v___x_1725_ = l_Lean_isRecCore(v_env_1724_, v_declName_1720_);
v___x_1726_ = lean_box(v___x_1725_);
v___x_1727_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1727_, 0, v___x_1726_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg___boxed(lean_object* v_declName_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_){
_start:
{
lean_object* v_res_1731_; 
v_res_1731_ = lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg(v_declName_1728_, v___y_1729_);
lean_dec(v___y_1729_);
return v_res_1731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg(lean_object* v_declName_1732_, lean_object* v___y_1733_){
_start:
{
lean_object* v___x_1735_; lean_object* v_env_1736_; uint8_t v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; 
v___x_1735_ = lean_st_ref_get(v___y_1733_);
v_env_1736_ = lean_ctor_get(v___x_1735_, 0);
lean_inc_ref(v_env_1736_);
lean_dec(v___x_1735_);
v___x_1737_ = l_Lean_Meta_isMatcherCore(v_env_1736_, v_declName_1732_);
v___x_1738_ = lean_box(v___x_1737_);
v___x_1739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1739_, 0, v___x_1738_);
return v___x_1739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg___boxed(lean_object* v_declName_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_){
_start:
{
lean_object* v_res_1743_; 
v_res_1743_ = lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg(v_declName_1740_, v___y_1741_);
lean_dec(v___y_1741_);
return v_res_1743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1(lean_object* v_declName_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_){
_start:
{
lean_object* v___x_1753_; uint8_t v___x_1754_; uint8_t v___x_1755_; 
v___x_1753_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__1));
v___x_1754_ = lean_name_eq(v_declName_1749_, v___x_1753_);
v___x_1755_ = 1;
if (v___x_1754_ == 0)
{
if (lean_obj_tag(v_declName_1749_) == 1)
{
lean_object* v_str_1778_; lean_object* v___x_1779_; uint8_t v___x_1780_; 
v_str_1778_ = lean_ctor_get(v_declName_1749_, 1);
v___x_1779_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__3));
v___x_1780_ = lean_string_dec_eq(v_str_1778_, v___x_1779_);
if (v___x_1780_ == 0)
{
goto v___jp_1772_;
}
else
{
lean_object* v___x_1781_; lean_object* v___x_1782_; 
lean_dec_ref_known(v_declName_1749_, 2);
v___x_1781_ = lean_box(v___x_1755_);
v___x_1782_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1782_, 0, v___x_1781_);
return v___x_1782_;
}
}
else
{
goto v___jp_1772_;
}
}
else
{
lean_object* v___x_1783_; lean_object* v___x_1784_; 
lean_dec(v_declName_1749_);
v___x_1783_ = lean_box(v___x_1755_);
v___x_1784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1784_, 0, v___x_1783_);
return v___x_1784_;
}
v___jp_1756_:
{
lean_object* v___x_1757_; lean_object* v_env_1758_; uint8_t v___x_1759_; 
v___x_1757_ = lean_st_ref_get(v___y_1751_);
v_env_1758_ = lean_ctor_get(v___x_1757_, 0);
lean_inc_ref(v_env_1758_);
lean_dec(v___x_1757_);
lean_inc(v_declName_1749_);
v___x_1759_ = l_Lean_Name_isInternalDetail(v_declName_1749_);
if (v___x_1759_ == 0)
{
uint8_t v___x_1760_; 
lean_inc(v_declName_1749_);
lean_inc_ref(v_env_1758_);
v___x_1760_ = l_Lean_isAuxRecursor(v_env_1758_, v_declName_1749_);
if (v___x_1760_ == 0)
{
uint8_t v___x_1761_; 
lean_inc(v_declName_1749_);
v___x_1761_ = l_Lean_isNoConfusion(v_env_1758_, v_declName_1749_);
if (v___x_1761_ == 0)
{
lean_object* v___x_1762_; lean_object* v_a_1763_; uint8_t v___x_1764_; 
lean_inc(v_declName_1749_);
v___x_1762_ = lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg(v_declName_1749_, v___y_1751_);
v_a_1763_ = lean_ctor_get(v___x_1762_, 0);
lean_inc(v_a_1763_);
v___x_1764_ = lean_unbox(v_a_1763_);
lean_dec(v_a_1763_);
if (v___x_1764_ == 0)
{
lean_object* v___x_1765_; 
lean_dec_ref(v___x_1762_);
v___x_1765_ = lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg(v_declName_1749_, v___y_1751_);
return v___x_1765_;
}
else
{
lean_dec(v_declName_1749_);
return v___x_1762_;
}
}
else
{
lean_object* v___x_1766_; lean_object* v___x_1767_; 
lean_dec(v_declName_1749_);
v___x_1766_ = lean_box(v___x_1761_);
v___x_1767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1767_, 0, v___x_1766_);
return v___x_1767_;
}
}
else
{
lean_object* v___x_1768_; lean_object* v___x_1769_; 
lean_dec_ref(v_env_1758_);
lean_dec(v_declName_1749_);
v___x_1768_ = lean_box(v___x_1755_);
v___x_1769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1769_, 0, v___x_1768_);
return v___x_1769_;
}
}
else
{
lean_object* v___x_1770_; lean_object* v___x_1771_; 
lean_dec_ref(v_env_1758_);
lean_dec(v_declName_1749_);
v___x_1770_ = lean_box(v___x_1755_);
v___x_1771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1771_, 0, v___x_1770_);
return v___x_1771_;
}
}
v___jp_1772_:
{
if (lean_obj_tag(v_declName_1749_) == 1)
{
lean_object* v_str_1773_; lean_object* v___x_1774_; uint8_t v___x_1775_; 
v_str_1773_ = lean_ctor_get(v_declName_1749_, 1);
v___x_1774_ = ((lean_object*)(lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___closed__2));
v___x_1775_ = lean_string_dec_eq(v_str_1773_, v___x_1774_);
if (v___x_1775_ == 0)
{
goto v___jp_1756_;
}
else
{
lean_object* v___x_1776_; lean_object* v___x_1777_; 
lean_dec_ref_known(v_declName_1749_, 2);
v___x_1776_ = lean_box(v___x_1755_);
v___x_1777_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1777_, 0, v___x_1776_);
return v___x_1777_;
}
}
else
{
goto v___jp_1756_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1___boxed(lean_object* v_declName_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_){
_start:
{
lean_object* v_res_1789_; 
v_res_1789_ = lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1(v_declName_1785_, v___y_1786_, v___y_1787_);
lean_dec(v___y_1787_);
lean_dec_ref(v___y_1786_);
return v_res_1789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19(lean_object* v_as_1790_, size_t v_i_1791_, size_t v_stop_1792_, lean_object* v_b_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_){
_start:
{
uint8_t v___x_1797_; 
v___x_1797_ = lean_usize_dec_eq(v_i_1791_, v_stop_1792_);
if (v___x_1797_ == 0)
{
lean_object* v___x_1798_; lean_object* v___x_1799_; 
v___x_1798_ = lean_array_uget_borrowed(v_as_1790_, v_i_1791_);
lean_inc(v___x_1798_);
v___x_1799_ = lp_mathlib_Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1(v___x_1798_, v___y_1794_, v___y_1795_);
if (lean_obj_tag(v___x_1799_) == 0)
{
lean_object* v_a_1800_; lean_object* v_a_1802_; uint8_t v___x_1806_; 
v_a_1800_ = lean_ctor_get(v___x_1799_, 0);
lean_inc(v_a_1800_);
lean_dec_ref_known(v___x_1799_, 1);
v___x_1806_ = lean_unbox(v_a_1800_);
lean_dec(v_a_1800_);
if (v___x_1806_ == 0)
{
v_a_1802_ = v_b_1793_;
goto v___jp_1801_;
}
else
{
lean_object* v___x_1807_; 
lean_inc(v___x_1798_);
v___x_1807_ = lean_array_push(v_b_1793_, v___x_1798_);
v_a_1802_ = v___x_1807_;
goto v___jp_1801_;
}
v___jp_1801_:
{
size_t v___x_1803_; size_t v___x_1804_; 
v___x_1803_ = ((size_t)1ULL);
v___x_1804_ = lean_usize_add(v_i_1791_, v___x_1803_);
v_i_1791_ = v___x_1804_;
v_b_1793_ = v_a_1802_;
goto _start;
}
}
else
{
lean_object* v_a_1808_; lean_object* v___x_1810_; uint8_t v_isShared_1811_; uint8_t v_isSharedCheck_1815_; 
lean_dec_ref(v_b_1793_);
v_a_1808_ = lean_ctor_get(v___x_1799_, 0);
v_isSharedCheck_1815_ = !lean_is_exclusive(v___x_1799_);
if (v_isSharedCheck_1815_ == 0)
{
v___x_1810_ = v___x_1799_;
v_isShared_1811_ = v_isSharedCheck_1815_;
goto v_resetjp_1809_;
}
else
{
lean_inc(v_a_1808_);
lean_dec(v___x_1799_);
v___x_1810_ = lean_box(0);
v_isShared_1811_ = v_isSharedCheck_1815_;
goto v_resetjp_1809_;
}
v_resetjp_1809_:
{
lean_object* v___x_1813_; 
if (v_isShared_1811_ == 0)
{
v___x_1813_ = v___x_1810_;
goto v_reusejp_1812_;
}
else
{
lean_object* v_reuseFailAlloc_1814_; 
v_reuseFailAlloc_1814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1814_, 0, v_a_1808_);
v___x_1813_ = v_reuseFailAlloc_1814_;
goto v_reusejp_1812_;
}
v_reusejp_1812_:
{
return v___x_1813_;
}
}
}
}
else
{
lean_object* v___x_1816_; 
v___x_1816_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1816_, 0, v_b_1793_);
return v___x_1816_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19___boxed(lean_object* v_as_1817_, lean_object* v_i_1818_, lean_object* v_stop_1819_, lean_object* v_b_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_){
_start:
{
size_t v_i_boxed_1824_; size_t v_stop_boxed_1825_; lean_object* v_res_1826_; 
v_i_boxed_1824_ = lean_unbox_usize(v_i_1818_);
lean_dec(v_i_1818_);
v_stop_boxed_1825_ = lean_unbox_usize(v_stop_1819_);
lean_dec(v_stop_1819_);
v_res_1826_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19(v_as_1817_, v_i_boxed_1824_, v_stop_boxed_1825_, v_b_1820_, v___y_1821_, v___y_1822_);
lean_dec(v___y_1822_);
lean_dec_ref(v___y_1821_);
lean_dec_ref(v_as_1817_);
return v_res_1826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14(uint8_t v___x_1830_, lean_object* v_a_1831_, lean_object* v_as_1832_, size_t v_i_1833_, size_t v_stop_1834_, lean_object* v_b_1835_){
_start:
{
lean_object* v___y_1837_; uint8_t v___x_1841_; 
v___x_1841_ = lean_usize_dec_eq(v_i_1833_, v_stop_1834_);
if (v___x_1841_ == 0)
{
lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v_fst_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; uint8_t v___x_1851_; 
v___x_1842_ = lean_unsigned_to_nat(0u);
v___x_1843_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___closed__0));
v___x_1844_ = l_List_get_x21Internal___redArg(v___x_1843_, v_a_1831_, v___x_1842_);
v_fst_1845_ = lean_ctor_get(v___x_1844_, 0);
lean_inc(v_fst_1845_);
lean_dec(v___x_1844_);
v___x_1846_ = lean_array_uget_borrowed(v_as_1832_, v_i_1833_);
lean_inc(v___x_1846_);
v___x_1847_ = l_Lean_Name_toString(v___x_1846_, v___x_1830_);
v___x_1848_ = l_Lean_Name_toString(v_fst_1845_, v___x_1830_);
v___x_1849_ = lean_string_utf8_byte_size(v___x_1847_);
v___x_1850_ = lean_string_utf8_byte_size(v___x_1848_);
v___x_1851_ = lean_nat_dec_le(v___x_1850_, v___x_1849_);
if (v___x_1851_ == 0)
{
lean_dec_ref(v___x_1848_);
lean_dec_ref(v___x_1847_);
v___y_1837_ = v_b_1835_;
goto v___jp_1836_;
}
else
{
lean_object* v___x_1852_; uint8_t v___x_1853_; 
v___x_1852_ = lean_nat_sub(v___x_1849_, v___x_1850_);
v___x_1853_ = lean_string_memcmp(v___x_1847_, v___x_1848_, v___x_1852_, v___x_1842_, v___x_1850_);
lean_dec(v___x_1852_);
lean_dec_ref(v___x_1848_);
lean_dec_ref(v___x_1847_);
if (v___x_1853_ == 0)
{
v___y_1837_ = v_b_1835_;
goto v___jp_1836_;
}
else
{
lean_object* v___x_1854_; 
lean_inc(v___x_1846_);
v___x_1854_ = lean_array_push(v_b_1835_, v___x_1846_);
v___y_1837_ = v___x_1854_;
goto v___jp_1836_;
}
}
}
else
{
return v_b_1835_;
}
v___jp_1836_:
{
size_t v___x_1838_; size_t v___x_1839_; 
v___x_1838_ = ((size_t)1ULL);
v___x_1839_ = lean_usize_add(v_i_1833_, v___x_1838_);
v_i_1833_ = v___x_1839_;
v_b_1835_ = v___y_1837_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___boxed(lean_object* v___x_1855_, lean_object* v_a_1856_, lean_object* v_as_1857_, lean_object* v_i_1858_, lean_object* v_stop_1859_, lean_object* v_b_1860_){
_start:
{
uint8_t v___x_23995__boxed_1861_; size_t v_i_boxed_1862_; size_t v_stop_boxed_1863_; lean_object* v_res_1864_; 
v___x_23995__boxed_1861_ = lean_unbox(v___x_1855_);
v_i_boxed_1862_ = lean_unbox_usize(v_i_1858_);
lean_dec(v_i_1858_);
v_stop_boxed_1863_ = lean_unbox_usize(v_stop_1859_);
lean_dec(v_stop_1859_);
v_res_1864_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14(v___x_23995__boxed_1861_, v_a_1856_, v_as_1857_, v_i_boxed_1862_, v_stop_boxed_1863_, v_b_1860_);
lean_dec_ref(v_as_1857_);
lean_dec(v_a_1856_);
return v_res_1864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__4(lean_object* v___y_1865_, size_t v_sz_1866_, size_t v_i_1867_, lean_object* v_bs_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_){
_start:
{
uint8_t v___x_1872_; 
v___x_1872_ = lean_usize_dec_lt(v_i_1867_, v_sz_1866_);
if (v___x_1872_ == 0)
{
lean_object* v___x_1873_; 
lean_dec(v___y_1865_);
v___x_1873_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1873_, 0, v_bs_1868_);
return v___x_1873_;
}
else
{
lean_object* v_v_1874_; lean_object* v_fst_1875_; lean_object* v_snd_1876_; lean_object* v___x_1877_; 
v_v_1874_ = lean_array_uget_borrowed(v_bs_1868_, v_i_1867_);
v_fst_1875_ = lean_ctor_get(v_v_1874_, 0);
v_snd_1876_ = lean_ctor_get(v_v_1874_, 1);
lean_inc(v___y_1865_);
lean_inc(v_snd_1876_);
lean_inc(v_fst_1875_);
v___x_1877_ = lp_mathlib_Mathlib_Tactic_DeprecateTo_mkDeprecationStx(v_fst_1875_, v_snd_1876_, v___y_1865_, v___y_1869_, v___y_1870_);
if (lean_obj_tag(v___x_1877_) == 0)
{
lean_object* v_a_1878_; lean_object* v___x_1879_; lean_object* v_bs_x27_1880_; size_t v___x_1881_; size_t v___x_1882_; lean_object* v___x_1883_; 
v_a_1878_ = lean_ctor_get(v___x_1877_, 0);
lean_inc(v_a_1878_);
lean_dec_ref_known(v___x_1877_, 1);
v___x_1879_ = lean_unsigned_to_nat(0u);
v_bs_x27_1880_ = lean_array_uset(v_bs_1868_, v_i_1867_, v___x_1879_);
v___x_1881_ = ((size_t)1ULL);
v___x_1882_ = lean_usize_add(v_i_1867_, v___x_1881_);
v___x_1883_ = lean_array_uset(v_bs_x27_1880_, v_i_1867_, v_a_1878_);
v_i_1867_ = v___x_1882_;
v_bs_1868_ = v___x_1883_;
goto _start;
}
else
{
lean_object* v_a_1885_; lean_object* v___x_1887_; uint8_t v_isShared_1888_; uint8_t v_isSharedCheck_1892_; 
lean_dec_ref(v_bs_1868_);
lean_dec(v___y_1865_);
v_a_1885_ = lean_ctor_get(v___x_1877_, 0);
v_isSharedCheck_1892_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1892_ == 0)
{
v___x_1887_ = v___x_1877_;
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
else
{
lean_inc(v_a_1885_);
lean_dec(v___x_1877_);
v___x_1887_ = lean_box(0);
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
v_resetjp_1886_:
{
lean_object* v___x_1890_; 
if (v_isShared_1888_ == 0)
{
v___x_1890_ = v___x_1887_;
goto v_reusejp_1889_;
}
else
{
lean_object* v_reuseFailAlloc_1891_; 
v_reuseFailAlloc_1891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1891_, 0, v_a_1885_);
v___x_1890_ = v_reuseFailAlloc_1891_;
goto v_reusejp_1889_;
}
v_reusejp_1889_:
{
return v___x_1890_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__4___boxed(lean_object* v___y_1893_, lean_object* v_sz_1894_, lean_object* v_i_1895_, lean_object* v_bs_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_){
_start:
{
size_t v_sz_boxed_1900_; size_t v_i_boxed_1901_; lean_object* v_res_1902_; 
v_sz_boxed_1900_ = lean_unbox_usize(v_sz_1894_);
lean_dec(v_sz_1894_);
v_i_boxed_1901_ = lean_unbox_usize(v_i_1895_);
lean_dec(v_i_1895_);
v_res_1902_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__4(v___y_1893_, v_sz_boxed_1900_, v_i_boxed_1901_, v_bs_1896_, v___y_1897_, v___y_1898_);
lean_dec(v___y_1898_);
lean_dec_ref(v___y_1897_);
return v_res_1902_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1905_; lean_object* v___x_1906_; 
v___x_1905_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__1));
v___x_1906_ = l_Lean_stringToMessageData(v___x_1905_);
return v___x_1906_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__4(void){
_start:
{
lean_object* v___x_1908_; lean_object* v___x_1909_; 
v___x_1908_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__3));
v___x_1909_ = l_Lean_stringToMessageData(v___x_1908_);
return v___x_1909_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; 
v___x_1914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__8));
v___x_1915_ = l_Lean_stringToMessageData(v___x_1914_);
return v___x_1915_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1917_; lean_object* v___x_1918_; 
v___x_1917_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__10));
v___x_1918_ = l_Lean_stringToMessageData(v___x_1917_);
return v___x_1918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1(lean_object* v_a_1921_, lean_object* v_id_1922_, lean_object* v___x_1923_, lean_object* v___x_1924_, lean_object* v_tk_1925_, uint8_t v___x_1926_, lean_object* v_cmd_1927_, lean_object* v___y_1928_, lean_object* v_a_1929_, lean_object* v_env_1930_, lean_object* v_a_x3f_1931_){
_start:
{
lean_object* v___y_1934_; lean_object* v___y_1935_; lean_object* v___y_1936_; lean_object* v___y_1937_; lean_object* v___y_1942_; size_t v___y_1943_; lean_object* v___y_1944_; lean_object* v___y_1945_; lean_object* v___y_1946_; uint8_t v___y_1947_; size_t v___y_1958_; lean_object* v___y_1959_; lean_object* v___y_1960_; lean_object* v___y_1961_; size_t v___y_1962_; lean_object* v___y_1963_; lean_object* v___y_1964_; lean_object* v___y_1965_; lean_object* v___y_1966_; size_t v___y_1978_; lean_object* v___y_1979_; lean_object* v___y_1980_; lean_object* v___y_1981_; lean_object* v___y_1982_; size_t v___y_1983_; lean_object* v___y_1984_; lean_object* v___y_1985_; lean_object* v___y_1986_; size_t v___y_1987_; lean_object* v___y_1988_; lean_object* v___y_1989_; lean_object* v___y_2008_; lean_object* v___y_2009_; size_t v___y_2010_; size_t v___y_2011_; lean_object* v___y_2012_; lean_object* v___y_2013_; lean_object* v___y_2014_; lean_object* v___y_2015_; lean_object* v___y_2016_; lean_object* v___y_2017_; size_t v___y_2018_; size_t v___y_2021_; lean_object* v___y_2022_; lean_object* v___y_2023_; lean_object* v___y_2024_; size_t v___y_2025_; lean_object* v___y_2026_; lean_object* v___y_2027_; lean_object* v___y_2028_; size_t v___y_2029_; lean_object* v___y_2030_; lean_object* v___y_2031_; lean_object* v___y_2032_; lean_object* v___y_2044_; size_t v___y_2045_; lean_object* v___y_2046_; lean_object* v___y_2047_; lean_object* v___y_2048_; lean_object* v___y_2049_; lean_object* v___y_2050_; lean_object* v___y_2051_; size_t v___y_2052_; lean_object* v___y_2053_; lean_object* v___y_2056_; lean_object* v___y_2057_; lean_object* v___y_2058_; lean_object* v___y_2059_; lean_object* v___y_2060_; lean_object* v_news_2061_; lean_object* v___y_2062_; lean_object* v___y_2063_; lean_object* v___y_2082_; lean_object* v___y_2083_; lean_object* v___y_2084_; lean_object* v___y_2085_; lean_object* v___y_2086_; lean_object* v___y_2087_; lean_object* v___y_2088_; lean_object* v___y_2089_; lean_object* v___y_2090_; lean_object* v___y_2091_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v___y_2104_; lean_object* v___y_2105_; lean_object* v___y_2106_; lean_object* v___y_2107_; lean_object* v___y_2108_; lean_object* v___y_2109_; lean_object* v___y_2110_; lean_object* v___y_2111_; lean_object* v___y_2137_; lean_object* v___y_2138_; lean_object* v___y_2139_; lean_object* v_warn_2140_; lean_object* v___y_2141_; lean_object* v___y_2142_; lean_object* v___y_2172_; lean_object* v___y_2173_; lean_object* v___y_2174_; lean_object* v_warn_2175_; lean_object* v___y_2176_; lean_object* v___y_2177_; lean_object* v___y_2190_; lean_object* v___y_2191_; lean_object* v___x_2202_; lean_object* v_env_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v_a_2207_; lean_object* v___y_2218_; lean_object* v___x_2228_; uint8_t v___x_2229_; 
v___x_2202_ = lean_st_ref_get(v_a_1921_);
v_env_2203_ = lean_ctor_get(v___x_2202_, 0);
lean_inc_ref(v_env_2203_);
lean_dec(v___x_2202_);
v___x_2204_ = lp_mathlib_Mathlib_Tactic_DeprecateTo_newNames(v_env_1930_, v_env_2203_);
v___x_2205_ = lean_array_get_size(v___x_2204_);
v___x_2228_ = lean_mk_empty_array_with_capacity(v___x_1924_);
v___x_2229_ = lean_nat_dec_lt(v___x_1924_, v___x_2205_);
if (v___x_2229_ == 0)
{
v_a_2207_ = v___x_2228_;
goto v___jp_2206_;
}
else
{
uint8_t v___x_2230_; 
v___x_2230_ = lean_nat_dec_le(v___x_2205_, v___x_2205_);
if (v___x_2230_ == 0)
{
if (v___x_2229_ == 0)
{
v_a_2207_ = v___x_2228_;
goto v___jp_2206_;
}
else
{
size_t v___x_2231_; size_t v___x_2232_; lean_object* v___x_2233_; 
v___x_2231_ = ((size_t)0ULL);
v___x_2232_ = lean_usize_of_nat(v___x_2205_);
v___x_2233_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19(v___x_2204_, v___x_2231_, v___x_2232_, v___x_2228_, v_a_1929_, v_a_1921_);
v___y_2218_ = v___x_2233_;
goto v___jp_2217_;
}
}
else
{
size_t v___x_2234_; size_t v___x_2235_; lean_object* v___x_2236_; 
v___x_2234_ = ((size_t)0ULL);
v___x_2235_ = lean_usize_of_nat(v___x_2205_);
v___x_2236_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__19(v___x_2204_, v___x_2234_, v___x_2235_, v___x_2228_, v_a_1929_, v_a_1921_);
v___y_2218_ = v___x_2236_;
goto v___jp_2217_;
}
}
v___jp_1933_:
{
lean_object* v___x_1938_; lean_object* v___x_1939_; 
v___x_1938_ = l_Lean_stringToMessageData(v___y_1937_);
v___x_1939_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7(v_tk_1925_, v___x_1938_, v___y_1935_, v___y_1936_);
if (lean_obj_tag(v___x_1939_) == 0)
{
lean_object* v___x_1940_; 
lean_dec_ref_known(v___x_1939_, 1);
v___x_1940_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___y_1934_, v___y_1935_, v___y_1936_);
return v___x_1940_;
}
else
{
lean_dec_ref(v___y_1934_);
return v___x_1939_;
}
}
v___jp_1941_:
{
if (v___y_1947_ == 0)
{
lean_object* v___x_1948_; 
lean_dec_ref(v___y_1944_);
lean_dec(v___x_1924_);
v___x_1948_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___y_1942_, v___y_1946_, v___y_1945_);
return v___x_1948_;
}
else
{
lean_object* v___x_1949_; lean_object* v___x_1950_; uint8_t v___x_1951_; 
v___x_1949_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__0));
v___x_1950_ = lean_array_get_size(v___y_1944_);
v___x_1951_ = lean_nat_dec_lt(v___x_1924_, v___x_1950_);
lean_dec(v___x_1924_);
if (v___x_1951_ == 0)
{
lean_dec_ref(v___y_1944_);
v___y_1934_ = v___y_1942_;
v___y_1935_ = v___y_1946_;
v___y_1936_ = v___y_1945_;
v___y_1937_ = v___x_1949_;
goto v___jp_1933_;
}
else
{
uint8_t v___x_1952_; 
v___x_1952_ = lean_nat_dec_le(v___x_1950_, v___x_1950_);
if (v___x_1952_ == 0)
{
if (v___x_1951_ == 0)
{
lean_dec_ref(v___y_1944_);
v___y_1934_ = v___y_1942_;
v___y_1935_ = v___y_1946_;
v___y_1936_ = v___y_1945_;
v___y_1937_ = v___x_1949_;
goto v___jp_1933_;
}
else
{
size_t v___x_1953_; lean_object* v___x_1954_; 
v___x_1953_ = lean_usize_of_nat(v___x_1950_);
v___x_1954_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8(v___y_1944_, v___y_1943_, v___x_1953_, v___x_1949_);
lean_dec_ref(v___y_1944_);
v___y_1934_ = v___y_1942_;
v___y_1935_ = v___y_1946_;
v___y_1936_ = v___y_1945_;
v___y_1937_ = v___x_1954_;
goto v___jp_1933_;
}
}
else
{
size_t v___x_1955_; lean_object* v___x_1956_; 
v___x_1955_ = lean_usize_of_nat(v___x_1950_);
v___x_1956_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__8(v___y_1944_, v___y_1943_, v___x_1955_, v___x_1949_);
lean_dec_ref(v___y_1944_);
v___y_1934_ = v___y_1942_;
v___y_1935_ = v___y_1946_;
v___y_1936_ = v___y_1945_;
v___y_1937_ = v___x_1956_;
goto v___jp_1933_;
}
}
}
}
v___jp_1957_:
{
lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___f_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; uint8_t v___x_1974_; 
v___x_1967_ = lean_mk_empty_array_with_capacity(v___x_1923_);
v___x_1968_ = lean_array_push(v___x_1967_, v___y_1961_);
v___x_1969_ = l_Array_append___redArg(v___x_1968_, v___y_1960_);
lean_dec_ref(v___y_1960_);
v___x_1970_ = lean_box_usize(v___y_1958_);
lean_inc(v___x_1924_);
v___f_1971_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___boxed), 12, 5);
lean_closure_set(v___f_1971_, 0, v___x_1969_);
lean_closure_set(v___f_1971_, 1, v___x_1970_);
lean_closure_set(v___f_1971_, 2, v___x_1924_);
lean_closure_set(v___f_1971_, 3, v___x_1923_);
lean_closure_set(v___f_1971_, 4, v___y_1959_);
v___x_1972_ = lean_array_get_size(v___y_1963_);
v___x_1973_ = lean_array_get_size(v___y_1964_);
v___x_1974_ = lean_nat_dec_eq(v___x_1972_, v___x_1973_);
if (v___x_1974_ == 0)
{
lean_dec_ref(v___y_1964_);
v___y_1942_ = v___f_1971_;
v___y_1943_ = v___y_1962_;
v___y_1944_ = v___y_1963_;
v___y_1945_ = v___y_1966_;
v___y_1946_ = v___y_1965_;
v___y_1947_ = v___x_1926_;
goto v___jp_1941_;
}
else
{
uint8_t v___x_1975_; 
v___x_1975_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg(v___y_1963_, v___y_1964_, v___x_1972_);
lean_dec_ref(v___y_1964_);
if (v___x_1975_ == 0)
{
v___y_1942_ = v___f_1971_;
v___y_1943_ = v___y_1962_;
v___y_1944_ = v___y_1963_;
v___y_1945_ = v___y_1966_;
v___y_1946_ = v___y_1965_;
v___y_1947_ = v___x_1926_;
goto v___jp_1941_;
}
else
{
lean_object* v___x_1976_; 
lean_dec_ref(v___y_1963_);
lean_dec(v___x_1924_);
v___x_1976_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_1971_, v___y_1965_, v___y_1966_);
return v___x_1976_;
}
}
}
v___jp_1977_:
{
lean_object* v___x_1990_; 
v___x_1990_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__4(v___y_1989_, v___y_1987_, v___y_1983_, v___y_1982_, v___y_1980_, v___y_1986_);
if (lean_obj_tag(v___x_1990_) == 0)
{
lean_object* v_a_1991_; uint8_t v___x_1992_; 
v_a_1991_ = lean_ctor_get(v___x_1990_, 0);
lean_inc(v_a_1991_);
lean_dec_ref_known(v___x_1990_, 1);
v___x_1992_ = l_Lean_Syntax_structEq(v___y_1981_, v_cmd_1927_);
if (v___x_1992_ == 0)
{
lean_dec(v___y_1984_);
lean_dec(v_cmd_1927_);
v___y_1958_ = v___y_1978_;
v___y_1959_ = v___y_1979_;
v___y_1960_ = v_a_1991_;
v___y_1961_ = v___y_1981_;
v___y_1962_ = v___y_1983_;
v___y_1963_ = v___y_1985_;
v___y_1964_ = v___y_1988_;
v___y_1965_ = v___y_1980_;
v___y_1966_ = v___y_1986_;
goto v___jp_1957_;
}
else
{
lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; 
v___x_1993_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__2);
v___x_1994_ = l_Lean_MessageData_ofSyntax(v___y_1984_);
v___x_1995_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1995_, 0, v___x_1993_);
lean_ctor_set(v___x_1995_, 1, v___x_1994_);
v___x_1996_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__4, &lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__4);
v___x_1997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1997_, 0, v___x_1995_);
lean_ctor_set(v___x_1997_, 1, v___x_1996_);
v___x_1998_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7(v_cmd_1927_, v___x_1997_, v___y_1980_, v___y_1986_);
lean_dec(v_cmd_1927_);
if (lean_obj_tag(v___x_1998_) == 0)
{
lean_dec_ref_known(v___x_1998_, 1);
v___y_1958_ = v___y_1978_;
v___y_1959_ = v___y_1979_;
v___y_1960_ = v_a_1991_;
v___y_1961_ = v___y_1981_;
v___y_1962_ = v___y_1983_;
v___y_1963_ = v___y_1985_;
v___y_1964_ = v___y_1988_;
v___y_1965_ = v___y_1980_;
v___y_1966_ = v___y_1986_;
goto v___jp_1957_;
}
else
{
lean_dec(v_a_1991_);
lean_dec_ref(v___y_1988_);
lean_dec_ref(v___y_1985_);
lean_dec(v___y_1981_);
lean_dec_ref(v___y_1979_);
lean_dec(v___x_1924_);
lean_dec(v___x_1923_);
return v___x_1998_;
}
}
}
else
{
lean_object* v_a_1999_; lean_object* v___x_2001_; uint8_t v_isShared_2002_; uint8_t v_isSharedCheck_2006_; 
lean_dec_ref(v___y_1988_);
lean_dec_ref(v___y_1985_);
lean_dec(v___y_1984_);
lean_dec(v___y_1981_);
lean_dec_ref(v___y_1979_);
lean_dec(v_cmd_1927_);
lean_dec(v___x_1924_);
lean_dec(v___x_1923_);
v_a_1999_ = lean_ctor_get(v___x_1990_, 0);
v_isSharedCheck_2006_ = !lean_is_exclusive(v___x_1990_);
if (v_isSharedCheck_2006_ == 0)
{
v___x_2001_ = v___x_1990_;
v_isShared_2002_ = v_isSharedCheck_2006_;
goto v_resetjp_2000_;
}
else
{
lean_inc(v_a_1999_);
lean_dec(v___x_1990_);
v___x_2001_ = lean_box(0);
v_isShared_2002_ = v_isSharedCheck_2006_;
goto v_resetjp_2000_;
}
v_resetjp_2000_:
{
lean_object* v___x_2004_; 
if (v_isShared_2002_ == 0)
{
v___x_2004_ = v___x_2001_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2005_; 
v_reuseFailAlloc_2005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2005_, 0, v_a_1999_);
v___x_2004_ = v_reuseFailAlloc_2005_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
return v___x_2004_;
}
}
}
}
v___jp_2007_:
{
lean_object* v___x_2019_; 
v___x_2019_ = lean_box(0);
v___y_1978_ = v___y_2011_;
v___y_1979_ = v___y_2013_;
v___y_1980_ = v___y_2015_;
v___y_1981_ = v___y_2014_;
v___y_1982_ = v___y_2008_;
v___y_1983_ = v___y_2018_;
v___y_1984_ = v___y_2012_;
v___y_1985_ = v___y_2017_;
v___y_1986_ = v___y_2016_;
v___y_1987_ = v___y_2010_;
v___y_1988_ = v___y_2009_;
v___y_1989_ = v___x_2019_;
goto v___jp_1977_;
}
v___jp_2020_:
{
lean_object* v___x_2033_; 
v___x_2033_ = lean_string_append(v___y_2030_, v___y_2032_);
lean_dec_ref(v___y_2032_);
if (lean_obj_tag(v___y_1928_) == 0)
{
v___y_2008_ = v___y_2024_;
v___y_2009_ = v___y_2031_;
v___y_2010_ = v___y_2029_;
v___y_2011_ = v___y_2021_;
v___y_2012_ = v___y_2026_;
v___y_2013_ = v___x_2033_;
v___y_2014_ = v___y_2023_;
v___y_2015_ = v___y_2022_;
v___y_2016_ = v___y_2028_;
v___y_2017_ = v___y_2027_;
v___y_2018_ = v___y_2025_;
goto v___jp_2007_;
}
else
{
if (v___x_1926_ == 0)
{
lean_dec_ref_known(v___y_1928_, 1);
v___y_2008_ = v___y_2024_;
v___y_2009_ = v___y_2031_;
v___y_2010_ = v___y_2029_;
v___y_2011_ = v___y_2021_;
v___y_2012_ = v___y_2026_;
v___y_2013_ = v___x_2033_;
v___y_2014_ = v___y_2023_;
v___y_2015_ = v___y_2022_;
v___y_2016_ = v___y_2028_;
v___y_2017_ = v___y_2027_;
v___y_2018_ = v___y_2025_;
goto v___jp_2007_;
}
else
{
lean_object* v_val_2034_; lean_object* v___x_2036_; uint8_t v_isShared_2037_; uint8_t v_isSharedCheck_2042_; 
v_val_2034_ = lean_ctor_get(v___y_1928_, 0);
v_isSharedCheck_2042_ = !lean_is_exclusive(v___y_1928_);
if (v_isSharedCheck_2042_ == 0)
{
v___x_2036_ = v___y_1928_;
v_isShared_2037_ = v_isSharedCheck_2042_;
goto v_resetjp_2035_;
}
else
{
lean_inc(v_val_2034_);
lean_dec(v___y_1928_);
v___x_2036_ = lean_box(0);
v_isShared_2037_ = v_isSharedCheck_2042_;
goto v_resetjp_2035_;
}
v_resetjp_2035_:
{
lean_object* v___x_2038_; lean_object* v___x_2040_; 
v___x_2038_ = l_Lean_TSyntax_getString(v_val_2034_);
lean_dec(v_val_2034_);
if (v_isShared_2037_ == 0)
{
lean_ctor_set(v___x_2036_, 0, v___x_2038_);
v___x_2040_ = v___x_2036_;
goto v_reusejp_2039_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v___x_2038_);
v___x_2040_ = v_reuseFailAlloc_2041_;
goto v_reusejp_2039_;
}
v_reusejp_2039_:
{
v___y_1978_ = v___y_2021_;
v___y_1979_ = v___x_2033_;
v___y_1980_ = v___y_2022_;
v___y_1981_ = v___y_2023_;
v___y_1982_ = v___y_2024_;
v___y_1983_ = v___y_2025_;
v___y_1984_ = v___y_2026_;
v___y_1985_ = v___y_2027_;
v___y_1986_ = v___y_2028_;
v___y_1987_ = v___y_2029_;
v___y_1988_ = v___y_2031_;
v___y_1989_ = v___x_2040_;
goto v___jp_1977_;
}
}
}
}
}
v___jp_2043_:
{
lean_object* v___x_2054_; 
v___x_2054_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__0___closed__0));
v___y_2021_ = v___y_2045_;
v___y_2022_ = v___y_2050_;
v___y_2023_ = v___y_2049_;
v___y_2024_ = v___y_2047_;
v___y_2025_ = v___y_2045_;
v___y_2026_ = v___y_2044_;
v___y_2027_ = v___y_2051_;
v___y_2028_ = v___y_2046_;
v___y_2029_ = v___y_2052_;
v___y_2030_ = v___y_2053_;
v___y_2031_ = v___y_2048_;
v___y_2032_ = v___x_2054_;
goto v___jp_2020_;
}
v___jp_2055_:
{
lean_object* v___x_2064_; lean_object* v___x_2065_; size_t v_sz_2066_; size_t v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; uint8_t v___x_2075_; 
v___x_2064_ = l_Array_zip___redArg(v_id_1922_, v_news_2061_);
lean_dec_ref(v_news_2061_);
lean_dec_ref(v_id_1922_);
v___x_2065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__5));
v_sz_2066_ = lean_array_size(v___x_2064_);
v___x_2067_ = ((size_t)0ULL);
lean_inc_ref(v___x_2064_);
v___x_2068_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__2(v_sz_2066_, v___x_2067_, v___x_2064_);
v___x_2069_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__6));
v___x_2070_ = lean_array_to_list(v___x_2068_);
v___x_2071_ = lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__3(v___x_2070_);
v___x_2072_ = lean_string_append(v___x_2069_, v___x_2071_);
lean_dec_ref(v___x_2071_);
v___x_2073_ = lean_string_append(v___x_2065_, v___x_2072_);
lean_dec_ref(v___x_2072_);
v___x_2074_ = lean_array_get_size(v___y_2058_);
v___x_2075_ = lean_nat_dec_eq(v___x_2074_, v___x_1924_);
if (v___x_2075_ == 0)
{
if (v___x_1926_ == 0)
{
lean_dec_ref(v___y_2058_);
v___y_2044_ = v___y_2057_;
v___y_2045_ = v___x_2067_;
v___y_2046_ = v___y_2063_;
v___y_2047_ = v___x_2064_;
v___y_2048_ = v___y_2060_;
v___y_2049_ = v___y_2056_;
v___y_2050_ = v___y_2062_;
v___y_2051_ = v___y_2059_;
v___y_2052_ = v_sz_2066_;
v___y_2053_ = v___x_2073_;
goto v___jp_2043_;
}
else
{
lean_object* v___x_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; 
v___x_2076_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__7));
v___x_2077_ = lean_array_to_list(v___y_2058_);
v___x_2078_ = lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10(v___x_2077_);
v___x_2079_ = lean_string_append(v___x_2069_, v___x_2078_);
lean_dec_ref(v___x_2078_);
v___x_2080_ = lean_string_append(v___x_2076_, v___x_2079_);
lean_dec_ref(v___x_2079_);
v___y_2021_ = v___x_2067_;
v___y_2022_ = v___y_2062_;
v___y_2023_ = v___y_2056_;
v___y_2024_ = v___x_2064_;
v___y_2025_ = v___x_2067_;
v___y_2026_ = v___y_2057_;
v___y_2027_ = v___y_2059_;
v___y_2028_ = v___y_2063_;
v___y_2029_ = v_sz_2066_;
v___y_2030_ = v___x_2073_;
v___y_2031_ = v___y_2060_;
v___y_2032_ = v___x_2080_;
goto v___jp_2020_;
}
}
else
{
lean_dec_ref(v___y_2058_);
v___y_2044_ = v___y_2057_;
v___y_2045_ = v___x_2067_;
v___y_2046_ = v___y_2063_;
v___y_2047_ = v___x_2064_;
v___y_2048_ = v___y_2060_;
v___y_2049_ = v___y_2056_;
v___y_2050_ = v___y_2062_;
v___y_2051_ = v___y_2059_;
v___y_2052_ = v_sz_2066_;
v___y_2053_ = v___x_2073_;
goto v___jp_2043_;
}
}
v___jp_2081_:
{
lean_object* v___x_2092_; uint8_t v___x_2093_; 
v___x_2092_ = lean_box(0);
v___x_2093_ = l_Lean_Syntax_structEq(v___y_2087_, v___x_2092_);
lean_dec(v___y_2087_);
if (v___x_2093_ == 0)
{
if (v___x_1926_ == 0)
{
lean_dec_ref(v___y_2083_);
v___y_2056_ = v___y_2082_;
v___y_2057_ = v___y_2084_;
v___y_2058_ = v___y_2085_;
v___y_2059_ = v___y_2086_;
v___y_2060_ = v___y_2089_;
v_news_2061_ = v___y_2088_;
v___y_2062_ = v___y_2090_;
v___y_2063_ = v___y_2091_;
goto v___jp_2055_;
}
else
{
lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; 
v___x_2094_ = lean_box(0);
v___x_2095_ = lean_array_get(v___x_2094_, v___y_2083_, v___x_1924_);
lean_dec_ref(v___y_2083_);
v___x_2096_ = lean_mk_empty_array_with_capacity(v___x_1923_);
lean_inc(v___x_2095_);
v___x_2097_ = lean_array_push(v___x_2096_, v___x_2095_);
v___x_2098_ = lp_mathlib_Array_erase___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__12(v___y_2088_, v___x_2095_);
lean_dec(v___x_2095_);
v___x_2099_ = l_Array_append___redArg(v___x_2097_, v___x_2098_);
lean_dec_ref(v___x_2098_);
v___y_2056_ = v___y_2082_;
v___y_2057_ = v___y_2084_;
v___y_2058_ = v___y_2085_;
v___y_2059_ = v___y_2086_;
v___y_2060_ = v___y_2089_;
v_news_2061_ = v___x_2099_;
v___y_2062_ = v___y_2090_;
v___y_2063_ = v___y_2091_;
goto v___jp_2055_;
}
}
else
{
lean_dec_ref(v___y_2083_);
v___y_2056_ = v___y_2082_;
v___y_2057_ = v___y_2084_;
v___y_2058_ = v___y_2085_;
v___y_2059_ = v___y_2086_;
v___y_2060_ = v___y_2089_;
v_news_2061_ = v___y_2088_;
v___y_2062_ = v___y_2090_;
v___y_2063_ = v___y_2091_;
goto v___jp_2055_;
}
}
v___jp_2100_:
{
lean_object* v___x_2112_; uint8_t v___x_2113_; 
v___x_2112_ = lean_array_get_size(v___y_2111_);
v___x_2113_ = lean_nat_dec_eq(v___x_2112_, v___x_1923_);
if (v___x_2113_ == 0)
{
if (v___x_1926_ == 0)
{
lean_dec(v___y_2103_);
v___y_2082_ = v___y_2101_;
v___y_2083_ = v___y_2111_;
v___y_2084_ = v___y_2105_;
v___y_2085_ = v___y_2106_;
v___y_2086_ = v___y_2108_;
v___y_2087_ = v___y_2107_;
v___y_2088_ = v___y_2109_;
v___y_2089_ = v___y_2110_;
v___y_2090_ = v___y_2104_;
v___y_2091_ = v___y_2102_;
goto v___jp_2081_;
}
else
{
lean_object* v___x_2114_; uint8_t v___x_2115_; 
v___x_2114_ = lean_box(0);
v___x_2115_ = l_Lean_Syntax_structEq(v___y_2107_, v___x_2114_);
if (v___x_2115_ == 0)
{
lean_object* v___x_2116_; lean_object* v___x_2117_; lean_object* v_fst_2118_; lean_object* v___x_2120_; uint8_t v_isShared_2121_; uint8_t v_isSharedCheck_2134_; 
v___x_2116_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14___closed__0));
lean_inc(v___x_1924_);
v___x_2117_ = l_List_get_x21Internal___redArg(v___x_2116_, v___y_2103_, v___x_1924_);
lean_dec(v___y_2103_);
v_fst_2118_ = lean_ctor_get(v___x_2117_, 0);
v_isSharedCheck_2134_ = !lean_is_exclusive(v___x_2117_);
if (v_isSharedCheck_2134_ == 0)
{
lean_object* v_unused_2135_; 
v_unused_2135_ = lean_ctor_get(v___x_2117_, 1);
lean_dec(v_unused_2135_);
v___x_2120_ = v___x_2117_;
v_isShared_2121_ = v_isSharedCheck_2134_;
goto v_resetjp_2119_;
}
else
{
lean_inc(v_fst_2118_);
lean_dec(v___x_2117_);
v___x_2120_ = lean_box(0);
v_isShared_2121_ = v_isSharedCheck_2134_;
goto v_resetjp_2119_;
}
v_resetjp_2119_:
{
lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2125_; 
v___x_2122_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__9, &lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__9);
v___x_2123_ = l_Lean_MessageData_ofName(v_fst_2118_);
if (v_isShared_2121_ == 0)
{
lean_ctor_set_tag(v___x_2120_, 7);
lean_ctor_set(v___x_2120_, 1, v___x_2123_);
lean_ctor_set(v___x_2120_, 0, v___x_2122_);
v___x_2125_ = v___x_2120_;
goto v_reusejp_2124_;
}
else
{
lean_object* v_reuseFailAlloc_2133_; 
v_reuseFailAlloc_2133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2133_, 0, v___x_2122_);
lean_ctor_set(v_reuseFailAlloc_2133_, 1, v___x_2123_);
v___x_2125_ = v_reuseFailAlloc_2133_;
goto v_reusejp_2124_;
}
v_reusejp_2124_:
{
lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; 
v___x_2126_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__11, &lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__11);
v___x_2127_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2127_, 0, v___x_2125_);
lean_ctor_set(v___x_2127_, 1, v___x_2126_);
v___x_2128_ = l_Nat_reprFast(v___x_2112_);
v___x_2129_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2129_, 0, v___x_2128_);
v___x_2130_ = l_Lean_MessageData_ofFormat(v___x_2129_);
v___x_2131_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2131_, 0, v___x_2127_);
lean_ctor_set(v___x_2131_, 1, v___x_2130_);
v___x_2132_ = lp_mathlib_Lean_logError___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__13(v___x_2131_, v___y_2104_, v___y_2102_);
if (lean_obj_tag(v___x_2132_) == 0)
{
lean_dec_ref_known(v___x_2132_, 1);
v___y_2082_ = v___y_2101_;
v___y_2083_ = v___y_2111_;
v___y_2084_ = v___y_2105_;
v___y_2085_ = v___y_2106_;
v___y_2086_ = v___y_2108_;
v___y_2087_ = v___y_2107_;
v___y_2088_ = v___y_2109_;
v___y_2089_ = v___y_2110_;
v___y_2090_ = v___y_2104_;
v___y_2091_ = v___y_2102_;
goto v___jp_2081_;
}
else
{
lean_dec_ref(v___y_2111_);
lean_dec_ref(v___y_2110_);
lean_dec_ref(v___y_2109_);
lean_dec_ref(v___y_2108_);
lean_dec(v___y_2107_);
lean_dec_ref(v___y_2106_);
lean_dec(v___y_2105_);
lean_dec(v___y_2101_);
lean_dec(v___y_1928_);
lean_dec(v_cmd_1927_);
lean_dec(v___x_1924_);
lean_dec(v___x_1923_);
lean_dec_ref(v_id_1922_);
return v___x_2132_;
}
}
}
}
else
{
lean_dec(v___y_2103_);
v___y_2082_ = v___y_2101_;
v___y_2083_ = v___y_2111_;
v___y_2084_ = v___y_2105_;
v___y_2085_ = v___y_2106_;
v___y_2086_ = v___y_2108_;
v___y_2087_ = v___y_2107_;
v___y_2088_ = v___y_2109_;
v___y_2089_ = v___y_2110_;
v___y_2090_ = v___y_2104_;
v___y_2091_ = v___y_2102_;
goto v___jp_2081_;
}
}
}
else
{
lean_dec(v___y_2103_);
v___y_2082_ = v___y_2101_;
v___y_2083_ = v___y_2111_;
v___y_2084_ = v___y_2105_;
v___y_2085_ = v___y_2106_;
v___y_2086_ = v___y_2108_;
v___y_2087_ = v___y_2107_;
v___y_2088_ = v___y_2109_;
v___y_2089_ = v___y_2110_;
v___y_2090_ = v___y_2104_;
v___y_2091_ = v___y_2102_;
goto v___jp_2081_;
}
}
v___jp_2136_:
{
lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v_fst_2146_; lean_object* v_snd_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; 
v___x_2143_ = lean_box(0);
v___x_2144_ = lean_array_get_borrowed(v___x_2143_, v_id_1922_, v___x_1924_);
lean_inc(v_cmd_1927_);
lean_inc(v___x_2144_);
v___x_2145_ = lp_mathlib_Mathlib_Tactic_DeprecateTo_renameTheorem(v___x_2144_, v_cmd_1927_);
v_fst_2146_ = lean_ctor_get(v___x_2145_, 0);
lean_inc(v_fst_2146_);
v_snd_2147_ = lean_ctor_get(v___x_2145_, 1);
lean_inc(v_snd_2147_);
lean_dec_ref(v___x_2145_);
v___x_2148_ = l_Lean_Syntax_getArg(v_fst_2146_, v___x_1924_);
v___x_2149_ = l_Lean_Syntax_getId(v___x_2148_);
v___x_2150_ = l_Lean_Name_eraseMacroScopes(v___x_2149_);
lean_dec(v___x_2149_);
v___x_2151_ = lp_mathlib_Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11(v___x_2150_, v___x_1926_, v___y_2141_, v___y_2142_);
if (lean_obj_tag(v___x_2151_) == 0)
{
lean_object* v_a_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; uint8_t v___x_2155_; 
v_a_2152_ = lean_ctor_get(v___x_2151_, 0);
lean_inc(v_a_2152_);
lean_dec_ref_known(v___x_2151_, 1);
v___x_2153_ = lean_array_get_size(v___y_2138_);
v___x_2154_ = lean_mk_empty_array_with_capacity(v___x_1924_);
v___x_2155_ = lean_nat_dec_lt(v___x_1924_, v___x_2153_);
if (v___x_2155_ == 0)
{
v___y_2101_ = v_snd_2147_;
v___y_2102_ = v___y_2142_;
v___y_2103_ = v_a_2152_;
v___y_2104_ = v___y_2141_;
v___y_2105_ = v___x_2148_;
v___y_2106_ = v___y_2137_;
v___y_2107_ = v_fst_2146_;
v___y_2108_ = v_warn_2140_;
v___y_2109_ = v___y_2138_;
v___y_2110_ = v___y_2139_;
v___y_2111_ = v___x_2154_;
goto v___jp_2100_;
}
else
{
uint8_t v___x_2156_; 
v___x_2156_ = lean_nat_dec_le(v___x_2153_, v___x_2153_);
if (v___x_2156_ == 0)
{
if (v___x_2155_ == 0)
{
v___y_2101_ = v_snd_2147_;
v___y_2102_ = v___y_2142_;
v___y_2103_ = v_a_2152_;
v___y_2104_ = v___y_2141_;
v___y_2105_ = v___x_2148_;
v___y_2106_ = v___y_2137_;
v___y_2107_ = v_fst_2146_;
v___y_2108_ = v_warn_2140_;
v___y_2109_ = v___y_2138_;
v___y_2110_ = v___y_2139_;
v___y_2111_ = v___x_2154_;
goto v___jp_2100_;
}
else
{
size_t v___x_2157_; size_t v___x_2158_; lean_object* v___x_2159_; 
v___x_2157_ = ((size_t)0ULL);
v___x_2158_ = lean_usize_of_nat(v___x_2153_);
v___x_2159_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14(v___x_1926_, v_a_2152_, v___y_2138_, v___x_2157_, v___x_2158_, v___x_2154_);
v___y_2101_ = v_snd_2147_;
v___y_2102_ = v___y_2142_;
v___y_2103_ = v_a_2152_;
v___y_2104_ = v___y_2141_;
v___y_2105_ = v___x_2148_;
v___y_2106_ = v___y_2137_;
v___y_2107_ = v_fst_2146_;
v___y_2108_ = v_warn_2140_;
v___y_2109_ = v___y_2138_;
v___y_2110_ = v___y_2139_;
v___y_2111_ = v___x_2159_;
goto v___jp_2100_;
}
}
else
{
size_t v___x_2160_; size_t v___x_2161_; lean_object* v___x_2162_; 
v___x_2160_ = ((size_t)0ULL);
v___x_2161_ = lean_usize_of_nat(v___x_2153_);
v___x_2162_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__14(v___x_1926_, v_a_2152_, v___y_2138_, v___x_2160_, v___x_2161_, v___x_2154_);
v___y_2101_ = v_snd_2147_;
v___y_2102_ = v___y_2142_;
v___y_2103_ = v_a_2152_;
v___y_2104_ = v___y_2141_;
v___y_2105_ = v___x_2148_;
v___y_2106_ = v___y_2137_;
v___y_2107_ = v_fst_2146_;
v___y_2108_ = v_warn_2140_;
v___y_2109_ = v___y_2138_;
v___y_2110_ = v___y_2139_;
v___y_2111_ = v___x_2162_;
goto v___jp_2100_;
}
}
}
else
{
lean_object* v_a_2163_; lean_object* v___x_2165_; uint8_t v_isShared_2166_; uint8_t v_isSharedCheck_2170_; 
lean_dec(v___x_2148_);
lean_dec(v_snd_2147_);
lean_dec(v_fst_2146_);
lean_dec_ref(v_warn_2140_);
lean_dec_ref(v___y_2139_);
lean_dec_ref(v___y_2138_);
lean_dec_ref(v___y_2137_);
lean_dec(v___y_1928_);
lean_dec(v_cmd_1927_);
lean_dec(v___x_1924_);
lean_dec(v___x_1923_);
lean_dec_ref(v_id_1922_);
v_a_2163_ = lean_ctor_get(v___x_2151_, 0);
v_isSharedCheck_2170_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2170_ == 0)
{
v___x_2165_ = v___x_2151_;
v_isShared_2166_ = v_isSharedCheck_2170_;
goto v_resetjp_2164_;
}
else
{
lean_inc(v_a_2163_);
lean_dec(v___x_2151_);
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
v___jp_2171_:
{
lean_object* v___x_2178_; lean_object* v___x_2179_; uint8_t v___x_2180_; 
v___x_2178_ = lean_array_get_size(v___y_2173_);
v___x_2179_ = lean_array_get_size(v_id_1922_);
v___x_2180_ = lean_nat_dec_lt(v___x_2178_, v___x_2179_);
if (v___x_2180_ == 0)
{
v___y_2137_ = v___y_2172_;
v___y_2138_ = v___y_2173_;
v___y_2139_ = v___y_2174_;
v_warn_2140_ = v_warn_2175_;
v___y_2141_ = v___y_2176_;
v___y_2142_ = v___y_2177_;
goto v___jp_2136_;
}
else
{
lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; 
lean_inc_ref(v_id_1922_);
v___x_2181_ = lean_array_to_list(v_id_1922_);
v___x_2182_ = l_List_drop___redArg(v___x_2178_, v___x_2181_);
lean_dec(v___x_2181_);
v___x_2183_ = lean_box(0);
v___x_2184_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg(v___x_2182_, v___x_2183_, v___y_2176_, v___y_2177_);
if (lean_obj_tag(v___x_2184_) == 0)
{
lean_object* v___x_2185_; lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; 
lean_dec_ref_known(v___x_2184_, 1);
v___x_2185_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__12));
v___x_2186_ = lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__17(v___x_2182_);
v___x_2187_ = lean_string_append(v___x_2185_, v___x_2186_);
lean_dec_ref(v___x_2186_);
v___x_2188_ = lean_array_push(v_warn_2175_, v___x_2187_);
v___y_2137_ = v___y_2172_;
v___y_2138_ = v___y_2173_;
v___y_2139_ = v___y_2174_;
v_warn_2140_ = v___x_2188_;
v___y_2141_ = v___y_2176_;
v___y_2142_ = v___y_2177_;
goto v___jp_2136_;
}
else
{
lean_dec(v___x_2182_);
lean_dec_ref(v_warn_2175_);
lean_dec_ref(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec_ref(v___y_2172_);
lean_dec(v___y_1928_);
lean_dec(v_cmd_1927_);
lean_dec(v___x_1924_);
lean_dec(v___x_1923_);
lean_dec_ref(v_id_1922_);
return v___x_2184_;
}
}
}
v___jp_2189_:
{
lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; uint8_t v___x_2195_; 
v___x_2192_ = lean_mk_empty_array_with_capacity(v___x_1924_);
v___x_2193_ = lean_array_get_size(v_id_1922_);
v___x_2194_ = lean_array_get_size(v___y_2191_);
v___x_2195_ = lean_nat_dec_lt(v___x_2193_, v___x_2194_);
if (v___x_2195_ == 0)
{
lean_inc_ref(v___x_2192_);
v___y_2172_ = v___y_2190_;
v___y_2173_ = v___y_2191_;
v___y_2174_ = v___x_2192_;
v_warn_2175_ = v___x_2192_;
v___y_2176_ = v_a_1929_;
v___y_2177_ = v_a_1921_;
goto v___jp_2171_;
}
else
{
lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; 
v___x_2196_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___closed__13));
lean_inc_ref(v___y_2191_);
v___x_2197_ = lean_array_to_list(v___y_2191_);
v___x_2198_ = l_List_drop___redArg(v___x_2193_, v___x_2197_);
lean_dec(v___x_2197_);
v___x_2199_ = lp_mathlib_List_toString___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__10(v___x_2198_);
v___x_2200_ = lean_string_append(v___x_2196_, v___x_2199_);
lean_dec_ref(v___x_2199_);
lean_inc_ref(v___x_2192_);
v___x_2201_ = lean_array_push(v___x_2192_, v___x_2200_);
v___y_2172_ = v___y_2190_;
v___y_2173_ = v___y_2191_;
v___y_2174_ = v___x_2192_;
v_warn_2175_ = v___x_2201_;
v___y_2176_ = v_a_1929_;
v___y_2177_ = v_a_1921_;
goto v___jp_2171_;
}
}
v___jp_2206_:
{
lean_object* v___x_2208_; uint8_t v___x_2209_; 
v___x_2208_ = lean_mk_empty_array_with_capacity(v___x_1924_);
v___x_2209_ = lean_nat_dec_lt(v___x_1924_, v___x_2205_);
if (v___x_2209_ == 0)
{
lean_dec_ref(v___x_2204_);
v___y_2190_ = v_a_2207_;
v___y_2191_ = v___x_2208_;
goto v___jp_2189_;
}
else
{
uint8_t v___x_2210_; 
v___x_2210_ = lean_nat_dec_le(v___x_2205_, v___x_2205_);
if (v___x_2210_ == 0)
{
if (v___x_2209_ == 0)
{
lean_dec_ref(v___x_2204_);
v___y_2190_ = v_a_2207_;
v___y_2191_ = v___x_2208_;
goto v___jp_2189_;
}
else
{
size_t v___x_2211_; size_t v___x_2212_; lean_object* v___x_2213_; 
v___x_2211_ = ((size_t)0ULL);
v___x_2212_ = lean_usize_of_nat(v___x_2205_);
lean_inc_ref(v_a_2207_);
v___x_2213_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18(v_a_2207_, v___x_2204_, v___x_2211_, v___x_2212_, v___x_2208_);
lean_dec_ref(v___x_2204_);
v___y_2190_ = v_a_2207_;
v___y_2191_ = v___x_2213_;
goto v___jp_2189_;
}
}
else
{
size_t v___x_2214_; size_t v___x_2215_; lean_object* v___x_2216_; 
v___x_2214_ = ((size_t)0ULL);
v___x_2215_ = lean_usize_of_nat(v___x_2205_);
lean_inc_ref(v_a_2207_);
v___x_2216_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__18(v_a_2207_, v___x_2204_, v___x_2214_, v___x_2215_, v___x_2208_);
lean_dec_ref(v___x_2204_);
v___y_2190_ = v_a_2207_;
v___y_2191_ = v___x_2216_;
goto v___jp_2189_;
}
}
}
v___jp_2217_:
{
if (lean_obj_tag(v___y_2218_) == 0)
{
lean_object* v_a_2219_; 
v_a_2219_ = lean_ctor_get(v___y_2218_, 0);
lean_inc(v_a_2219_);
lean_dec_ref_known(v___y_2218_, 1);
v_a_2207_ = v_a_2219_;
goto v___jp_2206_;
}
else
{
lean_object* v_a_2220_; lean_object* v___x_2222_; uint8_t v_isShared_2223_; uint8_t v_isSharedCheck_2227_; 
lean_dec_ref(v___x_2204_);
lean_dec(v___y_1928_);
lean_dec(v_cmd_1927_);
lean_dec(v___x_1924_);
lean_dec(v___x_1923_);
lean_dec_ref(v_id_1922_);
v_a_2220_ = lean_ctor_get(v___y_2218_, 0);
v_isSharedCheck_2227_ = !lean_is_exclusive(v___y_2218_);
if (v_isSharedCheck_2227_ == 0)
{
v___x_2222_ = v___y_2218_;
v_isShared_2223_ = v_isSharedCheck_2227_;
goto v_resetjp_2221_;
}
else
{
lean_inc(v_a_2220_);
lean_dec(v___y_2218_);
v___x_2222_ = lean_box(0);
v_isShared_2223_ = v_isSharedCheck_2227_;
goto v_resetjp_2221_;
}
v_resetjp_2221_:
{
lean_object* v___x_2225_; 
if (v_isShared_2223_ == 0)
{
v___x_2225_ = v___x_2222_;
goto v_reusejp_2224_;
}
else
{
lean_object* v_reuseFailAlloc_2226_; 
v_reuseFailAlloc_2226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2226_, 0, v_a_2220_);
v___x_2225_ = v_reuseFailAlloc_2226_;
goto v_reusejp_2224_;
}
v_reusejp_2224_:
{
return v___x_2225_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1___boxed(lean_object* v_a_2237_, lean_object* v_id_2238_, lean_object* v___x_2239_, lean_object* v___x_2240_, lean_object* v_tk_2241_, lean_object* v___x_2242_, lean_object* v_cmd_2243_, lean_object* v___y_2244_, lean_object* v_a_2245_, lean_object* v_env_2246_, lean_object* v_a_x3f_2247_, lean_object* v___y_2248_){
_start:
{
uint8_t v___x_24166__boxed_2249_; lean_object* v_res_2250_; 
v___x_24166__boxed_2249_ = lean_unbox(v___x_2242_);
v_res_2250_ = lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1(v_a_2237_, v_id_2238_, v___x_2239_, v___x_2240_, v_tk_2241_, v___x_24166__boxed_2249_, v_cmd_2243_, v___y_2244_, v_a_2245_, v_env_2246_, v_a_x3f_2247_);
lean_dec(v_a_x3f_2247_);
lean_dec_ref(v_a_2245_);
lean_dec(v_tk_2241_);
lean_dec(v_a_2237_);
return v_res_2250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1(lean_object* v_x_2251_, lean_object* v_a_2252_, lean_object* v_a_2253_){
_start:
{
lean_object* v___x_2255_; uint8_t v___x_2256_; 
v___x_2255_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_DeprecateTo_commandDeprecateTo_____________00__closed__3));
lean_inc(v_x_2251_);
v___x_2256_ = l_Lean_Syntax_isOfKind(v_x_2251_, v___x_2255_);
if (v___x_2256_ == 0)
{
lean_object* v___x_2257_; 
lean_dec(v_x_2251_);
v___x_2257_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__0___redArg();
return v___x_2257_;
}
else
{
lean_object* v___x_2258_; lean_object* v_tk_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v_cmd_2266_; lean_object* v___y_2268_; lean_object* v___x_2301_; 
v___x_2258_ = lean_unsigned_to_nat(0u);
v_tk_2259_ = l_Lean_Syntax_getArg(v_x_2251_, v___x_2258_);
v___x_2260_ = lean_unsigned_to_nat(1u);
v___x_2261_ = lean_unsigned_to_nat(2u);
v___x_2262_ = l_Lean_Syntax_getArg(v_x_2251_, v___x_2261_);
v___x_2263_ = lean_unsigned_to_nat(3u);
v___x_2264_ = l_Lean_Syntax_getArg(v_x_2251_, v___x_2263_);
v___x_2265_ = lean_unsigned_to_nat(5u);
v_cmd_2266_ = l_Lean_Syntax_getArg(v_x_2251_, v___x_2265_);
lean_dec(v_x_2251_);
v___x_2301_ = l_Lean_Syntax_getOptional_x3f(v___x_2264_);
lean_dec(v___x_2264_);
if (lean_obj_tag(v___x_2301_) == 0)
{
lean_object* v___x_2302_; 
v___x_2302_ = lean_box(0);
v___y_2268_ = v___x_2302_;
goto v___jp_2267_;
}
else
{
lean_object* v_val_2303_; lean_object* v___x_2305_; uint8_t v_isShared_2306_; uint8_t v_isSharedCheck_2310_; 
v_val_2303_ = lean_ctor_get(v___x_2301_, 0);
v_isSharedCheck_2310_ = !lean_is_exclusive(v___x_2301_);
if (v_isSharedCheck_2310_ == 0)
{
v___x_2305_ = v___x_2301_;
v_isShared_2306_ = v_isSharedCheck_2310_;
goto v_resetjp_2304_;
}
else
{
lean_inc(v_val_2303_);
lean_dec(v___x_2301_);
v___x_2305_ = lean_box(0);
v_isShared_2306_ = v_isSharedCheck_2310_;
goto v_resetjp_2304_;
}
v_resetjp_2304_:
{
lean_object* v___x_2308_; 
if (v_isShared_2306_ == 0)
{
v___x_2308_ = v___x_2305_;
goto v_reusejp_2307_;
}
else
{
lean_object* v_reuseFailAlloc_2309_; 
v_reuseFailAlloc_2309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2309_, 0, v_val_2303_);
v___x_2308_ = v_reuseFailAlloc_2309_;
goto v_reusejp_2307_;
}
v_reusejp_2307_:
{
v___y_2268_ = v___x_2308_;
goto v___jp_2267_;
}
}
}
v___jp_2267_:
{
lean_object* v___x_2269_; lean_object* v_env_2270_; lean_object* v_id_2271_; lean_object* v_r_2272_; 
v___x_2269_ = lean_st_ref_get(v_a_2253_);
v_env_2270_ = lean_ctor_get(v___x_2269_, 0);
lean_inc_ref(v_env_2270_);
lean_dec(v___x_2269_);
v_id_2271_ = l_Lean_Syntax_getArgs(v___x_2262_);
lean_dec(v___x_2262_);
lean_inc(v_cmd_2266_);
v_r_2272_ = l_Lean_Elab_Command_elabCommand(v_cmd_2266_, v_a_2252_, v_a_2253_);
if (lean_obj_tag(v_r_2272_) == 0)
{
lean_object* v_a_2273_; lean_object* v___x_2275_; uint8_t v_isShared_2276_; uint8_t v_isSharedCheck_2289_; 
v_a_2273_ = lean_ctor_get(v_r_2272_, 0);
v_isSharedCheck_2289_ = !lean_is_exclusive(v_r_2272_);
if (v_isSharedCheck_2289_ == 0)
{
v___x_2275_ = v_r_2272_;
v_isShared_2276_ = v_isSharedCheck_2289_;
goto v_resetjp_2274_;
}
else
{
lean_inc(v_a_2273_);
lean_dec(v_r_2272_);
v___x_2275_ = lean_box(0);
v_isShared_2276_ = v_isSharedCheck_2289_;
goto v_resetjp_2274_;
}
v_resetjp_2274_:
{
lean_object* v___x_2278_; 
lean_inc(v_a_2273_);
if (v_isShared_2276_ == 0)
{
lean_ctor_set_tag(v___x_2275_, 1);
v___x_2278_ = v___x_2275_;
goto v_reusejp_2277_;
}
else
{
lean_object* v_reuseFailAlloc_2288_; 
v_reuseFailAlloc_2288_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2288_, 0, v_a_2273_);
v___x_2278_ = v_reuseFailAlloc_2288_;
goto v_reusejp_2277_;
}
v_reusejp_2277_:
{
lean_object* v___x_2279_; 
v___x_2279_ = lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1(v_a_2253_, v_id_2271_, v___x_2260_, v___x_2258_, v_tk_2259_, v___x_2256_, v_cmd_2266_, v___y_2268_, v_a_2252_, v_env_2270_, v___x_2278_);
lean_dec_ref(v___x_2278_);
lean_dec(v_tk_2259_);
if (lean_obj_tag(v___x_2279_) == 0)
{
lean_object* v___x_2281_; uint8_t v_isShared_2282_; uint8_t v_isSharedCheck_2286_; 
v_isSharedCheck_2286_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2286_ == 0)
{
lean_object* v_unused_2287_; 
v_unused_2287_ = lean_ctor_get(v___x_2279_, 0);
lean_dec(v_unused_2287_);
v___x_2281_ = v___x_2279_;
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
else
{
lean_dec(v___x_2279_);
v___x_2281_ = lean_box(0);
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
v_resetjp_2280_:
{
lean_object* v___x_2284_; 
if (v_isShared_2282_ == 0)
{
lean_ctor_set(v___x_2281_, 0, v_a_2273_);
v___x_2284_ = v___x_2281_;
goto v_reusejp_2283_;
}
else
{
lean_object* v_reuseFailAlloc_2285_; 
v_reuseFailAlloc_2285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2285_, 0, v_a_2273_);
v___x_2284_ = v_reuseFailAlloc_2285_;
goto v_reusejp_2283_;
}
v_reusejp_2283_:
{
return v___x_2284_;
}
}
}
else
{
lean_dec(v_a_2273_);
return v___x_2279_;
}
}
}
}
else
{
lean_object* v_a_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; 
v_a_2290_ = lean_ctor_get(v_r_2272_, 0);
lean_inc(v_a_2290_);
lean_dec_ref_known(v_r_2272_, 1);
v___x_2291_ = lean_box(0);
v___x_2292_ = lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___lam__1(v_a_2253_, v_id_2271_, v___x_2260_, v___x_2258_, v_tk_2259_, v___x_2256_, v_cmd_2266_, v___y_2268_, v_a_2252_, v_env_2270_, v___x_2291_);
lean_dec(v_tk_2259_);
if (lean_obj_tag(v___x_2292_) == 0)
{
lean_object* v___x_2294_; uint8_t v_isShared_2295_; uint8_t v_isSharedCheck_2299_; 
v_isSharedCheck_2299_ = !lean_is_exclusive(v___x_2292_);
if (v_isSharedCheck_2299_ == 0)
{
lean_object* v_unused_2300_; 
v_unused_2300_ = lean_ctor_get(v___x_2292_, 0);
lean_dec(v_unused_2300_);
v___x_2294_ = v___x_2292_;
v_isShared_2295_ = v_isSharedCheck_2299_;
goto v_resetjp_2293_;
}
else
{
lean_dec(v___x_2292_);
v___x_2294_ = lean_box(0);
v_isShared_2295_ = v_isSharedCheck_2299_;
goto v_resetjp_2293_;
}
v_resetjp_2293_:
{
lean_object* v___x_2297_; 
if (v_isShared_2295_ == 0)
{
lean_ctor_set_tag(v___x_2294_, 1);
lean_ctor_set(v___x_2294_, 0, v_a_2290_);
v___x_2297_ = v___x_2294_;
goto v_reusejp_2296_;
}
else
{
lean_object* v_reuseFailAlloc_2298_; 
v_reuseFailAlloc_2298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2298_, 0, v_a_2290_);
v___x_2297_ = v_reuseFailAlloc_2298_;
goto v_reusejp_2296_;
}
v_reusejp_2296_:
{
return v___x_2297_;
}
}
}
else
{
lean_dec(v_a_2290_);
return v___x_2292_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1___boxed(lean_object* v_x_2311_, lean_object* v_a_2312_, lean_object* v_a_2313_, lean_object* v_a_2314_){
_start:
{
lean_object* v_res_2315_; 
v_res_2315_ = lp_mathlib_Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1(v_x_2311_, v_a_2312_, v_a_2313_);
lean_dec(v_a_2313_);
lean_dec_ref(v_a_2312_);
return v_res_2315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1(lean_object* v_declName_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_){
_start:
{
lean_object* v___x_2320_; 
v___x_2320_ = lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___redArg(v_declName_2316_, v___y_2318_);
return v___x_2320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1___boxed(lean_object* v_declName_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_){
_start:
{
lean_object* v_res_2325_; 
v_res_2325_ = lp_mathlib_Lean_isRec___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__1(v_declName_2321_, v___y_2322_, v___y_2323_);
lean_dec(v___y_2323_);
lean_dec_ref(v___y_2322_);
return v_res_2325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2(lean_object* v_declName_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_){
_start:
{
lean_object* v___x_2330_; 
v___x_2330_ = lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___redArg(v_declName_2326_, v___y_2328_);
return v___x_2330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2___boxed(lean_object* v_declName_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_){
_start:
{
lean_object* v_res_2335_; 
v_res_2335_ = lp_mathlib_Lean_Meta_isMatcher___at___00Lean_Name_isBlackListed___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__1_spec__2(v_declName_2331_, v___y_2332_, v___y_2333_);
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
return v_res_2335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5(size_t v_sz_2336_, size_t v_i_2337_, lean_object* v_bs_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_){
_start:
{
lean_object* v___x_2346_; 
v___x_2346_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___redArg(v_sz_2336_, v_i_2337_, v_bs_2338_, v___y_2343_, v___y_2344_);
return v___x_2346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5___boxed(lean_object* v_sz_2347_, lean_object* v_i_2348_, lean_object* v_bs_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
size_t v_sz_boxed_2357_; size_t v_i_boxed_2358_; lean_object* v_res_2359_; 
v_sz_boxed_2357_ = lean_unbox_usize(v_sz_2347_);
lean_dec(v_sz_2347_);
v_i_boxed_2358_ = lean_unbox_usize(v_i_2348_);
lean_dec(v_i_2348_);
v_res_2359_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__5(v_sz_boxed_2357_, v_i_boxed_2358_, v_bs_2349_, v___y_2350_, v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_);
lean_dec(v___y_2355_);
lean_dec_ref(v___y_2354_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
lean_dec(v___y_2351_);
lean_dec_ref(v___y_2350_);
return v_res_2359_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9(lean_object* v_xs_2360_, lean_object* v_ys_2361_, lean_object* v_hsz_2362_, lean_object* v_x_2363_, lean_object* v_x_2364_){
_start:
{
uint8_t v___x_2365_; 
v___x_2365_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___redArg(v_xs_2360_, v_ys_2361_, v_x_2363_);
return v___x_2365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9___boxed(lean_object* v_xs_2366_, lean_object* v_ys_2367_, lean_object* v_hsz_2368_, lean_object* v_x_2369_, lean_object* v_x_2370_){
_start:
{
uint8_t v_res_2371_; lean_object* v_r_2372_; 
v_res_2371_ = lp_mathlib_Array_isEqvAux___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__9(v_xs_2366_, v_ys_2367_, v_hsz_2368_, v_x_2369_, v_x_2370_);
lean_dec_ref(v_ys_2367_);
lean_dec_ref(v_xs_2366_);
v_r_2372_ = lean_box(v_res_2371_);
return v_r_2372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16(lean_object* v_as_2373_, lean_object* v_as_x27_2374_, lean_object* v_b_2375_, lean_object* v_a_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_){
_start:
{
lean_object* v___x_2380_; 
v___x_2380_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___redArg(v_as_x27_2374_, v_b_2375_, v___y_2377_, v___y_2378_);
return v___x_2380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16___boxed(lean_object* v_as_2381_, lean_object* v_as_x27_2382_, lean_object* v_b_2383_, lean_object* v_a_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_){
_start:
{
lean_object* v_res_2388_; 
v_res_2388_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__16(v_as_2381_, v_as_x27_2382_, v_b_2383_, v_a_2384_, v___y_2385_, v___y_2386_);
lean_dec(v___y_2386_);
lean_dec_ref(v___y_2385_);
lean_dec(v_as_x27_2382_);
lean_dec(v_as_2381_);
return v_res_2388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11(lean_object* v_msgData_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_){
_start:
{
lean_object* v___x_2393_; 
v___x_2393_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___redArg(v_msgData_2389_, v___y_2391_);
return v___x_2393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11___boxed(lean_object* v_msgData_2394_, lean_object* v___y_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_){
_start:
{
lean_object* v_res_2398_; 
v_res_2398_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__7_spec__10_spec__11(v_msgData_2394_, v___y_2395_, v___y_2396_);
lean_dec(v___y_2396_);
lean_dec_ref(v___y_2395_);
return v_res_2398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20(lean_object* v_opt_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_){
_start:
{
lean_object* v___x_2403_; 
v___x_2403_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___redArg(v_opt_2399_, v___y_2401_);
return v___x_2403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20___boxed(lean_object* v_opt_2404_, lean_object* v___y_2405_, lean_object* v___y_2406_, lean_object* v___y_2407_){
_start:
{
lean_object* v_res_2408_; 
v_res_2408_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00Mathlib_Tactic_DeprecateTo___aux__Mathlib__Tactic__DeprecateTo______elabRules__Mathlib__Tactic__DeprecateTo__commandDeprecateTo______________1_spec__11_spec__17_spec__20(v_opt_2404_, v___y_2405_, v___y_2406_);
lean_dec(v___y_2406_);
lean_dec_ref(v___y_2405_);
lean_dec_ref(v_opt_2404_);
return v_res_2408_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DeprecateTo(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Std_Time_Format(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_DeprecateTo(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Time_Format(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Std_Time_Format(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_DeprecateTo(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Time_Format(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DeprecateTo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_DeprecateTo(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_DeprecateTo(builtin);
}
#ifdef __cplusplus
}
#endif
