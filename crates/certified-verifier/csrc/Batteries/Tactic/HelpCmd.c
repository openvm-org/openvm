// Lean compiler output
// Module: Batteries.Tactic.HelpCmd
// Imports: public import Init public meta import Init public meta import Lean.Elab.Syntax public meta import Lean.DocString public meta import Batteries.Util.LibraryNote
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
lean_object* l_Array_instInhabited(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint64_t lean_string_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_findDocString_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_mkIdentFrom(lean_object*, lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_data_value_to_string(lean_object*);
lean_object* l_String_quote(lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Bool_repr___redArg(uint8_t);
lean_object* l_Lean_Name_reprPrec(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_instRepr_repr(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_ParserExtension_instInhabitedState_default;
extern lean_object* l_Lean_Parser_parserExtension;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Syntax_getId(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
extern lean_object* l_Lean_Elab_Tactic_tacticElabAttribute;
lean_object* l_Lean_KeyedDeclsAttribute_getEntries___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_macroAttribute;
extern lean_object* l_Lean_Elab_Command_commandElabAttribute;
extern lean_object* l_Lean_Elab_Term_termElabAttribute;
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_balance___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_batteries_Batteries_Util_LibraryNote_encodeNameForExport(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_getOptionDecls();
lean_object* l_String_decEq___boxed(lean_object*, lean_object*);
lean_object* l_List_eraseDupsBy___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t l_String_decLE(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Parser_mkParserOfConstant(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_attributeMapRef;
lean_object* l_Lean_instInhabitedPersistentEnvExtensionState___redArg(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Elab_Term_addCategoryInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_TSyntax_getString(lean_object*);
extern lean_object* lp_batteries_Batteries_Util_LibraryNote_libraryNoteExt;
lean_object* l_Lean_SimplePersistentEnvExtension_getEntries___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_MergeSort_Internal_mergeSortTR_u2082___redArg(lean_object*, lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "command#help_Option___"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 40, 198, 102, 90, 185, 163)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "withPosition"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(246, 171, 180, 145, 132, 143, 108, 238)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "#help "};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "option"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__16_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__18_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__19_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__20_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__21 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__21_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__21_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__22 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__22_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__23 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__23_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__24 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__24_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rawIdent"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__25 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__25_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__23_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__24_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__25_value),LEAN_SCALAR_PTR_LITERAL(220, 2, 179, 39, 67, 204, 226, 154)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__26_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__27 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__27_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__22_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__27_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__28 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__28_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__18_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__28_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__29 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__29_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__16_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__29_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__30 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__30_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__30_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__31 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__31_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__31_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__32 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__32_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Option______ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__32_value;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "option "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__1 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__1_value)}};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__2 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__2_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__3 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__3_value;
static const lean_ctor_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__3_value)}};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__4 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__4_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = " (currently: "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__5 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__5_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__6 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__6_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "String := "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__7 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__7_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Bool := "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__8 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__8_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Name := "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__9 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__9_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Nat := "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__10 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__10_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Int := "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__11 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__11_value;
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__12;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Syntax := "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__13 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__13_value;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0;
static const lean_string_object lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__1 = (const lean_object*)&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__1_value;
static const lean_ctor_object lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__1_value)}};
static const lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__2 = (const lean_object*)&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__2_value;
static lean_once_cell_t lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__3;
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "no options found (!)"};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__0_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__1;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "no options start with "};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__2_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "command#help_AttrAttribute___"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(40, 175, 164, 134, 117, 141, 248, 54)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "attr"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "token"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(15, 57, 25, 131, 204, 83, 142, 228)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "attribute"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(12, 118, 110, 9, 86, 239, 246, 250)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__14_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__29_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__17_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute______ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__17_value;
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__0 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__0_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "]: "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__1 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__1_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__2 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "no attributes found (!)"};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__0_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__1;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "no attributes start with "};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__2_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__AttrAttribute________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__AttrAttribute________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "command#help_Cats___"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(147, 149, 189, 11, 93, 80, 146, 185)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cats"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__29_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__7_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cats______ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__7_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__0 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "category "};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__2 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__3;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__4 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__4_value;
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__5;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__6 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__6_value;
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "no syntax categories found (!)"};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__0_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__1;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "no syntax categories start with "};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__2_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cats________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cats________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "command#help_Cat+______"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 107, 91, 27, 184, 55, 159, 109)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "cat"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "+"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__18_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__27_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__22_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__17_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__18_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__20_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__21 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__21_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__21_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__22 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__22_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b____________ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__22_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_tokensToList(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_tokensToList___boxed(lean_object*);
static const lean_closure_object lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_String_decEq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3___closed__0 = (const lean_object*)&lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_minOn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_minOn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "$"};
static const lean_object* lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2___closed__0 = (const lean_object*)&lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0___boxed(lean_object*);
static lean_once_cell_t lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8_spec__21___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "+ "};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__0 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__1;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__2 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic elab"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__0 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__0_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "macro"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__1 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__1_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "command elab"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__6 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__6_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term elab"};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__7 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__7_value;
static const lean_string_object lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "syntax ... ["};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__8 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__8_value;
static lean_once_cell_t lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10_spec__15(lean_object*);
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Std.Data.DHashMap.Internal.AssocList.Basic"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__0 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__0_value;
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Std.DHashMap.Internal.AssocList.get!"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__1 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__1_value;
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "key is not present in hash table"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__2 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__2_value;
static lean_once_cell_t lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "syntax ..."};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__1;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "... ["};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "syntax "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__4_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__5;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__0_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__1;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__2;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__3;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__4;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "no "};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__5 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__5_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = " declarations found"};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__7 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__7_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__8;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = " declarations start with "};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__9 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__9_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__10;
static const lean_string_object lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = " is not a syntax category"};
static const lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__11 = (const lean_object*)&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__11_value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__12;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8_spec__21(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cat_x2b______________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cat_x2b______________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "command#help_Note___"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(13, 175, 107, 243, 210, 243, 92, 43)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "note"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__21_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strLit"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__23_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__24_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(63, 157, 94, 66, 135, 29, 115, 44)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__11_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__16_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Note______ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__16_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LibraryNote"};
static const lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__0 = (const lean_object*)&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(134, 197, 35, 181, 239, 168, 57, 237)}};
static const lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__1 = (const lean_object*)&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__1_value;
static lean_once_cell_t lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__2;
static const lean_string_object lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "library_note "};
static const lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__3 = (const lean_object*)&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__3_value;
static const lean_string_object lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "/-- "};
static const lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__4 = (const lean_object*)&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__4_value;
static const lean_string_object lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " -/"};
static const lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__5 = (const lean_object*)&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__1;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__2;
static const lean_closure_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__3_value;
static const lean_closure_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__1___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__3_value)} };
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__4_value;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Note not found"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__9;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__10_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "command#help_Term+____"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 78, 238, 80, 63, 213, 149, 66)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__7_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Term_x2b________ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__0_value;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#help"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "command#help_Tactic+____"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 114, 62, 75, 196, 158, 31, 94)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__7_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b________ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "command#help_Conv+____"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 95, 249, 14, 158, 154, 204, 98)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__7_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b________ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5_value),LEAN_SCALAR_PTR_LITERAL(232, 67, 39, 189, 45, 247, 54, 81)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "command#help_Command+____"};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(164, 194, 243, 111, 222, 225, 28, 42)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__7_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_command_x23help__Command_x2b________ = (const lean_object*)&lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(lean_object* v_k_74_, lean_object* v_v_75_, lean_object* v_t_76_){
_start:
{
if (lean_obj_tag(v_t_76_) == 0)
{
lean_object* v_size_77_; lean_object* v_k_78_; lean_object* v_v_79_; lean_object* v_l_80_; lean_object* v_r_81_; lean_object* v___x_83_; uint8_t v_isShared_84_; uint8_t v_isSharedCheck_361_; 
v_size_77_ = lean_ctor_get(v_t_76_, 0);
v_k_78_ = lean_ctor_get(v_t_76_, 1);
v_v_79_ = lean_ctor_get(v_t_76_, 2);
v_l_80_ = lean_ctor_get(v_t_76_, 3);
v_r_81_ = lean_ctor_get(v_t_76_, 4);
v_isSharedCheck_361_ = !lean_is_exclusive(v_t_76_);
if (v_isSharedCheck_361_ == 0)
{
v___x_83_ = v_t_76_;
v_isShared_84_ = v_isSharedCheck_361_;
goto v_resetjp_82_;
}
else
{
lean_inc(v_r_81_);
lean_inc(v_l_80_);
lean_inc(v_v_79_);
lean_inc(v_k_78_);
lean_inc(v_size_77_);
lean_dec(v_t_76_);
v___x_83_ = lean_box(0);
v_isShared_84_ = v_isSharedCheck_361_;
goto v_resetjp_82_;
}
v_resetjp_82_:
{
uint8_t v___x_85_; 
v___x_85_ = lean_string_compare(v_k_74_, v_k_78_);
switch(v___x_85_)
{
case 0:
{
lean_object* v_impl_86_; lean_object* v___x_87_; 
lean_dec(v_size_77_);
v_impl_86_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v_k_74_, v_v_75_, v_l_80_);
v___x_87_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_81_) == 0)
{
lean_object* v_size_88_; lean_object* v_size_89_; lean_object* v_k_90_; lean_object* v_v_91_; lean_object* v_l_92_; lean_object* v_r_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v_size_88_ = lean_ctor_get(v_r_81_, 0);
v_size_89_ = lean_ctor_get(v_impl_86_, 0);
lean_inc(v_size_89_);
v_k_90_ = lean_ctor_get(v_impl_86_, 1);
lean_inc(v_k_90_);
v_v_91_ = lean_ctor_get(v_impl_86_, 2);
lean_inc(v_v_91_);
v_l_92_ = lean_ctor_get(v_impl_86_, 3);
lean_inc(v_l_92_);
v_r_93_ = lean_ctor_get(v_impl_86_, 4);
lean_inc(v_r_93_);
v___x_94_ = lean_unsigned_to_nat(3u);
v___x_95_ = lean_nat_mul(v___x_94_, v_size_88_);
v___x_96_ = lean_nat_dec_lt(v___x_95_, v_size_89_);
lean_dec(v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_100_; 
lean_dec(v_r_93_);
lean_dec(v_l_92_);
lean_dec(v_v_91_);
lean_dec(v_k_90_);
v___x_97_ = lean_nat_add(v___x_87_, v_size_89_);
lean_dec(v_size_89_);
v___x_98_ = lean_nat_add(v___x_97_, v_size_88_);
lean_dec(v___x_97_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 3, v_impl_86_);
lean_ctor_set(v___x_83_, 0, v___x_98_);
v___x_100_ = v___x_83_;
goto v_reusejp_99_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v___x_98_);
lean_ctor_set(v_reuseFailAlloc_101_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_101_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_101_, 3, v_impl_86_);
lean_ctor_set(v_reuseFailAlloc_101_, 4, v_r_81_);
v___x_100_ = v_reuseFailAlloc_101_;
goto v_reusejp_99_;
}
v_reusejp_99_:
{
return v___x_100_;
}
}
else
{
lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_167_; 
v_isSharedCheck_167_ = !lean_is_exclusive(v_impl_86_);
if (v_isSharedCheck_167_ == 0)
{
lean_object* v_unused_168_; lean_object* v_unused_169_; lean_object* v_unused_170_; lean_object* v_unused_171_; lean_object* v_unused_172_; 
v_unused_168_ = lean_ctor_get(v_impl_86_, 4);
lean_dec(v_unused_168_);
v_unused_169_ = lean_ctor_get(v_impl_86_, 3);
lean_dec(v_unused_169_);
v_unused_170_ = lean_ctor_get(v_impl_86_, 2);
lean_dec(v_unused_170_);
v_unused_171_ = lean_ctor_get(v_impl_86_, 1);
lean_dec(v_unused_171_);
v_unused_172_ = lean_ctor_get(v_impl_86_, 0);
lean_dec(v_unused_172_);
v___x_103_ = v_impl_86_;
v_isShared_104_ = v_isSharedCheck_167_;
goto v_resetjp_102_;
}
else
{
lean_dec(v_impl_86_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_167_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v_size_105_; lean_object* v_size_106_; lean_object* v_k_107_; lean_object* v_v_108_; lean_object* v_l_109_; lean_object* v_r_110_; lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v_size_105_ = lean_ctor_get(v_l_92_, 0);
v_size_106_ = lean_ctor_get(v_r_93_, 0);
v_k_107_ = lean_ctor_get(v_r_93_, 1);
v_v_108_ = lean_ctor_get(v_r_93_, 2);
v_l_109_ = lean_ctor_get(v_r_93_, 3);
v_r_110_ = lean_ctor_get(v_r_93_, 4);
v___x_111_ = lean_unsigned_to_nat(2u);
v___x_112_ = lean_nat_mul(v___x_111_, v_size_105_);
v___x_113_ = lean_nat_dec_lt(v_size_106_, v___x_112_);
lean_dec(v___x_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_142_; 
lean_inc(v_r_110_);
lean_inc(v_l_109_);
lean_inc(v_v_108_);
lean_inc(v_k_107_);
v_isSharedCheck_142_ = !lean_is_exclusive(v_r_93_);
if (v_isSharedCheck_142_ == 0)
{
lean_object* v_unused_143_; lean_object* v_unused_144_; lean_object* v_unused_145_; lean_object* v_unused_146_; lean_object* v_unused_147_; 
v_unused_143_ = lean_ctor_get(v_r_93_, 4);
lean_dec(v_unused_143_);
v_unused_144_ = lean_ctor_get(v_r_93_, 3);
lean_dec(v_unused_144_);
v_unused_145_ = lean_ctor_get(v_r_93_, 2);
lean_dec(v_unused_145_);
v_unused_146_ = lean_ctor_get(v_r_93_, 1);
lean_dec(v_unused_146_);
v_unused_147_ = lean_ctor_get(v_r_93_, 0);
lean_dec(v_unused_147_);
v___x_115_ = v_r_93_;
v_isShared_116_ = v_isSharedCheck_142_;
goto v_resetjp_114_;
}
else
{
lean_dec(v_r_93_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_142_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___y_120_; lean_object* v___y_121_; lean_object* v___y_122_; lean_object* v___x_130_; lean_object* v___y_132_; 
v___x_117_ = lean_nat_add(v___x_87_, v_size_89_);
lean_dec(v_size_89_);
v___x_118_ = lean_nat_add(v___x_117_, v_size_88_);
lean_dec(v___x_117_);
v___x_130_ = lean_nat_add(v___x_87_, v_size_105_);
if (lean_obj_tag(v_l_109_) == 0)
{
lean_object* v_size_140_; 
v_size_140_ = lean_ctor_get(v_l_109_, 0);
lean_inc(v_size_140_);
v___y_132_ = v_size_140_;
goto v___jp_131_;
}
else
{
lean_object* v___x_141_; 
v___x_141_ = lean_unsigned_to_nat(0u);
v___y_132_ = v___x_141_;
goto v___jp_131_;
}
v___jp_119_:
{
lean_object* v___x_123_; lean_object* v___x_125_; 
v___x_123_ = lean_nat_add(v___y_120_, v___y_122_);
lean_dec(v___y_122_);
lean_dec(v___y_120_);
if (v_isShared_116_ == 0)
{
lean_ctor_set(v___x_115_, 4, v_r_81_);
lean_ctor_set(v___x_115_, 3, v_r_110_);
lean_ctor_set(v___x_115_, 2, v_v_79_);
lean_ctor_set(v___x_115_, 1, v_k_78_);
lean_ctor_set(v___x_115_, 0, v___x_123_);
v___x_125_ = v___x_115_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_129_; 
v_reuseFailAlloc_129_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_129_, 0, v___x_123_);
lean_ctor_set(v_reuseFailAlloc_129_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_129_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_129_, 3, v_r_110_);
lean_ctor_set(v_reuseFailAlloc_129_, 4, v_r_81_);
v___x_125_ = v_reuseFailAlloc_129_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
lean_object* v___x_127_; 
if (v_isShared_104_ == 0)
{
lean_ctor_set(v___x_103_, 4, v___x_125_);
lean_ctor_set(v___x_103_, 3, v___y_121_);
lean_ctor_set(v___x_103_, 2, v_v_108_);
lean_ctor_set(v___x_103_, 1, v_k_107_);
lean_ctor_set(v___x_103_, 0, v___x_118_);
v___x_127_ = v___x_103_;
goto v_reusejp_126_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v___x_118_);
lean_ctor_set(v_reuseFailAlloc_128_, 1, v_k_107_);
lean_ctor_set(v_reuseFailAlloc_128_, 2, v_v_108_);
lean_ctor_set(v_reuseFailAlloc_128_, 3, v___y_121_);
lean_ctor_set(v_reuseFailAlloc_128_, 4, v___x_125_);
v___x_127_ = v_reuseFailAlloc_128_;
goto v_reusejp_126_;
}
v_reusejp_126_:
{
return v___x_127_;
}
}
}
v___jp_131_:
{
lean_object* v___x_133_; lean_object* v___x_135_; 
v___x_133_ = lean_nat_add(v___x_130_, v___y_132_);
lean_dec(v___y_132_);
lean_dec(v___x_130_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v_l_109_);
lean_ctor_set(v___x_83_, 3, v_l_92_);
lean_ctor_set(v___x_83_, 2, v_v_91_);
lean_ctor_set(v___x_83_, 1, v_k_90_);
lean_ctor_set(v___x_83_, 0, v___x_133_);
v___x_135_ = v___x_83_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v___x_133_);
lean_ctor_set(v_reuseFailAlloc_139_, 1, v_k_90_);
lean_ctor_set(v_reuseFailAlloc_139_, 2, v_v_91_);
lean_ctor_set(v_reuseFailAlloc_139_, 3, v_l_92_);
lean_ctor_set(v_reuseFailAlloc_139_, 4, v_l_109_);
v___x_135_ = v_reuseFailAlloc_139_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
lean_object* v___x_136_; 
v___x_136_ = lean_nat_add(v___x_87_, v_size_88_);
if (lean_obj_tag(v_r_110_) == 0)
{
lean_object* v_size_137_; 
v_size_137_ = lean_ctor_get(v_r_110_, 0);
lean_inc(v_size_137_);
v___y_120_ = v___x_136_;
v___y_121_ = v___x_135_;
v___y_122_ = v_size_137_;
goto v___jp_119_;
}
else
{
lean_object* v___x_138_; 
v___x_138_ = lean_unsigned_to_nat(0u);
v___y_120_ = v___x_136_;
v___y_121_ = v___x_135_;
v___y_122_ = v___x_138_;
goto v___jp_119_;
}
}
}
}
}
else
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_153_; 
lean_del_object(v___x_83_);
v___x_148_ = lean_nat_add(v___x_87_, v_size_89_);
lean_dec(v_size_89_);
v___x_149_ = lean_nat_add(v___x_148_, v_size_88_);
lean_dec(v___x_148_);
v___x_150_ = lean_nat_add(v___x_87_, v_size_88_);
v___x_151_ = lean_nat_add(v___x_150_, v_size_106_);
lean_dec(v___x_150_);
lean_inc_ref(v_r_81_);
if (v_isShared_104_ == 0)
{
lean_ctor_set(v___x_103_, 4, v_r_81_);
lean_ctor_set(v___x_103_, 3, v_r_93_);
lean_ctor_set(v___x_103_, 2, v_v_79_);
lean_ctor_set(v___x_103_, 1, v_k_78_);
lean_ctor_set(v___x_103_, 0, v___x_151_);
v___x_153_ = v___x_103_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_151_);
lean_ctor_set(v_reuseFailAlloc_166_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_166_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_166_, 3, v_r_93_);
lean_ctor_set(v_reuseFailAlloc_166_, 4, v_r_81_);
v___x_153_ = v_reuseFailAlloc_166_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_160_; 
v_isSharedCheck_160_ = !lean_is_exclusive(v_r_81_);
if (v_isSharedCheck_160_ == 0)
{
lean_object* v_unused_161_; lean_object* v_unused_162_; lean_object* v_unused_163_; lean_object* v_unused_164_; lean_object* v_unused_165_; 
v_unused_161_ = lean_ctor_get(v_r_81_, 4);
lean_dec(v_unused_161_);
v_unused_162_ = lean_ctor_get(v_r_81_, 3);
lean_dec(v_unused_162_);
v_unused_163_ = lean_ctor_get(v_r_81_, 2);
lean_dec(v_unused_163_);
v_unused_164_ = lean_ctor_get(v_r_81_, 1);
lean_dec(v_unused_164_);
v_unused_165_ = lean_ctor_get(v_r_81_, 0);
lean_dec(v_unused_165_);
v___x_155_ = v_r_81_;
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
else
{
lean_dec(v_r_81_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v___x_158_; 
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 4, v___x_153_);
lean_ctor_set(v___x_155_, 3, v_l_92_);
lean_ctor_set(v___x_155_, 2, v_v_91_);
lean_ctor_set(v___x_155_, 1, v_k_90_);
lean_ctor_set(v___x_155_, 0, v___x_149_);
v___x_158_ = v___x_155_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v___x_149_);
lean_ctor_set(v_reuseFailAlloc_159_, 1, v_k_90_);
lean_ctor_set(v_reuseFailAlloc_159_, 2, v_v_91_);
lean_ctor_set(v_reuseFailAlloc_159_, 3, v_l_92_);
lean_ctor_set(v_reuseFailAlloc_159_, 4, v___x_153_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_173_; 
v_l_173_ = lean_ctor_get(v_impl_86_, 3);
lean_inc(v_l_173_);
if (lean_obj_tag(v_l_173_) == 0)
{
lean_object* v_r_174_; lean_object* v_k_175_; lean_object* v_v_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_187_; 
v_r_174_ = lean_ctor_get(v_impl_86_, 4);
v_k_175_ = lean_ctor_get(v_impl_86_, 1);
v_v_176_ = lean_ctor_get(v_impl_86_, 2);
v_isSharedCheck_187_ = !lean_is_exclusive(v_impl_86_);
if (v_isSharedCheck_187_ == 0)
{
lean_object* v_unused_188_; lean_object* v_unused_189_; 
v_unused_188_ = lean_ctor_get(v_impl_86_, 3);
lean_dec(v_unused_188_);
v_unused_189_ = lean_ctor_get(v_impl_86_, 0);
lean_dec(v_unused_189_);
v___x_178_ = v_impl_86_;
v_isShared_179_ = v_isSharedCheck_187_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_r_174_);
lean_inc(v_v_176_);
lean_inc(v_k_175_);
lean_dec(v_impl_86_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_187_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___x_180_; lean_object* v___x_182_; 
v___x_180_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_174_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 3, v_r_174_);
lean_ctor_set(v___x_178_, 2, v_v_79_);
lean_ctor_set(v___x_178_, 1, v_k_78_);
lean_ctor_set(v___x_178_, 0, v___x_87_);
v___x_182_ = v___x_178_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_87_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v_r_174_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v_r_174_);
v___x_182_ = v_reuseFailAlloc_186_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
lean_object* v___x_184_; 
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v___x_182_);
lean_ctor_set(v___x_83_, 3, v_l_173_);
lean_ctor_set(v___x_83_, 2, v_v_176_);
lean_ctor_set(v___x_83_, 1, v_k_175_);
lean_ctor_set(v___x_83_, 0, v___x_180_);
v___x_184_ = v___x_83_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v_k_175_);
lean_ctor_set(v_reuseFailAlloc_185_, 2, v_v_176_);
lean_ctor_set(v_reuseFailAlloc_185_, 3, v_l_173_);
lean_ctor_set(v_reuseFailAlloc_185_, 4, v___x_182_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
else
{
lean_object* v_r_190_; 
v_r_190_ = lean_ctor_get(v_impl_86_, 4);
lean_inc(v_r_190_);
if (lean_obj_tag(v_r_190_) == 0)
{
lean_object* v_k_191_; lean_object* v_v_192_; lean_object* v___x_194_; uint8_t v_isShared_195_; uint8_t v_isSharedCheck_215_; 
v_k_191_ = lean_ctor_get(v_impl_86_, 1);
v_v_192_ = lean_ctor_get(v_impl_86_, 2);
v_isSharedCheck_215_ = !lean_is_exclusive(v_impl_86_);
if (v_isSharedCheck_215_ == 0)
{
lean_object* v_unused_216_; lean_object* v_unused_217_; lean_object* v_unused_218_; 
v_unused_216_ = lean_ctor_get(v_impl_86_, 4);
lean_dec(v_unused_216_);
v_unused_217_ = lean_ctor_get(v_impl_86_, 3);
lean_dec(v_unused_217_);
v_unused_218_ = lean_ctor_get(v_impl_86_, 0);
lean_dec(v_unused_218_);
v___x_194_ = v_impl_86_;
v_isShared_195_ = v_isSharedCheck_215_;
goto v_resetjp_193_;
}
else
{
lean_inc(v_v_192_);
lean_inc(v_k_191_);
lean_dec(v_impl_86_);
v___x_194_ = lean_box(0);
v_isShared_195_ = v_isSharedCheck_215_;
goto v_resetjp_193_;
}
v_resetjp_193_:
{
lean_object* v_k_196_; lean_object* v_v_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_211_; 
v_k_196_ = lean_ctor_get(v_r_190_, 1);
v_v_197_ = lean_ctor_get(v_r_190_, 2);
v_isSharedCheck_211_ = !lean_is_exclusive(v_r_190_);
if (v_isSharedCheck_211_ == 0)
{
lean_object* v_unused_212_; lean_object* v_unused_213_; lean_object* v_unused_214_; 
v_unused_212_ = lean_ctor_get(v_r_190_, 4);
lean_dec(v_unused_212_);
v_unused_213_ = lean_ctor_get(v_r_190_, 3);
lean_dec(v_unused_213_);
v_unused_214_ = lean_ctor_get(v_r_190_, 0);
lean_dec(v_unused_214_);
v___x_199_ = v_r_190_;
v_isShared_200_ = v_isSharedCheck_211_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_v_197_);
lean_inc(v_k_196_);
lean_dec(v_r_190_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_211_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_201_; lean_object* v___x_203_; 
v___x_201_ = lean_unsigned_to_nat(3u);
if (v_isShared_200_ == 0)
{
lean_ctor_set(v___x_199_, 4, v_l_173_);
lean_ctor_set(v___x_199_, 3, v_l_173_);
lean_ctor_set(v___x_199_, 2, v_v_192_);
lean_ctor_set(v___x_199_, 1, v_k_191_);
lean_ctor_set(v___x_199_, 0, v___x_87_);
v___x_203_ = v___x_199_;
goto v_reusejp_202_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_87_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v_k_191_);
lean_ctor_set(v_reuseFailAlloc_210_, 2, v_v_192_);
lean_ctor_set(v_reuseFailAlloc_210_, 3, v_l_173_);
lean_ctor_set(v_reuseFailAlloc_210_, 4, v_l_173_);
v___x_203_ = v_reuseFailAlloc_210_;
goto v_reusejp_202_;
}
v_reusejp_202_:
{
lean_object* v___x_205_; 
if (v_isShared_195_ == 0)
{
lean_ctor_set(v___x_194_, 4, v_l_173_);
lean_ctor_set(v___x_194_, 2, v_v_79_);
lean_ctor_set(v___x_194_, 1, v_k_78_);
lean_ctor_set(v___x_194_, 0, v___x_87_);
v___x_205_ = v___x_194_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_209_; 
v_reuseFailAlloc_209_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_209_, 0, v___x_87_);
lean_ctor_set(v_reuseFailAlloc_209_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_209_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_209_, 3, v_l_173_);
lean_ctor_set(v_reuseFailAlloc_209_, 4, v_l_173_);
v___x_205_ = v_reuseFailAlloc_209_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
lean_object* v___x_207_; 
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v___x_205_);
lean_ctor_set(v___x_83_, 3, v___x_203_);
lean_ctor_set(v___x_83_, 2, v_v_197_);
lean_ctor_set(v___x_83_, 1, v_k_196_);
lean_ctor_set(v___x_83_, 0, v___x_201_);
v___x_207_ = v___x_83_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v___x_201_);
lean_ctor_set(v_reuseFailAlloc_208_, 1, v_k_196_);
lean_ctor_set(v_reuseFailAlloc_208_, 2, v_v_197_);
lean_ctor_set(v_reuseFailAlloc_208_, 3, v___x_203_);
lean_ctor_set(v_reuseFailAlloc_208_, 4, v___x_205_);
v___x_207_ = v_reuseFailAlloc_208_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
return v___x_207_;
}
}
}
}
}
}
else
{
lean_object* v___x_219_; lean_object* v___x_221_; 
v___x_219_ = lean_unsigned_to_nat(2u);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v_r_190_);
lean_ctor_set(v___x_83_, 3, v_impl_86_);
lean_ctor_set(v___x_83_, 0, v___x_219_);
v___x_221_ = v___x_83_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v___x_219_);
lean_ctor_set(v_reuseFailAlloc_222_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_222_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_222_, 3, v_impl_86_);
lean_ctor_set(v_reuseFailAlloc_222_, 4, v_r_190_);
v___x_221_ = v_reuseFailAlloc_222_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
return v___x_221_;
}
}
}
}
}
case 1:
{
lean_object* v___x_224_; 
lean_dec(v_v_79_);
lean_dec(v_k_78_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 2, v_v_75_);
lean_ctor_set(v___x_83_, 1, v_k_74_);
v___x_224_ = v___x_83_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v_size_77_);
lean_ctor_set(v_reuseFailAlloc_225_, 1, v_k_74_);
lean_ctor_set(v_reuseFailAlloc_225_, 2, v_v_75_);
lean_ctor_set(v_reuseFailAlloc_225_, 3, v_l_80_);
lean_ctor_set(v_reuseFailAlloc_225_, 4, v_r_81_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
default: 
{
lean_object* v_impl_226_; lean_object* v___x_227_; 
lean_dec(v_size_77_);
v_impl_226_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v_k_74_, v_v_75_, v_r_81_);
v___x_227_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_80_) == 0)
{
lean_object* v_size_228_; lean_object* v_size_229_; lean_object* v_k_230_; lean_object* v_v_231_; lean_object* v_l_232_; lean_object* v_r_233_; lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; 
v_size_228_ = lean_ctor_get(v_l_80_, 0);
v_size_229_ = lean_ctor_get(v_impl_226_, 0);
lean_inc(v_size_229_);
v_k_230_ = lean_ctor_get(v_impl_226_, 1);
lean_inc(v_k_230_);
v_v_231_ = lean_ctor_get(v_impl_226_, 2);
lean_inc(v_v_231_);
v_l_232_ = lean_ctor_get(v_impl_226_, 3);
lean_inc(v_l_232_);
v_r_233_ = lean_ctor_get(v_impl_226_, 4);
lean_inc(v_r_233_);
v___x_234_ = lean_unsigned_to_nat(3u);
v___x_235_ = lean_nat_mul(v___x_234_, v_size_228_);
v___x_236_ = lean_nat_dec_lt(v___x_235_, v_size_229_);
lean_dec(v___x_235_);
if (v___x_236_ == 0)
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_240_; 
lean_dec(v_r_233_);
lean_dec(v_l_232_);
lean_dec(v_v_231_);
lean_dec(v_k_230_);
v___x_237_ = lean_nat_add(v___x_227_, v_size_228_);
v___x_238_ = lean_nat_add(v___x_237_, v_size_229_);
lean_dec(v_size_229_);
lean_dec(v___x_237_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v_impl_226_);
lean_ctor_set(v___x_83_, 0, v___x_238_);
v___x_240_ = v___x_83_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v___x_238_);
lean_ctor_set(v_reuseFailAlloc_241_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_241_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_241_, 3, v_l_80_);
lean_ctor_set(v_reuseFailAlloc_241_, 4, v_impl_226_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
else
{
lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_305_; 
v_isSharedCheck_305_ = !lean_is_exclusive(v_impl_226_);
if (v_isSharedCheck_305_ == 0)
{
lean_object* v_unused_306_; lean_object* v_unused_307_; lean_object* v_unused_308_; lean_object* v_unused_309_; lean_object* v_unused_310_; 
v_unused_306_ = lean_ctor_get(v_impl_226_, 4);
lean_dec(v_unused_306_);
v_unused_307_ = lean_ctor_get(v_impl_226_, 3);
lean_dec(v_unused_307_);
v_unused_308_ = lean_ctor_get(v_impl_226_, 2);
lean_dec(v_unused_308_);
v_unused_309_ = lean_ctor_get(v_impl_226_, 1);
lean_dec(v_unused_309_);
v_unused_310_ = lean_ctor_get(v_impl_226_, 0);
lean_dec(v_unused_310_);
v___x_243_ = v_impl_226_;
v_isShared_244_ = v_isSharedCheck_305_;
goto v_resetjp_242_;
}
else
{
lean_dec(v_impl_226_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_305_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v_size_245_; lean_object* v_k_246_; lean_object* v_v_247_; lean_object* v_l_248_; lean_object* v_r_249_; lean_object* v_size_250_; lean_object* v___x_251_; lean_object* v___x_252_; uint8_t v___x_253_; 
v_size_245_ = lean_ctor_get(v_l_232_, 0);
v_k_246_ = lean_ctor_get(v_l_232_, 1);
v_v_247_ = lean_ctor_get(v_l_232_, 2);
v_l_248_ = lean_ctor_get(v_l_232_, 3);
v_r_249_ = lean_ctor_get(v_l_232_, 4);
v_size_250_ = lean_ctor_get(v_r_233_, 0);
v___x_251_ = lean_unsigned_to_nat(2u);
v___x_252_ = lean_nat_mul(v___x_251_, v_size_250_);
v___x_253_ = lean_nat_dec_lt(v_size_245_, v___x_252_);
lean_dec(v___x_252_);
if (v___x_253_ == 0)
{
lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_281_; 
lean_inc(v_r_249_);
lean_inc(v_l_248_);
lean_inc(v_v_247_);
lean_inc(v_k_246_);
v_isSharedCheck_281_ = !lean_is_exclusive(v_l_232_);
if (v_isSharedCheck_281_ == 0)
{
lean_object* v_unused_282_; lean_object* v_unused_283_; lean_object* v_unused_284_; lean_object* v_unused_285_; lean_object* v_unused_286_; 
v_unused_282_ = lean_ctor_get(v_l_232_, 4);
lean_dec(v_unused_282_);
v_unused_283_ = lean_ctor_get(v_l_232_, 3);
lean_dec(v_unused_283_);
v_unused_284_ = lean_ctor_get(v_l_232_, 2);
lean_dec(v_unused_284_);
v_unused_285_ = lean_ctor_get(v_l_232_, 1);
lean_dec(v_unused_285_);
v_unused_286_ = lean_ctor_get(v_l_232_, 0);
lean_dec(v_unused_286_);
v___x_255_ = v_l_232_;
v_isShared_256_ = v_isSharedCheck_281_;
goto v_resetjp_254_;
}
else
{
lean_dec(v_l_232_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_281_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___y_260_; lean_object* v___y_261_; lean_object* v___y_262_; lean_object* v___y_271_; 
v___x_257_ = lean_nat_add(v___x_227_, v_size_228_);
v___x_258_ = lean_nat_add(v___x_257_, v_size_229_);
lean_dec(v_size_229_);
if (lean_obj_tag(v_l_248_) == 0)
{
lean_object* v_size_279_; 
v_size_279_ = lean_ctor_get(v_l_248_, 0);
lean_inc(v_size_279_);
v___y_271_ = v_size_279_;
goto v___jp_270_;
}
else
{
lean_object* v___x_280_; 
v___x_280_ = lean_unsigned_to_nat(0u);
v___y_271_ = v___x_280_;
goto v___jp_270_;
}
v___jp_259_:
{
lean_object* v___x_263_; lean_object* v___x_265_; 
v___x_263_ = lean_nat_add(v___y_261_, v___y_262_);
lean_dec(v___y_262_);
lean_dec(v___y_261_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 4, v_r_233_);
lean_ctor_set(v___x_255_, 3, v_r_249_);
lean_ctor_set(v___x_255_, 2, v_v_231_);
lean_ctor_set(v___x_255_, 1, v_k_230_);
lean_ctor_set(v___x_255_, 0, v___x_263_);
v___x_265_ = v___x_255_;
goto v_reusejp_264_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v___x_263_);
lean_ctor_set(v_reuseFailAlloc_269_, 1, v_k_230_);
lean_ctor_set(v_reuseFailAlloc_269_, 2, v_v_231_);
lean_ctor_set(v_reuseFailAlloc_269_, 3, v_r_249_);
lean_ctor_set(v_reuseFailAlloc_269_, 4, v_r_233_);
v___x_265_ = v_reuseFailAlloc_269_;
goto v_reusejp_264_;
}
v_reusejp_264_:
{
lean_object* v___x_267_; 
if (v_isShared_244_ == 0)
{
lean_ctor_set(v___x_243_, 4, v___x_265_);
lean_ctor_set(v___x_243_, 3, v___y_260_);
lean_ctor_set(v___x_243_, 2, v_v_247_);
lean_ctor_set(v___x_243_, 1, v_k_246_);
lean_ctor_set(v___x_243_, 0, v___x_258_);
v___x_267_ = v___x_243_;
goto v_reusejp_266_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v___x_258_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v_k_246_);
lean_ctor_set(v_reuseFailAlloc_268_, 2, v_v_247_);
lean_ctor_set(v_reuseFailAlloc_268_, 3, v___y_260_);
lean_ctor_set(v_reuseFailAlloc_268_, 4, v___x_265_);
v___x_267_ = v_reuseFailAlloc_268_;
goto v_reusejp_266_;
}
v_reusejp_266_:
{
return v___x_267_;
}
}
}
v___jp_270_:
{
lean_object* v___x_272_; lean_object* v___x_274_; 
v___x_272_ = lean_nat_add(v___x_257_, v___y_271_);
lean_dec(v___y_271_);
lean_dec(v___x_257_);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v_l_248_);
lean_ctor_set(v___x_83_, 0, v___x_272_);
v___x_274_ = v___x_83_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_272_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_278_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_278_, 3, v_l_80_);
lean_ctor_set(v_reuseFailAlloc_278_, 4, v_l_248_);
v___x_274_ = v_reuseFailAlloc_278_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
lean_object* v___x_275_; 
v___x_275_ = lean_nat_add(v___x_227_, v_size_250_);
if (lean_obj_tag(v_r_249_) == 0)
{
lean_object* v_size_276_; 
v_size_276_ = lean_ctor_get(v_r_249_, 0);
lean_inc(v_size_276_);
v___y_260_ = v___x_274_;
v___y_261_ = v___x_275_;
v___y_262_ = v_size_276_;
goto v___jp_259_;
}
else
{
lean_object* v___x_277_; 
v___x_277_ = lean_unsigned_to_nat(0u);
v___y_260_ = v___x_274_;
v___y_261_ = v___x_275_;
v___y_262_ = v___x_277_;
goto v___jp_259_;
}
}
}
}
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_291_; 
lean_del_object(v___x_83_);
v___x_287_ = lean_nat_add(v___x_227_, v_size_228_);
v___x_288_ = lean_nat_add(v___x_287_, v_size_229_);
lean_dec(v_size_229_);
v___x_289_ = lean_nat_add(v___x_287_, v_size_245_);
lean_dec(v___x_287_);
lean_inc_ref(v_l_80_);
if (v_isShared_244_ == 0)
{
lean_ctor_set(v___x_243_, 4, v_l_232_);
lean_ctor_set(v___x_243_, 3, v_l_80_);
lean_ctor_set(v___x_243_, 2, v_v_79_);
lean_ctor_set(v___x_243_, 1, v_k_78_);
lean_ctor_set(v___x_243_, 0, v___x_289_);
v___x_291_ = v___x_243_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v___x_289_);
lean_ctor_set(v_reuseFailAlloc_304_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_304_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_304_, 3, v_l_80_);
lean_ctor_set(v_reuseFailAlloc_304_, 4, v_l_232_);
v___x_291_ = v_reuseFailAlloc_304_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_298_; 
v_isSharedCheck_298_ = !lean_is_exclusive(v_l_80_);
if (v_isSharedCheck_298_ == 0)
{
lean_object* v_unused_299_; lean_object* v_unused_300_; lean_object* v_unused_301_; lean_object* v_unused_302_; lean_object* v_unused_303_; 
v_unused_299_ = lean_ctor_get(v_l_80_, 4);
lean_dec(v_unused_299_);
v_unused_300_ = lean_ctor_get(v_l_80_, 3);
lean_dec(v_unused_300_);
v_unused_301_ = lean_ctor_get(v_l_80_, 2);
lean_dec(v_unused_301_);
v_unused_302_ = lean_ctor_get(v_l_80_, 1);
lean_dec(v_unused_302_);
v_unused_303_ = lean_ctor_get(v_l_80_, 0);
lean_dec(v_unused_303_);
v___x_293_ = v_l_80_;
v_isShared_294_ = v_isSharedCheck_298_;
goto v_resetjp_292_;
}
else
{
lean_dec(v_l_80_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_298_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___x_296_; 
if (v_isShared_294_ == 0)
{
lean_ctor_set(v___x_293_, 4, v_r_233_);
lean_ctor_set(v___x_293_, 3, v___x_291_);
lean_ctor_set(v___x_293_, 2, v_v_231_);
lean_ctor_set(v___x_293_, 1, v_k_230_);
lean_ctor_set(v___x_293_, 0, v___x_288_);
v___x_296_ = v___x_293_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v___x_288_);
lean_ctor_set(v_reuseFailAlloc_297_, 1, v_k_230_);
lean_ctor_set(v_reuseFailAlloc_297_, 2, v_v_231_);
lean_ctor_set(v_reuseFailAlloc_297_, 3, v___x_291_);
lean_ctor_set(v_reuseFailAlloc_297_, 4, v_r_233_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_311_; 
v_l_311_ = lean_ctor_get(v_impl_226_, 3);
lean_inc(v_l_311_);
if (lean_obj_tag(v_l_311_) == 0)
{
lean_object* v_r_312_; lean_object* v_k_313_; lean_object* v_v_314_; lean_object* v___x_316_; uint8_t v_isShared_317_; uint8_t v_isSharedCheck_337_; 
v_r_312_ = lean_ctor_get(v_impl_226_, 4);
v_k_313_ = lean_ctor_get(v_impl_226_, 1);
v_v_314_ = lean_ctor_get(v_impl_226_, 2);
v_isSharedCheck_337_ = !lean_is_exclusive(v_impl_226_);
if (v_isSharedCheck_337_ == 0)
{
lean_object* v_unused_338_; lean_object* v_unused_339_; 
v_unused_338_ = lean_ctor_get(v_impl_226_, 3);
lean_dec(v_unused_338_);
v_unused_339_ = lean_ctor_get(v_impl_226_, 0);
lean_dec(v_unused_339_);
v___x_316_ = v_impl_226_;
v_isShared_317_ = v_isSharedCheck_337_;
goto v_resetjp_315_;
}
else
{
lean_inc(v_r_312_);
lean_inc(v_v_314_);
lean_inc(v_k_313_);
lean_dec(v_impl_226_);
v___x_316_ = lean_box(0);
v_isShared_317_ = v_isSharedCheck_337_;
goto v_resetjp_315_;
}
v_resetjp_315_:
{
lean_object* v_k_318_; lean_object* v_v_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_333_; 
v_k_318_ = lean_ctor_get(v_l_311_, 1);
v_v_319_ = lean_ctor_get(v_l_311_, 2);
v_isSharedCheck_333_ = !lean_is_exclusive(v_l_311_);
if (v_isSharedCheck_333_ == 0)
{
lean_object* v_unused_334_; lean_object* v_unused_335_; lean_object* v_unused_336_; 
v_unused_334_ = lean_ctor_get(v_l_311_, 4);
lean_dec(v_unused_334_);
v_unused_335_ = lean_ctor_get(v_l_311_, 3);
lean_dec(v_unused_335_);
v_unused_336_ = lean_ctor_get(v_l_311_, 0);
lean_dec(v_unused_336_);
v___x_321_ = v_l_311_;
v_isShared_322_ = v_isSharedCheck_333_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_v_319_);
lean_inc(v_k_318_);
lean_dec(v_l_311_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_333_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_323_; lean_object* v___x_325_; 
v___x_323_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_312_, 2);
if (v_isShared_322_ == 0)
{
lean_ctor_set(v___x_321_, 4, v_r_312_);
lean_ctor_set(v___x_321_, 3, v_r_312_);
lean_ctor_set(v___x_321_, 2, v_v_79_);
lean_ctor_set(v___x_321_, 1, v_k_78_);
lean_ctor_set(v___x_321_, 0, v___x_227_);
v___x_325_ = v___x_321_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_332_; 
v_reuseFailAlloc_332_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_332_, 0, v___x_227_);
lean_ctor_set(v_reuseFailAlloc_332_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_332_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_332_, 3, v_r_312_);
lean_ctor_set(v_reuseFailAlloc_332_, 4, v_r_312_);
v___x_325_ = v_reuseFailAlloc_332_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
lean_object* v___x_327_; 
lean_inc(v_r_312_);
if (v_isShared_317_ == 0)
{
lean_ctor_set(v___x_316_, 3, v_r_312_);
lean_ctor_set(v___x_316_, 0, v___x_227_);
v___x_327_ = v___x_316_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___x_227_);
lean_ctor_set(v_reuseFailAlloc_331_, 1, v_k_313_);
lean_ctor_set(v_reuseFailAlloc_331_, 2, v_v_314_);
lean_ctor_set(v_reuseFailAlloc_331_, 3, v_r_312_);
lean_ctor_set(v_reuseFailAlloc_331_, 4, v_r_312_);
v___x_327_ = v_reuseFailAlloc_331_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
lean_object* v___x_329_; 
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v___x_327_);
lean_ctor_set(v___x_83_, 3, v___x_325_);
lean_ctor_set(v___x_83_, 2, v_v_319_);
lean_ctor_set(v___x_83_, 1, v_k_318_);
lean_ctor_set(v___x_83_, 0, v___x_323_);
v___x_329_ = v___x_83_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v___x_323_);
lean_ctor_set(v_reuseFailAlloc_330_, 1, v_k_318_);
lean_ctor_set(v_reuseFailAlloc_330_, 2, v_v_319_);
lean_ctor_set(v_reuseFailAlloc_330_, 3, v___x_325_);
lean_ctor_set(v_reuseFailAlloc_330_, 4, v___x_327_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
}
}
}
}
else
{
lean_object* v_r_340_; 
v_r_340_ = lean_ctor_get(v_impl_226_, 4);
lean_inc(v_r_340_);
if (lean_obj_tag(v_r_340_) == 0)
{
lean_object* v_k_341_; lean_object* v_v_342_; lean_object* v___x_344_; uint8_t v_isShared_345_; uint8_t v_isSharedCheck_353_; 
v_k_341_ = lean_ctor_get(v_impl_226_, 1);
v_v_342_ = lean_ctor_get(v_impl_226_, 2);
v_isSharedCheck_353_ = !lean_is_exclusive(v_impl_226_);
if (v_isSharedCheck_353_ == 0)
{
lean_object* v_unused_354_; lean_object* v_unused_355_; lean_object* v_unused_356_; 
v_unused_354_ = lean_ctor_get(v_impl_226_, 4);
lean_dec(v_unused_354_);
v_unused_355_ = lean_ctor_get(v_impl_226_, 3);
lean_dec(v_unused_355_);
v_unused_356_ = lean_ctor_get(v_impl_226_, 0);
lean_dec(v_unused_356_);
v___x_344_ = v_impl_226_;
v_isShared_345_ = v_isSharedCheck_353_;
goto v_resetjp_343_;
}
else
{
lean_inc(v_v_342_);
lean_inc(v_k_341_);
lean_dec(v_impl_226_);
v___x_344_ = lean_box(0);
v_isShared_345_ = v_isSharedCheck_353_;
goto v_resetjp_343_;
}
v_resetjp_343_:
{
lean_object* v___x_346_; lean_object* v___x_348_; 
v___x_346_ = lean_unsigned_to_nat(3u);
if (v_isShared_345_ == 0)
{
lean_ctor_set(v___x_344_, 4, v_l_311_);
lean_ctor_set(v___x_344_, 2, v_v_79_);
lean_ctor_set(v___x_344_, 1, v_k_78_);
lean_ctor_set(v___x_344_, 0, v___x_227_);
v___x_348_ = v___x_344_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v___x_227_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_352_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_352_, 3, v_l_311_);
lean_ctor_set(v_reuseFailAlloc_352_, 4, v_l_311_);
v___x_348_ = v_reuseFailAlloc_352_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
lean_object* v___x_350_; 
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v_r_340_);
lean_ctor_set(v___x_83_, 3, v___x_348_);
lean_ctor_set(v___x_83_, 2, v_v_342_);
lean_ctor_set(v___x_83_, 1, v_k_341_);
lean_ctor_set(v___x_83_, 0, v___x_346_);
v___x_350_ = v___x_83_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v___x_346_);
lean_ctor_set(v_reuseFailAlloc_351_, 1, v_k_341_);
lean_ctor_set(v_reuseFailAlloc_351_, 2, v_v_342_);
lean_ctor_set(v_reuseFailAlloc_351_, 3, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_351_, 4, v_r_340_);
v___x_350_ = v_reuseFailAlloc_351_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
return v___x_350_;
}
}
}
}
else
{
lean_object* v___x_357_; lean_object* v___x_359_; 
v___x_357_ = lean_unsigned_to_nat(2u);
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 4, v_impl_226_);
lean_ctor_set(v___x_83_, 3, v_r_340_);
lean_ctor_set(v___x_83_, 0, v___x_357_);
v___x_359_ = v___x_83_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_357_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v_k_78_);
lean_ctor_set(v_reuseFailAlloc_360_, 2, v_v_79_);
lean_ctor_set(v_reuseFailAlloc_360_, 3, v_r_340_);
lean_ctor_set(v_reuseFailAlloc_360_, 4, v_impl_226_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
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
lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_362_ = lean_unsigned_to_nat(1u);
v___x_363_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v_k_74_);
lean_ctor_set(v___x_363_, 2, v_v_75_);
lean_ctor_set(v___x_363_, 3, v_t_76_);
lean_ctor_set(v___x_363_, 4, v_t_76_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg(lean_object* v___y_364_, lean_object* v_init_365_, lean_object* v_x_366_){
_start:
{
lean_object* v_d_369_; 
if (lean_obj_tag(v_x_366_) == 0)
{
lean_object* v_k_372_; lean_object* v_v_373_; lean_object* v_l_374_; lean_object* v_r_375_; lean_object* v___y_377_; lean_object* v___x_382_; lean_object* v_a_383_; 
v_k_372_ = lean_ctor_get(v_x_366_, 1);
lean_inc(v_k_372_);
v_v_373_ = lean_ctor_get(v_x_366_, 2);
lean_inc(v_v_373_);
v_l_374_ = lean_ctor_get(v_x_366_, 3);
lean_inc(v_l_374_);
v_r_375_ = lean_ctor_get(v_x_366_, 4);
lean_inc(v_r_375_);
lean_dec_ref_known(v_x_366_, 5);
v___x_382_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg(v___y_364_, v_init_365_, v_l_374_);
v_a_383_ = lean_ctor_get(v___x_382_, 0);
lean_inc(v_a_383_);
if (lean_obj_tag(v_a_383_) == 0)
{
lean_object* v_a_384_; 
lean_dec_ref(v___x_382_);
lean_dec(v_r_375_);
lean_dec(v_v_373_);
lean_dec(v_k_372_);
v_a_384_ = lean_ctor_get(v_a_383_, 0);
lean_inc(v_a_384_);
lean_dec_ref_known(v_a_383_, 1);
v_d_369_ = v_a_384_;
goto v___jp_368_;
}
else
{
lean_object* v_a_385_; uint8_t v___x_386_; lean_object* v___x_387_; 
v_a_385_ = lean_ctor_get(v_a_383_, 0);
lean_inc(v_a_385_);
lean_dec_ref_known(v_a_383_, 1);
v___x_386_ = 0;
lean_inc(v_k_372_);
v___x_387_ = l_Lean_Name_toString(v_k_372_, v___x_386_);
if (lean_obj_tag(v___y_364_) == 1)
{
lean_object* v_val_392_; lean_object* v___x_393_; lean_object* v___x_394_; uint8_t v___x_395_; 
v_val_392_ = lean_ctor_get(v___y_364_, 0);
v___x_393_ = lean_string_utf8_byte_size(v___x_387_);
v___x_394_ = lean_string_utf8_byte_size(v_val_392_);
v___x_395_ = lean_nat_dec_le(v___x_394_, v___x_393_);
if (v___x_395_ == 0)
{
lean_dec_ref(v___x_387_);
lean_dec(v_a_385_);
lean_dec(v_v_373_);
lean_dec(v_k_372_);
v___y_377_ = v___x_382_;
goto v___jp_376_;
}
else
{
lean_object* v___x_396_; uint8_t v___x_397_; 
v___x_396_ = lean_unsigned_to_nat(0u);
v___x_397_ = lean_string_memcmp(v___x_387_, v_val_392_, v___x_396_, v___x_396_, v___x_394_);
if (v___x_397_ == 0)
{
lean_dec_ref(v___x_387_);
lean_dec(v_a_385_);
lean_dec(v_v_373_);
lean_dec(v_k_372_);
v___y_377_ = v___x_382_;
goto v___jp_376_;
}
else
{
lean_dec_ref(v___x_382_);
goto v___jp_388_;
}
}
}
else
{
lean_dec_ref(v___x_382_);
goto v___jp_388_;
}
v___jp_388_:
{
lean_object* v___x_389_; lean_object* v___x_390_; 
v___x_389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_389_, 0, v_k_372_);
lean_ctor_set(v___x_389_, 1, v_v_373_);
v___x_390_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v___x_387_, v___x_389_, v_a_385_);
v_init_365_ = v___x_390_;
v_x_366_ = v_r_375_;
goto _start;
}
}
v___jp_376_:
{
lean_object* v_a_378_; 
v_a_378_ = lean_ctor_get(v___y_377_, 0);
lean_inc(v_a_378_);
lean_dec_ref(v___y_377_);
if (lean_obj_tag(v_a_378_) == 0)
{
lean_object* v_a_379_; 
lean_dec(v_r_375_);
v_a_379_ = lean_ctor_get(v_a_378_, 0);
lean_inc(v_a_379_);
lean_dec_ref_known(v_a_378_, 1);
v_d_369_ = v_a_379_;
goto v___jp_368_;
}
else
{
lean_object* v_a_380_; 
v_a_380_ = lean_ctor_get(v_a_378_, 0);
lean_inc(v_a_380_);
lean_dec_ref_known(v_a_378_, 1);
v_init_365_ = v_a_380_;
v_x_366_ = v_r_375_;
goto _start;
}
}
}
else
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_398_, 0, v_init_365_);
v___x_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
return v___x_399_;
}
v___jp_368_:
{
lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_370_, 0, v_d_369_);
v___x_371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
return v___x_371_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg___boxed(lean_object* v___y_400_, lean_object* v_init_401_, lean_object* v_x_402_, lean_object* v___y_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg(v___y_400_, v_init_401_, v_x_402_);
lean_dec(v___y_400_);
return v_res_404_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0(uint8_t v___y_406_, uint8_t v_suppressElabErrors_407_, lean_object* v_x_408_){
_start:
{
if (lean_obj_tag(v_x_408_) == 1)
{
lean_object* v_pre_409_; 
v_pre_409_ = lean_ctor_get(v_x_408_, 0);
if (lean_obj_tag(v_pre_409_) == 0)
{
lean_object* v_str_410_; lean_object* v___x_411_; uint8_t v___x_412_; 
v_str_410_ = lean_ctor_get(v_x_408_, 1);
v___x_411_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___closed__0));
v___x_412_ = lean_string_dec_eq(v_str_410_, v___x_411_);
if (v___x_412_ == 0)
{
return v___y_406_;
}
else
{
return v_suppressElabErrors_407_;
}
}
else
{
return v___y_406_;
}
}
else
{
return v___y_406_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___boxed(lean_object* v___y_413_, lean_object* v_suppressElabErrors_414_, lean_object* v_x_415_){
_start:
{
uint8_t v___y_9524__boxed_416_; uint8_t v_suppressElabErrors_boxed_417_; uint8_t v_res_418_; lean_object* v_r_419_; 
v___y_9524__boxed_416_ = lean_unbox(v___y_413_);
v_suppressElabErrors_boxed_417_ = lean_unbox(v_suppressElabErrors_414_);
v_res_418_ = lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0(v___y_9524__boxed_416_, v_suppressElabErrors_boxed_417_, v_x_415_);
lean_dec(v_x_415_);
v_r_419_ = lean_box(v_res_418_);
return v_r_419_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_420_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__0);
v___x_422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
return v___x_422_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_423_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1);
v___x_424_ = lean_unsigned_to_nat(0u);
v___x_425_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_425_, 0, v___x_424_);
lean_ctor_set(v___x_425_, 1, v___x_424_);
lean_ctor_set(v___x_425_, 2, v___x_424_);
lean_ctor_set(v___x_425_, 3, v___x_424_);
lean_ctor_set(v___x_425_, 4, v___x_423_);
lean_ctor_set(v___x_425_, 5, v___x_423_);
lean_ctor_set(v___x_425_, 6, v___x_423_);
lean_ctor_set(v___x_425_, 7, v___x_423_);
lean_ctor_set(v___x_425_, 8, v___x_423_);
lean_ctor_set(v___x_425_, 9, v___x_423_);
return v___x_425_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_426_ = lean_unsigned_to_nat(32u);
v___x_427_ = lean_mk_empty_array_with_capacity(v___x_426_);
v___x_428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
return v___x_428_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_429_ = ((size_t)5ULL);
v___x_430_ = lean_unsigned_to_nat(0u);
v___x_431_ = lean_unsigned_to_nat(32u);
v___x_432_ = lean_mk_empty_array_with_capacity(v___x_431_);
v___x_433_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__3);
v___x_434_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_434_, 0, v___x_433_);
lean_ctor_set(v___x_434_, 1, v___x_432_);
lean_ctor_set(v___x_434_, 2, v___x_430_);
lean_ctor_set(v___x_434_, 3, v___x_430_);
lean_ctor_set_usize(v___x_434_, 4, v___x_429_);
return v___x_434_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_435_ = lean_box(1);
v___x_436_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__4);
v___x_437_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__1);
v___x_438_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_438_, 0, v___x_437_);
lean_ctor_set(v___x_438_, 1, v___x_436_);
lean_ctor_set(v___x_438_, 2, v___x_435_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg(lean_object* v_msgData_439_, lean_object* v___y_440_){
_start:
{
lean_object* v___x_442_; lean_object* v_env_443_; lean_object* v___x_444_; lean_object* v_scopes_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v_opts_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; 
v___x_442_ = lean_st_ref_get(v___y_440_);
v_env_443_ = lean_ctor_get(v___x_442_, 0);
lean_inc_ref(v_env_443_);
lean_dec(v___x_442_);
v___x_444_ = lean_st_ref_get(v___y_440_);
v_scopes_445_ = lean_ctor_get(v___x_444_, 2);
lean_inc(v_scopes_445_);
lean_dec(v___x_444_);
v___x_446_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_447_ = l_List_head_x21___redArg(v___x_446_, v_scopes_445_);
lean_dec(v_scopes_445_);
v_opts_448_ = lean_ctor_get(v___x_447_, 1);
lean_inc_ref(v_opts_448_);
lean_dec(v___x_447_);
v___x_449_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__2);
v___x_450_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___closed__5);
v___x_451_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_451_, 0, v_env_443_);
lean_ctor_set(v___x_451_, 1, v___x_449_);
lean_ctor_set(v___x_451_, 2, v___x_450_);
lean_ctor_set(v___x_451_, 3, v_opts_448_);
v___x_452_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_451_);
lean_ctor_set(v___x_452_, 1, v_msgData_439_);
v___x_453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_453_, 0, v___x_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg(v_msgData_454_, v___y_455_);
lean_dec(v___y_455_);
return v_res_457_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6(lean_object* v_opts_458_, lean_object* v_opt_459_){
_start:
{
lean_object* v_name_460_; lean_object* v_defValue_461_; lean_object* v_map_462_; lean_object* v___x_463_; 
v_name_460_ = lean_ctor_get(v_opt_459_, 0);
v_defValue_461_ = lean_ctor_get(v_opt_459_, 1);
v_map_462_ = lean_ctor_get(v_opts_458_, 0);
v___x_463_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_462_, v_name_460_);
if (lean_obj_tag(v___x_463_) == 0)
{
uint8_t v___x_464_; 
v___x_464_ = lean_unbox(v_defValue_461_);
return v___x_464_;
}
else
{
lean_object* v_val_465_; 
v_val_465_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_val_465_);
lean_dec_ref_known(v___x_463_, 1);
if (lean_obj_tag(v_val_465_) == 1)
{
uint8_t v_v_466_; 
v_v_466_ = lean_ctor_get_uint8(v_val_465_, 0);
lean_dec_ref_known(v_val_465_, 0);
return v_v_466_;
}
else
{
uint8_t v___x_467_; 
lean_dec(v_val_465_);
v___x_467_ = lean_unbox(v_defValue_461_);
return v___x_467_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6___boxed(lean_object* v_opts_468_, lean_object* v_opt_469_){
_start:
{
uint8_t v_res_470_; lean_object* v_r_471_; 
v_res_470_ = lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6(v_opts_468_, v_opt_469_);
lean_dec_ref(v_opt_469_);
lean_dec_ref(v_opts_468_);
v_r_471_ = lean_box(v_res_470_);
return v_r_471_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1(lean_object* v_ref_473_, lean_object* v_msgData_474_, uint8_t v_severity_475_, uint8_t v_isSilent_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
uint8_t v___y_481_; lean_object* v___y_482_; lean_object* v___y_483_; lean_object* v___y_484_; uint8_t v___y_485_; lean_object* v___y_486_; lean_object* v___y_487_; lean_object* v___y_488_; uint8_t v___y_545_; uint8_t v___y_546_; lean_object* v___y_547_; uint8_t v___y_548_; lean_object* v___y_549_; uint8_t v___y_573_; uint8_t v___y_574_; lean_object* v___y_575_; uint8_t v___y_576_; lean_object* v___y_577_; uint8_t v___y_581_; uint8_t v___y_582_; uint8_t v___y_583_; uint8_t v___x_598_; uint8_t v___y_600_; uint8_t v___y_601_; uint8_t v___y_602_; uint8_t v___y_604_; uint8_t v___x_616_; 
v___x_598_ = 2;
v___x_616_ = l_Lean_instBEqMessageSeverity_beq(v_severity_475_, v___x_598_);
if (v___x_616_ == 0)
{
v___y_604_ = v___x_616_;
goto v___jp_603_;
}
else
{
uint8_t v___x_617_; 
lean_inc_ref(v_msgData_474_);
v___x_617_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_474_);
v___y_604_ = v___x_617_;
goto v___jp_603_;
}
v___jp_480_:
{
lean_object* v___x_489_; 
v___x_489_ = l_Lean_Elab_Command_getScope___redArg(v___y_488_);
if (lean_obj_tag(v___x_489_) == 0)
{
lean_object* v_a_490_; lean_object* v___x_491_; 
v_a_490_ = lean_ctor_get(v___x_489_, 0);
lean_inc(v_a_490_);
lean_dec_ref_known(v___x_489_, 1);
v___x_491_ = l_Lean_Elab_Command_getScope___redArg(v___y_488_);
if (lean_obj_tag(v___x_491_) == 0)
{
lean_object* v_a_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_527_; 
v_a_492_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_527_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_527_ == 0)
{
v___x_494_ = v___x_491_;
v_isShared_495_ = v_isSharedCheck_527_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_a_492_);
lean_dec(v___x_491_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_527_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_496_; lean_object* v_currNamespace_497_; lean_object* v_openDecls_498_; lean_object* v_env_499_; lean_object* v_messages_500_; lean_object* v_scopes_501_; lean_object* v_usedQuotCtxts_502_; lean_object* v_nextMacroScope_503_; lean_object* v_maxRecDepth_504_; lean_object* v_ngen_505_; lean_object* v_auxDeclNGen_506_; lean_object* v_infoState_507_; lean_object* v_traceState_508_; lean_object* v_snapshotTasks_509_; lean_object* v_prevLinterStates_510_; lean_object* v___x_512_; uint8_t v_isShared_513_; uint8_t v_isSharedCheck_526_; 
v___x_496_ = lean_st_ref_take(v___y_488_);
v_currNamespace_497_ = lean_ctor_get(v_a_490_, 2);
lean_inc(v_currNamespace_497_);
lean_dec(v_a_490_);
v_openDecls_498_ = lean_ctor_get(v_a_492_, 3);
lean_inc(v_openDecls_498_);
lean_dec(v_a_492_);
v_env_499_ = lean_ctor_get(v___x_496_, 0);
v_messages_500_ = lean_ctor_get(v___x_496_, 1);
v_scopes_501_ = lean_ctor_get(v___x_496_, 2);
v_usedQuotCtxts_502_ = lean_ctor_get(v___x_496_, 3);
v_nextMacroScope_503_ = lean_ctor_get(v___x_496_, 4);
v_maxRecDepth_504_ = lean_ctor_get(v___x_496_, 5);
v_ngen_505_ = lean_ctor_get(v___x_496_, 6);
v_auxDeclNGen_506_ = lean_ctor_get(v___x_496_, 7);
v_infoState_507_ = lean_ctor_get(v___x_496_, 8);
v_traceState_508_ = lean_ctor_get(v___x_496_, 9);
v_snapshotTasks_509_ = lean_ctor_get(v___x_496_, 10);
v_prevLinterStates_510_ = lean_ctor_get(v___x_496_, 11);
v_isSharedCheck_526_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_526_ == 0)
{
v___x_512_ = v___x_496_;
v_isShared_513_ = v_isSharedCheck_526_;
goto v_resetjp_511_;
}
else
{
lean_inc(v_prevLinterStates_510_);
lean_inc(v_snapshotTasks_509_);
lean_inc(v_traceState_508_);
lean_inc(v_infoState_507_);
lean_inc(v_auxDeclNGen_506_);
lean_inc(v_ngen_505_);
lean_inc(v_maxRecDepth_504_);
lean_inc(v_nextMacroScope_503_);
lean_inc(v_usedQuotCtxts_502_);
lean_inc(v_scopes_501_);
lean_inc(v_messages_500_);
lean_inc(v_env_499_);
lean_dec(v___x_496_);
v___x_512_ = lean_box(0);
v_isShared_513_ = v_isSharedCheck_526_;
goto v_resetjp_511_;
}
v_resetjp_511_:
{
lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_519_; 
v___x_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_514_, 0, v_currNamespace_497_);
lean_ctor_set(v___x_514_, 1, v_openDecls_498_);
v___x_515_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_515_, 0, v___x_514_);
lean_ctor_set(v___x_515_, 1, v___y_484_);
lean_inc_ref(v___y_487_);
lean_inc_ref(v___y_482_);
v___x_516_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_516_, 0, v___y_482_);
lean_ctor_set(v___x_516_, 1, v___y_486_);
lean_ctor_set(v___x_516_, 2, v___y_483_);
lean_ctor_set(v___x_516_, 3, v___y_487_);
lean_ctor_set(v___x_516_, 4, v___x_515_);
lean_ctor_set_uint8(v___x_516_, sizeof(void*)*5, v___y_485_);
lean_ctor_set_uint8(v___x_516_, sizeof(void*)*5 + 1, v___y_481_);
lean_ctor_set_uint8(v___x_516_, sizeof(void*)*5 + 2, v_isSilent_476_);
v___x_517_ = l_Lean_MessageLog_add(v___x_516_, v_messages_500_);
if (v_isShared_513_ == 0)
{
lean_ctor_set(v___x_512_, 1, v___x_517_);
v___x_519_ = v___x_512_;
goto v_reusejp_518_;
}
else
{
lean_object* v_reuseFailAlloc_525_; 
v_reuseFailAlloc_525_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_525_, 0, v_env_499_);
lean_ctor_set(v_reuseFailAlloc_525_, 1, v___x_517_);
lean_ctor_set(v_reuseFailAlloc_525_, 2, v_scopes_501_);
lean_ctor_set(v_reuseFailAlloc_525_, 3, v_usedQuotCtxts_502_);
lean_ctor_set(v_reuseFailAlloc_525_, 4, v_nextMacroScope_503_);
lean_ctor_set(v_reuseFailAlloc_525_, 5, v_maxRecDepth_504_);
lean_ctor_set(v_reuseFailAlloc_525_, 6, v_ngen_505_);
lean_ctor_set(v_reuseFailAlloc_525_, 7, v_auxDeclNGen_506_);
lean_ctor_set(v_reuseFailAlloc_525_, 8, v_infoState_507_);
lean_ctor_set(v_reuseFailAlloc_525_, 9, v_traceState_508_);
lean_ctor_set(v_reuseFailAlloc_525_, 10, v_snapshotTasks_509_);
lean_ctor_set(v_reuseFailAlloc_525_, 11, v_prevLinterStates_510_);
v___x_519_ = v_reuseFailAlloc_525_;
goto v_reusejp_518_;
}
v_reusejp_518_:
{
lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_523_; 
v___x_520_ = lean_st_ref_set(v___y_488_, v___x_519_);
v___x_521_ = lean_box(0);
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 0, v___x_521_);
v___x_523_ = v___x_494_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v___x_521_);
v___x_523_ = v_reuseFailAlloc_524_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
return v___x_523_;
}
}
}
}
}
else
{
lean_object* v_a_528_; lean_object* v___x_530_; uint8_t v_isShared_531_; uint8_t v_isSharedCheck_535_; 
lean_dec(v_a_490_);
lean_dec_ref(v___y_486_);
lean_dec_ref(v___y_484_);
lean_dec(v___y_483_);
v_a_528_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_535_ == 0)
{
v___x_530_ = v___x_491_;
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
else
{
lean_inc(v_a_528_);
lean_dec(v___x_491_);
v___x_530_ = lean_box(0);
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
v_resetjp_529_:
{
lean_object* v___x_533_; 
if (v_isShared_531_ == 0)
{
v___x_533_ = v___x_530_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v_a_528_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
}
}
else
{
lean_object* v_a_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_543_; 
lean_dec_ref(v___y_486_);
lean_dec_ref(v___y_484_);
lean_dec(v___y_483_);
v_a_536_ = lean_ctor_get(v___x_489_, 0);
v_isSharedCheck_543_ = !lean_is_exclusive(v___x_489_);
if (v_isSharedCheck_543_ == 0)
{
v___x_538_ = v___x_489_;
v_isShared_539_ = v_isSharedCheck_543_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_a_536_);
lean_dec(v___x_489_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_543_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_541_; 
if (v_isShared_539_ == 0)
{
v___x_541_ = v___x_538_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v_a_536_);
v___x_541_ = v_reuseFailAlloc_542_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
return v___x_541_;
}
}
}
}
v___jp_544_:
{
lean_object* v_fileName_550_; lean_object* v_fileMap_551_; uint8_t v_suppressElabErrors_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v_a_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_571_; 
v_fileName_550_ = lean_ctor_get(v___y_477_, 0);
v_fileMap_551_ = lean_ctor_get(v___y_477_, 1);
v_suppressElabErrors_552_ = lean_ctor_get_uint8(v___y_477_, sizeof(void*)*10);
v___x_553_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_474_);
v___x_554_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg(v___x_553_, v___y_478_);
v_a_555_ = lean_ctor_get(v___x_554_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_554_);
if (v_isSharedCheck_571_ == 0)
{
v___x_557_ = v___x_554_;
v_isShared_558_ = v_isSharedCheck_571_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_a_555_);
lean_dec(v___x_554_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_571_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; 
lean_inc_ref_n(v_fileMap_551_, 2);
v___x_559_ = l_Lean_FileMap_toPosition(v_fileMap_551_, v___y_547_);
lean_dec(v___y_547_);
v___x_560_ = l_Lean_FileMap_toPosition(v_fileMap_551_, v___y_549_);
lean_dec(v___y_549_);
v___x_561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_561_, 0, v___x_560_);
v___x_562_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___closed__0));
if (v_suppressElabErrors_552_ == 0)
{
lean_del_object(v___x_557_);
v___y_481_ = v___y_546_;
v___y_482_ = v_fileName_550_;
v___y_483_ = v___x_561_;
v___y_484_ = v_a_555_;
v___y_485_ = v___y_548_;
v___y_486_ = v___x_559_;
v___y_487_ = v___x_562_;
v___y_488_ = v___y_478_;
goto v___jp_480_;
}
else
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___f_565_; uint8_t v___x_566_; 
v___x_563_ = lean_box(v___y_545_);
v___x_564_ = lean_box(v_suppressElabErrors_552_);
v___f_565_ = lean_alloc_closure((void*)(lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___lam__0___boxed), 3, 2);
lean_closure_set(v___f_565_, 0, v___x_563_);
lean_closure_set(v___f_565_, 1, v___x_564_);
lean_inc(v_a_555_);
v___x_566_ = l_Lean_MessageData_hasTag(v___f_565_, v_a_555_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; lean_object* v___x_569_; 
lean_dec_ref_known(v___x_561_, 1);
lean_dec_ref(v___x_559_);
lean_dec(v_a_555_);
v___x_567_ = lean_box(0);
if (v_isShared_558_ == 0)
{
lean_ctor_set(v___x_557_, 0, v___x_567_);
v___x_569_ = v___x_557_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_567_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
else
{
lean_del_object(v___x_557_);
v___y_481_ = v___y_546_;
v___y_482_ = v_fileName_550_;
v___y_483_ = v___x_561_;
v___y_484_ = v_a_555_;
v___y_485_ = v___y_548_;
v___y_486_ = v___x_559_;
v___y_487_ = v___x_562_;
v___y_488_ = v___y_478_;
goto v___jp_480_;
}
}
}
}
v___jp_572_:
{
lean_object* v___x_578_; 
v___x_578_ = l_Lean_Syntax_getTailPos_x3f(v___y_575_, v___y_576_);
lean_dec(v___y_575_);
if (lean_obj_tag(v___x_578_) == 0)
{
lean_inc(v___y_577_);
v___y_545_ = v___y_573_;
v___y_546_ = v___y_574_;
v___y_547_ = v___y_577_;
v___y_548_ = v___y_576_;
v___y_549_ = v___y_577_;
goto v___jp_544_;
}
else
{
lean_object* v_val_579_; 
v_val_579_ = lean_ctor_get(v___x_578_, 0);
lean_inc(v_val_579_);
lean_dec_ref_known(v___x_578_, 1);
v___y_545_ = v___y_573_;
v___y_546_ = v___y_574_;
v___y_547_ = v___y_577_;
v___y_548_ = v___y_576_;
v___y_549_ = v_val_579_;
goto v___jp_544_;
}
}
v___jp_580_:
{
lean_object* v___x_584_; 
v___x_584_ = l_Lean_Elab_Command_getRef___redArg(v___y_477_);
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v_a_585_; lean_object* v_ref_586_; lean_object* v___x_587_; 
v_a_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_a_585_);
lean_dec_ref_known(v___x_584_, 1);
v_ref_586_ = l_Lean_replaceRef(v_ref_473_, v_a_585_);
lean_dec(v_a_585_);
v___x_587_ = l_Lean_Syntax_getPos_x3f(v_ref_586_, v___y_582_);
if (lean_obj_tag(v___x_587_) == 0)
{
lean_object* v___x_588_; 
v___x_588_ = lean_unsigned_to_nat(0u);
v___y_573_ = v___y_581_;
v___y_574_ = v___y_583_;
v___y_575_ = v_ref_586_;
v___y_576_ = v___y_582_;
v___y_577_ = v___x_588_;
goto v___jp_572_;
}
else
{
lean_object* v_val_589_; 
v_val_589_ = lean_ctor_get(v___x_587_, 0);
lean_inc(v_val_589_);
lean_dec_ref_known(v___x_587_, 1);
v___y_573_ = v___y_581_;
v___y_574_ = v___y_583_;
v___y_575_ = v_ref_586_;
v___y_576_ = v___y_582_;
v___y_577_ = v_val_589_;
goto v___jp_572_;
}
}
else
{
lean_object* v_a_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_597_; 
lean_dec_ref(v_msgData_474_);
v_a_590_ = lean_ctor_get(v___x_584_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_597_ == 0)
{
v___x_592_ = v___x_584_;
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_a_590_);
lean_dec(v___x_584_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
lean_object* v___x_595_; 
if (v_isShared_593_ == 0)
{
v___x_595_ = v___x_592_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_a_590_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
}
v___jp_599_:
{
if (v___y_602_ == 0)
{
v___y_581_ = v___y_600_;
v___y_582_ = v___y_601_;
v___y_583_ = v_severity_475_;
goto v___jp_580_;
}
else
{
v___y_581_ = v___y_600_;
v___y_582_ = v___y_601_;
v___y_583_ = v___x_598_;
goto v___jp_580_;
}
}
v___jp_603_:
{
if (v___y_604_ == 0)
{
lean_object* v___x_605_; lean_object* v_scopes_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v_opts_609_; uint8_t v___x_610_; uint8_t v___x_611_; 
v___x_605_ = lean_st_ref_get(v___y_478_);
v_scopes_606_ = lean_ctor_get(v___x_605_, 2);
lean_inc(v_scopes_606_);
lean_dec(v___x_605_);
v___x_607_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_608_ = l_List_head_x21___redArg(v___x_607_, v_scopes_606_);
lean_dec(v_scopes_606_);
v_opts_609_ = lean_ctor_get(v___x_608_, 1);
lean_inc_ref(v_opts_609_);
lean_dec(v___x_608_);
v___x_610_ = 1;
v___x_611_ = l_Lean_instBEqMessageSeverity_beq(v_severity_475_, v___x_610_);
if (v___x_611_ == 0)
{
lean_dec_ref(v_opts_609_);
v___y_600_ = v___y_604_;
v___y_601_ = v___y_604_;
v___y_602_ = v___x_611_;
goto v___jp_599_;
}
else
{
lean_object* v___x_612_; uint8_t v___x_613_; 
v___x_612_ = l_Lean_warningAsError;
v___x_613_ = lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6(v_opts_609_, v___x_612_);
lean_dec_ref(v_opts_609_);
v___y_600_ = v___y_604_;
v___y_601_ = v___y_604_;
v___y_602_ = v___x_613_;
goto v___jp_599_;
}
}
else
{
lean_object* v___x_614_; lean_object* v___x_615_; 
lean_dec_ref(v_msgData_474_);
v___x_614_ = lean_box(0);
v___x_615_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_615_, 0, v___x_614_);
return v___x_615_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___boxed(lean_object* v_ref_618_, lean_object* v_msgData_619_, lean_object* v_severity_620_, lean_object* v_isSilent_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_){
_start:
{
uint8_t v_severity_boxed_625_; uint8_t v_isSilent_boxed_626_; lean_object* v_res_627_; 
v_severity_boxed_625_ = lean_unbox(v_severity_620_);
v_isSilent_boxed_626_ = lean_unbox(v_isSilent_621_);
v_res_627_ = lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1(v_ref_618_, v_msgData_619_, v_severity_boxed_625_, v_isSilent_boxed_626_, v___y_622_, v___y_623_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
lean_dec(v_ref_618_);
return v_res_627_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0(lean_object* v_msgData_628_, uint8_t v_severity_629_, uint8_t v_isSilent_630_, lean_object* v___y_631_, lean_object* v___y_632_){
_start:
{
lean_object* v___x_634_; 
v___x_634_ = l_Lean_Elab_Command_getRef___redArg(v___y_631_);
if (lean_obj_tag(v___x_634_) == 0)
{
lean_object* v_a_635_; lean_object* v___x_636_; 
v_a_635_ = lean_ctor_get(v___x_634_, 0);
lean_inc(v_a_635_);
lean_dec_ref_known(v___x_634_, 1);
v___x_636_ = lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1(v_a_635_, v_msgData_628_, v_severity_629_, v_isSilent_630_, v___y_631_, v___y_632_);
lean_dec(v_a_635_);
return v___x_636_;
}
else
{
lean_object* v_a_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_644_; 
lean_dec_ref(v_msgData_628_);
v_a_637_ = lean_ctor_get(v___x_634_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_634_);
if (v_isSharedCheck_644_ == 0)
{
v___x_639_ = v___x_634_;
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_a_637_);
lean_dec(v___x_634_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_642_; 
if (v_isShared_640_ == 0)
{
v___x_642_ = v___x_639_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v_a_637_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0___boxed(lean_object* v_msgData_645_, lean_object* v_severity_646_, lean_object* v_isSilent_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_){
_start:
{
uint8_t v_severity_boxed_651_; uint8_t v_isSilent_boxed_652_; lean_object* v_res_653_; 
v_severity_boxed_651_ = lean_unbox(v_severity_646_);
v_isSilent_boxed_652_ = lean_unbox(v_isSilent_647_);
v_res_653_ = lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0(v_msgData_645_, v_severity_boxed_651_, v_isSilent_boxed_652_, v___y_648_, v___y_649_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(lean_object* v_msgData_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
uint8_t v___x_658_; uint8_t v___x_659_; lean_object* v___x_660_; 
v___x_658_ = 0;
v___x_659_ = 0;
v___x_660_ = lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0(v_msgData_654_, v___x_658_, v___x_659_, v___y_655_, v___y_656_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0___boxed(lean_object* v_msgData_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(v_msgData_661_, v___y_662_, v___y_663_);
lean_dec(v___y_663_);
lean_dec_ref(v___y_662_);
return v_res_665_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_666_ = lean_unsigned_to_nat(2u);
v___x_667_ = lean_nat_to_int(v___x_666_);
return v___x_667_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__12(void){
_start:
{
lean_object* v___x_681_; lean_object* v___x_682_; 
v___x_681_ = lean_unsigned_to_nat(0u);
v___x_682_ = lean_nat_to_int(v___x_681_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg(lean_object* v___x_684_, lean_object* v_init_685_, lean_object* v_x_686_){
_start:
{
if (lean_obj_tag(v_x_686_) == 0)
{
lean_object* v_k_688_; lean_object* v_v_689_; lean_object* v_l_690_; lean_object* v_r_691_; lean_object* v___x_692_; lean_object* v_a_693_; lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_820_; 
v_k_688_ = lean_ctor_get(v_x_686_, 1);
lean_inc(v_k_688_);
v_v_689_ = lean_ctor_get(v_x_686_, 2);
lean_inc(v_v_689_);
v_l_690_ = lean_ctor_get(v_x_686_, 3);
lean_inc(v_l_690_);
v_r_691_ = lean_ctor_get(v_x_686_, 4);
lean_inc(v_r_691_);
lean_dec_ref_known(v_x_686_, 5);
v___x_692_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg(v___x_684_, v_init_685_, v_l_690_);
v_a_693_ = lean_ctor_get(v___x_692_, 0);
v_isSharedCheck_820_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_820_ == 0)
{
v___x_695_ = v___x_692_;
v_isShared_696_ = v_isSharedCheck_820_;
goto v_resetjp_694_;
}
else
{
lean_inc(v_a_693_);
lean_dec(v___x_692_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_820_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v_snd_697_; lean_object* v_a_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_819_; 
v_snd_697_ = lean_ctor_get(v_v_689_, 1);
lean_inc(v_snd_697_);
v_a_698_ = lean_ctor_get(v_a_693_, 0);
v_isSharedCheck_819_ = !lean_is_exclusive(v_a_693_);
if (v_isSharedCheck_819_ == 0)
{
v___x_700_ = v_a_693_;
v_isShared_701_ = v_isSharedCheck_819_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_a_698_);
lean_dec(v_a_693_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_819_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v_fst_702_; lean_object* v___x_704_; uint8_t v_isShared_705_; uint8_t v_isSharedCheck_817_; 
v_fst_702_ = lean_ctor_get(v_v_689_, 0);
v_isSharedCheck_817_ = !lean_is_exclusive(v_v_689_);
if (v_isSharedCheck_817_ == 0)
{
lean_object* v_unused_818_; 
v_unused_818_ = lean_ctor_get(v_v_689_, 1);
lean_dec(v_unused_818_);
v___x_704_ = v_v_689_;
v_isShared_705_ = v_isSharedCheck_817_;
goto v_resetjp_703_;
}
else
{
lean_inc(v_fst_702_);
lean_dec(v_v_689_);
v___x_704_ = lean_box(0);
v_isShared_705_ = v_isSharedCheck_817_;
goto v_resetjp_703_;
}
v_resetjp_703_:
{
lean_object* v_defValue_706_; lean_object* v_descr_707_; lean_object* v_msg1_709_; lean_object* v___y_734_; 
v_defValue_706_ = lean_ctor_get(v_snd_697_, 2);
lean_inc_ref(v_defValue_706_);
v_descr_707_ = lean_ctor_get(v_snd_697_, 3);
lean_inc_ref(v_descr_707_);
lean_dec(v_snd_697_);
switch(lean_obj_tag(v_defValue_706_))
{
case 0:
{
lean_object* v_v_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_757_; 
v_v_744_ = lean_ctor_get(v_defValue_706_, 0);
v_isSharedCheck_757_ = !lean_is_exclusive(v_defValue_706_);
if (v_isSharedCheck_757_ == 0)
{
v___x_746_ = v_defValue_706_;
v_isShared_747_ = v_isSharedCheck_757_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_v_744_);
lean_dec(v_defValue_706_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_757_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_751_; 
v___x_748_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__7));
v___x_749_ = l_String_quote(v_v_744_);
if (v_isShared_747_ == 0)
{
lean_ctor_set_tag(v___x_746_, 3);
lean_ctor_set(v___x_746_, 0, v___x_749_);
v___x_751_ = v___x_746_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v___x_749_);
v___x_751_ = v_reuseFailAlloc_756_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; 
v___x_752_ = l_Std_Format_defWidth;
v___x_753_ = lean_unsigned_to_nat(0u);
v___x_754_ = l_Std_Format_pretty(v___x_751_, v___x_752_, v___x_753_, v___x_753_);
v___x_755_ = lean_string_append(v___x_748_, v___x_754_);
lean_dec_ref(v___x_754_);
v___y_734_ = v___x_755_;
goto v___jp_733_;
}
}
}
case 1:
{
uint8_t v_v_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
v_v_758_ = lean_ctor_get_uint8(v_defValue_706_, 0);
lean_dec_ref_known(v_defValue_706_, 0);
v___x_759_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__8));
v___x_760_ = lean_unsigned_to_nat(0u);
v___x_761_ = l_Bool_repr___redArg(v_v_758_);
v___x_762_ = l_Std_Format_defWidth;
v___x_763_ = l_Std_Format_pretty(v___x_761_, v___x_762_, v___x_760_, v___x_760_);
v___x_764_ = lean_string_append(v___x_759_, v___x_763_);
lean_dec_ref(v___x_763_);
v___y_734_ = v___x_764_;
goto v___jp_733_;
}
case 2:
{
lean_object* v_v_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; 
v_v_765_ = lean_ctor_get(v_defValue_706_, 0);
lean_inc(v_v_765_);
lean_dec_ref_known(v_defValue_706_, 1);
v___x_766_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__9));
v___x_767_ = lean_unsigned_to_nat(0u);
v___x_768_ = l_Lean_Name_reprPrec(v_v_765_, v___x_767_);
v___x_769_ = l_Std_Format_defWidth;
v___x_770_ = l_Std_Format_pretty(v___x_768_, v___x_769_, v___x_767_, v___x_767_);
v___x_771_ = lean_string_append(v___x_766_, v___x_770_);
lean_dec_ref(v___x_770_);
v___y_734_ = v___x_771_;
goto v___jp_733_;
}
case 3:
{
lean_object* v_v_772_; lean_object* v___x_774_; uint8_t v_isShared_775_; uint8_t v_isSharedCheck_785_; 
v_v_772_ = lean_ctor_get(v_defValue_706_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v_defValue_706_);
if (v_isSharedCheck_785_ == 0)
{
v___x_774_ = v_defValue_706_;
v_isShared_775_ = v_isSharedCheck_785_;
goto v_resetjp_773_;
}
else
{
lean_inc(v_v_772_);
lean_dec(v_defValue_706_);
v___x_774_ = lean_box(0);
v_isShared_775_ = v_isSharedCheck_785_;
goto v_resetjp_773_;
}
v_resetjp_773_:
{
lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_779_; 
v___x_776_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__10));
v___x_777_ = l_Nat_reprFast(v_v_772_);
if (v_isShared_775_ == 0)
{
lean_ctor_set(v___x_774_, 0, v___x_777_);
v___x_779_ = v___x_774_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v___x_777_);
v___x_779_ = v_reuseFailAlloc_784_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v___x_780_ = l_Std_Format_defWidth;
v___x_781_ = lean_unsigned_to_nat(0u);
v___x_782_ = l_Std_Format_pretty(v___x_779_, v___x_780_, v___x_781_, v___x_781_);
v___x_783_ = lean_string_append(v___x_776_, v___x_782_);
lean_dec_ref(v___x_782_);
v___y_734_ = v___x_783_;
goto v___jp_733_;
}
}
}
case 4:
{
lean_object* v_v_786_; lean_object* v___x_788_; uint8_t v_isShared_789_; uint8_t v_isSharedCheck_809_; 
v_v_786_ = lean_ctor_get(v_defValue_706_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v_defValue_706_);
if (v_isSharedCheck_809_ == 0)
{
v___x_788_ = v_defValue_706_;
v_isShared_789_ = v_isSharedCheck_809_;
goto v_resetjp_787_;
}
else
{
lean_inc(v_v_786_);
lean_dec(v_defValue_706_);
v___x_788_ = lean_box(0);
v_isShared_789_ = v_isSharedCheck_809_;
goto v_resetjp_787_;
}
v_resetjp_787_:
{
lean_object* v___x_790_; lean_object* v___y_792_; lean_object* v___x_797_; lean_object* v___x_798_; uint8_t v___x_799_; 
v___x_790_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__11));
v___x_797_ = lean_unsigned_to_nat(0u);
v___x_798_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__12, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__12_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__12);
v___x_799_ = lean_int_dec_lt(v_v_786_, v___x_798_);
if (v___x_799_ == 0)
{
lean_object* v___x_800_; lean_object* v___x_802_; 
v___x_800_ = l_Int_repr(v_v_786_);
lean_dec(v_v_786_);
if (v_isShared_789_ == 0)
{
lean_ctor_set_tag(v___x_788_, 3);
lean_ctor_set(v___x_788_, 0, v___x_800_);
v___x_802_ = v___x_788_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v___x_800_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
v___y_792_ = v___x_802_;
goto v___jp_791_;
}
}
else
{
lean_object* v___x_804_; lean_object* v___x_806_; 
v___x_804_ = l_Int_repr(v_v_786_);
lean_dec(v_v_786_);
if (v_isShared_789_ == 0)
{
lean_ctor_set_tag(v___x_788_, 3);
lean_ctor_set(v___x_788_, 0, v___x_804_);
v___x_806_ = v___x_788_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v___x_804_);
v___x_806_ = v_reuseFailAlloc_808_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
lean_object* v___x_807_; 
v___x_807_ = l_Repr_addAppParen(v___x_806_, v___x_797_);
v___y_792_ = v___x_807_;
goto v___jp_791_;
}
}
v___jp_791_:
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v___x_793_ = l_Std_Format_defWidth;
v___x_794_ = lean_unsigned_to_nat(0u);
v___x_795_ = l_Std_Format_pretty(v___y_792_, v___x_793_, v___x_794_, v___x_794_);
v___x_796_ = lean_string_append(v___x_790_, v___x_795_);
lean_dec_ref(v___x_795_);
v___y_734_ = v___x_796_;
goto v___jp_733_;
}
}
}
default: 
{
lean_object* v_v_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; 
v_v_810_ = lean_ctor_get(v_defValue_706_, 0);
lean_inc(v_v_810_);
lean_dec_ref_known(v_defValue_706_, 1);
v___x_811_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__13));
v___x_812_ = lean_unsigned_to_nat(0u);
v___x_813_ = l_Lean_Syntax_instRepr_repr(v_v_810_, v___x_812_);
v___x_814_ = l_Std_Format_defWidth;
v___x_815_ = l_Std_Format_pretty(v___x_813_, v___x_814_, v___x_812_, v___x_812_);
v___x_816_ = lean_string_append(v___x_811_, v___x_815_);
lean_dec_ref(v___x_815_);
v___y_734_ = v___x_816_;
goto v___jp_733_;
}
}
v___jp_708_:
{
lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_713_; 
v___x_710_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0);
v___x_711_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__2));
if (v_isShared_701_ == 0)
{
lean_ctor_set_tag(v___x_700_, 3);
lean_ctor_set(v___x_700_, 0, v_k_688_);
v___x_713_ = v___x_700_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v_k_688_);
v___x_713_ = v_reuseFailAlloc_732_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
lean_object* v___x_715_; 
if (v_isShared_705_ == 0)
{
lean_ctor_set_tag(v___x_704_, 5);
lean_ctor_set(v___x_704_, 1, v___x_713_);
lean_ctor_set(v___x_704_, 0, v___x_711_);
v___x_715_ = v___x_704_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v___x_711_);
lean_ctor_set(v_reuseFailAlloc_731_, 1, v___x_713_);
v___x_715_ = v_reuseFailAlloc_731_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_719_; 
v___x_716_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__4));
v___x_717_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_717_, 0, v___x_715_);
lean_ctor_set(v___x_717_, 1, v___x_716_);
if (v_isShared_696_ == 0)
{
lean_ctor_set_tag(v___x_695_, 3);
lean_ctor_set(v___x_695_, 0, v_msg1_709_);
v___x_719_ = v___x_695_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_730_; 
v_reuseFailAlloc_730_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_730_, 0, v_msg1_709_);
v___x_719_ = v_reuseFailAlloc_730_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_720_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_720_, 0, v___x_717_);
lean_ctor_set(v___x_720_, 1, v___x_719_);
v___x_721_ = lean_box(1);
v___x_722_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_722_, 0, v___x_720_);
lean_ctor_set(v___x_722_, 1, v___x_721_);
v___x_723_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_723_, 0, v_descr_707_);
v___x_724_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_722_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_725_, 0, v___x_710_);
lean_ctor_set(v___x_725_, 1, v___x_724_);
v___x_726_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_726_, 0, v_a_698_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
v___x_727_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_727_, 0, v___x_726_);
lean_ctor_set(v___x_727_, 1, v___x_721_);
v___x_728_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_727_);
lean_ctor_set(v___x_728_, 1, v___x_721_);
v_init_685_ = v___x_728_;
v_x_686_ = v_r_691_;
goto _start;
}
}
}
}
v___jp_733_:
{
lean_object* v_map_735_; lean_object* v___x_736_; 
v_map_735_ = lean_ctor_get(v___x_684_, 0);
v___x_736_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_735_, v_fst_702_);
lean_dec(v_fst_702_);
if (lean_obj_tag(v___x_736_) == 1)
{
lean_object* v_val_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; 
v_val_737_ = lean_ctor_get(v___x_736_, 0);
lean_inc(v_val_737_);
lean_dec_ref_known(v___x_736_, 1);
v___x_738_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__5));
v___x_739_ = lean_string_append(v___y_734_, v___x_738_);
v___x_740_ = lean_data_value_to_string(v_val_737_);
v___x_741_ = lean_string_append(v___x_739_, v___x_740_);
lean_dec_ref(v___x_740_);
v___x_742_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__6));
v___x_743_ = lean_string_append(v___x_741_, v___x_742_);
v_msg1_709_ = v___x_743_;
goto v___jp_708_;
}
else
{
lean_dec(v___x_736_);
v_msg1_709_ = v___y_734_;
goto v___jp_708_;
}
}
}
}
}
}
else
{
lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_821_, 0, v_init_685_);
v___x_822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_822_, 0, v___x_821_);
return v___x_822_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___boxed(lean_object* v___x_823_, lean_object* v_init_824_, lean_object* v_x_825_, lean_object* v___y_826_){
_start:
{
lean_object* v_res_827_; 
v_res_827_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg(v___x_823_, v_init_824_, v_x_825_);
lean_dec_ref(v___x_823_);
return v_res_827_;
}
}
static lean_object* _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0(void){
_start:
{
lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_828_ = lean_box(1);
v___x_829_ = l_Lean_MessageData_ofFormat(v___x_828_);
return v___x_829_;
}
}
static lean_object* _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__3(void){
_start:
{
lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_833_ = ((lean_object*)(lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__2));
v___x_834_ = l_Lean_MessageData_ofFormat(v___x_833_);
return v___x_834_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7(lean_object* v_x_835_, lean_object* v_x_836_){
_start:
{
if (lean_obj_tag(v_x_836_) == 0)
{
return v_x_835_;
}
else
{
lean_object* v_head_837_; lean_object* v_tail_838_; lean_object* v___x_840_; uint8_t v_isShared_841_; uint8_t v_isSharedCheck_860_; 
v_head_837_ = lean_ctor_get(v_x_836_, 0);
v_tail_838_ = lean_ctor_get(v_x_836_, 1);
v_isSharedCheck_860_ = !lean_is_exclusive(v_x_836_);
if (v_isSharedCheck_860_ == 0)
{
v___x_840_ = v_x_836_;
v_isShared_841_ = v_isSharedCheck_860_;
goto v_resetjp_839_;
}
else
{
lean_inc(v_tail_838_);
lean_inc(v_head_837_);
lean_dec(v_x_836_);
v___x_840_ = lean_box(0);
v_isShared_841_ = v_isSharedCheck_860_;
goto v_resetjp_839_;
}
v_resetjp_839_:
{
lean_object* v_before_842_; lean_object* v___x_844_; uint8_t v_isShared_845_; uint8_t v_isSharedCheck_858_; 
v_before_842_ = lean_ctor_get(v_head_837_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v_head_837_);
if (v_isSharedCheck_858_ == 0)
{
lean_object* v_unused_859_; 
v_unused_859_ = lean_ctor_get(v_head_837_, 1);
lean_dec(v_unused_859_);
v___x_844_ = v_head_837_;
v_isShared_845_ = v_isSharedCheck_858_;
goto v_resetjp_843_;
}
else
{
lean_inc(v_before_842_);
lean_dec(v_head_837_);
v___x_844_ = lean_box(0);
v_isShared_845_ = v_isSharedCheck_858_;
goto v_resetjp_843_;
}
v_resetjp_843_:
{
lean_object* v___x_846_; lean_object* v___x_848_; 
v___x_846_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
if (v_isShared_845_ == 0)
{
lean_ctor_set_tag(v___x_844_, 7);
lean_ctor_set(v___x_844_, 1, v___x_846_);
lean_ctor_set(v___x_844_, 0, v_x_835_);
v___x_848_ = v___x_844_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_857_; 
v_reuseFailAlloc_857_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_857_, 0, v_x_835_);
lean_ctor_set(v_reuseFailAlloc_857_, 1, v___x_846_);
v___x_848_ = v_reuseFailAlloc_857_;
goto v_reusejp_847_;
}
v_reusejp_847_:
{
lean_object* v___x_849_; lean_object* v___x_851_; 
v___x_849_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__3, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__3_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__3);
if (v_isShared_841_ == 0)
{
lean_ctor_set_tag(v___x_840_, 7);
lean_ctor_set(v___x_840_, 1, v___x_849_);
lean_ctor_set(v___x_840_, 0, v___x_848_);
v___x_851_ = v___x_840_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v___x_848_);
lean_ctor_set(v_reuseFailAlloc_856_, 1, v___x_849_);
v___x_851_ = v_reuseFailAlloc_856_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; 
v___x_852_ = l_Lean_MessageData_ofSyntax(v_before_842_);
v___x_853_ = l_Lean_indentD(v___x_852_);
v___x_854_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_854_, 0, v___x_851_);
lean_ctor_set(v___x_854_, 1, v___x_853_);
v_x_835_ = v___x_854_;
v_x_836_ = v_tail_838_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; 
v___x_864_ = ((lean_object*)(lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__1));
v___x_865_ = l_Lean_MessageData_ofFormat(v___x_864_);
return v___x_865_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg(lean_object* v_msgData_866_, lean_object* v_macroStack_867_, lean_object* v___y_868_){
_start:
{
lean_object* v___x_870_; lean_object* v_scopes_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v_opts_874_; lean_object* v___x_875_; uint8_t v___x_876_; 
v___x_870_ = lean_st_ref_get(v___y_868_);
v_scopes_871_ = lean_ctor_get(v___x_870_, 2);
lean_inc(v_scopes_871_);
lean_dec(v___x_870_);
v___x_872_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_873_ = l_List_head_x21___redArg(v___x_872_, v_scopes_871_);
lean_dec(v_scopes_871_);
v_opts_874_ = lean_ctor_get(v___x_873_, 1);
lean_inc_ref(v_opts_874_);
lean_dec(v___x_873_);
v___x_875_ = l_Lean_Elab_pp_macroStack;
v___x_876_ = lp_batteries_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__6(v_opts_874_, v___x_875_);
lean_dec_ref(v_opts_874_);
if (v___x_876_ == 0)
{
lean_object* v___x_877_; 
lean_dec(v_macroStack_867_);
v___x_877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_877_, 0, v_msgData_866_);
return v___x_877_;
}
else
{
if (lean_obj_tag(v_macroStack_867_) == 0)
{
lean_object* v___x_878_; 
v___x_878_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_878_, 0, v_msgData_866_);
return v___x_878_;
}
else
{
lean_object* v_head_879_; lean_object* v_after_880_; lean_object* v___x_882_; uint8_t v_isShared_883_; uint8_t v_isSharedCheck_895_; 
v_head_879_ = lean_ctor_get(v_macroStack_867_, 0);
lean_inc(v_head_879_);
v_after_880_ = lean_ctor_get(v_head_879_, 1);
v_isSharedCheck_895_ = !lean_is_exclusive(v_head_879_);
if (v_isSharedCheck_895_ == 0)
{
lean_object* v_unused_896_; 
v_unused_896_ = lean_ctor_get(v_head_879_, 0);
lean_dec(v_unused_896_);
v___x_882_ = v_head_879_;
v_isShared_883_ = v_isSharedCheck_895_;
goto v_resetjp_881_;
}
else
{
lean_inc(v_after_880_);
lean_dec(v_head_879_);
v___x_882_ = lean_box(0);
v_isShared_883_ = v_isSharedCheck_895_;
goto v_resetjp_881_;
}
v_resetjp_881_:
{
lean_object* v___x_884_; lean_object* v___x_886_; 
v___x_884_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
if (v_isShared_883_ == 0)
{
lean_ctor_set_tag(v___x_882_, 7);
lean_ctor_set(v___x_882_, 1, v___x_884_);
lean_ctor_set(v___x_882_, 0, v_msgData_866_);
v___x_886_ = v___x_882_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v_msgData_866_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v___x_884_);
v___x_886_ = v_reuseFailAlloc_894_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v_msgData_891_; lean_object* v___x_892_; lean_object* v___x_893_; 
v___x_887_ = lean_obj_once(&lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__2, &lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__2_once, _init_lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___closed__2);
v___x_888_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_888_, 0, v___x_886_);
lean_ctor_set(v___x_888_, 1, v___x_887_);
v___x_889_ = l_Lean_MessageData_ofSyntax(v_after_880_);
v___x_890_ = l_Lean_indentD(v___x_889_);
v_msgData_891_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_891_, 0, v___x_888_);
lean_ctor_set(v_msgData_891_, 1, v___x_890_);
v___x_892_ = lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7(v_msgData_891_, v_macroStack_867_);
v___x_893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_893_, 0, v___x_892_);
return v___x_893_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg___boxed(lean_object* v_msgData_897_, lean_object* v_macroStack_898_, lean_object* v___y_899_, lean_object* v___y_900_){
_start:
{
lean_object* v_res_901_; 
v_res_901_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg(v_msgData_897_, v_macroStack_898_, v___y_899_);
lean_dec(v___y_899_);
return v_res_901_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(lean_object* v_msg_902_, lean_object* v___y_903_, lean_object* v___y_904_){
_start:
{
lean_object* v___x_906_; 
v___x_906_ = l_Lean_Elab_Command_getRef___redArg(v___y_903_);
if (lean_obj_tag(v___x_906_) == 0)
{
lean_object* v_a_907_; lean_object* v_macroStack_908_; lean_object* v___x_909_; lean_object* v_a_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v_a_913_; lean_object* v___x_915_; uint8_t v_isShared_916_; uint8_t v_isSharedCheck_921_; 
v_a_907_ = lean_ctor_get(v___x_906_, 0);
lean_inc(v_a_907_);
lean_dec_ref_known(v___x_906_, 1);
v_macroStack_908_ = lean_ctor_get(v___y_903_, 4);
v___x_909_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg(v_msg_902_, v___y_904_);
v_a_910_ = lean_ctor_get(v___x_909_, 0);
lean_inc(v_a_910_);
lean_dec_ref(v___x_909_);
v___x_911_ = l_Lean_Elab_getBetterRef(v_a_907_, v_macroStack_908_);
lean_dec(v_a_907_);
lean_inc(v_macroStack_908_);
v___x_912_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg(v_a_910_, v_macroStack_908_, v___y_904_);
v_a_913_ = lean_ctor_get(v___x_912_, 0);
v_isSharedCheck_921_ = !lean_is_exclusive(v___x_912_);
if (v_isSharedCheck_921_ == 0)
{
v___x_915_ = v___x_912_;
v_isShared_916_ = v_isSharedCheck_921_;
goto v_resetjp_914_;
}
else
{
lean_inc(v_a_913_);
lean_dec(v___x_912_);
v___x_915_ = lean_box(0);
v_isShared_916_ = v_isSharedCheck_921_;
goto v_resetjp_914_;
}
v_resetjp_914_:
{
lean_object* v___x_917_; lean_object* v___x_919_; 
v___x_917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_917_, 0, v___x_911_);
lean_ctor_set(v___x_917_, 1, v_a_913_);
if (v_isShared_916_ == 0)
{
lean_ctor_set_tag(v___x_915_, 1);
lean_ctor_set(v___x_915_, 0, v___x_917_);
v___x_919_ = v___x_915_;
goto v_reusejp_918_;
}
else
{
lean_object* v_reuseFailAlloc_920_; 
v_reuseFailAlloc_920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_920_, 0, v___x_917_);
v___x_919_ = v_reuseFailAlloc_920_;
goto v_reusejp_918_;
}
v_reusejp_918_:
{
return v___x_919_;
}
}
}
else
{
lean_object* v_a_922_; lean_object* v___x_924_; uint8_t v_isShared_925_; uint8_t v_isSharedCheck_929_; 
lean_dec_ref(v_msg_902_);
v_a_922_ = lean_ctor_get(v___x_906_, 0);
v_isSharedCheck_929_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_929_ == 0)
{
v___x_924_ = v___x_906_;
v_isShared_925_ = v_isSharedCheck_929_;
goto v_resetjp_923_;
}
else
{
lean_inc(v_a_922_);
lean_dec(v___x_906_);
v___x_924_ = lean_box(0);
v_isShared_925_ = v_isSharedCheck_929_;
goto v_resetjp_923_;
}
v_resetjp_923_:
{
lean_object* v___x_927_; 
if (v_isShared_925_ == 0)
{
v___x_927_ = v___x_924_;
goto v_reusejp_926_;
}
else
{
lean_object* v_reuseFailAlloc_928_; 
v_reuseFailAlloc_928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_928_, 0, v_a_922_);
v___x_927_ = v_reuseFailAlloc_928_;
goto v_reusejp_926_;
}
v_reusejp_926_:
{
return v___x_927_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg___boxed(lean_object* v_msg_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v_msg_930_, v___y_931_, v___y_932_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
return v_res_934_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__1(void){
_start:
{
lean_object* v___x_936_; lean_object* v___x_937_; 
v___x_936_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__0));
v___x_937_ = l_Lean_stringToMessageData(v___x_936_);
return v___x_937_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__3(void){
_start:
{
lean_object* v___x_939_; lean_object* v___x_940_; 
v___x_939_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__2));
v___x_940_ = l_Lean_stringToMessageData(v___x_939_);
return v___x_940_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption(lean_object* v_id_941_, lean_object* v_a_942_, lean_object* v_a_943_){
_start:
{
lean_object* v___y_946_; lean_object* v___y_947_; lean_object* v_a_948_; lean_object* v___y_952_; lean_object* v___y_953_; lean_object* v___y_954_; lean_object* v___y_955_; lean_object* v___y_956_; lean_object* v___y_961_; lean_object* v_a_962_; lean_object* v___y_977_; 
if (lean_obj_tag(v_id_941_) == 0)
{
lean_object* v___x_997_; 
v___x_997_ = lean_box(0);
v___y_977_ = v___x_997_;
goto v___jp_976_;
}
else
{
lean_object* v_val_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1008_; 
v_val_998_ = lean_ctor_get(v_id_941_, 0);
v_isSharedCheck_1008_ = !lean_is_exclusive(v_id_941_);
if (v_isSharedCheck_1008_ == 0)
{
v___x_1000_ = v_id_941_;
v_isShared_1001_ = v_isSharedCheck_1008_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_val_998_);
lean_dec(v_id_941_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1008_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1002_; uint8_t v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1006_; 
v___x_1002_ = l_Lean_Syntax_getId(v_val_998_);
lean_dec(v_val_998_);
v___x_1003_ = 0;
v___x_1004_ = l_Lean_Name_toString(v___x_1002_, v___x_1003_);
if (v_isShared_1001_ == 0)
{
lean_ctor_set(v___x_1000_, 0, v___x_1004_);
v___x_1006_ = v___x_1000_;
goto v_reusejp_1005_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v___x_1004_);
v___x_1006_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1005_;
}
v_reusejp_1005_:
{
v___y_977_ = v___x_1006_;
goto v___jp_976_;
}
}
}
v___jp_945_:
{
lean_object* v___x_949_; lean_object* v___x_950_; 
v___x_949_ = l_Lean_MessageData_ofFormat(v_a_948_);
v___x_950_ = lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(v___x_949_, v___y_946_, v___y_947_);
return v___x_950_;
}
v___jp_951_:
{
lean_object* v___x_957_; lean_object* v_a_958_; lean_object* v_a_959_; 
lean_inc(v___y_952_);
v___x_957_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg(v___y_954_, v___y_952_, v___y_953_);
lean_dec_ref(v___y_954_);
v_a_958_ = lean_ctor_get(v___x_957_, 0);
lean_inc(v_a_958_);
lean_dec_ref(v___x_957_);
v_a_959_ = lean_ctor_get(v_a_958_, 0);
lean_inc(v_a_959_);
lean_dec(v_a_958_);
v___y_946_ = v___y_955_;
v___y_947_ = v___y_956_;
v_a_948_ = v_a_959_;
goto v___jp_945_;
}
v___jp_960_:
{
lean_object* v___x_963_; lean_object* v_scopes_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v_opts_967_; lean_object* v___x_968_; 
v___x_963_ = lean_st_ref_get(v_a_943_);
v_scopes_964_ = lean_ctor_get(v___x_963_, 2);
lean_inc(v_scopes_964_);
lean_dec(v___x_963_);
v___x_965_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_966_ = l_List_head_x21___redArg(v___x_965_, v_scopes_964_);
lean_dec(v_scopes_964_);
v_opts_967_ = lean_ctor_get(v___x_966_, 1);
lean_inc_ref(v_opts_967_);
lean_dec(v___x_966_);
v___x_968_ = lean_box(0);
if (lean_obj_tag(v_a_962_) == 0)
{
lean_dec(v___y_961_);
v___y_952_ = v___x_968_;
v___y_953_ = v_a_962_;
v___y_954_ = v_opts_967_;
v___y_955_ = v_a_942_;
v___y_956_ = v_a_943_;
goto v___jp_951_;
}
else
{
lean_dec_ref(v_opts_967_);
if (lean_obj_tag(v___y_961_) == 0)
{
lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_969_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__1, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__1);
v___x_970_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_969_, v_a_942_, v_a_943_);
return v___x_970_;
}
else
{
lean_object* v_val_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; 
v_val_971_ = lean_ctor_get(v___y_961_, 0);
lean_inc(v_val_971_);
lean_dec_ref_known(v___y_961_, 1);
v___x_972_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__3, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__3_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___closed__3);
v___x_973_ = l_Lean_stringToMessageData(v_val_971_);
v___x_974_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_974_, 0, v___x_972_);
lean_ctor_set(v___x_974_, 1, v___x_973_);
v___x_975_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_974_, v_a_942_, v_a_943_);
return v___x_975_;
}
}
}
v___jp_976_:
{
lean_object* v___x_978_; 
v___x_978_ = l_Lean_getOptionDecls();
if (lean_obj_tag(v___x_978_) == 0)
{
lean_object* v_a_979_; lean_object* v_decls_980_; lean_object* v___x_981_; lean_object* v_a_982_; lean_object* v_a_983_; 
v_a_979_ = lean_ctor_get(v___x_978_, 0);
lean_inc(v_a_979_);
lean_dec_ref_known(v___x_978_, 1);
v_decls_980_ = lean_box(1);
v___x_981_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg(v___y_977_, v_decls_980_, v_a_979_);
v_a_982_ = lean_ctor_get(v___x_981_, 0);
lean_inc(v_a_982_);
lean_dec_ref(v___x_981_);
v_a_983_ = lean_ctor_get(v_a_982_, 0);
lean_inc(v_a_983_);
lean_dec(v_a_982_);
v___y_961_ = v___y_977_;
v_a_962_ = v_a_983_;
goto v___jp_960_;
}
else
{
lean_object* v_a_984_; lean_object* v___x_986_; uint8_t v_isShared_987_; uint8_t v_isSharedCheck_996_; 
lean_dec(v___y_977_);
v_a_984_ = lean_ctor_get(v___x_978_, 0);
v_isSharedCheck_996_ = !lean_is_exclusive(v___x_978_);
if (v_isSharedCheck_996_ == 0)
{
v___x_986_ = v___x_978_;
v_isShared_987_ = v_isSharedCheck_996_;
goto v_resetjp_985_;
}
else
{
lean_inc(v_a_984_);
lean_dec(v___x_978_);
v___x_986_ = lean_box(0);
v_isShared_987_ = v_isSharedCheck_996_;
goto v_resetjp_985_;
}
v_resetjp_985_:
{
lean_object* v_ref_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_994_; 
v_ref_988_ = lean_ctor_get(v_a_942_, 7);
v___x_989_ = lean_io_error_to_string(v_a_984_);
v___x_990_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_990_, 0, v___x_989_);
v___x_991_ = l_Lean_MessageData_ofFormat(v___x_990_);
lean_inc(v_ref_988_);
v___x_992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_992_, 0, v_ref_988_);
lean_ctor_set(v___x_992_, 1, v___x_991_);
if (v_isShared_987_ == 0)
{
lean_ctor_set(v___x_986_, 0, v___x_992_);
v___x_994_ = v___x_986_;
goto v_reusejp_993_;
}
else
{
lean_object* v_reuseFailAlloc_995_; 
v_reuseFailAlloc_995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_995_, 0, v___x_992_);
v___x_994_ = v_reuseFailAlloc_995_;
goto v_reusejp_993_;
}
v_reusejp_993_:
{
return v___x_994_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption___boxed(lean_object* v_id_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_){
_start:
{
lean_object* v_res_1013_; 
v_res_1013_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption(v_id_1009_, v_a_1010_, v_a_1011_);
lean_dec(v_a_1011_);
lean_dec_ref(v_a_1010_);
return v_res_1013_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1(lean_object* v___x_1014_, lean_object* v_init_1015_, lean_object* v_x_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_){
_start:
{
lean_object* v___x_1020_; 
v___x_1020_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg(v___x_1014_, v_init_1015_, v_x_1016_);
return v___x_1020_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___boxed(lean_object* v___x_1021_, lean_object* v_init_1022_, lean_object* v_x_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_){
_start:
{
lean_object* v_res_1027_; 
v_res_1027_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1(v___x_1021_, v_init_1022_, v_x_1023_, v___y_1024_, v___y_1025_);
lean_dec(v___y_1025_);
lean_dec_ref(v___y_1024_);
lean_dec_ref(v___x_1021_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3(lean_object* v_msgData_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___redArg(v_msgData_1028_, v___y_1030_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3___boxed(lean_object* v_msgData_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_){
_start:
{
lean_object* v_res_1037_; 
v_res_1037_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__3(v_msgData_1033_, v___y_1034_, v___y_1035_);
lean_dec(v___y_1035_);
lean_dec_ref(v___y_1034_);
return v_res_1037_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2(lean_object* v_00_u03b1_1038_, lean_object* v_msg_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v___x_1043_; 
v___x_1043_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v_msg_1039_, v___y_1040_, v___y_1041_);
return v___x_1043_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___boxed(lean_object* v_00_u03b1_1044_, lean_object* v_msg_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_){
_start:
{
lean_object* v_res_1049_; 
v_res_1049_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2(v_00_u03b1_1044_, v_msg_1045_, v___y_1046_, v___y_1047_);
lean_dec(v___y_1047_);
lean_dec_ref(v___y_1046_);
return v_res_1049_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3(lean_object* v_00_u03b2_1050_, lean_object* v_k_1051_, lean_object* v_v_1052_, lean_object* v_t_1053_, lean_object* v_hl_1054_){
_start:
{
lean_object* v___x_1055_; 
v___x_1055_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v_k_1051_, v_v_1052_, v_t_1053_);
return v___x_1055_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4(lean_object* v___y_1056_, lean_object* v_init_1057_, lean_object* v_x_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_){
_start:
{
lean_object* v___x_1062_; 
v___x_1062_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___redArg(v___y_1056_, v_init_1057_, v_x_1058_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4___boxed(lean_object* v___y_1063_, lean_object* v_init_1064_, lean_object* v_x_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
lean_object* v_res_1069_; 
v_res_1069_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__4(v___y_1063_, v_init_1064_, v_x_1065_, v___y_1066_, v___y_1067_);
lean_dec(v___y_1067_);
lean_dec_ref(v___y_1066_);
lean_dec(v___y_1063_);
return v_res_1069_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4(lean_object* v_msgData_1070_, lean_object* v_macroStack_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_){
_start:
{
lean_object* v___x_1075_; 
v___x_1075_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___redArg(v_msgData_1070_, v_macroStack_1071_, v___y_1073_);
return v___x_1075_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4___boxed(lean_object* v_msgData_1076_, lean_object* v_macroStack_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v_res_1081_; 
v_res_1081_ = lp_batteries_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4(v_msgData_1076_, v_macroStack_1077_, v___y_1078_, v___y_1079_);
lean_dec(v___y_1079_);
lean_dec_ref(v___y_1078_);
return v_res_1081_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; 
v___x_1082_ = lean_box(0);
v___x_1083_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1084_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1084_, 0, v___x_1083_);
lean_ctor_set(v___x_1084_, 1, v___x_1082_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg(){
_start:
{
lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1086_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___closed__0);
v___x_1087_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1087_, 0, v___x_1086_);
return v___x_1087_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg___boxed(lean_object* v___y_1088_){
_start:
{
lean_object* v_res_1089_; 
v_res_1089_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v_res_1089_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0(lean_object* v_00_u03b1_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v___x_1094_; 
v___x_1094_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_1094_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___boxed(lean_object* v_00_u03b1_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_){
_start:
{
lean_object* v_res_1099_; 
v_res_1099_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0(v_00_u03b1_1095_, v___y_1096_, v___y_1097_);
lean_dec(v___y_1097_);
lean_dec_ref(v___y_1096_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1(lean_object* v_x_1100_, lean_object* v_a_1101_, lean_object* v_a_1102_){
_start:
{
lean_object* v___x_1104_; uint8_t v___x_1105_; 
v___x_1104_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Option_______00__closed__3));
lean_inc(v_x_1100_);
v___x_1105_ = l_Lean_Syntax_isOfKind(v_x_1100_, v___x_1104_);
if (v___x_1105_ == 0)
{
lean_object* v___x_1106_; 
lean_dec(v_x_1100_);
v___x_1106_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_1106_;
}
else
{
lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; 
v___x_1107_ = lean_unsigned_to_nat(2u);
v___x_1108_ = l_Lean_Syntax_getArg(v_x_1100_, v___x_1107_);
lean_dec(v_x_1100_);
v___x_1109_ = l_Lean_Syntax_getOptional_x3f(v___x_1108_);
lean_dec(v___x_1108_);
if (lean_obj_tag(v___x_1109_) == 0)
{
lean_object* v___x_1110_; lean_object* v___x_1111_; 
v___x_1110_ = lean_box(0);
v___x_1111_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption(v___x_1110_, v_a_1101_, v_a_1102_);
return v___x_1111_;
}
else
{
lean_object* v_val_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1120_; 
v_val_1112_ = lean_ctor_get(v___x_1109_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_1109_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1114_ = v___x_1109_;
v_isShared_1115_ = v_isSharedCheck_1120_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_val_1112_);
lean_dec(v___x_1109_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1120_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v___x_1117_; 
if (v_isShared_1115_ == 0)
{
v___x_1117_ = v___x_1114_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v_val_1112_);
v___x_1117_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
lean_object* v___x_1118_; 
v___x_1118_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption(v___x_1117_, v_a_1101_, v_a_1102_);
return v___x_1118_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1___boxed(lean_object* v_x_1121_, lean_object* v_a_1122_, lean_object* v_a_1123_, lean_object* v_a_1124_){
_start:
{
lean_object* v_res_1125_; 
v_res_1125_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1(v_x_1121_, v_a_1122_, v_a_1123_);
lean_dec(v_a_1123_);
lean_dec_ref(v_a_1122_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg(lean_object* v___y_1177_, lean_object* v_as_x27_1178_, lean_object* v_b_1179_){
_start:
{
if (lean_obj_tag(v_as_x27_1178_) == 0)
{
lean_object* v___x_1181_; 
v___x_1181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1181_, 0, v_b_1179_);
return v___x_1181_;
}
else
{
lean_object* v_head_1182_; lean_object* v_tail_1183_; lean_object* v_fst_1184_; lean_object* v_snd_1185_; uint8_t v___x_1186_; lean_object* v___x_1187_; 
v_head_1182_ = lean_ctor_get(v_as_x27_1178_, 0);
v_tail_1183_ = lean_ctor_get(v_as_x27_1178_, 1);
v_fst_1184_ = lean_ctor_get(v_head_1182_, 0);
v_snd_1185_ = lean_ctor_get(v_head_1182_, 1);
v___x_1186_ = 0;
lean_inc(v_fst_1184_);
v___x_1187_ = l_Lean_Name_toString(v_fst_1184_, v___x_1186_);
if (lean_obj_tag(v___y_1177_) == 1)
{
lean_object* v_val_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; uint8_t v___x_1194_; 
v_val_1191_ = lean_ctor_get(v___y_1177_, 0);
v___x_1192_ = lean_string_utf8_byte_size(v___x_1187_);
v___x_1193_ = lean_string_utf8_byte_size(v_val_1191_);
v___x_1194_ = lean_nat_dec_le(v___x_1193_, v___x_1192_);
if (v___x_1194_ == 0)
{
lean_dec_ref(v___x_1187_);
v_as_x27_1178_ = v_tail_1183_;
goto _start;
}
else
{
lean_object* v___x_1196_; uint8_t v___x_1197_; 
v___x_1196_ = lean_unsigned_to_nat(0u);
v___x_1197_ = lean_string_memcmp(v___x_1187_, v_val_1191_, v___x_1196_, v___x_1196_, v___x_1193_);
if (v___x_1197_ == 0)
{
lean_dec_ref(v___x_1187_);
v_as_x27_1178_ = v_tail_1183_;
goto _start;
}
else
{
goto v___jp_1188_;
}
}
}
else
{
goto v___jp_1188_;
}
v___jp_1188_:
{
lean_object* v___x_1189_; 
lean_inc(v_snd_1185_);
v___x_1189_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v___x_1187_, v_snd_1185_, v_b_1179_);
v_as_x27_1178_ = v_tail_1183_;
v_b_1179_ = v___x_1189_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg___boxed(lean_object* v___y_1199_, lean_object* v_as_x27_1200_, lean_object* v_b_1201_, lean_object* v___y_1202_){
_start:
{
lean_object* v_res_1203_; 
v_res_1203_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg(v___y_1199_, v_as_x27_1200_, v_b_1201_);
lean_dec(v_as_x27_1200_);
lean_dec(v___y_1199_);
return v_res_1203_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2(lean_object* v_x_1204_, lean_object* v_x_1205_){
_start:
{
if (lean_obj_tag(v_x_1205_) == 0)
{
lean_inc(v_x_1204_);
return v_x_1204_;
}
else
{
lean_object* v_key_1206_; lean_object* v_value_1207_; lean_object* v_tail_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
v_key_1206_ = lean_ctor_get(v_x_1205_, 0);
v_value_1207_ = lean_ctor_get(v_x_1205_, 1);
v_tail_1208_ = lean_ctor_get(v_x_1205_, 2);
v___x_1209_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2(v_x_1204_, v_tail_1208_);
lean_inc(v_value_1207_);
lean_inc(v_key_1206_);
v___x_1210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1210_, 0, v_key_1206_);
lean_ctor_set(v___x_1210_, 1, v_value_1207_);
v___x_1211_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1211_, 0, v___x_1210_);
lean_ctor_set(v___x_1211_, 1, v___x_1209_);
return v___x_1211_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2___boxed(lean_object* v_x_1212_, lean_object* v_x_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2(v_x_1212_, v_x_1213_);
lean_dec(v_x_1213_);
lean_dec(v_x_1212_);
return v_res_1214_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__3(lean_object* v_as_1215_, size_t v_i_1216_, size_t v_stop_1217_, lean_object* v_b_1218_){
_start:
{
uint8_t v___x_1219_; 
v___x_1219_ = lean_usize_dec_eq(v_i_1216_, v_stop_1217_);
if (v___x_1219_ == 0)
{
size_t v___x_1220_; size_t v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; 
v___x_1220_ = ((size_t)1ULL);
v___x_1221_ = lean_usize_sub(v_i_1216_, v___x_1220_);
v___x_1222_ = lean_array_uget_borrowed(v_as_1215_, v___x_1221_);
v___x_1223_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldrM___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__2(v_b_1218_, v___x_1222_);
lean_dec(v_b_1218_);
v_i_1216_ = v___x_1221_;
v_b_1218_ = v___x_1223_;
goto _start;
}
else
{
return v_b_1218_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__3___boxed(lean_object* v_as_1225_, lean_object* v_i_1226_, lean_object* v_stop_1227_, lean_object* v_b_1228_){
_start:
{
size_t v_i_boxed_1229_; size_t v_stop_boxed_1230_; lean_object* v_res_1231_; 
v_i_boxed_1229_ = lean_unbox_usize(v_i_1226_);
lean_dec(v_i_1226_);
v_stop_boxed_1230_ = lean_unbox_usize(v_stop_1227_);
lean_dec(v_stop_1227_);
v_res_1231_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__3(v_as_1225_, v_i_boxed_1229_, v_stop_boxed_1230_, v_b_1228_);
lean_dec_ref(v_as_1225_);
return v_res_1231_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg(lean_object* v___x_1235_, lean_object* v_init_1236_, lean_object* v_x_1237_, lean_object* v___y_1238_){
_start:
{
if (lean_obj_tag(v_x_1237_) == 0)
{
lean_object* v_k_1240_; lean_object* v_v_1241_; lean_object* v_l_1242_; lean_object* v_r_1243_; lean_object* v___x_1244_; 
v_k_1240_ = lean_ctor_get(v_x_1237_, 1);
lean_inc(v_k_1240_);
v_v_1241_ = lean_ctor_get(v_x_1237_, 2);
lean_inc(v_v_1241_);
v_l_1242_ = lean_ctor_get(v_x_1237_, 3);
lean_inc(v_l_1242_);
v_r_1243_ = lean_ctor_get(v_x_1237_, 4);
lean_inc(v_r_1243_);
lean_dec_ref_known(v_x_1237_, 5);
lean_inc_ref(v___x_1235_);
v___x_1244_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg(v___x_1235_, v_init_1236_, v_l_1242_, v___y_1238_);
if (lean_obj_tag(v___x_1244_) == 0)
{
lean_object* v_a_1245_; lean_object* v_toAttributeImplCore_1246_; lean_object* v___x_1248_; uint8_t v_isShared_1249_; uint8_t v_isSharedCheck_1306_; 
v_a_1245_ = lean_ctor_get(v___x_1244_, 0);
lean_inc(v_a_1245_);
lean_dec_ref_known(v___x_1244_, 1);
v_toAttributeImplCore_1246_ = lean_ctor_get(v_v_1241_, 0);
v_isSharedCheck_1306_ = !lean_is_exclusive(v_v_1241_);
if (v_isSharedCheck_1306_ == 0)
{
lean_object* v_unused_1307_; lean_object* v_unused_1308_; 
v_unused_1307_ = lean_ctor_get(v_v_1241_, 2);
lean_dec(v_unused_1307_);
v_unused_1308_ = lean_ctor_get(v_v_1241_, 1);
lean_dec(v_unused_1308_);
v___x_1248_ = v_v_1241_;
v_isShared_1249_ = v_isSharedCheck_1306_;
goto v_resetjp_1247_;
}
else
{
lean_inc(v_toAttributeImplCore_1246_);
lean_dec(v_v_1241_);
v___x_1248_ = lean_box(0);
v_isShared_1249_ = v_isSharedCheck_1306_;
goto v_resetjp_1247_;
}
v_resetjp_1247_:
{
lean_object* v_a_1250_; lean_object* v___x_1252_; uint8_t v_isShared_1253_; uint8_t v_isSharedCheck_1305_; 
v_a_1250_ = lean_ctor_get(v_a_1245_, 0);
v_isSharedCheck_1305_ = !lean_is_exclusive(v_a_1245_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1252_ = v_a_1245_;
v_isShared_1253_ = v_isSharedCheck_1305_;
goto v_resetjp_1251_;
}
else
{
lean_inc(v_a_1250_);
lean_dec(v_a_1245_);
v___x_1252_ = lean_box(0);
v_isShared_1253_ = v_isSharedCheck_1305_;
goto v_resetjp_1251_;
}
v_resetjp_1251_:
{
lean_object* v_ref_1254_; lean_object* v_descr_1255_; uint8_t v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; 
v_ref_1254_ = lean_ctor_get(v_toAttributeImplCore_1246_, 0);
lean_inc(v_ref_1254_);
v_descr_1255_ = lean_ctor_get(v_toAttributeImplCore_1246_, 2);
lean_inc_ref(v_descr_1255_);
lean_dec_ref(v_toAttributeImplCore_1246_);
v___x_1256_ = 1;
v___x_1257_ = l_Lean_Options_empty;
v___x_1258_ = lean_box(0);
v___x_1259_ = lean_box(0);
lean_inc_ref(v___x_1235_);
v___x_1260_ = l_Lean_findDocString_x3f(v___x_1235_, v_ref_1254_, v___x_1256_, v___x_1257_, v___x_1258_, v___x_1259_);
if (lean_obj_tag(v___x_1260_) == 0)
{
lean_object* v_a_1261_; lean_object* v_msg1_1263_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; 
v_a_1261_ = lean_ctor_get(v___x_1260_, 0);
lean_inc(v_a_1261_);
lean_dec_ref_known(v___x_1260_, 1);
v___x_1274_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__0));
v___x_1275_ = lean_string_append(v___x_1274_, v_k_1240_);
lean_dec(v_k_1240_);
v___x_1276_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__1));
v___x_1277_ = lean_string_append(v___x_1275_, v___x_1276_);
v___x_1278_ = lean_string_append(v___x_1277_, v_descr_1255_);
lean_dec_ref(v_descr_1255_);
if (lean_obj_tag(v_a_1261_) == 1)
{
lean_object* v_val_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1285_; 
v_val_1279_ = lean_ctor_get(v_a_1261_, 0);
lean_inc(v_val_1279_);
lean_dec_ref_known(v_a_1261_, 1);
v___x_1280_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__2));
v___x_1281_ = lean_string_append(v___x_1278_, v___x_1280_);
v___x_1282_ = lean_unsigned_to_nat(0u);
v___x_1283_ = lean_string_utf8_byte_size(v_val_1279_);
if (v_isShared_1249_ == 0)
{
lean_ctor_set(v___x_1248_, 2, v___x_1283_);
lean_ctor_set(v___x_1248_, 1, v___x_1282_);
lean_ctor_set(v___x_1248_, 0, v_val_1279_);
v___x_1285_ = v___x_1248_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v_val_1279_);
lean_ctor_set(v_reuseFailAlloc_1289_, 1, v___x_1282_);
lean_ctor_set(v_reuseFailAlloc_1289_, 2, v___x_1283_);
v___x_1285_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1286_ = l_String_Slice_trimAscii(v___x_1285_);
v___x_1287_ = l_String_Slice_toString(v___x_1286_);
lean_dec_ref(v___x_1286_);
v___x_1288_ = lean_string_append(v___x_1281_, v___x_1287_);
lean_dec_ref(v___x_1287_);
v_msg1_1263_ = v___x_1288_;
goto v___jp_1262_;
}
}
else
{
lean_dec(v_a_1261_);
lean_del_object(v___x_1248_);
v_msg1_1263_ = v___x_1278_;
goto v___jp_1262_;
}
v___jp_1262_:
{
lean_object* v___x_1264_; lean_object* v___x_1266_; 
v___x_1264_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__1___redArg___closed__0);
if (v_isShared_1253_ == 0)
{
lean_ctor_set_tag(v___x_1252_, 3);
lean_ctor_set(v___x_1252_, 0, v_msg1_1263_);
v___x_1266_ = v___x_1252_;
goto v_reusejp_1265_;
}
else
{
lean_object* v_reuseFailAlloc_1273_; 
v_reuseFailAlloc_1273_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1273_, 0, v_msg1_1263_);
v___x_1266_ = v_reuseFailAlloc_1273_;
goto v_reusejp_1265_;
}
v_reusejp_1265_:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; 
v___x_1267_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1267_, 0, v___x_1264_);
lean_ctor_set(v___x_1267_, 1, v___x_1266_);
v___x_1268_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1268_, 0, v_a_1250_);
lean_ctor_set(v___x_1268_, 1, v___x_1267_);
v___x_1269_ = lean_box(1);
v___x_1270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1268_);
lean_ctor_set(v___x_1270_, 1, v___x_1269_);
v___x_1271_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1270_);
lean_ctor_set(v___x_1271_, 1, v___x_1269_);
v_init_1236_ = v___x_1271_;
v_x_1237_ = v_r_1243_;
goto _start;
}
}
}
else
{
lean_object* v_a_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1304_; 
lean_dec_ref(v_descr_1255_);
lean_dec(v_a_1250_);
lean_del_object(v___x_1248_);
lean_dec(v_r_1243_);
lean_dec(v_k_1240_);
lean_dec_ref(v___x_1235_);
v_a_1290_ = lean_ctor_get(v___x_1260_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1260_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1292_ = v___x_1260_;
v_isShared_1293_ = v_isSharedCheck_1304_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_a_1290_);
lean_dec(v___x_1260_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1304_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
lean_object* v_ref_1294_; lean_object* v___x_1295_; lean_object* v___x_1297_; 
v_ref_1294_ = lean_ctor_get(v___y_1238_, 7);
v___x_1295_ = lean_io_error_to_string(v_a_1290_);
if (v_isShared_1253_ == 0)
{
lean_ctor_set_tag(v___x_1252_, 3);
lean_ctor_set(v___x_1252_, 0, v___x_1295_);
v___x_1297_ = v___x_1252_;
goto v_reusejp_1296_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v___x_1295_);
v___x_1297_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1296_;
}
v_reusejp_1296_:
{
lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1301_; 
v___x_1298_ = l_Lean_MessageData_ofFormat(v___x_1297_);
lean_inc(v_ref_1294_);
v___x_1299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1299_, 0, v_ref_1294_);
lean_ctor_set(v___x_1299_, 1, v___x_1298_);
if (v_isShared_1293_ == 0)
{
lean_ctor_set(v___x_1292_, 0, v___x_1299_);
v___x_1301_ = v___x_1292_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v___x_1299_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
}
}
}
}
}
else
{
lean_dec(v_r_1243_);
lean_dec(v_v_1241_);
lean_dec(v_k_1240_);
lean_dec_ref(v___x_1235_);
return v___x_1244_;
}
}
else
{
lean_object* v___x_1309_; lean_object* v___x_1310_; 
lean_dec_ref(v___x_1235_);
v___x_1309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1309_, 0, v_init_1236_);
v___x_1310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1310_, 0, v___x_1309_);
return v___x_1310_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___boxed(lean_object* v___x_1311_, lean_object* v_init_1312_, lean_object* v_x_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_){
_start:
{
lean_object* v_res_1316_; 
v_res_1316_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg(v___x_1311_, v_init_1312_, v_x_1313_, v___y_1314_);
lean_dec_ref(v___y_1314_);
return v_res_1316_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__1(void){
_start:
{
lean_object* v___x_1318_; lean_object* v___x_1319_; 
v___x_1318_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__0));
v___x_1319_ = l_Lean_stringToMessageData(v___x_1318_);
return v___x_1319_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__3(void){
_start:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; 
v___x_1321_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__2));
v___x_1322_ = l_Lean_stringToMessageData(v___x_1321_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(lean_object* v_id_1323_, lean_object* v_a_1324_, lean_object* v_a_1325_){
_start:
{
lean_object* v___y_1328_; lean_object* v___y_1329_; lean_object* v_a_1330_; lean_object* v___y_1334_; lean_object* v___y_1335_; lean_object* v___y_1336_; lean_object* v___y_1337_; lean_object* v___y_1338_; lean_object* v___y_1351_; lean_object* v___y_1352_; lean_object* v___y_1353_; lean_object* v___y_1367_; 
if (lean_obj_tag(v_id_1323_) == 0)
{
lean_object* v___x_1379_; 
v___x_1379_ = lean_box(0);
v___y_1367_ = v___x_1379_;
goto v___jp_1366_;
}
else
{
lean_object* v_val_1380_; lean_object* v___x_1382_; uint8_t v_isShared_1383_; uint8_t v_isSharedCheck_1390_; 
v_val_1380_ = lean_ctor_get(v_id_1323_, 0);
v_isSharedCheck_1390_ = !lean_is_exclusive(v_id_1323_);
if (v_isSharedCheck_1390_ == 0)
{
v___x_1382_ = v_id_1323_;
v_isShared_1383_ = v_isSharedCheck_1390_;
goto v_resetjp_1381_;
}
else
{
lean_inc(v_val_1380_);
lean_dec(v_id_1323_);
v___x_1382_ = lean_box(0);
v_isShared_1383_ = v_isSharedCheck_1390_;
goto v_resetjp_1381_;
}
v_resetjp_1381_:
{
lean_object* v___x_1384_; uint8_t v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1388_; 
v___x_1384_ = l_Lean_Syntax_getId(v_val_1380_);
lean_dec(v_val_1380_);
v___x_1385_ = 0;
v___x_1386_ = l_Lean_Name_toString(v___x_1384_, v___x_1385_);
if (v_isShared_1383_ == 0)
{
lean_ctor_set(v___x_1382_, 0, v___x_1386_);
v___x_1388_ = v___x_1382_;
goto v_reusejp_1387_;
}
else
{
lean_object* v_reuseFailAlloc_1389_; 
v_reuseFailAlloc_1389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1389_, 0, v___x_1386_);
v___x_1388_ = v_reuseFailAlloc_1389_;
goto v_reusejp_1387_;
}
v_reusejp_1387_:
{
v___y_1367_ = v___x_1388_;
goto v___jp_1366_;
}
}
}
v___jp_1327_:
{
lean_object* v___x_1331_; lean_object* v___x_1332_; 
v___x_1331_ = l_Lean_MessageData_ofFormat(v_a_1330_);
v___x_1332_ = lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(v___x_1331_, v___y_1329_, v___y_1328_);
return v___x_1332_;
}
v___jp_1333_:
{
lean_object* v___x_1339_; 
lean_inc(v___y_1334_);
v___x_1339_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg(v___y_1336_, v___y_1334_, v___y_1335_, v___y_1337_);
if (lean_obj_tag(v___x_1339_) == 0)
{
lean_object* v_a_1340_; lean_object* v_a_1341_; 
v_a_1340_ = lean_ctor_get(v___x_1339_, 0);
lean_inc(v_a_1340_);
lean_dec_ref_known(v___x_1339_, 1);
v_a_1341_ = lean_ctor_get(v_a_1340_, 0);
lean_inc(v_a_1341_);
lean_dec(v_a_1340_);
v___y_1328_ = v___y_1338_;
v___y_1329_ = v___y_1337_;
v_a_1330_ = v_a_1341_;
goto v___jp_1327_;
}
else
{
lean_object* v_a_1342_; lean_object* v___x_1344_; uint8_t v_isShared_1345_; uint8_t v_isSharedCheck_1349_; 
v_a_1342_ = lean_ctor_get(v___x_1339_, 0);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1339_);
if (v_isSharedCheck_1349_ == 0)
{
v___x_1344_ = v___x_1339_;
v_isShared_1345_ = v_isSharedCheck_1349_;
goto v_resetjp_1343_;
}
else
{
lean_inc(v_a_1342_);
lean_dec(v___x_1339_);
v___x_1344_ = lean_box(0);
v_isShared_1345_ = v_isSharedCheck_1349_;
goto v_resetjp_1343_;
}
v_resetjp_1343_:
{
lean_object* v___x_1347_; 
if (v_isShared_1345_ == 0)
{
v___x_1347_ = v___x_1344_;
goto v_reusejp_1346_;
}
else
{
lean_object* v_reuseFailAlloc_1348_; 
v_reuseFailAlloc_1348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1348_, 0, v_a_1342_);
v___x_1347_ = v_reuseFailAlloc_1348_;
goto v_reusejp_1346_;
}
v_reusejp_1346_:
{
return v___x_1347_;
}
}
}
}
v___jp_1350_:
{
lean_object* v___x_1354_; lean_object* v_a_1355_; lean_object* v___x_1356_; lean_object* v_env_1357_; lean_object* v___x_1358_; 
lean_inc(v___y_1352_);
v___x_1354_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg(v___y_1351_, v___y_1353_, v___y_1352_);
lean_dec(v___y_1353_);
v_a_1355_ = lean_ctor_get(v___x_1354_, 0);
lean_inc(v_a_1355_);
lean_dec_ref(v___x_1354_);
v___x_1356_ = lean_st_ref_get(v_a_1325_);
v_env_1357_ = lean_ctor_get(v___x_1356_, 0);
lean_inc_ref(v_env_1357_);
lean_dec(v___x_1356_);
v___x_1358_ = lean_box(0);
if (lean_obj_tag(v_a_1355_) == 0)
{
lean_dec(v___y_1351_);
v___y_1334_ = v___x_1358_;
v___y_1335_ = v_a_1355_;
v___y_1336_ = v_env_1357_;
v___y_1337_ = v_a_1324_;
v___y_1338_ = v_a_1325_;
goto v___jp_1333_;
}
else
{
lean_dec_ref(v_env_1357_);
if (lean_obj_tag(v___y_1351_) == 0)
{
lean_object* v___x_1359_; lean_object* v___x_1360_; 
v___x_1359_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__1, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__1);
v___x_1360_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_1359_, v_a_1324_, v_a_1325_);
return v___x_1360_;
}
else
{
lean_object* v_val_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; 
v_val_1361_ = lean_ctor_get(v___y_1351_, 0);
lean_inc(v_val_1361_);
lean_dec_ref_known(v___y_1351_, 1);
v___x_1362_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__3, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__3_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___closed__3);
v___x_1363_ = l_Lean_stringToMessageData(v_val_1361_);
v___x_1364_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1364_, 0, v___x_1362_);
lean_ctor_set(v___x_1364_, 1, v___x_1363_);
v___x_1365_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_1364_, v_a_1324_, v_a_1325_);
return v___x_1365_;
}
}
}
v___jp_1366_:
{
lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v_buckets_1370_; lean_object* v_decls_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; uint8_t v___x_1375_; 
v___x_1368_ = l_Lean_attributeMapRef;
v___x_1369_ = lean_st_ref_get(v___x_1368_);
v_buckets_1370_ = lean_ctor_get(v___x_1369_, 1);
lean_inc_ref(v_buckets_1370_);
lean_dec(v___x_1369_);
v_decls_1371_ = lean_box(1);
v___x_1372_ = lean_box(0);
v___x_1373_ = lean_array_get_size(v_buckets_1370_);
v___x_1374_ = lean_unsigned_to_nat(0u);
v___x_1375_ = lean_nat_dec_lt(v___x_1374_, v___x_1373_);
if (v___x_1375_ == 0)
{
lean_dec_ref(v_buckets_1370_);
v___y_1351_ = v___y_1367_;
v___y_1352_ = v_decls_1371_;
v___y_1353_ = v___x_1372_;
goto v___jp_1350_;
}
else
{
size_t v___x_1376_; size_t v___x_1377_; lean_object* v___x_1378_; 
v___x_1376_ = lean_usize_of_nat(v___x_1373_);
v___x_1377_ = ((size_t)0ULL);
v___x_1378_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__3(v_buckets_1370_, v___x_1376_, v___x_1377_, v___x_1372_);
lean_dec_ref(v_buckets_1370_);
v___y_1351_ = v___y_1367_;
v___y_1352_ = v_decls_1371_;
v___y_1353_ = v___x_1378_;
goto v___jp_1350_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr___boxed(lean_object* v_id_1391_, lean_object* v_a_1392_, lean_object* v_a_1393_, lean_object* v_a_1394_){
_start:
{
lean_object* v_res_1395_; 
v_res_1395_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(v_id_1391_, v_a_1392_, v_a_1393_);
lean_dec(v_a_1393_);
lean_dec_ref(v_a_1392_);
return v_res_1395_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0(lean_object* v___y_1396_, lean_object* v_as_1397_, lean_object* v_as_x27_1398_, lean_object* v_b_1399_, lean_object* v_a_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_){
_start:
{
lean_object* v___x_1404_; 
v___x_1404_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___redArg(v___y_1396_, v_as_x27_1398_, v_b_1399_);
return v___x_1404_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0___boxed(lean_object* v___y_1405_, lean_object* v_as_1406_, lean_object* v_as_x27_1407_, lean_object* v_b_1408_, lean_object* v_a_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_){
_start:
{
lean_object* v_res_1413_; 
v_res_1413_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__0(v___y_1405_, v_as_1406_, v_as_x27_1407_, v_b_1408_, v_a_1409_, v___y_1410_, v___y_1411_);
lean_dec(v___y_1411_);
lean_dec_ref(v___y_1410_);
lean_dec(v_as_x27_1407_);
lean_dec(v_as_1406_);
lean_dec(v___y_1405_);
return v_res_1413_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1(lean_object* v___x_1414_, lean_object* v_init_1415_, lean_object* v_x_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_){
_start:
{
lean_object* v___x_1420_; 
v___x_1420_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg(v___x_1414_, v_init_1415_, v_x_1416_, v___y_1417_);
return v___x_1420_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___boxed(lean_object* v___x_1421_, lean_object* v_init_1422_, lean_object* v_x_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_){
_start:
{
lean_object* v_res_1427_; 
v_res_1427_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1(v___x_1421_, v_init_1422_, v_x_1423_, v___y_1424_, v___y_1425_);
lean_dec(v___y_1425_);
lean_dec_ref(v___y_1424_);
return v_res_1427_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__AttrAttribute________1(lean_object* v_x_1428_, lean_object* v_a_1429_, lean_object* v_a_1430_){
_start:
{
lean_object* v___x_1432_; uint8_t v___x_1433_; 
v___x_1432_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__1));
lean_inc(v_x_1428_);
v___x_1433_ = l_Lean_Syntax_isOfKind(v_x_1428_, v___x_1432_);
if (v___x_1433_ == 0)
{
lean_object* v___x_1434_; 
lean_dec(v_x_1428_);
v___x_1434_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_1434_;
}
else
{
lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; uint8_t v___x_1438_; 
v___x_1435_ = lean_unsigned_to_nat(1u);
v___x_1436_ = l_Lean_Syntax_getArg(v_x_1428_, v___x_1435_);
v___x_1437_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__6));
lean_inc(v___x_1436_);
v___x_1438_ = l_Lean_Syntax_isOfKind(v___x_1436_, v___x_1437_);
if (v___x_1438_ == 0)
{
lean_object* v___x_1439_; uint8_t v___x_1440_; 
v___x_1439_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__AttrAttribute_______00__closed__10));
v___x_1440_ = l_Lean_Syntax_isOfKind(v___x_1436_, v___x_1439_);
if (v___x_1440_ == 0)
{
lean_object* v___x_1441_; 
lean_dec(v_x_1428_);
v___x_1441_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_1441_;
}
else
{
lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; 
v___x_1442_ = lean_unsigned_to_nat(2u);
v___x_1443_ = l_Lean_Syntax_getArg(v_x_1428_, v___x_1442_);
lean_dec(v_x_1428_);
v___x_1444_ = l_Lean_Syntax_getOptional_x3f(v___x_1443_);
lean_dec(v___x_1443_);
if (lean_obj_tag(v___x_1444_) == 0)
{
lean_object* v___x_1445_; lean_object* v___x_1446_; 
v___x_1445_ = lean_box(0);
v___x_1446_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(v___x_1445_, v_a_1429_, v_a_1430_);
return v___x_1446_;
}
else
{
lean_object* v_val_1447_; lean_object* v___x_1449_; uint8_t v_isShared_1450_; uint8_t v_isSharedCheck_1455_; 
v_val_1447_ = lean_ctor_get(v___x_1444_, 0);
v_isSharedCheck_1455_ = !lean_is_exclusive(v___x_1444_);
if (v_isSharedCheck_1455_ == 0)
{
v___x_1449_ = v___x_1444_;
v_isShared_1450_ = v_isSharedCheck_1455_;
goto v_resetjp_1448_;
}
else
{
lean_inc(v_val_1447_);
lean_dec(v___x_1444_);
v___x_1449_ = lean_box(0);
v_isShared_1450_ = v_isSharedCheck_1455_;
goto v_resetjp_1448_;
}
v_resetjp_1448_:
{
lean_object* v___x_1452_; 
if (v_isShared_1450_ == 0)
{
v___x_1452_ = v___x_1449_;
goto v_reusejp_1451_;
}
else
{
lean_object* v_reuseFailAlloc_1454_; 
v_reuseFailAlloc_1454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1454_, 0, v_val_1447_);
v___x_1452_ = v_reuseFailAlloc_1454_;
goto v_reusejp_1451_;
}
v_reusejp_1451_:
{
lean_object* v___x_1453_; 
v___x_1453_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(v___x_1452_, v_a_1429_, v_a_1430_);
return v___x_1453_;
}
}
}
}
}
else
{
lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; 
lean_dec(v___x_1436_);
v___x_1456_ = lean_unsigned_to_nat(2u);
v___x_1457_ = l_Lean_Syntax_getArg(v_x_1428_, v___x_1456_);
lean_dec(v_x_1428_);
v___x_1458_ = l_Lean_Syntax_getOptional_x3f(v___x_1457_);
lean_dec(v___x_1457_);
if (lean_obj_tag(v___x_1458_) == 0)
{
lean_object* v___x_1459_; lean_object* v___x_1460_; 
v___x_1459_ = lean_box(0);
v___x_1460_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(v___x_1459_, v_a_1429_, v_a_1430_);
return v___x_1460_;
}
else
{
lean_object* v_val_1461_; lean_object* v___x_1463_; uint8_t v_isShared_1464_; uint8_t v_isSharedCheck_1469_; 
v_val_1461_ = lean_ctor_get(v___x_1458_, 0);
v_isSharedCheck_1469_ = !lean_is_exclusive(v___x_1458_);
if (v_isSharedCheck_1469_ == 0)
{
v___x_1463_ = v___x_1458_;
v_isShared_1464_ = v_isSharedCheck_1469_;
goto v_resetjp_1462_;
}
else
{
lean_inc(v_val_1461_);
lean_dec(v___x_1458_);
v___x_1463_ = lean_box(0);
v_isShared_1464_ = v_isSharedCheck_1469_;
goto v_resetjp_1462_;
}
v_resetjp_1462_:
{
lean_object* v___x_1466_; 
if (v_isShared_1464_ == 0)
{
v___x_1466_ = v___x_1463_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v_val_1461_);
v___x_1466_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1465_;
}
v_reusejp_1465_:
{
lean_object* v___x_1467_; 
v___x_1467_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr(v___x_1466_, v_a_1429_, v_a_1430_);
return v___x_1467_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__AttrAttribute________1___boxed(lean_object* v_x_1470_, lean_object* v_a_1471_, lean_object* v_a_1472_, lean_object* v_a_1473_){
_start:
{
lean_object* v_res_1474_; 
v_res_1474_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__AttrAttribute________1(v_x_1470_, v_a_1471_, v_a_1472_);
lean_dec(v_a_1472_);
lean_dec_ref(v_a_1471_);
return v_res_1474_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___lam__0(lean_object* v___y_1500_, lean_object* v_x_1501_, lean_object* v_____s_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_){
_start:
{
lean_object* v_fst_1509_; lean_object* v_snd_1510_; uint8_t v___x_1511_; lean_object* v___x_1512_; 
v_fst_1509_ = lean_ctor_get(v_x_1501_, 0);
lean_inc(v_fst_1509_);
v_snd_1510_ = lean_ctor_get(v_x_1501_, 1);
lean_inc(v_snd_1510_);
lean_dec_ref(v_x_1501_);
v___x_1511_ = 0;
v___x_1512_ = l_Lean_Name_toString(v_fst_1509_, v___x_1511_);
if (lean_obj_tag(v___y_1500_) == 1)
{
lean_object* v_val_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; uint8_t v___x_1520_; 
v_val_1517_ = lean_ctor_get(v___y_1500_, 0);
v___x_1518_ = lean_string_utf8_byte_size(v___x_1512_);
v___x_1519_ = lean_string_utf8_byte_size(v_val_1517_);
v___x_1520_ = lean_nat_dec_le(v___x_1519_, v___x_1518_);
if (v___x_1520_ == 0)
{
lean_dec_ref(v___x_1512_);
lean_dec(v_snd_1510_);
goto v___jp_1506_;
}
else
{
lean_object* v___x_1521_; uint8_t v___x_1522_; 
v___x_1521_ = lean_unsigned_to_nat(0u);
v___x_1522_ = lean_string_memcmp(v___x_1512_, v_val_1517_, v___x_1521_, v___x_1521_, v___x_1519_);
if (v___x_1522_ == 0)
{
lean_dec_ref(v___x_1512_);
lean_dec(v_snd_1510_);
goto v___jp_1506_;
}
else
{
goto v___jp_1513_;
}
}
}
else
{
goto v___jp_1513_;
}
v___jp_1506_:
{
lean_object* v___x_1507_; lean_object* v___x_1508_; 
v___x_1507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1507_, 0, v_____s_1502_);
v___x_1508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1508_, 0, v___x_1507_);
return v___x_1508_;
}
v___jp_1513_:
{
lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1514_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v___x_1512_, v_snd_1510_, v_____s_1502_);
v___x_1515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1515_, 0, v___x_1514_);
v___x_1516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1516_, 0, v___x_1515_);
return v___x_1516_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___lam__0___boxed(lean_object* v___y_1523_, lean_object* v_x_1524_, lean_object* v_____s_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_){
_start:
{
lean_object* v_res_1529_; 
v_res_1529_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___lam__0(v___y_1523_, v_x_1524_, v_____s_1525_, v___y_1526_, v___y_1527_);
lean_dec(v___y_1527_);
lean_dec_ref(v___y_1526_);
lean_dec(v___y_1523_);
return v_res_1529_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_1532_; lean_object* v___x_1533_; 
v___x_1532_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__0));
v___x_1533_ = l_Lean_MessageData_ofFormat(v___x_1532_);
return v___x_1533_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_1535_; lean_object* v___x_1536_; 
v___x_1535_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__2));
v___x_1536_ = l_Lean_stringToMessageData(v___x_1535_);
return v___x_1536_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__5(void){
_start:
{
lean_object* v___x_1538_; lean_object* v___x_1539_; 
v___x_1538_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__4));
v___x_1539_ = l_Lean_stringToMessageData(v___x_1538_);
return v___x_1539_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7(void){
_start:
{
lean_object* v___x_1541_; lean_object* v___x_1542_; 
v___x_1541_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__6));
v___x_1542_ = l_Lean_stringToMessageData(v___x_1541_);
return v___x_1542_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg(lean_object* v___x_1543_, lean_object* v_init_1544_, lean_object* v_x_1545_, lean_object* v___y_1546_){
_start:
{
if (lean_obj_tag(v_x_1545_) == 0)
{
lean_object* v_k_1548_; lean_object* v_v_1549_; lean_object* v_l_1550_; lean_object* v_r_1551_; lean_object* v___x_1552_; 
v_k_1548_ = lean_ctor_get(v_x_1545_, 1);
lean_inc(v_k_1548_);
v_v_1549_ = lean_ctor_get(v_x_1545_, 2);
lean_inc(v_v_1549_);
v_l_1550_ = lean_ctor_get(v_x_1545_, 3);
lean_inc(v_l_1550_);
v_r_1551_ = lean_ctor_get(v_x_1545_, 4);
lean_inc(v_r_1551_);
lean_dec_ref_known(v_x_1545_, 5);
lean_inc_ref(v___x_1543_);
v___x_1552_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg(v___x_1543_, v_init_1544_, v_l_1550_, v___y_1546_);
if (lean_obj_tag(v___x_1552_) == 0)
{
lean_object* v_a_1553_; lean_object* v_a_1554_; lean_object* v___x_1556_; uint8_t v_isShared_1557_; uint8_t v_isSharedCheck_1618_; 
v_a_1553_ = lean_ctor_get(v___x_1552_, 0);
lean_inc(v_a_1553_);
lean_dec_ref_known(v___x_1552_, 1);
v_a_1554_ = lean_ctor_get(v_a_1553_, 0);
v_isSharedCheck_1618_ = !lean_is_exclusive(v_a_1553_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1556_ = v_a_1553_;
v_isShared_1557_ = v_isSharedCheck_1618_;
goto v_resetjp_1555_;
}
else
{
lean_inc(v_a_1554_);
lean_dec(v_a_1553_);
v___x_1556_ = lean_box(0);
v_isShared_1557_ = v_isSharedCheck_1618_;
goto v_resetjp_1555_;
}
v_resetjp_1555_:
{
lean_object* v_declName_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; uint8_t v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; 
v_declName_1558_ = lean_ctor_get(v_v_1549_, 0);
lean_inc_n(v_declName_1558_, 2);
lean_dec(v_v_1549_);
v___x_1559_ = lean_box(0);
v___x_1560_ = l_Lean_mkConst(v_declName_1558_, v___x_1559_);
v___x_1561_ = 1;
v___x_1562_ = l_Lean_Options_empty;
v___x_1563_ = lean_box(0);
lean_inc_ref(v___x_1543_);
v___x_1564_ = l_Lean_findDocString_x3f(v___x_1543_, v_declName_1558_, v___x_1561_, v___x_1562_, v___x_1563_, v___x_1559_);
if (lean_obj_tag(v___x_1564_) == 0)
{
lean_object* v_a_1565_; lean_object* v_msg1_1567_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; 
lean_del_object(v___x_1556_);
v_a_1565_ = lean_ctor_get(v___x_1564_, 0);
lean_inc(v_a_1565_);
lean_dec_ref_known(v___x_1564_, 1);
v___x_1574_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__3, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__3_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__3);
v___x_1575_ = l_Lean_stringToMessageData(v_k_1548_);
v___x_1576_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1574_);
lean_ctor_set(v___x_1576_, 1, v___x_1575_);
v___x_1577_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__5, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__5_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__5);
v___x_1578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1578_, 0, v___x_1576_);
lean_ctor_set(v___x_1578_, 1, v___x_1577_);
v___x_1579_ = l_Lean_MessageData_ofExpr(v___x_1560_);
v___x_1580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1580_, 0, v___x_1578_);
lean_ctor_set(v___x_1580_, 1, v___x_1579_);
v___x_1581_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7);
v___x_1582_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1582_, 0, v___x_1580_);
lean_ctor_set(v___x_1582_, 1, v___x_1581_);
if (lean_obj_tag(v_a_1565_) == 1)
{
lean_object* v_val_1583_; lean_object* v___x_1585_; uint8_t v_isShared_1586_; uint8_t v_isSharedCheck_1602_; 
v_val_1583_ = lean_ctor_get(v_a_1565_, 0);
v_isSharedCheck_1602_ = !lean_is_exclusive(v_a_1565_);
if (v_isSharedCheck_1602_ == 0)
{
v___x_1585_ = v_a_1565_;
v_isShared_1586_ = v_isSharedCheck_1602_;
goto v_resetjp_1584_;
}
else
{
lean_inc(v_val_1583_);
lean_dec(v_a_1565_);
v___x_1585_ = lean_box(0);
v_isShared_1586_ = v_isSharedCheck_1602_;
goto v_resetjp_1584_;
}
v_resetjp_1584_:
{
lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v_str_1591_; lean_object* v_startInclusive_1592_; lean_object* v_endExclusive_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1598_; 
v___x_1587_ = lean_unsigned_to_nat(0u);
v___x_1588_ = lean_string_utf8_byte_size(v_val_1583_);
v___x_1589_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1589_, 0, v_val_1583_);
lean_ctor_set(v___x_1589_, 1, v___x_1587_);
lean_ctor_set(v___x_1589_, 2, v___x_1588_);
v___x_1590_ = l_String_Slice_trimAscii(v___x_1589_);
v_str_1591_ = lean_ctor_get(v___x_1590_, 0);
lean_inc_ref(v_str_1591_);
v_startInclusive_1592_ = lean_ctor_get(v___x_1590_, 1);
lean_inc(v_startInclusive_1592_);
v_endExclusive_1593_ = lean_ctor_get(v___x_1590_, 2);
lean_inc(v_endExclusive_1593_);
lean_dec_ref(v___x_1590_);
v___x_1594_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
v___x_1595_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1595_, 0, v___x_1582_);
lean_ctor_set(v___x_1595_, 1, v___x_1594_);
v___x_1596_ = lean_string_utf8_extract_fast(v_str_1591_, v_startInclusive_1592_, v_endExclusive_1593_);
lean_dec(v_endExclusive_1593_);
lean_dec(v_startInclusive_1592_);
lean_dec_ref(v_str_1591_);
if (v_isShared_1586_ == 0)
{
lean_ctor_set_tag(v___x_1585_, 3);
lean_ctor_set(v___x_1585_, 0, v___x_1596_);
v___x_1598_ = v___x_1585_;
goto v_reusejp_1597_;
}
else
{
lean_object* v_reuseFailAlloc_1601_; 
v_reuseFailAlloc_1601_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1601_, 0, v___x_1596_);
v___x_1598_ = v_reuseFailAlloc_1601_;
goto v_reusejp_1597_;
}
v_reusejp_1597_:
{
lean_object* v___x_1599_; lean_object* v___x_1600_; 
v___x_1599_ = l_Lean_MessageData_ofFormat(v___x_1598_);
v___x_1600_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1600_, 0, v___x_1595_);
lean_ctor_set(v___x_1600_, 1, v___x_1599_);
v_msg1_1567_ = v___x_1600_;
goto v___jp_1566_;
}
}
}
else
{
lean_dec(v_a_1565_);
v_msg1_1567_ = v___x_1582_;
goto v___jp_1566_;
}
v___jp_1566_:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; 
v___x_1568_ = lean_unsigned_to_nat(2u);
v___x_1569_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1569_, 0, v___x_1568_);
lean_ctor_set(v___x_1569_, 1, v_msg1_1567_);
v___x_1570_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1570_, 0, v_a_1554_);
lean_ctor_set(v___x_1570_, 1, v___x_1569_);
v___x_1571_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1);
v___x_1572_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1572_, 0, v___x_1570_);
lean_ctor_set(v___x_1572_, 1, v___x_1571_);
v_init_1544_ = v___x_1572_;
v_x_1545_ = v_r_1551_;
goto _start;
}
}
else
{
lean_object* v_a_1603_; lean_object* v___x_1605_; uint8_t v_isShared_1606_; uint8_t v_isSharedCheck_1617_; 
lean_dec_ref(v___x_1560_);
lean_dec(v_a_1554_);
lean_dec(v_r_1551_);
lean_dec(v_k_1548_);
lean_dec_ref(v___x_1543_);
v_a_1603_ = lean_ctor_get(v___x_1564_, 0);
v_isSharedCheck_1617_ = !lean_is_exclusive(v___x_1564_);
if (v_isSharedCheck_1617_ == 0)
{
v___x_1605_ = v___x_1564_;
v_isShared_1606_ = v_isSharedCheck_1617_;
goto v_resetjp_1604_;
}
else
{
lean_inc(v_a_1603_);
lean_dec(v___x_1564_);
v___x_1605_ = lean_box(0);
v_isShared_1606_ = v_isSharedCheck_1617_;
goto v_resetjp_1604_;
}
v_resetjp_1604_:
{
lean_object* v_ref_1607_; lean_object* v___x_1608_; lean_object* v___x_1610_; 
v_ref_1607_ = lean_ctor_get(v___y_1546_, 7);
v___x_1608_ = lean_io_error_to_string(v_a_1603_);
if (v_isShared_1557_ == 0)
{
lean_ctor_set_tag(v___x_1556_, 3);
lean_ctor_set(v___x_1556_, 0, v___x_1608_);
v___x_1610_ = v___x_1556_;
goto v_reusejp_1609_;
}
else
{
lean_object* v_reuseFailAlloc_1616_; 
v_reuseFailAlloc_1616_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1616_, 0, v___x_1608_);
v___x_1610_ = v_reuseFailAlloc_1616_;
goto v_reusejp_1609_;
}
v_reusejp_1609_:
{
lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1614_; 
v___x_1611_ = l_Lean_MessageData_ofFormat(v___x_1610_);
lean_inc(v_ref_1607_);
v___x_1612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1612_, 0, v_ref_1607_);
lean_ctor_set(v___x_1612_, 1, v___x_1611_);
if (v_isShared_1606_ == 0)
{
lean_ctor_set(v___x_1605_, 0, v___x_1612_);
v___x_1614_ = v___x_1605_;
goto v_reusejp_1613_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v___x_1612_);
v___x_1614_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1613_;
}
v_reusejp_1613_:
{
return v___x_1614_;
}
}
}
}
}
}
else
{
lean_dec(v_r_1551_);
lean_dec(v_v_1549_);
lean_dec(v_k_1548_);
lean_dec_ref(v___x_1543_);
return v___x_1552_;
}
}
else
{
lean_object* v___x_1619_; lean_object* v___x_1620_; 
lean_dec_ref(v___x_1543_);
v___x_1619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1619_, 0, v_init_1544_);
v___x_1620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1620_, 0, v___x_1619_);
return v___x_1620_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___boxed(lean_object* v___x_1621_, lean_object* v_init_1622_, lean_object* v_x_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v_res_1626_; 
v_res_1626_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg(v___x_1621_, v_init_1622_, v_x_1623_, v___y_1624_);
lean_dec_ref(v___y_1624_);
return v_res_1626_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___lam__0(lean_object* v_f_1627_, lean_object* v_s_1628_, lean_object* v_a_1629_, lean_object* v_b_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_){
_start:
{
lean_object* v___x_1634_; lean_object* v___x_1635_; 
v___x_1634_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1634_, 0, v_a_1629_);
lean_ctor_set(v___x_1634_, 1, v_b_1630_);
lean_inc(v___y_1632_);
lean_inc_ref(v___y_1631_);
v___x_1635_ = lean_apply_5(v_f_1627_, v___x_1634_, v_s_1628_, v___y_1631_, v___y_1632_, lean_box(0));
if (lean_obj_tag(v___x_1635_) == 0)
{
lean_object* v_a_1636_; lean_object* v___x_1638_; uint8_t v_isShared_1639_; uint8_t v_isSharedCheck_1662_; 
v_a_1636_ = lean_ctor_get(v___x_1635_, 0);
v_isSharedCheck_1662_ = !lean_is_exclusive(v___x_1635_);
if (v_isSharedCheck_1662_ == 0)
{
v___x_1638_ = v___x_1635_;
v_isShared_1639_ = v_isSharedCheck_1662_;
goto v_resetjp_1637_;
}
else
{
lean_inc(v_a_1636_);
lean_dec(v___x_1635_);
v___x_1638_ = lean_box(0);
v_isShared_1639_ = v_isSharedCheck_1662_;
goto v_resetjp_1637_;
}
v_resetjp_1637_:
{
if (lean_obj_tag(v_a_1636_) == 0)
{
lean_object* v_a_1640_; lean_object* v___x_1642_; uint8_t v_isShared_1643_; uint8_t v_isSharedCheck_1650_; 
v_a_1640_ = lean_ctor_get(v_a_1636_, 0);
v_isSharedCheck_1650_ = !lean_is_exclusive(v_a_1636_);
if (v_isSharedCheck_1650_ == 0)
{
v___x_1642_ = v_a_1636_;
v_isShared_1643_ = v_isSharedCheck_1650_;
goto v_resetjp_1641_;
}
else
{
lean_inc(v_a_1640_);
lean_dec(v_a_1636_);
v___x_1642_ = lean_box(0);
v_isShared_1643_ = v_isSharedCheck_1650_;
goto v_resetjp_1641_;
}
v_resetjp_1641_:
{
lean_object* v___x_1645_; 
if (v_isShared_1643_ == 0)
{
v___x_1645_ = v___x_1642_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1649_; 
v_reuseFailAlloc_1649_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1649_, 0, v_a_1640_);
v___x_1645_ = v_reuseFailAlloc_1649_;
goto v_reusejp_1644_;
}
v_reusejp_1644_:
{
lean_object* v___x_1647_; 
if (v_isShared_1639_ == 0)
{
lean_ctor_set(v___x_1638_, 0, v___x_1645_);
v___x_1647_ = v___x_1638_;
goto v_reusejp_1646_;
}
else
{
lean_object* v_reuseFailAlloc_1648_; 
v_reuseFailAlloc_1648_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1648_, 0, v___x_1645_);
v___x_1647_ = v_reuseFailAlloc_1648_;
goto v_reusejp_1646_;
}
v_reusejp_1646_:
{
return v___x_1647_;
}
}
}
}
else
{
lean_object* v_a_1651_; lean_object* v___x_1653_; uint8_t v_isShared_1654_; uint8_t v_isSharedCheck_1661_; 
v_a_1651_ = lean_ctor_get(v_a_1636_, 0);
v_isSharedCheck_1661_ = !lean_is_exclusive(v_a_1636_);
if (v_isSharedCheck_1661_ == 0)
{
v___x_1653_ = v_a_1636_;
v_isShared_1654_ = v_isSharedCheck_1661_;
goto v_resetjp_1652_;
}
else
{
lean_inc(v_a_1651_);
lean_dec(v_a_1636_);
v___x_1653_ = lean_box(0);
v_isShared_1654_ = v_isSharedCheck_1661_;
goto v_resetjp_1652_;
}
v_resetjp_1652_:
{
lean_object* v___x_1656_; 
if (v_isShared_1654_ == 0)
{
v___x_1656_ = v___x_1653_;
goto v_reusejp_1655_;
}
else
{
lean_object* v_reuseFailAlloc_1660_; 
v_reuseFailAlloc_1660_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1660_, 0, v_a_1651_);
v___x_1656_ = v_reuseFailAlloc_1660_;
goto v_reusejp_1655_;
}
v_reusejp_1655_:
{
lean_object* v___x_1658_; 
if (v_isShared_1639_ == 0)
{
lean_ctor_set(v___x_1638_, 0, v___x_1656_);
v___x_1658_ = v___x_1638_;
goto v_reusejp_1657_;
}
else
{
lean_object* v_reuseFailAlloc_1659_; 
v_reuseFailAlloc_1659_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1659_, 0, v___x_1656_);
v___x_1658_ = v_reuseFailAlloc_1659_;
goto v_reusejp_1657_;
}
v_reusejp_1657_:
{
return v___x_1658_;
}
}
}
}
}
}
else
{
lean_object* v_a_1663_; lean_object* v___x_1665_; uint8_t v_isShared_1666_; uint8_t v_isSharedCheck_1670_; 
v_a_1663_ = lean_ctor_get(v___x_1635_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1635_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1665_ = v___x_1635_;
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
else
{
lean_inc(v_a_1663_);
lean_dec(v___x_1635_);
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
v_reuseFailAlloc_1669_ = lean_alloc_ctor(1, 1, 0);
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
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___lam__0___boxed(lean_object* v_f_1671_, lean_object* v_s_1672_, lean_object* v_a_1673_, lean_object* v_b_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_){
_start:
{
lean_object* v_res_1678_; 
v_res_1678_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___lam__0(v_f_1671_, v_s_1672_, v_a_1673_, v_b_1674_, v___y_1675_, v___y_1676_);
lean_dec(v___y_1676_);
lean_dec_ref(v___y_1675_);
return v_res_1678_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg(lean_object* v_f_1679_, lean_object* v_keys_1680_, lean_object* v_vals_1681_, lean_object* v_i_1682_, lean_object* v_acc_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_){
_start:
{
lean_object* v___x_1687_; uint8_t v___x_1688_; 
v___x_1687_ = lean_array_get_size(v_keys_1680_);
v___x_1688_ = lean_nat_dec_lt(v_i_1682_, v___x_1687_);
if (v___x_1688_ == 0)
{
lean_object* v___x_1689_; lean_object* v___x_1690_; 
lean_dec(v_i_1682_);
lean_dec_ref(v_f_1679_);
v___x_1689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1689_, 0, v_acc_1683_);
v___x_1690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1690_, 0, v___x_1689_);
return v___x_1690_;
}
else
{
lean_object* v_k_1691_; lean_object* v_v_1692_; lean_object* v___x_1693_; 
v_k_1691_ = lean_array_fget_borrowed(v_keys_1680_, v_i_1682_);
v_v_1692_ = lean_array_fget_borrowed(v_vals_1681_, v_i_1682_);
lean_inc_ref(v_f_1679_);
lean_inc(v___y_1685_);
lean_inc_ref(v___y_1684_);
lean_inc(v_v_1692_);
lean_inc(v_k_1691_);
v___x_1693_ = lean_apply_6(v_f_1679_, v_acc_1683_, v_k_1691_, v_v_1692_, v___y_1684_, v___y_1685_, lean_box(0));
if (lean_obj_tag(v___x_1693_) == 0)
{
lean_object* v_a_1694_; 
v_a_1694_ = lean_ctor_get(v___x_1693_, 0);
lean_inc(v_a_1694_);
if (lean_obj_tag(v_a_1694_) == 0)
{
lean_dec_ref_known(v_a_1694_, 1);
lean_dec(v_i_1682_);
lean_dec_ref(v_f_1679_);
return v___x_1693_;
}
else
{
lean_object* v_a_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; 
lean_dec_ref_known(v___x_1693_, 1);
v_a_1695_ = lean_ctor_get(v_a_1694_, 0);
lean_inc(v_a_1695_);
lean_dec_ref_known(v_a_1694_, 1);
v___x_1696_ = lean_unsigned_to_nat(1u);
v___x_1697_ = lean_nat_add(v_i_1682_, v___x_1696_);
lean_dec(v_i_1682_);
v_i_1682_ = v___x_1697_;
v_acc_1683_ = v_a_1695_;
goto _start;
}
}
else
{
lean_dec(v_i_1682_);
lean_dec_ref(v_f_1679_);
return v___x_1693_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_f_1699_, lean_object* v_keys_1700_, lean_object* v_vals_1701_, lean_object* v_i_1702_, lean_object* v_acc_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_){
_start:
{
lean_object* v_res_1707_; 
v_res_1707_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg(v_f_1699_, v_keys_1700_, v_vals_1701_, v_i_1702_, v_acc_1703_, v___y_1704_, v___y_1705_);
lean_dec(v___y_1705_);
lean_dec_ref(v___y_1704_);
lean_dec_ref(v_vals_1701_);
lean_dec_ref(v_keys_1700_);
return v_res_1707_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(lean_object* v_f_1708_, lean_object* v_x_1709_, lean_object* v_x_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_){
_start:
{
if (lean_obj_tag(v_x_1709_) == 0)
{
lean_object* v_es_1714_; lean_object* v___x_1716_; uint8_t v_isShared_1717_; uint8_t v_isSharedCheck_1736_; 
v_es_1714_ = lean_ctor_get(v_x_1709_, 0);
v_isSharedCheck_1736_ = !lean_is_exclusive(v_x_1709_);
if (v_isSharedCheck_1736_ == 0)
{
v___x_1716_ = v_x_1709_;
v_isShared_1717_ = v_isSharedCheck_1736_;
goto v_resetjp_1715_;
}
else
{
lean_inc(v_es_1714_);
lean_dec(v_x_1709_);
v___x_1716_ = lean_box(0);
v_isShared_1717_ = v_isSharedCheck_1736_;
goto v_resetjp_1715_;
}
v_resetjp_1715_:
{
lean_object* v___x_1718_; lean_object* v___x_1719_; uint8_t v___x_1720_; 
v___x_1718_ = lean_unsigned_to_nat(0u);
v___x_1719_ = lean_array_get_size(v_es_1714_);
v___x_1720_ = lean_nat_dec_lt(v___x_1718_, v___x_1719_);
if (v___x_1720_ == 0)
{
lean_object* v___x_1722_; 
lean_dec_ref(v_es_1714_);
lean_dec_ref(v_f_1708_);
if (v_isShared_1717_ == 0)
{
lean_ctor_set_tag(v___x_1716_, 1);
lean_ctor_set(v___x_1716_, 0, v_x_1710_);
v___x_1722_ = v___x_1716_;
goto v_reusejp_1721_;
}
else
{
lean_object* v_reuseFailAlloc_1724_; 
v_reuseFailAlloc_1724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1724_, 0, v_x_1710_);
v___x_1722_ = v_reuseFailAlloc_1724_;
goto v_reusejp_1721_;
}
v_reusejp_1721_:
{
lean_object* v___x_1723_; 
v___x_1723_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1723_, 0, v___x_1722_);
return v___x_1723_;
}
}
else
{
uint8_t v___x_1725_; 
v___x_1725_ = lean_nat_dec_le(v___x_1719_, v___x_1719_);
if (v___x_1725_ == 0)
{
if (v___x_1720_ == 0)
{
lean_object* v___x_1727_; 
lean_dec_ref(v_es_1714_);
lean_dec_ref(v_f_1708_);
if (v_isShared_1717_ == 0)
{
lean_ctor_set_tag(v___x_1716_, 1);
lean_ctor_set(v___x_1716_, 0, v_x_1710_);
v___x_1727_ = v___x_1716_;
goto v_reusejp_1726_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v_x_1710_);
v___x_1727_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1726_;
}
v_reusejp_1726_:
{
lean_object* v___x_1728_; 
v___x_1728_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1728_, 0, v___x_1727_);
return v___x_1728_;
}
}
else
{
size_t v___x_1730_; size_t v___x_1731_; lean_object* v___x_1732_; 
lean_del_object(v___x_1716_);
v___x_1730_ = ((size_t)0ULL);
v___x_1731_ = lean_usize_of_nat(v___x_1719_);
v___x_1732_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg(v_f_1708_, v_es_1714_, v___x_1730_, v___x_1731_, v_x_1710_, v___y_1711_, v___y_1712_);
lean_dec_ref(v_es_1714_);
return v___x_1732_;
}
}
else
{
size_t v___x_1733_; size_t v___x_1734_; lean_object* v___x_1735_; 
lean_del_object(v___x_1716_);
v___x_1733_ = ((size_t)0ULL);
v___x_1734_ = lean_usize_of_nat(v___x_1719_);
v___x_1735_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg(v_f_1708_, v_es_1714_, v___x_1733_, v___x_1734_, v_x_1710_, v___y_1711_, v___y_1712_);
lean_dec_ref(v_es_1714_);
return v___x_1735_;
}
}
}
}
else
{
lean_object* v_ks_1737_; lean_object* v_vs_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; 
v_ks_1737_ = lean_ctor_get(v_x_1709_, 0);
lean_inc_ref(v_ks_1737_);
v_vs_1738_ = lean_ctor_get(v_x_1709_, 1);
lean_inc_ref(v_vs_1738_);
lean_dec_ref_known(v_x_1709_, 2);
v___x_1739_ = lean_unsigned_to_nat(0u);
v___x_1740_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg(v_f_1708_, v_ks_1737_, v_vs_1738_, v___x_1739_, v_x_1710_, v___y_1711_, v___y_1712_);
lean_dec_ref(v_vs_1738_);
lean_dec_ref(v_ks_1737_);
return v___x_1740_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_f_1741_, lean_object* v_as_1742_, size_t v_i_1743_, size_t v_stop_1744_, lean_object* v_b_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_){
_start:
{
lean_object* v_a_1750_; lean_object* v___y_1755_; uint8_t v___x_1758_; 
v___x_1758_ = lean_usize_dec_eq(v_i_1743_, v_stop_1744_);
if (v___x_1758_ == 0)
{
lean_object* v___x_1759_; 
v___x_1759_ = lean_array_uget_borrowed(v_as_1742_, v_i_1743_);
switch(lean_obj_tag(v___x_1759_))
{
case 0:
{
lean_object* v_key_1760_; lean_object* v_val_1761_; lean_object* v___x_1762_; 
v_key_1760_ = lean_ctor_get(v___x_1759_, 0);
v_val_1761_ = lean_ctor_get(v___x_1759_, 1);
lean_inc_ref(v_f_1741_);
lean_inc(v___y_1747_);
lean_inc_ref(v___y_1746_);
lean_inc(v_val_1761_);
lean_inc(v_key_1760_);
v___x_1762_ = lean_apply_6(v_f_1741_, v_b_1745_, v_key_1760_, v_val_1761_, v___y_1746_, v___y_1747_, lean_box(0));
v___y_1755_ = v___x_1762_;
goto v___jp_1754_;
}
case 1:
{
lean_object* v_node_1763_; lean_object* v___x_1764_; 
v_node_1763_ = lean_ctor_get(v___x_1759_, 0);
lean_inc(v_node_1763_);
lean_inc_ref(v_f_1741_);
v___x_1764_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(v_f_1741_, v_node_1763_, v_b_1745_, v___y_1746_, v___y_1747_);
v___y_1755_ = v___x_1764_;
goto v___jp_1754_;
}
default: 
{
v_a_1750_ = v_b_1745_;
goto v___jp_1749_;
}
}
}
else
{
lean_object* v___x_1765_; lean_object* v___x_1766_; 
lean_dec_ref(v_f_1741_);
v___x_1765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1765_, 0, v_b_1745_);
v___x_1766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1766_, 0, v___x_1765_);
return v___x_1766_;
}
v___jp_1749_:
{
size_t v___x_1751_; size_t v___x_1752_; 
v___x_1751_ = ((size_t)1ULL);
v___x_1752_ = lean_usize_add(v_i_1743_, v___x_1751_);
v_i_1743_ = v___x_1752_;
v_b_1745_ = v_a_1750_;
goto _start;
}
v___jp_1754_:
{
if (lean_obj_tag(v___y_1755_) == 0)
{
lean_object* v_a_1756_; 
v_a_1756_ = lean_ctor_get(v___y_1755_, 0);
if (lean_obj_tag(v_a_1756_) == 0)
{
lean_dec_ref(v_f_1741_);
return v___y_1755_;
}
else
{
lean_object* v_a_1757_; 
lean_inc_ref(v_a_1756_);
lean_dec_ref_known(v___y_1755_, 1);
v_a_1757_ = lean_ctor_get(v_a_1756_, 0);
lean_inc(v_a_1757_);
lean_dec_ref_known(v_a_1756_, 1);
v_a_1750_ = v_a_1757_;
goto v___jp_1749_;
}
}
else
{
lean_dec_ref(v_f_1741_);
return v___y_1755_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_f_1767_, lean_object* v_as_1768_, lean_object* v_i_1769_, lean_object* v_stop_1770_, lean_object* v_b_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_){
_start:
{
size_t v_i_boxed_1775_; size_t v_stop_boxed_1776_; lean_object* v_res_1777_; 
v_i_boxed_1775_ = lean_unbox_usize(v_i_1769_);
lean_dec(v_i_1769_);
v_stop_boxed_1776_ = lean_unbox_usize(v_stop_1770_);
lean_dec(v_stop_1770_);
v_res_1777_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg(v_f_1767_, v_as_1768_, v_i_boxed_1775_, v_stop_boxed_1776_, v_b_1771_, v___y_1772_, v___y_1773_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec_ref(v_as_1768_);
return v_res_1777_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_f_1778_, lean_object* v_x_1779_, lean_object* v_x_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_){
_start:
{
lean_object* v_res_1784_; 
v_res_1784_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(v_f_1778_, v_x_1779_, v_x_1780_, v___y_1781_, v___y_1782_);
lean_dec(v___y_1782_);
lean_dec_ref(v___y_1781_);
return v_res_1784_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg(lean_object* v_map_1785_, lean_object* v_init_1786_, lean_object* v_f_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_){
_start:
{
lean_object* v___f_1791_; lean_object* v___x_1792_; 
v___f_1791_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1791_, 0, v_f_1787_);
lean_inc_ref(v_map_1785_);
v___x_1792_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(v___f_1791_, v_map_1785_, v_init_1786_, v___y_1788_, v___y_1789_);
if (lean_obj_tag(v___x_1792_) == 0)
{
lean_object* v_a_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1801_; 
v_a_1793_ = lean_ctor_get(v___x_1792_, 0);
v_isSharedCheck_1801_ = !lean_is_exclusive(v___x_1792_);
if (v_isSharedCheck_1801_ == 0)
{
v___x_1795_ = v___x_1792_;
v_isShared_1796_ = v_isSharedCheck_1801_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_a_1793_);
lean_dec(v___x_1792_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1801_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v_a_1797_; lean_object* v___x_1799_; 
v_a_1797_ = lean_ctor_get(v_a_1793_, 0);
lean_inc(v_a_1797_);
lean_dec(v_a_1793_);
if (v_isShared_1796_ == 0)
{
lean_ctor_set(v___x_1795_, 0, v_a_1797_);
v___x_1799_ = v___x_1795_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v_a_1797_);
v___x_1799_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
return v___x_1799_;
}
}
}
else
{
lean_object* v_a_1802_; lean_object* v___x_1804_; uint8_t v_isShared_1805_; uint8_t v_isSharedCheck_1809_; 
v_a_1802_ = lean_ctor_get(v___x_1792_, 0);
v_isSharedCheck_1809_ = !lean_is_exclusive(v___x_1792_);
if (v_isSharedCheck_1809_ == 0)
{
v___x_1804_ = v___x_1792_;
v_isShared_1805_ = v_isSharedCheck_1809_;
goto v_resetjp_1803_;
}
else
{
lean_inc(v_a_1802_);
lean_dec(v___x_1792_);
v___x_1804_ = lean_box(0);
v_isShared_1805_ = v_isSharedCheck_1809_;
goto v_resetjp_1803_;
}
v_resetjp_1803_:
{
lean_object* v___x_1807_; 
if (v_isShared_1805_ == 0)
{
v___x_1807_ = v___x_1804_;
goto v_reusejp_1806_;
}
else
{
lean_object* v_reuseFailAlloc_1808_; 
v_reuseFailAlloc_1808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1808_, 0, v_a_1802_);
v___x_1807_ = v_reuseFailAlloc_1808_;
goto v_reusejp_1806_;
}
v_reusejp_1806_:
{
return v___x_1807_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg___boxed(lean_object* v_map_1810_, lean_object* v_init_1811_, lean_object* v_f_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_){
_start:
{
lean_object* v_res_1816_; 
v_res_1816_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg(v_map_1810_, v_init_1811_, v_f_1812_, v___y_1813_, v___y_1814_);
lean_dec(v___y_1814_);
lean_dec_ref(v___y_1813_);
lean_dec_ref(v_map_1810_);
return v_res_1816_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__1(void){
_start:
{
lean_object* v___x_1818_; lean_object* v___x_1819_; 
v___x_1818_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__0));
v___x_1819_ = l_Lean_stringToMessageData(v___x_1818_);
return v___x_1819_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__3(void){
_start:
{
lean_object* v___x_1821_; lean_object* v___x_1822_; 
v___x_1821_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__2));
v___x_1822_ = l_Lean_stringToMessageData(v___x_1821_);
return v___x_1822_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats(lean_object* v_id_1823_, lean_object* v_a_1824_, lean_object* v_a_1825_){
_start:
{
lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1830_; lean_object* v___y_1831_; lean_object* v___y_1832_; lean_object* v___x_1845_; lean_object* v___y_1847_; 
v___x_1845_ = l_Lean_Parser_ParserExtension_instInhabitedState_default;
if (lean_obj_tag(v_id_1823_) == 0)
{
lean_object* v___x_1878_; 
v___x_1878_ = lean_box(0);
v___y_1847_ = v___x_1878_;
goto v___jp_1846_;
}
else
{
lean_object* v_val_1879_; lean_object* v___x_1881_; uint8_t v_isShared_1882_; uint8_t v_isSharedCheck_1889_; 
v_val_1879_ = lean_ctor_get(v_id_1823_, 0);
v_isSharedCheck_1889_ = !lean_is_exclusive(v_id_1823_);
if (v_isSharedCheck_1889_ == 0)
{
v___x_1881_ = v_id_1823_;
v_isShared_1882_ = v_isSharedCheck_1889_;
goto v_resetjp_1880_;
}
else
{
lean_inc(v_val_1879_);
lean_dec(v_id_1823_);
v___x_1881_ = lean_box(0);
v_isShared_1882_ = v_isSharedCheck_1889_;
goto v_resetjp_1880_;
}
v_resetjp_1880_:
{
lean_object* v___x_1883_; uint8_t v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1887_; 
v___x_1883_ = l_Lean_Syntax_getId(v_val_1879_);
lean_dec(v_val_1879_);
v___x_1884_ = 0;
v___x_1885_ = l_Lean_Name_toString(v___x_1883_, v___x_1884_);
if (v_isShared_1882_ == 0)
{
lean_ctor_set(v___x_1881_, 0, v___x_1885_);
v___x_1887_ = v___x_1881_;
goto v_reusejp_1886_;
}
else
{
lean_object* v_reuseFailAlloc_1888_; 
v_reuseFailAlloc_1888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1888_, 0, v___x_1885_);
v___x_1887_ = v_reuseFailAlloc_1888_;
goto v_reusejp_1886_;
}
v_reusejp_1886_:
{
v___y_1847_ = v___x_1887_;
goto v___jp_1846_;
}
}
}
v___jp_1827_:
{
lean_object* v___x_1833_; 
lean_inc_ref(v___y_1830_);
v___x_1833_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg(v___y_1829_, v___y_1830_, v___y_1828_, v___y_1831_);
if (lean_obj_tag(v___x_1833_) == 0)
{
lean_object* v_a_1834_; lean_object* v_a_1835_; lean_object* v___x_1836_; 
v_a_1834_ = lean_ctor_get(v___x_1833_, 0);
lean_inc(v_a_1834_);
lean_dec_ref_known(v___x_1833_, 1);
v_a_1835_ = lean_ctor_get(v_a_1834_, 0);
lean_inc(v_a_1835_);
lean_dec(v_a_1834_);
v___x_1836_ = lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(v_a_1835_, v___y_1831_, v___y_1832_);
return v___x_1836_;
}
else
{
lean_object* v_a_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1844_; 
v_a_1837_ = lean_ctor_get(v___x_1833_, 0);
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1833_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1839_ = v___x_1833_;
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_a_1837_);
lean_dec(v___x_1833_);
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
v___jp_1846_:
{
lean_object* v___x_1848_; lean_object* v_env_1849_; lean_object* v___x_1850_; lean_object* v_ext_1851_; lean_object* v_toEnvExtension_1852_; lean_object* v_asyncMode_1853_; lean_object* v___x_1854_; lean_object* v_categories_1855_; lean_object* v___f_1856_; lean_object* v_decls_1857_; lean_object* v___x_1858_; 
v___x_1848_ = lean_st_ref_get(v_a_1825_);
v_env_1849_ = lean_ctor_get(v___x_1848_, 0);
lean_inc_ref(v_env_1849_);
lean_dec(v___x_1848_);
v___x_1850_ = l_Lean_Parser_parserExtension;
v_ext_1851_ = lean_ctor_get(v___x_1850_, 1);
v_toEnvExtension_1852_ = lean_ctor_get(v_ext_1851_, 0);
v_asyncMode_1853_ = lean_ctor_get(v_toEnvExtension_1852_, 2);
v___x_1854_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1845_, v___x_1850_, v_env_1849_, v_asyncMode_1853_);
v_categories_1855_ = lean_ctor_get(v___x_1854_, 2);
lean_inc_ref(v_categories_1855_);
lean_dec(v___x_1854_);
lean_inc(v___y_1847_);
v___f_1856_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___lam__0___boxed), 6, 1);
lean_closure_set(v___f_1856_, 0, v___y_1847_);
v_decls_1857_ = lean_box(1);
v___x_1858_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg(v_categories_1855_, v_decls_1857_, v___f_1856_, v_a_1824_, v_a_1825_);
lean_dec_ref(v_categories_1855_);
if (lean_obj_tag(v___x_1858_) == 0)
{
lean_object* v_a_1859_; lean_object* v___x_1860_; lean_object* v_env_1861_; lean_object* v___x_1862_; 
v_a_1859_ = lean_ctor_get(v___x_1858_, 0);
lean_inc(v_a_1859_);
lean_dec_ref_known(v___x_1858_, 1);
v___x_1860_ = lean_st_ref_get(v_a_1825_);
v_env_1861_ = lean_ctor_get(v___x_1860_, 0);
lean_inc_ref(v_env_1861_);
lean_dec(v___x_1860_);
v___x_1862_ = l_Lean_MessageData_nil;
if (lean_obj_tag(v_a_1859_) == 0)
{
lean_dec(v___y_1847_);
v___y_1828_ = v_a_1859_;
v___y_1829_ = v_env_1861_;
v___y_1830_ = v___x_1862_;
v___y_1831_ = v_a_1824_;
v___y_1832_ = v_a_1825_;
goto v___jp_1827_;
}
else
{
lean_dec_ref(v_env_1861_);
if (lean_obj_tag(v___y_1847_) == 0)
{
lean_object* v___x_1863_; lean_object* v___x_1864_; 
v___x_1863_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__1, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__1);
v___x_1864_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_1863_, v_a_1824_, v_a_1825_);
return v___x_1864_;
}
else
{
lean_object* v_val_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; 
v_val_1865_ = lean_ctor_get(v___y_1847_, 0);
lean_inc(v_val_1865_);
lean_dec_ref_known(v___y_1847_, 1);
v___x_1866_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__3, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__3_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___closed__3);
v___x_1867_ = l_Lean_stringToMessageData(v_val_1865_);
v___x_1868_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1868_, 0, v___x_1866_);
lean_ctor_set(v___x_1868_, 1, v___x_1867_);
v___x_1869_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_1868_, v_a_1824_, v_a_1825_);
return v___x_1869_;
}
}
}
else
{
lean_object* v_a_1870_; lean_object* v___x_1872_; uint8_t v_isShared_1873_; uint8_t v_isSharedCheck_1877_; 
lean_dec(v___y_1847_);
v_a_1870_ = lean_ctor_get(v___x_1858_, 0);
v_isSharedCheck_1877_ = !lean_is_exclusive(v___x_1858_);
if (v_isSharedCheck_1877_ == 0)
{
v___x_1872_ = v___x_1858_;
v_isShared_1873_ = v_isSharedCheck_1877_;
goto v_resetjp_1871_;
}
else
{
lean_inc(v_a_1870_);
lean_dec(v___x_1858_);
v___x_1872_ = lean_box(0);
v_isShared_1873_ = v_isSharedCheck_1877_;
goto v_resetjp_1871_;
}
v_resetjp_1871_:
{
lean_object* v___x_1875_; 
if (v_isShared_1873_ == 0)
{
v___x_1875_ = v___x_1872_;
goto v_reusejp_1874_;
}
else
{
lean_object* v_reuseFailAlloc_1876_; 
v_reuseFailAlloc_1876_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1876_, 0, v_a_1870_);
v___x_1875_ = v_reuseFailAlloc_1876_;
goto v_reusejp_1874_;
}
v_reusejp_1874_:
{
return v___x_1875_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats___boxed(lean_object* v_id_1890_, lean_object* v_a_1891_, lean_object* v_a_1892_, lean_object* v_a_1893_){
_start:
{
lean_object* v_res_1894_; 
v_res_1894_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats(v_id_1890_, v_a_1891_, v_a_1892_);
lean_dec(v_a_1892_);
lean_dec_ref(v_a_1891_);
return v_res_1894_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0(lean_object* v_00_u03c3_1895_, lean_object* v_00_u03b2_1896_, lean_object* v_map_1897_, lean_object* v_init_1898_, lean_object* v_f_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_){
_start:
{
lean_object* v___x_1903_; 
v___x_1903_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg(v_map_1897_, v_init_1898_, v_f_1899_, v___y_1900_, v___y_1901_);
return v___x_1903_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___boxed(lean_object* v_00_u03c3_1904_, lean_object* v_00_u03b2_1905_, lean_object* v_map_1906_, lean_object* v_init_1907_, lean_object* v_f_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_){
_start:
{
lean_object* v_res_1912_; 
v_res_1912_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0(v_00_u03c3_1904_, v_00_u03b2_1905_, v_map_1906_, v_init_1907_, v_f_1908_, v___y_1909_, v___y_1910_);
lean_dec(v___y_1910_);
lean_dec_ref(v___y_1909_);
lean_dec_ref(v_map_1906_);
return v_res_1912_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1(lean_object* v___x_1913_, lean_object* v_init_1914_, lean_object* v_x_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v___x_1919_; 
v___x_1919_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg(v___x_1913_, v_init_1914_, v_x_1915_, v___y_1916_);
return v___x_1919_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___boxed(lean_object* v___x_1920_, lean_object* v_init_1921_, lean_object* v_x_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_){
_start:
{
lean_object* v_res_1926_; 
v_res_1926_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1(v___x_1920_, v_init_1921_, v_x_1922_, v___y_1923_, v___y_1924_);
lean_dec(v___y_1924_);
lean_dec_ref(v___y_1923_);
return v_res_1926_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___redArg(lean_object* v_map_1927_, lean_object* v_f_1928_, lean_object* v_init_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_){
_start:
{
lean_object* v___x_1933_; 
v___x_1933_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(v_f_1928_, v_map_1927_, v_init_1929_, v___y_1930_, v___y_1931_);
return v___x_1933_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___redArg___boxed(lean_object* v_map_1934_, lean_object* v_f_1935_, lean_object* v_init_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_){
_start:
{
lean_object* v_res_1940_; 
v_res_1940_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___redArg(v_map_1934_, v_f_1935_, v_init_1936_, v___y_1937_, v___y_1938_);
lean_dec(v___y_1938_);
lean_dec_ref(v___y_1937_);
return v_res_1940_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0(lean_object* v_00_u03c3_1941_, lean_object* v_00_u03c3_1942_, lean_object* v_00_u03b2_1943_, lean_object* v_map_1944_, lean_object* v_f_1945_, lean_object* v_init_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_){
_start:
{
lean_object* v___x_1950_; 
v___x_1950_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(v_f_1945_, v_map_1944_, v_init_1946_, v___y_1947_, v___y_1948_);
return v___x_1950_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0___boxed(lean_object* v_00_u03c3_1951_, lean_object* v_00_u03c3_1952_, lean_object* v_00_u03b2_1953_, lean_object* v_map_1954_, lean_object* v_f_1955_, lean_object* v_init_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_){
_start:
{
lean_object* v_res_1960_; 
v_res_1960_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0(v_00_u03c3_1951_, v_00_u03c3_1952_, v_00_u03b2_1953_, v_map_1954_, v_f_1955_, v_init_1956_, v___y_1957_, v___y_1958_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
return v_res_1960_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1(lean_object* v_00_u03c3_1961_, lean_object* v_00_u03c3_1962_, lean_object* v_00_u03b1_1963_, lean_object* v_00_u03b2_1964_, lean_object* v_f_1965_, lean_object* v_x_1966_, lean_object* v_x_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_){
_start:
{
lean_object* v___x_1971_; 
v___x_1971_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___redArg(v_f_1965_, v_x_1966_, v_x_1967_, v___y_1968_, v___y_1969_);
return v___x_1971_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03c3_1972_, lean_object* v_00_u03c3_1973_, lean_object* v_00_u03b1_1974_, lean_object* v_00_u03b2_1975_, lean_object* v_f_1976_, lean_object* v_x_1977_, lean_object* v_x_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
lean_object* v_res_1982_; 
v_res_1982_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1(v_00_u03c3_1972_, v_00_u03c3_1973_, v_00_u03b1_1974_, v_00_u03b2_1975_, v_f_1976_, v_x_1977_, v_x_1978_, v___y_1979_, v___y_1980_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
return v_res_1982_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b1_1983_, lean_object* v_00_u03b2_1984_, lean_object* v_00_u03c3_1985_, lean_object* v_00_u03c3_1986_, lean_object* v_f_1987_, lean_object* v_as_1988_, size_t v_i_1989_, size_t v_stop_1990_, lean_object* v_b_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_){
_start:
{
lean_object* v___x_1995_; 
v___x_1995_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___redArg(v_f_1987_, v_as_1988_, v_i_1989_, v_stop_1990_, v_b_1991_, v___y_1992_, v___y_1993_);
return v___x_1995_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b1_1996_, lean_object* v_00_u03b2_1997_, lean_object* v_00_u03c3_1998_, lean_object* v_00_u03c3_1999_, lean_object* v_f_2000_, lean_object* v_as_2001_, lean_object* v_i_2002_, lean_object* v_stop_2003_, lean_object* v_b_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_){
_start:
{
size_t v_i_boxed_2008_; size_t v_stop_boxed_2009_; lean_object* v_res_2010_; 
v_i_boxed_2008_ = lean_unbox_usize(v_i_2002_);
lean_dec(v_i_2002_);
v_stop_boxed_2009_ = lean_unbox_usize(v_stop_2003_);
lean_dec(v_stop_2003_);
v_res_2010_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__3(v_00_u03b1_1996_, v_00_u03b2_1997_, v_00_u03c3_1998_, v_00_u03c3_1999_, v_f_2000_, v_as_2001_, v_i_boxed_2008_, v_stop_boxed_2009_, v_b_2004_, v___y_2005_, v___y_2006_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec_ref(v_as_2001_);
return v_res_2010_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4(lean_object* v_00_u03c3_2011_, lean_object* v_00_u03c3_2012_, lean_object* v_00_u03b1_2013_, lean_object* v_00_u03b2_2014_, lean_object* v_f_2015_, lean_object* v_keys_2016_, lean_object* v_vals_2017_, lean_object* v_heq_2018_, lean_object* v_i_2019_, lean_object* v_acc_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_){
_start:
{
lean_object* v___x_2024_; 
v___x_2024_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___redArg(v_f_2015_, v_keys_2016_, v_vals_2017_, v_i_2019_, v_acc_2020_, v___y_2021_, v___y_2022_);
return v___x_2024_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_00_u03c3_2025_, lean_object* v_00_u03c3_2026_, lean_object* v_00_u03b1_2027_, lean_object* v_00_u03b2_2028_, lean_object* v_f_2029_, lean_object* v_keys_2030_, lean_object* v_vals_2031_, lean_object* v_heq_2032_, lean_object* v_i_2033_, lean_object* v_acc_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_, lean_object* v___y_2037_){
_start:
{
lean_object* v_res_2038_; 
v_res_2038_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0_spec__0_spec__1_spec__4(v_00_u03c3_2025_, v_00_u03c3_2026_, v_00_u03b1_2027_, v_00_u03b2_2028_, v_f_2029_, v_keys_2030_, v_vals_2031_, v_heq_2032_, v_i_2033_, v_acc_2034_, v___y_2035_, v___y_2036_);
lean_dec(v___y_2036_);
lean_dec_ref(v___y_2035_);
lean_dec_ref(v_vals_2031_);
lean_dec_ref(v_keys_2030_);
return v_res_2038_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cats________1(lean_object* v_x_2039_, lean_object* v_a_2040_, lean_object* v_a_2041_){
_start:
{
lean_object* v___x_2043_; uint8_t v___x_2044_; 
v___x_2043_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cats_______00__closed__1));
lean_inc(v_x_2039_);
v___x_2044_ = l_Lean_Syntax_isOfKind(v_x_2039_, v___x_2043_);
if (v___x_2044_ == 0)
{
lean_object* v___x_2045_; 
lean_dec(v_x_2039_);
v___x_2045_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_2045_;
}
else
{
lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; 
v___x_2046_ = lean_unsigned_to_nat(2u);
v___x_2047_ = l_Lean_Syntax_getArg(v_x_2039_, v___x_2046_);
lean_dec(v_x_2039_);
v___x_2048_ = l_Lean_Syntax_getOptional_x3f(v___x_2047_);
lean_dec(v___x_2047_);
if (lean_obj_tag(v___x_2048_) == 0)
{
lean_object* v___x_2049_; lean_object* v___x_2050_; 
v___x_2049_ = lean_box(0);
v___x_2050_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats(v___x_2049_, v_a_2040_, v_a_2041_);
return v___x_2050_;
}
else
{
lean_object* v_val_2051_; lean_object* v___x_2053_; uint8_t v_isShared_2054_; uint8_t v_isSharedCheck_2059_; 
v_val_2051_ = lean_ctor_get(v___x_2048_, 0);
v_isSharedCheck_2059_ = !lean_is_exclusive(v___x_2048_);
if (v_isSharedCheck_2059_ == 0)
{
v___x_2053_ = v___x_2048_;
v_isShared_2054_ = v_isSharedCheck_2059_;
goto v_resetjp_2052_;
}
else
{
lean_inc(v_val_2051_);
lean_dec(v___x_2048_);
v___x_2053_ = lean_box(0);
v_isShared_2054_ = v_isSharedCheck_2059_;
goto v_resetjp_2052_;
}
v_resetjp_2052_:
{
lean_object* v___x_2056_; 
if (v_isShared_2054_ == 0)
{
v___x_2056_ = v___x_2053_;
goto v_reusejp_2055_;
}
else
{
lean_object* v_reuseFailAlloc_2058_; 
v_reuseFailAlloc_2058_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2058_, 0, v_val_2051_);
v___x_2056_ = v_reuseFailAlloc_2058_;
goto v_reusejp_2055_;
}
v_reusejp_2055_:
{
lean_object* v___x_2057_; 
v___x_2057_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats(v___x_2056_, v_a_2040_, v_a_2041_);
return v___x_2057_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cats________1___boxed(lean_object* v_x_2060_, lean_object* v_a_2061_, lean_object* v_a_2062_, lean_object* v_a_2063_){
_start:
{
lean_object* v_res_2064_; 
v_res_2064_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cats________1(v_x_2060_, v_a_2061_, v_a_2062_);
lean_dec(v_a_2062_);
lean_dec_ref(v_a_2061_);
return v_res_2064_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_tokensToList(lean_object* v_tks_2129_){
_start:
{
switch(lean_obj_tag(v_tks_2129_))
{
case 2:
{
lean_object* v_a_2130_; 
v_a_2130_ = lean_ctor_get(v_tks_2129_, 0);
lean_inc(v_a_2130_);
return v_a_2130_;
}
case 3:
{
lean_object* v_a_2131_; 
v_a_2131_ = lean_ctor_get(v_tks_2129_, 0);
lean_inc(v_a_2131_);
return v_a_2131_;
}
default: 
{
lean_object* v___x_2132_; 
v___x_2132_ = lean_box(0);
return v___x_2132_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_tokensToList___boxed(lean_object* v_tks_2133_){
_start:
{
lean_object* v_res_2134_; 
v_res_2134_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_tokensToList(v_tks_2133_);
lean_dec(v_tks_2133_);
return v_res_2134_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3(lean_object* v_as_2136_){
_start:
{
lean_object* v___f_2137_; lean_object* v___x_2138_; 
v___f_2137_ = ((lean_object*)(lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3___closed__0));
v___x_2138_ = l_List_eraseDupsBy___redArg(v___f_2137_, v_as_2136_);
return v___x_2138_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__8(lean_object* v_msg_2139_){
_start:
{
lean_object* v___x_2140_; lean_object* v___x_2141_; 
v___x_2140_ = ((lean_object*)(lp_batteries_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0_spec__1___closed__0));
v___x_2141_ = lean_panic_fn_borrowed(v___x_2140_, v_msg_2139_);
return v___x_2141_;
}
}
LEAN_EXPORT lean_object* lp_batteries_minOn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__9___redArg(lean_object* v_f_2142_, lean_object* v_x_2143_, lean_object* v_y_2144_){
_start:
{
lean_object* v___x_2145_; lean_object* v___x_2146_; uint8_t v___x_2147_; 
lean_inc_ref(v_f_2142_);
lean_inc(v_x_2143_);
v___x_2145_ = lean_apply_1(v_f_2142_, v_x_2143_);
lean_inc(v_y_2144_);
v___x_2146_ = lean_apply_1(v_f_2142_, v_y_2144_);
v___x_2147_ = lean_nat_dec_le(v___x_2145_, v___x_2146_);
lean_dec(v___x_2146_);
lean_dec(v___x_2145_);
if (v___x_2147_ == 0)
{
lean_dec(v_x_2143_);
return v_y_2144_;
}
else
{
lean_dec(v_y_2144_);
return v_x_2143_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_minOn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__9(lean_object* v_00_u03b1_2148_, lean_object* v_f_2149_, lean_object* v_x_2150_, lean_object* v_y_2151_){
_start:
{
lean_object* v___x_2152_; 
v___x_2152_ = lp_batteries_minOn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__9___redArg(v_f_2149_, v_x_2150_, v_y_2151_);
return v___x_2152_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__0(lean_object* v_categories_2153_, lean_object* v_fst_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_){
_start:
{
lean_object* v___x_2158_; lean_object* v_env_2159_; lean_object* v_options_2160_; lean_object* v_ref_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; 
v___x_2158_ = lean_st_ref_get(v___y_2156_);
v_env_2159_ = lean_ctor_get(v___x_2158_, 0);
lean_inc_ref(v_env_2159_);
lean_dec(v___x_2158_);
v_options_2160_ = lean_ctor_get(v___y_2155_, 2);
v_ref_2161_ = lean_ctor_get(v___y_2155_, 5);
lean_inc_ref(v_options_2160_);
v___x_2162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2162_, 0, v_env_2159_);
lean_ctor_set(v___x_2162_, 1, v_options_2160_);
v___x_2163_ = l_Lean_Parser_mkParserOfConstant(v_categories_2153_, v_fst_2154_, v___x_2162_);
lean_dec_ref_known(v___x_2162_, 2);
if (lean_obj_tag(v___x_2163_) == 0)
{
lean_object* v_a_2164_; lean_object* v___x_2166_; uint8_t v_isShared_2167_; uint8_t v_isSharedCheck_2171_; 
v_a_2164_ = lean_ctor_get(v___x_2163_, 0);
v_isSharedCheck_2171_ = !lean_is_exclusive(v___x_2163_);
if (v_isSharedCheck_2171_ == 0)
{
v___x_2166_ = v___x_2163_;
v_isShared_2167_ = v_isSharedCheck_2171_;
goto v_resetjp_2165_;
}
else
{
lean_inc(v_a_2164_);
lean_dec(v___x_2163_);
v___x_2166_ = lean_box(0);
v_isShared_2167_ = v_isSharedCheck_2171_;
goto v_resetjp_2165_;
}
v_resetjp_2165_:
{
lean_object* v___x_2169_; 
if (v_isShared_2167_ == 0)
{
v___x_2169_ = v___x_2166_;
goto v_reusejp_2168_;
}
else
{
lean_object* v_reuseFailAlloc_2170_; 
v_reuseFailAlloc_2170_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2170_, 0, v_a_2164_);
v___x_2169_ = v_reuseFailAlloc_2170_;
goto v_reusejp_2168_;
}
v_reusejp_2168_:
{
return v___x_2169_;
}
}
}
else
{
lean_object* v_a_2172_; lean_object* v___x_2174_; uint8_t v_isShared_2175_; uint8_t v_isSharedCheck_2183_; 
v_a_2172_ = lean_ctor_get(v___x_2163_, 0);
v_isSharedCheck_2183_ = !lean_is_exclusive(v___x_2163_);
if (v_isSharedCheck_2183_ == 0)
{
v___x_2174_ = v___x_2163_;
v_isShared_2175_ = v_isSharedCheck_2183_;
goto v_resetjp_2173_;
}
else
{
lean_inc(v_a_2172_);
lean_dec(v___x_2163_);
v___x_2174_ = lean_box(0);
v_isShared_2175_ = v_isSharedCheck_2183_;
goto v_resetjp_2173_;
}
v_resetjp_2173_:
{
lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2181_; 
v___x_2176_ = lean_io_error_to_string(v_a_2172_);
v___x_2177_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2177_, 0, v___x_2176_);
v___x_2178_ = l_Lean_MessageData_ofFormat(v___x_2177_);
lean_inc(v_ref_2161_);
v___x_2179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2179_, 0, v_ref_2161_);
lean_ctor_set(v___x_2179_, 1, v___x_2178_);
if (v_isShared_2175_ == 0)
{
lean_ctor_set(v___x_2174_, 0, v___x_2179_);
v___x_2181_ = v___x_2174_;
goto v_reusejp_2180_;
}
else
{
lean_object* v_reuseFailAlloc_2182_; 
v_reuseFailAlloc_2182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2182_, 0, v___x_2179_);
v___x_2181_ = v_reuseFailAlloc_2182_;
goto v_reusejp_2180_;
}
v_reusejp_2180_:
{
return v___x_2181_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__0___boxed(lean_object* v_categories_2184_, lean_object* v_fst_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_){
_start:
{
lean_object* v_res_2189_; 
v_res_2189_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__0(v_categories_2184_, v_fst_2185_, v___y_2186_, v___y_2187_);
lean_dec(v___y_2187_);
lean_dec_ref(v___y_2186_);
return v_res_2189_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1(lean_object* v___x_2192_, uint8_t v_fst_2193_, lean_object* v_fst_2194_, lean_object* v_fst_2195_, lean_object* v_a_2196_, uint8_t v___x_2197_, lean_object* v_____r_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_){
_start:
{
uint8_t v___x_2202_; 
v___x_2202_ = l_List_isEmpty___redArg(v___x_2192_);
if (v___x_2202_ == 0)
{
uint8_t v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; 
v___x_2203_ = 1;
v___x_2204_ = lean_box(v_fst_2193_);
v___x_2205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2205_, 0, v___x_2192_);
lean_ctor_set(v___x_2205_, 1, v___x_2204_);
v___x_2206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2206_, 0, v_fst_2194_);
lean_ctor_set(v___x_2206_, 1, v___x_2205_);
v___x_2207_ = lean_array_push(v_fst_2195_, v___x_2206_);
v___x_2208_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___closed__0));
v___x_2209_ = lean_box(v___x_2203_);
v___x_2210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2210_, 0, v_a_2196_);
lean_ctor_set(v___x_2210_, 1, v___x_2209_);
v___x_2211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2211_, 0, v___x_2207_);
lean_ctor_set(v___x_2211_, 1, v___x_2210_);
v___x_2212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2212_, 0, v___x_2208_);
lean_ctor_set(v___x_2212_, 1, v___x_2211_);
v___x_2213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2213_, 0, v___x_2212_);
return v___x_2213_;
}
else
{
lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; 
lean_dec(v_fst_2194_);
lean_dec(v___x_2192_);
v___x_2214_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___closed__0));
v___x_2215_ = lean_box(v___x_2197_);
v___x_2216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2216_, 0, v_a_2196_);
lean_ctor_set(v___x_2216_, 1, v___x_2215_);
v___x_2217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2217_, 0, v_fst_2195_);
lean_ctor_set(v___x_2217_, 1, v___x_2216_);
v___x_2218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2218_, 0, v___x_2214_);
lean_ctor_set(v___x_2218_, 1, v___x_2217_);
v___x_2219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2219_, 0, v___x_2218_);
return v___x_2219_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1___boxed(lean_object* v___x_2220_, lean_object* v_fst_2221_, lean_object* v_fst_2222_, lean_object* v_fst_2223_, lean_object* v_a_2224_, lean_object* v___x_2225_, lean_object* v_____r_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_){
_start:
{
uint8_t v_fst_20835__boxed_2230_; uint8_t v___x_20838__boxed_2231_; lean_object* v_res_2232_; 
v_fst_20835__boxed_2230_ = lean_unbox(v_fst_2221_);
v___x_20838__boxed_2231_ = lean_unbox(v___x_2225_);
v_res_2232_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1(v___x_2220_, v_fst_20835__boxed_2230_, v_fst_2222_, v_fst_2223_, v_a_2224_, v___x_20838__boxed_2231_, v_____r_2226_, v___y_2227_, v___y_2228_);
lean_dec(v___y_2228_);
lean_dec_ref(v___y_2227_);
return v_res_2232_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__5(lean_object* v_val_2233_, lean_object* v_x_2234_){
_start:
{
if (lean_obj_tag(v_x_2234_) == 0)
{
uint8_t v___x_2235_; 
v___x_2235_ = 0;
return v___x_2235_;
}
else
{
lean_object* v_head_2236_; lean_object* v_tail_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; uint8_t v___x_2240_; 
v_head_2236_ = lean_ctor_get(v_x_2234_, 0);
v_tail_2237_ = lean_ctor_get(v_x_2234_, 1);
v___x_2238_ = lean_string_utf8_byte_size(v_head_2236_);
v___x_2239_ = lean_string_utf8_byte_size(v_val_2233_);
v___x_2240_ = lean_nat_dec_le(v___x_2239_, v___x_2238_);
if (v___x_2240_ == 0)
{
v_x_2234_ = v_tail_2237_;
goto _start;
}
else
{
lean_object* v___x_2242_; uint8_t v___x_2243_; 
v___x_2242_ = lean_unsigned_to_nat(0u);
v___x_2243_ = lean_string_memcmp(v_head_2236_, v_val_2233_, v___x_2242_, v___x_2242_, v___x_2239_);
if (v___x_2243_ == 0)
{
v_x_2234_ = v_tail_2237_;
goto _start;
}
else
{
return v___x_2243_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__5___boxed(lean_object* v_val_2245_, lean_object* v_x_2246_){
_start:
{
uint8_t v_res_2247_; lean_object* v_r_2248_; 
v_res_2247_ = lp_batteries_List_any___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__5(v_val_2245_, v_x_2246_);
lean_dec(v_x_2246_);
lean_dec_ref(v_val_2245_);
v_r_2248_ = lean_box(v_res_2247_);
return v_r_2248_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2(lean_object* v_a_2250_, lean_object* v_a_2251_){
_start:
{
if (lean_obj_tag(v_a_2250_) == 0)
{
lean_object* v___x_2252_; 
v___x_2252_ = l_List_reverse___redArg(v_a_2251_);
return v___x_2252_;
}
else
{
lean_object* v_head_2253_; lean_object* v_tail_2254_; lean_object* v___x_2256_; uint8_t v_isShared_2257_; uint8_t v_isSharedCheck_2265_; 
v_head_2253_ = lean_ctor_get(v_a_2250_, 0);
v_tail_2254_ = lean_ctor_get(v_a_2250_, 1);
v_isSharedCheck_2265_ = !lean_is_exclusive(v_a_2250_);
if (v_isSharedCheck_2265_ == 0)
{
v___x_2256_ = v_a_2250_;
v_isShared_2257_ = v_isSharedCheck_2265_;
goto v_resetjp_2255_;
}
else
{
lean_inc(v_tail_2254_);
lean_inc(v_head_2253_);
lean_dec(v_a_2250_);
v___x_2256_ = lean_box(0);
v_isShared_2257_ = v_isSharedCheck_2265_;
goto v_resetjp_2255_;
}
v_resetjp_2255_:
{
lean_object* v___x_2258_; uint8_t v___x_2259_; 
v___x_2258_ = ((lean_object*)(lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2___closed__0));
v___x_2259_ = lean_string_dec_eq(v_head_2253_, v___x_2258_);
if (v___x_2259_ == 0)
{
lean_object* v___x_2261_; 
if (v_isShared_2257_ == 0)
{
lean_ctor_set(v___x_2256_, 1, v_a_2251_);
v___x_2261_ = v___x_2256_;
goto v_reusejp_2260_;
}
else
{
lean_object* v_reuseFailAlloc_2263_; 
v_reuseFailAlloc_2263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2263_, 0, v_head_2253_);
lean_ctor_set(v_reuseFailAlloc_2263_, 1, v_a_2251_);
v___x_2261_ = v_reuseFailAlloc_2263_;
goto v_reusejp_2260_;
}
v_reusejp_2260_:
{
v_a_2250_ = v_tail_2254_;
v_a_2251_ = v___x_2261_;
goto _start;
}
}
else
{
lean_del_object(v___x_2256_);
lean_dec(v_head_2253_);
v_a_2250_ = v_tail_2254_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0(lean_object* v_n_2266_){
_start:
{
lean_object* v___y_2268_; 
if (lean_obj_tag(v_n_2266_) == 0)
{
lean_object* v___x_2272_; 
v___x_2272_ = lean_unsigned_to_nat(0u);
v___y_2268_ = v___x_2272_;
goto v___jp_2267_;
}
else
{
lean_object* v_val_2273_; 
v_val_2273_ = lean_ctor_get(v_n_2266_, 0);
v___y_2268_ = v_val_2273_;
goto v___jp_2267_;
}
v___jp_2267_:
{
lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; 
v___x_2269_ = lean_unsigned_to_nat(1u);
v___x_2270_ = lean_nat_add(v___y_2268_, v___x_2269_);
v___x_2271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2271_, 0, v___x_2270_);
return v___x_2271_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0___boxed(lean_object* v_n_2274_){
_start:
{
lean_object* v_res_2275_; 
v_res_2275_ = lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0(v_n_2274_);
lean_dec(v_n_2274_);
return v_res_2275_;
}
}
static lean_object* _init_lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___closed__0(void){
_start:
{
lean_object* v___x_2276_; lean_object* v___x_2277_; 
v___x_2276_ = lean_box(0);
v___x_2277_ = lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0(v___x_2276_);
return v___x_2277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4(lean_object* v_a_2278_, lean_object* v_x_2279_){
_start:
{
if (lean_obj_tag(v_x_2279_) == 0)
{
lean_object* v___x_2280_; lean_object* v_val_2281_; lean_object* v___x_2282_; 
v___x_2280_ = lean_obj_once(&lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___closed__0, &lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___closed__0_once, _init_lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___closed__0);
v_val_2281_ = lean_ctor_get(v___x_2280_, 0);
lean_inc(v_val_2281_);
v___x_2282_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2282_, 0, v_a_2278_);
lean_ctor_set(v___x_2282_, 1, v_val_2281_);
lean_ctor_set(v___x_2282_, 2, v_x_2279_);
return v___x_2282_;
}
else
{
lean_object* v_key_2283_; lean_object* v_value_2284_; lean_object* v_tail_2285_; lean_object* v___x_2287_; uint8_t v_isShared_2288_; uint8_t v_isSharedCheck_2300_; 
v_key_2283_ = lean_ctor_get(v_x_2279_, 0);
v_value_2284_ = lean_ctor_get(v_x_2279_, 1);
v_tail_2285_ = lean_ctor_get(v_x_2279_, 2);
v_isSharedCheck_2300_ = !lean_is_exclusive(v_x_2279_);
if (v_isSharedCheck_2300_ == 0)
{
v___x_2287_ = v_x_2279_;
v_isShared_2288_ = v_isSharedCheck_2300_;
goto v_resetjp_2286_;
}
else
{
lean_inc(v_tail_2285_);
lean_inc(v_value_2284_);
lean_inc(v_key_2283_);
lean_dec(v_x_2279_);
v___x_2287_ = lean_box(0);
v_isShared_2288_ = v_isSharedCheck_2300_;
goto v_resetjp_2286_;
}
v_resetjp_2286_:
{
uint8_t v___x_2289_; 
v___x_2289_ = lean_string_dec_eq(v_key_2283_, v_a_2278_);
if (v___x_2289_ == 0)
{
lean_object* v_tail_2290_; lean_object* v___x_2292_; 
v_tail_2290_ = lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4(v_a_2278_, v_tail_2285_);
if (v_isShared_2288_ == 0)
{
lean_ctor_set(v___x_2287_, 2, v_tail_2290_);
v___x_2292_ = v___x_2287_;
goto v_reusejp_2291_;
}
else
{
lean_object* v_reuseFailAlloc_2293_; 
v_reuseFailAlloc_2293_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2293_, 0, v_key_2283_);
lean_ctor_set(v_reuseFailAlloc_2293_, 1, v_value_2284_);
lean_ctor_set(v_reuseFailAlloc_2293_, 2, v_tail_2290_);
v___x_2292_ = v_reuseFailAlloc_2293_;
goto v_reusejp_2291_;
}
v_reusejp_2291_:
{
return v___x_2292_;
}
}
else
{
lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v_val_2296_; lean_object* v___x_2298_; 
lean_dec(v_key_2283_);
v___x_2294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2294_, 0, v_value_2284_);
v___x_2295_ = lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4___lam__0(v___x_2294_);
lean_dec_ref_known(v___x_2294_, 1);
v_val_2296_ = lean_ctor_get(v___x_2295_, 0);
lean_inc(v_val_2296_);
lean_dec(v___x_2295_);
if (v_isShared_2288_ == 0)
{
lean_ctor_set(v___x_2287_, 1, v_val_2296_);
lean_ctor_set(v___x_2287_, 0, v_a_2278_);
v___x_2298_ = v___x_2287_;
goto v_reusejp_2297_;
}
else
{
lean_object* v_reuseFailAlloc_2299_; 
v_reuseFailAlloc_2299_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2299_, 0, v_a_2278_);
lean_ctor_set(v_reuseFailAlloc_2299_, 1, v_val_2296_);
lean_ctor_set(v_reuseFailAlloc_2299_, 2, v_tail_2285_);
v___x_2298_ = v_reuseFailAlloc_2299_;
goto v_reusejp_2297_;
}
v_reusejp_2297_:
{
return v___x_2298_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8_spec__21___redArg(lean_object* v_x_2301_, lean_object* v_x_2302_){
_start:
{
if (lean_obj_tag(v_x_2302_) == 0)
{
return v_x_2301_;
}
else
{
lean_object* v_key_2303_; lean_object* v_value_2304_; lean_object* v_tail_2305_; lean_object* v___x_2307_; uint8_t v_isShared_2308_; uint8_t v_isSharedCheck_2328_; 
v_key_2303_ = lean_ctor_get(v_x_2302_, 0);
v_value_2304_ = lean_ctor_get(v_x_2302_, 1);
v_tail_2305_ = lean_ctor_get(v_x_2302_, 2);
v_isSharedCheck_2328_ = !lean_is_exclusive(v_x_2302_);
if (v_isSharedCheck_2328_ == 0)
{
v___x_2307_ = v_x_2302_;
v_isShared_2308_ = v_isSharedCheck_2328_;
goto v_resetjp_2306_;
}
else
{
lean_inc(v_tail_2305_);
lean_inc(v_value_2304_);
lean_inc(v_key_2303_);
lean_dec(v_x_2302_);
v___x_2307_ = lean_box(0);
v_isShared_2308_ = v_isSharedCheck_2328_;
goto v_resetjp_2306_;
}
v_resetjp_2306_:
{
lean_object* v___x_2309_; uint64_t v___x_2310_; uint64_t v___x_2311_; uint64_t v___x_2312_; uint64_t v_fold_2313_; uint64_t v___x_2314_; uint64_t v___x_2315_; uint64_t v___x_2316_; size_t v___x_2317_; size_t v___x_2318_; size_t v___x_2319_; size_t v___x_2320_; size_t v___x_2321_; lean_object* v___x_2322_; lean_object* v___x_2324_; 
v___x_2309_ = lean_array_get_size(v_x_2301_);
v___x_2310_ = lean_string_hash(v_key_2303_);
v___x_2311_ = 32ULL;
v___x_2312_ = lean_uint64_shift_right(v___x_2310_, v___x_2311_);
v_fold_2313_ = lean_uint64_xor(v___x_2310_, v___x_2312_);
v___x_2314_ = 16ULL;
v___x_2315_ = lean_uint64_shift_right(v_fold_2313_, v___x_2314_);
v___x_2316_ = lean_uint64_xor(v_fold_2313_, v___x_2315_);
v___x_2317_ = lean_uint64_to_usize(v___x_2316_);
v___x_2318_ = lean_usize_of_nat(v___x_2309_);
v___x_2319_ = ((size_t)1ULL);
v___x_2320_ = lean_usize_sub(v___x_2318_, v___x_2319_);
v___x_2321_ = lean_usize_land(v___x_2317_, v___x_2320_);
v___x_2322_ = lean_array_uget_borrowed(v_x_2301_, v___x_2321_);
lean_inc(v___x_2322_);
if (v_isShared_2308_ == 0)
{
lean_ctor_set(v___x_2307_, 2, v___x_2322_);
v___x_2324_ = v___x_2307_;
goto v_reusejp_2323_;
}
else
{
lean_object* v_reuseFailAlloc_2327_; 
v_reuseFailAlloc_2327_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2327_, 0, v_key_2303_);
lean_ctor_set(v_reuseFailAlloc_2327_, 1, v_value_2304_);
lean_ctor_set(v_reuseFailAlloc_2327_, 2, v___x_2322_);
v___x_2324_ = v_reuseFailAlloc_2327_;
goto v_reusejp_2323_;
}
v_reusejp_2323_:
{
lean_object* v___x_2325_; 
v___x_2325_ = lean_array_uset(v_x_2301_, v___x_2321_, v___x_2324_);
v_x_2301_ = v___x_2325_;
v_x_2302_ = v_tail_2305_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8___redArg(lean_object* v_i_2329_, lean_object* v_source_2330_, lean_object* v_target_2331_){
_start:
{
lean_object* v___x_2332_; uint8_t v___x_2333_; 
v___x_2332_ = lean_array_get_size(v_source_2330_);
v___x_2333_ = lean_nat_dec_lt(v_i_2329_, v___x_2332_);
if (v___x_2333_ == 0)
{
lean_dec_ref(v_source_2330_);
lean_dec(v_i_2329_);
return v_target_2331_;
}
else
{
lean_object* v_es_2334_; lean_object* v___x_2335_; lean_object* v_source_2336_; lean_object* v_target_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; 
v_es_2334_ = lean_array_fget(v_source_2330_, v_i_2329_);
v___x_2335_ = lean_box(0);
v_source_2336_ = lean_array_fset(v_source_2330_, v_i_2329_, v___x_2335_);
v_target_2337_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8_spec__21___redArg(v_target_2331_, v_es_2334_);
v___x_2338_ = lean_unsigned_to_nat(1u);
v___x_2339_ = lean_nat_add(v_i_2329_, v___x_2338_);
lean_dec(v_i_2329_);
v_i_2329_ = v___x_2339_;
v_source_2330_ = v_source_2336_;
v_target_2331_ = v_target_2337_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3___redArg(lean_object* v_data_2341_){
_start:
{
lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v_nbuckets_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; 
v___x_2342_ = lean_array_get_size(v_data_2341_);
v___x_2343_ = lean_unsigned_to_nat(2u);
v_nbuckets_2344_ = lean_nat_mul(v___x_2342_, v___x_2343_);
v___x_2345_ = lean_unsigned_to_nat(0u);
v___x_2346_ = lean_box(0);
v___x_2347_ = lean_mk_array(v_nbuckets_2344_, v___x_2346_);
v___x_2348_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8___redArg(v___x_2345_, v_data_2341_, v___x_2347_);
return v___x_2348_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg(lean_object* v_a_2349_, lean_object* v_x_2350_){
_start:
{
if (lean_obj_tag(v_x_2350_) == 0)
{
uint8_t v___x_2351_; 
v___x_2351_ = 0;
return v___x_2351_;
}
else
{
lean_object* v_key_2352_; lean_object* v_tail_2353_; uint8_t v___x_2354_; 
v_key_2352_ = lean_ctor_get(v_x_2350_, 0);
v_tail_2353_ = lean_ctor_get(v_x_2350_, 2);
v___x_2354_ = lean_string_dec_eq(v_key_2352_, v_a_2349_);
if (v___x_2354_ == 0)
{
v_x_2350_ = v_tail_2353_;
goto _start;
}
else
{
return v___x_2354_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg___boxed(lean_object* v_a_2356_, lean_object* v_x_2357_){
_start:
{
uint8_t v_res_2358_; lean_object* v_r_2359_; 
v_res_2358_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg(v_a_2356_, v_x_2357_);
lean_dec(v_x_2357_);
lean_dec_ref(v_a_2356_);
v_r_2359_ = lean_box(v_res_2358_);
return v_r_2359_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1(lean_object* v_m_2360_, lean_object* v_a_2361_){
_start:
{
lean_object* v_size_2362_; lean_object* v_buckets_2363_; lean_object* v___x_2365_; uint8_t v_isShared_2366_; uint8_t v_isSharedCheck_2411_; 
v_size_2362_ = lean_ctor_get(v_m_2360_, 0);
v_buckets_2363_ = lean_ctor_get(v_m_2360_, 1);
v_isSharedCheck_2411_ = !lean_is_exclusive(v_m_2360_);
if (v_isSharedCheck_2411_ == 0)
{
v___x_2365_ = v_m_2360_;
v_isShared_2366_ = v_isSharedCheck_2411_;
goto v_resetjp_2364_;
}
else
{
lean_inc(v_buckets_2363_);
lean_inc(v_size_2362_);
lean_dec(v_m_2360_);
v___x_2365_ = lean_box(0);
v_isShared_2366_ = v_isSharedCheck_2411_;
goto v_resetjp_2364_;
}
v_resetjp_2364_:
{
lean_object* v___x_2367_; uint64_t v___x_2368_; uint64_t v___x_2369_; uint64_t v___x_2370_; uint64_t v_fold_2371_; uint64_t v___x_2372_; uint64_t v___x_2373_; uint64_t v___x_2374_; size_t v___x_2375_; size_t v___x_2376_; size_t v___x_2377_; size_t v___x_2378_; size_t v___x_2379_; lean_object* v_bkt_2380_; uint8_t v___x_2381_; 
v___x_2367_ = lean_array_get_size(v_buckets_2363_);
v___x_2368_ = lean_string_hash(v_a_2361_);
v___x_2369_ = 32ULL;
v___x_2370_ = lean_uint64_shift_right(v___x_2368_, v___x_2369_);
v_fold_2371_ = lean_uint64_xor(v___x_2368_, v___x_2370_);
v___x_2372_ = 16ULL;
v___x_2373_ = lean_uint64_shift_right(v_fold_2371_, v___x_2372_);
v___x_2374_ = lean_uint64_xor(v_fold_2371_, v___x_2373_);
v___x_2375_ = lean_uint64_to_usize(v___x_2374_);
v___x_2376_ = lean_usize_of_nat(v___x_2367_);
v___x_2377_ = ((size_t)1ULL);
v___x_2378_ = lean_usize_sub(v___x_2376_, v___x_2377_);
v___x_2379_ = lean_usize_land(v___x_2375_, v___x_2378_);
v_bkt_2380_ = lean_array_uget_borrowed(v_buckets_2363_, v___x_2379_);
v___x_2381_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg(v_a_2361_, v_bkt_2380_);
if (v___x_2381_ == 0)
{
lean_object* v___x_2382_; lean_object* v_size_x27_2383_; lean_object* v___x_2384_; lean_object* v_buckets_x27_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; uint8_t v___x_2391_; 
v___x_2382_ = lean_unsigned_to_nat(1u);
v_size_x27_2383_ = lean_nat_add(v_size_2362_, v___x_2382_);
lean_dec(v_size_2362_);
lean_inc(v_bkt_2380_);
v___x_2384_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2384_, 0, v_a_2361_);
lean_ctor_set(v___x_2384_, 1, v___x_2382_);
lean_ctor_set(v___x_2384_, 2, v_bkt_2380_);
v_buckets_x27_2385_ = lean_array_uset(v_buckets_2363_, v___x_2379_, v___x_2384_);
v___x_2386_ = lean_unsigned_to_nat(4u);
v___x_2387_ = lean_nat_mul(v_size_x27_2383_, v___x_2386_);
v___x_2388_ = lean_unsigned_to_nat(3u);
v___x_2389_ = lean_nat_div(v___x_2387_, v___x_2388_);
lean_dec(v___x_2387_);
v___x_2390_ = lean_array_get_size(v_buckets_x27_2385_);
v___x_2391_ = lean_nat_dec_le(v___x_2389_, v___x_2390_);
lean_dec(v___x_2389_);
if (v___x_2391_ == 0)
{
lean_object* v_val_2392_; lean_object* v___x_2394_; 
v_val_2392_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3___redArg(v_buckets_x27_2385_);
if (v_isShared_2366_ == 0)
{
lean_ctor_set(v___x_2365_, 1, v_val_2392_);
lean_ctor_set(v___x_2365_, 0, v_size_x27_2383_);
v___x_2394_ = v___x_2365_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v_size_x27_2383_);
lean_ctor_set(v_reuseFailAlloc_2395_, 1, v_val_2392_);
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
lean_object* v___x_2397_; 
if (v_isShared_2366_ == 0)
{
lean_ctor_set(v___x_2365_, 1, v_buckets_x27_2385_);
lean_ctor_set(v___x_2365_, 0, v_size_x27_2383_);
v___x_2397_ = v___x_2365_;
goto v_reusejp_2396_;
}
else
{
lean_object* v_reuseFailAlloc_2398_; 
v_reuseFailAlloc_2398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2398_, 0, v_size_x27_2383_);
lean_ctor_set(v_reuseFailAlloc_2398_, 1, v_buckets_x27_2385_);
v___x_2397_ = v_reuseFailAlloc_2398_;
goto v_reusejp_2396_;
}
v_reusejp_2396_:
{
return v___x_2397_;
}
}
}
else
{
lean_object* v___x_2399_; lean_object* v_buckets_x27_2400_; lean_object* v_bkt_x27_2401_; lean_object* v___y_2403_; uint8_t v___x_2408_; 
lean_inc(v_bkt_2380_);
v___x_2399_ = lean_box(0);
v_buckets_x27_2400_ = lean_array_uset(v_buckets_2363_, v___x_2379_, v___x_2399_);
lean_inc_ref(v_a_2361_);
v_bkt_x27_2401_ = lp_batteries_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__4(v_a_2361_, v_bkt_2380_);
v___x_2408_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg(v_a_2361_, v_bkt_x27_2401_);
lean_dec_ref(v_a_2361_);
if (v___x_2408_ == 0)
{
lean_object* v___x_2409_; lean_object* v___x_2410_; 
v___x_2409_ = lean_unsigned_to_nat(1u);
v___x_2410_ = lean_nat_sub(v_size_2362_, v___x_2409_);
lean_dec(v_size_2362_);
v___y_2403_ = v___x_2410_;
goto v___jp_2402_;
}
else
{
v___y_2403_ = v_size_2362_;
goto v___jp_2402_;
}
v___jp_2402_:
{
lean_object* v___x_2404_; lean_object* v___x_2406_; 
v___x_2404_ = lean_array_uset(v_buckets_x27_2400_, v___x_2379_, v_bkt_x27_2401_);
if (v_isShared_2366_ == 0)
{
lean_ctor_set(v___x_2365_, 1, v___x_2404_);
lean_ctor_set(v___x_2365_, 0, v___y_2403_);
v___x_2406_ = v___x_2365_;
goto v_reusejp_2405_;
}
else
{
lean_object* v_reuseFailAlloc_2407_; 
v_reuseFailAlloc_2407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2407_, 0, v___y_2403_);
lean_ctor_set(v_reuseFailAlloc_2407_, 1, v___x_2404_);
v___x_2406_ = v_reuseFailAlloc_2407_;
goto v_reusejp_2405_;
}
v_reusejp_2405_:
{
return v___x_2406_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg(lean_object* v_as_x27_2412_, lean_object* v_b_2413_){
_start:
{
if (lean_obj_tag(v_as_x27_2412_) == 0)
{
lean_object* v___x_2415_; 
v___x_2415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2415_, 0, v_b_2413_);
return v___x_2415_;
}
else
{
lean_object* v_head_2416_; lean_object* v_tail_2417_; lean_object* v___x_2418_; 
v_head_2416_ = lean_ctor_get(v_as_x27_2412_, 0);
v_tail_2417_ = lean_ctor_get(v_as_x27_2412_, 1);
lean_inc(v_head_2416_);
v___x_2418_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1(v_b_2413_, v_head_2416_);
v_as_x27_2412_ = v_tail_2417_;
v_b_2413_ = v___x_2418_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg___boxed(lean_object* v_as_x27_2420_, lean_object* v_b_2421_, lean_object* v___y_2422_){
_start:
{
lean_object* v_res_2423_; 
v_res_2423_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg(v_as_x27_2420_, v_b_2421_);
lean_dec(v_as_x27_2420_);
return v_res_2423_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__2(lean_object* v_categories_2424_, lean_object* v_id_2425_, lean_object* v_x_2426_, lean_object* v_____s_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_){
_start:
{
lean_object* v_fst_2431_; lean_object* v___x_2433_; uint8_t v_isShared_2434_; uint8_t v_isSharedCheck_2518_; 
v_fst_2431_ = lean_ctor_get(v_x_2426_, 0);
v_isSharedCheck_2518_ = !lean_is_exclusive(v_x_2426_);
if (v_isSharedCheck_2518_ == 0)
{
lean_object* v_unused_2519_; 
v_unused_2519_ = lean_ctor_get(v_x_2426_, 1);
lean_dec(v_unused_2519_);
v___x_2433_ = v_x_2426_;
v_isShared_2434_ = v_isSharedCheck_2518_;
goto v_resetjp_2432_;
}
else
{
lean_inc(v_fst_2431_);
lean_dec(v_x_2426_);
v___x_2433_ = lean_box(0);
v_isShared_2434_ = v_isSharedCheck_2518_;
goto v_resetjp_2432_;
}
v_resetjp_2432_:
{
lean_object* v_fst_2435_; lean_object* v_snd_2436_; lean_object* v___x_2438_; uint8_t v_isShared_2439_; uint8_t v_isSharedCheck_2517_; 
v_fst_2435_ = lean_ctor_get(v_____s_2427_, 0);
v_snd_2436_ = lean_ctor_get(v_____s_2427_, 1);
v_isSharedCheck_2517_ = !lean_is_exclusive(v_____s_2427_);
if (v_isSharedCheck_2517_ == 0)
{
v___x_2438_ = v_____s_2427_;
v_isShared_2439_ = v_isSharedCheck_2517_;
goto v_resetjp_2437_;
}
else
{
lean_inc(v_snd_2436_);
lean_inc(v_fst_2435_);
lean_dec(v_____s_2427_);
v___x_2438_ = lean_box(0);
v_isShared_2439_ = v_isSharedCheck_2517_;
goto v_resetjp_2437_;
}
v_resetjp_2437_:
{
lean_object* v_fst_2441_; lean_object* v_fst_2442_; lean_object* v___y_2452_; lean_object* v___y_2453_; lean_object* v_fst_2458_; lean_object* v_snd_2459_; lean_object* v___x_2461_; uint8_t v_isShared_2462_; uint8_t v_isSharedCheck_2516_; 
v_fst_2458_ = lean_ctor_get(v_snd_2436_, 0);
v_snd_2459_ = lean_ctor_get(v_snd_2436_, 1);
v_isSharedCheck_2516_ = !lean_is_exclusive(v_snd_2436_);
if (v_isSharedCheck_2516_ == 0)
{
v___x_2461_ = v_snd_2436_;
v_isShared_2462_ = v_isSharedCheck_2516_;
goto v_resetjp_2460_;
}
else
{
lean_inc(v_snd_2459_);
lean_inc(v_fst_2458_);
lean_dec(v_snd_2436_);
v___x_2461_ = lean_box(0);
v_isShared_2462_ = v_isSharedCheck_2516_;
goto v_resetjp_2460_;
}
v___jp_2440_:
{
lean_object* v___x_2444_; 
if (v_isShared_2439_ == 0)
{
lean_ctor_set(v___x_2438_, 1, v_fst_2442_);
lean_ctor_set(v___x_2438_, 0, v_fst_2441_);
v___x_2444_ = v___x_2438_;
goto v_reusejp_2443_;
}
else
{
lean_object* v_reuseFailAlloc_2450_; 
v_reuseFailAlloc_2450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2450_, 0, v_fst_2441_);
lean_ctor_set(v_reuseFailAlloc_2450_, 1, v_fst_2442_);
v___x_2444_ = v_reuseFailAlloc_2450_;
goto v_reusejp_2443_;
}
v_reusejp_2443_:
{
lean_object* v___x_2446_; 
if (v_isShared_2434_ == 0)
{
lean_ctor_set(v___x_2433_, 1, v___x_2444_);
lean_ctor_set(v___x_2433_, 0, v_fst_2435_);
v___x_2446_ = v___x_2433_;
goto v_reusejp_2445_;
}
else
{
lean_object* v_reuseFailAlloc_2449_; 
v_reuseFailAlloc_2449_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2449_, 0, v_fst_2435_);
lean_ctor_set(v_reuseFailAlloc_2449_, 1, v___x_2444_);
v___x_2446_ = v_reuseFailAlloc_2449_;
goto v_reusejp_2445_;
}
v_reusejp_2445_:
{
lean_object* v___x_2447_; lean_object* v___x_2448_; 
v___x_2447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2447_, 0, v___x_2446_);
v___x_2448_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2448_, 0, v___x_2447_);
return v___x_2448_;
}
}
}
v___jp_2451_:
{
lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; 
v___x_2454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2454_, 0, v___y_2453_);
lean_ctor_set(v___x_2454_, 1, v___y_2452_);
v___x_2455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2455_, 0, v_fst_2435_);
lean_ctor_set(v___x_2455_, 1, v___x_2454_);
v___x_2456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2456_, 0, v___x_2455_);
v___x_2457_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2457_, 0, v___x_2456_);
return v___x_2457_;
}
v_resetjp_2460_:
{
lean_object* v___f_2463_; uint8_t v___x_2464_; lean_object* v_fst_2466_; lean_object* v_fst_2467_; lean_object* v_a_2477_; lean_object* v___y_2481_; lean_object* v___x_2494_; 
lean_inc(v_fst_2431_);
v___f_2463_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__0___boxed), 5, 2);
lean_closure_set(v___f_2463_, 0, v_categories_2424_);
lean_closure_set(v___f_2463_, 1, v_fst_2431_);
v___x_2464_ = 0;
v___x_2494_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_2463_, v___y_2428_, v___y_2429_);
if (lean_obj_tag(v___x_2494_) == 0)
{
lean_object* v_a_2495_; lean_object* v_snd_2496_; lean_object* v_info_2497_; lean_object* v_fst_2498_; lean_object* v_firstTokens_2499_; lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; 
v_a_2495_ = lean_ctor_get(v___x_2494_, 0);
lean_inc(v_a_2495_);
lean_dec_ref_known(v___x_2494_, 1);
v_snd_2496_ = lean_ctor_get(v_a_2495_, 1);
v_info_2497_ = lean_ctor_get(v_snd_2496_, 0);
lean_inc_ref(v_info_2497_);
v_fst_2498_ = lean_ctor_get(v_a_2495_, 0);
lean_inc(v_fst_2498_);
lean_dec(v_a_2495_);
v_firstTokens_2499_ = lean_ctor_get(v_info_2497_, 2);
lean_inc(v_firstTokens_2499_);
lean_dec_ref(v_info_2497_);
v___x_2500_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_tokensToList(v_firstTokens_2499_);
lean_dec(v_firstTokens_2499_);
v___x_2501_ = lean_box(0);
v___x_2502_ = lp_batteries_List_filterTR_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__2(v___x_2500_, v___x_2501_);
lean_inc(v___x_2502_);
v___x_2503_ = lp_batteries_List_eraseDups___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__3(v___x_2502_);
v___x_2504_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg(v___x_2503_, v_snd_2459_);
lean_dec(v___x_2503_);
if (lean_obj_tag(v_id_2425_) == 1)
{
lean_object* v_a_2505_; lean_object* v_val_2506_; uint8_t v___x_2507_; 
v_a_2505_ = lean_ctor_get(v___x_2504_, 0);
lean_inc(v_a_2505_);
lean_dec_ref(v___x_2504_);
v_val_2506_ = lean_ctor_get(v_id_2425_, 0);
v___x_2507_ = lp_batteries_List_any___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__5(v_val_2506_, v___x_2502_);
if (v___x_2507_ == 0)
{
lean_dec(v___x_2502_);
lean_dec(v_fst_2498_);
lean_del_object(v___x_2461_);
lean_dec(v_fst_2431_);
v_fst_2441_ = v_fst_2458_;
v_fst_2442_ = v_a_2505_;
goto v___jp_2440_;
}
else
{
lean_object* v___x_2508_; uint8_t v___x_2509_; lean_object* v___x_2510_; 
v___x_2508_ = lean_box(0);
v___x_2509_ = lean_unbox(v_fst_2498_);
lean_dec(v_fst_2498_);
lean_inc(v_fst_2431_);
v___x_2510_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1(v___x_2502_, v___x_2509_, v_fst_2431_, v_fst_2458_, v_a_2505_, v___x_2464_, v___x_2508_, v___y_2428_, v___y_2429_);
v___y_2481_ = v___x_2510_;
goto v___jp_2480_;
}
}
else
{
lean_object* v_a_2511_; lean_object* v___x_2512_; uint8_t v___x_2513_; lean_object* v___x_2514_; 
v_a_2511_ = lean_ctor_get(v___x_2504_, 0);
lean_inc(v_a_2511_);
lean_dec_ref(v___x_2504_);
v___x_2512_ = lean_box(0);
v___x_2513_ = lean_unbox(v_fst_2498_);
lean_dec(v_fst_2498_);
lean_inc(v_fst_2431_);
v___x_2514_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__1(v___x_2502_, v___x_2513_, v_fst_2431_, v_fst_2458_, v_a_2511_, v___x_2464_, v___x_2512_, v___y_2428_, v___y_2429_);
v___y_2481_ = v___x_2514_;
goto v___jp_2480_;
}
}
else
{
lean_object* v_a_2515_; 
lean_del_object(v___x_2438_);
lean_del_object(v___x_2433_);
v_a_2515_ = lean_ctor_get(v___x_2494_, 0);
lean_inc(v_a_2515_);
lean_dec_ref_known(v___x_2494_, 1);
v_a_2477_ = v_a_2515_;
goto v___jp_2476_;
}
v___jp_2465_:
{
if (lean_obj_tag(v_id_2425_) == 0)
{
lean_object* v___x_2468_; lean_object* v___x_2469_; lean_object* v___x_2471_; 
lean_inc(v_fst_2431_);
v___x_2468_ = l_Lean_Name_toString(v_fst_2431_, v___x_2464_);
v___x_2469_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__3___redArg(v___x_2468_, v_fst_2431_, v_fst_2435_);
if (v_isShared_2462_ == 0)
{
lean_ctor_set(v___x_2461_, 1, v_fst_2467_);
lean_ctor_set(v___x_2461_, 0, v_fst_2466_);
v___x_2471_ = v___x_2461_;
goto v_reusejp_2470_;
}
else
{
lean_object* v_reuseFailAlloc_2475_; 
v_reuseFailAlloc_2475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2475_, 0, v_fst_2466_);
lean_ctor_set(v_reuseFailAlloc_2475_, 1, v_fst_2467_);
v___x_2471_ = v_reuseFailAlloc_2475_;
goto v_reusejp_2470_;
}
v_reusejp_2470_:
{
lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; 
v___x_2472_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2472_, 0, v___x_2469_);
lean_ctor_set(v___x_2472_, 1, v___x_2471_);
v___x_2473_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2473_, 0, v___x_2472_);
v___x_2474_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2474_, 0, v___x_2473_);
return v___x_2474_;
}
}
else
{
lean_del_object(v___x_2461_);
lean_dec(v_fst_2431_);
v___y_2452_ = v_fst_2467_;
v___y_2453_ = v_fst_2466_;
goto v___jp_2451_;
}
}
v___jp_2476_:
{
uint8_t v___x_2478_; 
v___x_2478_ = l_Lean_Exception_isInterrupt(v_a_2477_);
if (v___x_2478_ == 0)
{
lean_dec_ref(v_a_2477_);
v_fst_2466_ = v_fst_2458_;
v_fst_2467_ = v_snd_2459_;
goto v___jp_2465_;
}
else
{
lean_object* v___x_2479_; 
lean_del_object(v___x_2461_);
lean_dec(v_snd_2459_);
lean_dec(v_fst_2458_);
lean_dec(v_fst_2435_);
lean_dec(v_fst_2431_);
v___x_2479_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2479_, 0, v_a_2477_);
return v___x_2479_;
}
}
v___jp_2480_:
{
lean_object* v_a_2482_; lean_object* v_snd_2483_; lean_object* v_snd_2484_; lean_object* v_fst_2485_; 
v_a_2482_ = lean_ctor_get(v___y_2481_, 0);
lean_inc(v_a_2482_);
lean_dec_ref(v___y_2481_);
v_snd_2483_ = lean_ctor_get(v_a_2482_, 1);
lean_inc(v_snd_2483_);
v_snd_2484_ = lean_ctor_get(v_snd_2483_, 1);
lean_inc(v_snd_2484_);
v_fst_2485_ = lean_ctor_get(v_a_2482_, 0);
lean_inc(v_fst_2485_);
lean_dec(v_a_2482_);
if (lean_obj_tag(v_fst_2485_) == 0)
{
lean_object* v_fst_2486_; lean_object* v_fst_2487_; 
lean_del_object(v___x_2461_);
lean_dec(v_fst_2431_);
v_fst_2486_ = lean_ctor_get(v_snd_2483_, 0);
lean_inc(v_fst_2486_);
lean_dec(v_snd_2483_);
v_fst_2487_ = lean_ctor_get(v_snd_2484_, 0);
lean_inc(v_fst_2487_);
lean_dec(v_snd_2484_);
v_fst_2441_ = v_fst_2486_;
v_fst_2442_ = v_fst_2487_;
goto v___jp_2440_;
}
else
{
lean_object* v_snd_2488_; uint8_t v___x_2489_; 
lean_dec_ref_known(v_fst_2485_, 1);
lean_del_object(v___x_2438_);
lean_del_object(v___x_2433_);
v_snd_2488_ = lean_ctor_get(v_snd_2484_, 1);
v___x_2489_ = lean_unbox(v_snd_2488_);
if (v___x_2489_ == 0)
{
lean_object* v_fst_2490_; lean_object* v_fst_2491_; 
v_fst_2490_ = lean_ctor_get(v_snd_2483_, 0);
lean_inc(v_fst_2490_);
lean_dec(v_snd_2483_);
v_fst_2491_ = lean_ctor_get(v_snd_2484_, 0);
lean_inc(v_fst_2491_);
lean_dec(v_snd_2484_);
v_fst_2466_ = v_fst_2490_;
v_fst_2467_ = v_fst_2491_;
goto v___jp_2465_;
}
else
{
lean_object* v_fst_2492_; lean_object* v_fst_2493_; 
lean_del_object(v___x_2461_);
lean_dec(v_fst_2431_);
v_fst_2492_ = lean_ctor_get(v_snd_2483_, 0);
lean_inc(v_fst_2492_);
lean_dec(v_snd_2483_);
v_fst_2493_ = lean_ctor_get(v_snd_2484_, 0);
lean_inc(v_fst_2493_);
lean_dec(v_snd_2484_);
v___y_2452_ = v_fst_2493_;
v___y_2453_ = v_fst_2492_;
goto v___jp_2451_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__2___boxed(lean_object* v_categories_2520_, lean_object* v_id_2521_, lean_object* v_x_2522_, lean_object* v_____s_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_, lean_object* v___y_2526_){
_start:
{
lean_object* v_res_2527_; 
v_res_2527_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__2(v_categories_2520_, v_id_2521_, v_x_2522_, v_____s_2523_, v___y_2524_, v___y_2525_);
lean_dec(v___y_2525_);
lean_dec_ref(v___y_2524_);
lean_dec(v_id_2521_);
return v_res_2527_;
}
}
static lean_object* _init_lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__1(void){
_start:
{
lean_object* v___x_2529_; lean_object* v___x_2530_; 
v___x_2529_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__0));
v___x_2530_ = l_Lean_stringToMessageData(v___x_2529_);
return v___x_2530_;
}
}
static lean_object* _init_lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__3(void){
_start:
{
lean_object* v___x_2532_; lean_object* v___x_2533_; 
v___x_2532_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__2));
v___x_2533_ = l_Lean_stringToMessageData(v___x_2532_);
return v___x_2533_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg(lean_object* v___x_2534_, lean_object* v_type_2535_, lean_object* v_as_x27_2536_, lean_object* v_b_2537_, lean_object* v___y_2538_){
_start:
{
if (lean_obj_tag(v_as_x27_2536_) == 0)
{
lean_object* v___x_2540_; 
lean_dec_ref(v_type_2535_);
lean_dec_ref(v___x_2534_);
v___x_2540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2540_, 0, v_b_2537_);
return v___x_2540_;
}
else
{
lean_object* v_head_2541_; lean_object* v_toOLeanEntry_2542_; lean_object* v_tail_2543_; lean_object* v_declName_2544_; uint8_t v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; 
v_head_2541_ = lean_ctor_get(v_as_x27_2536_, 0);
v_toOLeanEntry_2542_ = lean_ctor_get(v_head_2541_, 0);
v_tail_2543_ = lean_ctor_get(v_as_x27_2536_, 1);
v_declName_2544_ = lean_ctor_get(v_toOLeanEntry_2542_, 1);
v___x_2545_ = 1;
v___x_2546_ = l_Lean_Options_empty;
v___x_2547_ = lean_box(0);
v___x_2548_ = lean_box(0);
lean_inc_n(v_declName_2544_, 2);
v___x_2549_ = l_Lean_mkConst(v_declName_2544_, v___x_2548_);
lean_inc_ref(v___x_2534_);
v___x_2550_ = l_Lean_findDocString_x3f(v___x_2534_, v_declName_2544_, v___x_2545_, v___x_2546_, v___x_2547_, v___x_2548_);
if (lean_obj_tag(v___x_2550_) == 0)
{
lean_object* v_a_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; 
v_a_2551_ = lean_ctor_get(v___x_2550_, 0);
lean_inc(v_a_2551_);
lean_dec_ref_known(v___x_2550_, 1);
v___x_2552_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
v___x_2553_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2553_, 0, v_b_2537_);
lean_ctor_set(v___x_2553_, 1, v___x_2552_);
v___x_2554_ = lean_obj_once(&lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__1, &lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__1_once, _init_lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__1);
lean_inc_ref(v_type_2535_);
v___x_2555_ = l_Lean_stringToMessageData(v_type_2535_);
v___x_2556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2556_, 0, v___x_2554_);
lean_ctor_set(v___x_2556_, 1, v___x_2555_);
v___x_2557_ = lean_obj_once(&lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__3, &lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__3_once, _init_lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___closed__3);
v___x_2558_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2558_, 0, v___x_2556_);
lean_ctor_set(v___x_2558_, 1, v___x_2557_);
v___x_2559_ = l_Lean_MessageData_ofExpr(v___x_2549_);
v___x_2560_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2560_, 0, v___x_2558_);
lean_ctor_set(v___x_2560_, 1, v___x_2559_);
v___x_2561_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2561_, 0, v___x_2553_);
lean_ctor_set(v___x_2561_, 1, v___x_2560_);
if (lean_obj_tag(v_a_2551_) == 1)
{
lean_object* v_val_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2583_; 
v_val_2562_ = lean_ctor_get(v_a_2551_, 0);
v_isSharedCheck_2583_ = !lean_is_exclusive(v_a_2551_);
if (v_isSharedCheck_2583_ == 0)
{
v___x_2564_ = v_a_2551_;
v_isShared_2565_ = v_isSharedCheck_2583_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_val_2562_);
lean_dec(v_a_2551_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2583_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v_str_2570_; lean_object* v_startInclusive_2571_; lean_object* v_endExclusive_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2576_; 
v___x_2566_ = lean_unsigned_to_nat(0u);
v___x_2567_ = lean_string_utf8_byte_size(v_val_2562_);
v___x_2568_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2568_, 0, v_val_2562_);
lean_ctor_set(v___x_2568_, 1, v___x_2566_);
lean_ctor_set(v___x_2568_, 2, v___x_2567_);
v___x_2569_ = l_String_Slice_trimAscii(v___x_2568_);
v_str_2570_ = lean_ctor_get(v___x_2569_, 0);
lean_inc_ref(v_str_2570_);
v_startInclusive_2571_ = lean_ctor_get(v___x_2569_, 1);
lean_inc(v_startInclusive_2571_);
v_endExclusive_2572_ = lean_ctor_get(v___x_2569_, 2);
lean_inc(v_endExclusive_2572_);
lean_dec_ref(v___x_2569_);
v___x_2573_ = lean_unsigned_to_nat(2u);
v___x_2574_ = lean_string_utf8_extract_fast(v_str_2570_, v_startInclusive_2571_, v_endExclusive_2572_);
lean_dec(v_endExclusive_2572_);
lean_dec(v_startInclusive_2571_);
lean_dec_ref(v_str_2570_);
if (v_isShared_2565_ == 0)
{
lean_ctor_set_tag(v___x_2564_, 3);
lean_ctor_set(v___x_2564_, 0, v___x_2574_);
v___x_2576_ = v___x_2564_;
goto v_reusejp_2575_;
}
else
{
lean_object* v_reuseFailAlloc_2582_; 
v_reuseFailAlloc_2582_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2582_, 0, v___x_2574_);
v___x_2576_ = v_reuseFailAlloc_2582_;
goto v_reusejp_2575_;
}
v_reusejp_2575_:
{
lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; 
v___x_2577_ = l_Lean_MessageData_ofFormat(v___x_2576_);
v___x_2578_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2578_, 0, v___x_2552_);
lean_ctor_set(v___x_2578_, 1, v___x_2577_);
v___x_2579_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2579_, 0, v___x_2573_);
lean_ctor_set(v___x_2579_, 1, v___x_2578_);
v___x_2580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2580_, 0, v___x_2561_);
lean_ctor_set(v___x_2580_, 1, v___x_2579_);
v_as_x27_2536_ = v_tail_2543_;
v_b_2537_ = v___x_2580_;
goto _start;
}
}
}
else
{
lean_dec(v_a_2551_);
v_as_x27_2536_ = v_tail_2543_;
v_b_2537_ = v___x_2561_;
goto _start;
}
}
else
{
lean_object* v_a_2585_; lean_object* v___x_2587_; uint8_t v_isShared_2588_; uint8_t v_isSharedCheck_2597_; 
lean_dec_ref(v___x_2549_);
lean_dec_ref(v_b_2537_);
lean_dec_ref(v_type_2535_);
lean_dec_ref(v___x_2534_);
v_a_2585_ = lean_ctor_get(v___x_2550_, 0);
v_isSharedCheck_2597_ = !lean_is_exclusive(v___x_2550_);
if (v_isSharedCheck_2597_ == 0)
{
v___x_2587_ = v___x_2550_;
v_isShared_2588_ = v_isSharedCheck_2597_;
goto v_resetjp_2586_;
}
else
{
lean_inc(v_a_2585_);
lean_dec(v___x_2550_);
v___x_2587_ = lean_box(0);
v_isShared_2588_ = v_isSharedCheck_2597_;
goto v_resetjp_2586_;
}
v_resetjp_2586_:
{
lean_object* v_ref_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2595_; 
v_ref_2589_ = lean_ctor_get(v___y_2538_, 7);
v___x_2590_ = lean_io_error_to_string(v_a_2585_);
v___x_2591_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2591_, 0, v___x_2590_);
v___x_2592_ = l_Lean_MessageData_ofFormat(v___x_2591_);
lean_inc(v_ref_2589_);
v___x_2593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2593_, 0, v_ref_2589_);
lean_ctor_set(v___x_2593_, 1, v___x_2592_);
if (v_isShared_2588_ == 0)
{
lean_ctor_set(v___x_2587_, 0, v___x_2593_);
v___x_2595_ = v___x_2587_;
goto v_reusejp_2594_;
}
else
{
lean_object* v_reuseFailAlloc_2596_; 
v_reuseFailAlloc_2596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2596_, 0, v___x_2593_);
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
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg___boxed(lean_object* v___x_2598_, lean_object* v_type_2599_, lean_object* v_as_x27_2600_, lean_object* v_b_2601_, lean_object* v___y_2602_, lean_object* v___y_2603_){
_start:
{
lean_object* v_res_2604_; 
v_res_2604_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg(v___x_2598_, v_type_2599_, v_as_x27_2600_, v_b_2601_, v___y_2602_);
lean_dec_ref(v___y_2602_);
lean_dec(v_as_x27_2600_);
return v_res_2604_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(lean_object* v___x_2605_, lean_object* v_v_2606_, lean_object* v_00_u03b1_2607_, lean_object* v_type_2608_, lean_object* v_attr_2609_, lean_object* v_msg_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_){
_start:
{
lean_object* v___x_2614_; lean_object* v___x_2615_; 
lean_inc_ref(v___x_2605_);
v___x_2614_ = l_Lean_KeyedDeclsAttribute_getEntries___redArg(v_attr_2609_, v___x_2605_, v_v_2606_);
v___x_2615_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg(v___x_2605_, v_type_2608_, v___x_2614_, v_msg_2610_, v___y_2611_);
lean_dec(v___x_2614_);
return v___x_2615_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0___boxed(lean_object* v___x_2616_, lean_object* v_v_2617_, lean_object* v_00_u03b1_2618_, lean_object* v_type_2619_, lean_object* v_attr_2620_, lean_object* v_msg_2621_, lean_object* v___y_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_){
_start:
{
lean_object* v_res_2625_; 
v_res_2625_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2616_, v_v_2617_, v_00_u03b1_2618_, v_type_2619_, v_attr_2620_, v_msg_2621_, v___y_2622_, v___y_2623_);
lean_dec(v___y_2623_);
lean_dec_ref(v___y_2622_);
lean_dec_ref(v_attr_2620_);
lean_dec(v_v_2617_);
return v_res_2625_;
}
}
static lean_object* _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9(void){
_start:
{
lean_object* v___x_2635_; lean_object* v___x_2636_; 
v___x_2635_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__8));
v___x_2636_ = l_Lean_stringToMessageData(v___x_2635_);
return v___x_2636_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19(lean_object* v___x_2637_, lean_object* v_more_2638_, lean_object* v_catName_2639_, lean_object* v_init_2640_, lean_object* v_x_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_){
_start:
{
if (lean_obj_tag(v_x_2641_) == 0)
{
lean_object* v_v_2645_; lean_object* v_l_2646_; lean_object* v_r_2647_; lean_object* v___x_2648_; 
v_v_2645_ = lean_ctor_get(v_x_2641_, 2);
lean_inc(v_v_2645_);
v_l_2646_ = lean_ctor_get(v_x_2641_, 3);
lean_inc(v_l_2646_);
v_r_2647_ = lean_ctor_get(v_x_2641_, 4);
lean_inc(v_r_2647_);
lean_dec_ref_known(v_x_2641_, 5);
lean_inc_ref(v___x_2637_);
v___x_2648_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19(v___x_2637_, v_more_2638_, v_catName_2639_, v_init_2640_, v_l_2646_, v___y_2642_, v___y_2643_);
if (lean_obj_tag(v___x_2648_) == 0)
{
lean_object* v_a_2649_; lean_object* v_a_2650_; lean_object* v___x_2652_; uint8_t v_isShared_2653_; uint8_t v_isSharedCheck_2777_; 
v_a_2649_ = lean_ctor_get(v___x_2648_, 0);
lean_inc(v_a_2649_);
lean_dec_ref_known(v___x_2648_, 1);
v_a_2650_ = lean_ctor_get(v_a_2649_, 0);
v_isSharedCheck_2777_ = !lean_is_exclusive(v_a_2649_);
if (v_isSharedCheck_2777_ == 0)
{
v___x_2652_ = v_a_2649_;
v_isShared_2653_ = v_isSharedCheck_2777_;
goto v_resetjp_2651_;
}
else
{
lean_inc(v_a_2650_);
lean_dec(v_a_2649_);
v___x_2652_ = lean_box(0);
v_isShared_2653_ = v_isSharedCheck_2777_;
goto v_resetjp_2651_;
}
v_resetjp_2651_:
{
lean_object* v_msg1_2655_; lean_object* v___x_2660_; lean_object* v___x_2661_; uint8_t v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; 
v___x_2660_ = lean_box(0);
lean_inc_n(v_v_2645_, 2);
v___x_2661_ = l_Lean_mkConst(v_v_2645_, v___x_2660_);
v___x_2662_ = 1;
v___x_2663_ = l_Lean_Options_empty;
v___x_2664_ = lean_box(0);
lean_inc_ref(v___x_2637_);
v___x_2665_ = l_Lean_findDocString_x3f(v___x_2637_, v_v_2645_, v___x_2662_, v___x_2663_, v___x_2664_, v___x_2660_);
if (lean_obj_tag(v___x_2665_) == 0)
{
lean_object* v_a_2666_; lean_object* v___y_2668_; lean_object* v___y_2669_; lean_object* v___y_2670_; lean_object* v_msg1_2684_; lean_object* v___y_2685_; lean_object* v___y_2686_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; lean_object* v___x_2741_; 
lean_del_object(v___x_2652_);
v_a_2666_ = lean_ctor_get(v___x_2665_, 0);
lean_inc(v_a_2666_);
lean_dec_ref_known(v___x_2665_, 1);
v___x_2737_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9);
v___x_2738_ = l_Lean_MessageData_ofExpr(v___x_2661_);
v___x_2739_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2739_, 0, v___x_2737_);
lean_ctor_set(v___x_2739_, 1, v___x_2738_);
v___x_2740_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7);
v___x_2741_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2741_, 0, v___x_2739_);
lean_ctor_set(v___x_2741_, 1, v___x_2740_);
if (lean_obj_tag(v_a_2666_) == 1)
{
lean_object* v_val_2742_; lean_object* v___x_2744_; uint8_t v_isShared_2745_; uint8_t v_isSharedCheck_2761_; 
v_val_2742_ = lean_ctor_get(v_a_2666_, 0);
v_isSharedCheck_2761_ = !lean_is_exclusive(v_a_2666_);
if (v_isSharedCheck_2761_ == 0)
{
v___x_2744_ = v_a_2666_;
v_isShared_2745_ = v_isSharedCheck_2761_;
goto v_resetjp_2743_;
}
else
{
lean_inc(v_val_2742_);
lean_dec(v_a_2666_);
v___x_2744_ = lean_box(0);
v_isShared_2745_ = v_isSharedCheck_2761_;
goto v_resetjp_2743_;
}
v_resetjp_2743_:
{
lean_object* v___x_2746_; lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2749_; lean_object* v_str_2750_; lean_object* v_startInclusive_2751_; lean_object* v_endExclusive_2752_; lean_object* v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___x_2757_; 
v___x_2746_ = lean_unsigned_to_nat(0u);
v___x_2747_ = lean_string_utf8_byte_size(v_val_2742_);
v___x_2748_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2748_, 0, v_val_2742_);
lean_ctor_set(v___x_2748_, 1, v___x_2746_);
lean_ctor_set(v___x_2748_, 2, v___x_2747_);
v___x_2749_ = l_String_Slice_trimAscii(v___x_2748_);
v_str_2750_ = lean_ctor_get(v___x_2749_, 0);
lean_inc_ref(v_str_2750_);
v_startInclusive_2751_ = lean_ctor_get(v___x_2749_, 1);
lean_inc(v_startInclusive_2751_);
v_endExclusive_2752_ = lean_ctor_get(v___x_2749_, 2);
lean_inc(v_endExclusive_2752_);
lean_dec_ref(v___x_2749_);
v___x_2753_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
v___x_2754_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2754_, 0, v___x_2741_);
lean_ctor_set(v___x_2754_, 1, v___x_2753_);
v___x_2755_ = lean_string_utf8_extract_fast(v_str_2750_, v_startInclusive_2751_, v_endExclusive_2752_);
lean_dec(v_endExclusive_2752_);
lean_dec(v_startInclusive_2751_);
lean_dec_ref(v_str_2750_);
if (v_isShared_2745_ == 0)
{
lean_ctor_set_tag(v___x_2744_, 3);
lean_ctor_set(v___x_2744_, 0, v___x_2755_);
v___x_2757_ = v___x_2744_;
goto v_reusejp_2756_;
}
else
{
lean_object* v_reuseFailAlloc_2760_; 
v_reuseFailAlloc_2760_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2760_, 0, v___x_2755_);
v___x_2757_ = v_reuseFailAlloc_2760_;
goto v_reusejp_2756_;
}
v_reusejp_2756_:
{
lean_object* v___x_2758_; lean_object* v___x_2759_; 
v___x_2758_ = l_Lean_MessageData_ofFormat(v___x_2757_);
v___x_2759_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2759_, 0, v___x_2754_);
lean_ctor_set(v___x_2759_, 1, v___x_2758_);
v_msg1_2684_ = v___x_2759_;
v___y_2685_ = v___y_2642_;
v___y_2686_ = v___y_2643_;
goto v___jp_2683_;
}
}
}
else
{
lean_dec(v_a_2666_);
v_msg1_2684_ = v___x_2741_;
v___y_2685_ = v___y_2642_;
v___y_2686_ = v___y_2643_;
goto v___jp_2683_;
}
v___jp_2667_:
{
lean_object* v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; 
v___x_2671_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__0));
v___x_2672_ = l_Lean_Elab_Tactic_tacticElabAttribute;
lean_inc_ref(v___x_2637_);
v___x_2673_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2637_, v_v_2645_, lean_box(0), v___x_2671_, v___x_2672_, v___y_2668_, v___y_2669_, v___y_2670_);
lean_dec(v_v_2645_);
if (lean_obj_tag(v___x_2673_) == 0)
{
lean_object* v_a_2674_; 
v_a_2674_ = lean_ctor_get(v___x_2673_, 0);
lean_inc(v_a_2674_);
lean_dec_ref_known(v___x_2673_, 1);
v_msg1_2655_ = v_a_2674_;
goto v___jp_2654_;
}
else
{
lean_object* v_a_2675_; lean_object* v___x_2677_; uint8_t v_isShared_2678_; uint8_t v_isSharedCheck_2682_; 
lean_dec(v_a_2650_);
lean_dec(v_r_2647_);
lean_dec_ref(v___x_2637_);
v_a_2675_ = lean_ctor_get(v___x_2673_, 0);
v_isSharedCheck_2682_ = !lean_is_exclusive(v___x_2673_);
if (v_isSharedCheck_2682_ == 0)
{
v___x_2677_ = v___x_2673_;
v_isShared_2678_ = v_isSharedCheck_2682_;
goto v_resetjp_2676_;
}
else
{
lean_inc(v_a_2675_);
lean_dec(v___x_2673_);
v___x_2677_ = lean_box(0);
v_isShared_2678_ = v_isSharedCheck_2682_;
goto v_resetjp_2676_;
}
v_resetjp_2676_:
{
lean_object* v___x_2680_; 
if (v_isShared_2678_ == 0)
{
v___x_2680_ = v___x_2677_;
goto v_reusejp_2679_;
}
else
{
lean_object* v_reuseFailAlloc_2681_; 
v_reuseFailAlloc_2681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2681_, 0, v_a_2675_);
v___x_2680_ = v_reuseFailAlloc_2681_;
goto v_reusejp_2679_;
}
v_reusejp_2679_:
{
return v___x_2680_;
}
}
}
}
v___jp_2683_:
{
lean_object* v___x_2687_; lean_object* v___x_2688_; 
v___x_2687_ = lean_unsigned_to_nat(2u);
v___x_2688_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2688_, 0, v___x_2687_);
lean_ctor_set(v___x_2688_, 1, v_msg1_2684_);
if (lean_obj_tag(v_more_2638_) == 0)
{
lean_dec(v_v_2645_);
v_msg1_2655_ = v___x_2688_;
goto v___jp_2654_;
}
else
{
lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; 
v___x_2689_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__1));
v___x_2690_ = l_Lean_Elab_macroAttribute;
lean_inc_ref(v___x_2637_);
v___x_2691_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2637_, v_v_2645_, lean_box(0), v___x_2689_, v___x_2690_, v___x_2688_, v___y_2685_, v___y_2686_);
if (lean_obj_tag(v___x_2691_) == 0)
{
if (lean_obj_tag(v_catName_2639_) == 1)
{
lean_object* v_pre_2692_; 
v_pre_2692_ = lean_ctor_get(v_catName_2639_, 0);
if (lean_obj_tag(v_pre_2692_) == 0)
{
lean_object* v_a_2693_; lean_object* v_str_2694_; lean_object* v___x_2695_; uint8_t v___x_2696_; 
v_a_2693_ = lean_ctor_get(v___x_2691_, 0);
lean_inc(v_a_2693_);
lean_dec_ref_known(v___x_2691_, 1);
v_str_2694_ = lean_ctor_get(v_catName_2639_, 1);
v___x_2695_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2));
v___x_2696_ = lean_string_dec_eq(v_str_2694_, v___x_2695_);
if (v___x_2696_ == 0)
{
lean_object* v___x_2697_; uint8_t v___x_2698_; 
v___x_2697_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3));
v___x_2698_ = lean_string_dec_eq(v_str_2694_, v___x_2697_);
if (v___x_2698_ == 0)
{
lean_object* v___x_2699_; uint8_t v___x_2700_; 
v___x_2699_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4));
v___x_2700_ = lean_string_dec_eq(v_str_2694_, v___x_2699_);
if (v___x_2700_ == 0)
{
lean_object* v___x_2701_; uint8_t v___x_2702_; 
v___x_2701_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5));
v___x_2702_ = lean_string_dec_eq(v_str_2694_, v___x_2701_);
if (v___x_2702_ == 0)
{
lean_dec(v_v_2645_);
v_msg1_2655_ = v_a_2693_;
goto v___jp_2654_;
}
else
{
v___y_2668_ = v_a_2693_;
v___y_2669_ = v___y_2685_;
v___y_2670_ = v___y_2686_;
goto v___jp_2667_;
}
}
else
{
v___y_2668_ = v_a_2693_;
v___y_2669_ = v___y_2685_;
v___y_2670_ = v___y_2686_;
goto v___jp_2667_;
}
}
else
{
lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; 
v___x_2703_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__6));
v___x_2704_ = l_Lean_Elab_Command_commandElabAttribute;
lean_inc_ref(v___x_2637_);
v___x_2705_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2637_, v_v_2645_, lean_box(0), v___x_2703_, v___x_2704_, v_a_2693_, v___y_2685_, v___y_2686_);
lean_dec(v_v_2645_);
if (lean_obj_tag(v___x_2705_) == 0)
{
lean_object* v_a_2706_; 
v_a_2706_ = lean_ctor_get(v___x_2705_, 0);
lean_inc(v_a_2706_);
lean_dec_ref_known(v___x_2705_, 1);
v_msg1_2655_ = v_a_2706_;
goto v___jp_2654_;
}
else
{
lean_object* v_a_2707_; lean_object* v___x_2709_; uint8_t v_isShared_2710_; uint8_t v_isSharedCheck_2714_; 
lean_dec(v_a_2650_);
lean_dec(v_r_2647_);
lean_dec_ref(v___x_2637_);
v_a_2707_ = lean_ctor_get(v___x_2705_, 0);
v_isSharedCheck_2714_ = !lean_is_exclusive(v___x_2705_);
if (v_isSharedCheck_2714_ == 0)
{
v___x_2709_ = v___x_2705_;
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
else
{
lean_inc(v_a_2707_);
lean_dec(v___x_2705_);
v___x_2709_ = lean_box(0);
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
v_resetjp_2708_:
{
lean_object* v___x_2712_; 
if (v_isShared_2710_ == 0)
{
v___x_2712_ = v___x_2709_;
goto v_reusejp_2711_;
}
else
{
lean_object* v_reuseFailAlloc_2713_; 
v_reuseFailAlloc_2713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2713_, 0, v_a_2707_);
v___x_2712_ = v_reuseFailAlloc_2713_;
goto v_reusejp_2711_;
}
v_reusejp_2711_:
{
return v___x_2712_;
}
}
}
}
}
else
{
lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; 
v___x_2715_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__7));
v___x_2716_ = l_Lean_Elab_Term_termElabAttribute;
lean_inc_ref(v___x_2637_);
v___x_2717_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2637_, v_v_2645_, lean_box(0), v___x_2715_, v___x_2716_, v_a_2693_, v___y_2685_, v___y_2686_);
lean_dec(v_v_2645_);
if (lean_obj_tag(v___x_2717_) == 0)
{
lean_object* v_a_2718_; 
v_a_2718_ = lean_ctor_get(v___x_2717_, 0);
lean_inc(v_a_2718_);
lean_dec_ref_known(v___x_2717_, 1);
v_msg1_2655_ = v_a_2718_;
goto v___jp_2654_;
}
else
{
lean_object* v_a_2719_; lean_object* v___x_2721_; uint8_t v_isShared_2722_; uint8_t v_isSharedCheck_2726_; 
lean_dec(v_a_2650_);
lean_dec(v_r_2647_);
lean_dec_ref(v___x_2637_);
v_a_2719_ = lean_ctor_get(v___x_2717_, 0);
v_isSharedCheck_2726_ = !lean_is_exclusive(v___x_2717_);
if (v_isSharedCheck_2726_ == 0)
{
v___x_2721_ = v___x_2717_;
v_isShared_2722_ = v_isSharedCheck_2726_;
goto v_resetjp_2720_;
}
else
{
lean_inc(v_a_2719_);
lean_dec(v___x_2717_);
v___x_2721_ = lean_box(0);
v_isShared_2722_ = v_isSharedCheck_2726_;
goto v_resetjp_2720_;
}
v_resetjp_2720_:
{
lean_object* v___x_2724_; 
if (v_isShared_2722_ == 0)
{
v___x_2724_ = v___x_2721_;
goto v_reusejp_2723_;
}
else
{
lean_object* v_reuseFailAlloc_2725_; 
v_reuseFailAlloc_2725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2725_, 0, v_a_2719_);
v___x_2724_ = v_reuseFailAlloc_2725_;
goto v_reusejp_2723_;
}
v_reusejp_2723_:
{
return v___x_2724_;
}
}
}
}
}
else
{
lean_object* v_a_2727_; 
lean_dec(v_v_2645_);
v_a_2727_ = lean_ctor_get(v___x_2691_, 0);
lean_inc(v_a_2727_);
lean_dec_ref_known(v___x_2691_, 1);
v_msg1_2655_ = v_a_2727_;
goto v___jp_2654_;
}
}
else
{
lean_object* v_a_2728_; 
lean_dec(v_v_2645_);
v_a_2728_ = lean_ctor_get(v___x_2691_, 0);
lean_inc(v_a_2728_);
lean_dec_ref_known(v___x_2691_, 1);
v_msg1_2655_ = v_a_2728_;
goto v___jp_2654_;
}
}
else
{
lean_object* v_a_2729_; lean_object* v___x_2731_; uint8_t v_isShared_2732_; uint8_t v_isSharedCheck_2736_; 
lean_dec(v_a_2650_);
lean_dec(v_r_2647_);
lean_dec(v_v_2645_);
lean_dec_ref(v___x_2637_);
v_a_2729_ = lean_ctor_get(v___x_2691_, 0);
v_isSharedCheck_2736_ = !lean_is_exclusive(v___x_2691_);
if (v_isSharedCheck_2736_ == 0)
{
v___x_2731_ = v___x_2691_;
v_isShared_2732_ = v_isSharedCheck_2736_;
goto v_resetjp_2730_;
}
else
{
lean_inc(v_a_2729_);
lean_dec(v___x_2691_);
v___x_2731_ = lean_box(0);
v_isShared_2732_ = v_isSharedCheck_2736_;
goto v_resetjp_2730_;
}
v_resetjp_2730_:
{
lean_object* v___x_2734_; 
if (v_isShared_2732_ == 0)
{
v___x_2734_ = v___x_2731_;
goto v_reusejp_2733_;
}
else
{
lean_object* v_reuseFailAlloc_2735_; 
v_reuseFailAlloc_2735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2735_, 0, v_a_2729_);
v___x_2734_ = v_reuseFailAlloc_2735_;
goto v_reusejp_2733_;
}
v_reusejp_2733_:
{
return v___x_2734_;
}
}
}
}
}
}
else
{
lean_object* v_a_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2776_; 
lean_dec_ref(v___x_2661_);
lean_dec(v_a_2650_);
lean_dec(v_r_2647_);
lean_dec(v_v_2645_);
lean_dec_ref(v___x_2637_);
v_a_2762_ = lean_ctor_get(v___x_2665_, 0);
v_isSharedCheck_2776_ = !lean_is_exclusive(v___x_2665_);
if (v_isSharedCheck_2776_ == 0)
{
v___x_2764_ = v___x_2665_;
v_isShared_2765_ = v_isSharedCheck_2776_;
goto v_resetjp_2763_;
}
else
{
lean_inc(v_a_2762_);
lean_dec(v___x_2665_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2776_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v_ref_2766_; lean_object* v___x_2767_; lean_object* v___x_2769_; 
v_ref_2766_ = lean_ctor_get(v___y_2642_, 7);
v___x_2767_ = lean_io_error_to_string(v_a_2762_);
if (v_isShared_2653_ == 0)
{
lean_ctor_set_tag(v___x_2652_, 3);
lean_ctor_set(v___x_2652_, 0, v___x_2767_);
v___x_2769_ = v___x_2652_;
goto v_reusejp_2768_;
}
else
{
lean_object* v_reuseFailAlloc_2775_; 
v_reuseFailAlloc_2775_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2775_, 0, v___x_2767_);
v___x_2769_ = v_reuseFailAlloc_2775_;
goto v_reusejp_2768_;
}
v_reusejp_2768_:
{
lean_object* v___x_2770_; lean_object* v___x_2771_; lean_object* v___x_2773_; 
v___x_2770_ = l_Lean_MessageData_ofFormat(v___x_2769_);
lean_inc(v_ref_2766_);
v___x_2771_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2771_, 0, v_ref_2766_);
lean_ctor_set(v___x_2771_, 1, v___x_2770_);
if (v_isShared_2765_ == 0)
{
lean_ctor_set(v___x_2764_, 0, v___x_2771_);
v___x_2773_ = v___x_2764_;
goto v_reusejp_2772_;
}
else
{
lean_object* v_reuseFailAlloc_2774_; 
v_reuseFailAlloc_2774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2774_, 0, v___x_2771_);
v___x_2773_ = v_reuseFailAlloc_2774_;
goto v_reusejp_2772_;
}
v_reusejp_2772_:
{
return v___x_2773_;
}
}
}
}
v___jp_2654_:
{
lean_object* v___x_2656_; lean_object* v___x_2657_; lean_object* v___x_2658_; 
v___x_2656_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2656_, 0, v_a_2650_);
lean_ctor_set(v___x_2656_, 1, v_msg1_2655_);
v___x_2657_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1);
v___x_2658_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2658_, 0, v___x_2656_);
lean_ctor_set(v___x_2658_, 1, v___x_2657_);
v_init_2640_ = v___x_2658_;
v_x_2641_ = v_r_2647_;
goto _start;
}
}
}
else
{
lean_dec(v_r_2647_);
lean_dec(v_v_2645_);
lean_dec_ref(v___x_2637_);
return v___x_2648_;
}
}
else
{
lean_object* v___x_2778_; lean_object* v___x_2779_; 
lean_dec_ref(v___x_2637_);
v___x_2778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2778_, 0, v_init_2640_);
v___x_2779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2779_, 0, v___x_2778_);
return v___x_2779_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___boxed(lean_object* v___x_2780_, lean_object* v_more_2781_, lean_object* v_catName_2782_, lean_object* v_init_2783_, lean_object* v_x_2784_, lean_object* v___y_2785_, lean_object* v___y_2786_, lean_object* v___y_2787_){
_start:
{
lean_object* v_res_2788_; 
v_res_2788_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19(v___x_2780_, v_more_2781_, v_catName_2782_, v_init_2783_, v_x_2784_, v___y_2785_, v___y_2786_);
lean_dec(v___y_2786_);
lean_dec_ref(v___y_2785_);
lean_dec(v_catName_2782_);
lean_dec(v_more_2781_);
return v_res_2788_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14(lean_object* v___x_2789_, lean_object* v_more_2790_, lean_object* v_catName_2791_, lean_object* v_init_2792_, lean_object* v_x_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_){
_start:
{
if (lean_obj_tag(v_x_2793_) == 0)
{
lean_object* v_v_2797_; lean_object* v_l_2798_; lean_object* v_r_2799_; lean_object* v___x_2800_; 
v_v_2797_ = lean_ctor_get(v_x_2793_, 2);
lean_inc(v_v_2797_);
v_l_2798_ = lean_ctor_get(v_x_2793_, 3);
lean_inc(v_l_2798_);
v_r_2799_ = lean_ctor_get(v_x_2793_, 4);
lean_inc(v_r_2799_);
lean_dec_ref_known(v_x_2793_, 5);
lean_inc_ref(v___x_2789_);
v___x_2800_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19(v___x_2789_, v_more_2790_, v_catName_2791_, v_init_2792_, v_l_2798_, v___y_2794_, v___y_2795_);
if (lean_obj_tag(v___x_2800_) == 0)
{
lean_object* v_a_2801_; lean_object* v_a_2802_; lean_object* v___x_2804_; uint8_t v_isShared_2805_; uint8_t v_isSharedCheck_2929_; 
v_a_2801_ = lean_ctor_get(v___x_2800_, 0);
lean_inc(v_a_2801_);
lean_dec_ref_known(v___x_2800_, 1);
v_a_2802_ = lean_ctor_get(v_a_2801_, 0);
v_isSharedCheck_2929_ = !lean_is_exclusive(v_a_2801_);
if (v_isSharedCheck_2929_ == 0)
{
v___x_2804_ = v_a_2801_;
v_isShared_2805_ = v_isSharedCheck_2929_;
goto v_resetjp_2803_;
}
else
{
lean_inc(v_a_2802_);
lean_dec(v_a_2801_);
v___x_2804_ = lean_box(0);
v_isShared_2805_ = v_isSharedCheck_2929_;
goto v_resetjp_2803_;
}
v_resetjp_2803_:
{
lean_object* v_msg1_2807_; lean_object* v___x_2812_; lean_object* v___x_2813_; uint8_t v___x_2814_; lean_object* v___x_2815_; lean_object* v___x_2816_; lean_object* v___x_2817_; 
v___x_2812_ = lean_box(0);
lean_inc_n(v_v_2797_, 2);
v___x_2813_ = l_Lean_mkConst(v_v_2797_, v___x_2812_);
v___x_2814_ = 1;
v___x_2815_ = l_Lean_Options_empty;
v___x_2816_ = lean_box(0);
lean_inc_ref(v___x_2789_);
v___x_2817_ = l_Lean_findDocString_x3f(v___x_2789_, v_v_2797_, v___x_2814_, v___x_2815_, v___x_2816_, v___x_2812_);
if (lean_obj_tag(v___x_2817_) == 0)
{
lean_object* v_a_2818_; lean_object* v___y_2820_; lean_object* v___y_2821_; lean_object* v___y_2822_; lean_object* v_msg1_2836_; lean_object* v___y_2837_; lean_object* v___y_2838_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; 
lean_del_object(v___x_2804_);
v_a_2818_ = lean_ctor_get(v___x_2817_, 0);
lean_inc(v_a_2818_);
lean_dec_ref_known(v___x_2817_, 1);
v___x_2889_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__9);
v___x_2890_ = l_Lean_MessageData_ofExpr(v___x_2813_);
v___x_2891_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2891_, 0, v___x_2889_);
lean_ctor_set(v___x_2891_, 1, v___x_2890_);
v___x_2892_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7);
v___x_2893_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2893_, 0, v___x_2891_);
lean_ctor_set(v___x_2893_, 1, v___x_2892_);
if (lean_obj_tag(v_a_2818_) == 1)
{
lean_object* v_val_2894_; lean_object* v___x_2896_; uint8_t v_isShared_2897_; uint8_t v_isSharedCheck_2913_; 
v_val_2894_ = lean_ctor_get(v_a_2818_, 0);
v_isSharedCheck_2913_ = !lean_is_exclusive(v_a_2818_);
if (v_isSharedCheck_2913_ == 0)
{
v___x_2896_ = v_a_2818_;
v_isShared_2897_ = v_isSharedCheck_2913_;
goto v_resetjp_2895_;
}
else
{
lean_inc(v_val_2894_);
lean_dec(v_a_2818_);
v___x_2896_ = lean_box(0);
v_isShared_2897_ = v_isSharedCheck_2913_;
goto v_resetjp_2895_;
}
v_resetjp_2895_:
{
lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v_str_2902_; lean_object* v_startInclusive_2903_; lean_object* v_endExclusive_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; lean_object* v___x_2909_; 
v___x_2898_ = lean_unsigned_to_nat(0u);
v___x_2899_ = lean_string_utf8_byte_size(v_val_2894_);
v___x_2900_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2900_, 0, v_val_2894_);
lean_ctor_set(v___x_2900_, 1, v___x_2898_);
lean_ctor_set(v___x_2900_, 2, v___x_2899_);
v___x_2901_ = l_String_Slice_trimAscii(v___x_2900_);
v_str_2902_ = lean_ctor_get(v___x_2901_, 0);
lean_inc_ref(v_str_2902_);
v_startInclusive_2903_ = lean_ctor_get(v___x_2901_, 1);
lean_inc(v_startInclusive_2903_);
v_endExclusive_2904_ = lean_ctor_get(v___x_2901_, 2);
lean_inc(v_endExclusive_2904_);
lean_dec_ref(v___x_2901_);
v___x_2905_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
v___x_2906_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2906_, 0, v___x_2893_);
lean_ctor_set(v___x_2906_, 1, v___x_2905_);
v___x_2907_ = lean_string_utf8_extract_fast(v_str_2902_, v_startInclusive_2903_, v_endExclusive_2904_);
lean_dec(v_endExclusive_2904_);
lean_dec(v_startInclusive_2903_);
lean_dec_ref(v_str_2902_);
if (v_isShared_2897_ == 0)
{
lean_ctor_set_tag(v___x_2896_, 3);
lean_ctor_set(v___x_2896_, 0, v___x_2907_);
v___x_2909_ = v___x_2896_;
goto v_reusejp_2908_;
}
else
{
lean_object* v_reuseFailAlloc_2912_; 
v_reuseFailAlloc_2912_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2912_, 0, v___x_2907_);
v___x_2909_ = v_reuseFailAlloc_2912_;
goto v_reusejp_2908_;
}
v_reusejp_2908_:
{
lean_object* v___x_2910_; lean_object* v___x_2911_; 
v___x_2910_ = l_Lean_MessageData_ofFormat(v___x_2909_);
v___x_2911_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2911_, 0, v___x_2906_);
lean_ctor_set(v___x_2911_, 1, v___x_2910_);
v_msg1_2836_ = v___x_2911_;
v___y_2837_ = v___y_2794_;
v___y_2838_ = v___y_2795_;
goto v___jp_2835_;
}
}
}
else
{
lean_dec(v_a_2818_);
v_msg1_2836_ = v___x_2893_;
v___y_2837_ = v___y_2794_;
v___y_2838_ = v___y_2795_;
goto v___jp_2835_;
}
v___jp_2819_:
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; 
v___x_2823_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__0));
v___x_2824_ = l_Lean_Elab_Tactic_tacticElabAttribute;
lean_inc_ref(v___x_2789_);
v___x_2825_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2789_, v_v_2797_, lean_box(0), v___x_2823_, v___x_2824_, v___y_2820_, v___y_2821_, v___y_2822_);
lean_dec(v_v_2797_);
if (lean_obj_tag(v___x_2825_) == 0)
{
lean_object* v_a_2826_; 
v_a_2826_ = lean_ctor_get(v___x_2825_, 0);
lean_inc(v_a_2826_);
lean_dec_ref_known(v___x_2825_, 1);
v_msg1_2807_ = v_a_2826_;
goto v___jp_2806_;
}
else
{
lean_object* v_a_2827_; lean_object* v___x_2829_; uint8_t v_isShared_2830_; uint8_t v_isSharedCheck_2834_; 
lean_dec(v_a_2802_);
lean_dec(v_r_2799_);
lean_dec_ref(v___x_2789_);
v_a_2827_ = lean_ctor_get(v___x_2825_, 0);
v_isSharedCheck_2834_ = !lean_is_exclusive(v___x_2825_);
if (v_isSharedCheck_2834_ == 0)
{
v___x_2829_ = v___x_2825_;
v_isShared_2830_ = v_isSharedCheck_2834_;
goto v_resetjp_2828_;
}
else
{
lean_inc(v_a_2827_);
lean_dec(v___x_2825_);
v___x_2829_ = lean_box(0);
v_isShared_2830_ = v_isSharedCheck_2834_;
goto v_resetjp_2828_;
}
v_resetjp_2828_:
{
lean_object* v___x_2832_; 
if (v_isShared_2830_ == 0)
{
v___x_2832_ = v___x_2829_;
goto v_reusejp_2831_;
}
else
{
lean_object* v_reuseFailAlloc_2833_; 
v_reuseFailAlloc_2833_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2833_, 0, v_a_2827_);
v___x_2832_ = v_reuseFailAlloc_2833_;
goto v_reusejp_2831_;
}
v_reusejp_2831_:
{
return v___x_2832_;
}
}
}
}
v___jp_2835_:
{
lean_object* v___x_2839_; lean_object* v___x_2840_; 
v___x_2839_ = lean_unsigned_to_nat(2u);
v___x_2840_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2840_, 0, v___x_2839_);
lean_ctor_set(v___x_2840_, 1, v_msg1_2836_);
if (lean_obj_tag(v_more_2790_) == 0)
{
lean_dec(v_v_2797_);
v_msg1_2807_ = v___x_2840_;
goto v___jp_2806_;
}
else
{
lean_object* v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; 
v___x_2841_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__1));
v___x_2842_ = l_Lean_Elab_macroAttribute;
lean_inc_ref(v___x_2789_);
v___x_2843_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2789_, v_v_2797_, lean_box(0), v___x_2841_, v___x_2842_, v___x_2840_, v___y_2837_, v___y_2838_);
if (lean_obj_tag(v___x_2843_) == 0)
{
if (lean_obj_tag(v_catName_2791_) == 1)
{
lean_object* v_pre_2844_; 
v_pre_2844_ = lean_ctor_get(v_catName_2791_, 0);
if (lean_obj_tag(v_pre_2844_) == 0)
{
lean_object* v_a_2845_; lean_object* v_str_2846_; lean_object* v___x_2847_; uint8_t v___x_2848_; 
v_a_2845_ = lean_ctor_get(v___x_2843_, 0);
lean_inc(v_a_2845_);
lean_dec_ref_known(v___x_2843_, 1);
v_str_2846_ = lean_ctor_get(v_catName_2791_, 1);
v___x_2847_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2));
v___x_2848_ = lean_string_dec_eq(v_str_2846_, v___x_2847_);
if (v___x_2848_ == 0)
{
lean_object* v___x_2849_; uint8_t v___x_2850_; 
v___x_2849_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3));
v___x_2850_ = lean_string_dec_eq(v_str_2846_, v___x_2849_);
if (v___x_2850_ == 0)
{
lean_object* v___x_2851_; uint8_t v___x_2852_; 
v___x_2851_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4));
v___x_2852_ = lean_string_dec_eq(v_str_2846_, v___x_2851_);
if (v___x_2852_ == 0)
{
lean_object* v___x_2853_; uint8_t v___x_2854_; 
v___x_2853_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5));
v___x_2854_ = lean_string_dec_eq(v_str_2846_, v___x_2853_);
if (v___x_2854_ == 0)
{
lean_dec(v_v_2797_);
v_msg1_2807_ = v_a_2845_;
goto v___jp_2806_;
}
else
{
v___y_2820_ = v_a_2845_;
v___y_2821_ = v___y_2837_;
v___y_2822_ = v___y_2838_;
goto v___jp_2819_;
}
}
else
{
v___y_2820_ = v_a_2845_;
v___y_2821_ = v___y_2837_;
v___y_2822_ = v___y_2838_;
goto v___jp_2819_;
}
}
else
{
lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v___x_2857_; 
v___x_2855_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__6));
v___x_2856_ = l_Lean_Elab_Command_commandElabAttribute;
lean_inc_ref(v___x_2789_);
v___x_2857_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2789_, v_v_2797_, lean_box(0), v___x_2855_, v___x_2856_, v_a_2845_, v___y_2837_, v___y_2838_);
lean_dec(v_v_2797_);
if (lean_obj_tag(v___x_2857_) == 0)
{
lean_object* v_a_2858_; 
v_a_2858_ = lean_ctor_get(v___x_2857_, 0);
lean_inc(v_a_2858_);
lean_dec_ref_known(v___x_2857_, 1);
v_msg1_2807_ = v_a_2858_;
goto v___jp_2806_;
}
else
{
lean_object* v_a_2859_; lean_object* v___x_2861_; uint8_t v_isShared_2862_; uint8_t v_isSharedCheck_2866_; 
lean_dec(v_a_2802_);
lean_dec(v_r_2799_);
lean_dec_ref(v___x_2789_);
v_a_2859_ = lean_ctor_get(v___x_2857_, 0);
v_isSharedCheck_2866_ = !lean_is_exclusive(v___x_2857_);
if (v_isSharedCheck_2866_ == 0)
{
v___x_2861_ = v___x_2857_;
v_isShared_2862_ = v_isSharedCheck_2866_;
goto v_resetjp_2860_;
}
else
{
lean_inc(v_a_2859_);
lean_dec(v___x_2857_);
v___x_2861_ = lean_box(0);
v_isShared_2862_ = v_isSharedCheck_2866_;
goto v_resetjp_2860_;
}
v_resetjp_2860_:
{
lean_object* v___x_2864_; 
if (v_isShared_2862_ == 0)
{
v___x_2864_ = v___x_2861_;
goto v_reusejp_2863_;
}
else
{
lean_object* v_reuseFailAlloc_2865_; 
v_reuseFailAlloc_2865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2865_, 0, v_a_2859_);
v___x_2864_ = v_reuseFailAlloc_2865_;
goto v_reusejp_2863_;
}
v_reusejp_2863_:
{
return v___x_2864_;
}
}
}
}
}
else
{
lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; 
v___x_2867_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__7));
v___x_2868_ = l_Lean_Elab_Term_termElabAttribute;
lean_inc_ref(v___x_2789_);
v___x_2869_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___lam__0(v___x_2789_, v_v_2797_, lean_box(0), v___x_2867_, v___x_2868_, v_a_2845_, v___y_2837_, v___y_2838_);
lean_dec(v_v_2797_);
if (lean_obj_tag(v___x_2869_) == 0)
{
lean_object* v_a_2870_; 
v_a_2870_ = lean_ctor_get(v___x_2869_, 0);
lean_inc(v_a_2870_);
lean_dec_ref_known(v___x_2869_, 1);
v_msg1_2807_ = v_a_2870_;
goto v___jp_2806_;
}
else
{
lean_object* v_a_2871_; lean_object* v___x_2873_; uint8_t v_isShared_2874_; uint8_t v_isSharedCheck_2878_; 
lean_dec(v_a_2802_);
lean_dec(v_r_2799_);
lean_dec_ref(v___x_2789_);
v_a_2871_ = lean_ctor_get(v___x_2869_, 0);
v_isSharedCheck_2878_ = !lean_is_exclusive(v___x_2869_);
if (v_isSharedCheck_2878_ == 0)
{
v___x_2873_ = v___x_2869_;
v_isShared_2874_ = v_isSharedCheck_2878_;
goto v_resetjp_2872_;
}
else
{
lean_inc(v_a_2871_);
lean_dec(v___x_2869_);
v___x_2873_ = lean_box(0);
v_isShared_2874_ = v_isSharedCheck_2878_;
goto v_resetjp_2872_;
}
v_resetjp_2872_:
{
lean_object* v___x_2876_; 
if (v_isShared_2874_ == 0)
{
v___x_2876_ = v___x_2873_;
goto v_reusejp_2875_;
}
else
{
lean_object* v_reuseFailAlloc_2877_; 
v_reuseFailAlloc_2877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2877_, 0, v_a_2871_);
v___x_2876_ = v_reuseFailAlloc_2877_;
goto v_reusejp_2875_;
}
v_reusejp_2875_:
{
return v___x_2876_;
}
}
}
}
}
else
{
lean_object* v_a_2879_; 
lean_dec(v_v_2797_);
v_a_2879_ = lean_ctor_get(v___x_2843_, 0);
lean_inc(v_a_2879_);
lean_dec_ref_known(v___x_2843_, 1);
v_msg1_2807_ = v_a_2879_;
goto v___jp_2806_;
}
}
else
{
lean_object* v_a_2880_; 
lean_dec(v_v_2797_);
v_a_2880_ = lean_ctor_get(v___x_2843_, 0);
lean_inc(v_a_2880_);
lean_dec_ref_known(v___x_2843_, 1);
v_msg1_2807_ = v_a_2880_;
goto v___jp_2806_;
}
}
else
{
lean_object* v_a_2881_; lean_object* v___x_2883_; uint8_t v_isShared_2884_; uint8_t v_isSharedCheck_2888_; 
lean_dec(v_a_2802_);
lean_dec(v_r_2799_);
lean_dec(v_v_2797_);
lean_dec_ref(v___x_2789_);
v_a_2881_ = lean_ctor_get(v___x_2843_, 0);
v_isSharedCheck_2888_ = !lean_is_exclusive(v___x_2843_);
if (v_isSharedCheck_2888_ == 0)
{
v___x_2883_ = v___x_2843_;
v_isShared_2884_ = v_isSharedCheck_2888_;
goto v_resetjp_2882_;
}
else
{
lean_inc(v_a_2881_);
lean_dec(v___x_2843_);
v___x_2883_ = lean_box(0);
v_isShared_2884_ = v_isSharedCheck_2888_;
goto v_resetjp_2882_;
}
v_resetjp_2882_:
{
lean_object* v___x_2886_; 
if (v_isShared_2884_ == 0)
{
v___x_2886_ = v___x_2883_;
goto v_reusejp_2885_;
}
else
{
lean_object* v_reuseFailAlloc_2887_; 
v_reuseFailAlloc_2887_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2887_, 0, v_a_2881_);
v___x_2886_ = v_reuseFailAlloc_2887_;
goto v_reusejp_2885_;
}
v_reusejp_2885_:
{
return v___x_2886_;
}
}
}
}
}
}
else
{
lean_object* v_a_2914_; lean_object* v___x_2916_; uint8_t v_isShared_2917_; uint8_t v_isSharedCheck_2928_; 
lean_dec_ref(v___x_2813_);
lean_dec(v_a_2802_);
lean_dec(v_r_2799_);
lean_dec(v_v_2797_);
lean_dec_ref(v___x_2789_);
v_a_2914_ = lean_ctor_get(v___x_2817_, 0);
v_isSharedCheck_2928_ = !lean_is_exclusive(v___x_2817_);
if (v_isSharedCheck_2928_ == 0)
{
v___x_2916_ = v___x_2817_;
v_isShared_2917_ = v_isSharedCheck_2928_;
goto v_resetjp_2915_;
}
else
{
lean_inc(v_a_2914_);
lean_dec(v___x_2817_);
v___x_2916_ = lean_box(0);
v_isShared_2917_ = v_isSharedCheck_2928_;
goto v_resetjp_2915_;
}
v_resetjp_2915_:
{
lean_object* v_ref_2918_; lean_object* v___x_2919_; lean_object* v___x_2921_; 
v_ref_2918_ = lean_ctor_get(v___y_2794_, 7);
v___x_2919_ = lean_io_error_to_string(v_a_2914_);
if (v_isShared_2805_ == 0)
{
lean_ctor_set_tag(v___x_2804_, 3);
lean_ctor_set(v___x_2804_, 0, v___x_2919_);
v___x_2921_ = v___x_2804_;
goto v_reusejp_2920_;
}
else
{
lean_object* v_reuseFailAlloc_2927_; 
v_reuseFailAlloc_2927_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2927_, 0, v___x_2919_);
v___x_2921_ = v_reuseFailAlloc_2927_;
goto v_reusejp_2920_;
}
v_reusejp_2920_:
{
lean_object* v___x_2922_; lean_object* v___x_2923_; lean_object* v___x_2925_; 
v___x_2922_ = l_Lean_MessageData_ofFormat(v___x_2921_);
lean_inc(v_ref_2918_);
v___x_2923_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2923_, 0, v_ref_2918_);
lean_ctor_set(v___x_2923_, 1, v___x_2922_);
if (v_isShared_2917_ == 0)
{
lean_ctor_set(v___x_2916_, 0, v___x_2923_);
v___x_2925_ = v___x_2916_;
goto v_reusejp_2924_;
}
else
{
lean_object* v_reuseFailAlloc_2926_; 
v_reuseFailAlloc_2926_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2926_, 0, v___x_2923_);
v___x_2925_ = v_reuseFailAlloc_2926_;
goto v_reusejp_2924_;
}
v_reusejp_2924_:
{
return v___x_2925_;
}
}
}
}
v___jp_2806_:
{
lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; 
v___x_2808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2808_, 0, v_a_2802_);
lean_ctor_set(v___x_2808_, 1, v_msg1_2807_);
v___x_2809_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1);
v___x_2810_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2810_, 0, v___x_2808_);
lean_ctor_set(v___x_2810_, 1, v___x_2809_);
v___x_2811_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19(v___x_2789_, v_more_2790_, v_catName_2791_, v___x_2810_, v_r_2799_, v___y_2794_, v___y_2795_);
return v___x_2811_;
}
}
}
else
{
lean_dec(v_r_2799_);
lean_dec(v_v_2797_);
lean_dec_ref(v___x_2789_);
return v___x_2800_;
}
}
else
{
lean_object* v___x_2930_; lean_object* v___x_2931_; 
lean_dec_ref(v___x_2789_);
v___x_2930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2930_, 0, v_init_2792_);
v___x_2931_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2931_, 0, v___x_2930_);
return v___x_2931_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14___boxed(lean_object* v___x_2932_, lean_object* v_more_2933_, lean_object* v_catName_2934_, lean_object* v_init_2935_, lean_object* v_x_2936_, lean_object* v___y_2937_, lean_object* v___y_2938_, lean_object* v___y_2939_){
_start:
{
lean_object* v_res_2940_; 
v_res_2940_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14(v___x_2932_, v_more_2933_, v_catName_2934_, v_init_2935_, v_x_2936_, v___y_2937_, v___y_2938_);
lean_dec(v___y_2938_);
lean_dec_ref(v___y_2937_);
lean_dec(v_catName_2934_);
lean_dec(v_more_2933_);
return v_res_2940_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10_spec__15(lean_object* v_msg_2941_){
_start:
{
lean_object* v___x_2942_; lean_object* v___x_2943_; 
v___x_2942_ = lean_unsigned_to_nat(0u);
v___x_2943_ = lean_panic_fn_borrowed(v___x_2942_, v_msg_2941_);
return v___x_2943_;
}
}
static lean_object* _init_lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__3(void){
_start:
{
lean_object* v___x_2947_; lean_object* v___x_2948_; lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; lean_object* v___x_2952_; 
v___x_2947_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__2));
v___x_2948_ = lean_unsigned_to_nat(11u);
v___x_2949_ = lean_unsigned_to_nat(163u);
v___x_2950_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__1));
v___x_2951_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__0));
v___x_2952_ = l_mkPanicMessageWithDecl(v___x_2951_, v___x_2950_, v___x_2949_, v___x_2948_, v___x_2947_);
return v___x_2952_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10(lean_object* v_a_2953_, lean_object* v_x_2954_){
_start:
{
if (lean_obj_tag(v_x_2954_) == 0)
{
lean_object* v___x_2955_; lean_object* v___x_2956_; 
v___x_2955_ = lean_obj_once(&lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__3, &lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__3_once, _init_lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___closed__3);
v___x_2956_ = lp_batteries_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10_spec__15(v___x_2955_);
return v___x_2956_;
}
else
{
lean_object* v_key_2957_; lean_object* v_value_2958_; lean_object* v_tail_2959_; uint8_t v___x_2960_; 
v_key_2957_ = lean_ctor_get(v_x_2954_, 0);
v_value_2958_ = lean_ctor_get(v_x_2954_, 1);
v_tail_2959_ = lean_ctor_get(v_x_2954_, 2);
v___x_2960_ = lean_string_dec_eq(v_key_2957_, v_a_2953_);
if (v___x_2960_ == 0)
{
v_x_2954_ = v_tail_2959_;
goto _start;
}
else
{
lean_inc(v_value_2958_);
return v_value_2958_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10___boxed(lean_object* v_a_2962_, lean_object* v_x_2963_){
_start:
{
lean_object* v_res_2964_; 
v_res_2964_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10(v_a_2962_, v_x_2963_);
lean_dec(v_x_2963_);
lean_dec_ref(v_a_2962_);
return v_res_2964_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6(lean_object* v_m_2965_, lean_object* v_a_2966_){
_start:
{
lean_object* v_buckets_2967_; lean_object* v___x_2968_; uint64_t v___x_2969_; uint64_t v___x_2970_; uint64_t v___x_2971_; uint64_t v_fold_2972_; uint64_t v___x_2973_; uint64_t v___x_2974_; uint64_t v___x_2975_; size_t v___x_2976_; size_t v___x_2977_; size_t v___x_2978_; size_t v___x_2979_; size_t v___x_2980_; lean_object* v___x_2981_; lean_object* v___x_2982_; 
v_buckets_2967_ = lean_ctor_get(v_m_2965_, 1);
v___x_2968_ = lean_array_get_size(v_buckets_2967_);
v___x_2969_ = lean_string_hash(v_a_2966_);
v___x_2970_ = 32ULL;
v___x_2971_ = lean_uint64_shift_right(v___x_2969_, v___x_2970_);
v_fold_2972_ = lean_uint64_xor(v___x_2969_, v___x_2971_);
v___x_2973_ = 16ULL;
v___x_2974_ = lean_uint64_shift_right(v_fold_2972_, v___x_2973_);
v___x_2975_ = lean_uint64_xor(v_fold_2972_, v___x_2974_);
v___x_2976_ = lean_uint64_to_usize(v___x_2975_);
v___x_2977_ = lean_usize_of_nat(v___x_2968_);
v___x_2978_ = ((size_t)1ULL);
v___x_2979_ = lean_usize_sub(v___x_2977_, v___x_2978_);
v___x_2980_ = lean_usize_land(v___x_2976_, v___x_2979_);
v___x_2981_ = lean_array_uget_borrowed(v_buckets_2967_, v___x_2980_);
v___x_2982_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6_spec__10(v_a_2966_, v___x_2981_);
return v___x_2982_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6___boxed(lean_object* v_m_2983_, lean_object* v_a_2984_){
_start:
{
lean_object* v_res_2985_; 
v_res_2985_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6(v_m_2983_, v_a_2984_);
lean_dec_ref(v_a_2984_);
lean_dec_ref(v_m_2983_);
return v_res_2985_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10___lam__0(lean_object* v___x_2986_, lean_object* v_x_2987_){
_start:
{
lean_object* v___x_2988_; 
v___x_2988_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__6(v___x_2986_, v_x_2987_);
return v___x_2988_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10___lam__0___boxed(lean_object* v___x_2989_, lean_object* v_x_2990_){
_start:
{
lean_object* v_res_2991_; 
v_res_2991_ = lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10___lam__0(v___x_2989_, v_x_2990_);
lean_dec_ref(v_x_2990_);
lean_dec_ref(v___x_2989_);
return v_res_2991_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10(lean_object* v___x_2992_, lean_object* v_x_2993_, lean_object* v_x_2994_){
_start:
{
if (lean_obj_tag(v_x_2994_) == 0)
{
lean_dec_ref(v___x_2992_);
return v_x_2993_;
}
else
{
lean_object* v_head_2995_; lean_object* v_tail_2996_; lean_object* v___f_2997_; lean_object* v___x_2998_; 
v_head_2995_ = lean_ctor_get(v_x_2994_, 0);
lean_inc(v_head_2995_);
v_tail_2996_ = lean_ctor_get(v_x_2994_, 1);
lean_inc(v_tail_2996_);
lean_dec_ref_known(v_x_2994_, 2);
lean_inc_ref(v___x_2992_);
v___f_2997_ = lean_alloc_closure((void*)(lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10___lam__0___boxed), 2, 1);
lean_closure_set(v___f_2997_, 0, v___x_2992_);
v___x_2998_ = lp_batteries_minOn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__9___redArg(v___f_2997_, v_x_2993_, v_head_2995_);
v_x_2993_ = v___x_2998_;
v_x_2994_ = v_tail_2996_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0(lean_object* v_fst_3002_, uint8_t v_snd_3003_, lean_object* v_arr_3004_){
_start:
{
lean_object* v___y_3006_; 
if (lean_obj_tag(v_arr_3004_) == 0)
{
lean_object* v___x_3011_; 
v___x_3011_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0___closed__0));
v___y_3006_ = v___x_3011_;
goto v___jp_3005_;
}
else
{
lean_object* v_val_3012_; 
v_val_3012_ = lean_ctor_get(v_arr_3004_, 0);
lean_inc(v_val_3012_);
lean_dec_ref_known(v_arr_3004_, 1);
v___y_3006_ = v_val_3012_;
goto v___jp_3005_;
}
v___jp_3005_:
{
lean_object* v___x_3007_; lean_object* v___x_3008_; lean_object* v___x_3009_; lean_object* v___x_3010_; 
v___x_3007_ = lean_box(v_snd_3003_);
v___x_3008_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3008_, 0, v_fst_3002_);
lean_ctor_set(v___x_3008_, 1, v___x_3007_);
v___x_3009_ = lean_array_push(v___y_3006_, v___x_3008_);
v___x_3010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3010_, 0, v___x_3009_);
return v___x_3010_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0___boxed(lean_object* v_fst_3013_, lean_object* v_snd_3014_, lean_object* v_arr_3015_){
_start:
{
uint8_t v_snd_22256__boxed_3016_; lean_object* v_res_3017_; 
v_snd_22256__boxed_3016_ = lean_unbox(v_snd_3014_);
v_res_3017_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0(v_fst_3013_, v_snd_22256__boxed_3016_, v_arr_3015_);
return v_res_3017_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(lean_object* v_fst_3018_, uint8_t v_snd_3019_, lean_object* v_k_3020_, lean_object* v_t_3021_){
_start:
{
if (lean_obj_tag(v_t_3021_) == 0)
{
lean_object* v_size_3022_; lean_object* v_k_3023_; lean_object* v_v_3024_; lean_object* v_l_3025_; lean_object* v_r_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3041_; 
v_size_3022_ = lean_ctor_get(v_t_3021_, 0);
v_k_3023_ = lean_ctor_get(v_t_3021_, 1);
v_v_3024_ = lean_ctor_get(v_t_3021_, 2);
v_l_3025_ = lean_ctor_get(v_t_3021_, 3);
v_r_3026_ = lean_ctor_get(v_t_3021_, 4);
v_isSharedCheck_3041_ = !lean_is_exclusive(v_t_3021_);
if (v_isSharedCheck_3041_ == 0)
{
v___x_3028_ = v_t_3021_;
v_isShared_3029_ = v_isSharedCheck_3041_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_r_3026_);
lean_inc(v_l_3025_);
lean_inc(v_v_3024_);
lean_inc(v_k_3023_);
lean_inc(v_size_3022_);
lean_dec(v_t_3021_);
v___x_3028_ = lean_box(0);
v_isShared_3029_ = v_isSharedCheck_3041_;
goto v_resetjp_3027_;
}
v_resetjp_3027_:
{
uint8_t v___x_3030_; 
v___x_3030_ = lean_string_compare(v_k_3020_, v_k_3023_);
switch(v___x_3030_)
{
case 0:
{
lean_object* v_impl_3031_; lean_object* v___x_3032_; 
lean_del_object(v___x_3028_);
lean_dec(v_size_3022_);
v_impl_3031_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(v_fst_3018_, v_snd_3019_, v_k_3020_, v_l_3025_);
v___x_3032_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_3023_, v_v_3024_, v_impl_3031_, v_r_3026_);
return v___x_3032_;
}
case 1:
{
lean_object* v___x_3033_; lean_object* v___x_3034_; lean_object* v_val_3035_; lean_object* v___x_3037_; 
lean_dec(v_k_3023_);
v___x_3033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3033_, 0, v_v_3024_);
v___x_3034_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0(v_fst_3018_, v_snd_3019_, v___x_3033_);
v_val_3035_ = lean_ctor_get(v___x_3034_, 0);
lean_inc(v_val_3035_);
lean_dec(v___x_3034_);
if (v_isShared_3029_ == 0)
{
lean_ctor_set(v___x_3028_, 2, v_val_3035_);
lean_ctor_set(v___x_3028_, 1, v_k_3020_);
v___x_3037_ = v___x_3028_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3038_; 
v_reuseFailAlloc_3038_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3038_, 0, v_size_3022_);
lean_ctor_set(v_reuseFailAlloc_3038_, 1, v_k_3020_);
lean_ctor_set(v_reuseFailAlloc_3038_, 2, v_val_3035_);
lean_ctor_set(v_reuseFailAlloc_3038_, 3, v_l_3025_);
lean_ctor_set(v_reuseFailAlloc_3038_, 4, v_r_3026_);
v___x_3037_ = v_reuseFailAlloc_3038_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
return v___x_3037_;
}
}
default: 
{
lean_object* v_impl_3039_; lean_object* v___x_3040_; 
lean_del_object(v___x_3028_);
lean_dec(v_size_3022_);
v_impl_3039_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(v_fst_3018_, v_snd_3019_, v_k_3020_, v_r_3026_);
v___x_3040_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_3023_, v_v_3024_, v_l_3025_, v_impl_3039_);
return v___x_3040_;
}
}
}
}
else
{
lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v_val_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; 
v___x_3042_ = lean_box(0);
v___x_3043_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___lam__0(v_fst_3018_, v_snd_3019_, v___x_3042_);
v_val_3044_ = lean_ctor_get(v___x_3043_, 0);
lean_inc(v_val_3044_);
lean_dec(v___x_3043_);
v___x_3045_ = lean_unsigned_to_nat(1u);
v___x_3046_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3046_, 0, v___x_3045_);
lean_ctor_set(v___x_3046_, 1, v_k_3020_);
lean_ctor_set(v___x_3046_, 2, v_val_3044_);
lean_ctor_set(v___x_3046_, 3, v_t_3021_);
lean_ctor_set(v___x_3046_, 4, v_t_3021_);
return v___x_3046_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg___boxed(lean_object* v_fst_3047_, lean_object* v_snd_3048_, lean_object* v_k_3049_, lean_object* v_t_3050_){
_start:
{
uint8_t v_snd_22280__boxed_3051_; lean_object* v_res_3052_; 
v_snd_22280__boxed_3051_ = lean_unbox(v_snd_3048_);
v_res_3052_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(v_fst_3047_, v_snd_22280__boxed_3051_, v_k_3049_, v_t_3050_);
return v_res_3052_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; lean_object* v___x_3060_; lean_object* v___x_3061_; 
v___x_3056_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__2));
v___x_3057_ = lean_unsigned_to_nat(14u);
v___x_3058_ = lean_unsigned_to_nat(22u);
v___x_3059_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__1));
v___x_3060_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__0));
v___x_3061_ = l_mkPanicMessageWithDecl(v___x_3060_, v___x_3059_, v___x_3058_, v___x_3057_, v___x_3056_);
return v___x_3061_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg(lean_object* v___x_3062_, lean_object* v_as_3063_, size_t v_sz_3064_, size_t v_i_3065_, lean_object* v_b_3066_){
_start:
{
uint8_t v___x_3068_; 
v___x_3068_ = lean_usize_dec_lt(v_i_3065_, v_sz_3064_);
if (v___x_3068_ == 0)
{
lean_object* v___x_3069_; 
lean_dec_ref(v___x_3062_);
v___x_3069_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3069_, 0, v_b_3066_);
return v___x_3069_;
}
else
{
lean_object* v_a_3070_; lean_object* v_snd_3071_; lean_object* v_fst_3072_; lean_object* v_fst_3073_; lean_object* v_snd_3074_; lean_object* v___y_3076_; 
v_a_3070_ = lean_array_uget_borrowed(v_as_3063_, v_i_3065_);
v_snd_3071_ = lean_ctor_get(v_a_3070_, 1);
v_fst_3072_ = lean_ctor_get(v_a_3070_, 0);
v_fst_3073_ = lean_ctor_get(v_snd_3071_, 0);
v_snd_3074_ = lean_ctor_get(v_snd_3071_, 1);
if (lean_obj_tag(v_fst_3073_) == 0)
{
lean_object* v___x_3082_; lean_object* v___x_3083_; 
v___x_3082_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___closed__3);
v___x_3083_ = lp_batteries_panic___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__8(v___x_3082_);
v___y_3076_ = v___x_3083_;
goto v___jp_3075_;
}
else
{
lean_object* v_head_3084_; lean_object* v_tail_3085_; lean_object* v___x_3086_; 
v_head_3084_ = lean_ctor_get(v_fst_3073_, 0);
v_tail_3085_ = lean_ctor_get(v_fst_3073_, 1);
lean_inc(v_tail_3085_);
lean_inc(v_head_3084_);
lean_inc_ref(v___x_3062_);
v___x_3086_ = lp_batteries_List_foldl___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__10(v___x_3062_, v_head_3084_, v_tail_3085_);
v___y_3076_ = v___x_3086_;
goto v___jp_3075_;
}
v___jp_3075_:
{
uint8_t v___x_3077_; lean_object* v___x_3078_; size_t v___x_3079_; size_t v___x_3080_; 
v___x_3077_ = lean_unbox(v_snd_3074_);
lean_inc(v_fst_3072_);
v___x_3078_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(v_fst_3072_, v___x_3077_, v___y_3076_, v_b_3066_);
v___x_3079_ = ((size_t)1ULL);
v___x_3080_ = lean_usize_add(v_i_3065_, v___x_3079_);
v_i_3065_ = v___x_3080_;
v_b_3066_ = v___x_3078_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg___boxed(lean_object* v___x_3087_, lean_object* v_as_3088_, lean_object* v_sz_3089_, lean_object* v_i_3090_, lean_object* v_b_3091_, lean_object* v___y_3092_){
_start:
{
size_t v_sz_boxed_3093_; size_t v_i_boxed_3094_; lean_object* v_res_3095_; 
v_sz_boxed_3093_ = lean_unbox_usize(v_sz_3089_);
lean_dec(v_sz_3089_);
v_i_boxed_3094_ = lean_unbox_usize(v_i_3090_);
lean_dec(v_i_3090_);
v_res_3095_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg(v___x_3087_, v_as_3088_, v_sz_boxed_3093_, v_i_boxed_3094_, v_b_3091_);
lean_dec_ref(v_as_3088_);
return v_res_3095_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg(lean_object* v_ref_3096_, lean_object* v_msg_3097_, lean_object* v___y_3098_, lean_object* v___y_3099_){
_start:
{
lean_object* v___x_3101_; 
v___x_3101_ = l_Lean_Elab_Command_getRef___redArg(v___y_3098_);
if (lean_obj_tag(v___x_3101_) == 0)
{
lean_object* v_a_3102_; lean_object* v_fileName_3103_; lean_object* v_fileMap_3104_; lean_object* v_currRecDepth_3105_; lean_object* v_cmdPos_3106_; lean_object* v_macroStack_3107_; lean_object* v_quotContext_x3f_3108_; lean_object* v_currMacroScope_3109_; lean_object* v_snap_x3f_3110_; lean_object* v_cancelTk_x3f_3111_; uint8_t v_suppressElabErrors_3112_; lean_object* v_ref_3113_; lean_object* v___x_3114_; lean_object* v___x_3115_; 
v_a_3102_ = lean_ctor_get(v___x_3101_, 0);
lean_inc(v_a_3102_);
lean_dec_ref_known(v___x_3101_, 1);
v_fileName_3103_ = lean_ctor_get(v___y_3098_, 0);
v_fileMap_3104_ = lean_ctor_get(v___y_3098_, 1);
v_currRecDepth_3105_ = lean_ctor_get(v___y_3098_, 2);
v_cmdPos_3106_ = lean_ctor_get(v___y_3098_, 3);
v_macroStack_3107_ = lean_ctor_get(v___y_3098_, 4);
v_quotContext_x3f_3108_ = lean_ctor_get(v___y_3098_, 5);
v_currMacroScope_3109_ = lean_ctor_get(v___y_3098_, 6);
v_snap_x3f_3110_ = lean_ctor_get(v___y_3098_, 8);
v_cancelTk_x3f_3111_ = lean_ctor_get(v___y_3098_, 9);
v_suppressElabErrors_3112_ = lean_ctor_get_uint8(v___y_3098_, sizeof(void*)*10);
v_ref_3113_ = l_Lean_replaceRef(v_ref_3096_, v_a_3102_);
lean_dec(v_a_3102_);
lean_inc(v_cancelTk_x3f_3111_);
lean_inc(v_snap_x3f_3110_);
lean_inc(v_currMacroScope_3109_);
lean_inc(v_quotContext_x3f_3108_);
lean_inc(v_macroStack_3107_);
lean_inc(v_cmdPos_3106_);
lean_inc(v_currRecDepth_3105_);
lean_inc_ref(v_fileMap_3104_);
lean_inc_ref(v_fileName_3103_);
v___x_3114_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_3114_, 0, v_fileName_3103_);
lean_ctor_set(v___x_3114_, 1, v_fileMap_3104_);
lean_ctor_set(v___x_3114_, 2, v_currRecDepth_3105_);
lean_ctor_set(v___x_3114_, 3, v_cmdPos_3106_);
lean_ctor_set(v___x_3114_, 4, v_macroStack_3107_);
lean_ctor_set(v___x_3114_, 5, v_quotContext_x3f_3108_);
lean_ctor_set(v___x_3114_, 6, v_currMacroScope_3109_);
lean_ctor_set(v___x_3114_, 7, v_ref_3113_);
lean_ctor_set(v___x_3114_, 8, v_snap_x3f_3110_);
lean_ctor_set(v___x_3114_, 9, v_cancelTk_x3f_3111_);
lean_ctor_set_uint8(v___x_3114_, sizeof(void*)*10, v_suppressElabErrors_3112_);
v___x_3115_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v_msg_3097_, v___x_3114_, v___y_3099_);
lean_dec_ref_known(v___x_3114_, 10);
return v___x_3115_;
}
else
{
lean_object* v_a_3116_; lean_object* v___x_3118_; uint8_t v_isShared_3119_; uint8_t v_isSharedCheck_3123_; 
lean_dec_ref(v_msg_3097_);
v_a_3116_ = lean_ctor_get(v___x_3101_, 0);
v_isSharedCheck_3123_ = !lean_is_exclusive(v___x_3101_);
if (v_isSharedCheck_3123_ == 0)
{
v___x_3118_ = v___x_3101_;
v_isShared_3119_ = v_isSharedCheck_3123_;
goto v_resetjp_3117_;
}
else
{
lean_inc(v_a_3116_);
lean_dec(v___x_3101_);
v___x_3118_ = lean_box(0);
v_isShared_3119_ = v_isSharedCheck_3123_;
goto v_resetjp_3117_;
}
v_resetjp_3117_:
{
lean_object* v___x_3121_; 
if (v_isShared_3119_ == 0)
{
v___x_3121_ = v___x_3118_;
goto v_reusejp_3120_;
}
else
{
lean_object* v_reuseFailAlloc_3122_; 
v_reuseFailAlloc_3122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3122_, 0, v_a_3116_);
v___x_3121_ = v_reuseFailAlloc_3122_;
goto v_reusejp_3120_;
}
v_reusejp_3120_:
{
return v___x_3121_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg___boxed(lean_object* v_ref_3124_, lean_object* v_msg_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_, lean_object* v___y_3128_){
_start:
{
lean_object* v_res_3129_; 
v_res_3129_ = lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg(v_ref_3124_, v_msg_3125_, v___y_3126_, v___y_3127_);
lean_dec(v___y_3127_);
lean_dec_ref(v___y_3126_);
lean_dec(v_ref_3124_);
return v_res_3129_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0(lean_object* v___x_3130_, lean_object* v_k_3131_, lean_object* v_00_u03b1_3132_, lean_object* v_type_3133_, lean_object* v_attr_3134_, lean_object* v_msg_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_){
_start:
{
lean_object* v___x_3139_; lean_object* v___x_3140_; 
lean_inc_ref(v___x_3130_);
v___x_3139_ = l_Lean_KeyedDeclsAttribute_getEntries___redArg(v_attr_3134_, v___x_3130_, v_k_3131_);
v___x_3140_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg(v___x_3130_, v_type_3133_, v___x_3139_, v_msg_3135_, v___y_3136_);
lean_dec(v___x_3139_);
return v___x_3140_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0___boxed(lean_object* v___x_3141_, lean_object* v_k_3142_, lean_object* v_00_u03b1_3143_, lean_object* v_type_3144_, lean_object* v_attr_3145_, lean_object* v_msg_3146_, lean_object* v___y_3147_, lean_object* v___y_3148_, lean_object* v___y_3149_){
_start:
{
lean_object* v_res_3150_; 
v_res_3150_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0(v___x_3141_, v_k_3142_, v_00_u03b1_3143_, v_type_3144_, v_attr_3145_, v_msg_3146_, v___y_3147_, v___y_3148_);
lean_dec(v___y_3148_);
lean_dec_ref(v___y_3147_);
lean_dec_ref(v_attr_3145_);
lean_dec(v_k_3142_);
return v_res_3150_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1(lean_object* v___x_3151_, uint8_t v___x_3152_, lean_object* v_more_3153_, lean_object* v_catName_3154_, lean_object* v_k_3155_, lean_object* v_msg_3156_, lean_object* v_msg1_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_){
_start:
{
lean_object* v_msg1_3162_; lean_object* v___y_3168_; lean_object* v___y_3169_; lean_object* v___y_3170_; lean_object* v___y_3171_; lean_object* v___x_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; 
v___x_3176_ = l_Lean_Options_empty;
v___x_3177_ = lean_box(0);
v___x_3178_ = lean_box(0);
lean_inc(v_k_3155_);
lean_inc_ref(v___x_3151_);
v___x_3179_ = l_Lean_findDocString_x3f(v___x_3151_, v_k_3155_, v___x_3152_, v___x_3176_, v___x_3177_, v___x_3178_);
if (lean_obj_tag(v___x_3179_) == 0)
{
lean_object* v_a_3180_; lean_object* v___f_3181_; lean_object* v_msg1_3183_; lean_object* v___y_3184_; lean_object* v___y_3185_; 
v_a_3180_ = lean_ctor_get(v___x_3179_, 0);
lean_inc(v_a_3180_);
lean_dec_ref_known(v___x_3179_, 1);
lean_inc(v_k_3155_);
lean_inc_ref(v___x_3151_);
v___f_3181_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0___boxed), 9, 2);
lean_closure_set(v___f_3181_, 0, v___x_3151_);
lean_closure_set(v___f_3181_, 1, v_k_3155_);
if (lean_obj_tag(v_a_3180_) == 1)
{
lean_object* v_val_3212_; lean_object* v___x_3214_; uint8_t v_isShared_3215_; uint8_t v_isSharedCheck_3231_; 
v_val_3212_ = lean_ctor_get(v_a_3180_, 0);
v_isSharedCheck_3231_ = !lean_is_exclusive(v_a_3180_);
if (v_isSharedCheck_3231_ == 0)
{
v___x_3214_ = v_a_3180_;
v_isShared_3215_ = v_isSharedCheck_3231_;
goto v_resetjp_3213_;
}
else
{
lean_inc(v_val_3212_);
lean_dec(v_a_3180_);
v___x_3214_ = lean_box(0);
v_isShared_3215_ = v_isSharedCheck_3231_;
goto v_resetjp_3213_;
}
v_resetjp_3213_:
{
lean_object* v___x_3216_; lean_object* v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v_str_3220_; lean_object* v_startInclusive_3221_; lean_object* v_endExclusive_3222_; lean_object* v___x_3223_; lean_object* v___x_3224_; lean_object* v___x_3225_; lean_object* v___x_3227_; 
v___x_3216_ = lean_unsigned_to_nat(0u);
v___x_3217_ = lean_string_utf8_byte_size(v_val_3212_);
v___x_3218_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3218_, 0, v_val_3212_);
lean_ctor_set(v___x_3218_, 1, v___x_3216_);
lean_ctor_set(v___x_3218_, 2, v___x_3217_);
v___x_3219_ = l_String_Slice_trimAscii(v___x_3218_);
v_str_3220_ = lean_ctor_get(v___x_3219_, 0);
lean_inc_ref(v_str_3220_);
v_startInclusive_3221_ = lean_ctor_get(v___x_3219_, 1);
lean_inc(v_startInclusive_3221_);
v_endExclusive_3222_ = lean_ctor_get(v___x_3219_, 2);
lean_inc(v_endExclusive_3222_);
lean_dec_ref(v___x_3219_);
v___x_3223_ = lean_obj_once(&lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0, &lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0_once, _init_lp_batteries_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2_spec__4_spec__7___closed__0);
v___x_3224_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3224_, 0, v_msg1_3157_);
lean_ctor_set(v___x_3224_, 1, v___x_3223_);
v___x_3225_ = lean_string_utf8_extract_fast(v_str_3220_, v_startInclusive_3221_, v_endExclusive_3222_);
lean_dec(v_endExclusive_3222_);
lean_dec(v_startInclusive_3221_);
lean_dec_ref(v_str_3220_);
if (v_isShared_3215_ == 0)
{
lean_ctor_set_tag(v___x_3214_, 3);
lean_ctor_set(v___x_3214_, 0, v___x_3225_);
v___x_3227_ = v___x_3214_;
goto v_reusejp_3226_;
}
else
{
lean_object* v_reuseFailAlloc_3230_; 
v_reuseFailAlloc_3230_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3230_, 0, v___x_3225_);
v___x_3227_ = v_reuseFailAlloc_3230_;
goto v_reusejp_3226_;
}
v_reusejp_3226_:
{
lean_object* v___x_3228_; lean_object* v___x_3229_; 
v___x_3228_ = l_Lean_MessageData_ofFormat(v___x_3227_);
v___x_3229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3229_, 0, v___x_3224_);
lean_ctor_set(v___x_3229_, 1, v___x_3228_);
v_msg1_3183_ = v___x_3229_;
v___y_3184_ = v___y_3158_;
v___y_3185_ = v___y_3159_;
goto v___jp_3182_;
}
}
}
else
{
lean_dec(v_a_3180_);
v_msg1_3183_ = v_msg1_3157_;
v___y_3184_ = v___y_3158_;
v___y_3185_ = v___y_3159_;
goto v___jp_3182_;
}
v___jp_3182_:
{
lean_object* v___x_3186_; lean_object* v___x_3187_; 
v___x_3186_ = lean_unsigned_to_nat(2u);
v___x_3187_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_3187_, 0, v___x_3186_);
lean_ctor_set(v___x_3187_, 1, v_msg1_3183_);
if (lean_obj_tag(v_more_3153_) == 0)
{
lean_dec_ref(v___f_3181_);
lean_dec(v_k_3155_);
lean_dec_ref(v___x_3151_);
v_msg1_3162_ = v___x_3187_;
goto v___jp_3161_;
}
else
{
lean_object* v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; 
v___x_3188_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__1));
v___x_3189_ = l_Lean_Elab_macroAttribute;
lean_inc_ref(v___x_3151_);
v___x_3190_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0(v___x_3151_, v_k_3155_, lean_box(0), v___x_3188_, v___x_3189_, v___x_3187_, v___y_3184_, v___y_3185_);
if (lean_obj_tag(v___x_3190_) == 0)
{
if (lean_obj_tag(v_catName_3154_) == 1)
{
lean_object* v_pre_3191_; 
v_pre_3191_ = lean_ctor_get(v_catName_3154_, 0);
if (lean_obj_tag(v_pre_3191_) == 0)
{
lean_object* v_a_3192_; lean_object* v_str_3193_; lean_object* v___x_3194_; uint8_t v___x_3195_; 
v_a_3192_ = lean_ctor_get(v___x_3190_, 0);
lean_inc(v_a_3192_);
lean_dec_ref_known(v___x_3190_, 1);
v_str_3193_ = lean_ctor_get(v_catName_3154_, 1);
v___x_3194_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__2));
v___x_3195_ = lean_string_dec_eq(v_str_3193_, v___x_3194_);
if (v___x_3195_ == 0)
{
lean_object* v___x_3196_; uint8_t v___x_3197_; 
v___x_3196_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__3));
v___x_3197_ = lean_string_dec_eq(v_str_3193_, v___x_3196_);
if (v___x_3197_ == 0)
{
lean_object* v___x_3198_; uint8_t v___x_3199_; 
lean_dec(v_k_3155_);
lean_dec_ref(v___x_3151_);
v___x_3198_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__4));
v___x_3199_ = lean_string_dec_eq(v_str_3193_, v___x_3198_);
if (v___x_3199_ == 0)
{
lean_object* v___x_3200_; uint8_t v___x_3201_; 
v___x_3200_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__5));
v___x_3201_ = lean_string_dec_eq(v_str_3193_, v___x_3200_);
if (v___x_3201_ == 0)
{
lean_dec_ref(v___f_3181_);
v_msg1_3162_ = v_a_3192_;
goto v___jp_3161_;
}
else
{
v___y_3168_ = v_a_3192_;
v___y_3169_ = v___f_3181_;
v___y_3170_ = v___y_3184_;
v___y_3171_ = v___y_3185_;
goto v___jp_3167_;
}
}
else
{
v___y_3168_ = v_a_3192_;
v___y_3169_ = v___f_3181_;
v___y_3170_ = v___y_3184_;
v___y_3171_ = v___y_3185_;
goto v___jp_3167_;
}
}
else
{
lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; 
lean_dec_ref(v___f_3181_);
v___x_3202_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__6));
v___x_3203_ = l_Lean_Elab_Command_commandElabAttribute;
v___x_3204_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0(v___x_3151_, v_k_3155_, lean_box(0), v___x_3202_, v___x_3203_, v_a_3192_, v___y_3184_, v___y_3185_);
lean_dec(v_k_3155_);
if (lean_obj_tag(v___x_3204_) == 0)
{
lean_object* v_a_3205_; 
v_a_3205_ = lean_ctor_get(v___x_3204_, 0);
lean_inc(v_a_3205_);
lean_dec_ref_known(v___x_3204_, 1);
v_msg1_3162_ = v_a_3205_;
goto v___jp_3161_;
}
else
{
lean_dec_ref(v_msg_3156_);
return v___x_3204_;
}
}
}
else
{
lean_object* v___x_3206_; lean_object* v___x_3207_; lean_object* v___x_3208_; 
lean_dec_ref(v___f_3181_);
v___x_3206_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__7));
v___x_3207_ = l_Lean_Elab_Term_termElabAttribute;
v___x_3208_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__0(v___x_3151_, v_k_3155_, lean_box(0), v___x_3206_, v___x_3207_, v_a_3192_, v___y_3184_, v___y_3185_);
lean_dec(v_k_3155_);
if (lean_obj_tag(v___x_3208_) == 0)
{
lean_object* v_a_3209_; 
v_a_3209_ = lean_ctor_get(v___x_3208_, 0);
lean_inc(v_a_3209_);
lean_dec_ref_known(v___x_3208_, 1);
v_msg1_3162_ = v_a_3209_;
goto v___jp_3161_;
}
else
{
lean_dec_ref(v_msg_3156_);
return v___x_3208_;
}
}
}
else
{
lean_object* v_a_3210_; 
lean_dec_ref(v___f_3181_);
lean_dec(v_k_3155_);
lean_dec_ref(v___x_3151_);
v_a_3210_ = lean_ctor_get(v___x_3190_, 0);
lean_inc(v_a_3210_);
lean_dec_ref_known(v___x_3190_, 1);
v_msg1_3162_ = v_a_3210_;
goto v___jp_3161_;
}
}
else
{
lean_object* v_a_3211_; 
lean_dec_ref(v___f_3181_);
lean_dec(v_k_3155_);
lean_dec_ref(v___x_3151_);
v_a_3211_ = lean_ctor_get(v___x_3190_, 0);
lean_inc(v_a_3211_);
lean_dec_ref_known(v___x_3190_, 1);
v_msg1_3162_ = v_a_3211_;
goto v___jp_3161_;
}
}
else
{
lean_dec_ref(v___f_3181_);
lean_dec_ref(v_msg_3156_);
lean_dec(v_k_3155_);
lean_dec_ref(v___x_3151_);
return v___x_3190_;
}
}
}
}
else
{
lean_object* v_a_3232_; lean_object* v___x_3234_; uint8_t v_isShared_3235_; uint8_t v_isSharedCheck_3244_; 
lean_dec_ref(v_msg1_3157_);
lean_dec_ref(v_msg_3156_);
lean_dec(v_k_3155_);
lean_dec_ref(v___x_3151_);
v_a_3232_ = lean_ctor_get(v___x_3179_, 0);
v_isSharedCheck_3244_ = !lean_is_exclusive(v___x_3179_);
if (v_isSharedCheck_3244_ == 0)
{
v___x_3234_ = v___x_3179_;
v_isShared_3235_ = v_isSharedCheck_3244_;
goto v_resetjp_3233_;
}
else
{
lean_inc(v_a_3232_);
lean_dec(v___x_3179_);
v___x_3234_ = lean_box(0);
v_isShared_3235_ = v_isSharedCheck_3244_;
goto v_resetjp_3233_;
}
v_resetjp_3233_:
{
lean_object* v_ref_3236_; lean_object* v___x_3237_; lean_object* v___x_3238_; lean_object* v___x_3239_; lean_object* v___x_3240_; lean_object* v___x_3242_; 
v_ref_3236_ = lean_ctor_get(v___y_3158_, 7);
v___x_3237_ = lean_io_error_to_string(v_a_3232_);
v___x_3238_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3238_, 0, v___x_3237_);
v___x_3239_ = l_Lean_MessageData_ofFormat(v___x_3238_);
lean_inc(v_ref_3236_);
v___x_3240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3240_, 0, v_ref_3236_);
lean_ctor_set(v___x_3240_, 1, v___x_3239_);
if (v_isShared_3235_ == 0)
{
lean_ctor_set(v___x_3234_, 0, v___x_3240_);
v___x_3242_ = v___x_3234_;
goto v_reusejp_3241_;
}
else
{
lean_object* v_reuseFailAlloc_3243_; 
v_reuseFailAlloc_3243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3243_, 0, v___x_3240_);
v___x_3242_ = v_reuseFailAlloc_3243_;
goto v_reusejp_3241_;
}
v_reusejp_3241_:
{
return v___x_3242_;
}
}
}
v___jp_3161_:
{
lean_object* v___x_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; 
v___x_3163_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3163_, 0, v_msg_3156_);
lean_ctor_set(v___x_3163_, 1, v_msg1_3162_);
v___x_3164_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__1);
v___x_3165_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3165_, 0, v___x_3163_);
lean_ctor_set(v___x_3165_, 1, v___x_3164_);
v___x_3166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3166_, 0, v___x_3165_);
return v___x_3166_;
}
v___jp_3167_:
{
lean_object* v___x_3172_; lean_object* v___x_3173_; lean_object* v___x_3174_; 
v___x_3172_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14_spec__19___closed__0));
v___x_3173_ = l_Lean_Elab_Tactic_tacticElabAttribute;
lean_inc(v___y_3171_);
lean_inc_ref(v___y_3170_);
v___x_3174_ = lean_apply_7(v___y_3169_, lean_box(0), v___x_3172_, v___x_3173_, v___y_3168_, v___y_3170_, v___y_3171_, lean_box(0));
if (lean_obj_tag(v___x_3174_) == 0)
{
lean_object* v_a_3175_; 
v_a_3175_ = lean_ctor_get(v___x_3174_, 0);
lean_inc(v_a_3175_);
lean_dec_ref_known(v___x_3174_, 1);
v_msg1_3162_ = v_a_3175_;
goto v___jp_3161_;
}
else
{
lean_dec_ref(v_msg_3156_);
return v___x_3174_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1___boxed(lean_object* v___x_3245_, lean_object* v___x_3246_, lean_object* v_more_3247_, lean_object* v_catName_3248_, lean_object* v_k_3249_, lean_object* v_msg_3250_, lean_object* v_msg1_3251_, lean_object* v___y_3252_, lean_object* v___y_3253_, lean_object* v___y_3254_){
_start:
{
uint8_t v___x_22469__boxed_3255_; lean_object* v_res_3256_; 
v___x_22469__boxed_3255_ = lean_unbox(v___x_3246_);
v_res_3256_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1(v___x_3245_, v___x_22469__boxed_3255_, v_more_3247_, v_catName_3248_, v_k_3249_, v_msg_3250_, v_msg1_3251_, v___y_3252_, v___y_3253_);
lean_dec(v___y_3253_);
lean_dec_ref(v___y_3252_);
lean_dec(v_catName_3248_);
lean_dec(v_more_3247_);
return v_res_3256_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__1(void){
_start:
{
lean_object* v___x_3258_; lean_object* v___x_3259_; 
v___x_3258_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__0));
v___x_3259_ = l_Lean_stringToMessageData(v___x_3258_);
return v___x_3259_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3(void){
_start:
{
lean_object* v___x_3261_; lean_object* v___x_3262_; 
v___x_3261_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__2));
v___x_3262_ = l_Lean_stringToMessageData(v___x_3261_);
return v___x_3262_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__5(void){
_start:
{
lean_object* v___x_3264_; lean_object* v___x_3265_; 
v___x_3264_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__4));
v___x_3265_ = l_Lean_stringToMessageData(v___x_3264_);
return v___x_3265_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13(lean_object* v_a_3266_, lean_object* v___x_3267_, lean_object* v_more_3268_, lean_object* v_catName_3269_, lean_object* v_as_3270_, size_t v_sz_3271_, size_t v_i_3272_, lean_object* v_b_3273_, lean_object* v___y_3274_, lean_object* v___y_3275_){
_start:
{
lean_object* v_a_3278_; uint8_t v___x_3282_; 
v___x_3282_ = lean_usize_dec_lt(v_i_3272_, v_sz_3271_);
if (v___x_3282_ == 0)
{
lean_object* v___x_3283_; 
lean_dec_ref(v___x_3267_);
lean_dec_ref(v_a_3266_);
v___x_3283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3283_, 0, v_b_3273_);
return v___x_3283_;
}
else
{
lean_object* v_a_3284_; lean_object* v_snd_3285_; uint8_t v___x_3286_; 
v_a_3284_ = lean_array_uget(v_as_3270_, v_i_3272_);
v_snd_3285_ = lean_ctor_get(v_a_3284_, 1);
v___x_3286_ = lean_unbox(v_snd_3285_);
if (v___x_3286_ == 0)
{
lean_object* v_fst_3287_; lean_object* v___x_3289_; uint8_t v_isShared_3290_; uint8_t v_isSharedCheck_3308_; 
v_fst_3287_ = lean_ctor_get(v_a_3284_, 0);
v_isSharedCheck_3308_ = !lean_is_exclusive(v_a_3284_);
if (v_isSharedCheck_3308_ == 0)
{
lean_object* v_unused_3309_; 
v_unused_3309_ = lean_ctor_get(v_a_3284_, 1);
lean_dec(v_unused_3309_);
v___x_3289_ = v_a_3284_;
v_isShared_3290_ = v_isSharedCheck_3308_;
goto v_resetjp_3288_;
}
else
{
lean_inc(v_fst_3287_);
lean_dec(v_a_3284_);
v___x_3289_ = lean_box(0);
v_isShared_3290_ = v_isSharedCheck_3308_;
goto v_resetjp_3288_;
}
v_resetjp_3288_:
{
lean_object* v___x_3291_; lean_object* v___x_3292_; lean_object* v___x_3293_; lean_object* v___x_3294_; lean_object* v___x_3296_; 
v___x_3291_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__1);
lean_inc_ref(v_a_3266_);
v___x_3292_ = l_String_quote(v_a_3266_);
v___x_3293_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3293_, 0, v___x_3292_);
v___x_3294_ = l_Lean_MessageData_ofFormat(v___x_3293_);
if (v_isShared_3290_ == 0)
{
lean_ctor_set_tag(v___x_3289_, 7);
lean_ctor_set(v___x_3289_, 1, v___x_3294_);
lean_ctor_set(v___x_3289_, 0, v___x_3291_);
v___x_3296_ = v___x_3289_;
goto v_reusejp_3295_;
}
else
{
lean_object* v_reuseFailAlloc_3307_; 
v_reuseFailAlloc_3307_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3307_, 0, v___x_3291_);
lean_ctor_set(v_reuseFailAlloc_3307_, 1, v___x_3294_);
v___x_3296_ = v_reuseFailAlloc_3307_;
goto v_reusejp_3295_;
}
v_reusejp_3295_:
{
lean_object* v___x_3297_; lean_object* v___x_3298_; lean_object* v___x_3299_; lean_object* v___x_3300_; lean_object* v___x_3301_; lean_object* v___x_3302_; lean_object* v___x_3303_; lean_object* v___x_3304_; lean_object* v___x_3305_; 
v___x_3297_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3);
v___x_3298_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3298_, 0, v___x_3296_);
lean_ctor_set(v___x_3298_, 1, v___x_3297_);
v___x_3299_ = lean_box(0);
lean_inc(v_fst_3287_);
v___x_3300_ = l_Lean_mkConst(v_fst_3287_, v___x_3299_);
v___x_3301_ = l_Lean_MessageData_ofExpr(v___x_3300_);
v___x_3302_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3302_, 0, v___x_3298_);
lean_ctor_set(v___x_3302_, 1, v___x_3301_);
v___x_3303_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7);
v___x_3304_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3304_, 0, v___x_3302_);
lean_ctor_set(v___x_3304_, 1, v___x_3303_);
lean_inc_ref(v___x_3267_);
v___x_3305_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1(v___x_3267_, v___x_3282_, v_more_3268_, v_catName_3269_, v_fst_3287_, v_b_3273_, v___x_3304_, v___y_3274_, v___y_3275_);
if (lean_obj_tag(v___x_3305_) == 0)
{
lean_object* v_a_3306_; 
v_a_3306_ = lean_ctor_get(v___x_3305_, 0);
lean_inc(v_a_3306_);
lean_dec_ref_known(v___x_3305_, 1);
v_a_3278_ = v_a_3306_;
goto v___jp_3277_;
}
else
{
lean_dec_ref(v___x_3267_);
lean_dec_ref(v_a_3266_);
return v___x_3305_;
}
}
}
}
else
{
lean_object* v_fst_3310_; lean_object* v___x_3312_; uint8_t v_isShared_3313_; uint8_t v_isSharedCheck_3331_; 
v_fst_3310_ = lean_ctor_get(v_a_3284_, 0);
v_isSharedCheck_3331_ = !lean_is_exclusive(v_a_3284_);
if (v_isSharedCheck_3331_ == 0)
{
lean_object* v_unused_3332_; 
v_unused_3332_ = lean_ctor_get(v_a_3284_, 1);
lean_dec(v_unused_3332_);
v___x_3312_ = v_a_3284_;
v_isShared_3313_ = v_isSharedCheck_3331_;
goto v_resetjp_3311_;
}
else
{
lean_inc(v_fst_3310_);
lean_dec(v_a_3284_);
v___x_3312_ = lean_box(0);
v_isShared_3313_ = v_isSharedCheck_3331_;
goto v_resetjp_3311_;
}
v_resetjp_3311_:
{
lean_object* v___x_3314_; lean_object* v___x_3315_; lean_object* v___x_3316_; lean_object* v___x_3317_; lean_object* v___x_3319_; 
v___x_3314_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__5);
lean_inc_ref(v_a_3266_);
v___x_3315_ = l_String_quote(v_a_3266_);
v___x_3316_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3316_, 0, v___x_3315_);
v___x_3317_ = l_Lean_MessageData_ofFormat(v___x_3316_);
if (v_isShared_3313_ == 0)
{
lean_ctor_set_tag(v___x_3312_, 7);
lean_ctor_set(v___x_3312_, 1, v___x_3317_);
lean_ctor_set(v___x_3312_, 0, v___x_3314_);
v___x_3319_ = v___x_3312_;
goto v_reusejp_3318_;
}
else
{
lean_object* v_reuseFailAlloc_3330_; 
v_reuseFailAlloc_3330_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3330_, 0, v___x_3314_);
lean_ctor_set(v_reuseFailAlloc_3330_, 1, v___x_3317_);
v___x_3319_ = v_reuseFailAlloc_3330_;
goto v_reusejp_3318_;
}
v_reusejp_3318_:
{
lean_object* v___x_3320_; lean_object* v___x_3321_; lean_object* v___x_3322_; lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; 
v___x_3320_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___closed__3);
v___x_3321_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3321_, 0, v___x_3319_);
lean_ctor_set(v___x_3321_, 1, v___x_3320_);
v___x_3322_ = lean_box(0);
lean_inc(v_fst_3310_);
v___x_3323_ = l_Lean_mkConst(v_fst_3310_, v___x_3322_);
v___x_3324_ = l_Lean_MessageData_ofExpr(v___x_3323_);
v___x_3325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3325_, 0, v___x_3321_);
lean_ctor_set(v___x_3325_, 1, v___x_3324_);
v___x_3326_ = lean_obj_once(&lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7, &lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7_once, _init_lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__1___redArg___closed__7);
v___x_3327_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3327_, 0, v___x_3325_);
lean_ctor_set(v___x_3327_, 1, v___x_3326_);
lean_inc_ref(v___x_3267_);
v___x_3328_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___lam__1(v___x_3267_, v___x_3282_, v_more_3268_, v_catName_3269_, v_fst_3310_, v_b_3273_, v___x_3327_, v___y_3274_, v___y_3275_);
if (lean_obj_tag(v___x_3328_) == 0)
{
lean_object* v_a_3329_; 
v_a_3329_ = lean_ctor_get(v___x_3328_, 0);
lean_inc(v_a_3329_);
lean_dec_ref_known(v___x_3328_, 1);
v_a_3278_ = v_a_3329_;
goto v___jp_3277_;
}
else
{
lean_dec_ref(v___x_3267_);
lean_dec_ref(v_a_3266_);
return v___x_3328_;
}
}
}
}
}
v___jp_3277_:
{
size_t v___x_3279_; size_t v___x_3280_; 
v___x_3279_ = ((size_t)1ULL);
v___x_3280_ = lean_usize_add(v_i_3272_, v___x_3279_);
v_i_3272_ = v___x_3280_;
v_b_3273_ = v_a_3278_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13___boxed(lean_object* v_a_3333_, lean_object* v___x_3334_, lean_object* v_more_3335_, lean_object* v_catName_3336_, lean_object* v_as_3337_, lean_object* v_sz_3338_, lean_object* v_i_3339_, lean_object* v_b_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_, lean_object* v___y_3343_){
_start:
{
size_t v_sz_boxed_3344_; size_t v_i_boxed_3345_; lean_object* v_res_3346_; 
v_sz_boxed_3344_ = lean_unbox_usize(v_sz_3338_);
lean_dec(v_sz_3338_);
v_i_boxed_3345_ = lean_unbox_usize(v_i_3339_);
lean_dec(v_i_3339_);
v_res_3346_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13(v_a_3333_, v___x_3334_, v_more_3335_, v_catName_3336_, v_as_3337_, v_sz_boxed_3344_, v_i_boxed_3345_, v_b_3340_, v___y_3341_, v___y_3342_);
lean_dec(v___y_3342_);
lean_dec_ref(v___y_3341_);
lean_dec_ref(v_as_3337_);
lean_dec(v_catName_3336_);
lean_dec(v_more_3335_);
return v_res_3346_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15(lean_object* v___x_3347_, lean_object* v_more_3348_, lean_object* v_catName_3349_, lean_object* v_init_3350_, lean_object* v_x_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_){
_start:
{
if (lean_obj_tag(v_x_3351_) == 0)
{
lean_object* v_k_3355_; lean_object* v_v_3356_; lean_object* v_l_3357_; lean_object* v_r_3358_; lean_object* v___x_3359_; 
v_k_3355_ = lean_ctor_get(v_x_3351_, 1);
lean_inc(v_k_3355_);
v_v_3356_ = lean_ctor_get(v_x_3351_, 2);
lean_inc(v_v_3356_);
v_l_3357_ = lean_ctor_get(v_x_3351_, 3);
lean_inc(v_l_3357_);
v_r_3358_ = lean_ctor_get(v_x_3351_, 4);
lean_inc(v_r_3358_);
lean_dec_ref_known(v_x_3351_, 5);
lean_inc_ref(v___x_3347_);
v___x_3359_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15(v___x_3347_, v_more_3348_, v_catName_3349_, v_init_3350_, v_l_3357_, v___y_3352_, v___y_3353_);
if (lean_obj_tag(v___x_3359_) == 0)
{
lean_object* v_a_3360_; lean_object* v_a_3361_; size_t v_sz_3362_; size_t v___x_3363_; lean_object* v___x_3364_; 
v_a_3360_ = lean_ctor_get(v___x_3359_, 0);
lean_inc(v_a_3360_);
lean_dec_ref_known(v___x_3359_, 1);
v_a_3361_ = lean_ctor_get(v_a_3360_, 0);
lean_inc(v_a_3361_);
lean_dec(v_a_3360_);
v_sz_3362_ = lean_array_size(v_v_3356_);
v___x_3363_ = ((size_t)0ULL);
lean_inc_ref(v___x_3347_);
v___x_3364_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__13(v_k_3355_, v___x_3347_, v_more_3348_, v_catName_3349_, v_v_3356_, v_sz_3362_, v___x_3363_, v_a_3361_, v___y_3352_, v___y_3353_);
lean_dec(v_v_3356_);
if (lean_obj_tag(v___x_3364_) == 0)
{
lean_object* v_a_3365_; 
v_a_3365_ = lean_ctor_get(v___x_3364_, 0);
lean_inc(v_a_3365_);
lean_dec_ref_known(v___x_3364_, 1);
v_init_3350_ = v_a_3365_;
v_x_3351_ = v_r_3358_;
goto _start;
}
else
{
lean_object* v_a_3367_; lean_object* v___x_3369_; uint8_t v_isShared_3370_; uint8_t v_isSharedCheck_3374_; 
lean_dec(v_r_3358_);
lean_dec_ref(v___x_3347_);
v_a_3367_ = lean_ctor_get(v___x_3364_, 0);
v_isSharedCheck_3374_ = !lean_is_exclusive(v___x_3364_);
if (v_isSharedCheck_3374_ == 0)
{
v___x_3369_ = v___x_3364_;
v_isShared_3370_ = v_isSharedCheck_3374_;
goto v_resetjp_3368_;
}
else
{
lean_inc(v_a_3367_);
lean_dec(v___x_3364_);
v___x_3369_ = lean_box(0);
v_isShared_3370_ = v_isSharedCheck_3374_;
goto v_resetjp_3368_;
}
v_resetjp_3368_:
{
lean_object* v___x_3372_; 
if (v_isShared_3370_ == 0)
{
v___x_3372_ = v___x_3369_;
goto v_reusejp_3371_;
}
else
{
lean_object* v_reuseFailAlloc_3373_; 
v_reuseFailAlloc_3373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3373_, 0, v_a_3367_);
v___x_3372_ = v_reuseFailAlloc_3373_;
goto v_reusejp_3371_;
}
v_reusejp_3371_:
{
return v___x_3372_;
}
}
}
}
else
{
lean_dec(v_r_3358_);
lean_dec(v_v_3356_);
lean_dec(v_k_3355_);
lean_dec_ref(v___x_3347_);
return v___x_3359_;
}
}
else
{
lean_object* v___x_3375_; lean_object* v___x_3376_; 
lean_dec_ref(v___x_3347_);
v___x_3375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3375_, 0, v_init_3350_);
v___x_3376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3376_, 0, v___x_3375_);
return v___x_3376_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15___boxed(lean_object* v___x_3377_, lean_object* v_more_3378_, lean_object* v_catName_3379_, lean_object* v_init_3380_, lean_object* v_x_3381_, lean_object* v___y_3382_, lean_object* v___y_3383_, lean_object* v___y_3384_){
_start:
{
lean_object* v_res_3385_; 
v_res_3385_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15(v___x_3377_, v_more_3378_, v_catName_3379_, v_init_3380_, v_x_3381_, v___y_3382_, v___y_3383_);
lean_dec(v___y_3383_);
lean_dec_ref(v___y_3382_);
lean_dec(v_catName_3379_);
lean_dec(v_more_3378_);
return v_res_3385_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg(lean_object* v_keys_3386_, lean_object* v_vals_3387_, lean_object* v_i_3388_, lean_object* v_k_3389_){
_start:
{
lean_object* v___x_3390_; uint8_t v___x_3391_; 
v___x_3390_ = lean_array_get_size(v_keys_3386_);
v___x_3391_ = lean_nat_dec_lt(v_i_3388_, v___x_3390_);
if (v___x_3391_ == 0)
{
lean_object* v___x_3392_; 
lean_dec(v_i_3388_);
v___x_3392_ = lean_box(0);
return v___x_3392_;
}
else
{
lean_object* v_k_x27_3393_; uint8_t v___x_3394_; 
v_k_x27_3393_ = lean_array_fget_borrowed(v_keys_3386_, v_i_3388_);
v___x_3394_ = lean_name_eq(v_k_3389_, v_k_x27_3393_);
if (v___x_3394_ == 0)
{
lean_object* v___x_3395_; lean_object* v___x_3396_; 
v___x_3395_ = lean_unsigned_to_nat(1u);
v___x_3396_ = lean_nat_add(v_i_3388_, v___x_3395_);
lean_dec(v_i_3388_);
v_i_3388_ = v___x_3396_;
goto _start;
}
else
{
lean_object* v___x_3398_; lean_object* v___x_3399_; 
v___x_3398_ = lean_array_fget_borrowed(v_vals_3387_, v_i_3388_);
lean_dec(v_i_3388_);
lean_inc(v___x_3398_);
v___x_3399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3399_, 0, v___x_3398_);
return v___x_3399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg___boxed(lean_object* v_keys_3400_, lean_object* v_vals_3401_, lean_object* v_i_3402_, lean_object* v_k_3403_){
_start:
{
lean_object* v_res_3404_; 
v_res_3404_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg(v_keys_3400_, v_vals_3401_, v_i_3402_, v_k_3403_);
lean_dec(v_k_3403_);
lean_dec_ref(v_vals_3401_);
lean_dec_ref(v_keys_3400_);
return v_res_3404_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg(lean_object* v_x_3405_, size_t v_x_3406_, lean_object* v_x_3407_){
_start:
{
if (lean_obj_tag(v_x_3405_) == 0)
{
lean_object* v_es_3408_; lean_object* v___x_3409_; size_t v___x_3410_; size_t v___x_3411_; lean_object* v_j_3412_; lean_object* v___x_3413_; 
v_es_3408_ = lean_ctor_get(v_x_3405_, 0);
v___x_3409_ = lean_box(2);
v___x_3410_ = ((size_t)31ULL);
v___x_3411_ = lean_usize_land(v_x_3406_, v___x_3410_);
v_j_3412_ = lean_usize_to_nat(v___x_3411_);
v___x_3413_ = lean_array_get_borrowed(v___x_3409_, v_es_3408_, v_j_3412_);
lean_dec(v_j_3412_);
switch(lean_obj_tag(v___x_3413_))
{
case 0:
{
lean_object* v_key_3414_; lean_object* v_val_3415_; uint8_t v___x_3416_; 
v_key_3414_ = lean_ctor_get(v___x_3413_, 0);
v_val_3415_ = lean_ctor_get(v___x_3413_, 1);
v___x_3416_ = lean_name_eq(v_x_3407_, v_key_3414_);
if (v___x_3416_ == 0)
{
lean_object* v___x_3417_; 
v___x_3417_ = lean_box(0);
return v___x_3417_;
}
else
{
lean_object* v___x_3418_; 
lean_inc(v_val_3415_);
v___x_3418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3418_, 0, v_val_3415_);
return v___x_3418_;
}
}
case 1:
{
lean_object* v_node_3419_; size_t v___x_3420_; size_t v___x_3421_; 
v_node_3419_ = lean_ctor_get(v___x_3413_, 0);
v___x_3420_ = ((size_t)5ULL);
v___x_3421_ = lean_usize_shift_right(v_x_3406_, v___x_3420_);
v_x_3405_ = v_node_3419_;
v_x_3406_ = v___x_3421_;
goto _start;
}
default: 
{
lean_object* v___x_3423_; 
v___x_3423_ = lean_box(0);
return v___x_3423_;
}
}
}
else
{
lean_object* v_ks_3424_; lean_object* v_vs_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; 
v_ks_3424_ = lean_ctor_get(v_x_3405_, 0);
v_vs_3425_ = lean_ctor_get(v_x_3405_, 1);
v___x_3426_ = lean_unsigned_to_nat(0u);
v___x_3427_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg(v_ks_3424_, v_vs_3425_, v___x_3426_, v_x_3407_);
return v___x_3427_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg___boxed(lean_object* v_x_3428_, lean_object* v_x_3429_, lean_object* v_x_3430_){
_start:
{
size_t v_x_22886__boxed_3431_; lean_object* v_res_3432_; 
v_x_22886__boxed_3431_ = lean_unbox_usize(v_x_3429_);
lean_dec(v_x_3429_);
v_res_3432_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg(v_x_3428_, v_x_22886__boxed_3431_, v_x_3430_);
lean_dec(v_x_3430_);
lean_dec_ref(v_x_3428_);
return v_res_3432_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg(lean_object* v_x_3433_, lean_object* v_x_3434_){
_start:
{
uint64_t v___y_3436_; 
if (lean_obj_tag(v_x_3434_) == 0)
{
uint64_t v___x_3439_; 
v___x_3439_ = 1723ULL;
v___y_3436_ = v___x_3439_;
goto v___jp_3435_;
}
else
{
uint64_t v_hash_3440_; 
v_hash_3440_ = lean_ctor_get_uint64(v_x_3434_, sizeof(void*)*2);
v___y_3436_ = v_hash_3440_;
goto v___jp_3435_;
}
v___jp_3435_:
{
size_t v___x_3437_; lean_object* v___x_3438_; 
v___x_3437_ = lean_uint64_to_usize(v___y_3436_);
v___x_3438_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg(v_x_3433_, v___x_3437_, v_x_3434_);
return v___x_3438_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg___boxed(lean_object* v_x_3441_, lean_object* v_x_3442_){
_start:
{
lean_object* v_res_3443_; 
v_res_3443_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg(v_x_3441_, v_x_3442_);
lean_dec(v_x_3442_);
lean_dec_ref(v_x_3441_);
return v_res_3443_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__1(void){
_start:
{
lean_object* v___x_3446_; lean_object* v___x_3447_; lean_object* v___x_3448_; 
v___x_3446_ = lean_box(0);
v___x_3447_ = lean_unsigned_to_nat(16u);
v___x_3448_ = lean_mk_array(v___x_3447_, v___x_3446_);
return v___x_3448_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__2(void){
_start:
{
lean_object* v___x_3449_; lean_object* v___x_3450_; lean_object* v___x_3451_; 
v___x_3449_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__1, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__1_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__1);
v___x_3450_ = lean_unsigned_to_nat(0u);
v___x_3451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3451_, 0, v___x_3450_);
lean_ctor_set(v___x_3451_, 1, v___x_3449_);
return v___x_3451_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__3(void){
_start:
{
lean_object* v___x_3452_; lean_object* v___x_3453_; lean_object* v___x_3454_; 
v___x_3452_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__2, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__2_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__2);
v___x_3453_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__0));
v___x_3454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3454_, 0, v___x_3453_);
lean_ctor_set(v___x_3454_, 1, v___x_3452_);
return v___x_3454_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__4(void){
_start:
{
lean_object* v___x_3455_; lean_object* v_rest_3456_; lean_object* v___x_3457_; 
v___x_3455_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__3, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__3_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__3);
v_rest_3456_ = lean_box(1);
v___x_3457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3457_, 0, v_rest_3456_);
lean_ctor_set(v___x_3457_, 1, v___x_3455_);
return v___x_3457_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6(void){
_start:
{
lean_object* v___x_3459_; lean_object* v___x_3460_; 
v___x_3459_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__5));
v___x_3460_ = l_Lean_stringToMessageData(v___x_3459_);
return v___x_3460_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__8(void){
_start:
{
lean_object* v___x_3462_; lean_object* v___x_3463_; 
v___x_3462_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__7));
v___x_3463_ = l_Lean_stringToMessageData(v___x_3462_);
return v___x_3463_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__10(void){
_start:
{
lean_object* v___x_3465_; lean_object* v___x_3466_; 
v___x_3465_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__9));
v___x_3466_ = l_Lean_stringToMessageData(v___x_3465_);
return v___x_3466_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__12(void){
_start:
{
lean_object* v___x_3468_; lean_object* v___x_3469_; 
v___x_3468_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__11));
v___x_3469_ = l_Lean_stringToMessageData(v___x_3468_);
return v___x_3469_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat(lean_object* v_more_3470_, lean_object* v_catStx_3471_, lean_object* v_id_3472_, lean_object* v_a_3473_, lean_object* v_a_3474_){
_start:
{
lean_object* v___x_3476_; lean_object* v_env_3477_; lean_object* v___x_3478_; lean_object* v_ext_3479_; lean_object* v_toEnvExtension_3480_; lean_object* v_asyncMode_3481_; lean_object* v___x_3482_; lean_object* v___x_3483_; lean_object* v_categories_3484_; lean_object* v___x_3485_; lean_object* v_catName_3486_; lean_object* v___x_3487_; 
v___x_3476_ = lean_st_ref_get(v_a_3474_);
v_env_3477_ = lean_ctor_get(v___x_3476_, 0);
lean_inc_ref(v_env_3477_);
lean_dec(v___x_3476_);
v___x_3478_ = l_Lean_Parser_parserExtension;
v_ext_3479_ = lean_ctor_get(v___x_3478_, 1);
v_toEnvExtension_3480_ = lean_ctor_get(v_ext_3479_, 0);
v_asyncMode_3481_ = lean_ctor_get(v_toEnvExtension_3480_, 2);
v___x_3482_ = l_Lean_Parser_ParserExtension_instInhabitedState_default;
v___x_3483_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3482_, v___x_3478_, v_env_3477_, v_asyncMode_3481_);
v_categories_3484_ = lean_ctor_get(v___x_3483_, 2);
lean_inc_ref(v_categories_3484_);
lean_dec(v___x_3483_);
v___x_3485_ = l_Lean_TSyntax_getId(v_catStx_3471_);
v_catName_3486_ = l_Lean_Name_eraseMacroScopes(v___x_3485_);
lean_dec(v___x_3485_);
v___x_3487_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg(v_categories_3484_, v_catName_3486_);
if (lean_obj_tag(v___x_3487_) == 1)
{
lean_object* v_val_3488_; lean_object* v___x_3489_; lean_object* v___x_3490_; 
v_val_3488_ = lean_ctor_get(v___x_3487_, 0);
lean_inc(v_val_3488_);
lean_dec_ref_known(v___x_3487_, 1);
lean_inc(v_catName_3486_);
v___x_3489_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_addCategoryInfo___boxed), 9, 2);
lean_closure_set(v___x_3489_, 0, v_catStx_3471_);
lean_closure_set(v___x_3489_, 1, v_catName_3486_);
v___x_3490_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___x_3489_, v_a_3473_, v_a_3474_);
if (lean_obj_tag(v___x_3490_) == 0)
{
lean_object* v_kinds_3491_; lean_object* v_rest_3492_; lean_object* v___f_3493_; lean_object* v___x_3494_; lean_object* v___x_3495_; 
lean_dec_ref_known(v___x_3490_, 1);
v_kinds_3491_ = lean_ctor_get(v_val_3488_, 1);
lean_inc_ref(v_kinds_3491_);
lean_dec(v_val_3488_);
v_rest_3492_ = lean_box(1);
lean_inc(v_id_3472_);
v___f_3493_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___lam__2___boxed), 7, 2);
lean_closure_set(v___f_3493_, 0, v_categories_3484_);
lean_closure_set(v___f_3493_, 1, v_id_3472_);
v___x_3494_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__4, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__4_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__4);
v___x_3495_ = lp_batteries_Lean_PersistentHashMap_forIn___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCats_spec__0___redArg(v_kinds_3491_, v___x_3494_, v___f_3493_, v_a_3473_, v_a_3474_);
lean_dec_ref(v_kinds_3491_);
if (lean_obj_tag(v___x_3495_) == 0)
{
lean_object* v_a_3496_; lean_object* v_fst_3497_; lean_object* v_snd_3498_; lean_object* v___x_3500_; uint8_t v_isShared_3501_; uint8_t v_isSharedCheck_3577_; 
v_a_3496_ = lean_ctor_get(v___x_3495_, 0);
lean_inc(v_a_3496_);
lean_dec_ref_known(v___x_3495_, 1);
v_fst_3497_ = lean_ctor_get(v_a_3496_, 0);
v_snd_3498_ = lean_ctor_get(v_a_3496_, 1);
v_isSharedCheck_3577_ = !lean_is_exclusive(v_a_3496_);
if (v_isSharedCheck_3577_ == 0)
{
v___x_3500_ = v_a_3496_;
v_isShared_3501_ = v_isSharedCheck_3577_;
goto v_resetjp_3499_;
}
else
{
lean_inc(v_snd_3498_);
lean_inc(v_fst_3497_);
lean_dec(v_a_3496_);
v___x_3500_ = lean_box(0);
v_isShared_3501_ = v_isSharedCheck_3577_;
goto v_resetjp_3499_;
}
v_resetjp_3499_:
{
lean_object* v___y_3503_; lean_object* v___y_3504_; lean_object* v___y_3505_; lean_object* v_a_3506_; lean_object* v_fst_3519_; lean_object* v_snd_3520_; lean_object* v___x_3522_; uint8_t v_isShared_3523_; uint8_t v_isSharedCheck_3576_; 
v_fst_3519_ = lean_ctor_get(v_snd_3498_, 0);
v_snd_3520_ = lean_ctor_get(v_snd_3498_, 1);
v_isSharedCheck_3576_ = !lean_is_exclusive(v_snd_3498_);
if (v_isSharedCheck_3576_ == 0)
{
v___x_3522_ = v_snd_3498_;
v_isShared_3523_ = v_isSharedCheck_3576_;
goto v_resetjp_3521_;
}
else
{
lean_inc(v_snd_3520_);
lean_inc(v_fst_3519_);
lean_dec(v_snd_3498_);
v___x_3522_ = lean_box(0);
v_isShared_3523_ = v_isSharedCheck_3576_;
goto v_resetjp_3521_;
}
v___jp_3502_:
{
lean_object* v___x_3507_; 
v___x_3507_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__14(v___y_3505_, v_more_3470_, v_catName_3486_, v_a_3506_, v_fst_3497_, v___y_3503_, v___y_3504_);
lean_dec(v_catName_3486_);
if (lean_obj_tag(v___x_3507_) == 0)
{
lean_object* v_a_3508_; lean_object* v_a_3509_; lean_object* v___x_3510_; 
v_a_3508_ = lean_ctor_get(v___x_3507_, 0);
lean_inc(v_a_3508_);
lean_dec_ref_known(v___x_3507_, 1);
v_a_3509_ = lean_ctor_get(v_a_3508_, 0);
lean_inc(v_a_3509_);
lean_dec(v_a_3508_);
v___x_3510_ = lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(v_a_3509_, v___y_3503_, v___y_3504_);
return v___x_3510_;
}
else
{
lean_object* v_a_3511_; lean_object* v___x_3513_; uint8_t v_isShared_3514_; uint8_t v_isSharedCheck_3518_; 
v_a_3511_ = lean_ctor_get(v___x_3507_, 0);
v_isSharedCheck_3518_ = !lean_is_exclusive(v___x_3507_);
if (v_isSharedCheck_3518_ == 0)
{
v___x_3513_ = v___x_3507_;
v_isShared_3514_ = v_isSharedCheck_3518_;
goto v_resetjp_3512_;
}
else
{
lean_inc(v_a_3511_);
lean_dec(v___x_3507_);
v___x_3513_ = lean_box(0);
v_isShared_3514_ = v_isSharedCheck_3518_;
goto v_resetjp_3512_;
}
v_resetjp_3512_:
{
lean_object* v___x_3516_; 
if (v_isShared_3514_ == 0)
{
v___x_3516_ = v___x_3513_;
goto v_reusejp_3515_;
}
else
{
lean_object* v_reuseFailAlloc_3517_; 
v_reuseFailAlloc_3517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3517_, 0, v_a_3511_);
v___x_3516_ = v_reuseFailAlloc_3517_;
goto v_reusejp_3515_;
}
v_reusejp_3515_:
{
return v___x_3516_;
}
}
}
}
v_resetjp_3521_:
{
size_t v_sz_3524_; size_t v___x_3525_; lean_object* v___x_3526_; 
v_sz_3524_ = lean_array_size(v_fst_3519_);
v___x_3525_ = ((size_t)0ULL);
v___x_3526_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg(v_snd_3520_, v_fst_3519_, v_sz_3524_, v___x_3525_, v_rest_3492_);
lean_dec(v_fst_3519_);
if (lean_obj_tag(v___x_3526_) == 0)
{
lean_object* v_a_3527_; lean_object* v___x_3528_; lean_object* v___y_3530_; lean_object* v___y_3531_; 
v_a_3527_ = lean_ctor_get(v___x_3526_, 0);
lean_inc(v_a_3527_);
lean_dec_ref_known(v___x_3526_, 1);
v___x_3528_ = l_Lean_MessageData_nil;
if (lean_obj_tag(v_a_3527_) == 0)
{
lean_del_object(v___x_3522_);
lean_del_object(v___x_3500_);
lean_dec(v_id_3472_);
v___y_3530_ = v_a_3473_;
v___y_3531_ = v_a_3474_;
goto v___jp_3529_;
}
else
{
if (lean_obj_tag(v_fst_3497_) == 0)
{
lean_del_object(v___x_3522_);
lean_del_object(v___x_3500_);
lean_dec(v_id_3472_);
v___y_3530_ = v_a_3473_;
v___y_3531_ = v_a_3474_;
goto v___jp_3529_;
}
else
{
if (lean_obj_tag(v_id_3472_) == 0)
{
lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v___x_3548_; 
v___x_3545_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6);
v___x_3546_ = l_Lean_MessageData_ofName(v_catName_3486_);
if (v_isShared_3523_ == 0)
{
lean_ctor_set_tag(v___x_3522_, 7);
lean_ctor_set(v___x_3522_, 1, v___x_3546_);
lean_ctor_set(v___x_3522_, 0, v___x_3545_);
v___x_3548_ = v___x_3522_;
goto v_reusejp_3547_;
}
else
{
lean_object* v_reuseFailAlloc_3554_; 
v_reuseFailAlloc_3554_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3554_, 0, v___x_3545_);
lean_ctor_set(v_reuseFailAlloc_3554_, 1, v___x_3546_);
v___x_3548_ = v_reuseFailAlloc_3554_;
goto v_reusejp_3547_;
}
v_reusejp_3547_:
{
lean_object* v___x_3549_; lean_object* v___x_3551_; 
v___x_3549_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__8, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__8_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__8);
if (v_isShared_3501_ == 0)
{
lean_ctor_set_tag(v___x_3500_, 7);
lean_ctor_set(v___x_3500_, 1, v___x_3549_);
lean_ctor_set(v___x_3500_, 0, v___x_3548_);
v___x_3551_ = v___x_3500_;
goto v_reusejp_3550_;
}
else
{
lean_object* v_reuseFailAlloc_3553_; 
v_reuseFailAlloc_3553_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3553_, 0, v___x_3548_);
lean_ctor_set(v_reuseFailAlloc_3553_, 1, v___x_3549_);
v___x_3551_ = v_reuseFailAlloc_3553_;
goto v_reusejp_3550_;
}
v_reusejp_3550_:
{
lean_object* v___x_3552_; 
v___x_3552_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_3551_, v_a_3473_, v_a_3474_);
return v___x_3552_;
}
}
}
else
{
lean_object* v_val_3555_; lean_object* v___x_3556_; lean_object* v___x_3557_; lean_object* v___x_3559_; 
v_val_3555_ = lean_ctor_get(v_id_3472_, 0);
lean_inc(v_val_3555_);
lean_dec_ref_known(v_id_3472_, 1);
v___x_3556_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__6);
v___x_3557_ = l_Lean_MessageData_ofName(v_catName_3486_);
if (v_isShared_3523_ == 0)
{
lean_ctor_set_tag(v___x_3522_, 7);
lean_ctor_set(v___x_3522_, 1, v___x_3557_);
lean_ctor_set(v___x_3522_, 0, v___x_3556_);
v___x_3559_ = v___x_3522_;
goto v_reusejp_3558_;
}
else
{
lean_object* v_reuseFailAlloc_3567_; 
v_reuseFailAlloc_3567_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3567_, 0, v___x_3556_);
lean_ctor_set(v_reuseFailAlloc_3567_, 1, v___x_3557_);
v___x_3559_ = v_reuseFailAlloc_3567_;
goto v_reusejp_3558_;
}
v_reusejp_3558_:
{
lean_object* v___x_3560_; lean_object* v___x_3562_; 
v___x_3560_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__10, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__10_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__10);
if (v_isShared_3501_ == 0)
{
lean_ctor_set_tag(v___x_3500_, 7);
lean_ctor_set(v___x_3500_, 1, v___x_3560_);
lean_ctor_set(v___x_3500_, 0, v___x_3559_);
v___x_3562_ = v___x_3500_;
goto v_reusejp_3561_;
}
else
{
lean_object* v_reuseFailAlloc_3566_; 
v_reuseFailAlloc_3566_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3566_, 0, v___x_3559_);
lean_ctor_set(v_reuseFailAlloc_3566_, 1, v___x_3560_);
v___x_3562_ = v_reuseFailAlloc_3566_;
goto v_reusejp_3561_;
}
v_reusejp_3561_:
{
lean_object* v___x_3563_; lean_object* v___x_3564_; lean_object* v___x_3565_; 
v___x_3563_ = l_Lean_stringToMessageData(v_val_3555_);
v___x_3564_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3564_, 0, v___x_3562_);
lean_ctor_set(v___x_3564_, 1, v___x_3563_);
v___x_3565_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__2___redArg(v___x_3564_, v_a_3473_, v_a_3474_);
return v___x_3565_;
}
}
}
}
}
v___jp_3529_:
{
lean_object* v___x_3532_; lean_object* v_env_3533_; lean_object* v___x_3534_; 
v___x_3532_ = lean_st_ref_get(v___y_3531_);
v_env_3533_ = lean_ctor_get(v___x_3532_, 0);
lean_inc_ref_n(v_env_3533_, 2);
lean_dec(v___x_3532_);
v___x_3534_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__15(v_env_3533_, v_more_3470_, v_catName_3486_, v___x_3528_, v_a_3527_, v___y_3530_, v___y_3531_);
if (lean_obj_tag(v___x_3534_) == 0)
{
lean_object* v_a_3535_; lean_object* v_a_3536_; 
v_a_3535_ = lean_ctor_get(v___x_3534_, 0);
lean_inc(v_a_3535_);
lean_dec_ref_known(v___x_3534_, 1);
v_a_3536_ = lean_ctor_get(v_a_3535_, 0);
lean_inc(v_a_3536_);
lean_dec(v_a_3535_);
v___y_3503_ = v___y_3530_;
v___y_3504_ = v___y_3531_;
v___y_3505_ = v_env_3533_;
v_a_3506_ = v_a_3536_;
goto v___jp_3502_;
}
else
{
lean_object* v_a_3537_; lean_object* v___x_3539_; uint8_t v_isShared_3540_; uint8_t v_isSharedCheck_3544_; 
lean_dec_ref(v_env_3533_);
lean_dec(v_fst_3497_);
lean_dec(v_catName_3486_);
v_a_3537_ = lean_ctor_get(v___x_3534_, 0);
v_isSharedCheck_3544_ = !lean_is_exclusive(v___x_3534_);
if (v_isSharedCheck_3544_ == 0)
{
v___x_3539_ = v___x_3534_;
v_isShared_3540_ = v_isSharedCheck_3544_;
goto v_resetjp_3538_;
}
else
{
lean_inc(v_a_3537_);
lean_dec(v___x_3534_);
v___x_3539_ = lean_box(0);
v_isShared_3540_ = v_isSharedCheck_3544_;
goto v_resetjp_3538_;
}
v_resetjp_3538_:
{
lean_object* v___x_3542_; 
if (v_isShared_3540_ == 0)
{
v___x_3542_ = v___x_3539_;
goto v_reusejp_3541_;
}
else
{
lean_object* v_reuseFailAlloc_3543_; 
v_reuseFailAlloc_3543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3543_, 0, v_a_3537_);
v___x_3542_ = v_reuseFailAlloc_3543_;
goto v_reusejp_3541_;
}
v_reusejp_3541_:
{
return v___x_3542_;
}
}
}
}
}
else
{
lean_object* v_a_3568_; lean_object* v___x_3570_; uint8_t v_isShared_3571_; uint8_t v_isSharedCheck_3575_; 
lean_del_object(v___x_3522_);
lean_del_object(v___x_3500_);
lean_dec(v_fst_3497_);
lean_dec(v_catName_3486_);
lean_dec(v_id_3472_);
v_a_3568_ = lean_ctor_get(v___x_3526_, 0);
v_isSharedCheck_3575_ = !lean_is_exclusive(v___x_3526_);
if (v_isSharedCheck_3575_ == 0)
{
v___x_3570_ = v___x_3526_;
v_isShared_3571_ = v_isSharedCheck_3575_;
goto v_resetjp_3569_;
}
else
{
lean_inc(v_a_3568_);
lean_dec(v___x_3526_);
v___x_3570_ = lean_box(0);
v_isShared_3571_ = v_isSharedCheck_3575_;
goto v_resetjp_3569_;
}
v_resetjp_3569_:
{
lean_object* v___x_3573_; 
if (v_isShared_3571_ == 0)
{
v___x_3573_ = v___x_3570_;
goto v_reusejp_3572_;
}
else
{
lean_object* v_reuseFailAlloc_3574_; 
v_reuseFailAlloc_3574_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3574_, 0, v_a_3568_);
v___x_3573_ = v_reuseFailAlloc_3574_;
goto v_reusejp_3572_;
}
v_reusejp_3572_:
{
return v___x_3573_;
}
}
}
}
}
}
else
{
lean_object* v_a_3578_; lean_object* v___x_3580_; uint8_t v_isShared_3581_; uint8_t v_isSharedCheck_3585_; 
lean_dec(v_catName_3486_);
lean_dec(v_id_3472_);
v_a_3578_ = lean_ctor_get(v___x_3495_, 0);
v_isSharedCheck_3585_ = !lean_is_exclusive(v___x_3495_);
if (v_isSharedCheck_3585_ == 0)
{
v___x_3580_ = v___x_3495_;
v_isShared_3581_ = v_isSharedCheck_3585_;
goto v_resetjp_3579_;
}
else
{
lean_inc(v_a_3578_);
lean_dec(v___x_3495_);
v___x_3580_ = lean_box(0);
v_isShared_3581_ = v_isSharedCheck_3585_;
goto v_resetjp_3579_;
}
v_resetjp_3579_:
{
lean_object* v___x_3583_; 
if (v_isShared_3581_ == 0)
{
v___x_3583_ = v___x_3580_;
goto v_reusejp_3582_;
}
else
{
lean_object* v_reuseFailAlloc_3584_; 
v_reuseFailAlloc_3584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3584_, 0, v_a_3578_);
v___x_3583_ = v_reuseFailAlloc_3584_;
goto v_reusejp_3582_;
}
v_reusejp_3582_:
{
return v___x_3583_;
}
}
}
}
else
{
lean_dec(v_val_3488_);
lean_dec(v_catName_3486_);
lean_dec_ref(v_categories_3484_);
lean_dec(v_id_3472_);
return v___x_3490_;
}
}
else
{
lean_object* v___x_3586_; lean_object* v___x_3587_; lean_object* v___x_3588_; lean_object* v___x_3589_; 
lean_dec(v___x_3487_);
lean_dec(v_catName_3486_);
lean_dec_ref(v_categories_3484_);
lean_dec(v_id_3472_);
lean_inc(v_catStx_3471_);
v___x_3586_ = l_Lean_MessageData_ofSyntax(v_catStx_3471_);
v___x_3587_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__12, &lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__12_once, _init_lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___closed__12);
v___x_3588_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3588_, 0, v___x_3586_);
lean_ctor_set(v___x_3588_, 1, v___x_3587_);
v___x_3589_ = lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg(v_catStx_3471_, v___x_3588_, v_a_3473_, v_a_3474_);
lean_dec(v_catStx_3471_);
return v___x_3589_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat___boxed(lean_object* v_more_3590_, lean_object* v_catStx_3591_, lean_object* v_id_3592_, lean_object* v_a_3593_, lean_object* v_a_3594_, lean_object* v_a_3595_){
_start:
{
lean_object* v_res_3596_; 
v_res_3596_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat(v_more_3590_, v_catStx_3591_, v_id_3592_, v_a_3593_, v_a_3594_);
lean_dec(v_a_3594_);
lean_dec_ref(v_a_3593_);
lean_dec(v_more_3590_);
return v_res_3596_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0(lean_object* v_00_u03b2_3597_, lean_object* v_x_3598_, lean_object* v_x_3599_){
_start:
{
lean_object* v___x_3600_; 
v___x_3600_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___redArg(v_x_3598_, v_x_3599_);
return v___x_3600_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0___boxed(lean_object* v_00_u03b2_3601_, lean_object* v_x_3602_, lean_object* v_x_3603_){
_start:
{
lean_object* v_res_3604_; 
v_res_3604_ = lp_batteries_Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0(v_00_u03b2_3601_, v_x_3602_, v_x_3603_);
lean_dec(v_x_3603_);
lean_dec_ref(v_x_3602_);
return v_res_3604_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4(lean_object* v_as_3605_, lean_object* v_as_x27_3606_, lean_object* v_b_3607_, lean_object* v_a_3608_, lean_object* v___y_3609_, lean_object* v___y_3610_){
_start:
{
lean_object* v___x_3612_; 
v___x_3612_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___redArg(v_as_x27_3606_, v_b_3607_);
return v___x_3612_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4___boxed(lean_object* v_as_3613_, lean_object* v_as_x27_3614_, lean_object* v_b_3615_, lean_object* v_a_3616_, lean_object* v___y_3617_, lean_object* v___y_3618_, lean_object* v___y_3619_){
_start:
{
lean_object* v_res_3620_; 
v_res_3620_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__4(v_as_3613_, v_as_x27_3614_, v_b_3615_, v_a_3616_, v___y_3617_, v___y_3618_);
lean_dec(v___y_3618_);
lean_dec_ref(v___y_3617_);
lean_dec(v_as_x27_3614_);
lean_dec(v_as_3613_);
return v_res_3620_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7(lean_object* v_fst_3621_, uint8_t v_snd_3622_, lean_object* v_k_3623_, lean_object* v_t_3624_, lean_object* v_hl_3625_){
_start:
{
lean_object* v___x_3626_; 
v___x_3626_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___redArg(v_fst_3621_, v_snd_3622_, v_k_3623_, v_t_3624_);
return v___x_3626_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7___boxed(lean_object* v_fst_3627_, lean_object* v_snd_3628_, lean_object* v_k_3629_, lean_object* v_t_3630_, lean_object* v_hl_3631_){
_start:
{
uint8_t v_snd_23269__boxed_3632_; lean_object* v_res_3633_; 
v_snd_23269__boxed_3632_ = lean_unbox(v_snd_3628_);
v_res_3633_ = lp_batteries_Std_DTreeMap_Internal_Impl_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__7(v_fst_3627_, v_snd_23269__boxed_3632_, v_k_3629_, v_t_3630_, v_hl_3631_);
return v_res_3633_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11(lean_object* v___x_3634_, lean_object* v_as_3635_, size_t v_sz_3636_, size_t v_i_3637_, lean_object* v_b_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_){
_start:
{
lean_object* v___x_3642_; 
v___x_3642_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___redArg(v___x_3634_, v_as_3635_, v_sz_3636_, v_i_3637_, v_b_3638_);
return v___x_3642_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11___boxed(lean_object* v___x_3643_, lean_object* v_as_3644_, lean_object* v_sz_3645_, lean_object* v_i_3646_, lean_object* v_b_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_){
_start:
{
size_t v_sz_boxed_3651_; size_t v_i_boxed_3652_; lean_object* v_res_3653_; 
v_sz_boxed_3651_ = lean_unbox_usize(v_sz_3645_);
lean_dec(v_sz_3645_);
v_i_boxed_3652_ = lean_unbox_usize(v_i_3646_);
lean_dec(v_i_3646_);
v_res_3653_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__11(v___x_3643_, v_as_3644_, v_sz_boxed_3651_, v_i_boxed_3652_, v_b_3647_, v___y_3648_, v___y_3649_);
lean_dec(v___y_3649_);
lean_dec_ref(v___y_3648_);
lean_dec_ref(v_as_3644_);
return v_res_3653_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12(lean_object* v_00_u03b1_3654_, lean_object* v___x_3655_, lean_object* v_type_3656_, lean_object* v_as_3657_, lean_object* v_as_x27_3658_, lean_object* v_b_3659_, lean_object* v_a_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_){
_start:
{
lean_object* v___x_3664_; 
v___x_3664_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___redArg(v___x_3655_, v_type_3656_, v_as_x27_3658_, v_b_3659_, v___y_3661_);
return v___x_3664_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12___boxed(lean_object* v_00_u03b1_3665_, lean_object* v___x_3666_, lean_object* v_type_3667_, lean_object* v_as_3668_, lean_object* v_as_x27_3669_, lean_object* v_b_3670_, lean_object* v_a_3671_, lean_object* v___y_3672_, lean_object* v___y_3673_, lean_object* v___y_3674_){
_start:
{
lean_object* v_res_3675_; 
v_res_3675_ = lp_batteries_List_forIn_x27_loop___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__12(v_00_u03b1_3665_, v___x_3666_, v_type_3667_, v_as_3668_, v_as_x27_3669_, v_b_3670_, v_a_3671_, v___y_3672_, v___y_3673_);
lean_dec(v___y_3673_);
lean_dec_ref(v___y_3672_);
lean_dec(v_as_x27_3669_);
lean_dec(v_as_3668_);
return v_res_3675_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16(lean_object* v_00_u03b1_3676_, lean_object* v_ref_3677_, lean_object* v_msg_3678_, lean_object* v___y_3679_, lean_object* v___y_3680_){
_start:
{
lean_object* v___x_3682_; 
v___x_3682_ = lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___redArg(v_ref_3677_, v_msg_3678_, v___y_3679_, v___y_3680_);
return v___x_3682_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16___boxed(lean_object* v_00_u03b1_3683_, lean_object* v_ref_3684_, lean_object* v_msg_3685_, lean_object* v___y_3686_, lean_object* v___y_3687_, lean_object* v___y_3688_){
_start:
{
lean_object* v_res_3689_; 
v_res_3689_ = lp_batteries_Lean_throwErrorAt___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__16(v_00_u03b1_3683_, v_ref_3684_, v_msg_3685_, v___y_3686_, v___y_3687_);
lean_dec(v___y_3687_);
lean_dec_ref(v___y_3686_);
lean_dec(v_ref_3684_);
return v_res_3689_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0(lean_object* v_00_u03b2_3690_, lean_object* v_x_3691_, size_t v_x_3692_, lean_object* v_x_3693_){
_start:
{
lean_object* v___x_3694_; 
v___x_3694_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___redArg(v_x_3691_, v_x_3692_, v_x_3693_);
return v___x_3694_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0___boxed(lean_object* v_00_u03b2_3695_, lean_object* v_x_3696_, lean_object* v_x_3697_, lean_object* v_x_3698_){
_start:
{
size_t v_x_23319__boxed_3699_; lean_object* v_res_3700_; 
v_x_23319__boxed_3699_ = lean_unbox_usize(v_x_3697_);
lean_dec(v_x_3697_);
v_res_3700_ = lp_batteries_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0(v_00_u03b2_3695_, v_x_3696_, v_x_23319__boxed_3699_, v_x_3698_);
lean_dec(v_x_3698_);
lean_dec_ref(v_x_3696_);
return v_res_3700_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2(lean_object* v_00_u03b2_3701_, lean_object* v_a_3702_, lean_object* v_x_3703_){
_start:
{
uint8_t v___x_3704_; 
v___x_3704_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___redArg(v_a_3702_, v_x_3703_);
return v___x_3704_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2___boxed(lean_object* v_00_u03b2_3705_, lean_object* v_a_3706_, lean_object* v_x_3707_){
_start:
{
uint8_t v_res_3708_; lean_object* v_r_3709_; 
v_res_3708_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__2(v_00_u03b2_3705_, v_a_3706_, v_x_3707_);
lean_dec(v_x_3707_);
lean_dec_ref(v_a_3706_);
v_r_3709_ = lean_box(v_res_3708_);
return v_r_3709_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3(lean_object* v_00_u03b2_3710_, lean_object* v_data_3711_){
_start:
{
lean_object* v___x_3712_; 
v___x_3712_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3___redArg(v_data_3711_);
return v___x_3712_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4(lean_object* v_00_u03b2_3713_, lean_object* v_keys_3714_, lean_object* v_vals_3715_, lean_object* v_heq_3716_, lean_object* v_i_3717_, lean_object* v_k_3718_){
_start:
{
lean_object* v___x_3719_; 
v___x_3719_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___redArg(v_keys_3714_, v_vals_3715_, v_i_3717_, v_k_3718_);
return v___x_3719_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4___boxed(lean_object* v_00_u03b2_3720_, lean_object* v_keys_3721_, lean_object* v_vals_3722_, lean_object* v_heq_3723_, lean_object* v_i_3724_, lean_object* v_k_3725_){
_start:
{
lean_object* v_res_3726_; 
v_res_3726_ = lp_batteries_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__0_spec__0_spec__4(v_00_u03b2_3720_, v_keys_3721_, v_vals_3722_, v_heq_3723_, v_i_3724_, v_k_3725_);
lean_dec(v_k_3725_);
lean_dec_ref(v_vals_3722_);
lean_dec_ref(v_keys_3721_);
return v_res_3726_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8(lean_object* v_00_u03b2_3727_, lean_object* v_i_3728_, lean_object* v_source_3729_, lean_object* v_target_3730_){
_start:
{
lean_object* v___x_3731_; 
v___x_3731_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8___redArg(v_i_3728_, v_source_3729_, v_target_3730_);
return v___x_3731_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8_spec__21(lean_object* v_00_u03b2_3732_, lean_object* v_x_3733_, lean_object* v_x_3734_){
_start:
{
lean_object* v___x_3735_; 
v___x_3735_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat_spec__1_spec__3_spec__8_spec__21___redArg(v_x_3733_, v_x_3734_);
return v___x_3735_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cat_x2b______________1(lean_object* v_x_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_){
_start:
{
lean_object* v___x_3740_; uint8_t v___x_3741_; 
v___x_3740_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1));
lean_inc(v_x_3736_);
v___x_3741_ = l_Lean_Syntax_isOfKind(v_x_3736_, v___x_3740_);
if (v___x_3741_ == 0)
{
lean_object* v___x_3742_; 
lean_dec(v_x_3736_);
v___x_3742_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_3742_;
}
else
{
lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v_more_3746_; lean_object* v___y_3747_; lean_object* v___y_3748_; lean_object* v___x_3771_; lean_object* v___x_3772_; uint8_t v___x_3773_; 
v___x_3743_ = lean_unsigned_to_nat(0u);
v___x_3744_ = lean_unsigned_to_nat(1u);
v___x_3771_ = lean_unsigned_to_nat(2u);
v___x_3772_ = l_Lean_Syntax_getArg(v_x_3736_, v___x_3771_);
v___x_3773_ = l_Lean_Syntax_isNone(v___x_3772_);
if (v___x_3773_ == 0)
{
uint8_t v___x_3774_; 
lean_inc(v___x_3772_);
v___x_3774_ = l_Lean_Syntax_matchesNull(v___x_3772_, v___x_3744_);
if (v___x_3774_ == 0)
{
lean_object* v___x_3775_; 
lean_dec(v___x_3772_);
lean_dec(v_x_3736_);
v___x_3775_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_3775_;
}
else
{
lean_object* v_more_3776_; lean_object* v___x_3777_; 
v_more_3776_ = l_Lean_Syntax_getArg(v___x_3772_, v___x_3743_);
lean_dec(v___x_3772_);
v___x_3777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3777_, 0, v_more_3776_);
v_more_3746_ = v___x_3777_;
v___y_3747_ = v_a_3737_;
v___y_3748_ = v_a_3738_;
goto v___jp_3745_;
}
}
else
{
lean_object* v___x_3778_; 
lean_dec(v___x_3772_);
v___x_3778_ = lean_box(0);
v_more_3746_ = v___x_3778_;
v___y_3747_ = v_a_3737_;
v___y_3748_ = v_a_3738_;
goto v___jp_3745_;
}
v___jp_3745_:
{
lean_object* v___x_3749_; lean_object* v_cat_3750_; lean_object* v___x_3751_; lean_object* v___x_3752_; uint8_t v___x_3753_; 
v___x_3749_ = lean_unsigned_to_nat(3u);
v_cat_3750_ = l_Lean_Syntax_getArg(v_x_3736_, v___x_3749_);
v___x_3751_ = lean_unsigned_to_nat(4u);
v___x_3752_ = l_Lean_Syntax_getArg(v_x_3736_, v___x_3751_);
lean_dec(v_x_3736_);
lean_inc(v___x_3752_);
v___x_3753_ = l_Lean_Syntax_matchesNull(v___x_3752_, v___x_3743_);
if (v___x_3753_ == 0)
{
uint8_t v___x_3754_; 
lean_inc(v___x_3752_);
v___x_3754_ = l_Lean_Syntax_matchesNull(v___x_3752_, v___x_3744_);
if (v___x_3754_ == 0)
{
lean_object* v___x_3755_; 
lean_dec(v___x_3752_);
lean_dec(v_cat_3750_);
lean_dec(v_more_3746_);
v___x_3755_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_3755_;
}
else
{
lean_object* v_id_3756_; lean_object* v___x_3757_; uint8_t v___x_3758_; 
v_id_3756_ = l_Lean_Syntax_getArg(v___x_3752_, v___x_3743_);
lean_dec(v___x_3752_);
v___x_3757_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__11));
lean_inc(v_id_3756_);
v___x_3758_ = l_Lean_Syntax_isOfKind(v_id_3756_, v___x_3757_);
if (v___x_3758_ == 0)
{
lean_object* v___x_3759_; uint8_t v___x_3760_; 
v___x_3759_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__15));
lean_inc(v_id_3756_);
v___x_3760_ = l_Lean_Syntax_isOfKind(v_id_3756_, v___x_3759_);
if (v___x_3760_ == 0)
{
lean_object* v___x_3761_; 
lean_dec(v_id_3756_);
lean_dec(v_cat_3750_);
lean_dec(v_more_3746_);
v___x_3761_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_3761_;
}
else
{
lean_object* v___x_3762_; lean_object* v___x_3763_; lean_object* v___x_3764_; 
v___x_3762_ = l_Lean_TSyntax_getString(v_id_3756_);
lean_dec(v_id_3756_);
v___x_3763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3763_, 0, v___x_3762_);
v___x_3764_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat(v_more_3746_, v_cat_3750_, v___x_3763_, v___y_3747_, v___y_3748_);
lean_dec(v_more_3746_);
return v___x_3764_;
}
}
else
{
lean_object* v___x_3765_; lean_object* v___x_3766_; lean_object* v___x_3767_; lean_object* v___x_3768_; 
v___x_3765_ = l_Lean_TSyntax_getId(v_id_3756_);
lean_dec(v_id_3756_);
v___x_3766_ = l_Lean_Name_toString(v___x_3765_, v___x_3753_);
v___x_3767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3767_, 0, v___x_3766_);
v___x_3768_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat(v_more_3746_, v_cat_3750_, v___x_3767_, v___y_3747_, v___y_3748_);
lean_dec(v_more_3746_);
return v___x_3768_;
}
}
}
else
{
lean_object* v___x_3769_; lean_object* v___x_3770_; 
lean_dec(v___x_3752_);
v___x_3769_ = lean_box(0);
v___x_3770_ = lp_batteries___private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpCat(v_more_3746_, v_cat_3750_, v___x_3769_, v___y_3747_, v___y_3748_);
lean_dec(v_more_3746_);
return v___x_3770_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cat_x2b______________1___boxed(lean_object* v_x_3779_, lean_object* v_a_3780_, lean_object* v_a_3781_, lean_object* v_a_3782_){
_start:
{
lean_object* v_res_3783_; 
v_res_3783_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Cat_x2b______________1(v_x_3779_, v_a_3780_, v_a_3781_);
lean_dec(v_a_3781_);
lean_dec_ref(v_a_3780_);
return v_res_3783_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__0(lean_object* v_n_3834_){
_start:
{
uint8_t v___x_3835_; lean_object* v___x_3836_; 
v___x_3835_ = 0;
v___x_3836_ = l_Lean_Name_toString(v_n_3834_, v___x_3835_);
return v___x_3836_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__1(lean_object* v___f_3837_, lean_object* v_x1_3838_, lean_object* v_x2_3839_){
_start:
{
lean_object* v___x_3840_; lean_object* v___x_3841_; uint8_t v___x_3842_; 
lean_inc_ref(v___f_3837_);
v___x_3840_ = lean_apply_1(v___f_3837_, v_x1_3838_);
v___x_3841_ = lean_apply_1(v___f_3837_, v_x2_3839_);
v___x_3842_ = l_String_decLE(v___x_3840_, v___x_3841_);
lean_dec_ref(v___x_3841_);
lean_dec_ref(v___x_3840_);
return v___x_3842_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__1___boxed(lean_object* v___f_3843_, lean_object* v_x1_3844_, lean_object* v_x2_3845_){
_start:
{
uint8_t v_res_3846_; lean_object* v_r_3847_; 
v_res_3846_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___lam__1(v___f_3843_, v_x1_3844_, v_x2_3845_);
v_r_3847_ = lean_box(v_res_3846_);
return v_r_3847_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__2(lean_object* v_msgData_3848_, lean_object* v___y_3849_, lean_object* v___y_3850_){
_start:
{
uint8_t v___x_3852_; uint8_t v___x_3853_; lean_object* v___x_3854_; 
v___x_3852_ = 2;
v___x_3853_ = 0;
v___x_3854_ = lp_batteries_Lean_log___at___00Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0_spec__0(v_msgData_3848_, v___x_3852_, v___x_3853_, v___y_3849_, v___y_3850_);
return v___x_3854_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__2___boxed(lean_object* v_msgData_3855_, lean_object* v___y_3856_, lean_object* v___y_3857_, lean_object* v___y_3858_){
_start:
{
lean_object* v_res_3859_; 
v_res_3859_ = lp_batteries_Lean_logError___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__2(v_msgData_3855_, v___y_3856_, v___y_3857_);
lean_dec(v___y_3857_);
lean_dec_ref(v___y_3856_);
return v_res_3859_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3(lean_object* v_as_3860_, size_t v_i_3861_, size_t v_stop_3862_, lean_object* v_b_3863_){
_start:
{
uint8_t v___x_3864_; 
v___x_3864_ = lean_usize_dec_eq(v_i_3861_, v_stop_3862_);
if (v___x_3864_ == 0)
{
lean_object* v___x_3865_; lean_object* v___x_3866_; size_t v___x_3867_; size_t v___x_3868_; 
v___x_3865_ = lean_array_uget_borrowed(v_as_3860_, v_i_3861_);
v___x_3866_ = l_Array_append___redArg(v_b_3863_, v___x_3865_);
v___x_3867_ = ((size_t)1ULL);
v___x_3868_ = lean_usize_add(v_i_3861_, v___x_3867_);
v_i_3861_ = v___x_3868_;
v_b_3863_ = v___x_3866_;
goto _start;
}
else
{
return v_b_3863_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3___boxed(lean_object* v_as_3870_, lean_object* v_i_3871_, lean_object* v_stop_3872_, lean_object* v_b_3873_){
_start:
{
size_t v_i_boxed_3874_; size_t v_stop_boxed_3875_; lean_object* v_res_3876_; 
v_i_boxed_3874_ = lean_unbox_usize(v_i_3871_);
lean_dec(v_i_3871_);
v_stop_boxed_3875_ = lean_unbox_usize(v_stop_3872_);
lean_dec(v_stop_3872_);
v_res_3876_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3(v_as_3870_, v_i_boxed_3874_, v_stop_boxed_3875_, v_b_3873_);
lean_dec_ref(v_as_3870_);
return v_res_3876_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0(lean_object* v___x_3877_, lean_object* v_a_3878_, lean_object* v_a_3879_){
_start:
{
if (lean_obj_tag(v_a_3878_) == 0)
{
lean_object* v___x_3880_; 
v___x_3880_ = lean_array_to_list(v_a_3879_);
return v___x_3880_;
}
else
{
lean_object* v_head_3881_; lean_object* v_tail_3882_; uint8_t v___x_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; uint8_t v___x_3887_; 
v_head_3881_ = lean_ctor_get(v_a_3878_, 0);
lean_inc_n(v_head_3881_, 2);
v_tail_3882_ = lean_ctor_get(v_a_3878_, 1);
lean_inc(v_tail_3882_);
lean_dec_ref_known(v_a_3878_, 2);
v___x_3883_ = 0;
v___x_3884_ = l_Lean_Name_toString(v_head_3881_, v___x_3883_);
v___x_3885_ = lean_string_utf8_byte_size(v___x_3884_);
v___x_3886_ = lean_string_utf8_byte_size(v___x_3877_);
v___x_3887_ = lean_nat_dec_le(v___x_3886_, v___x_3885_);
if (v___x_3887_ == 0)
{
lean_dec_ref(v___x_3884_);
lean_dec(v_head_3881_);
v_a_3878_ = v_tail_3882_;
goto _start;
}
else
{
lean_object* v___x_3889_; uint8_t v___x_3890_; 
v___x_3889_ = lean_unsigned_to_nat(0u);
v___x_3890_ = lean_string_memcmp(v___x_3884_, v___x_3877_, v___x_3889_, v___x_3889_, v___x_3886_);
lean_dec_ref(v___x_3884_);
if (v___x_3890_ == 0)
{
lean_dec(v_head_3881_);
v_a_3878_ = v_tail_3882_;
goto _start;
}
else
{
lean_object* v___x_3892_; 
v___x_3892_ = lean_array_push(v_a_3879_, v_head_3881_);
v_a_3878_ = v_tail_3882_;
v_a_3879_ = v___x_3892_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0___boxed(lean_object* v___x_3894_, lean_object* v_a_3895_, lean_object* v_a_3896_){
_start:
{
lean_object* v_res_3897_; 
v_res_3897_ = lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0(v___x_3894_, v_a_3895_, v_a_3896_);
lean_dec_ref(v___x_3894_);
return v_res_3897_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0(lean_object* v___x_3898_, lean_object* v_a_3899_, lean_object* v_a_3900_){
_start:
{
if (lean_obj_tag(v_a_3899_) == 0)
{
lean_object* v___x_3901_; 
v___x_3901_ = lean_array_to_list(v_a_3900_);
return v___x_3901_;
}
else
{
lean_object* v_head_3902_; lean_object* v_tail_3903_; uint8_t v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; uint8_t v___x_3908_; 
v_head_3902_ = lean_ctor_get(v_a_3899_, 0);
lean_inc_n(v_head_3902_, 2);
v_tail_3903_ = lean_ctor_get(v_a_3899_, 1);
lean_inc(v_tail_3903_);
lean_dec_ref_known(v_a_3899_, 2);
v___x_3904_ = 0;
v___x_3905_ = l_Lean_Name_toString(v_head_3902_, v___x_3904_);
v___x_3906_ = lean_string_utf8_byte_size(v___x_3905_);
v___x_3907_ = lean_string_utf8_byte_size(v___x_3898_);
v___x_3908_ = lean_nat_dec_le(v___x_3907_, v___x_3906_);
if (v___x_3908_ == 0)
{
lean_object* v___x_3909_; 
lean_dec_ref(v___x_3905_);
lean_dec(v_head_3902_);
v___x_3909_ = lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0(v___x_3898_, v_tail_3903_, v_a_3900_);
return v___x_3909_;
}
else
{
lean_object* v___x_3910_; uint8_t v___x_3911_; 
v___x_3910_ = lean_unsigned_to_nat(0u);
v___x_3911_ = lean_string_memcmp(v___x_3905_, v___x_3898_, v___x_3910_, v___x_3910_, v___x_3907_);
lean_dec_ref(v___x_3905_);
if (v___x_3911_ == 0)
{
lean_object* v___x_3912_; 
lean_dec(v_head_3902_);
v___x_3912_ = lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0(v___x_3898_, v_tail_3903_, v_a_3900_);
return v___x_3912_;
}
else
{
lean_object* v___x_3913_; lean_object* v___x_3914_; 
v___x_3913_ = lean_array_push(v_a_3900_, v_head_3902_);
v___x_3914_ = lp_batteries_List_filterMapTR_go___at___00List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0_spec__0(v___x_3898_, v_tail_3903_, v___x_3913_);
return v___x_3914_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0___boxed(lean_object* v___x_3915_, lean_object* v_a_3916_, lean_object* v_a_3917_){
_start:
{
lean_object* v_res_3918_; 
v_res_3918_ = lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0(v___x_3915_, v_a_3916_, v_a_3917_);
lean_dec_ref(v___x_3915_);
return v_res_3918_;
}
}
static lean_object* _init_lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_3922_; lean_object* v___x_3923_; 
v___x_3922_ = ((lean_object*)(lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__1));
v___x_3923_ = l_Lean_Name_eraseMacroScopes(v___x_3922_);
return v___x_3923_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg(lean_object* v___x_3927_, uint8_t v___x_3928_, lean_object* v_x_3929_, lean_object* v_x_3930_, lean_object* v___y_3931_){
_start:
{
if (lean_obj_tag(v_x_3929_) == 0)
{
lean_object* v___x_3933_; lean_object* v___x_3934_; 
lean_dec_ref(v___x_3927_);
v___x_3933_ = l_List_reverse___redArg(v_x_3930_);
v___x_3934_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3934_, 0, v___x_3933_);
return v___x_3934_;
}
else
{
lean_object* v_head_3935_; lean_object* v_tail_3936_; lean_object* v___x_3938_; uint8_t v_isShared_3939_; uint8_t v_isSharedCheck_3985_; 
v_head_3935_ = lean_ctor_get(v_x_3929_, 0);
v_tail_3936_ = lean_ctor_get(v_x_3929_, 1);
v_isSharedCheck_3985_ = !lean_is_exclusive(v_x_3929_);
if (v_isSharedCheck_3985_ == 0)
{
v___x_3938_ = v_x_3929_;
v_isShared_3939_ = v_isSharedCheck_3985_;
goto v_resetjp_3937_;
}
else
{
lean_inc(v_tail_3936_);
lean_inc(v_head_3935_);
lean_dec(v_x_3929_);
v___x_3938_ = lean_box(0);
v_isShared_3939_ = v_isSharedCheck_3985_;
goto v_resetjp_3937_;
}
v_resetjp_3937_:
{
lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; lean_object* v___x_3943_; lean_object* v___x_3944_; lean_object* v___x_3945_; lean_object* v___x_3946_; 
v___x_3940_ = lean_box(0);
lean_inc(v_head_3935_);
v___x_3941_ = lp_batteries_Batteries_Util_LibraryNote_encodeNameForExport(v_head_3935_);
v___x_3942_ = lean_obj_once(&lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__2, &lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__2_once, _init_lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__2);
v___x_3943_ = l_Lean_Name_append(v___x_3942_, v___x_3941_);
v___x_3944_ = l_Lean_Options_empty;
v___x_3945_ = lean_box(0);
lean_inc_ref(v___x_3927_);
v___x_3946_ = l_Lean_findDocString_x3f(v___x_3927_, v___x_3943_, v___x_3928_, v___x_3944_, v___x_3940_, v___x_3945_);
if (lean_obj_tag(v___x_3946_) == 0)
{
lean_object* v_a_3947_; 
v_a_3947_ = lean_ctor_get(v___x_3946_, 0);
lean_inc(v_a_3947_);
lean_dec_ref_known(v___x_3946_, 1);
if (lean_obj_tag(v_a_3947_) == 1)
{
lean_object* v_val_3948_; lean_object* v___x_3949_; lean_object* v___x_3950_; lean_object* v___x_3951_; lean_object* v___x_3952_; lean_object* v_str_3953_; lean_object* v_startInclusive_3954_; lean_object* v_endExclusive_3955_; lean_object* v___x_3956_; lean_object* v___x_3957_; lean_object* v___x_3958_; lean_object* v___x_3959_; lean_object* v___x_3960_; lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; lean_object* v___x_3964_; lean_object* v___x_3965_; lean_object* v___x_3966_; lean_object* v___x_3968_; 
v_val_3948_ = lean_ctor_get(v_a_3947_, 0);
lean_inc(v_val_3948_);
lean_dec_ref_known(v_a_3947_, 1);
v___x_3949_ = lean_unsigned_to_nat(0u);
v___x_3950_ = lean_string_utf8_byte_size(v_val_3948_);
v___x_3951_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3951_, 0, v_val_3948_);
lean_ctor_set(v___x_3951_, 1, v___x_3949_);
lean_ctor_set(v___x_3951_, 2, v___x_3950_);
v___x_3952_ = l_String_Slice_trimAscii(v___x_3951_);
v_str_3953_ = lean_ctor_get(v___x_3952_, 0);
lean_inc_ref(v_str_3953_);
v_startInclusive_3954_ = lean_ctor_get(v___x_3952_, 1);
lean_inc(v_startInclusive_3954_);
v_endExclusive_3955_ = lean_ctor_get(v___x_3952_, 2);
lean_inc(v_endExclusive_3955_);
lean_dec_ref(v___x_3952_);
v___x_3956_ = ((lean_object*)(lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__3));
v___x_3957_ = l_Lean_Name_toString(v_head_3935_, v___x_3928_);
v___x_3958_ = lean_string_append(v___x_3956_, v___x_3957_);
lean_dec_ref(v___x_3957_);
v___x_3959_ = ((lean_object*)(lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpAttr_spec__1___redArg___closed__2));
v___x_3960_ = lean_string_append(v___x_3958_, v___x_3959_);
v___x_3961_ = ((lean_object*)(lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__4));
v___x_3962_ = lean_string_append(v___x_3960_, v___x_3961_);
v___x_3963_ = lean_string_utf8_extract_fast(v_str_3953_, v_startInclusive_3954_, v_endExclusive_3955_);
lean_dec(v_endExclusive_3955_);
lean_dec(v_startInclusive_3954_);
lean_dec_ref(v_str_3953_);
v___x_3964_ = lean_string_append(v___x_3962_, v___x_3963_);
lean_dec_ref(v___x_3963_);
v___x_3965_ = ((lean_object*)(lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___closed__5));
v___x_3966_ = lean_string_append(v___x_3964_, v___x_3965_);
if (v_isShared_3939_ == 0)
{
lean_ctor_set(v___x_3938_, 1, v_x_3930_);
lean_ctor_set(v___x_3938_, 0, v___x_3966_);
v___x_3968_ = v___x_3938_;
goto v_reusejp_3967_;
}
else
{
lean_object* v_reuseFailAlloc_3970_; 
v_reuseFailAlloc_3970_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3970_, 0, v___x_3966_);
lean_ctor_set(v_reuseFailAlloc_3970_, 1, v_x_3930_);
v___x_3968_ = v_reuseFailAlloc_3970_;
goto v_reusejp_3967_;
}
v_reusejp_3967_:
{
v_x_3929_ = v_tail_3936_;
v_x_3930_ = v___x_3968_;
goto _start;
}
}
else
{
lean_dec(v_a_3947_);
lean_del_object(v___x_3938_);
lean_dec(v_head_3935_);
v_x_3929_ = v_tail_3936_;
goto _start;
}
}
else
{
lean_object* v_a_3972_; lean_object* v___x_3974_; uint8_t v_isShared_3975_; uint8_t v_isSharedCheck_3984_; 
lean_del_object(v___x_3938_);
lean_dec(v_tail_3936_);
lean_dec(v_head_3935_);
lean_dec(v_x_3930_);
lean_dec_ref(v___x_3927_);
v_a_3972_ = lean_ctor_get(v___x_3946_, 0);
v_isSharedCheck_3984_ = !lean_is_exclusive(v___x_3946_);
if (v_isSharedCheck_3984_ == 0)
{
v___x_3974_ = v___x_3946_;
v_isShared_3975_ = v_isSharedCheck_3984_;
goto v_resetjp_3973_;
}
else
{
lean_inc(v_a_3972_);
lean_dec(v___x_3946_);
v___x_3974_ = lean_box(0);
v_isShared_3975_ = v_isSharedCheck_3984_;
goto v_resetjp_3973_;
}
v_resetjp_3973_:
{
lean_object* v_ref_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v___x_3980_; lean_object* v___x_3982_; 
v_ref_3976_ = lean_ctor_get(v___y_3931_, 7);
v___x_3977_ = lean_io_error_to_string(v_a_3972_);
v___x_3978_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3978_, 0, v___x_3977_);
v___x_3979_ = l_Lean_MessageData_ofFormat(v___x_3978_);
lean_inc(v_ref_3976_);
v___x_3980_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3980_, 0, v_ref_3976_);
lean_ctor_set(v___x_3980_, 1, v___x_3979_);
if (v_isShared_3975_ == 0)
{
lean_ctor_set(v___x_3974_, 0, v___x_3980_);
v___x_3982_ = v___x_3974_;
goto v_reusejp_3981_;
}
else
{
lean_object* v_reuseFailAlloc_3983_; 
v_reuseFailAlloc_3983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3983_, 0, v___x_3980_);
v___x_3982_ = v_reuseFailAlloc_3983_;
goto v_reusejp_3981_;
}
v_reusejp_3981_:
{
return v___x_3982_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg___boxed(lean_object* v___x_3986_, lean_object* v___x_3987_, lean_object* v_x_3988_, lean_object* v_x_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_){
_start:
{
uint8_t v___x_2636__boxed_3992_; lean_object* v_res_3993_; 
v___x_2636__boxed_3992_ = lean_unbox(v___x_3987_);
v_res_3993_ = lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg(v___x_3986_, v___x_2636__boxed_3992_, v_x_3988_, v_x_3989_, v___y_3990_);
lean_dec_ref(v___y_3990_);
return v_res_3993_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0(void){
_start:
{
lean_object* v___x_3994_; 
v___x_3994_ = l_Array_instInhabited(lean_box(0));
return v___x_3994_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__1(void){
_start:
{
lean_object* v___x_3995_; lean_object* v___x_3996_; lean_object* v___x_3997_; 
v___x_3995_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0);
v___x_3996_ = lean_box(0);
v___x_3997_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3997_, 0, v___x_3996_);
lean_ctor_set(v___x_3997_, 1, v___x_3995_);
return v___x_3997_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__2(void){
_start:
{
lean_object* v___x_3998_; lean_object* v___x_3999_; 
v___x_3998_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__1, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__1_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__1);
v___x_3999_ = l_Lean_instInhabitedPersistentEnvExtensionState___redArg(v___x_3998_);
return v___x_3999_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__9(void){
_start:
{
lean_object* v___x_4009_; lean_object* v___x_4010_; 
v___x_4009_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__8));
v___x_4010_ = l_Lean_MessageData_ofFormat(v___x_4009_);
return v___x_4010_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1(lean_object* v_x_4013_, lean_object* v_a_4014_, lean_object* v_a_4015_){
_start:
{
lean_object* v___x_4017_; uint8_t v___x_4018_; 
v___x_4017_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Note_______00__closed__1));
lean_inc(v_x_4013_);
v___x_4018_ = l_Lean_Syntax_isOfKind(v_x_4013_, v___x_4017_);
if (v___x_4018_ == 0)
{
lean_object* v___x_4019_; 
lean_dec(v_x_4013_);
v___x_4019_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Option________1_spec__0___redArg();
return v___x_4019_;
}
else
{
lean_object* v___x_4020_; lean_object* v_env_4021_; lean_object* v___x_4022_; lean_object* v_toEnvExtension_4023_; lean_object* v_asyncMode_4024_; lean_object* v___x_4025_; lean_object* v___x_4026_; lean_object* v___x_4027_; lean_object* v___x_4028_; lean_object* v___x_4029_; lean_object* v___x_4030_; lean_object* v_importedEntries_4031_; lean_object* v___f_4032_; lean_object* v___x_4033_; lean_object* v_name_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; lean_object* v___y_4038_; lean_object* v___x_4063_; lean_object* v___x_4064_; lean_object* v___x_4065_; uint8_t v___x_4066_; 
v___x_4020_ = lean_st_ref_get(v_a_4015_);
v_env_4021_ = lean_ctor_get(v___x_4020_, 0);
lean_inc_ref_n(v_env_4021_, 3);
lean_dec(v___x_4020_);
v___x_4022_ = lp_batteries_Batteries_Util_LibraryNote_libraryNoteExt;
v_toEnvExtension_4023_ = lean_ctor_get(v___x_4022_, 0);
v_asyncMode_4024_ = lean_ctor_get(v_toEnvExtension_4023_, 2);
v___x_4025_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__0);
v___x_4026_ = l_Lean_SimplePersistentEnvExtension_getEntries___redArg(v___x_4025_, v___x_4022_, v_env_4021_, v_asyncMode_4024_);
v___x_4027_ = lean_box(0);
v___x_4028_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__2, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__2_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__2);
v___x_4029_ = lean_box(0);
v___x_4030_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_4028_, v_toEnvExtension_4023_, v_env_4021_, v_asyncMode_4024_, v___x_4029_);
v_importedEntries_4031_ = lean_ctor_get(v___x_4030_, 0);
lean_inc_ref(v_importedEntries_4031_);
lean_dec(v___x_4030_);
v___f_4032_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__4));
v___x_4033_ = lean_unsigned_to_nat(5u);
v_name_4034_ = l_Lean_Syntax_getArg(v_x_4013_, v___x_4033_);
lean_dec(v_x_4013_);
v___x_4035_ = l_List_reverse___redArg(v___x_4026_);
v___x_4036_ = l_Lean_TSyntax_getString(v_name_4034_);
lean_dec(v_name_4034_);
v___x_4063_ = lean_unsigned_to_nat(0u);
v___x_4064_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__10));
v___x_4065_ = lean_array_get_size(v_importedEntries_4031_);
v___x_4066_ = lean_nat_dec_lt(v___x_4063_, v___x_4065_);
if (v___x_4066_ == 0)
{
lean_dec_ref(v_importedEntries_4031_);
v___y_4038_ = v___x_4064_;
goto v___jp_4037_;
}
else
{
uint8_t v___x_4067_; 
v___x_4067_ = lean_nat_dec_le(v___x_4065_, v___x_4065_);
if (v___x_4067_ == 0)
{
if (v___x_4066_ == 0)
{
lean_dec_ref(v_importedEntries_4031_);
v___y_4038_ = v___x_4064_;
goto v___jp_4037_;
}
else
{
size_t v___x_4068_; size_t v___x_4069_; lean_object* v___x_4070_; 
v___x_4068_ = ((size_t)0ULL);
v___x_4069_ = lean_usize_of_nat(v___x_4065_);
v___x_4070_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3(v_importedEntries_4031_, v___x_4068_, v___x_4069_, v___x_4064_);
lean_dec_ref(v_importedEntries_4031_);
v___y_4038_ = v___x_4070_;
goto v___jp_4037_;
}
}
else
{
size_t v___x_4071_; size_t v___x_4072_; lean_object* v___x_4073_; 
v___x_4071_ = ((size_t)0ULL);
v___x_4072_ = lean_usize_of_nat(v___x_4065_);
v___x_4073_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__3(v_importedEntries_4031_, v___x_4071_, v___x_4072_, v___x_4064_);
lean_dec_ref(v_importedEntries_4031_);
v___y_4038_ = v___x_4073_;
goto v___jp_4037_;
}
}
v___jp_4037_:
{
lean_object* v___x_4039_; lean_object* v___x_4040_; lean_object* v___x_4041_; lean_object* v___x_4042_; lean_object* v___x_4043_; lean_object* v___x_4044_; uint8_t v___x_4045_; 
v___x_4039_ = lean_array_to_list(v___y_4038_);
v___x_4040_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__5));
v___x_4041_ = lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0(v___x_4036_, v___x_4039_, v___x_4040_);
v___x_4042_ = lp_batteries_List_filterMapTR_go___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__0(v___x_4036_, v___x_4035_, v___x_4040_);
lean_dec_ref(v___x_4036_);
v___x_4043_ = l_List_appendTR___redArg(v___x_4041_, v___x_4042_);
v___x_4044_ = l_List_MergeSort_Internal_mergeSortTR_u2082___redArg(v___x_4043_, v___f_4032_);
v___x_4045_ = l_List_isEmpty___redArg(v___x_4044_);
if (v___x_4045_ == 0)
{
lean_object* v___x_4046_; 
v___x_4046_ = lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg(v_env_4021_, v___x_4018_, v___x_4044_, v___x_4027_, v_a_4014_);
if (lean_obj_tag(v___x_4046_) == 0)
{
lean_object* v_a_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; lean_object* v___x_4051_; lean_object* v___x_4052_; 
v_a_4047_ = lean_ctor_get(v___x_4046_, 0);
lean_inc(v_a_4047_);
lean_dec_ref_known(v___x_4046_, 1);
v___x_4048_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__6));
v___x_4049_ = l_String_intercalate(v___x_4048_, v_a_4047_);
v___x_4050_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4050_, 0, v___x_4049_);
v___x_4051_ = l_Lean_MessageData_ofFormat(v___x_4050_);
v___x_4052_ = lp_batteries_Lean_logInfo___at___00__private_Batteries_Tactic_HelpCmd_0__Batteries_Tactic_elabHelpOption_spec__0(v___x_4051_, v_a_4014_, v_a_4015_);
return v___x_4052_;
}
else
{
lean_object* v_a_4053_; lean_object* v___x_4055_; uint8_t v_isShared_4056_; uint8_t v_isSharedCheck_4060_; 
v_a_4053_ = lean_ctor_get(v___x_4046_, 0);
v_isSharedCheck_4060_ = !lean_is_exclusive(v___x_4046_);
if (v_isSharedCheck_4060_ == 0)
{
v___x_4055_ = v___x_4046_;
v_isShared_4056_ = v_isSharedCheck_4060_;
goto v_resetjp_4054_;
}
else
{
lean_inc(v_a_4053_);
lean_dec(v___x_4046_);
v___x_4055_ = lean_box(0);
v_isShared_4056_ = v_isSharedCheck_4060_;
goto v_resetjp_4054_;
}
v_resetjp_4054_:
{
lean_object* v___x_4058_; 
if (v_isShared_4056_ == 0)
{
v___x_4058_ = v___x_4055_;
goto v_reusejp_4057_;
}
else
{
lean_object* v_reuseFailAlloc_4059_; 
v_reuseFailAlloc_4059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4059_, 0, v_a_4053_);
v___x_4058_ = v_reuseFailAlloc_4059_;
goto v_reusejp_4057_;
}
v_reusejp_4057_:
{
return v___x_4058_;
}
}
}
}
else
{
lean_object* v___x_4061_; lean_object* v___x_4062_; 
lean_dec(v___x_4044_);
lean_dec_ref(v_env_4021_);
v___x_4061_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__9, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__9_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___closed__9);
v___x_4062_ = lp_batteries_Lean_logError___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__2(v___x_4061_, v_a_4014_, v_a_4015_);
return v___x_4062_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1___boxed(lean_object* v_x_4074_, lean_object* v_a_4075_, lean_object* v_a_4076_, lean_object* v_a_4077_){
_start:
{
lean_object* v_res_4078_; 
v_res_4078_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1(v_x_4074_, v_a_4075_, v_a_4076_);
lean_dec(v_a_4076_);
lean_dec_ref(v_a_4075_);
return v_res_4078_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1(lean_object* v___x_4079_, uint8_t v___x_4080_, lean_object* v_x_4081_, lean_object* v_x_4082_, lean_object* v___y_4083_, lean_object* v___y_4084_){
_start:
{
lean_object* v___x_4086_; 
v___x_4086_ = lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___redArg(v___x_4079_, v___x_4080_, v_x_4081_, v_x_4082_, v___y_4083_);
return v___x_4086_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1___boxed(lean_object* v___x_4087_, lean_object* v___x_4088_, lean_object* v_x_4089_, lean_object* v_x_4090_, lean_object* v___y_4091_, lean_object* v___y_4092_, lean_object* v___y_4093_){
_start:
{
uint8_t v___x_2934__boxed_4094_; lean_object* v_res_4095_; 
v___x_2934__boxed_4094_ = lean_unbox(v___x_4088_);
v_res_4095_ = lp_batteries_List_filterMapM_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______elabRules__Batteries__Tactic__command_x23help__Note________1_spec__1(v___x_4087_, v___x_2934__boxed_4094_, v_x_4089_, v_x_4090_, v___y_4091_, v___y_4092_);
lean_dec(v___y_4092_);
lean_dec_ref(v___y_4091_);
return v_res_4095_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5(void){
_start:
{
lean_object* v___x_4132_; 
v___x_4132_ = l_Array_mkArray0(lean_box(0));
return v___x_4132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1(lean_object* v_x_4133_, lean_object* v_a_4134_, lean_object* v_a_4135_){
_start:
{
lean_object* v___y_4137_; lean_object* v___y_4138_; lean_object* v___y_4139_; lean_object* v___y_4140_; lean_object* v___y_4141_; lean_object* v___y_4142_; lean_object* v___y_4143_; lean_object* v___y_4144_; lean_object* v___y_4145_; lean_object* v___y_4146_; lean_object* v___x_4151_; uint8_t v___x_4152_; 
v___x_4151_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Term_x2b_________00__closed__1));
lean_inc(v_x_4133_);
v___x_4152_ = l_Lean_Syntax_isOfKind(v_x_4133_, v___x_4151_);
if (v___x_4152_ == 0)
{
lean_object* v___x_4153_; lean_object* v___x_4154_; 
lean_dec(v_x_4133_);
v___x_4153_ = lean_box(1);
v___x_4154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4154_, 0, v___x_4153_);
lean_ctor_set(v___x_4154_, 1, v_a_4135_);
return v___x_4154_;
}
else
{
lean_object* v___x_4155_; lean_object* v_tk_4156_; lean_object* v___y_4158_; lean_object* v___y_4159_; lean_object* v___y_4160_; lean_object* v___y_4161_; uint8_t v___y_4162_; lean_object* v___y_4163_; lean_object* v___y_4164_; lean_object* v___y_4165_; lean_object* v___y_4166_; lean_object* v___y_4167_; lean_object* v___y_4177_; lean_object* v___y_4178_; lean_object* v___y_4179_; lean_object* v___y_4180_; lean_object* v_more_4198_; lean_object* v___y_4199_; lean_object* v___y_4200_; lean_object* v___x_4213_; lean_object* v___x_4214_; uint8_t v___x_4215_; 
v___x_4155_ = lean_unsigned_to_nat(1u);
v_tk_4156_ = l_Lean_Syntax_getArg(v_x_4133_, v___x_4155_);
v___x_4213_ = lean_unsigned_to_nat(2u);
v___x_4214_ = l_Lean_Syntax_getArg(v_x_4133_, v___x_4213_);
v___x_4215_ = l_Lean_Syntax_isNone(v___x_4214_);
if (v___x_4215_ == 0)
{
uint8_t v___x_4216_; 
lean_inc(v___x_4214_);
v___x_4216_ = l_Lean_Syntax_matchesNull(v___x_4214_, v___x_4155_);
if (v___x_4216_ == 0)
{
lean_object* v___x_4217_; lean_object* v___x_4218_; 
lean_dec(v___x_4214_);
lean_dec(v_tk_4156_);
lean_dec(v_x_4133_);
v___x_4217_ = lean_box(1);
v___x_4218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4218_, 0, v___x_4217_);
lean_ctor_set(v___x_4218_, 1, v_a_4135_);
return v___x_4218_;
}
else
{
lean_object* v___x_4219_; lean_object* v_more_4220_; lean_object* v___x_4221_; 
v___x_4219_ = lean_unsigned_to_nat(0u);
v_more_4220_ = l_Lean_Syntax_getArg(v___x_4214_, v___x_4219_);
lean_dec(v___x_4214_);
v___x_4221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4221_, 0, v_more_4220_);
v_more_4198_ = v___x_4221_;
v___y_4199_ = v_a_4134_;
v___y_4200_ = v_a_4135_;
goto v___jp_4197_;
}
}
else
{
lean_object* v___x_4222_; 
lean_dec(v___x_4214_);
v___x_4222_ = lean_box(0);
v_more_4198_ = v___x_4222_;
v___y_4199_ = v_a_4134_;
v___y_4200_ = v_a_4135_;
goto v___jp_4197_;
}
v___jp_4157_:
{
lean_object* v___x_4168_; lean_object* v___x_4169_; lean_object* v___x_4170_; lean_object* v___x_4171_; 
lean_inc_ref(v___y_4164_);
v___x_4168_ = l_Array_append___redArg(v___y_4164_, v___y_4167_);
lean_dec_ref(v___y_4167_);
lean_inc(v___y_4160_);
lean_inc(v___y_4166_);
v___x_4169_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4169_, 0, v___y_4166_);
lean_ctor_set(v___x_4169_, 1, v___y_4160_);
lean_ctor_set(v___x_4169_, 2, v___x_4168_);
v___x_4170_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__0));
v___x_4171_ = l_Lean_mkIdentFrom(v_tk_4156_, v___x_4170_, v___y_4162_);
lean_dec(v_tk_4156_);
if (lean_obj_tag(v___y_4163_) == 0)
{
lean_object* v___x_4172_; 
v___x_4172_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4137_ = v___x_4169_;
v___y_4138_ = v___y_4158_;
v___y_4139_ = v___y_4159_;
v___y_4140_ = v___y_4160_;
v___y_4141_ = v___y_4161_;
v___y_4142_ = v___x_4171_;
v___y_4143_ = v___y_4164_;
v___y_4144_ = v___y_4165_;
v___y_4145_ = v___y_4166_;
v___y_4146_ = v___x_4172_;
goto v___jp_4136_;
}
else
{
lean_object* v_val_4173_; lean_object* v___x_4174_; lean_object* v___x_4175_; 
v_val_4173_ = lean_ctor_get(v___y_4163_, 0);
lean_inc(v_val_4173_);
lean_dec_ref_known(v___y_4163_, 1);
v___x_4174_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___x_4175_ = lean_array_push(v___x_4174_, v_val_4173_);
v___y_4137_ = v___x_4169_;
v___y_4138_ = v___y_4158_;
v___y_4139_ = v___y_4159_;
v___y_4140_ = v___y_4160_;
v___y_4141_ = v___y_4161_;
v___y_4142_ = v___x_4171_;
v___y_4143_ = v___y_4164_;
v___y_4144_ = v___y_4165_;
v___y_4145_ = v___y_4166_;
v___y_4146_ = v___x_4175_;
goto v___jp_4136_;
}
}
v___jp_4176_:
{
lean_object* v_ref_4181_; uint8_t v___x_4182_; lean_object* v___x_4183_; lean_object* v___x_4184_; lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; 
v_ref_4181_ = lean_ctor_get(v___y_4178_, 5);
v___x_4182_ = 0;
v___x_4183_ = l_Lean_SourceInfo_fromRef(v_ref_4181_, v___x_4182_);
v___x_4184_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1));
v___x_4185_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2));
lean_inc_n(v___x_4183_, 2);
v___x_4186_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4186_, 0, v___x_4183_);
lean_ctor_set(v___x_4186_, 1, v___x_4185_);
v___x_4187_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2));
v___x_4188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4188_, 0, v___x_4183_);
lean_ctor_set(v___x_4188_, 1, v___x_4187_);
v___x_4189_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4));
v___x_4190_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5);
if (lean_obj_tag(v___y_4179_) == 1)
{
lean_object* v_val_4191_; lean_object* v___x_4192_; lean_object* v___x_4193_; lean_object* v___x_4194_; lean_object* v___x_4195_; 
v_val_4191_ = lean_ctor_get(v___y_4179_, 0);
lean_inc(v_val_4191_);
lean_dec_ref_known(v___y_4179_, 1);
v___x_4192_ = l_Lean_SourceInfo_fromRef(v_val_4191_, v___x_4152_);
lean_dec(v_val_4191_);
v___x_4193_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5));
v___x_4194_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4194_, 0, v___x_4192_);
lean_ctor_set(v___x_4194_, 1, v___x_4193_);
v___x_4195_ = l_Array_mkArray1___redArg(v___x_4194_);
v___y_4158_ = v___x_4186_;
v___y_4159_ = v___y_4177_;
v___y_4160_ = v___x_4189_;
v___y_4161_ = v___x_4188_;
v___y_4162_ = v___x_4182_;
v___y_4163_ = v___y_4180_;
v___y_4164_ = v___x_4190_;
v___y_4165_ = v___x_4184_;
v___y_4166_ = v___x_4183_;
v___y_4167_ = v___x_4195_;
goto v___jp_4157_;
}
else
{
lean_object* v___x_4196_; 
lean_dec(v___y_4179_);
v___x_4196_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4158_ = v___x_4186_;
v___y_4159_ = v___y_4177_;
v___y_4160_ = v___x_4189_;
v___y_4161_ = v___x_4188_;
v___y_4162_ = v___x_4182_;
v___y_4163_ = v___y_4180_;
v___y_4164_ = v___x_4190_;
v___y_4165_ = v___x_4184_;
v___y_4166_ = v___x_4183_;
v___y_4167_ = v___x_4196_;
goto v___jp_4157_;
}
}
v___jp_4197_:
{
lean_object* v___x_4201_; lean_object* v___x_4202_; lean_object* v___x_4203_; 
v___x_4201_ = lean_unsigned_to_nat(3u);
v___x_4202_ = l_Lean_Syntax_getArg(v_x_4133_, v___x_4201_);
lean_dec(v_x_4133_);
v___x_4203_ = l_Lean_Syntax_getOptional_x3f(v___x_4202_);
lean_dec(v___x_4202_);
if (lean_obj_tag(v___x_4203_) == 0)
{
lean_object* v___x_4204_; 
v___x_4204_ = lean_box(0);
v___y_4177_ = v___y_4200_;
v___y_4178_ = v___y_4199_;
v___y_4179_ = v_more_4198_;
v___y_4180_ = v___x_4204_;
goto v___jp_4176_;
}
else
{
lean_object* v_val_4205_; lean_object* v___x_4207_; uint8_t v_isShared_4208_; uint8_t v_isSharedCheck_4212_; 
v_val_4205_ = lean_ctor_get(v___x_4203_, 0);
v_isSharedCheck_4212_ = !lean_is_exclusive(v___x_4203_);
if (v_isSharedCheck_4212_ == 0)
{
v___x_4207_ = v___x_4203_;
v_isShared_4208_ = v_isSharedCheck_4212_;
goto v_resetjp_4206_;
}
else
{
lean_inc(v_val_4205_);
lean_dec(v___x_4203_);
v___x_4207_ = lean_box(0);
v_isShared_4208_ = v_isSharedCheck_4212_;
goto v_resetjp_4206_;
}
v_resetjp_4206_:
{
lean_object* v___x_4210_; 
if (v_isShared_4208_ == 0)
{
v___x_4210_ = v___x_4207_;
goto v_reusejp_4209_;
}
else
{
lean_object* v_reuseFailAlloc_4211_; 
v_reuseFailAlloc_4211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4211_, 0, v_val_4205_);
v___x_4210_ = v_reuseFailAlloc_4211_;
goto v_reusejp_4209_;
}
v_reusejp_4209_:
{
v___y_4177_ = v___y_4200_;
v___y_4178_ = v___y_4199_;
v___y_4179_ = v_more_4198_;
v___y_4180_ = v___x_4210_;
goto v___jp_4176_;
}
}
}
}
}
v___jp_4136_:
{
lean_object* v___x_4147_; lean_object* v___x_4148_; lean_object* v___x_4149_; lean_object* v___x_4150_; 
lean_inc_ref(v___y_4143_);
v___x_4147_ = l_Array_append___redArg(v___y_4143_, v___y_4146_);
lean_dec_ref(v___y_4146_);
lean_inc(v___y_4140_);
lean_inc(v___y_4145_);
v___x_4148_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4148_, 0, v___y_4145_);
lean_ctor_set(v___x_4148_, 1, v___y_4140_);
lean_ctor_set(v___x_4148_, 2, v___x_4147_);
lean_inc(v___y_4144_);
v___x_4149_ = l_Lean_Syntax_node5(v___y_4145_, v___y_4144_, v___y_4138_, v___y_4141_, v___y_4137_, v___y_4142_, v___x_4148_);
v___x_4150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4150_, 0, v___x_4149_);
lean_ctor_set(v___x_4150_, 1, v___y_4139_);
return v___x_4150_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___boxed(lean_object* v_x_4223_, lean_object* v_a_4224_, lean_object* v_a_4225_){
_start:
{
lean_object* v_res_4226_; 
v_res_4226_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1(v_x_4223_, v_a_4224_, v_a_4225_);
lean_dec_ref(v_a_4224_);
return v_res_4226_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1(lean_object* v_x_4257_, lean_object* v_a_4258_, lean_object* v_a_4259_){
_start:
{
lean_object* v___y_4261_; lean_object* v___y_4262_; lean_object* v___y_4263_; lean_object* v___y_4264_; lean_object* v___y_4265_; lean_object* v___y_4266_; lean_object* v___y_4267_; lean_object* v___y_4268_; lean_object* v___y_4269_; lean_object* v___y_4270_; lean_object* v___x_4275_; uint8_t v___x_4276_; 
v___x_4275_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Tactic_x2b_________00__closed__1));
lean_inc(v_x_4257_);
v___x_4276_ = l_Lean_Syntax_isOfKind(v_x_4257_, v___x_4275_);
if (v___x_4276_ == 0)
{
lean_object* v___x_4277_; lean_object* v___x_4278_; 
lean_dec(v_x_4257_);
v___x_4277_ = lean_box(1);
v___x_4278_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4278_, 0, v___x_4277_);
lean_ctor_set(v___x_4278_, 1, v_a_4259_);
return v___x_4278_;
}
else
{
lean_object* v___x_4279_; lean_object* v_tk_4280_; lean_object* v___y_4282_; uint8_t v___y_4283_; lean_object* v___y_4284_; lean_object* v___y_4285_; lean_object* v___y_4286_; lean_object* v___y_4287_; lean_object* v___y_4288_; lean_object* v___y_4289_; lean_object* v___y_4290_; lean_object* v___y_4291_; lean_object* v___y_4301_; lean_object* v___y_4302_; lean_object* v___y_4303_; lean_object* v___y_4304_; lean_object* v_more_4322_; lean_object* v___y_4323_; lean_object* v___y_4324_; lean_object* v___x_4337_; lean_object* v___x_4338_; uint8_t v___x_4339_; 
v___x_4279_ = lean_unsigned_to_nat(1u);
v_tk_4280_ = l_Lean_Syntax_getArg(v_x_4257_, v___x_4279_);
v___x_4337_ = lean_unsigned_to_nat(2u);
v___x_4338_ = l_Lean_Syntax_getArg(v_x_4257_, v___x_4337_);
v___x_4339_ = l_Lean_Syntax_isNone(v___x_4338_);
if (v___x_4339_ == 0)
{
uint8_t v___x_4340_; 
lean_inc(v___x_4338_);
v___x_4340_ = l_Lean_Syntax_matchesNull(v___x_4338_, v___x_4279_);
if (v___x_4340_ == 0)
{
lean_object* v___x_4341_; lean_object* v___x_4342_; 
lean_dec(v___x_4338_);
lean_dec(v_tk_4280_);
lean_dec(v_x_4257_);
v___x_4341_ = lean_box(1);
v___x_4342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4342_, 0, v___x_4341_);
lean_ctor_set(v___x_4342_, 1, v_a_4259_);
return v___x_4342_;
}
else
{
lean_object* v___x_4343_; lean_object* v_more_4344_; lean_object* v___x_4345_; 
v___x_4343_ = lean_unsigned_to_nat(0u);
v_more_4344_ = l_Lean_Syntax_getArg(v___x_4338_, v___x_4343_);
lean_dec(v___x_4338_);
v___x_4345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4345_, 0, v_more_4344_);
v_more_4322_ = v___x_4345_;
v___y_4323_ = v_a_4258_;
v___y_4324_ = v_a_4259_;
goto v___jp_4321_;
}
}
else
{
lean_object* v___x_4346_; 
lean_dec(v___x_4338_);
v___x_4346_ = lean_box(0);
v_more_4322_ = v___x_4346_;
v___y_4323_ = v_a_4258_;
v___y_4324_ = v_a_4259_;
goto v___jp_4321_;
}
v___jp_4281_:
{
lean_object* v___x_4292_; lean_object* v___x_4293_; lean_object* v___x_4294_; lean_object* v___x_4295_; 
lean_inc_ref(v___y_4288_);
v___x_4292_ = l_Array_append___redArg(v___y_4288_, v___y_4291_);
lean_dec_ref(v___y_4291_);
lean_inc(v___y_4285_);
lean_inc(v___y_4290_);
v___x_4293_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4293_, 0, v___y_4290_);
lean_ctor_set(v___x_4293_, 1, v___y_4285_);
lean_ctor_set(v___x_4293_, 2, v___x_4292_);
v___x_4294_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1___closed__0));
v___x_4295_ = l_Lean_mkIdentFrom(v_tk_4280_, v___x_4294_, v___y_4283_);
lean_dec(v_tk_4280_);
if (lean_obj_tag(v___y_4282_) == 0)
{
lean_object* v___x_4296_; 
v___x_4296_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4261_ = v___x_4293_;
v___y_4262_ = v___y_4284_;
v___y_4263_ = v___y_4285_;
v___y_4264_ = v___x_4295_;
v___y_4265_ = v___y_4286_;
v___y_4266_ = v___y_4287_;
v___y_4267_ = v___y_4288_;
v___y_4268_ = v___y_4289_;
v___y_4269_ = v___y_4290_;
v___y_4270_ = v___x_4296_;
goto v___jp_4260_;
}
else
{
lean_object* v_val_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; 
v_val_4297_ = lean_ctor_get(v___y_4282_, 0);
lean_inc(v_val_4297_);
lean_dec_ref_known(v___y_4282_, 1);
v___x_4298_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___x_4299_ = lean_array_push(v___x_4298_, v_val_4297_);
v___y_4261_ = v___x_4293_;
v___y_4262_ = v___y_4284_;
v___y_4263_ = v___y_4285_;
v___y_4264_ = v___x_4295_;
v___y_4265_ = v___y_4286_;
v___y_4266_ = v___y_4287_;
v___y_4267_ = v___y_4288_;
v___y_4268_ = v___y_4289_;
v___y_4269_ = v___y_4290_;
v___y_4270_ = v___x_4299_;
goto v___jp_4260_;
}
}
v___jp_4300_:
{
lean_object* v_ref_4305_; uint8_t v___x_4306_; lean_object* v___x_4307_; lean_object* v___x_4308_; lean_object* v___x_4309_; lean_object* v___x_4310_; lean_object* v___x_4311_; lean_object* v___x_4312_; lean_object* v___x_4313_; lean_object* v___x_4314_; 
v_ref_4305_ = lean_ctor_get(v___y_4301_, 5);
v___x_4306_ = 0;
v___x_4307_ = l_Lean_SourceInfo_fromRef(v_ref_4305_, v___x_4306_);
v___x_4308_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1));
v___x_4309_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2));
lean_inc_n(v___x_4307_, 2);
v___x_4310_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4310_, 0, v___x_4307_);
lean_ctor_set(v___x_4310_, 1, v___x_4309_);
v___x_4311_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2));
v___x_4312_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4312_, 0, v___x_4307_);
lean_ctor_set(v___x_4312_, 1, v___x_4311_);
v___x_4313_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4));
v___x_4314_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5);
if (lean_obj_tag(v___y_4303_) == 1)
{
lean_object* v_val_4315_; lean_object* v___x_4316_; lean_object* v___x_4317_; lean_object* v___x_4318_; lean_object* v___x_4319_; 
v_val_4315_ = lean_ctor_get(v___y_4303_, 0);
lean_inc(v_val_4315_);
lean_dec_ref_known(v___y_4303_, 1);
v___x_4316_ = l_Lean_SourceInfo_fromRef(v_val_4315_, v___x_4276_);
lean_dec(v_val_4315_);
v___x_4317_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5));
v___x_4318_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4318_, 0, v___x_4316_);
lean_ctor_set(v___x_4318_, 1, v___x_4317_);
v___x_4319_ = l_Array_mkArray1___redArg(v___x_4318_);
v___y_4282_ = v___y_4304_;
v___y_4283_ = v___x_4306_;
v___y_4284_ = v___x_4312_;
v___y_4285_ = v___x_4313_;
v___y_4286_ = v___y_4302_;
v___y_4287_ = v___x_4308_;
v___y_4288_ = v___x_4314_;
v___y_4289_ = v___x_4310_;
v___y_4290_ = v___x_4307_;
v___y_4291_ = v___x_4319_;
goto v___jp_4281_;
}
else
{
lean_object* v___x_4320_; 
lean_dec(v___y_4303_);
v___x_4320_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4282_ = v___y_4304_;
v___y_4283_ = v___x_4306_;
v___y_4284_ = v___x_4312_;
v___y_4285_ = v___x_4313_;
v___y_4286_ = v___y_4302_;
v___y_4287_ = v___x_4308_;
v___y_4288_ = v___x_4314_;
v___y_4289_ = v___x_4310_;
v___y_4290_ = v___x_4307_;
v___y_4291_ = v___x_4320_;
goto v___jp_4281_;
}
}
v___jp_4321_:
{
lean_object* v___x_4325_; lean_object* v___x_4326_; lean_object* v___x_4327_; 
v___x_4325_ = lean_unsigned_to_nat(3u);
v___x_4326_ = l_Lean_Syntax_getArg(v_x_4257_, v___x_4325_);
lean_dec(v_x_4257_);
v___x_4327_ = l_Lean_Syntax_getOptional_x3f(v___x_4326_);
lean_dec(v___x_4326_);
if (lean_obj_tag(v___x_4327_) == 0)
{
lean_object* v___x_4328_; 
v___x_4328_ = lean_box(0);
v___y_4301_ = v___y_4323_;
v___y_4302_ = v___y_4324_;
v___y_4303_ = v_more_4322_;
v___y_4304_ = v___x_4328_;
goto v___jp_4300_;
}
else
{
lean_object* v_val_4329_; lean_object* v___x_4331_; uint8_t v_isShared_4332_; uint8_t v_isSharedCheck_4336_; 
v_val_4329_ = lean_ctor_get(v___x_4327_, 0);
v_isSharedCheck_4336_ = !lean_is_exclusive(v___x_4327_);
if (v_isSharedCheck_4336_ == 0)
{
v___x_4331_ = v___x_4327_;
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
else
{
lean_inc(v_val_4329_);
lean_dec(v___x_4327_);
v___x_4331_ = lean_box(0);
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
v_resetjp_4330_:
{
lean_object* v___x_4334_; 
if (v_isShared_4332_ == 0)
{
v___x_4334_ = v___x_4331_;
goto v_reusejp_4333_;
}
else
{
lean_object* v_reuseFailAlloc_4335_; 
v_reuseFailAlloc_4335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4335_, 0, v_val_4329_);
v___x_4334_ = v_reuseFailAlloc_4335_;
goto v_reusejp_4333_;
}
v_reusejp_4333_:
{
v___y_4301_ = v___y_4323_;
v___y_4302_ = v___y_4324_;
v___y_4303_ = v_more_4322_;
v___y_4304_ = v___x_4334_;
goto v___jp_4300_;
}
}
}
}
}
v___jp_4260_:
{
lean_object* v___x_4271_; lean_object* v___x_4272_; lean_object* v___x_4273_; lean_object* v___x_4274_; 
lean_inc_ref(v___y_4267_);
v___x_4271_ = l_Array_append___redArg(v___y_4267_, v___y_4270_);
lean_dec_ref(v___y_4270_);
lean_inc(v___y_4263_);
lean_inc(v___y_4269_);
v___x_4272_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4272_, 0, v___y_4269_);
lean_ctor_set(v___x_4272_, 1, v___y_4263_);
lean_ctor_set(v___x_4272_, 2, v___x_4271_);
lean_inc(v___y_4266_);
v___x_4273_ = l_Lean_Syntax_node5(v___y_4269_, v___y_4266_, v___y_4268_, v___y_4262_, v___y_4261_, v___y_4264_, v___x_4272_);
v___x_4274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4274_, 0, v___x_4273_);
lean_ctor_set(v___x_4274_, 1, v___y_4265_);
return v___x_4274_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1___boxed(lean_object* v_x_4347_, lean_object* v_a_4348_, lean_object* v_a_4349_){
_start:
{
lean_object* v_res_4350_; 
v_res_4350_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Tactic_x2b__________1(v_x_4347_, v_a_4348_, v_a_4349_);
lean_dec_ref(v_a_4348_);
return v_res_4350_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1(lean_object* v_x_4381_, lean_object* v_a_4382_, lean_object* v_a_4383_){
_start:
{
lean_object* v___y_4385_; lean_object* v___y_4386_; lean_object* v___y_4387_; lean_object* v___y_4388_; lean_object* v___y_4389_; lean_object* v___y_4390_; lean_object* v___y_4391_; lean_object* v___y_4392_; lean_object* v___y_4393_; lean_object* v___y_4394_; lean_object* v___x_4399_; uint8_t v___x_4400_; 
v___x_4399_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Conv_x2b_________00__closed__1));
lean_inc(v_x_4381_);
v___x_4400_ = l_Lean_Syntax_isOfKind(v_x_4381_, v___x_4399_);
if (v___x_4400_ == 0)
{
lean_object* v___x_4401_; lean_object* v___x_4402_; 
lean_dec(v_x_4381_);
v___x_4401_ = lean_box(1);
v___x_4402_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4402_, 0, v___x_4401_);
lean_ctor_set(v___x_4402_, 1, v_a_4383_);
return v___x_4402_;
}
else
{
lean_object* v___x_4403_; lean_object* v_tk_4404_; lean_object* v___y_4406_; lean_object* v___y_4407_; lean_object* v___y_4408_; lean_object* v___y_4409_; uint8_t v___y_4410_; lean_object* v___y_4411_; lean_object* v___y_4412_; lean_object* v___y_4413_; lean_object* v___y_4414_; lean_object* v___y_4415_; lean_object* v___y_4425_; lean_object* v___y_4426_; lean_object* v___y_4427_; lean_object* v___y_4428_; lean_object* v_more_4446_; lean_object* v___y_4447_; lean_object* v___y_4448_; lean_object* v___x_4461_; lean_object* v___x_4462_; uint8_t v___x_4463_; 
v___x_4403_ = lean_unsigned_to_nat(1u);
v_tk_4404_ = l_Lean_Syntax_getArg(v_x_4381_, v___x_4403_);
v___x_4461_ = lean_unsigned_to_nat(2u);
v___x_4462_ = l_Lean_Syntax_getArg(v_x_4381_, v___x_4461_);
v___x_4463_ = l_Lean_Syntax_isNone(v___x_4462_);
if (v___x_4463_ == 0)
{
uint8_t v___x_4464_; 
lean_inc(v___x_4462_);
v___x_4464_ = l_Lean_Syntax_matchesNull(v___x_4462_, v___x_4403_);
if (v___x_4464_ == 0)
{
lean_object* v___x_4465_; lean_object* v___x_4466_; 
lean_dec(v___x_4462_);
lean_dec(v_tk_4404_);
lean_dec(v_x_4381_);
v___x_4465_ = lean_box(1);
v___x_4466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4466_, 0, v___x_4465_);
lean_ctor_set(v___x_4466_, 1, v_a_4383_);
return v___x_4466_;
}
else
{
lean_object* v___x_4467_; lean_object* v_more_4468_; lean_object* v___x_4469_; 
v___x_4467_ = lean_unsigned_to_nat(0u);
v_more_4468_ = l_Lean_Syntax_getArg(v___x_4462_, v___x_4467_);
lean_dec(v___x_4462_);
v___x_4469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4469_, 0, v_more_4468_);
v_more_4446_ = v___x_4469_;
v___y_4447_ = v_a_4382_;
v___y_4448_ = v_a_4383_;
goto v___jp_4445_;
}
}
else
{
lean_object* v___x_4470_; 
lean_dec(v___x_4462_);
v___x_4470_ = lean_box(0);
v_more_4446_ = v___x_4470_;
v___y_4447_ = v_a_4382_;
v___y_4448_ = v_a_4383_;
goto v___jp_4445_;
}
v___jp_4405_:
{
lean_object* v___x_4416_; lean_object* v___x_4417_; lean_object* v___x_4418_; lean_object* v___x_4419_; 
lean_inc_ref(v___y_4412_);
v___x_4416_ = l_Array_append___redArg(v___y_4412_, v___y_4415_);
lean_dec_ref(v___y_4415_);
lean_inc(v___y_4407_);
lean_inc(v___y_4411_);
v___x_4417_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4417_, 0, v___y_4411_);
lean_ctor_set(v___x_4417_, 1, v___y_4407_);
lean_ctor_set(v___x_4417_, 2, v___x_4416_);
v___x_4418_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1___closed__0));
v___x_4419_ = l_Lean_mkIdentFrom(v_tk_4404_, v___x_4418_, v___y_4410_);
lean_dec(v_tk_4404_);
if (lean_obj_tag(v___y_4406_) == 0)
{
lean_object* v___x_4420_; 
v___x_4420_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4385_ = v___x_4419_;
v___y_4386_ = v___y_4407_;
v___y_4387_ = v___x_4417_;
v___y_4388_ = v___y_4408_;
v___y_4389_ = v___y_4409_;
v___y_4390_ = v___y_4411_;
v___y_4391_ = v___y_4412_;
v___y_4392_ = v___y_4413_;
v___y_4393_ = v___y_4414_;
v___y_4394_ = v___x_4420_;
goto v___jp_4384_;
}
else
{
lean_object* v_val_4421_; lean_object* v___x_4422_; lean_object* v___x_4423_; 
v_val_4421_ = lean_ctor_get(v___y_4406_, 0);
lean_inc(v_val_4421_);
lean_dec_ref_known(v___y_4406_, 1);
v___x_4422_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___x_4423_ = lean_array_push(v___x_4422_, v_val_4421_);
v___y_4385_ = v___x_4419_;
v___y_4386_ = v___y_4407_;
v___y_4387_ = v___x_4417_;
v___y_4388_ = v___y_4408_;
v___y_4389_ = v___y_4409_;
v___y_4390_ = v___y_4411_;
v___y_4391_ = v___y_4412_;
v___y_4392_ = v___y_4413_;
v___y_4393_ = v___y_4414_;
v___y_4394_ = v___x_4423_;
goto v___jp_4384_;
}
}
v___jp_4424_:
{
lean_object* v_ref_4429_; uint8_t v___x_4430_; lean_object* v___x_4431_; lean_object* v___x_4432_; lean_object* v___x_4433_; lean_object* v___x_4434_; lean_object* v___x_4435_; lean_object* v___x_4436_; lean_object* v___x_4437_; lean_object* v___x_4438_; 
v_ref_4429_ = lean_ctor_get(v___y_4425_, 5);
v___x_4430_ = 0;
v___x_4431_ = l_Lean_SourceInfo_fromRef(v_ref_4429_, v___x_4430_);
v___x_4432_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1));
v___x_4433_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2));
lean_inc_n(v___x_4431_, 2);
v___x_4434_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4434_, 0, v___x_4431_);
lean_ctor_set(v___x_4434_, 1, v___x_4433_);
v___x_4435_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2));
v___x_4436_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4436_, 0, v___x_4431_);
lean_ctor_set(v___x_4436_, 1, v___x_4435_);
v___x_4437_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4));
v___x_4438_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5);
if (lean_obj_tag(v___y_4426_) == 1)
{
lean_object* v_val_4439_; lean_object* v___x_4440_; lean_object* v___x_4441_; lean_object* v___x_4442_; lean_object* v___x_4443_; 
v_val_4439_ = lean_ctor_get(v___y_4426_, 0);
lean_inc(v_val_4439_);
lean_dec_ref_known(v___y_4426_, 1);
v___x_4440_ = l_Lean_SourceInfo_fromRef(v_val_4439_, v___x_4400_);
lean_dec(v_val_4439_);
v___x_4441_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5));
v___x_4442_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4442_, 0, v___x_4440_);
lean_ctor_set(v___x_4442_, 1, v___x_4441_);
v___x_4443_ = l_Array_mkArray1___redArg(v___x_4442_);
v___y_4406_ = v___y_4428_;
v___y_4407_ = v___x_4437_;
v___y_4408_ = v___x_4436_;
v___y_4409_ = v___x_4432_;
v___y_4410_ = v___x_4430_;
v___y_4411_ = v___x_4431_;
v___y_4412_ = v___x_4438_;
v___y_4413_ = v___y_4427_;
v___y_4414_ = v___x_4434_;
v___y_4415_ = v___x_4443_;
goto v___jp_4405_;
}
else
{
lean_object* v___x_4444_; 
lean_dec(v___y_4426_);
v___x_4444_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4406_ = v___y_4428_;
v___y_4407_ = v___x_4437_;
v___y_4408_ = v___x_4436_;
v___y_4409_ = v___x_4432_;
v___y_4410_ = v___x_4430_;
v___y_4411_ = v___x_4431_;
v___y_4412_ = v___x_4438_;
v___y_4413_ = v___y_4427_;
v___y_4414_ = v___x_4434_;
v___y_4415_ = v___x_4444_;
goto v___jp_4405_;
}
}
v___jp_4445_:
{
lean_object* v___x_4449_; lean_object* v___x_4450_; lean_object* v___x_4451_; 
v___x_4449_ = lean_unsigned_to_nat(3u);
v___x_4450_ = l_Lean_Syntax_getArg(v_x_4381_, v___x_4449_);
lean_dec(v_x_4381_);
v___x_4451_ = l_Lean_Syntax_getOptional_x3f(v___x_4450_);
lean_dec(v___x_4450_);
if (lean_obj_tag(v___x_4451_) == 0)
{
lean_object* v___x_4452_; 
v___x_4452_ = lean_box(0);
v___y_4425_ = v___y_4447_;
v___y_4426_ = v_more_4446_;
v___y_4427_ = v___y_4448_;
v___y_4428_ = v___x_4452_;
goto v___jp_4424_;
}
else
{
lean_object* v_val_4453_; lean_object* v___x_4455_; uint8_t v_isShared_4456_; uint8_t v_isSharedCheck_4460_; 
v_val_4453_ = lean_ctor_get(v___x_4451_, 0);
v_isSharedCheck_4460_ = !lean_is_exclusive(v___x_4451_);
if (v_isSharedCheck_4460_ == 0)
{
v___x_4455_ = v___x_4451_;
v_isShared_4456_ = v_isSharedCheck_4460_;
goto v_resetjp_4454_;
}
else
{
lean_inc(v_val_4453_);
lean_dec(v___x_4451_);
v___x_4455_ = lean_box(0);
v_isShared_4456_ = v_isSharedCheck_4460_;
goto v_resetjp_4454_;
}
v_resetjp_4454_:
{
lean_object* v___x_4458_; 
if (v_isShared_4456_ == 0)
{
v___x_4458_ = v___x_4455_;
goto v_reusejp_4457_;
}
else
{
lean_object* v_reuseFailAlloc_4459_; 
v_reuseFailAlloc_4459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4459_, 0, v_val_4453_);
v___x_4458_ = v_reuseFailAlloc_4459_;
goto v_reusejp_4457_;
}
v_reusejp_4457_:
{
v___y_4425_ = v___y_4447_;
v___y_4426_ = v_more_4446_;
v___y_4427_ = v___y_4448_;
v___y_4428_ = v___x_4458_;
goto v___jp_4424_;
}
}
}
}
}
v___jp_4384_:
{
lean_object* v___x_4395_; lean_object* v___x_4396_; lean_object* v___x_4397_; lean_object* v___x_4398_; 
lean_inc_ref(v___y_4391_);
v___x_4395_ = l_Array_append___redArg(v___y_4391_, v___y_4394_);
lean_dec_ref(v___y_4394_);
lean_inc(v___y_4386_);
lean_inc(v___y_4390_);
v___x_4396_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4396_, 0, v___y_4390_);
lean_ctor_set(v___x_4396_, 1, v___y_4386_);
lean_ctor_set(v___x_4396_, 2, v___x_4395_);
lean_inc(v___y_4389_);
v___x_4397_ = l_Lean_Syntax_node5(v___y_4390_, v___y_4389_, v___y_4393_, v___y_4388_, v___y_4387_, v___y_4385_, v___x_4396_);
v___x_4398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4398_, 0, v___x_4397_);
lean_ctor_set(v___x_4398_, 1, v___y_4392_);
return v___x_4398_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1___boxed(lean_object* v_x_4471_, lean_object* v_a_4472_, lean_object* v_a_4473_){
_start:
{
lean_object* v_res_4474_; 
v_res_4474_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Conv_x2b__________1(v_x_4471_, v_a_4472_, v_a_4473_);
lean_dec_ref(v_a_4472_);
return v_res_4474_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1(lean_object* v_x_4505_, lean_object* v_a_4506_, lean_object* v_a_4507_){
_start:
{
lean_object* v___y_4509_; lean_object* v___y_4510_; lean_object* v___y_4511_; lean_object* v___y_4512_; lean_object* v___y_4513_; lean_object* v___y_4514_; lean_object* v___y_4515_; lean_object* v___y_4516_; lean_object* v___y_4517_; lean_object* v___y_4518_; lean_object* v___x_4523_; uint8_t v___x_4524_; 
v___x_4523_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Command_x2b_________00__closed__1));
lean_inc(v_x_4505_);
v___x_4524_ = l_Lean_Syntax_isOfKind(v_x_4505_, v___x_4523_);
if (v___x_4524_ == 0)
{
lean_object* v___x_4525_; lean_object* v___x_4526_; 
lean_dec(v_x_4505_);
v___x_4525_ = lean_box(1);
v___x_4526_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4526_, 0, v___x_4525_);
lean_ctor_set(v___x_4526_, 1, v_a_4507_);
return v___x_4526_;
}
else
{
lean_object* v___x_4527_; lean_object* v_tk_4528_; lean_object* v___y_4530_; lean_object* v___y_4531_; lean_object* v___y_4532_; uint8_t v___y_4533_; lean_object* v___y_4534_; lean_object* v___y_4535_; lean_object* v___y_4536_; lean_object* v___y_4537_; lean_object* v___y_4538_; lean_object* v___y_4539_; lean_object* v___y_4540_; lean_object* v___y_4549_; lean_object* v___y_4550_; lean_object* v___y_4551_; lean_object* v___y_4552_; lean_object* v_more_4571_; lean_object* v___y_4572_; lean_object* v___y_4573_; lean_object* v___x_4586_; lean_object* v___x_4587_; uint8_t v___x_4588_; 
v___x_4527_ = lean_unsigned_to_nat(1u);
v_tk_4528_ = l_Lean_Syntax_getArg(v_x_4505_, v___x_4527_);
v___x_4586_ = lean_unsigned_to_nat(2u);
v___x_4587_ = l_Lean_Syntax_getArg(v_x_4505_, v___x_4586_);
v___x_4588_ = l_Lean_Syntax_isNone(v___x_4587_);
if (v___x_4588_ == 0)
{
uint8_t v___x_4589_; 
lean_inc(v___x_4587_);
v___x_4589_ = l_Lean_Syntax_matchesNull(v___x_4587_, v___x_4527_);
if (v___x_4589_ == 0)
{
lean_object* v___x_4590_; lean_object* v___x_4591_; 
lean_dec(v___x_4587_);
lean_dec(v_tk_4528_);
lean_dec(v_x_4505_);
v___x_4590_ = lean_box(1);
v___x_4591_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4591_, 0, v___x_4590_);
lean_ctor_set(v___x_4591_, 1, v_a_4507_);
return v___x_4591_;
}
else
{
lean_object* v___x_4592_; lean_object* v_more_4593_; lean_object* v___x_4594_; 
v___x_4592_ = lean_unsigned_to_nat(0u);
v_more_4593_ = l_Lean_Syntax_getArg(v___x_4587_, v___x_4592_);
lean_dec(v___x_4587_);
v___x_4594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4594_, 0, v_more_4593_);
v_more_4571_ = v___x_4594_;
v___y_4572_ = v_a_4506_;
v___y_4573_ = v_a_4507_;
goto v___jp_4570_;
}
}
else
{
lean_object* v___x_4595_; 
lean_dec(v___x_4587_);
v___x_4595_ = lean_box(0);
v_more_4571_ = v___x_4595_;
v___y_4572_ = v_a_4506_;
v___y_4573_ = v_a_4507_;
goto v___jp_4570_;
}
v___jp_4529_:
{
lean_object* v___x_4541_; lean_object* v___x_4542_; lean_object* v___x_4543_; 
lean_inc_ref(v___y_4532_);
v___x_4541_ = l_Array_append___redArg(v___y_4532_, v___y_4540_);
lean_dec_ref(v___y_4540_);
lean_inc(v___y_4530_);
lean_inc(v___y_4534_);
v___x_4542_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4542_, 0, v___y_4534_);
lean_ctor_set(v___x_4542_, 1, v___y_4530_);
lean_ctor_set(v___x_4542_, 2, v___x_4541_);
lean_inc(v___y_4539_);
v___x_4543_ = l_Lean_mkIdentFrom(v_tk_4528_, v___y_4539_, v___y_4533_);
lean_dec(v_tk_4528_);
if (lean_obj_tag(v___y_4535_) == 0)
{
lean_object* v___x_4544_; 
v___x_4544_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4509_ = v___x_4542_;
v___y_4510_ = v___y_4530_;
v___y_4511_ = v___y_4531_;
v___y_4512_ = v___y_4532_;
v___y_4513_ = v___y_4534_;
v___y_4514_ = v___x_4543_;
v___y_4515_ = v___y_4536_;
v___y_4516_ = v___y_4537_;
v___y_4517_ = v___y_4538_;
v___y_4518_ = v___x_4544_;
goto v___jp_4508_;
}
else
{
lean_object* v_val_4545_; lean_object* v___x_4546_; lean_object* v___x_4547_; 
v_val_4545_ = lean_ctor_get(v___y_4535_, 0);
lean_inc(v_val_4545_);
lean_dec_ref_known(v___y_4535_, 1);
v___x_4546_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___x_4547_ = lean_array_push(v___x_4546_, v_val_4545_);
v___y_4509_ = v___x_4542_;
v___y_4510_ = v___y_4530_;
v___y_4511_ = v___y_4531_;
v___y_4512_ = v___y_4532_;
v___y_4513_ = v___y_4534_;
v___y_4514_ = v___x_4543_;
v___y_4515_ = v___y_4536_;
v___y_4516_ = v___y_4537_;
v___y_4517_ = v___y_4538_;
v___y_4518_ = v___x_4547_;
goto v___jp_4508_;
}
}
v___jp_4548_:
{
lean_object* v_ref_4553_; uint8_t v___x_4554_; lean_object* v___x_4555_; lean_object* v___x_4556_; lean_object* v___x_4557_; lean_object* v___x_4558_; lean_object* v___x_4559_; lean_object* v___x_4560_; lean_object* v___x_4561_; lean_object* v___x_4562_; lean_object* v___x_4563_; 
v_ref_4553_ = lean_ctor_get(v___y_4550_, 5);
v___x_4554_ = 0;
v___x_4555_ = l_Lean_SourceInfo_fromRef(v_ref_4553_, v___x_4554_);
v___x_4556_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1___closed__0));
v___x_4557_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__1));
v___x_4558_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__2));
lean_inc_n(v___x_4555_, 2);
v___x_4559_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4559_, 0, v___x_4555_);
lean_ctor_set(v___x_4559_, 1, v___x_4558_);
v___x_4560_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__2));
v___x_4561_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4561_, 0, v___x_4555_);
lean_ctor_set(v___x_4561_, 1, v___x_4560_);
v___x_4562_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__4));
v___x_4563_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__5);
if (lean_obj_tag(v___y_4549_) == 1)
{
lean_object* v_val_4564_; lean_object* v___x_4565_; lean_object* v___x_4566_; lean_object* v___x_4567_; lean_object* v___x_4568_; 
v_val_4564_ = lean_ctor_get(v___y_4549_, 0);
lean_inc(v_val_4564_);
lean_dec_ref_known(v___y_4549_, 1);
v___x_4565_ = l_Lean_SourceInfo_fromRef(v_val_4564_, v___x_4524_);
lean_dec(v_val_4564_);
v___x_4566_ = ((lean_object*)(lp_batteries_Batteries_Tactic_command_x23help__Cat_x2b_____________00__closed__5));
v___x_4567_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4567_, 0, v___x_4565_);
lean_ctor_set(v___x_4567_, 1, v___x_4566_);
v___x_4568_ = l_Array_mkArray1___redArg(v___x_4567_);
v___y_4530_ = v___x_4562_;
v___y_4531_ = v___x_4559_;
v___y_4532_ = v___x_4563_;
v___y_4533_ = v___x_4554_;
v___y_4534_ = v___x_4555_;
v___y_4535_ = v___y_4552_;
v___y_4536_ = v___x_4561_;
v___y_4537_ = v___y_4551_;
v___y_4538_ = v___x_4557_;
v___y_4539_ = v___x_4556_;
v___y_4540_ = v___x_4568_;
goto v___jp_4529_;
}
else
{
lean_object* v___x_4569_; 
lean_dec(v___y_4549_);
v___x_4569_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Term_x2b__________1___closed__1));
v___y_4530_ = v___x_4562_;
v___y_4531_ = v___x_4559_;
v___y_4532_ = v___x_4563_;
v___y_4533_ = v___x_4554_;
v___y_4534_ = v___x_4555_;
v___y_4535_ = v___y_4552_;
v___y_4536_ = v___x_4561_;
v___y_4537_ = v___y_4551_;
v___y_4538_ = v___x_4557_;
v___y_4539_ = v___x_4556_;
v___y_4540_ = v___x_4569_;
goto v___jp_4529_;
}
}
v___jp_4570_:
{
lean_object* v___x_4574_; lean_object* v___x_4575_; lean_object* v___x_4576_; 
v___x_4574_ = lean_unsigned_to_nat(3u);
v___x_4575_ = l_Lean_Syntax_getArg(v_x_4505_, v___x_4574_);
lean_dec(v_x_4505_);
v___x_4576_ = l_Lean_Syntax_getOptional_x3f(v___x_4575_);
lean_dec(v___x_4575_);
if (lean_obj_tag(v___x_4576_) == 0)
{
lean_object* v___x_4577_; 
v___x_4577_ = lean_box(0);
v___y_4549_ = v_more_4571_;
v___y_4550_ = v___y_4572_;
v___y_4551_ = v___y_4573_;
v___y_4552_ = v___x_4577_;
goto v___jp_4548_;
}
else
{
lean_object* v_val_4578_; lean_object* v___x_4580_; uint8_t v_isShared_4581_; uint8_t v_isSharedCheck_4585_; 
v_val_4578_ = lean_ctor_get(v___x_4576_, 0);
v_isSharedCheck_4585_ = !lean_is_exclusive(v___x_4576_);
if (v_isSharedCheck_4585_ == 0)
{
v___x_4580_ = v___x_4576_;
v_isShared_4581_ = v_isSharedCheck_4585_;
goto v_resetjp_4579_;
}
else
{
lean_inc(v_val_4578_);
lean_dec(v___x_4576_);
v___x_4580_ = lean_box(0);
v_isShared_4581_ = v_isSharedCheck_4585_;
goto v_resetjp_4579_;
}
v_resetjp_4579_:
{
lean_object* v___x_4583_; 
if (v_isShared_4581_ == 0)
{
v___x_4583_ = v___x_4580_;
goto v_reusejp_4582_;
}
else
{
lean_object* v_reuseFailAlloc_4584_; 
v_reuseFailAlloc_4584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4584_, 0, v_val_4578_);
v___x_4583_ = v_reuseFailAlloc_4584_;
goto v_reusejp_4582_;
}
v_reusejp_4582_:
{
v___y_4549_ = v_more_4571_;
v___y_4550_ = v___y_4572_;
v___y_4551_ = v___y_4573_;
v___y_4552_ = v___x_4583_;
goto v___jp_4548_;
}
}
}
}
}
v___jp_4508_:
{
lean_object* v___x_4519_; lean_object* v___x_4520_; lean_object* v___x_4521_; lean_object* v___x_4522_; 
lean_inc_ref(v___y_4512_);
v___x_4519_ = l_Array_append___redArg(v___y_4512_, v___y_4518_);
lean_dec_ref(v___y_4518_);
lean_inc(v___y_4510_);
lean_inc(v___y_4513_);
v___x_4520_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4520_, 0, v___y_4513_);
lean_ctor_set(v___x_4520_, 1, v___y_4510_);
lean_ctor_set(v___x_4520_, 2, v___x_4519_);
lean_inc(v___y_4517_);
v___x_4521_ = l_Lean_Syntax_node5(v___y_4513_, v___y_4517_, v___y_4511_, v___y_4515_, v___y_4509_, v___y_4514_, v___x_4520_);
v___x_4522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4522_, 0, v___x_4521_);
lean_ctor_set(v___x_4522_, 1, v___y_4516_);
return v___x_4522_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1___boxed(lean_object* v_x_4596_, lean_object* v_a_4597_, lean_object* v_a_4598_){
_start:
{
lean_object* v_res_4599_; 
v_res_4599_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__HelpCmd______macroRules__Batteries__Tactic__command_x23help__Command_x2b__________1(v_x_4596_, v_a_4597_, v_a_4598_);
lean_dec_ref(v_a_4597_);
return v_res_4599_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Syntax(uint8_t builtin);
lean_object* runtime_initialize_Lean_DocString(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Syntax(uint8_t builtin);
lean_object* initialize_Lean_DocString(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_LibraryNote(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_LibraryNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
}
#ifdef __cplusplus
}
#endif
