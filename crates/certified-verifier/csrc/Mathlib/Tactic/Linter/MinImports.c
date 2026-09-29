// Lean compiler output
// Module: Mathlib.Tactic.Linter.MinImports
// Imports: public import Init public meta import Init public meta import ImportGraph.Imports.ImportGraph public meta import ImportGraph.Graph.TransitiveClosure public import Mathlib.Tactic.MinImports
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
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
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
lean_object* l_Lean_Syntax_getId(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_find_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_NameSet_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* lp_importGraph_Lean_Environment_importGraph(lean_object*);
lean_object* lp_importGraph_Lean_NameMap_transitiveClosure(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lp_importGraph_Lean_Environment_findRedundantImports(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_maxView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_minView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getId(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getAllImports(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(lean_object*, lean_object*);
lean_object* l_Lean_NameSet_filter(lean_object*, lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_IO_FS_readFile(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_Parser_mkInputContext___redArg(lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Parser_parseHeader(lean_object*);
lean_object* l_Lean_NameSet_ofArray(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_Syntax_getKind(lean_object*);
lean_object* l_Lean_Environment_imports(lean_object*);
uint8_t lp_mathlib_Mathlib_Command_MinImports_isInitImport(lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t l_Lean_MessageLog_hasErrors(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_withSetOptionIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_addLinter(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_instInhabitedImportState_default;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_instInhabitedImportState;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2382540021____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2382540021____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_minImportsRef;
static const lean_string_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Linter"};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "command#reset_min_imports"};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 211, 21, 175, 26, 248, 95, 50)}};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "#reset_min_imports"};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_command_x23reset__min__imports = (const lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "minImports"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(232, 248, 199, 125, 230, 70, 85, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "enable the minImports linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(227, 82, 44, 147, 4, 169, 143, 163)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_minImports;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "increases"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(232, 248, 199, 125, 230, 70, 85, 160)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(22, 76, 163, 134, 184, 3, 13, 36)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "enable reporting increase-size change in the minImports linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(101, 237, 90, 120, 51, 59, 46, 172)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(227, 82, 44, 147, 4, 169, 143, 163)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__0_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(217, 179, 49, 46, 76, 78, 227, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_linter_minImports_increases;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "MinImports"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "command#import_bumps"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value),LEAN_SCALAR_PTR_LITERAL(120, 131, 127, 204, 79, 169, 80, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__0_value),LEAN_SCALAR_PTR_LITERAL(110, 179, 185, 47, 136, 123, 218, 112)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__1_value),LEAN_SCALAR_PTR_LITERAL(68, 216, 104, 164, 119, 27, 169, 145)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "#import_bumps"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "runCmd"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(65, 158, 215, 209, 131, 110, 142, 142)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "run_cmd"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "doSeqIndent"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(93, 115, 138, 230, 225, 195, 43, 46)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "doSeqItem"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(10, 94, 50, 120, 46, 251, 13, 13)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "doExpr"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(130, 168, 60, 255, 153, 218, 88, 77)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "logInfo"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__17;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(202, 91, 142, 254, 66, 53, 122, 238)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(203, 37, 56, 83, 4, 68, 47, 204)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "\"Counting imports from here.\""};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "set_option"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(216, 223, 149, 245, 150, 86, 134, 198)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Elab.async"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__29_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__30;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "async"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(6, 0, 36, 68, 138, 2, 151, 20)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(163, 142, 149, 180, 91, 16, 128, 108)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__35_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "linter.minImports"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__38_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__39;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__41_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__40_value),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__42_value)}};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__44_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "import"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(237, 201, 190, 222, 246, 15, 232, 234)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__1(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__21___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16_spec__23(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__4(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "import "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "unneeded import '"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__0 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__0_value;
static lean_once_cell_t lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__1;
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__2 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__3;
static const lean_closure_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__1___boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__4 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__4_value;
static const lean_ctor_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__2_value)}};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__5 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__5_value;
static const lean_string_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "' not found"};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__6 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__6_value;
static const lean_ctor_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__6_value)}};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__7 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10_spec__15(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__11(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Imports increased "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "to\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "\n\nNew imports: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "\nNow redundant: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__9;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "by "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__12;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "-- missing imports\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__15_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__16;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "eoi"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "exit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__18_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__19_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__20;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 194, .m_capacity = 194, .m_length = 193, .m_data = "Try using '#import_bumps', instead of manually setting the linter option: the linter works best with linear parsing of the file and '#import_bumps' also sets the `Elab.async` option to `false`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__21_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__21_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__22_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__23;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__0_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_withSetOptionIn___boxed, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__6_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 204, 154, 235, 250, 222, 148, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 14, 48, 79, 49, 230, 247, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(58, 15, 236, 171, 195, 45, 107, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(139, 118, 57, 236, 208, 89, 15, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__1_value),LEAN_SCALAR_PTR_LITERAL(201, 91, 209, 41, 240, 170, 208, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__0_value),LEAN_SCALAR_PTR_LITERAL(147, 179, 189, 106, 194, 63, 72, 226)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "minImportsLinter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__13_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__14_value),LEAN_SCALAR_PTR_LITERAL(93, 28, 212, 169, 213, 250, 133, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_1947129272____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_1947129272____hygCtx___hyg_2____boxed(lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_1_ = lean_unsigned_to_nat(0u);
v___x_2_ = l_Lean_NameSet_empty;
v___x_3_ = lean_box(0);
v___x_4_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4_, 0, v___x_3_);
lean_ctor_set(v___x_4_, 1, v___x_2_);
lean_ctor_set(v___x_4_, 2, v___x_1_);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_instInhabitedImportState_default(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0, &lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0_once, _init_lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0);
return v___x_5_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_instInhabitedImportState(void){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lp_mathlib_Mathlib_Linter_instInhabitedImportState_default;
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2382540021____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_8_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0, &lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0_once, _init_lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0);
v___x_9_ = lean_st_mk_ref(v___x_8_);
v___x_10_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_10_, 0, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2382540021____hygCtx___hyg_2____boxed(lean_object* v_a_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2382540021____hygCtx___hyg_2_();
return v_res_12_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_28_ = lean_box(0);
v___x_29_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_30_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___x_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg(){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___closed__0);
v___x_33_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg___boxed(lean_object* v___y_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg();
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0(lean_object* v_00_u03b1_36_, lean_object* v___y_37_, lean_object* v___y_38_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg();
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___boxed(lean_object* v_00_u03b1_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0(v_00_u03b1_41_, v___y_42_, v___y_43_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1(lean_object* v_x_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v___x_50_; uint8_t v___x_51_; 
v___x_50_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_command_x23reset__min__imports___closed__3));
v___x_51_ = l_Lean_Syntax_isOfKind(v_x_46_, v___x_50_);
if (v___x_51_ == 0)
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1_spec__0___redArg();
return v___x_52_;
}
else
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_minImportsRef;
v___x_54_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0, &lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0_once, _init_lp_mathlib_Mathlib_Linter_instInhabitedImportState_default___closed__0);
v___x_55_ = lean_st_ref_set(v___x_53_, v___x_54_);
v___x_56_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
return v___x_56_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1___boxed(lean_object* v_x_57_, lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Mathlib_Linter___aux__Mathlib__Tactic__Linter__MinImports______elabRules__Mathlib__Linter__command_x23reset__min__imports__1(v_x_57_, v_a_58_, v_a_59_);
lean_dec(v_a_59_);
lean_dec_ref(v_a_58_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0(lean_object* v_name_62_, lean_object* v_decl_63_, lean_object* v_ref_64_){
_start:
{
lean_object* v_defValue_66_; lean_object* v_descr_67_; lean_object* v_deprecation_x3f_68_; lean_object* v___x_69_; uint8_t v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_defValue_66_ = lean_ctor_get(v_decl_63_, 0);
v_descr_67_ = lean_ctor_get(v_decl_63_, 1);
v_deprecation_x3f_68_ = lean_ctor_get(v_decl_63_, 2);
v___x_69_ = lean_alloc_ctor(1, 0, 1);
v___x_70_ = lean_unbox(v_defValue_66_);
lean_ctor_set_uint8(v___x_69_, 0, v___x_70_);
lean_inc(v_deprecation_x3f_68_);
lean_inc_ref(v_descr_67_);
lean_inc_n(v_name_62_, 2);
v___x_71_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_71_, 0, v_name_62_);
lean_ctor_set(v___x_71_, 1, v_ref_64_);
lean_ctor_set(v___x_71_, 2, v___x_69_);
lean_ctor_set(v___x_71_, 3, v_descr_67_);
lean_ctor_set(v___x_71_, 4, v_deprecation_x3f_68_);
v___x_72_ = lean_register_option(v_name_62_, v___x_71_);
if (lean_obj_tag(v___x_72_) == 0)
{
lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_80_; 
v_isSharedCheck_80_ = !lean_is_exclusive(v___x_72_);
if (v_isSharedCheck_80_ == 0)
{
lean_object* v_unused_81_; 
v_unused_81_ = lean_ctor_get(v___x_72_, 0);
lean_dec(v_unused_81_);
v___x_74_ = v___x_72_;
v_isShared_75_ = v_isSharedCheck_80_;
goto v_resetjp_73_;
}
else
{
lean_dec(v___x_72_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_80_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___x_76_; lean_object* v___x_78_; 
lean_inc(v_defValue_66_);
v___x_76_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_76_, 0, v_name_62_);
lean_ctor_set(v___x_76_, 1, v_defValue_66_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_76_);
v___x_78_ = v___x_74_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_79_; 
v_reuseFailAlloc_79_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_79_, 0, v___x_76_);
v___x_78_ = v_reuseFailAlloc_79_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
return v___x_78_;
}
}
}
else
{
lean_object* v_a_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
lean_dec(v_name_62_);
v_a_82_ = lean_ctor_get(v___x_72_, 0);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_72_);
if (v_isSharedCheck_89_ == 0)
{
v___x_84_ = v___x_72_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_a_82_);
lean_dec(v___x_72_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_82_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_90_, lean_object* v_decl_91_, lean_object* v_ref_92_, lean_object* v_a_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0(v_name_90_, v_decl_91_, v_ref_92_);
lean_dec_ref(v_decl_91_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_112_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_));
v___x_113_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_));
v___x_114_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__5_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_));
v___x_115_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0(v___x_112_, v___x_113_, v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4____boxed(lean_object* v_a_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_();
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_136_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__1_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_));
v___x_137_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__3_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_));
v___x_138_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__4_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_));
v___x_139_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4__spec__0(v___x_136_, v___x_137_, v___x_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4____boxed(lean_object* v_a_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_();
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg(lean_object* v_t_142_, lean_object* v_k_143_, lean_object* v_fallback_144_){
_start:
{
if (lean_obj_tag(v_t_142_) == 0)
{
lean_object* v_k_145_; lean_object* v_v_146_; lean_object* v_l_147_; lean_object* v_r_148_; uint8_t v___x_149_; 
v_k_145_ = lean_ctor_get(v_t_142_, 1);
v_v_146_ = lean_ctor_get(v_t_142_, 2);
v_l_147_ = lean_ctor_get(v_t_142_, 3);
v_r_148_ = lean_ctor_get(v_t_142_, 4);
v___x_149_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_143_, v_k_145_);
switch(v___x_149_)
{
case 0:
{
v_t_142_ = v_l_147_;
goto _start;
}
case 1:
{
lean_inc(v_v_146_);
return v_v_146_;
}
default: 
{
v_t_142_ = v_r_148_;
goto _start;
}
}
}
else
{
lean_inc(v_fallback_144_);
return v_fallback_144_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg___boxed(lean_object* v_t_152_, lean_object* v_k_153_, lean_object* v_fallback_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg(v_t_152_, v_k_153_, v_fallback_154_);
lean_dec(v_fallback_154_);
lean_dec(v_k_153_);
lean_dec(v_t_152_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(lean_object* v_tc_156_, lean_object* v_init_157_, lean_object* v_x_158_){
_start:
{
if (lean_obj_tag(v_x_158_) == 0)
{
lean_object* v_k_159_; lean_object* v_l_160_; lean_object* v_r_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v_k_159_ = lean_ctor_get(v_x_158_, 1);
v_l_160_ = lean_ctor_get(v_x_158_, 3);
v_r_161_ = lean_ctor_get(v_x_158_, 4);
v___x_162_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(v_tc_156_, v_init_157_, v_l_160_);
v___x_163_ = l_Lean_NameSet_empty;
v___x_164_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg(v_tc_156_, v_k_159_, v___x_163_);
v___x_165_ = l_Lean_NameSet_append(v___x_162_, v___x_164_);
v_init_157_ = v___x_165_;
v_x_158_ = v_r_161_;
goto _start;
}
else
{
return v_init_157_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1___boxed(lean_object* v_tc_167_, lean_object* v_init_168_, lean_object* v_x_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(v_tc_167_, v_init_168_, v_x_169_);
lean_dec(v_x_169_);
lean_dec(v_tc_167_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow(lean_object* v_tc_171_, lean_object* v_ms_172_){
_start:
{
lean_object* v___x_173_; 
lean_inc(v_ms_172_);
v___x_173_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(v_tc_171_, v_ms_172_, v_ms_172_);
lean_dec(v_ms_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow___boxed(lean_object* v_tc_174_, lean_object* v_ms_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow(v_tc_174_, v_ms_175_);
lean_dec(v_tc_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0(lean_object* v_00_u03b4_177_, lean_object* v_t_178_, lean_object* v_k_179_, lean_object* v_fallback_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___redArg(v_t_178_, v_k_179_, v_fallback_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0___boxed(lean_object* v_00_u03b4_182_, lean_object* v_t_183_, lean_object* v_k_184_, lean_object* v_fallback_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__0(v_00_u03b4_182_, v_t_183_, v_k_184_, v_fallback_185_);
lean_dec(v_fallback_185_);
lean_dec(v_k_184_);
lean_dec(v_t_183_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1(lean_object* v_tc_187_, lean_object* v_init_188_, lean_object* v_t_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(v_tc_187_, v_init_188_, v_t_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1___boxed(lean_object* v_tc_191_, lean_object* v_init_192_, lean_object* v_t_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1(v_tc_191_, v_init_192_, v_t_193_);
lean_dec(v_t_193_);
lean_dec(v_tc_191_);
return v_res_194_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__17(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__16));
v___x_247_ = l_String_toRawSubstring_x27(v___x_246_);
return v___x_247_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25(void){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = l_Array_mkArray0(lean_box(0));
return v___x_263_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__30(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_272_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__29));
v___x_273_ = l_String_toRawSubstring_x27(v___x_272_);
return v___x_273_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__39(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__38));
v___x_292_ = l_String_toRawSubstring_x27(v___x_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1(lean_object* v_x_305_, lean_object* v_a_306_, lean_object* v_a_307_){
_start:
{
lean_object* v___x_308_; uint8_t v___x_309_; 
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2));
v___x_309_ = l_Lean_Syntax_isOfKind(v_x_305_, v___x_308_);
if (v___x_309_ == 0)
{
lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_310_ = lean_box(1);
v___x_311_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v_a_307_);
return v___x_311_;
}
else
{
lean_object* v_quotContext_312_; lean_object* v_currMacroScope_313_; lean_object* v_ref_314_; uint8_t v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v_quotContext_312_ = lean_ctor_get(v_a_306_, 1);
v_currMacroScope_313_ = lean_ctor_get(v_a_306_, 2);
v_ref_314_ = lean_ctor_get(v_a_306_, 5);
v___x_315_ = 0;
v___x_316_ = l_Lean_SourceInfo_fromRef(v_ref_314_, v___x_315_);
v___x_317_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__1));
v___x_318_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__4));
v___x_319_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__5));
lean_inc_n(v___x_316_, 19);
v___x_320_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_316_);
lean_ctor_set(v___x_320_, 1, v___x_319_);
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__9));
v___x_322_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__11));
v___x_323_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__13));
v___x_324_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__15));
v___x_325_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__17, &lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__17_once, _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__17);
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__18));
lean_inc_n(v_currMacroScope_313_, 3);
lean_inc_n(v_quotContext_312_, 3);
v___x_327_ = l_Lean_addMacroScope(v_quotContext_312_, v___x_326_, v_currMacroScope_313_);
v___x_328_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__21));
v___x_329_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_329_, 0, v___x_316_);
lean_ctor_set(v___x_329_, 1, v___x_325_);
lean_ctor_set(v___x_329_, 2, v___x_327_);
lean_ctor_set(v___x_329_, 3, v___x_328_);
v___x_330_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__23));
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__24));
v___x_332_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_316_);
lean_ctor_set(v___x_332_, 1, v___x_331_);
v___x_333_ = l_Lean_Syntax_node1(v___x_316_, v___x_330_, v___x_332_);
v___x_334_ = l_Lean_Syntax_node1(v___x_316_, v___x_317_, v___x_333_);
v___x_335_ = l_Lean_Syntax_node2(v___x_316_, v___x_324_, v___x_329_, v___x_334_);
v___x_336_ = l_Lean_Syntax_node1(v___x_316_, v___x_323_, v___x_335_);
v___x_337_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25, &lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25_once, _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25);
v___x_338_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_338_, 0, v___x_316_);
lean_ctor_set(v___x_338_, 1, v___x_317_);
lean_ctor_set(v___x_338_, 2, v___x_337_);
lean_inc_ref_n(v___x_338_, 2);
v___x_339_ = l_Lean_Syntax_node2(v___x_316_, v___x_322_, v___x_336_, v___x_338_);
v___x_340_ = l_Lean_Syntax_node1(v___x_316_, v___x_317_, v___x_339_);
v___x_341_ = l_Lean_Syntax_node1(v___x_316_, v___x_321_, v___x_340_);
v___x_342_ = l_Lean_Syntax_node2(v___x_316_, v___x_318_, v___x_320_, v___x_341_);
v___x_343_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__27));
v___x_344_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28));
v___x_345_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_316_);
lean_ctor_set(v___x_345_, 1, v___x_343_);
v___x_346_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__30, &lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__30_once, _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__30);
v___x_347_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__33));
v___x_348_ = l_Lean_addMacroScope(v_quotContext_312_, v___x_347_, v_currMacroScope_313_);
v___x_349_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__36));
v___x_350_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_350_, 0, v___x_316_);
lean_ctor_set(v___x_350_, 1, v___x_346_);
lean_ctor_set(v___x_350_, 2, v___x_348_);
lean_ctor_set(v___x_350_, 3, v___x_349_);
v___x_351_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__37));
v___x_352_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_316_);
lean_ctor_set(v___x_352_, 1, v___x_351_);
lean_inc_ref(v___x_345_);
v___x_353_ = l_Lean_Syntax_node4(v___x_316_, v___x_344_, v___x_345_, v___x_350_, v___x_338_, v___x_352_);
v___x_354_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__39, &lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__39_once, _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__39);
v___x_355_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_));
v___x_356_ = l_Lean_addMacroScope(v_quotContext_312_, v___x_355_, v_currMacroScope_313_);
v___x_357_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__43));
v___x_358_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_358_, 0, v___x_316_);
lean_ctor_set(v___x_358_, 1, v___x_354_);
lean_ctor_set(v___x_358_, 2, v___x_356_);
lean_ctor_set(v___x_358_, 3, v___x_357_);
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__44));
v___x_360_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_360_, 0, v___x_316_);
lean_ctor_set(v___x_360_, 1, v___x_359_);
v___x_361_ = l_Lean_Syntax_node4(v___x_316_, v___x_344_, v___x_345_, v___x_358_, v___x_338_, v___x_360_);
v___x_362_ = l_Lean_Syntax_node3(v___x_316_, v___x_317_, v___x_342_, v___x_353_, v___x_361_);
v___x_363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
lean_ctor_set(v___x_363_, 1, v_a_307_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___boxed(lean_object* v_x_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1(v_x_364_, v_a_365_, v_a_366_);
lean_dec_ref(v_a_365_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg(lean_object* v___y_368_){
_start:
{
lean_object* v___x_370_; lean_object* v_env_371_; lean_object* v___x_372_; lean_object* v_mainModule_373_; lean_object* v___x_374_; 
v___x_370_ = lean_st_ref_get(v___y_368_);
v_env_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc_ref(v_env_371_);
lean_dec(v___x_370_);
v___x_372_ = l_Lean_Environment_header(v_env_371_);
lean_dec_ref(v_env_371_);
v_mainModule_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc(v_mainModule_373_);
lean_dec_ref(v___x_372_);
v___x_374_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_374_, 0, v_mainModule_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg___boxed(lean_object* v___y_375_, lean_object* v___y_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg(v___y_375_);
lean_dec(v___y_375_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17(lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg(v___y_379_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___boxed(lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17(v___y_382_, v___y_383_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
return v_res_385_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0(lean_object* v_x_389_){
_start:
{
lean_object* v___x_390_; uint8_t v___x_391_; 
v___x_390_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___closed__1));
v___x_391_ = l_Lean_Syntax_isOfKind(v_x_389_, v___x_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0___boxed(lean_object* v_x_392_){
_start:
{
uint8_t v_res_393_; lean_object* v_r_394_; 
v_res_393_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__0(v_x_392_);
v_r_394_ = lean_box(v_res_393_);
return v_r_394_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__1(uint8_t v___x_395_, uint8_t v___x_396_, lean_object* v_x_397_){
_start:
{
uint8_t v___x_398_; 
v___x_398_ = lp_mathlib_Mathlib_Command_MinImports_isInitImport(v_x_397_);
if (v___x_398_ == 0)
{
return v___x_395_;
}
else
{
return v___x_396_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__1___boxed(lean_object* v___x_399_, lean_object* v___x_400_, lean_object* v_x_401_){
_start:
{
uint8_t v___x_39419__boxed_402_; uint8_t v___x_39420__boxed_403_; uint8_t v_res_404_; lean_object* v_r_405_; 
v___x_39419__boxed_402_ = lean_unbox(v___x_399_);
v___x_39420__boxed_403_ = lean_unbox(v___x_400_);
v_res_404_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__1(v___x_39419__boxed_402_, v___x_39420__boxed_403_, v_x_401_);
lean_dec(v_x_401_);
v_r_405_ = lean_box(v_res_404_);
return v_r_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg(lean_object* v_hi_406_, lean_object* v_pivot_407_, lean_object* v_as_408_, lean_object* v_i_409_, lean_object* v_k_410_){
_start:
{
uint8_t v___x_411_; 
v___x_411_ = lean_nat_dec_lt(v_k_410_, v_hi_406_);
if (v___x_411_ == 0)
{
lean_object* v___x_412_; lean_object* v___x_413_; 
lean_dec(v_k_410_);
v___x_412_ = lean_array_fswap(v_as_408_, v_i_409_, v_hi_406_);
v___x_413_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_413_, 0, v_i_409_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
return v___x_413_;
}
else
{
lean_object* v___x_414_; uint8_t v___x_415_; 
v___x_414_ = lean_array_fget_borrowed(v_as_408_, v_k_410_);
v___x_415_ = l_Lean_Name_lt(v___x_414_, v_pivot_407_);
if (v___x_415_ == 0)
{
lean_object* v___x_416_; lean_object* v___x_417_; 
v___x_416_ = lean_unsigned_to_nat(1u);
v___x_417_ = lean_nat_add(v_k_410_, v___x_416_);
lean_dec(v_k_410_);
v_k_410_ = v___x_417_;
goto _start;
}
else
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_419_ = lean_array_fswap(v_as_408_, v_i_409_, v_k_410_);
v___x_420_ = lean_unsigned_to_nat(1u);
v___x_421_ = lean_nat_add(v_i_409_, v___x_420_);
lean_dec(v_i_409_);
v___x_422_ = lean_nat_add(v_k_410_, v___x_420_);
lean_dec(v_k_410_);
v_as_408_ = v___x_419_;
v_i_409_ = v___x_421_;
v_k_410_ = v___x_422_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg___boxed(lean_object* v_hi_424_, lean_object* v_pivot_425_, lean_object* v_as_426_, lean_object* v_i_427_, lean_object* v_k_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg(v_hi_424_, v_pivot_425_, v_as_426_, v_i_427_, v_k_428_);
lean_dec(v_pivot_425_);
lean_dec(v_hi_424_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(lean_object* v_n_430_, lean_object* v_as_431_, lean_object* v_lo_432_, lean_object* v_hi_433_){
_start:
{
lean_object* v___y_435_; uint8_t v___x_445_; 
v___x_445_ = lean_nat_dec_lt(v_lo_432_, v_hi_433_);
if (v___x_445_ == 0)
{
lean_dec(v_lo_432_);
return v_as_431_;
}
else
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v_mid_448_; lean_object* v___y_450_; lean_object* v___y_456_; lean_object* v___x_461_; lean_object* v___x_462_; uint8_t v___x_463_; 
v___x_446_ = lean_nat_add(v_lo_432_, v_hi_433_);
v___x_447_ = lean_unsigned_to_nat(1u);
v_mid_448_ = lean_nat_shiftr(v___x_446_, v___x_447_);
lean_dec(v___x_446_);
v___x_461_ = lean_array_fget_borrowed(v_as_431_, v_mid_448_);
v___x_462_ = lean_array_fget_borrowed(v_as_431_, v_lo_432_);
v___x_463_ = l_Lean_Name_lt(v___x_461_, v___x_462_);
if (v___x_463_ == 0)
{
v___y_456_ = v_as_431_;
goto v___jp_455_;
}
else
{
lean_object* v___x_464_; 
v___x_464_ = lean_array_fswap(v_as_431_, v_lo_432_, v_mid_448_);
v___y_456_ = v___x_464_;
goto v___jp_455_;
}
v___jp_449_:
{
lean_object* v___x_451_; lean_object* v___x_452_; uint8_t v___x_453_; 
v___x_451_ = lean_array_fget_borrowed(v___y_450_, v_mid_448_);
v___x_452_ = lean_array_fget_borrowed(v___y_450_, v_hi_433_);
v___x_453_ = l_Lean_Name_lt(v___x_451_, v___x_452_);
if (v___x_453_ == 0)
{
lean_dec(v_mid_448_);
v___y_435_ = v___y_450_;
goto v___jp_434_;
}
else
{
lean_object* v___x_454_; 
v___x_454_ = lean_array_fswap(v___y_450_, v_mid_448_, v_hi_433_);
lean_dec(v_mid_448_);
v___y_435_ = v___x_454_;
goto v___jp_434_;
}
}
v___jp_455_:
{
lean_object* v___x_457_; lean_object* v___x_458_; uint8_t v___x_459_; 
v___x_457_ = lean_array_fget_borrowed(v___y_456_, v_hi_433_);
v___x_458_ = lean_array_fget_borrowed(v___y_456_, v_lo_432_);
v___x_459_ = l_Lean_Name_lt(v___x_457_, v___x_458_);
if (v___x_459_ == 0)
{
v___y_450_ = v___y_456_;
goto v___jp_449_;
}
else
{
lean_object* v___x_460_; 
v___x_460_ = lean_array_fswap(v___y_456_, v_lo_432_, v_hi_433_);
v___y_450_ = v___x_460_;
goto v___jp_449_;
}
}
}
v___jp_434_:
{
lean_object* v_pivot_436_; lean_object* v___x_437_; lean_object* v_fst_438_; lean_object* v_snd_439_; uint8_t v___x_440_; 
v_pivot_436_ = lean_array_fget(v___y_435_, v_hi_433_);
lean_inc_n(v_lo_432_, 2);
v___x_437_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg(v_hi_433_, v_pivot_436_, v___y_435_, v_lo_432_, v_lo_432_);
lean_dec(v_pivot_436_);
v_fst_438_ = lean_ctor_get(v___x_437_, 0);
lean_inc(v_fst_438_);
v_snd_439_ = lean_ctor_get(v___x_437_, 1);
lean_inc(v_snd_439_);
lean_dec_ref(v___x_437_);
v___x_440_ = lean_nat_dec_le(v_hi_433_, v_fst_438_);
if (v___x_440_ == 0)
{
lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_441_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(v_n_430_, v_snd_439_, v_lo_432_, v_fst_438_);
v___x_442_ = lean_unsigned_to_nat(1u);
v___x_443_ = lean_nat_add(v_fst_438_, v___x_442_);
lean_dec(v_fst_438_);
v_as_431_ = v___x_441_;
v_lo_432_ = v___x_443_;
goto _start;
}
else
{
lean_dec(v_fst_438_);
lean_dec(v_lo_432_);
return v_snd_439_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg___boxed(lean_object* v_n_465_, lean_object* v_as_466_, lean_object* v_lo_467_, lean_object* v_hi_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(v_n_465_, v_as_466_, v_lo_467_, v_hi_468_);
lean_dec(v_hi_468_);
lean_dec(v_n_465_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(lean_object* v___x_470_, lean_object* v_as_471_, size_t v_i_472_, size_t v_stop_473_, lean_object* v_b_474_){
_start:
{
lean_object* v___y_476_; uint8_t v___x_480_; 
v___x_480_ = lean_usize_dec_eq(v_i_472_, v_stop_473_);
if (v___x_480_ == 0)
{
lean_object* v___x_481_; uint8_t v___x_482_; 
v___x_481_ = lean_array_uget_borrowed(v_as_471_, v_i_472_);
v___x_482_ = l_Lean_NameSet_contains(v___x_470_, v___x_481_);
if (v___x_482_ == 0)
{
lean_object* v___x_483_; 
lean_inc(v___x_481_);
v___x_483_ = lean_array_push(v_b_474_, v___x_481_);
v___y_476_ = v___x_483_;
goto v___jp_475_;
}
else
{
v___y_476_ = v_b_474_;
goto v___jp_475_;
}
}
else
{
return v_b_474_;
}
v___jp_475_:
{
size_t v___x_477_; size_t v___x_478_; 
v___x_477_ = ((size_t)1ULL);
v___x_478_ = lean_usize_add(v_i_472_, v___x_477_);
v_i_472_ = v___x_478_;
v_b_474_ = v___y_476_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6___boxed(lean_object* v___x_484_, lean_object* v_as_485_, lean_object* v_i_486_, lean_object* v_stop_487_, lean_object* v_b_488_){
_start:
{
size_t v_i_boxed_489_; size_t v_stop_boxed_490_; lean_object* v_res_491_; 
v_i_boxed_489_ = lean_unbox_usize(v_i_486_);
lean_dec(v_i_486_);
v_stop_boxed_490_ = lean_unbox_usize(v_stop_487_);
lean_dec(v_stop_487_);
v_res_491_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(v___x_484_, v_as_485_, v_i_boxed_489_, v_stop_boxed_490_, v_b_488_);
lean_dec_ref(v_as_485_);
lean_dec(v___x_484_);
return v_res_491_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0(uint8_t v___y_493_, uint8_t v_suppressElabErrors_494_, lean_object* v_x_495_){
_start:
{
if (lean_obj_tag(v_x_495_) == 1)
{
lean_object* v_pre_496_; 
v_pre_496_ = lean_ctor_get(v_x_495_, 0);
if (lean_obj_tag(v_pre_496_) == 0)
{
lean_object* v_str_497_; lean_object* v___x_498_; uint8_t v___x_499_; 
v_str_497_ = lean_ctor_get(v_x_495_, 1);
v___x_498_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___closed__0));
v___x_499_ = lean_string_dec_eq(v_str_497_, v___x_498_);
if (v___x_499_ == 0)
{
return v___y_493_;
}
else
{
return v_suppressElabErrors_494_;
}
}
else
{
return v___y_493_;
}
}
else
{
return v___y_493_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___boxed(lean_object* v___y_500_, lean_object* v_suppressElabErrors_501_, lean_object* v_x_502_){
_start:
{
uint8_t v___y_39537__boxed_503_; uint8_t v_suppressElabErrors_boxed_504_; uint8_t v_res_505_; lean_object* v_r_506_; 
v___y_39537__boxed_503_ = lean_unbox(v___y_500_);
v_suppressElabErrors_boxed_504_ = lean_unbox(v_suppressElabErrors_501_);
v_res_505_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0(v___y_39537__boxed_503_, v_suppressElabErrors_boxed_504_, v_x_502_);
lean_dec(v_x_502_);
v_r_506_ = lean_box(v_res_505_);
return v_r_506_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__0(void){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_507_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1(void){
_start:
{
lean_object* v___x_508_; lean_object* v___x_509_; 
v___x_508_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__0);
v___x_509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
return v___x_509_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__2(void){
_start:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_510_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1);
v___x_511_ = lean_unsigned_to_nat(0u);
v___x_512_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_512_, 0, v___x_511_);
lean_ctor_set(v___x_512_, 1, v___x_511_);
lean_ctor_set(v___x_512_, 2, v___x_511_);
lean_ctor_set(v___x_512_, 3, v___x_511_);
lean_ctor_set(v___x_512_, 4, v___x_510_);
lean_ctor_set(v___x_512_, 5, v___x_510_);
lean_ctor_set(v___x_512_, 6, v___x_510_);
lean_ctor_set(v___x_512_, 7, v___x_510_);
lean_ctor_set(v___x_512_, 8, v___x_510_);
lean_ctor_set(v___x_512_, 9, v___x_510_);
return v___x_512_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__3(void){
_start:
{
lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; 
v___x_513_ = lean_unsigned_to_nat(32u);
v___x_514_ = lean_mk_empty_array_with_capacity(v___x_513_);
v___x_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_515_, 0, v___x_514_);
return v___x_515_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__4(void){
_start:
{
size_t v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_516_ = ((size_t)5ULL);
v___x_517_ = lean_unsigned_to_nat(0u);
v___x_518_ = lean_unsigned_to_nat(32u);
v___x_519_ = lean_mk_empty_array_with_capacity(v___x_518_);
v___x_520_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__3);
v___x_521_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_521_, 0, v___x_520_);
lean_ctor_set(v___x_521_, 1, v___x_519_);
lean_ctor_set(v___x_521_, 2, v___x_517_);
lean_ctor_set(v___x_521_, 3, v___x_517_);
lean_ctor_set_usize(v___x_521_, 4, v___x_516_);
return v___x_521_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__5(void){
_start:
{
lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_522_ = lean_box(1);
v___x_523_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__4);
v___x_524_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__1);
v___x_525_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
lean_ctor_set(v___x_525_, 1, v___x_523_);
lean_ctor_set(v___x_525_, 2, v___x_522_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg(lean_object* v_msgData_526_, lean_object* v___y_527_){
_start:
{
lean_object* v___x_529_; lean_object* v_env_530_; lean_object* v___x_531_; lean_object* v_scopes_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v_opts_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_529_ = lean_st_ref_get(v___y_527_);
v_env_530_ = lean_ctor_get(v___x_529_, 0);
lean_inc_ref(v_env_530_);
lean_dec(v___x_529_);
v___x_531_ = lean_st_ref_get(v___y_527_);
v_scopes_532_ = lean_ctor_get(v___x_531_, 2);
lean_inc(v_scopes_532_);
lean_dec(v___x_531_);
v___x_533_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_534_ = l_List_head_x21___redArg(v___x_533_, v_scopes_532_);
lean_dec(v_scopes_532_);
v_opts_535_ = lean_ctor_get(v___x_534_, 1);
lean_inc_ref(v_opts_535_);
lean_dec(v___x_534_);
v___x_536_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__2);
v___x_537_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___closed__5);
v___x_538_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_538_, 0, v_env_530_);
lean_ctor_set(v___x_538_, 1, v___x_536_);
lean_ctor_set(v___x_538_, 2, v___x_537_);
lean_ctor_set(v___x_538_, 3, v_opts_535_);
v___x_539_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_538_);
lean_ctor_set(v___x_539_, 1, v_msgData_526_);
v___x_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg___boxed(lean_object* v_msgData_541_, lean_object* v___y_542_, lean_object* v___y_543_){
_start:
{
lean_object* v_res_544_; 
v_res_544_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg(v_msgData_541_, v___y_542_);
lean_dec(v___y_542_);
return v_res_544_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__21(lean_object* v_opts_545_, lean_object* v_opt_546_){
_start:
{
lean_object* v_name_547_; lean_object* v_defValue_548_; lean_object* v_map_549_; lean_object* v___x_550_; 
v_name_547_ = lean_ctor_get(v_opt_546_, 0);
v_defValue_548_ = lean_ctor_get(v_opt_546_, 1);
v_map_549_ = lean_ctor_get(v_opts_545_, 0);
v___x_550_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_549_, v_name_547_);
if (lean_obj_tag(v___x_550_) == 0)
{
uint8_t v___x_551_; 
v___x_551_ = lean_unbox(v_defValue_548_);
return v___x_551_;
}
else
{
lean_object* v_val_552_; 
v_val_552_ = lean_ctor_get(v___x_550_, 0);
lean_inc(v_val_552_);
lean_dec_ref_known(v___x_550_, 1);
if (lean_obj_tag(v_val_552_) == 1)
{
uint8_t v_v_553_; 
v_v_553_ = lean_ctor_get_uint8(v_val_552_, 0);
lean_dec_ref_known(v_val_552_, 0);
return v_v_553_;
}
else
{
uint8_t v___x_554_; 
lean_dec(v_val_552_);
v___x_554_ = lean_unbox(v_defValue_548_);
return v___x_554_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__21___boxed(lean_object* v_opts_555_, lean_object* v_opt_556_){
_start:
{
uint8_t v_res_557_; lean_object* v_r_558_; 
v_res_557_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__21(v_opts_555_, v_opt_556_);
lean_dec_ref(v_opt_556_);
lean_dec_ref(v_opts_555_);
v_r_558_ = lean_box(v_res_557_);
return v_r_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18(lean_object* v_ref_560_, lean_object* v_msgData_561_, uint8_t v_severity_562_, uint8_t v_isSilent_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
uint8_t v___y_568_; lean_object* v___y_569_; lean_object* v___y_570_; lean_object* v___y_571_; uint8_t v___y_572_; lean_object* v___y_573_; lean_object* v___y_574_; lean_object* v___y_575_; uint8_t v___y_632_; uint8_t v___y_633_; lean_object* v___y_634_; uint8_t v___y_635_; lean_object* v___y_636_; uint8_t v___y_660_; uint8_t v___y_661_; lean_object* v___y_662_; uint8_t v___y_663_; lean_object* v___y_664_; uint8_t v___y_668_; uint8_t v___y_669_; uint8_t v___y_670_; uint8_t v___x_685_; uint8_t v___y_687_; uint8_t v___y_688_; uint8_t v___y_689_; uint8_t v___y_691_; uint8_t v___x_703_; 
v___x_685_ = 2;
v___x_703_ = l_Lean_instBEqMessageSeverity_beq(v_severity_562_, v___x_685_);
if (v___x_703_ == 0)
{
v___y_691_ = v___x_703_;
goto v___jp_690_;
}
else
{
uint8_t v___x_704_; 
lean_inc_ref(v_msgData_561_);
v___x_704_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_561_);
v___y_691_ = v___x_704_;
goto v___jp_690_;
}
v___jp_567_:
{
lean_object* v___x_576_; 
v___x_576_ = l_Lean_Elab_Command_getScope___redArg(v___y_575_);
if (lean_obj_tag(v___x_576_) == 0)
{
lean_object* v_a_577_; lean_object* v___x_578_; 
v_a_577_ = lean_ctor_get(v___x_576_, 0);
lean_inc(v_a_577_);
lean_dec_ref_known(v___x_576_, 1);
v___x_578_ = l_Lean_Elab_Command_getScope___redArg(v___y_575_);
if (lean_obj_tag(v___x_578_) == 0)
{
lean_object* v_a_579_; lean_object* v___x_581_; uint8_t v_isShared_582_; uint8_t v_isSharedCheck_614_; 
v_a_579_ = lean_ctor_get(v___x_578_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_578_);
if (v_isSharedCheck_614_ == 0)
{
v___x_581_ = v___x_578_;
v_isShared_582_ = v_isSharedCheck_614_;
goto v_resetjp_580_;
}
else
{
lean_inc(v_a_579_);
lean_dec(v___x_578_);
v___x_581_ = lean_box(0);
v_isShared_582_ = v_isSharedCheck_614_;
goto v_resetjp_580_;
}
v_resetjp_580_:
{
lean_object* v___x_583_; lean_object* v_currNamespace_584_; lean_object* v_openDecls_585_; lean_object* v_env_586_; lean_object* v_messages_587_; lean_object* v_scopes_588_; lean_object* v_usedQuotCtxts_589_; lean_object* v_nextMacroScope_590_; lean_object* v_maxRecDepth_591_; lean_object* v_ngen_592_; lean_object* v_auxDeclNGen_593_; lean_object* v_infoState_594_; lean_object* v_traceState_595_; lean_object* v_snapshotTasks_596_; lean_object* v_prevLinterStates_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_613_; 
v___x_583_ = lean_st_ref_take(v___y_575_);
v_currNamespace_584_ = lean_ctor_get(v_a_577_, 2);
lean_inc(v_currNamespace_584_);
lean_dec(v_a_577_);
v_openDecls_585_ = lean_ctor_get(v_a_579_, 3);
lean_inc(v_openDecls_585_);
lean_dec(v_a_579_);
v_env_586_ = lean_ctor_get(v___x_583_, 0);
v_messages_587_ = lean_ctor_get(v___x_583_, 1);
v_scopes_588_ = lean_ctor_get(v___x_583_, 2);
v_usedQuotCtxts_589_ = lean_ctor_get(v___x_583_, 3);
v_nextMacroScope_590_ = lean_ctor_get(v___x_583_, 4);
v_maxRecDepth_591_ = lean_ctor_get(v___x_583_, 5);
v_ngen_592_ = lean_ctor_get(v___x_583_, 6);
v_auxDeclNGen_593_ = lean_ctor_get(v___x_583_, 7);
v_infoState_594_ = lean_ctor_get(v___x_583_, 8);
v_traceState_595_ = lean_ctor_get(v___x_583_, 9);
v_snapshotTasks_596_ = lean_ctor_get(v___x_583_, 10);
v_prevLinterStates_597_ = lean_ctor_get(v___x_583_, 11);
v_isSharedCheck_613_ = !lean_is_exclusive(v___x_583_);
if (v_isSharedCheck_613_ == 0)
{
v___x_599_ = v___x_583_;
v_isShared_600_ = v_isSharedCheck_613_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_prevLinterStates_597_);
lean_inc(v_snapshotTasks_596_);
lean_inc(v_traceState_595_);
lean_inc(v_infoState_594_);
lean_inc(v_auxDeclNGen_593_);
lean_inc(v_ngen_592_);
lean_inc(v_maxRecDepth_591_);
lean_inc(v_nextMacroScope_590_);
lean_inc(v_usedQuotCtxts_589_);
lean_inc(v_scopes_588_);
lean_inc(v_messages_587_);
lean_inc(v_env_586_);
lean_dec(v___x_583_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_613_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_606_; 
v___x_601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_601_, 0, v_currNamespace_584_);
lean_ctor_set(v___x_601_, 1, v_openDecls_585_);
v___x_602_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_602_, 0, v___x_601_);
lean_ctor_set(v___x_602_, 1, v___y_570_);
lean_inc_ref(v___y_569_);
lean_inc_ref(v___y_571_);
v___x_603_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_603_, 0, v___y_571_);
lean_ctor_set(v___x_603_, 1, v___y_574_);
lean_ctor_set(v___x_603_, 2, v___y_573_);
lean_ctor_set(v___x_603_, 3, v___y_569_);
lean_ctor_set(v___x_603_, 4, v___x_602_);
lean_ctor_set_uint8(v___x_603_, sizeof(void*)*5, v___y_572_);
lean_ctor_set_uint8(v___x_603_, sizeof(void*)*5 + 1, v___y_568_);
lean_ctor_set_uint8(v___x_603_, sizeof(void*)*5 + 2, v_isSilent_563_);
v___x_604_ = l_Lean_MessageLog_add(v___x_603_, v_messages_587_);
if (v_isShared_600_ == 0)
{
lean_ctor_set(v___x_599_, 1, v___x_604_);
v___x_606_ = v___x_599_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_612_; 
v_reuseFailAlloc_612_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_612_, 0, v_env_586_);
lean_ctor_set(v_reuseFailAlloc_612_, 1, v___x_604_);
lean_ctor_set(v_reuseFailAlloc_612_, 2, v_scopes_588_);
lean_ctor_set(v_reuseFailAlloc_612_, 3, v_usedQuotCtxts_589_);
lean_ctor_set(v_reuseFailAlloc_612_, 4, v_nextMacroScope_590_);
lean_ctor_set(v_reuseFailAlloc_612_, 5, v_maxRecDepth_591_);
lean_ctor_set(v_reuseFailAlloc_612_, 6, v_ngen_592_);
lean_ctor_set(v_reuseFailAlloc_612_, 7, v_auxDeclNGen_593_);
lean_ctor_set(v_reuseFailAlloc_612_, 8, v_infoState_594_);
lean_ctor_set(v_reuseFailAlloc_612_, 9, v_traceState_595_);
lean_ctor_set(v_reuseFailAlloc_612_, 10, v_snapshotTasks_596_);
lean_ctor_set(v_reuseFailAlloc_612_, 11, v_prevLinterStates_597_);
v___x_606_ = v_reuseFailAlloc_612_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_610_; 
v___x_607_ = lean_st_ref_set(v___y_575_, v___x_606_);
v___x_608_ = lean_box(0);
if (v_isShared_582_ == 0)
{
lean_ctor_set(v___x_581_, 0, v___x_608_);
v___x_610_ = v___x_581_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_611_; 
v_reuseFailAlloc_611_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_611_, 0, v___x_608_);
v___x_610_ = v_reuseFailAlloc_611_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
return v___x_610_;
}
}
}
}
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
lean_dec(v_a_577_);
lean_dec_ref(v___y_574_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_570_);
v_a_615_ = lean_ctor_get(v___x_578_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_578_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_578_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_578_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_620_; 
if (v_isShared_618_ == 0)
{
v___x_620_ = v___x_617_;
goto v_reusejp_619_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_a_615_);
v___x_620_ = v_reuseFailAlloc_621_;
goto v_reusejp_619_;
}
v_reusejp_619_:
{
return v___x_620_;
}
}
}
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_dec_ref(v___y_574_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_570_);
v_a_623_ = lean_ctor_get(v___x_576_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_576_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_576_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_576_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
v___jp_631_:
{
lean_object* v_fileName_637_; lean_object* v_fileMap_638_; uint8_t v_suppressElabErrors_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v_a_642_; lean_object* v___x_644_; uint8_t v_isShared_645_; uint8_t v_isSharedCheck_658_; 
v_fileName_637_ = lean_ctor_get(v___y_564_, 0);
v_fileMap_638_ = lean_ctor_get(v___y_564_, 1);
v_suppressElabErrors_639_ = lean_ctor_get_uint8(v___y_564_, sizeof(void*)*10);
v___x_640_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_561_);
v___x_641_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg(v___x_640_, v___y_565_);
v_a_642_ = lean_ctor_get(v___x_641_, 0);
v_isSharedCheck_658_ = !lean_is_exclusive(v___x_641_);
if (v_isSharedCheck_658_ == 0)
{
v___x_644_ = v___x_641_;
v_isShared_645_ = v_isSharedCheck_658_;
goto v_resetjp_643_;
}
else
{
lean_inc(v_a_642_);
lean_dec(v___x_641_);
v___x_644_ = lean_box(0);
v_isShared_645_ = v_isSharedCheck_658_;
goto v_resetjp_643_;
}
v_resetjp_643_:
{
lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
lean_inc_ref_n(v_fileMap_638_, 2);
v___x_646_ = l_Lean_FileMap_toPosition(v_fileMap_638_, v___y_634_);
lean_dec(v___y_634_);
v___x_647_ = l_Lean_FileMap_toPosition(v_fileMap_638_, v___y_636_);
lean_dec(v___y_636_);
v___x_648_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_648_, 0, v___x_647_);
v___x_649_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___closed__0));
if (v_suppressElabErrors_639_ == 0)
{
lean_del_object(v___x_644_);
v___y_568_ = v___y_633_;
v___y_569_ = v___x_649_;
v___y_570_ = v_a_642_;
v___y_571_ = v_fileName_637_;
v___y_572_ = v___y_635_;
v___y_573_ = v___x_648_;
v___y_574_ = v___x_646_;
v___y_575_ = v___y_565_;
goto v___jp_567_;
}
else
{
lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___f_652_; uint8_t v___x_653_; 
v___x_650_ = lean_box(v___y_632_);
v___x_651_ = lean_box(v_suppressElabErrors_639_);
v___f_652_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___lam__0___boxed), 3, 2);
lean_closure_set(v___f_652_, 0, v___x_650_);
lean_closure_set(v___f_652_, 1, v___x_651_);
lean_inc(v_a_642_);
v___x_653_ = l_Lean_MessageData_hasTag(v___f_652_, v_a_642_);
if (v___x_653_ == 0)
{
lean_object* v___x_654_; lean_object* v___x_656_; 
lean_dec_ref_known(v___x_648_, 1);
lean_dec_ref(v___x_646_);
lean_dec(v_a_642_);
v___x_654_ = lean_box(0);
if (v_isShared_645_ == 0)
{
lean_ctor_set(v___x_644_, 0, v___x_654_);
v___x_656_ = v___x_644_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_657_; 
v_reuseFailAlloc_657_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_657_, 0, v___x_654_);
v___x_656_ = v_reuseFailAlloc_657_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
return v___x_656_;
}
}
else
{
lean_del_object(v___x_644_);
v___y_568_ = v___y_633_;
v___y_569_ = v___x_649_;
v___y_570_ = v_a_642_;
v___y_571_ = v_fileName_637_;
v___y_572_ = v___y_635_;
v___y_573_ = v___x_648_;
v___y_574_ = v___x_646_;
v___y_575_ = v___y_565_;
goto v___jp_567_;
}
}
}
}
v___jp_659_:
{
lean_object* v___x_665_; 
v___x_665_ = l_Lean_Syntax_getTailPos_x3f(v___y_662_, v___y_663_);
lean_dec(v___y_662_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_inc(v___y_664_);
v___y_632_ = v___y_660_;
v___y_633_ = v___y_661_;
v___y_634_ = v___y_664_;
v___y_635_ = v___y_663_;
v___y_636_ = v___y_664_;
goto v___jp_631_;
}
else
{
lean_object* v_val_666_; 
v_val_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc(v_val_666_);
lean_dec_ref_known(v___x_665_, 1);
v___y_632_ = v___y_660_;
v___y_633_ = v___y_661_;
v___y_634_ = v___y_664_;
v___y_635_ = v___y_663_;
v___y_636_ = v_val_666_;
goto v___jp_631_;
}
}
v___jp_667_:
{
lean_object* v___x_671_; 
v___x_671_ = l_Lean_Elab_Command_getRef___redArg(v___y_564_);
if (lean_obj_tag(v___x_671_) == 0)
{
lean_object* v_a_672_; lean_object* v_ref_673_; lean_object* v___x_674_; 
v_a_672_ = lean_ctor_get(v___x_671_, 0);
lean_inc(v_a_672_);
lean_dec_ref_known(v___x_671_, 1);
v_ref_673_ = l_Lean_replaceRef(v_ref_560_, v_a_672_);
lean_dec(v_a_672_);
v___x_674_ = l_Lean_Syntax_getPos_x3f(v_ref_673_, v___y_669_);
if (lean_obj_tag(v___x_674_) == 0)
{
lean_object* v___x_675_; 
v___x_675_ = lean_unsigned_to_nat(0u);
v___y_660_ = v___y_668_;
v___y_661_ = v___y_670_;
v___y_662_ = v_ref_673_;
v___y_663_ = v___y_669_;
v___y_664_ = v___x_675_;
goto v___jp_659_;
}
else
{
lean_object* v_val_676_; 
v_val_676_ = lean_ctor_get(v___x_674_, 0);
lean_inc(v_val_676_);
lean_dec_ref_known(v___x_674_, 1);
v___y_660_ = v___y_668_;
v___y_661_ = v___y_670_;
v___y_662_ = v_ref_673_;
v___y_663_ = v___y_669_;
v___y_664_ = v_val_676_;
goto v___jp_659_;
}
}
else
{
lean_object* v_a_677_; lean_object* v___x_679_; uint8_t v_isShared_680_; uint8_t v_isSharedCheck_684_; 
lean_dec_ref(v_msgData_561_);
v_a_677_ = lean_ctor_get(v___x_671_, 0);
v_isSharedCheck_684_ = !lean_is_exclusive(v___x_671_);
if (v_isSharedCheck_684_ == 0)
{
v___x_679_ = v___x_671_;
v_isShared_680_ = v_isSharedCheck_684_;
goto v_resetjp_678_;
}
else
{
lean_inc(v_a_677_);
lean_dec(v___x_671_);
v___x_679_ = lean_box(0);
v_isShared_680_ = v_isSharedCheck_684_;
goto v_resetjp_678_;
}
v_resetjp_678_:
{
lean_object* v___x_682_; 
if (v_isShared_680_ == 0)
{
v___x_682_ = v___x_679_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_683_; 
v_reuseFailAlloc_683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_683_, 0, v_a_677_);
v___x_682_ = v_reuseFailAlloc_683_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
return v___x_682_;
}
}
}
}
v___jp_686_:
{
if (v___y_689_ == 0)
{
v___y_668_ = v___y_687_;
v___y_669_ = v___y_688_;
v___y_670_ = v_severity_562_;
goto v___jp_667_;
}
else
{
v___y_668_ = v___y_687_;
v___y_669_ = v___y_688_;
v___y_670_ = v___x_685_;
goto v___jp_667_;
}
}
v___jp_690_:
{
if (v___y_691_ == 0)
{
lean_object* v___x_692_; lean_object* v_scopes_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v_opts_696_; uint8_t v___x_697_; uint8_t v___x_698_; 
v___x_692_ = lean_st_ref_get(v___y_565_);
v_scopes_693_ = lean_ctor_get(v___x_692_, 2);
lean_inc(v_scopes_693_);
lean_dec(v___x_692_);
v___x_694_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_695_ = l_List_head_x21___redArg(v___x_694_, v_scopes_693_);
lean_dec(v_scopes_693_);
v_opts_696_ = lean_ctor_get(v___x_695_, 1);
lean_inc_ref(v_opts_696_);
lean_dec(v___x_695_);
v___x_697_ = 1;
v___x_698_ = l_Lean_instBEqMessageSeverity_beq(v_severity_562_, v___x_697_);
if (v___x_698_ == 0)
{
lean_dec_ref(v_opts_696_);
v___y_687_ = v___y_691_;
v___y_688_ = v___y_691_;
v___y_689_ = v___x_698_;
goto v___jp_686_;
}
else
{
lean_object* v___x_699_; uint8_t v___x_700_; 
v___x_699_ = l_Lean_warningAsError;
v___x_700_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__21(v_opts_696_, v___x_699_);
lean_dec_ref(v_opts_696_);
v___y_687_ = v___y_691_;
v___y_688_ = v___y_691_;
v___y_689_ = v___x_700_;
goto v___jp_686_;
}
}
else
{
lean_object* v___x_701_; lean_object* v___x_702_; 
lean_dec_ref(v_msgData_561_);
v___x_701_ = lean_box(0);
v___x_702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
return v___x_702_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___boxed(lean_object* v_ref_705_, lean_object* v_msgData_706_, lean_object* v_severity_707_, lean_object* v_isSilent_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
uint8_t v_severity_boxed_712_; uint8_t v_isSilent_boxed_713_; lean_object* v_res_714_; 
v_severity_boxed_712_ = lean_unbox(v_severity_707_);
v_isSilent_boxed_713_ = lean_unbox(v_isSilent_708_);
v_res_714_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18(v_ref_705_, v_msgData_706_, v_severity_boxed_712_, v_isSilent_boxed_713_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v_ref_705_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16_spec__23(lean_object* v_msgData_715_, uint8_t v_severity_716_, uint8_t v_isSilent_717_, lean_object* v___y_718_, lean_object* v___y_719_){
_start:
{
lean_object* v___x_721_; 
v___x_721_ = l_Lean_Elab_Command_getRef___redArg(v___y_718_);
if (lean_obj_tag(v___x_721_) == 0)
{
lean_object* v_a_722_; lean_object* v___x_723_; 
v_a_722_ = lean_ctor_get(v___x_721_, 0);
lean_inc(v_a_722_);
lean_dec_ref_known(v___x_721_, 1);
v___x_723_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18(v_a_722_, v_msgData_715_, v_severity_716_, v_isSilent_717_, v___y_718_, v___y_719_);
lean_dec(v_a_722_);
return v___x_723_;
}
else
{
lean_object* v_a_724_; lean_object* v___x_726_; uint8_t v_isShared_727_; uint8_t v_isSharedCheck_731_; 
lean_dec_ref(v_msgData_715_);
v_a_724_ = lean_ctor_get(v___x_721_, 0);
v_isSharedCheck_731_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_731_ == 0)
{
v___x_726_ = v___x_721_;
v_isShared_727_ = v_isSharedCheck_731_;
goto v_resetjp_725_;
}
else
{
lean_inc(v_a_724_);
lean_dec(v___x_721_);
v___x_726_ = lean_box(0);
v_isShared_727_ = v_isSharedCheck_731_;
goto v_resetjp_725_;
}
v_resetjp_725_:
{
lean_object* v___x_729_; 
if (v_isShared_727_ == 0)
{
v___x_729_ = v___x_726_;
goto v_reusejp_728_;
}
else
{
lean_object* v_reuseFailAlloc_730_; 
v_reuseFailAlloc_730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_730_, 0, v_a_724_);
v___x_729_ = v_reuseFailAlloc_730_;
goto v_reusejp_728_;
}
v_reusejp_728_:
{
return v___x_729_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16_spec__23___boxed(lean_object* v_msgData_732_, lean_object* v_severity_733_, lean_object* v_isSilent_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_){
_start:
{
uint8_t v_severity_boxed_738_; uint8_t v_isSilent_boxed_739_; lean_object* v_res_740_; 
v_severity_boxed_738_ = lean_unbox(v_severity_733_);
v_isSilent_boxed_739_ = lean_unbox(v_isSilent_734_);
v_res_740_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16_spec__23(v_msgData_732_, v_severity_boxed_738_, v_isSilent_boxed_739_, v___y_735_, v___y_736_);
lean_dec(v___y_736_);
lean_dec_ref(v___y_735_);
return v_res_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16(lean_object* v_msgData_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
uint8_t v___x_745_; uint8_t v___x_746_; lean_object* v___x_747_; 
v___x_745_ = 0;
v___x_746_ = 0;
v___x_747_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16_spec__23(v_msgData_741_, v___x_745_, v___x_746_, v___y_742_, v___y_743_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16___boxed(lean_object* v_msgData_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_){
_start:
{
lean_object* v_res_752_; 
v_res_752_ = lp_mathlib_Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16(v_msgData_748_, v___y_749_, v___y_750_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(lean_object* v_k_753_, lean_object* v_t_754_){
_start:
{
if (lean_obj_tag(v_t_754_) == 0)
{
lean_object* v_k_755_; lean_object* v_v_756_; lean_object* v_l_757_; lean_object* v_r_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_1412_; 
v_k_755_ = lean_ctor_get(v_t_754_, 1);
v_v_756_ = lean_ctor_get(v_t_754_, 2);
v_l_757_ = lean_ctor_get(v_t_754_, 3);
v_r_758_ = lean_ctor_get(v_t_754_, 4);
v_isSharedCheck_1412_ = !lean_is_exclusive(v_t_754_);
if (v_isSharedCheck_1412_ == 0)
{
lean_object* v_unused_1413_; 
v_unused_1413_ = lean_ctor_get(v_t_754_, 0);
lean_dec(v_unused_1413_);
v___x_760_ = v_t_754_;
v_isShared_761_ = v_isSharedCheck_1412_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_r_758_);
lean_inc(v_l_757_);
lean_inc(v_v_756_);
lean_inc(v_k_755_);
lean_dec(v_t_754_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_1412_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
uint8_t v___x_762_; 
v___x_762_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_753_, v_k_755_);
switch(v___x_762_)
{
case 0:
{
lean_object* v_impl_763_; lean_object* v___x_764_; 
v_impl_763_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(v_k_753_, v_l_757_);
v___x_764_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_763_) == 0)
{
if (lean_obj_tag(v_r_758_) == 0)
{
lean_object* v_size_765_; lean_object* v_size_766_; lean_object* v_k_767_; lean_object* v_v_768_; lean_object* v_l_769_; lean_object* v_r_770_; lean_object* v___x_771_; lean_object* v___x_772_; uint8_t v___x_773_; 
v_size_765_ = lean_ctor_get(v_impl_763_, 0);
lean_inc(v_size_765_);
v_size_766_ = lean_ctor_get(v_r_758_, 0);
v_k_767_ = lean_ctor_get(v_r_758_, 1);
v_v_768_ = lean_ctor_get(v_r_758_, 2);
v_l_769_ = lean_ctor_get(v_r_758_, 3);
lean_inc(v_l_769_);
v_r_770_ = lean_ctor_get(v_r_758_, 4);
v___x_771_ = lean_unsigned_to_nat(3u);
v___x_772_ = lean_nat_mul(v___x_771_, v_size_765_);
v___x_773_ = lean_nat_dec_lt(v___x_772_, v_size_766_);
lean_dec(v___x_772_);
if (v___x_773_ == 0)
{
lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_777_; 
lean_dec(v_l_769_);
v___x_774_ = lean_nat_add(v___x_764_, v_size_765_);
lean_dec(v_size_765_);
v___x_775_ = lean_nat_add(v___x_774_, v_size_766_);
lean_dec(v___x_774_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 3, v_impl_763_);
lean_ctor_set(v___x_760_, 0, v___x_775_);
v___x_777_ = v___x_760_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v___x_775_);
lean_ctor_set(v_reuseFailAlloc_778_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_778_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_778_, 3, v_impl_763_);
lean_ctor_set(v_reuseFailAlloc_778_, 4, v_r_758_);
v___x_777_ = v_reuseFailAlloc_778_;
goto v_reusejp_776_;
}
v_reusejp_776_:
{
return v___x_777_;
}
}
else
{
lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_842_; 
lean_inc(v_r_770_);
lean_inc(v_v_768_);
lean_inc(v_k_767_);
lean_inc(v_size_766_);
v_isSharedCheck_842_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_842_ == 0)
{
lean_object* v_unused_843_; lean_object* v_unused_844_; lean_object* v_unused_845_; lean_object* v_unused_846_; lean_object* v_unused_847_; 
v_unused_843_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_843_);
v_unused_844_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_844_);
v_unused_845_ = lean_ctor_get(v_r_758_, 2);
lean_dec(v_unused_845_);
v_unused_846_ = lean_ctor_get(v_r_758_, 1);
lean_dec(v_unused_846_);
v_unused_847_ = lean_ctor_get(v_r_758_, 0);
lean_dec(v_unused_847_);
v___x_780_ = v_r_758_;
v_isShared_781_ = v_isSharedCheck_842_;
goto v_resetjp_779_;
}
else
{
lean_dec(v_r_758_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_842_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v_size_782_; lean_object* v_k_783_; lean_object* v_v_784_; lean_object* v_l_785_; lean_object* v_r_786_; lean_object* v_size_787_; lean_object* v___x_788_; lean_object* v___x_789_; uint8_t v___x_790_; 
v_size_782_ = lean_ctor_get(v_l_769_, 0);
v_k_783_ = lean_ctor_get(v_l_769_, 1);
v_v_784_ = lean_ctor_get(v_l_769_, 2);
v_l_785_ = lean_ctor_get(v_l_769_, 3);
v_r_786_ = lean_ctor_get(v_l_769_, 4);
v_size_787_ = lean_ctor_get(v_r_770_, 0);
v___x_788_ = lean_unsigned_to_nat(2u);
v___x_789_ = lean_nat_mul(v___x_788_, v_size_787_);
v___x_790_ = lean_nat_dec_lt(v_size_782_, v___x_789_);
lean_dec(v___x_789_);
if (v___x_790_ == 0)
{
lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_818_; 
lean_inc(v_r_786_);
lean_inc(v_l_785_);
lean_inc(v_v_784_);
lean_inc(v_k_783_);
v_isSharedCheck_818_ = !lean_is_exclusive(v_l_769_);
if (v_isSharedCheck_818_ == 0)
{
lean_object* v_unused_819_; lean_object* v_unused_820_; lean_object* v_unused_821_; lean_object* v_unused_822_; lean_object* v_unused_823_; 
v_unused_819_ = lean_ctor_get(v_l_769_, 4);
lean_dec(v_unused_819_);
v_unused_820_ = lean_ctor_get(v_l_769_, 3);
lean_dec(v_unused_820_);
v_unused_821_ = lean_ctor_get(v_l_769_, 2);
lean_dec(v_unused_821_);
v_unused_822_ = lean_ctor_get(v_l_769_, 1);
lean_dec(v_unused_822_);
v_unused_823_ = lean_ctor_get(v_l_769_, 0);
lean_dec(v_unused_823_);
v___x_792_ = v_l_769_;
v_isShared_793_ = v_isSharedCheck_818_;
goto v_resetjp_791_;
}
else
{
lean_dec(v_l_769_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_818_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___y_797_; lean_object* v___y_798_; lean_object* v___y_799_; lean_object* v___y_808_; 
v___x_794_ = lean_nat_add(v___x_764_, v_size_765_);
lean_dec(v_size_765_);
v___x_795_ = lean_nat_add(v___x_794_, v_size_766_);
lean_dec(v_size_766_);
if (lean_obj_tag(v_l_785_) == 0)
{
lean_object* v_size_816_; 
v_size_816_ = lean_ctor_get(v_l_785_, 0);
lean_inc(v_size_816_);
v___y_808_ = v_size_816_;
goto v___jp_807_;
}
else
{
lean_object* v___x_817_; 
v___x_817_ = lean_unsigned_to_nat(0u);
v___y_808_ = v___x_817_;
goto v___jp_807_;
}
v___jp_796_:
{
lean_object* v___x_800_; lean_object* v___x_802_; 
v___x_800_ = lean_nat_add(v___y_798_, v___y_799_);
lean_dec(v___y_799_);
lean_dec(v___y_798_);
if (v_isShared_793_ == 0)
{
lean_ctor_set(v___x_792_, 4, v_r_770_);
lean_ctor_set(v___x_792_, 3, v_r_786_);
lean_ctor_set(v___x_792_, 2, v_v_768_);
lean_ctor_set(v___x_792_, 1, v_k_767_);
lean_ctor_set(v___x_792_, 0, v___x_800_);
v___x_802_ = v___x_792_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_806_; 
v_reuseFailAlloc_806_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_806_, 0, v___x_800_);
lean_ctor_set(v_reuseFailAlloc_806_, 1, v_k_767_);
lean_ctor_set(v_reuseFailAlloc_806_, 2, v_v_768_);
lean_ctor_set(v_reuseFailAlloc_806_, 3, v_r_786_);
lean_ctor_set(v_reuseFailAlloc_806_, 4, v_r_770_);
v___x_802_ = v_reuseFailAlloc_806_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
lean_object* v___x_804_; 
if (v_isShared_781_ == 0)
{
lean_ctor_set(v___x_780_, 4, v___x_802_);
lean_ctor_set(v___x_780_, 3, v___y_797_);
lean_ctor_set(v___x_780_, 2, v_v_784_);
lean_ctor_set(v___x_780_, 1, v_k_783_);
lean_ctor_set(v___x_780_, 0, v___x_795_);
v___x_804_ = v___x_780_;
goto v_reusejp_803_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v___x_795_);
lean_ctor_set(v_reuseFailAlloc_805_, 1, v_k_783_);
lean_ctor_set(v_reuseFailAlloc_805_, 2, v_v_784_);
lean_ctor_set(v_reuseFailAlloc_805_, 3, v___y_797_);
lean_ctor_set(v_reuseFailAlloc_805_, 4, v___x_802_);
v___x_804_ = v_reuseFailAlloc_805_;
goto v_reusejp_803_;
}
v_reusejp_803_:
{
return v___x_804_;
}
}
}
v___jp_807_:
{
lean_object* v___x_809_; lean_object* v___x_811_; 
v___x_809_ = lean_nat_add(v___x_794_, v___y_808_);
lean_dec(v___y_808_);
lean_dec(v___x_794_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_l_785_);
lean_ctor_set(v___x_760_, 3, v_impl_763_);
lean_ctor_set(v___x_760_, 0, v___x_809_);
v___x_811_ = v___x_760_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_815_; 
v_reuseFailAlloc_815_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_815_, 0, v___x_809_);
lean_ctor_set(v_reuseFailAlloc_815_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_815_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_815_, 3, v_impl_763_);
lean_ctor_set(v_reuseFailAlloc_815_, 4, v_l_785_);
v___x_811_ = v_reuseFailAlloc_815_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
lean_object* v___x_812_; 
v___x_812_ = lean_nat_add(v___x_764_, v_size_787_);
if (lean_obj_tag(v_r_786_) == 0)
{
lean_object* v_size_813_; 
v_size_813_ = lean_ctor_get(v_r_786_, 0);
lean_inc(v_size_813_);
v___y_797_ = v___x_811_;
v___y_798_ = v___x_812_;
v___y_799_ = v_size_813_;
goto v___jp_796_;
}
else
{
lean_object* v___x_814_; 
v___x_814_ = lean_unsigned_to_nat(0u);
v___y_797_ = v___x_811_;
v___y_798_ = v___x_812_;
v___y_799_ = v___x_814_;
goto v___jp_796_;
}
}
}
}
}
else
{
lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_828_; 
lean_del_object(v___x_760_);
v___x_824_ = lean_nat_add(v___x_764_, v_size_765_);
lean_dec(v_size_765_);
v___x_825_ = lean_nat_add(v___x_824_, v_size_766_);
lean_dec(v_size_766_);
v___x_826_ = lean_nat_add(v___x_824_, v_size_782_);
lean_dec(v___x_824_);
lean_inc_ref(v_impl_763_);
if (v_isShared_781_ == 0)
{
lean_ctor_set(v___x_780_, 4, v_l_769_);
lean_ctor_set(v___x_780_, 3, v_impl_763_);
lean_ctor_set(v___x_780_, 2, v_v_756_);
lean_ctor_set(v___x_780_, 1, v_k_755_);
lean_ctor_set(v___x_780_, 0, v___x_826_);
v___x_828_ = v___x_780_;
goto v_reusejp_827_;
}
else
{
lean_object* v_reuseFailAlloc_841_; 
v_reuseFailAlloc_841_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_841_, 0, v___x_826_);
lean_ctor_set(v_reuseFailAlloc_841_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_841_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_841_, 3, v_impl_763_);
lean_ctor_set(v_reuseFailAlloc_841_, 4, v_l_769_);
v___x_828_ = v_reuseFailAlloc_841_;
goto v_reusejp_827_;
}
v_reusejp_827_:
{
lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_835_; 
v_isSharedCheck_835_ = !lean_is_exclusive(v_impl_763_);
if (v_isSharedCheck_835_ == 0)
{
lean_object* v_unused_836_; lean_object* v_unused_837_; lean_object* v_unused_838_; lean_object* v_unused_839_; lean_object* v_unused_840_; 
v_unused_836_ = lean_ctor_get(v_impl_763_, 4);
lean_dec(v_unused_836_);
v_unused_837_ = lean_ctor_get(v_impl_763_, 3);
lean_dec(v_unused_837_);
v_unused_838_ = lean_ctor_get(v_impl_763_, 2);
lean_dec(v_unused_838_);
v_unused_839_ = lean_ctor_get(v_impl_763_, 1);
lean_dec(v_unused_839_);
v_unused_840_ = lean_ctor_get(v_impl_763_, 0);
lean_dec(v_unused_840_);
v___x_830_ = v_impl_763_;
v_isShared_831_ = v_isSharedCheck_835_;
goto v_resetjp_829_;
}
else
{
lean_dec(v_impl_763_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_835_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
lean_object* v___x_833_; 
if (v_isShared_831_ == 0)
{
lean_ctor_set(v___x_830_, 4, v_r_770_);
lean_ctor_set(v___x_830_, 3, v___x_828_);
lean_ctor_set(v___x_830_, 2, v_v_768_);
lean_ctor_set(v___x_830_, 1, v_k_767_);
lean_ctor_set(v___x_830_, 0, v___x_825_);
v___x_833_ = v___x_830_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_825_);
lean_ctor_set(v_reuseFailAlloc_834_, 1, v_k_767_);
lean_ctor_set(v_reuseFailAlloc_834_, 2, v_v_768_);
lean_ctor_set(v_reuseFailAlloc_834_, 3, v___x_828_);
lean_ctor_set(v_reuseFailAlloc_834_, 4, v_r_770_);
v___x_833_ = v_reuseFailAlloc_834_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
return v___x_833_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_848_; lean_object* v___x_849_; lean_object* v___x_851_; 
v_size_848_ = lean_ctor_get(v_impl_763_, 0);
lean_inc(v_size_848_);
v___x_849_ = lean_nat_add(v___x_764_, v_size_848_);
lean_dec(v_size_848_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 3, v_impl_763_);
lean_ctor_set(v___x_760_, 0, v___x_849_);
v___x_851_ = v___x_760_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v___x_849_);
lean_ctor_set(v_reuseFailAlloc_852_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_852_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_852_, 3, v_impl_763_);
lean_ctor_set(v_reuseFailAlloc_852_, 4, v_r_758_);
v___x_851_ = v_reuseFailAlloc_852_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
return v___x_851_;
}
}
}
else
{
if (lean_obj_tag(v_r_758_) == 0)
{
lean_object* v_l_853_; 
v_l_853_ = lean_ctor_get(v_r_758_, 3);
lean_inc(v_l_853_);
if (lean_obj_tag(v_l_853_) == 0)
{
lean_object* v_r_854_; 
v_r_854_ = lean_ctor_get(v_r_758_, 4);
lean_inc(v_r_854_);
if (lean_obj_tag(v_r_854_) == 0)
{
lean_object* v_size_855_; lean_object* v_k_856_; lean_object* v_v_857_; lean_object* v___x_859_; uint8_t v_isShared_860_; uint8_t v_isSharedCheck_870_; 
v_size_855_ = lean_ctor_get(v_r_758_, 0);
v_k_856_ = lean_ctor_get(v_r_758_, 1);
v_v_857_ = lean_ctor_get(v_r_758_, 2);
v_isSharedCheck_870_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_870_ == 0)
{
lean_object* v_unused_871_; lean_object* v_unused_872_; 
v_unused_871_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_871_);
v_unused_872_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_872_);
v___x_859_ = v_r_758_;
v_isShared_860_ = v_isSharedCheck_870_;
goto v_resetjp_858_;
}
else
{
lean_inc(v_v_857_);
lean_inc(v_k_856_);
lean_inc(v_size_855_);
lean_dec(v_r_758_);
v___x_859_ = lean_box(0);
v_isShared_860_ = v_isSharedCheck_870_;
goto v_resetjp_858_;
}
v_resetjp_858_:
{
lean_object* v_size_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_865_; 
v_size_861_ = lean_ctor_get(v_l_853_, 0);
v___x_862_ = lean_nat_add(v___x_764_, v_size_855_);
lean_dec(v_size_855_);
v___x_863_ = lean_nat_add(v___x_764_, v_size_861_);
if (v_isShared_860_ == 0)
{
lean_ctor_set(v___x_859_, 4, v_l_853_);
lean_ctor_set(v___x_859_, 3, v_impl_763_);
lean_ctor_set(v___x_859_, 2, v_v_756_);
lean_ctor_set(v___x_859_, 1, v_k_755_);
lean_ctor_set(v___x_859_, 0, v___x_863_);
v___x_865_ = v___x_859_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_869_; 
v_reuseFailAlloc_869_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_869_, 0, v___x_863_);
lean_ctor_set(v_reuseFailAlloc_869_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_869_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_869_, 3, v_impl_763_);
lean_ctor_set(v_reuseFailAlloc_869_, 4, v_l_853_);
v___x_865_ = v_reuseFailAlloc_869_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
lean_object* v___x_867_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_r_854_);
lean_ctor_set(v___x_760_, 3, v___x_865_);
lean_ctor_set(v___x_760_, 2, v_v_857_);
lean_ctor_set(v___x_760_, 1, v_k_856_);
lean_ctor_set(v___x_760_, 0, v___x_862_);
v___x_867_ = v___x_760_;
goto v_reusejp_866_;
}
else
{
lean_object* v_reuseFailAlloc_868_; 
v_reuseFailAlloc_868_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_868_, 0, v___x_862_);
lean_ctor_set(v_reuseFailAlloc_868_, 1, v_k_856_);
lean_ctor_set(v_reuseFailAlloc_868_, 2, v_v_857_);
lean_ctor_set(v_reuseFailAlloc_868_, 3, v___x_865_);
lean_ctor_set(v_reuseFailAlloc_868_, 4, v_r_854_);
v___x_867_ = v_reuseFailAlloc_868_;
goto v_reusejp_866_;
}
v_reusejp_866_:
{
return v___x_867_;
}
}
}
}
else
{
lean_object* v_k_873_; lean_object* v_v_874_; lean_object* v___x_876_; uint8_t v_isShared_877_; uint8_t v_isSharedCheck_897_; 
v_k_873_ = lean_ctor_get(v_r_758_, 1);
v_v_874_ = lean_ctor_get(v_r_758_, 2);
v_isSharedCheck_897_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_897_ == 0)
{
lean_object* v_unused_898_; lean_object* v_unused_899_; lean_object* v_unused_900_; 
v_unused_898_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_898_);
v_unused_899_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_899_);
v_unused_900_ = lean_ctor_get(v_r_758_, 0);
lean_dec(v_unused_900_);
v___x_876_ = v_r_758_;
v_isShared_877_ = v_isSharedCheck_897_;
goto v_resetjp_875_;
}
else
{
lean_inc(v_v_874_);
lean_inc(v_k_873_);
lean_dec(v_r_758_);
v___x_876_ = lean_box(0);
v_isShared_877_ = v_isSharedCheck_897_;
goto v_resetjp_875_;
}
v_resetjp_875_:
{
lean_object* v_k_878_; lean_object* v_v_879_; lean_object* v___x_881_; uint8_t v_isShared_882_; uint8_t v_isSharedCheck_893_; 
v_k_878_ = lean_ctor_get(v_l_853_, 1);
v_v_879_ = lean_ctor_get(v_l_853_, 2);
v_isSharedCheck_893_ = !lean_is_exclusive(v_l_853_);
if (v_isSharedCheck_893_ == 0)
{
lean_object* v_unused_894_; lean_object* v_unused_895_; lean_object* v_unused_896_; 
v_unused_894_ = lean_ctor_get(v_l_853_, 4);
lean_dec(v_unused_894_);
v_unused_895_ = lean_ctor_get(v_l_853_, 3);
lean_dec(v_unused_895_);
v_unused_896_ = lean_ctor_get(v_l_853_, 0);
lean_dec(v_unused_896_);
v___x_881_ = v_l_853_;
v_isShared_882_ = v_isSharedCheck_893_;
goto v_resetjp_880_;
}
else
{
lean_inc(v_v_879_);
lean_inc(v_k_878_);
lean_dec(v_l_853_);
v___x_881_ = lean_box(0);
v_isShared_882_ = v_isSharedCheck_893_;
goto v_resetjp_880_;
}
v_resetjp_880_:
{
lean_object* v___x_883_; lean_object* v___x_885_; 
v___x_883_ = lean_unsigned_to_nat(3u);
if (v_isShared_882_ == 0)
{
lean_ctor_set(v___x_881_, 4, v_r_854_);
lean_ctor_set(v___x_881_, 3, v_r_854_);
lean_ctor_set(v___x_881_, 2, v_v_756_);
lean_ctor_set(v___x_881_, 1, v_k_755_);
lean_ctor_set(v___x_881_, 0, v___x_764_);
v___x_885_ = v___x_881_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v___x_764_);
lean_ctor_set(v_reuseFailAlloc_892_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_892_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_892_, 3, v_r_854_);
lean_ctor_set(v_reuseFailAlloc_892_, 4, v_r_854_);
v___x_885_ = v_reuseFailAlloc_892_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
lean_object* v___x_887_; 
if (v_isShared_877_ == 0)
{
lean_ctor_set(v___x_876_, 3, v_r_854_);
lean_ctor_set(v___x_876_, 0, v___x_764_);
v___x_887_ = v___x_876_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v___x_764_);
lean_ctor_set(v_reuseFailAlloc_891_, 1, v_k_873_);
lean_ctor_set(v_reuseFailAlloc_891_, 2, v_v_874_);
lean_ctor_set(v_reuseFailAlloc_891_, 3, v_r_854_);
lean_ctor_set(v_reuseFailAlloc_891_, 4, v_r_854_);
v___x_887_ = v_reuseFailAlloc_891_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
lean_object* v___x_889_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v___x_887_);
lean_ctor_set(v___x_760_, 3, v___x_885_);
lean_ctor_set(v___x_760_, 2, v_v_879_);
lean_ctor_set(v___x_760_, 1, v_k_878_);
lean_ctor_set(v___x_760_, 0, v___x_883_);
v___x_889_ = v___x_760_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_890_; 
v_reuseFailAlloc_890_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_890_, 0, v___x_883_);
lean_ctor_set(v_reuseFailAlloc_890_, 1, v_k_878_);
lean_ctor_set(v_reuseFailAlloc_890_, 2, v_v_879_);
lean_ctor_set(v_reuseFailAlloc_890_, 3, v___x_885_);
lean_ctor_set(v_reuseFailAlloc_890_, 4, v___x_887_);
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
lean_object* v_r_901_; 
v_r_901_ = lean_ctor_get(v_r_758_, 4);
lean_inc(v_r_901_);
if (lean_obj_tag(v_r_901_) == 0)
{
lean_object* v_k_902_; lean_object* v_v_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_914_; 
v_k_902_ = lean_ctor_get(v_r_758_, 1);
v_v_903_ = lean_ctor_get(v_r_758_, 2);
v_isSharedCheck_914_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_914_ == 0)
{
lean_object* v_unused_915_; lean_object* v_unused_916_; lean_object* v_unused_917_; 
v_unused_915_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_915_);
v_unused_916_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_916_);
v_unused_917_ = lean_ctor_get(v_r_758_, 0);
lean_dec(v_unused_917_);
v___x_905_ = v_r_758_;
v_isShared_906_ = v_isSharedCheck_914_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_v_903_);
lean_inc(v_k_902_);
lean_dec(v_r_758_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_914_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v___x_907_; lean_object* v___x_909_; 
v___x_907_ = lean_unsigned_to_nat(3u);
if (v_isShared_906_ == 0)
{
lean_ctor_set(v___x_905_, 4, v_l_853_);
lean_ctor_set(v___x_905_, 2, v_v_756_);
lean_ctor_set(v___x_905_, 1, v_k_755_);
lean_ctor_set(v___x_905_, 0, v___x_764_);
v___x_909_ = v___x_905_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v___x_764_);
lean_ctor_set(v_reuseFailAlloc_913_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_913_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_913_, 3, v_l_853_);
lean_ctor_set(v_reuseFailAlloc_913_, 4, v_l_853_);
v___x_909_ = v_reuseFailAlloc_913_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
lean_object* v___x_911_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_r_901_);
lean_ctor_set(v___x_760_, 3, v___x_909_);
lean_ctor_set(v___x_760_, 2, v_v_903_);
lean_ctor_set(v___x_760_, 1, v_k_902_);
lean_ctor_set(v___x_760_, 0, v___x_907_);
v___x_911_ = v___x_760_;
goto v_reusejp_910_;
}
else
{
lean_object* v_reuseFailAlloc_912_; 
v_reuseFailAlloc_912_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_912_, 0, v___x_907_);
lean_ctor_set(v_reuseFailAlloc_912_, 1, v_k_902_);
lean_ctor_set(v_reuseFailAlloc_912_, 2, v_v_903_);
lean_ctor_set(v_reuseFailAlloc_912_, 3, v___x_909_);
lean_ctor_set(v_reuseFailAlloc_912_, 4, v_r_901_);
v___x_911_ = v_reuseFailAlloc_912_;
goto v_reusejp_910_;
}
v_reusejp_910_:
{
return v___x_911_;
}
}
}
}
else
{
lean_object* v_size_918_; lean_object* v_k_919_; lean_object* v_v_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_931_; 
v_size_918_ = lean_ctor_get(v_r_758_, 0);
v_k_919_ = lean_ctor_get(v_r_758_, 1);
v_v_920_ = lean_ctor_get(v_r_758_, 2);
v_isSharedCheck_931_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_931_ == 0)
{
lean_object* v_unused_932_; lean_object* v_unused_933_; 
v_unused_932_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_932_);
v_unused_933_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_933_);
v___x_922_ = v_r_758_;
v_isShared_923_ = v_isSharedCheck_931_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_v_920_);
lean_inc(v_k_919_);
lean_inc(v_size_918_);
lean_dec(v_r_758_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_931_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_925_; 
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 3, v_r_901_);
v___x_925_ = v___x_922_;
goto v_reusejp_924_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v_size_918_);
lean_ctor_set(v_reuseFailAlloc_930_, 1, v_k_919_);
lean_ctor_set(v_reuseFailAlloc_930_, 2, v_v_920_);
lean_ctor_set(v_reuseFailAlloc_930_, 3, v_r_901_);
lean_ctor_set(v_reuseFailAlloc_930_, 4, v_r_901_);
v___x_925_ = v_reuseFailAlloc_930_;
goto v_reusejp_924_;
}
v_reusejp_924_:
{
lean_object* v___x_926_; lean_object* v___x_928_; 
v___x_926_ = lean_unsigned_to_nat(2u);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v___x_925_);
lean_ctor_set(v___x_760_, 3, v_r_901_);
lean_ctor_set(v___x_760_, 0, v___x_926_);
v___x_928_ = v___x_760_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v___x_926_);
lean_ctor_set(v_reuseFailAlloc_929_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_929_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_929_, 3, v_r_901_);
lean_ctor_set(v_reuseFailAlloc_929_, 4, v___x_925_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
}
}
}
else
{
lean_object* v___x_935_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 3, v_r_758_);
lean_ctor_set(v___x_760_, 0, v___x_764_);
v___x_935_ = v___x_760_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v___x_764_);
lean_ctor_set(v_reuseFailAlloc_936_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_936_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_936_, 3, v_r_758_);
lean_ctor_set(v_reuseFailAlloc_936_, 4, v_r_758_);
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
case 1:
{
lean_del_object(v___x_760_);
lean_dec(v_v_756_);
lean_dec(v_k_755_);
if (lean_obj_tag(v_l_757_) == 0)
{
if (lean_obj_tag(v_r_758_) == 0)
{
lean_object* v_size_937_; lean_object* v_k_938_; lean_object* v_v_939_; lean_object* v_l_940_; lean_object* v_r_941_; lean_object* v_size_942_; lean_object* v_k_943_; lean_object* v_v_944_; lean_object* v_l_945_; lean_object* v_r_946_; lean_object* v___x_947_; uint8_t v___x_948_; 
v_size_937_ = lean_ctor_get(v_l_757_, 0);
v_k_938_ = lean_ctor_get(v_l_757_, 1);
v_v_939_ = lean_ctor_get(v_l_757_, 2);
v_l_940_ = lean_ctor_get(v_l_757_, 3);
v_r_941_ = lean_ctor_get(v_l_757_, 4);
lean_inc(v_r_941_);
v_size_942_ = lean_ctor_get(v_r_758_, 0);
v_k_943_ = lean_ctor_get(v_r_758_, 1);
v_v_944_ = lean_ctor_get(v_r_758_, 2);
v_l_945_ = lean_ctor_get(v_r_758_, 3);
lean_inc(v_l_945_);
v_r_946_ = lean_ctor_get(v_r_758_, 4);
v___x_947_ = lean_unsigned_to_nat(1u);
v___x_948_ = lean_nat_dec_lt(v_size_937_, v_size_942_);
if (v___x_948_ == 0)
{
lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_1084_; 
lean_inc(v_l_940_);
lean_inc(v_v_939_);
lean_inc(v_k_938_);
v_isSharedCheck_1084_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1084_ == 0)
{
lean_object* v_unused_1085_; lean_object* v_unused_1086_; lean_object* v_unused_1087_; lean_object* v_unused_1088_; lean_object* v_unused_1089_; 
v_unused_1085_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1085_);
v_unused_1086_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1086_);
v_unused_1087_ = lean_ctor_get(v_l_757_, 2);
lean_dec(v_unused_1087_);
v_unused_1088_ = lean_ctor_get(v_l_757_, 1);
lean_dec(v_unused_1088_);
v_unused_1089_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1089_);
v___x_950_ = v_l_757_;
v_isShared_951_ = v_isSharedCheck_1084_;
goto v_resetjp_949_;
}
else
{
lean_dec(v_l_757_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_1084_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_952_; lean_object* v_tree_953_; 
v___x_952_ = l_Std_DTreeMap_Internal_Impl_maxView___redArg(v_k_938_, v_v_939_, v_l_940_, v_r_941_);
v_tree_953_ = lean_ctor_get(v___x_952_, 2);
lean_inc(v_tree_953_);
if (lean_obj_tag(v_tree_953_) == 0)
{
lean_object* v_k_954_; lean_object* v_v_955_; lean_object* v_size_956_; lean_object* v___x_957_; lean_object* v___x_958_; uint8_t v___x_959_; 
v_k_954_ = lean_ctor_get(v___x_952_, 0);
lean_inc(v_k_954_);
v_v_955_ = lean_ctor_get(v___x_952_, 1);
lean_inc(v_v_955_);
lean_dec_ref(v___x_952_);
v_size_956_ = lean_ctor_get(v_tree_953_, 0);
v___x_957_ = lean_unsigned_to_nat(3u);
v___x_958_ = lean_nat_mul(v___x_957_, v_size_956_);
v___x_959_ = lean_nat_dec_lt(v___x_958_, v_size_942_);
lean_dec(v___x_958_);
if (v___x_959_ == 0)
{
lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_963_; 
lean_dec(v_l_945_);
v___x_960_ = lean_nat_add(v___x_947_, v_size_956_);
v___x_961_ = lean_nat_add(v___x_960_, v_size_942_);
lean_dec(v___x_960_);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v_r_758_);
lean_ctor_set(v___x_950_, 3, v_tree_953_);
lean_ctor_set(v___x_950_, 2, v_v_955_);
lean_ctor_set(v___x_950_, 1, v_k_954_);
lean_ctor_set(v___x_950_, 0, v___x_961_);
v___x_963_ = v___x_950_;
goto v_reusejp_962_;
}
else
{
lean_object* v_reuseFailAlloc_964_; 
v_reuseFailAlloc_964_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_964_, 0, v___x_961_);
lean_ctor_set(v_reuseFailAlloc_964_, 1, v_k_954_);
lean_ctor_set(v_reuseFailAlloc_964_, 2, v_v_955_);
lean_ctor_set(v_reuseFailAlloc_964_, 3, v_tree_953_);
lean_ctor_set(v_reuseFailAlloc_964_, 4, v_r_758_);
v___x_963_ = v_reuseFailAlloc_964_;
goto v_reusejp_962_;
}
v_reusejp_962_:
{
return v___x_963_;
}
}
else
{
lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_1019_; 
lean_inc(v_r_946_);
lean_inc(v_v_944_);
lean_inc(v_k_943_);
lean_inc(v_size_942_);
v_isSharedCheck_1019_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_1019_ == 0)
{
lean_object* v_unused_1020_; lean_object* v_unused_1021_; lean_object* v_unused_1022_; lean_object* v_unused_1023_; lean_object* v_unused_1024_; 
v_unused_1020_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_1020_);
v_unused_1021_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_1021_);
v_unused_1022_ = lean_ctor_get(v_r_758_, 2);
lean_dec(v_unused_1022_);
v_unused_1023_ = lean_ctor_get(v_r_758_, 1);
lean_dec(v_unused_1023_);
v_unused_1024_ = lean_ctor_get(v_r_758_, 0);
lean_dec(v_unused_1024_);
v___x_966_ = v_r_758_;
v_isShared_967_ = v_isSharedCheck_1019_;
goto v_resetjp_965_;
}
else
{
lean_dec(v_r_758_);
v___x_966_ = lean_box(0);
v_isShared_967_ = v_isSharedCheck_1019_;
goto v_resetjp_965_;
}
v_resetjp_965_:
{
lean_object* v_size_968_; lean_object* v_k_969_; lean_object* v_v_970_; lean_object* v_l_971_; lean_object* v_r_972_; lean_object* v_size_973_; lean_object* v___x_974_; lean_object* v___x_975_; uint8_t v___x_976_; 
v_size_968_ = lean_ctor_get(v_l_945_, 0);
v_k_969_ = lean_ctor_get(v_l_945_, 1);
v_v_970_ = lean_ctor_get(v_l_945_, 2);
v_l_971_ = lean_ctor_get(v_l_945_, 3);
v_r_972_ = lean_ctor_get(v_l_945_, 4);
v_size_973_ = lean_ctor_get(v_r_946_, 0);
v___x_974_ = lean_unsigned_to_nat(2u);
v___x_975_ = lean_nat_mul(v___x_974_, v_size_973_);
v___x_976_ = lean_nat_dec_lt(v_size_968_, v___x_975_);
lean_dec(v___x_975_);
if (v___x_976_ == 0)
{
lean_object* v___x_978_; uint8_t v_isShared_979_; uint8_t v_isSharedCheck_1004_; 
lean_inc(v_r_972_);
lean_inc(v_l_971_);
lean_inc(v_v_970_);
lean_inc(v_k_969_);
v_isSharedCheck_1004_ = !lean_is_exclusive(v_l_945_);
if (v_isSharedCheck_1004_ == 0)
{
lean_object* v_unused_1005_; lean_object* v_unused_1006_; lean_object* v_unused_1007_; lean_object* v_unused_1008_; lean_object* v_unused_1009_; 
v_unused_1005_ = lean_ctor_get(v_l_945_, 4);
lean_dec(v_unused_1005_);
v_unused_1006_ = lean_ctor_get(v_l_945_, 3);
lean_dec(v_unused_1006_);
v_unused_1007_ = lean_ctor_get(v_l_945_, 2);
lean_dec(v_unused_1007_);
v_unused_1008_ = lean_ctor_get(v_l_945_, 1);
lean_dec(v_unused_1008_);
v_unused_1009_ = lean_ctor_get(v_l_945_, 0);
lean_dec(v_unused_1009_);
v___x_978_ = v_l_945_;
v_isShared_979_ = v_isSharedCheck_1004_;
goto v_resetjp_977_;
}
else
{
lean_dec(v_l_945_);
v___x_978_ = lean_box(0);
v_isShared_979_ = v_isSharedCheck_1004_;
goto v_resetjp_977_;
}
v_resetjp_977_:
{
lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___y_983_; lean_object* v___y_984_; lean_object* v___y_985_; lean_object* v___y_994_; 
v___x_980_ = lean_nat_add(v___x_947_, v_size_956_);
v___x_981_ = lean_nat_add(v___x_980_, v_size_942_);
lean_dec(v_size_942_);
if (lean_obj_tag(v_l_971_) == 0)
{
lean_object* v_size_1002_; 
v_size_1002_ = lean_ctor_get(v_l_971_, 0);
lean_inc(v_size_1002_);
v___y_994_ = v_size_1002_;
goto v___jp_993_;
}
else
{
lean_object* v___x_1003_; 
v___x_1003_ = lean_unsigned_to_nat(0u);
v___y_994_ = v___x_1003_;
goto v___jp_993_;
}
v___jp_982_:
{
lean_object* v___x_986_; lean_object* v___x_988_; 
v___x_986_ = lean_nat_add(v___y_984_, v___y_985_);
lean_dec(v___y_985_);
lean_dec(v___y_984_);
if (v_isShared_979_ == 0)
{
lean_ctor_set(v___x_978_, 4, v_r_946_);
lean_ctor_set(v___x_978_, 3, v_r_972_);
lean_ctor_set(v___x_978_, 2, v_v_944_);
lean_ctor_set(v___x_978_, 1, v_k_943_);
lean_ctor_set(v___x_978_, 0, v___x_986_);
v___x_988_ = v___x_978_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_992_; 
v_reuseFailAlloc_992_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_992_, 0, v___x_986_);
lean_ctor_set(v_reuseFailAlloc_992_, 1, v_k_943_);
lean_ctor_set(v_reuseFailAlloc_992_, 2, v_v_944_);
lean_ctor_set(v_reuseFailAlloc_992_, 3, v_r_972_);
lean_ctor_set(v_reuseFailAlloc_992_, 4, v_r_946_);
v___x_988_ = v_reuseFailAlloc_992_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
lean_object* v___x_990_; 
if (v_isShared_967_ == 0)
{
lean_ctor_set(v___x_966_, 4, v___x_988_);
lean_ctor_set(v___x_966_, 3, v___y_983_);
lean_ctor_set(v___x_966_, 2, v_v_970_);
lean_ctor_set(v___x_966_, 1, v_k_969_);
lean_ctor_set(v___x_966_, 0, v___x_981_);
v___x_990_ = v___x_966_;
goto v_reusejp_989_;
}
else
{
lean_object* v_reuseFailAlloc_991_; 
v_reuseFailAlloc_991_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_991_, 0, v___x_981_);
lean_ctor_set(v_reuseFailAlloc_991_, 1, v_k_969_);
lean_ctor_set(v_reuseFailAlloc_991_, 2, v_v_970_);
lean_ctor_set(v_reuseFailAlloc_991_, 3, v___y_983_);
lean_ctor_set(v_reuseFailAlloc_991_, 4, v___x_988_);
v___x_990_ = v_reuseFailAlloc_991_;
goto v_reusejp_989_;
}
v_reusejp_989_:
{
return v___x_990_;
}
}
}
v___jp_993_:
{
lean_object* v___x_995_; lean_object* v___x_997_; 
v___x_995_ = lean_nat_add(v___x_980_, v___y_994_);
lean_dec(v___y_994_);
lean_dec(v___x_980_);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v_l_971_);
lean_ctor_set(v___x_950_, 3, v_tree_953_);
lean_ctor_set(v___x_950_, 2, v_v_955_);
lean_ctor_set(v___x_950_, 1, v_k_954_);
lean_ctor_set(v___x_950_, 0, v___x_995_);
v___x_997_ = v___x_950_;
goto v_reusejp_996_;
}
else
{
lean_object* v_reuseFailAlloc_1001_; 
v_reuseFailAlloc_1001_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1001_, 0, v___x_995_);
lean_ctor_set(v_reuseFailAlloc_1001_, 1, v_k_954_);
lean_ctor_set(v_reuseFailAlloc_1001_, 2, v_v_955_);
lean_ctor_set(v_reuseFailAlloc_1001_, 3, v_tree_953_);
lean_ctor_set(v_reuseFailAlloc_1001_, 4, v_l_971_);
v___x_997_ = v_reuseFailAlloc_1001_;
goto v_reusejp_996_;
}
v_reusejp_996_:
{
lean_object* v___x_998_; 
v___x_998_ = lean_nat_add(v___x_947_, v_size_973_);
if (lean_obj_tag(v_r_972_) == 0)
{
lean_object* v_size_999_; 
v_size_999_ = lean_ctor_get(v_r_972_, 0);
lean_inc(v_size_999_);
v___y_983_ = v___x_997_;
v___y_984_ = v___x_998_;
v___y_985_ = v_size_999_;
goto v___jp_982_;
}
else
{
lean_object* v___x_1000_; 
v___x_1000_ = lean_unsigned_to_nat(0u);
v___y_983_ = v___x_997_;
v___y_984_ = v___x_998_;
v___y_985_ = v___x_1000_;
goto v___jp_982_;
}
}
}
}
}
else
{
lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1014_; 
v___x_1010_ = lean_nat_add(v___x_947_, v_size_956_);
v___x_1011_ = lean_nat_add(v___x_1010_, v_size_942_);
lean_dec(v_size_942_);
v___x_1012_ = lean_nat_add(v___x_1010_, v_size_968_);
lean_dec(v___x_1010_);
if (v_isShared_967_ == 0)
{
lean_ctor_set(v___x_966_, 4, v_l_945_);
lean_ctor_set(v___x_966_, 3, v_tree_953_);
lean_ctor_set(v___x_966_, 2, v_v_955_);
lean_ctor_set(v___x_966_, 1, v_k_954_);
lean_ctor_set(v___x_966_, 0, v___x_1012_);
v___x_1014_ = v___x_966_;
goto v_reusejp_1013_;
}
else
{
lean_object* v_reuseFailAlloc_1018_; 
v_reuseFailAlloc_1018_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1018_, 0, v___x_1012_);
lean_ctor_set(v_reuseFailAlloc_1018_, 1, v_k_954_);
lean_ctor_set(v_reuseFailAlloc_1018_, 2, v_v_955_);
lean_ctor_set(v_reuseFailAlloc_1018_, 3, v_tree_953_);
lean_ctor_set(v_reuseFailAlloc_1018_, 4, v_l_945_);
v___x_1014_ = v_reuseFailAlloc_1018_;
goto v_reusejp_1013_;
}
v_reusejp_1013_:
{
lean_object* v___x_1016_; 
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v_r_946_);
lean_ctor_set(v___x_950_, 3, v___x_1014_);
lean_ctor_set(v___x_950_, 2, v_v_944_);
lean_ctor_set(v___x_950_, 1, v_k_943_);
lean_ctor_set(v___x_950_, 0, v___x_1011_);
v___x_1016_ = v___x_950_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1017_; 
v_reuseFailAlloc_1017_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1017_, 0, v___x_1011_);
lean_ctor_set(v_reuseFailAlloc_1017_, 1, v_k_943_);
lean_ctor_set(v_reuseFailAlloc_1017_, 2, v_v_944_);
lean_ctor_set(v_reuseFailAlloc_1017_, 3, v___x_1014_);
lean_ctor_set(v_reuseFailAlloc_1017_, 4, v_r_946_);
v___x_1016_ = v_reuseFailAlloc_1017_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
return v___x_1016_;
}
}
}
}
}
}
else
{
lean_object* v___x_1026_; uint8_t v_isShared_1027_; uint8_t v_isSharedCheck_1078_; 
lean_inc(v_r_946_);
lean_inc(v_v_944_);
lean_inc(v_k_943_);
lean_inc(v_size_942_);
v_isSharedCheck_1078_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_1078_ == 0)
{
lean_object* v_unused_1079_; lean_object* v_unused_1080_; lean_object* v_unused_1081_; lean_object* v_unused_1082_; lean_object* v_unused_1083_; 
v_unused_1079_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_1079_);
v_unused_1080_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_1080_);
v_unused_1081_ = lean_ctor_get(v_r_758_, 2);
lean_dec(v_unused_1081_);
v_unused_1082_ = lean_ctor_get(v_r_758_, 1);
lean_dec(v_unused_1082_);
v_unused_1083_ = lean_ctor_get(v_r_758_, 0);
lean_dec(v_unused_1083_);
v___x_1026_ = v_r_758_;
v_isShared_1027_ = v_isSharedCheck_1078_;
goto v_resetjp_1025_;
}
else
{
lean_dec(v_r_758_);
v___x_1026_ = lean_box(0);
v_isShared_1027_ = v_isSharedCheck_1078_;
goto v_resetjp_1025_;
}
v_resetjp_1025_:
{
if (lean_obj_tag(v_l_945_) == 0)
{
if (lean_obj_tag(v_r_946_) == 0)
{
lean_object* v_k_1028_; lean_object* v_v_1029_; lean_object* v_size_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1034_; 
v_k_1028_ = lean_ctor_get(v___x_952_, 0);
lean_inc(v_k_1028_);
v_v_1029_ = lean_ctor_get(v___x_952_, 1);
lean_inc(v_v_1029_);
lean_dec_ref(v___x_952_);
v_size_1030_ = lean_ctor_get(v_l_945_, 0);
v___x_1031_ = lean_nat_add(v___x_947_, v_size_942_);
lean_dec(v_size_942_);
v___x_1032_ = lean_nat_add(v___x_947_, v_size_1030_);
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 4, v_l_945_);
lean_ctor_set(v___x_1026_, 3, v_tree_953_);
lean_ctor_set(v___x_1026_, 2, v_v_1029_);
lean_ctor_set(v___x_1026_, 1, v_k_1028_);
lean_ctor_set(v___x_1026_, 0, v___x_1032_);
v___x_1034_ = v___x_1026_;
goto v_reusejp_1033_;
}
else
{
lean_object* v_reuseFailAlloc_1038_; 
v_reuseFailAlloc_1038_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1038_, 0, v___x_1032_);
lean_ctor_set(v_reuseFailAlloc_1038_, 1, v_k_1028_);
lean_ctor_set(v_reuseFailAlloc_1038_, 2, v_v_1029_);
lean_ctor_set(v_reuseFailAlloc_1038_, 3, v_tree_953_);
lean_ctor_set(v_reuseFailAlloc_1038_, 4, v_l_945_);
v___x_1034_ = v_reuseFailAlloc_1038_;
goto v_reusejp_1033_;
}
v_reusejp_1033_:
{
lean_object* v___x_1036_; 
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v_r_946_);
lean_ctor_set(v___x_950_, 3, v___x_1034_);
lean_ctor_set(v___x_950_, 2, v_v_944_);
lean_ctor_set(v___x_950_, 1, v_k_943_);
lean_ctor_set(v___x_950_, 0, v___x_1031_);
v___x_1036_ = v___x_950_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1037_; 
v_reuseFailAlloc_1037_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1037_, 0, v___x_1031_);
lean_ctor_set(v_reuseFailAlloc_1037_, 1, v_k_943_);
lean_ctor_set(v_reuseFailAlloc_1037_, 2, v_v_944_);
lean_ctor_set(v_reuseFailAlloc_1037_, 3, v___x_1034_);
lean_ctor_set(v_reuseFailAlloc_1037_, 4, v_r_946_);
v___x_1036_ = v_reuseFailAlloc_1037_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
return v___x_1036_;
}
}
}
else
{
lean_object* v_k_1039_; lean_object* v_v_1040_; lean_object* v_k_1041_; lean_object* v_v_1042_; lean_object* v___x_1044_; uint8_t v_isShared_1045_; uint8_t v_isSharedCheck_1056_; 
lean_dec(v_size_942_);
v_k_1039_ = lean_ctor_get(v___x_952_, 0);
lean_inc(v_k_1039_);
v_v_1040_ = lean_ctor_get(v___x_952_, 1);
lean_inc(v_v_1040_);
lean_dec_ref(v___x_952_);
v_k_1041_ = lean_ctor_get(v_l_945_, 1);
v_v_1042_ = lean_ctor_get(v_l_945_, 2);
v_isSharedCheck_1056_ = !lean_is_exclusive(v_l_945_);
if (v_isSharedCheck_1056_ == 0)
{
lean_object* v_unused_1057_; lean_object* v_unused_1058_; lean_object* v_unused_1059_; 
v_unused_1057_ = lean_ctor_get(v_l_945_, 4);
lean_dec(v_unused_1057_);
v_unused_1058_ = lean_ctor_get(v_l_945_, 3);
lean_dec(v_unused_1058_);
v_unused_1059_ = lean_ctor_get(v_l_945_, 0);
lean_dec(v_unused_1059_);
v___x_1044_ = v_l_945_;
v_isShared_1045_ = v_isSharedCheck_1056_;
goto v_resetjp_1043_;
}
else
{
lean_inc(v_v_1042_);
lean_inc(v_k_1041_);
lean_dec(v_l_945_);
v___x_1044_ = lean_box(0);
v_isShared_1045_ = v_isSharedCheck_1056_;
goto v_resetjp_1043_;
}
v_resetjp_1043_:
{
lean_object* v___x_1046_; lean_object* v___x_1048_; 
v___x_1046_ = lean_unsigned_to_nat(3u);
if (v_isShared_1045_ == 0)
{
lean_ctor_set(v___x_1044_, 4, v_r_946_);
lean_ctor_set(v___x_1044_, 3, v_r_946_);
lean_ctor_set(v___x_1044_, 2, v_v_1040_);
lean_ctor_set(v___x_1044_, 1, v_k_1039_);
lean_ctor_set(v___x_1044_, 0, v___x_947_);
v___x_1048_ = v___x_1044_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1055_; 
v_reuseFailAlloc_1055_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1055_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_1055_, 1, v_k_1039_);
lean_ctor_set(v_reuseFailAlloc_1055_, 2, v_v_1040_);
lean_ctor_set(v_reuseFailAlloc_1055_, 3, v_r_946_);
lean_ctor_set(v_reuseFailAlloc_1055_, 4, v_r_946_);
v___x_1048_ = v_reuseFailAlloc_1055_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
lean_object* v___x_1050_; 
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 3, v_r_946_);
lean_ctor_set(v___x_1026_, 0, v___x_947_);
v___x_1050_ = v___x_1026_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1054_; 
v_reuseFailAlloc_1054_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1054_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_1054_, 1, v_k_943_);
lean_ctor_set(v_reuseFailAlloc_1054_, 2, v_v_944_);
lean_ctor_set(v_reuseFailAlloc_1054_, 3, v_r_946_);
lean_ctor_set(v_reuseFailAlloc_1054_, 4, v_r_946_);
v___x_1050_ = v_reuseFailAlloc_1054_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
lean_object* v___x_1052_; 
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v___x_1050_);
lean_ctor_set(v___x_950_, 3, v___x_1048_);
lean_ctor_set(v___x_950_, 2, v_v_1042_);
lean_ctor_set(v___x_950_, 1, v_k_1041_);
lean_ctor_set(v___x_950_, 0, v___x_1046_);
v___x_1052_ = v___x_950_;
goto v_reusejp_1051_;
}
else
{
lean_object* v_reuseFailAlloc_1053_; 
v_reuseFailAlloc_1053_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1053_, 0, v___x_1046_);
lean_ctor_set(v_reuseFailAlloc_1053_, 1, v_k_1041_);
lean_ctor_set(v_reuseFailAlloc_1053_, 2, v_v_1042_);
lean_ctor_set(v_reuseFailAlloc_1053_, 3, v___x_1048_);
lean_ctor_set(v_reuseFailAlloc_1053_, 4, v___x_1050_);
v___x_1052_ = v_reuseFailAlloc_1053_;
goto v_reusejp_1051_;
}
v_reusejp_1051_:
{
return v___x_1052_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_946_) == 0)
{
lean_object* v_k_1060_; lean_object* v_v_1061_; lean_object* v___x_1062_; lean_object* v___x_1064_; 
lean_dec(v_size_942_);
v_k_1060_ = lean_ctor_get(v___x_952_, 0);
lean_inc(v_k_1060_);
v_v_1061_ = lean_ctor_get(v___x_952_, 1);
lean_inc(v_v_1061_);
lean_dec_ref(v___x_952_);
v___x_1062_ = lean_unsigned_to_nat(3u);
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 4, v_l_945_);
lean_ctor_set(v___x_1026_, 2, v_v_1061_);
lean_ctor_set(v___x_1026_, 1, v_k_1060_);
lean_ctor_set(v___x_1026_, 0, v___x_947_);
v___x_1064_ = v___x_1026_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_1068_, 1, v_k_1060_);
lean_ctor_set(v_reuseFailAlloc_1068_, 2, v_v_1061_);
lean_ctor_set(v_reuseFailAlloc_1068_, 3, v_l_945_);
lean_ctor_set(v_reuseFailAlloc_1068_, 4, v_l_945_);
v___x_1064_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
lean_object* v___x_1066_; 
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v_r_946_);
lean_ctor_set(v___x_950_, 3, v___x_1064_);
lean_ctor_set(v___x_950_, 2, v_v_944_);
lean_ctor_set(v___x_950_, 1, v_k_943_);
lean_ctor_set(v___x_950_, 0, v___x_1062_);
v___x_1066_ = v___x_950_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1067_; 
v_reuseFailAlloc_1067_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1067_, 0, v___x_1062_);
lean_ctor_set(v_reuseFailAlloc_1067_, 1, v_k_943_);
lean_ctor_set(v_reuseFailAlloc_1067_, 2, v_v_944_);
lean_ctor_set(v_reuseFailAlloc_1067_, 3, v___x_1064_);
lean_ctor_set(v_reuseFailAlloc_1067_, 4, v_r_946_);
v___x_1066_ = v_reuseFailAlloc_1067_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
return v___x_1066_;
}
}
}
else
{
lean_object* v_k_1069_; lean_object* v_v_1070_; lean_object* v___x_1072_; 
v_k_1069_ = lean_ctor_get(v___x_952_, 0);
lean_inc(v_k_1069_);
v_v_1070_ = lean_ctor_get(v___x_952_, 1);
lean_inc(v_v_1070_);
lean_dec_ref(v___x_952_);
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 3, v_r_946_);
v___x_1072_ = v___x_1026_;
goto v_reusejp_1071_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_size_942_);
lean_ctor_set(v_reuseFailAlloc_1077_, 1, v_k_943_);
lean_ctor_set(v_reuseFailAlloc_1077_, 2, v_v_944_);
lean_ctor_set(v_reuseFailAlloc_1077_, 3, v_r_946_);
lean_ctor_set(v_reuseFailAlloc_1077_, 4, v_r_946_);
v___x_1072_ = v_reuseFailAlloc_1077_;
goto v_reusejp_1071_;
}
v_reusejp_1071_:
{
lean_object* v___x_1073_; lean_object* v___x_1075_; 
v___x_1073_ = lean_unsigned_to_nat(2u);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 4, v___x_1072_);
lean_ctor_set(v___x_950_, 3, v_r_946_);
lean_ctor_set(v___x_950_, 2, v_v_1070_);
lean_ctor_set(v___x_950_, 1, v_k_1069_);
lean_ctor_set(v___x_950_, 0, v___x_1073_);
v___x_1075_ = v___x_950_;
goto v_reusejp_1074_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v___x_1073_);
lean_ctor_set(v_reuseFailAlloc_1076_, 1, v_k_1069_);
lean_ctor_set(v_reuseFailAlloc_1076_, 2, v_v_1070_);
lean_ctor_set(v_reuseFailAlloc_1076_, 3, v_r_946_);
lean_ctor_set(v_reuseFailAlloc_1076_, 4, v___x_1072_);
v___x_1075_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1074_;
}
v_reusejp_1074_:
{
return v___x_1075_;
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
lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1242_; 
lean_inc(v_r_946_);
lean_inc(v_v_944_);
lean_inc(v_k_943_);
v_isSharedCheck_1242_ = !lean_is_exclusive(v_r_758_);
if (v_isSharedCheck_1242_ == 0)
{
lean_object* v_unused_1243_; lean_object* v_unused_1244_; lean_object* v_unused_1245_; lean_object* v_unused_1246_; lean_object* v_unused_1247_; 
v_unused_1243_ = lean_ctor_get(v_r_758_, 4);
lean_dec(v_unused_1243_);
v_unused_1244_ = lean_ctor_get(v_r_758_, 3);
lean_dec(v_unused_1244_);
v_unused_1245_ = lean_ctor_get(v_r_758_, 2);
lean_dec(v_unused_1245_);
v_unused_1246_ = lean_ctor_get(v_r_758_, 1);
lean_dec(v_unused_1246_);
v_unused_1247_ = lean_ctor_get(v_r_758_, 0);
lean_dec(v_unused_1247_);
v___x_1091_ = v_r_758_;
v_isShared_1092_ = v_isSharedCheck_1242_;
goto v_resetjp_1090_;
}
else
{
lean_dec(v_r_758_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1242_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1093_; lean_object* v_tree_1094_; 
v___x_1093_ = l_Std_DTreeMap_Internal_Impl_minView___redArg(v_k_943_, v_v_944_, v_l_945_, v_r_946_);
v_tree_1094_ = lean_ctor_get(v___x_1093_, 2);
lean_inc(v_tree_1094_);
if (lean_obj_tag(v_tree_1094_) == 0)
{
lean_object* v_k_1095_; lean_object* v_v_1096_; lean_object* v_size_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; uint8_t v___x_1100_; 
v_k_1095_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_k_1095_);
v_v_1096_ = lean_ctor_get(v___x_1093_, 1);
lean_inc(v_v_1096_);
lean_dec_ref(v___x_1093_);
v_size_1097_ = lean_ctor_get(v_tree_1094_, 0);
v___x_1098_ = lean_unsigned_to_nat(3u);
v___x_1099_ = lean_nat_mul(v___x_1098_, v_size_1097_);
v___x_1100_ = lean_nat_dec_lt(v___x_1099_, v_size_937_);
lean_dec(v___x_1099_);
if (v___x_1100_ == 0)
{
lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1104_; 
lean_dec(v_r_941_);
v___x_1101_ = lean_nat_add(v___x_947_, v_size_937_);
v___x_1102_ = lean_nat_add(v___x_1101_, v_size_1097_);
lean_dec(v___x_1101_);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_tree_1094_);
lean_ctor_set(v___x_1091_, 3, v_l_757_);
lean_ctor_set(v___x_1091_, 2, v_v_1096_);
lean_ctor_set(v___x_1091_, 1, v_k_1095_);
lean_ctor_set(v___x_1091_, 0, v___x_1102_);
v___x_1104_ = v___x_1091_;
goto v_reusejp_1103_;
}
else
{
lean_object* v_reuseFailAlloc_1105_; 
v_reuseFailAlloc_1105_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1105_, 0, v___x_1102_);
lean_ctor_set(v_reuseFailAlloc_1105_, 1, v_k_1095_);
lean_ctor_set(v_reuseFailAlloc_1105_, 2, v_v_1096_);
lean_ctor_set(v_reuseFailAlloc_1105_, 3, v_l_757_);
lean_ctor_set(v_reuseFailAlloc_1105_, 4, v_tree_1094_);
v___x_1104_ = v_reuseFailAlloc_1105_;
goto v_reusejp_1103_;
}
v_reusejp_1103_:
{
return v___x_1104_;
}
}
else
{
lean_object* v___x_1107_; uint8_t v_isShared_1108_; uint8_t v_isSharedCheck_1171_; 
lean_inc(v_l_940_);
lean_inc(v_v_939_);
lean_inc(v_k_938_);
lean_inc(v_size_937_);
v_isSharedCheck_1171_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1171_ == 0)
{
lean_object* v_unused_1172_; lean_object* v_unused_1173_; lean_object* v_unused_1174_; lean_object* v_unused_1175_; lean_object* v_unused_1176_; 
v_unused_1172_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1172_);
v_unused_1173_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1173_);
v_unused_1174_ = lean_ctor_get(v_l_757_, 2);
lean_dec(v_unused_1174_);
v_unused_1175_ = lean_ctor_get(v_l_757_, 1);
lean_dec(v_unused_1175_);
v_unused_1176_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1176_);
v___x_1107_ = v_l_757_;
v_isShared_1108_ = v_isSharedCheck_1171_;
goto v_resetjp_1106_;
}
else
{
lean_dec(v_l_757_);
v___x_1107_ = lean_box(0);
v_isShared_1108_ = v_isSharedCheck_1171_;
goto v_resetjp_1106_;
}
v_resetjp_1106_:
{
lean_object* v_size_1109_; lean_object* v_size_1110_; lean_object* v_k_1111_; lean_object* v_v_1112_; lean_object* v_l_1113_; lean_object* v_r_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; uint8_t v___x_1117_; 
v_size_1109_ = lean_ctor_get(v_l_940_, 0);
v_size_1110_ = lean_ctor_get(v_r_941_, 0);
v_k_1111_ = lean_ctor_get(v_r_941_, 1);
v_v_1112_ = lean_ctor_get(v_r_941_, 2);
v_l_1113_ = lean_ctor_get(v_r_941_, 3);
v_r_1114_ = lean_ctor_get(v_r_941_, 4);
v___x_1115_ = lean_unsigned_to_nat(2u);
v___x_1116_ = lean_nat_mul(v___x_1115_, v_size_1109_);
v___x_1117_ = lean_nat_dec_lt(v_size_1110_, v___x_1116_);
lean_dec(v___x_1116_);
if (v___x_1117_ == 0)
{
lean_object* v___x_1119_; uint8_t v_isShared_1120_; uint8_t v_isSharedCheck_1155_; 
lean_inc(v_r_1114_);
lean_inc(v_l_1113_);
lean_inc(v_v_1112_);
lean_inc(v_k_1111_);
lean_del_object(v___x_1107_);
v_isSharedCheck_1155_ = !lean_is_exclusive(v_r_941_);
if (v_isSharedCheck_1155_ == 0)
{
lean_object* v_unused_1156_; lean_object* v_unused_1157_; lean_object* v_unused_1158_; lean_object* v_unused_1159_; lean_object* v_unused_1160_; 
v_unused_1156_ = lean_ctor_get(v_r_941_, 4);
lean_dec(v_unused_1156_);
v_unused_1157_ = lean_ctor_get(v_r_941_, 3);
lean_dec(v_unused_1157_);
v_unused_1158_ = lean_ctor_get(v_r_941_, 2);
lean_dec(v_unused_1158_);
v_unused_1159_ = lean_ctor_get(v_r_941_, 1);
lean_dec(v_unused_1159_);
v_unused_1160_ = lean_ctor_get(v_r_941_, 0);
lean_dec(v_unused_1160_);
v___x_1119_ = v_r_941_;
v_isShared_1120_ = v_isSharedCheck_1155_;
goto v_resetjp_1118_;
}
else
{
lean_dec(v_r_941_);
v___x_1119_ = lean_box(0);
v_isShared_1120_ = v_isSharedCheck_1155_;
goto v_resetjp_1118_;
}
v_resetjp_1118_:
{
lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___y_1124_; lean_object* v___y_1125_; lean_object* v___y_1126_; lean_object* v___x_1143_; lean_object* v___y_1145_; 
v___x_1121_ = lean_nat_add(v___x_947_, v_size_937_);
lean_dec(v_size_937_);
v___x_1122_ = lean_nat_add(v___x_1121_, v_size_1097_);
lean_dec(v___x_1121_);
v___x_1143_ = lean_nat_add(v___x_947_, v_size_1109_);
if (lean_obj_tag(v_l_1113_) == 0)
{
lean_object* v_size_1153_; 
v_size_1153_ = lean_ctor_get(v_l_1113_, 0);
lean_inc(v_size_1153_);
v___y_1145_ = v_size_1153_;
goto v___jp_1144_;
}
else
{
lean_object* v___x_1154_; 
v___x_1154_ = lean_unsigned_to_nat(0u);
v___y_1145_ = v___x_1154_;
goto v___jp_1144_;
}
v___jp_1123_:
{
lean_object* v___x_1127_; lean_object* v___x_1129_; 
v___x_1127_ = lean_nat_add(v___y_1124_, v___y_1126_);
lean_dec(v___y_1126_);
lean_dec(v___y_1124_);
lean_inc_ref(v_tree_1094_);
if (v_isShared_1120_ == 0)
{
lean_ctor_set(v___x_1119_, 4, v_tree_1094_);
lean_ctor_set(v___x_1119_, 3, v_r_1114_);
lean_ctor_set(v___x_1119_, 2, v_v_1096_);
lean_ctor_set(v___x_1119_, 1, v_k_1095_);
lean_ctor_set(v___x_1119_, 0, v___x_1127_);
v___x_1129_ = v___x_1119_;
goto v_reusejp_1128_;
}
else
{
lean_object* v_reuseFailAlloc_1142_; 
v_reuseFailAlloc_1142_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1142_, 0, v___x_1127_);
lean_ctor_set(v_reuseFailAlloc_1142_, 1, v_k_1095_);
lean_ctor_set(v_reuseFailAlloc_1142_, 2, v_v_1096_);
lean_ctor_set(v_reuseFailAlloc_1142_, 3, v_r_1114_);
lean_ctor_set(v_reuseFailAlloc_1142_, 4, v_tree_1094_);
v___x_1129_ = v_reuseFailAlloc_1142_;
goto v_reusejp_1128_;
}
v_reusejp_1128_:
{
lean_object* v___x_1131_; uint8_t v_isShared_1132_; uint8_t v_isSharedCheck_1136_; 
v_isSharedCheck_1136_ = !lean_is_exclusive(v_tree_1094_);
if (v_isSharedCheck_1136_ == 0)
{
lean_object* v_unused_1137_; lean_object* v_unused_1138_; lean_object* v_unused_1139_; lean_object* v_unused_1140_; lean_object* v_unused_1141_; 
v_unused_1137_ = lean_ctor_get(v_tree_1094_, 4);
lean_dec(v_unused_1137_);
v_unused_1138_ = lean_ctor_get(v_tree_1094_, 3);
lean_dec(v_unused_1138_);
v_unused_1139_ = lean_ctor_get(v_tree_1094_, 2);
lean_dec(v_unused_1139_);
v_unused_1140_ = lean_ctor_get(v_tree_1094_, 1);
lean_dec(v_unused_1140_);
v_unused_1141_ = lean_ctor_get(v_tree_1094_, 0);
lean_dec(v_unused_1141_);
v___x_1131_ = v_tree_1094_;
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
else
{
lean_dec(v_tree_1094_);
v___x_1131_ = lean_box(0);
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
v_resetjp_1130_:
{
lean_object* v___x_1134_; 
if (v_isShared_1132_ == 0)
{
lean_ctor_set(v___x_1131_, 4, v___x_1129_);
lean_ctor_set(v___x_1131_, 3, v___y_1125_);
lean_ctor_set(v___x_1131_, 2, v_v_1112_);
lean_ctor_set(v___x_1131_, 1, v_k_1111_);
lean_ctor_set(v___x_1131_, 0, v___x_1122_);
v___x_1134_ = v___x_1131_;
goto v_reusejp_1133_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v___x_1122_);
lean_ctor_set(v_reuseFailAlloc_1135_, 1, v_k_1111_);
lean_ctor_set(v_reuseFailAlloc_1135_, 2, v_v_1112_);
lean_ctor_set(v_reuseFailAlloc_1135_, 3, v___y_1125_);
lean_ctor_set(v_reuseFailAlloc_1135_, 4, v___x_1129_);
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
v___jp_1144_:
{
lean_object* v___x_1146_; lean_object* v___x_1148_; 
v___x_1146_ = lean_nat_add(v___x_1143_, v___y_1145_);
lean_dec(v___y_1145_);
lean_dec(v___x_1143_);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_l_1113_);
lean_ctor_set(v___x_1091_, 3, v_l_940_);
lean_ctor_set(v___x_1091_, 2, v_v_939_);
lean_ctor_set(v___x_1091_, 1, v_k_938_);
lean_ctor_set(v___x_1091_, 0, v___x_1146_);
v___x_1148_ = v___x_1091_;
goto v_reusejp_1147_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v___x_1146_);
lean_ctor_set(v_reuseFailAlloc_1152_, 1, v_k_938_);
lean_ctor_set(v_reuseFailAlloc_1152_, 2, v_v_939_);
lean_ctor_set(v_reuseFailAlloc_1152_, 3, v_l_940_);
lean_ctor_set(v_reuseFailAlloc_1152_, 4, v_l_1113_);
v___x_1148_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1147_;
}
v_reusejp_1147_:
{
lean_object* v___x_1149_; 
v___x_1149_ = lean_nat_add(v___x_947_, v_size_1097_);
if (lean_obj_tag(v_r_1114_) == 0)
{
lean_object* v_size_1150_; 
v_size_1150_ = lean_ctor_get(v_r_1114_, 0);
lean_inc(v_size_1150_);
v___y_1124_ = v___x_1149_;
v___y_1125_ = v___x_1148_;
v___y_1126_ = v_size_1150_;
goto v___jp_1123_;
}
else
{
lean_object* v___x_1151_; 
v___x_1151_ = lean_unsigned_to_nat(0u);
v___y_1124_ = v___x_1149_;
v___y_1125_ = v___x_1148_;
v___y_1126_ = v___x_1151_;
goto v___jp_1123_;
}
}
}
}
}
else
{
lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1166_; 
v___x_1161_ = lean_nat_add(v___x_947_, v_size_937_);
lean_dec(v_size_937_);
v___x_1162_ = lean_nat_add(v___x_1161_, v_size_1097_);
lean_dec(v___x_1161_);
v___x_1163_ = lean_nat_add(v___x_947_, v_size_1097_);
v___x_1164_ = lean_nat_add(v___x_1163_, v_size_1110_);
lean_dec(v___x_1163_);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_tree_1094_);
lean_ctor_set(v___x_1091_, 3, v_r_941_);
lean_ctor_set(v___x_1091_, 2, v_v_1096_);
lean_ctor_set(v___x_1091_, 1, v_k_1095_);
lean_ctor_set(v___x_1091_, 0, v___x_1164_);
v___x_1166_ = v___x_1091_;
goto v_reusejp_1165_;
}
else
{
lean_object* v_reuseFailAlloc_1170_; 
v_reuseFailAlloc_1170_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1170_, 0, v___x_1164_);
lean_ctor_set(v_reuseFailAlloc_1170_, 1, v_k_1095_);
lean_ctor_set(v_reuseFailAlloc_1170_, 2, v_v_1096_);
lean_ctor_set(v_reuseFailAlloc_1170_, 3, v_r_941_);
lean_ctor_set(v_reuseFailAlloc_1170_, 4, v_tree_1094_);
v___x_1166_ = v_reuseFailAlloc_1170_;
goto v_reusejp_1165_;
}
v_reusejp_1165_:
{
lean_object* v___x_1168_; 
if (v_isShared_1108_ == 0)
{
lean_ctor_set(v___x_1107_, 4, v___x_1166_);
lean_ctor_set(v___x_1107_, 0, v___x_1162_);
v___x_1168_ = v___x_1107_;
goto v_reusejp_1167_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v___x_1162_);
lean_ctor_set(v_reuseFailAlloc_1169_, 1, v_k_938_);
lean_ctor_set(v_reuseFailAlloc_1169_, 2, v_v_939_);
lean_ctor_set(v_reuseFailAlloc_1169_, 3, v_l_940_);
lean_ctor_set(v_reuseFailAlloc_1169_, 4, v___x_1166_);
v___x_1168_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1167_;
}
v_reusejp_1167_:
{
return v___x_1168_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_l_940_) == 0)
{
lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1200_; 
lean_inc_ref(v_l_940_);
lean_inc(v_v_939_);
lean_inc(v_k_938_);
lean_inc(v_size_937_);
v_isSharedCheck_1200_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1200_ == 0)
{
lean_object* v_unused_1201_; lean_object* v_unused_1202_; lean_object* v_unused_1203_; lean_object* v_unused_1204_; lean_object* v_unused_1205_; 
v_unused_1201_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1201_);
v_unused_1202_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1202_);
v_unused_1203_ = lean_ctor_get(v_l_757_, 2);
lean_dec(v_unused_1203_);
v_unused_1204_ = lean_ctor_get(v_l_757_, 1);
lean_dec(v_unused_1204_);
v_unused_1205_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1205_);
v___x_1178_ = v_l_757_;
v_isShared_1179_ = v_isSharedCheck_1200_;
goto v_resetjp_1177_;
}
else
{
lean_dec(v_l_757_);
v___x_1178_ = lean_box(0);
v_isShared_1179_ = v_isSharedCheck_1200_;
goto v_resetjp_1177_;
}
v_resetjp_1177_:
{
if (lean_obj_tag(v_r_941_) == 0)
{
lean_object* v_k_1180_; lean_object* v_v_1181_; lean_object* v_size_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1186_; 
v_k_1180_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_k_1180_);
v_v_1181_ = lean_ctor_get(v___x_1093_, 1);
lean_inc(v_v_1181_);
lean_dec_ref(v___x_1093_);
v_size_1182_ = lean_ctor_get(v_r_941_, 0);
v___x_1183_ = lean_nat_add(v___x_947_, v_size_937_);
lean_dec(v_size_937_);
v___x_1184_ = lean_nat_add(v___x_947_, v_size_1182_);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_tree_1094_);
lean_ctor_set(v___x_1091_, 3, v_r_941_);
lean_ctor_set(v___x_1091_, 2, v_v_1181_);
lean_ctor_set(v___x_1091_, 1, v_k_1180_);
lean_ctor_set(v___x_1091_, 0, v___x_1184_);
v___x_1186_ = v___x_1091_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v___x_1184_);
lean_ctor_set(v_reuseFailAlloc_1190_, 1, v_k_1180_);
lean_ctor_set(v_reuseFailAlloc_1190_, 2, v_v_1181_);
lean_ctor_set(v_reuseFailAlloc_1190_, 3, v_r_941_);
lean_ctor_set(v_reuseFailAlloc_1190_, 4, v_tree_1094_);
v___x_1186_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
lean_object* v___x_1188_; 
if (v_isShared_1179_ == 0)
{
lean_ctor_set(v___x_1178_, 4, v___x_1186_);
lean_ctor_set(v___x_1178_, 0, v___x_1183_);
v___x_1188_ = v___x_1178_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v___x_1183_);
lean_ctor_set(v_reuseFailAlloc_1189_, 1, v_k_938_);
lean_ctor_set(v_reuseFailAlloc_1189_, 2, v_v_939_);
lean_ctor_set(v_reuseFailAlloc_1189_, 3, v_l_940_);
lean_ctor_set(v_reuseFailAlloc_1189_, 4, v___x_1186_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
return v___x_1188_;
}
}
}
else
{
lean_object* v_k_1191_; lean_object* v_v_1192_; lean_object* v___x_1193_; lean_object* v___x_1195_; 
lean_dec(v_size_937_);
v_k_1191_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_k_1191_);
v_v_1192_ = lean_ctor_get(v___x_1093_, 1);
lean_inc(v_v_1192_);
lean_dec_ref(v___x_1093_);
v___x_1193_ = lean_unsigned_to_nat(3u);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_r_941_);
lean_ctor_set(v___x_1091_, 3, v_r_941_);
lean_ctor_set(v___x_1091_, 2, v_v_1192_);
lean_ctor_set(v___x_1091_, 1, v_k_1191_);
lean_ctor_set(v___x_1091_, 0, v___x_947_);
v___x_1195_ = v___x_1091_;
goto v_reusejp_1194_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_1199_, 1, v_k_1191_);
lean_ctor_set(v_reuseFailAlloc_1199_, 2, v_v_1192_);
lean_ctor_set(v_reuseFailAlloc_1199_, 3, v_r_941_);
lean_ctor_set(v_reuseFailAlloc_1199_, 4, v_r_941_);
v___x_1195_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1194_;
}
v_reusejp_1194_:
{
lean_object* v___x_1197_; 
if (v_isShared_1179_ == 0)
{
lean_ctor_set(v___x_1178_, 4, v___x_1195_);
lean_ctor_set(v___x_1178_, 0, v___x_1193_);
v___x_1197_ = v___x_1178_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v___x_1193_);
lean_ctor_set(v_reuseFailAlloc_1198_, 1, v_k_938_);
lean_ctor_set(v_reuseFailAlloc_1198_, 2, v_v_939_);
lean_ctor_set(v_reuseFailAlloc_1198_, 3, v_l_940_);
lean_ctor_set(v_reuseFailAlloc_1198_, 4, v___x_1195_);
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
if (lean_obj_tag(v_r_941_) == 0)
{
lean_object* v___x_1207_; uint8_t v_isShared_1208_; uint8_t v_isSharedCheck_1230_; 
lean_inc(v_l_940_);
lean_inc(v_v_939_);
lean_inc(v_k_938_);
v_isSharedCheck_1230_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1230_ == 0)
{
lean_object* v_unused_1231_; lean_object* v_unused_1232_; lean_object* v_unused_1233_; lean_object* v_unused_1234_; lean_object* v_unused_1235_; 
v_unused_1231_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1231_);
v_unused_1232_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1232_);
v_unused_1233_ = lean_ctor_get(v_l_757_, 2);
lean_dec(v_unused_1233_);
v_unused_1234_ = lean_ctor_get(v_l_757_, 1);
lean_dec(v_unused_1234_);
v_unused_1235_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1235_);
v___x_1207_ = v_l_757_;
v_isShared_1208_ = v_isSharedCheck_1230_;
goto v_resetjp_1206_;
}
else
{
lean_dec(v_l_757_);
v___x_1207_ = lean_box(0);
v_isShared_1208_ = v_isSharedCheck_1230_;
goto v_resetjp_1206_;
}
v_resetjp_1206_:
{
lean_object* v_k_1209_; lean_object* v_v_1210_; lean_object* v_k_1211_; lean_object* v_v_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1226_; 
v_k_1209_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_k_1209_);
v_v_1210_ = lean_ctor_get(v___x_1093_, 1);
lean_inc(v_v_1210_);
lean_dec_ref(v___x_1093_);
v_k_1211_ = lean_ctor_get(v_r_941_, 1);
v_v_1212_ = lean_ctor_get(v_r_941_, 2);
v_isSharedCheck_1226_ = !lean_is_exclusive(v_r_941_);
if (v_isSharedCheck_1226_ == 0)
{
lean_object* v_unused_1227_; lean_object* v_unused_1228_; lean_object* v_unused_1229_; 
v_unused_1227_ = lean_ctor_get(v_r_941_, 4);
lean_dec(v_unused_1227_);
v_unused_1228_ = lean_ctor_get(v_r_941_, 3);
lean_dec(v_unused_1228_);
v_unused_1229_ = lean_ctor_get(v_r_941_, 0);
lean_dec(v_unused_1229_);
v___x_1214_ = v_r_941_;
v_isShared_1215_ = v_isSharedCheck_1226_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_v_1212_);
lean_inc(v_k_1211_);
lean_dec(v_r_941_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1226_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
lean_object* v___x_1216_; lean_object* v___x_1218_; 
v___x_1216_ = lean_unsigned_to_nat(3u);
if (v_isShared_1215_ == 0)
{
lean_ctor_set(v___x_1214_, 4, v_l_940_);
lean_ctor_set(v___x_1214_, 3, v_l_940_);
lean_ctor_set(v___x_1214_, 2, v_v_939_);
lean_ctor_set(v___x_1214_, 1, v_k_938_);
lean_ctor_set(v___x_1214_, 0, v___x_947_);
v___x_1218_ = v___x_1214_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1225_; 
v_reuseFailAlloc_1225_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1225_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_1225_, 1, v_k_938_);
lean_ctor_set(v_reuseFailAlloc_1225_, 2, v_v_939_);
lean_ctor_set(v_reuseFailAlloc_1225_, 3, v_l_940_);
lean_ctor_set(v_reuseFailAlloc_1225_, 4, v_l_940_);
v___x_1218_ = v_reuseFailAlloc_1225_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
lean_object* v___x_1220_; 
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_l_940_);
lean_ctor_set(v___x_1091_, 3, v_l_940_);
lean_ctor_set(v___x_1091_, 2, v_v_1210_);
lean_ctor_set(v___x_1091_, 1, v_k_1209_);
lean_ctor_set(v___x_1091_, 0, v___x_947_);
v___x_1220_ = v___x_1091_;
goto v_reusejp_1219_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_1224_, 1, v_k_1209_);
lean_ctor_set(v_reuseFailAlloc_1224_, 2, v_v_1210_);
lean_ctor_set(v_reuseFailAlloc_1224_, 3, v_l_940_);
lean_ctor_set(v_reuseFailAlloc_1224_, 4, v_l_940_);
v___x_1220_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1219_;
}
v_reusejp_1219_:
{
lean_object* v___x_1222_; 
if (v_isShared_1208_ == 0)
{
lean_ctor_set(v___x_1207_, 4, v___x_1220_);
lean_ctor_set(v___x_1207_, 3, v___x_1218_);
lean_ctor_set(v___x_1207_, 2, v_v_1212_);
lean_ctor_set(v___x_1207_, 1, v_k_1211_);
lean_ctor_set(v___x_1207_, 0, v___x_1216_);
v___x_1222_ = v___x_1207_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v___x_1216_);
lean_ctor_set(v_reuseFailAlloc_1223_, 1, v_k_1211_);
lean_ctor_set(v_reuseFailAlloc_1223_, 2, v_v_1212_);
lean_ctor_set(v_reuseFailAlloc_1223_, 3, v___x_1218_);
lean_ctor_set(v_reuseFailAlloc_1223_, 4, v___x_1220_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
}
}
else
{
lean_object* v_k_1236_; lean_object* v_v_1237_; lean_object* v___x_1238_; lean_object* v___x_1240_; 
v_k_1236_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_k_1236_);
v_v_1237_ = lean_ctor_get(v___x_1093_, 1);
lean_inc(v_v_1237_);
lean_dec_ref(v___x_1093_);
v___x_1238_ = lean_unsigned_to_nat(2u);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 4, v_r_941_);
lean_ctor_set(v___x_1091_, 3, v_l_757_);
lean_ctor_set(v___x_1091_, 2, v_v_1237_);
lean_ctor_set(v___x_1091_, 1, v_k_1236_);
lean_ctor_set(v___x_1091_, 0, v___x_1238_);
v___x_1240_ = v___x_1091_;
goto v_reusejp_1239_;
}
else
{
lean_object* v_reuseFailAlloc_1241_; 
v_reuseFailAlloc_1241_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1241_, 0, v___x_1238_);
lean_ctor_set(v_reuseFailAlloc_1241_, 1, v_k_1236_);
lean_ctor_set(v_reuseFailAlloc_1241_, 2, v_v_1237_);
lean_ctor_set(v_reuseFailAlloc_1241_, 3, v_l_757_);
lean_ctor_set(v_reuseFailAlloc_1241_, 4, v_r_941_);
v___x_1240_ = v_reuseFailAlloc_1241_;
goto v_reusejp_1239_;
}
v_reusejp_1239_:
{
return v___x_1240_;
}
}
}
}
}
}
}
else
{
return v_l_757_;
}
}
else
{
return v_r_758_;
}
}
default: 
{
lean_object* v_impl_1248_; lean_object* v___x_1249_; 
v_impl_1248_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(v_k_753_, v_r_758_);
v___x_1249_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_1248_) == 0)
{
if (lean_obj_tag(v_l_757_) == 0)
{
lean_object* v_size_1250_; lean_object* v_size_1251_; lean_object* v_k_1252_; lean_object* v_v_1253_; lean_object* v_l_1254_; lean_object* v_r_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; uint8_t v___x_1258_; 
v_size_1250_ = lean_ctor_get(v_impl_1248_, 0);
lean_inc(v_size_1250_);
v_size_1251_ = lean_ctor_get(v_l_757_, 0);
v_k_1252_ = lean_ctor_get(v_l_757_, 1);
v_v_1253_ = lean_ctor_get(v_l_757_, 2);
v_l_1254_ = lean_ctor_get(v_l_757_, 3);
v_r_1255_ = lean_ctor_get(v_l_757_, 4);
lean_inc(v_r_1255_);
v___x_1256_ = lean_unsigned_to_nat(3u);
v___x_1257_ = lean_nat_mul(v___x_1256_, v_size_1250_);
v___x_1258_ = lean_nat_dec_lt(v___x_1257_, v_size_1251_);
lean_dec(v___x_1257_);
if (v___x_1258_ == 0)
{
lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1262_; 
lean_dec(v_r_1255_);
v___x_1259_ = lean_nat_add(v___x_1249_, v_size_1251_);
v___x_1260_ = lean_nat_add(v___x_1259_, v_size_1250_);
lean_dec(v_size_1250_);
lean_dec(v___x_1259_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_impl_1248_);
lean_ctor_set(v___x_760_, 0, v___x_1260_);
v___x_1262_ = v___x_760_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v___x_1260_);
lean_ctor_set(v_reuseFailAlloc_1263_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1263_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1263_, 3, v_l_757_);
lean_ctor_set(v_reuseFailAlloc_1263_, 4, v_impl_1248_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
else
{
lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1329_; 
lean_inc(v_l_1254_);
lean_inc(v_v_1253_);
lean_inc(v_k_1252_);
lean_inc(v_size_1251_);
v_isSharedCheck_1329_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1329_ == 0)
{
lean_object* v_unused_1330_; lean_object* v_unused_1331_; lean_object* v_unused_1332_; lean_object* v_unused_1333_; lean_object* v_unused_1334_; 
v_unused_1330_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1330_);
v_unused_1331_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1331_);
v_unused_1332_ = lean_ctor_get(v_l_757_, 2);
lean_dec(v_unused_1332_);
v_unused_1333_ = lean_ctor_get(v_l_757_, 1);
lean_dec(v_unused_1333_);
v_unused_1334_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1334_);
v___x_1265_ = v_l_757_;
v_isShared_1266_ = v_isSharedCheck_1329_;
goto v_resetjp_1264_;
}
else
{
lean_dec(v_l_757_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1329_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v_size_1267_; lean_object* v_size_1268_; lean_object* v_k_1269_; lean_object* v_v_1270_; lean_object* v_l_1271_; lean_object* v_r_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; uint8_t v___x_1275_; 
v_size_1267_ = lean_ctor_get(v_l_1254_, 0);
v_size_1268_ = lean_ctor_get(v_r_1255_, 0);
v_k_1269_ = lean_ctor_get(v_r_1255_, 1);
v_v_1270_ = lean_ctor_get(v_r_1255_, 2);
v_l_1271_ = lean_ctor_get(v_r_1255_, 3);
v_r_1272_ = lean_ctor_get(v_r_1255_, 4);
v___x_1273_ = lean_unsigned_to_nat(2u);
v___x_1274_ = lean_nat_mul(v___x_1273_, v_size_1267_);
v___x_1275_ = lean_nat_dec_lt(v_size_1268_, v___x_1274_);
lean_dec(v___x_1274_);
if (v___x_1275_ == 0)
{
lean_object* v___x_1277_; uint8_t v_isShared_1278_; uint8_t v_isSharedCheck_1304_; 
lean_inc(v_r_1272_);
lean_inc(v_l_1271_);
lean_inc(v_v_1270_);
lean_inc(v_k_1269_);
v_isSharedCheck_1304_ = !lean_is_exclusive(v_r_1255_);
if (v_isSharedCheck_1304_ == 0)
{
lean_object* v_unused_1305_; lean_object* v_unused_1306_; lean_object* v_unused_1307_; lean_object* v_unused_1308_; lean_object* v_unused_1309_; 
v_unused_1305_ = lean_ctor_get(v_r_1255_, 4);
lean_dec(v_unused_1305_);
v_unused_1306_ = lean_ctor_get(v_r_1255_, 3);
lean_dec(v_unused_1306_);
v_unused_1307_ = lean_ctor_get(v_r_1255_, 2);
lean_dec(v_unused_1307_);
v_unused_1308_ = lean_ctor_get(v_r_1255_, 1);
lean_dec(v_unused_1308_);
v_unused_1309_ = lean_ctor_get(v_r_1255_, 0);
lean_dec(v_unused_1309_);
v___x_1277_ = v_r_1255_;
v_isShared_1278_ = v_isSharedCheck_1304_;
goto v_resetjp_1276_;
}
else
{
lean_dec(v_r_1255_);
v___x_1277_ = lean_box(0);
v_isShared_1278_ = v_isSharedCheck_1304_;
goto v_resetjp_1276_;
}
v_resetjp_1276_:
{
lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___y_1282_; lean_object* v___y_1283_; lean_object* v___y_1284_; lean_object* v___x_1292_; lean_object* v___y_1294_; 
v___x_1279_ = lean_nat_add(v___x_1249_, v_size_1251_);
lean_dec(v_size_1251_);
v___x_1280_ = lean_nat_add(v___x_1279_, v_size_1250_);
lean_dec(v___x_1279_);
v___x_1292_ = lean_nat_add(v___x_1249_, v_size_1267_);
if (lean_obj_tag(v_l_1271_) == 0)
{
lean_object* v_size_1302_; 
v_size_1302_ = lean_ctor_get(v_l_1271_, 0);
lean_inc(v_size_1302_);
v___y_1294_ = v_size_1302_;
goto v___jp_1293_;
}
else
{
lean_object* v___x_1303_; 
v___x_1303_ = lean_unsigned_to_nat(0u);
v___y_1294_ = v___x_1303_;
goto v___jp_1293_;
}
v___jp_1281_:
{
lean_object* v___x_1285_; lean_object* v___x_1287_; 
v___x_1285_ = lean_nat_add(v___y_1282_, v___y_1284_);
lean_dec(v___y_1284_);
lean_dec(v___y_1282_);
if (v_isShared_1278_ == 0)
{
lean_ctor_set(v___x_1277_, 4, v_impl_1248_);
lean_ctor_set(v___x_1277_, 3, v_r_1272_);
lean_ctor_set(v___x_1277_, 2, v_v_756_);
lean_ctor_set(v___x_1277_, 1, v_k_755_);
lean_ctor_set(v___x_1277_, 0, v___x_1285_);
v___x_1287_ = v___x_1277_;
goto v_reusejp_1286_;
}
else
{
lean_object* v_reuseFailAlloc_1291_; 
v_reuseFailAlloc_1291_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1291_, 0, v___x_1285_);
lean_ctor_set(v_reuseFailAlloc_1291_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1291_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1291_, 3, v_r_1272_);
lean_ctor_set(v_reuseFailAlloc_1291_, 4, v_impl_1248_);
v___x_1287_ = v_reuseFailAlloc_1291_;
goto v_reusejp_1286_;
}
v_reusejp_1286_:
{
lean_object* v___x_1289_; 
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 4, v___x_1287_);
lean_ctor_set(v___x_1265_, 3, v___y_1283_);
lean_ctor_set(v___x_1265_, 2, v_v_1270_);
lean_ctor_set(v___x_1265_, 1, v_k_1269_);
lean_ctor_set(v___x_1265_, 0, v___x_1280_);
v___x_1289_ = v___x_1265_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v___x_1280_);
lean_ctor_set(v_reuseFailAlloc_1290_, 1, v_k_1269_);
lean_ctor_set(v_reuseFailAlloc_1290_, 2, v_v_1270_);
lean_ctor_set(v_reuseFailAlloc_1290_, 3, v___y_1283_);
lean_ctor_set(v_reuseFailAlloc_1290_, 4, v___x_1287_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
}
v___jp_1293_:
{
lean_object* v___x_1295_; lean_object* v___x_1297_; 
v___x_1295_ = lean_nat_add(v___x_1292_, v___y_1294_);
lean_dec(v___y_1294_);
lean_dec(v___x_1292_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_l_1271_);
lean_ctor_set(v___x_760_, 3, v_l_1254_);
lean_ctor_set(v___x_760_, 2, v_v_1253_);
lean_ctor_set(v___x_760_, 1, v_k_1252_);
lean_ctor_set(v___x_760_, 0, v___x_1295_);
v___x_1297_ = v___x_760_;
goto v_reusejp_1296_;
}
else
{
lean_object* v_reuseFailAlloc_1301_; 
v_reuseFailAlloc_1301_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1301_, 0, v___x_1295_);
lean_ctor_set(v_reuseFailAlloc_1301_, 1, v_k_1252_);
lean_ctor_set(v_reuseFailAlloc_1301_, 2, v_v_1253_);
lean_ctor_set(v_reuseFailAlloc_1301_, 3, v_l_1254_);
lean_ctor_set(v_reuseFailAlloc_1301_, 4, v_l_1271_);
v___x_1297_ = v_reuseFailAlloc_1301_;
goto v_reusejp_1296_;
}
v_reusejp_1296_:
{
lean_object* v___x_1298_; 
v___x_1298_ = lean_nat_add(v___x_1249_, v_size_1250_);
lean_dec(v_size_1250_);
if (lean_obj_tag(v_r_1272_) == 0)
{
lean_object* v_size_1299_; 
v_size_1299_ = lean_ctor_get(v_r_1272_, 0);
lean_inc(v_size_1299_);
v___y_1282_ = v___x_1298_;
v___y_1283_ = v___x_1297_;
v___y_1284_ = v_size_1299_;
goto v___jp_1281_;
}
else
{
lean_object* v___x_1300_; 
v___x_1300_ = lean_unsigned_to_nat(0u);
v___y_1282_ = v___x_1298_;
v___y_1283_ = v___x_1297_;
v___y_1284_ = v___x_1300_;
goto v___jp_1281_;
}
}
}
}
}
else
{
lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1315_; 
lean_del_object(v___x_760_);
v___x_1310_ = lean_nat_add(v___x_1249_, v_size_1251_);
lean_dec(v_size_1251_);
v___x_1311_ = lean_nat_add(v___x_1310_, v_size_1250_);
lean_dec(v___x_1310_);
v___x_1312_ = lean_nat_add(v___x_1249_, v_size_1250_);
lean_dec(v_size_1250_);
v___x_1313_ = lean_nat_add(v___x_1312_, v_size_1268_);
lean_dec(v___x_1312_);
lean_inc_ref(v_impl_1248_);
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 4, v_impl_1248_);
lean_ctor_set(v___x_1265_, 3, v_r_1255_);
lean_ctor_set(v___x_1265_, 2, v_v_756_);
lean_ctor_set(v___x_1265_, 1, v_k_755_);
lean_ctor_set(v___x_1265_, 0, v___x_1313_);
v___x_1315_ = v___x_1265_;
goto v_reusejp_1314_;
}
else
{
lean_object* v_reuseFailAlloc_1328_; 
v_reuseFailAlloc_1328_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1328_, 0, v___x_1313_);
lean_ctor_set(v_reuseFailAlloc_1328_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1328_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1328_, 3, v_r_1255_);
lean_ctor_set(v_reuseFailAlloc_1328_, 4, v_impl_1248_);
v___x_1315_ = v_reuseFailAlloc_1328_;
goto v_reusejp_1314_;
}
v_reusejp_1314_:
{
lean_object* v___x_1317_; uint8_t v_isShared_1318_; uint8_t v_isSharedCheck_1322_; 
v_isSharedCheck_1322_ = !lean_is_exclusive(v_impl_1248_);
if (v_isSharedCheck_1322_ == 0)
{
lean_object* v_unused_1323_; lean_object* v_unused_1324_; lean_object* v_unused_1325_; lean_object* v_unused_1326_; lean_object* v_unused_1327_; 
v_unused_1323_ = lean_ctor_get(v_impl_1248_, 4);
lean_dec(v_unused_1323_);
v_unused_1324_ = lean_ctor_get(v_impl_1248_, 3);
lean_dec(v_unused_1324_);
v_unused_1325_ = lean_ctor_get(v_impl_1248_, 2);
lean_dec(v_unused_1325_);
v_unused_1326_ = lean_ctor_get(v_impl_1248_, 1);
lean_dec(v_unused_1326_);
v_unused_1327_ = lean_ctor_get(v_impl_1248_, 0);
lean_dec(v_unused_1327_);
v___x_1317_ = v_impl_1248_;
v_isShared_1318_ = v_isSharedCheck_1322_;
goto v_resetjp_1316_;
}
else
{
lean_dec(v_impl_1248_);
v___x_1317_ = lean_box(0);
v_isShared_1318_ = v_isSharedCheck_1322_;
goto v_resetjp_1316_;
}
v_resetjp_1316_:
{
lean_object* v___x_1320_; 
if (v_isShared_1318_ == 0)
{
lean_ctor_set(v___x_1317_, 4, v___x_1315_);
lean_ctor_set(v___x_1317_, 3, v_l_1254_);
lean_ctor_set(v___x_1317_, 2, v_v_1253_);
lean_ctor_set(v___x_1317_, 1, v_k_1252_);
lean_ctor_set(v___x_1317_, 0, v___x_1311_);
v___x_1320_ = v___x_1317_;
goto v_reusejp_1319_;
}
else
{
lean_object* v_reuseFailAlloc_1321_; 
v_reuseFailAlloc_1321_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1321_, 0, v___x_1311_);
lean_ctor_set(v_reuseFailAlloc_1321_, 1, v_k_1252_);
lean_ctor_set(v_reuseFailAlloc_1321_, 2, v_v_1253_);
lean_ctor_set(v_reuseFailAlloc_1321_, 3, v_l_1254_);
lean_ctor_set(v_reuseFailAlloc_1321_, 4, v___x_1315_);
v___x_1320_ = v_reuseFailAlloc_1321_;
goto v_reusejp_1319_;
}
v_reusejp_1319_:
{
return v___x_1320_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_1335_; lean_object* v___x_1336_; lean_object* v___x_1338_; 
v_size_1335_ = lean_ctor_get(v_impl_1248_, 0);
lean_inc(v_size_1335_);
v___x_1336_ = lean_nat_add(v___x_1249_, v_size_1335_);
lean_dec(v_size_1335_);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_impl_1248_);
lean_ctor_set(v___x_760_, 0, v___x_1336_);
v___x_1338_ = v___x_760_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v___x_1336_);
lean_ctor_set(v_reuseFailAlloc_1339_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1339_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1339_, 3, v_l_757_);
lean_ctor_set(v_reuseFailAlloc_1339_, 4, v_impl_1248_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
}
else
{
if (lean_obj_tag(v_l_757_) == 0)
{
lean_object* v_l_1340_; 
v_l_1340_ = lean_ctor_get(v_l_757_, 3);
if (lean_obj_tag(v_l_1340_) == 0)
{
lean_object* v_r_1341_; 
lean_inc_ref(v_l_1340_);
v_r_1341_ = lean_ctor_get(v_l_757_, 4);
lean_inc(v_r_1341_);
if (lean_obj_tag(v_r_1341_) == 0)
{
lean_object* v_size_1342_; lean_object* v_k_1343_; lean_object* v_v_1344_; lean_object* v___x_1346_; uint8_t v_isShared_1347_; uint8_t v_isSharedCheck_1357_; 
v_size_1342_ = lean_ctor_get(v_l_757_, 0);
v_k_1343_ = lean_ctor_get(v_l_757_, 1);
v_v_1344_ = lean_ctor_get(v_l_757_, 2);
v_isSharedCheck_1357_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1357_ == 0)
{
lean_object* v_unused_1358_; lean_object* v_unused_1359_; 
v_unused_1358_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1358_);
v_unused_1359_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1359_);
v___x_1346_ = v_l_757_;
v_isShared_1347_ = v_isSharedCheck_1357_;
goto v_resetjp_1345_;
}
else
{
lean_inc(v_v_1344_);
lean_inc(v_k_1343_);
lean_inc(v_size_1342_);
lean_dec(v_l_757_);
v___x_1346_ = lean_box(0);
v_isShared_1347_ = v_isSharedCheck_1357_;
goto v_resetjp_1345_;
}
v_resetjp_1345_:
{
lean_object* v_size_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1352_; 
v_size_1348_ = lean_ctor_get(v_r_1341_, 0);
v___x_1349_ = lean_nat_add(v___x_1249_, v_size_1342_);
lean_dec(v_size_1342_);
v___x_1350_ = lean_nat_add(v___x_1249_, v_size_1348_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 4, v_impl_1248_);
lean_ctor_set(v___x_1346_, 3, v_r_1341_);
lean_ctor_set(v___x_1346_, 2, v_v_756_);
lean_ctor_set(v___x_1346_, 1, v_k_755_);
lean_ctor_set(v___x_1346_, 0, v___x_1350_);
v___x_1352_ = v___x_1346_;
goto v_reusejp_1351_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v___x_1350_);
lean_ctor_set(v_reuseFailAlloc_1356_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1356_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1356_, 3, v_r_1341_);
lean_ctor_set(v_reuseFailAlloc_1356_, 4, v_impl_1248_);
v___x_1352_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1351_;
}
v_reusejp_1351_:
{
lean_object* v___x_1354_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v___x_1352_);
lean_ctor_set(v___x_760_, 3, v_l_1340_);
lean_ctor_set(v___x_760_, 2, v_v_1344_);
lean_ctor_set(v___x_760_, 1, v_k_1343_);
lean_ctor_set(v___x_760_, 0, v___x_1349_);
v___x_1354_ = v___x_760_;
goto v_reusejp_1353_;
}
else
{
lean_object* v_reuseFailAlloc_1355_; 
v_reuseFailAlloc_1355_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1355_, 0, v___x_1349_);
lean_ctor_set(v_reuseFailAlloc_1355_, 1, v_k_1343_);
lean_ctor_set(v_reuseFailAlloc_1355_, 2, v_v_1344_);
lean_ctor_set(v_reuseFailAlloc_1355_, 3, v_l_1340_);
lean_ctor_set(v_reuseFailAlloc_1355_, 4, v___x_1352_);
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
else
{
lean_object* v_k_1360_; lean_object* v_v_1361_; lean_object* v___x_1363_; uint8_t v_isShared_1364_; uint8_t v_isSharedCheck_1372_; 
v_k_1360_ = lean_ctor_get(v_l_757_, 1);
v_v_1361_ = lean_ctor_get(v_l_757_, 2);
v_isSharedCheck_1372_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1372_ == 0)
{
lean_object* v_unused_1373_; lean_object* v_unused_1374_; lean_object* v_unused_1375_; 
v_unused_1373_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1373_);
v_unused_1374_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1374_);
v_unused_1375_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1375_);
v___x_1363_ = v_l_757_;
v_isShared_1364_ = v_isSharedCheck_1372_;
goto v_resetjp_1362_;
}
else
{
lean_inc(v_v_1361_);
lean_inc(v_k_1360_);
lean_dec(v_l_757_);
v___x_1363_ = lean_box(0);
v_isShared_1364_ = v_isSharedCheck_1372_;
goto v_resetjp_1362_;
}
v_resetjp_1362_:
{
lean_object* v___x_1365_; lean_object* v___x_1367_; 
v___x_1365_ = lean_unsigned_to_nat(3u);
if (v_isShared_1364_ == 0)
{
lean_ctor_set(v___x_1363_, 3, v_r_1341_);
lean_ctor_set(v___x_1363_, 2, v_v_756_);
lean_ctor_set(v___x_1363_, 1, v_k_755_);
lean_ctor_set(v___x_1363_, 0, v___x_1249_);
v___x_1367_ = v___x_1363_;
goto v_reusejp_1366_;
}
else
{
lean_object* v_reuseFailAlloc_1371_; 
v_reuseFailAlloc_1371_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1371_, 0, v___x_1249_);
lean_ctor_set(v_reuseFailAlloc_1371_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1371_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1371_, 3, v_r_1341_);
lean_ctor_set(v_reuseFailAlloc_1371_, 4, v_r_1341_);
v___x_1367_ = v_reuseFailAlloc_1371_;
goto v_reusejp_1366_;
}
v_reusejp_1366_:
{
lean_object* v___x_1369_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v___x_1367_);
lean_ctor_set(v___x_760_, 3, v_l_1340_);
lean_ctor_set(v___x_760_, 2, v_v_1361_);
lean_ctor_set(v___x_760_, 1, v_k_1360_);
lean_ctor_set(v___x_760_, 0, v___x_1365_);
v___x_1369_ = v___x_760_;
goto v_reusejp_1368_;
}
else
{
lean_object* v_reuseFailAlloc_1370_; 
v_reuseFailAlloc_1370_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1370_, 0, v___x_1365_);
lean_ctor_set(v_reuseFailAlloc_1370_, 1, v_k_1360_);
lean_ctor_set(v_reuseFailAlloc_1370_, 2, v_v_1361_);
lean_ctor_set(v_reuseFailAlloc_1370_, 3, v_l_1340_);
lean_ctor_set(v_reuseFailAlloc_1370_, 4, v___x_1367_);
v___x_1369_ = v_reuseFailAlloc_1370_;
goto v_reusejp_1368_;
}
v_reusejp_1368_:
{
return v___x_1369_;
}
}
}
}
}
else
{
lean_object* v_r_1376_; 
v_r_1376_ = lean_ctor_get(v_l_757_, 4);
lean_inc(v_r_1376_);
if (lean_obj_tag(v_r_1376_) == 0)
{
lean_object* v_k_1377_; lean_object* v_v_1378_; lean_object* v___x_1380_; uint8_t v_isShared_1381_; uint8_t v_isSharedCheck_1401_; 
lean_inc(v_l_1340_);
v_k_1377_ = lean_ctor_get(v_l_757_, 1);
v_v_1378_ = lean_ctor_get(v_l_757_, 2);
v_isSharedCheck_1401_ = !lean_is_exclusive(v_l_757_);
if (v_isSharedCheck_1401_ == 0)
{
lean_object* v_unused_1402_; lean_object* v_unused_1403_; lean_object* v_unused_1404_; 
v_unused_1402_ = lean_ctor_get(v_l_757_, 4);
lean_dec(v_unused_1402_);
v_unused_1403_ = lean_ctor_get(v_l_757_, 3);
lean_dec(v_unused_1403_);
v_unused_1404_ = lean_ctor_get(v_l_757_, 0);
lean_dec(v_unused_1404_);
v___x_1380_ = v_l_757_;
v_isShared_1381_ = v_isSharedCheck_1401_;
goto v_resetjp_1379_;
}
else
{
lean_inc(v_v_1378_);
lean_inc(v_k_1377_);
lean_dec(v_l_757_);
v___x_1380_ = lean_box(0);
v_isShared_1381_ = v_isSharedCheck_1401_;
goto v_resetjp_1379_;
}
v_resetjp_1379_:
{
lean_object* v_k_1382_; lean_object* v_v_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1397_; 
v_k_1382_ = lean_ctor_get(v_r_1376_, 1);
v_v_1383_ = lean_ctor_get(v_r_1376_, 2);
v_isSharedCheck_1397_ = !lean_is_exclusive(v_r_1376_);
if (v_isSharedCheck_1397_ == 0)
{
lean_object* v_unused_1398_; lean_object* v_unused_1399_; lean_object* v_unused_1400_; 
v_unused_1398_ = lean_ctor_get(v_r_1376_, 4);
lean_dec(v_unused_1398_);
v_unused_1399_ = lean_ctor_get(v_r_1376_, 3);
lean_dec(v_unused_1399_);
v_unused_1400_ = lean_ctor_get(v_r_1376_, 0);
lean_dec(v_unused_1400_);
v___x_1385_ = v_r_1376_;
v_isShared_1386_ = v_isSharedCheck_1397_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_v_1383_);
lean_inc(v_k_1382_);
lean_dec(v_r_1376_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1397_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
lean_object* v___x_1387_; lean_object* v___x_1389_; 
v___x_1387_ = lean_unsigned_to_nat(3u);
if (v_isShared_1386_ == 0)
{
lean_ctor_set(v___x_1385_, 4, v_l_1340_);
lean_ctor_set(v___x_1385_, 3, v_l_1340_);
lean_ctor_set(v___x_1385_, 2, v_v_1378_);
lean_ctor_set(v___x_1385_, 1, v_k_1377_);
lean_ctor_set(v___x_1385_, 0, v___x_1249_);
v___x_1389_ = v___x_1385_;
goto v_reusejp_1388_;
}
else
{
lean_object* v_reuseFailAlloc_1396_; 
v_reuseFailAlloc_1396_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1396_, 0, v___x_1249_);
lean_ctor_set(v_reuseFailAlloc_1396_, 1, v_k_1377_);
lean_ctor_set(v_reuseFailAlloc_1396_, 2, v_v_1378_);
lean_ctor_set(v_reuseFailAlloc_1396_, 3, v_l_1340_);
lean_ctor_set(v_reuseFailAlloc_1396_, 4, v_l_1340_);
v___x_1389_ = v_reuseFailAlloc_1396_;
goto v_reusejp_1388_;
}
v_reusejp_1388_:
{
lean_object* v___x_1391_; 
if (v_isShared_1381_ == 0)
{
lean_ctor_set(v___x_1380_, 4, v_l_1340_);
lean_ctor_set(v___x_1380_, 2, v_v_756_);
lean_ctor_set(v___x_1380_, 1, v_k_755_);
lean_ctor_set(v___x_1380_, 0, v___x_1249_);
v___x_1391_ = v___x_1380_;
goto v_reusejp_1390_;
}
else
{
lean_object* v_reuseFailAlloc_1395_; 
v_reuseFailAlloc_1395_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1395_, 0, v___x_1249_);
lean_ctor_set(v_reuseFailAlloc_1395_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1395_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1395_, 3, v_l_1340_);
lean_ctor_set(v_reuseFailAlloc_1395_, 4, v_l_1340_);
v___x_1391_ = v_reuseFailAlloc_1395_;
goto v_reusejp_1390_;
}
v_reusejp_1390_:
{
lean_object* v___x_1393_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v___x_1391_);
lean_ctor_set(v___x_760_, 3, v___x_1389_);
lean_ctor_set(v___x_760_, 2, v_v_1383_);
lean_ctor_set(v___x_760_, 1, v_k_1382_);
lean_ctor_set(v___x_760_, 0, v___x_1387_);
v___x_1393_ = v___x_760_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v___x_1387_);
lean_ctor_set(v_reuseFailAlloc_1394_, 1, v_k_1382_);
lean_ctor_set(v_reuseFailAlloc_1394_, 2, v_v_1383_);
lean_ctor_set(v_reuseFailAlloc_1394_, 3, v___x_1389_);
lean_ctor_set(v_reuseFailAlloc_1394_, 4, v___x_1391_);
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
}
}
else
{
lean_object* v___x_1405_; lean_object* v___x_1407_; 
v___x_1405_ = lean_unsigned_to_nat(2u);
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_r_1376_);
lean_ctor_set(v___x_760_, 0, v___x_1405_);
v___x_1407_ = v___x_760_;
goto v_reusejp_1406_;
}
else
{
lean_object* v_reuseFailAlloc_1408_; 
v_reuseFailAlloc_1408_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1408_, 0, v___x_1405_);
lean_ctor_set(v_reuseFailAlloc_1408_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1408_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1408_, 3, v_l_757_);
lean_ctor_set(v_reuseFailAlloc_1408_, 4, v_r_1376_);
v___x_1407_ = v_reuseFailAlloc_1408_;
goto v_reusejp_1406_;
}
v_reusejp_1406_:
{
return v___x_1407_;
}
}
}
}
else
{
lean_object* v___x_1410_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 4, v_l_757_);
lean_ctor_set(v___x_760_, 0, v___x_1249_);
v___x_1410_ = v___x_760_;
goto v_reusejp_1409_;
}
else
{
lean_object* v_reuseFailAlloc_1411_; 
v_reuseFailAlloc_1411_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1411_, 0, v___x_1249_);
lean_ctor_set(v_reuseFailAlloc_1411_, 1, v_k_755_);
lean_ctor_set(v_reuseFailAlloc_1411_, 2, v_v_756_);
lean_ctor_set(v_reuseFailAlloc_1411_, 3, v_l_757_);
lean_ctor_set(v_reuseFailAlloc_1411_, 4, v_l_757_);
v___x_1410_ = v_reuseFailAlloc_1411_;
goto v_reusejp_1409_;
}
v_reusejp_1409_:
{
return v___x_1410_;
}
}
}
}
}
}
}
else
{
return v_t_754_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg___boxed(lean_object* v_k_1414_, lean_object* v_t_1415_){
_start:
{
lean_object* v_res_1416_; 
v_res_1416_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(v_k_1414_, v_t_1415_);
lean_dec(v_k_1414_);
return v_res_1416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(lean_object* v_init_1417_, lean_object* v_x_1418_){
_start:
{
if (lean_obj_tag(v_x_1418_) == 0)
{
lean_object* v_k_1419_; lean_object* v_l_1420_; lean_object* v_r_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; 
v_k_1419_ = lean_ctor_get(v_x_1418_, 1);
v_l_1420_ = lean_ctor_get(v_x_1418_, 3);
v_r_1421_ = lean_ctor_get(v_x_1418_, 4);
v___x_1422_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(v_init_1417_, v_l_1420_);
v___x_1423_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(v_k_1419_, v___x_1422_);
v_init_1417_ = v___x_1423_;
v_x_1418_ = v_r_1421_;
goto _start;
}
else
{
return v_init_1417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5___boxed(lean_object* v_init_1425_, lean_object* v_x_1426_){
_start:
{
lean_object* v_res_1427_; 
v_res_1427_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(v_init_1425_, v_x_1426_);
lean_dec(v_x_1426_);
return v_res_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15(lean_object* v_as_1428_, size_t v_i_1429_, size_t v_stop_1430_, lean_object* v_b_1431_){
_start:
{
lean_object* v___y_1433_; uint8_t v___x_1437_; 
v___x_1437_ = lean_usize_dec_eq(v_i_1429_, v_stop_1430_);
if (v___x_1437_ == 0)
{
lean_object* v___x_1438_; uint8_t v___x_1439_; 
v___x_1438_ = lean_array_uget_borrowed(v_as_1428_, v_i_1429_);
v___x_1439_ = lp_mathlib_Mathlib_Command_MinImports_isInitImport(v___x_1438_);
if (v___x_1439_ == 0)
{
lean_object* v___x_1440_; 
lean_inc(v___x_1438_);
v___x_1440_ = lean_array_push(v_b_1431_, v___x_1438_);
v___y_1433_ = v___x_1440_;
goto v___jp_1432_;
}
else
{
v___y_1433_ = v_b_1431_;
goto v___jp_1432_;
}
}
else
{
return v_b_1431_;
}
v___jp_1432_:
{
size_t v___x_1434_; size_t v___x_1435_; 
v___x_1434_ = ((size_t)1ULL);
v___x_1435_ = lean_usize_add(v_i_1429_, v___x_1434_);
v_i_1429_ = v___x_1435_;
v_b_1431_ = v___y_1433_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15___boxed(lean_object* v_as_1441_, lean_object* v_i_1442_, lean_object* v_stop_1443_, lean_object* v_b_1444_){
_start:
{
size_t v_i_boxed_1445_; size_t v_stop_boxed_1446_; lean_object* v_res_1447_; 
v_i_boxed_1445_ = lean_unbox_usize(v_i_1442_);
lean_dec(v_i_1442_);
v_stop_boxed_1446_ = lean_unbox_usize(v_stop_1443_);
lean_dec(v_stop_1443_);
v_res_1447_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15(v_as_1441_, v_i_boxed_1445_, v_stop_boxed_1446_, v_b_1444_);
lean_dec_ref(v_as_1441_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__4(lean_object* v_a_1448_, lean_object* v_a_1449_){
_start:
{
if (lean_obj_tag(v_a_1448_) == 0)
{
lean_object* v___x_1450_; 
v___x_1450_ = l_List_reverse___redArg(v_a_1449_);
return v___x_1450_;
}
else
{
lean_object* v_head_1451_; lean_object* v_tail_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1461_; 
v_head_1451_ = lean_ctor_get(v_a_1448_, 0);
v_tail_1452_ = lean_ctor_get(v_a_1448_, 1);
v_isSharedCheck_1461_ = !lean_is_exclusive(v_a_1448_);
if (v_isSharedCheck_1461_ == 0)
{
v___x_1454_ = v_a_1448_;
v_isShared_1455_ = v_isSharedCheck_1461_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_tail_1452_);
lean_inc(v_head_1451_);
lean_dec(v_a_1448_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1461_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1456_; lean_object* v___x_1458_; 
v___x_1456_ = l_Lean_MessageData_ofName(v_head_1451_);
if (v_isShared_1455_ == 0)
{
lean_ctor_set(v___x_1454_, 1, v_a_1449_);
lean_ctor_set(v___x_1454_, 0, v___x_1456_);
v___x_1458_ = v___x_1454_;
goto v_reusejp_1457_;
}
else
{
lean_object* v_reuseFailAlloc_1460_; 
v_reuseFailAlloc_1460_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1460_, 0, v___x_1456_);
lean_ctor_set(v_reuseFailAlloc_1460_, 1, v_a_1449_);
v___x_1458_ = v_reuseFailAlloc_1460_;
goto v_reusejp_1457_;
}
v_reusejp_1457_:
{
v_a_1448_ = v_tail_1452_;
v_a_1449_ = v___x_1458_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14(uint8_t v___y_1463_, size_t v_sz_1464_, size_t v_i_1465_, lean_object* v_bs_1466_){
_start:
{
uint8_t v___x_1467_; 
v___x_1467_ = lean_usize_dec_lt(v_i_1465_, v_sz_1464_);
if (v___x_1467_ == 0)
{
return v_bs_1466_;
}
else
{
lean_object* v_v_1468_; lean_object* v___x_1469_; lean_object* v_bs_x27_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; size_t v___x_1474_; size_t v___x_1475_; lean_object* v___x_1476_; 
v_v_1468_ = lean_array_uget(v_bs_1466_, v_i_1465_);
v___x_1469_ = lean_unsigned_to_nat(0u);
v_bs_x27_1470_ = lean_array_uset(v_bs_1466_, v_i_1465_, v___x_1469_);
v___x_1471_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14___closed__0));
v___x_1472_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_v_1468_, v___y_1463_);
v___x_1473_ = lean_string_append(v___x_1471_, v___x_1472_);
lean_dec_ref(v___x_1472_);
v___x_1474_ = ((size_t)1ULL);
v___x_1475_ = lean_usize_add(v_i_1465_, v___x_1474_);
v___x_1476_ = lean_array_uset(v_bs_x27_1470_, v_i_1465_, v___x_1473_);
v_i_1465_ = v___x_1475_;
v_bs_1466_ = v___x_1476_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14___boxed(lean_object* v___y_1478_, lean_object* v_sz_1479_, lean_object* v_i_1480_, lean_object* v_bs_1481_){
_start:
{
uint8_t v___y_41426__boxed_1482_; size_t v_sz_boxed_1483_; size_t v_i_boxed_1484_; lean_object* v_res_1485_; 
v___y_41426__boxed_1482_ = lean_unbox(v___y_1478_);
v_sz_boxed_1483_ = lean_unbox_usize(v_sz_1479_);
lean_dec(v_sz_1479_);
v_i_boxed_1484_ = lean_unbox_usize(v_i_1480_);
lean_dec(v_i_1480_);
v_res_1485_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14(v___y_41426__boxed_1482_, v_sz_boxed_1483_, v_i_boxed_1484_, v_bs_1481_);
return v_res_1485_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg(lean_object* v_xs_1486_, lean_object* v_ys_1487_, lean_object* v_x_1488_){
_start:
{
lean_object* v_zero_1489_; uint8_t v_isZero_1490_; 
v_zero_1489_ = lean_unsigned_to_nat(0u);
v_isZero_1490_ = lean_nat_dec_eq(v_x_1488_, v_zero_1489_);
if (v_isZero_1490_ == 1)
{
lean_dec(v_x_1488_);
return v_isZero_1490_;
}
else
{
lean_object* v_one_1491_; lean_object* v_n_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; uint8_t v___x_1495_; 
v_one_1491_ = lean_unsigned_to_nat(1u);
v_n_1492_ = lean_nat_sub(v_x_1488_, v_one_1491_);
lean_dec(v_x_1488_);
v___x_1493_ = lean_array_fget_borrowed(v_xs_1486_, v_n_1492_);
v___x_1494_ = lean_array_fget_borrowed(v_ys_1487_, v_n_1492_);
v___x_1495_ = lean_name_eq(v___x_1493_, v___x_1494_);
if (v___x_1495_ == 0)
{
lean_dec(v_n_1492_);
return v___x_1495_;
}
else
{
v_x_1488_ = v_n_1492_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg___boxed(lean_object* v_xs_1497_, lean_object* v_ys_1498_, lean_object* v_x_1499_){
_start:
{
uint8_t v_res_1500_; lean_object* v_r_1501_; 
v_res_1500_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg(v_xs_1497_, v_ys_1498_, v_x_1499_);
lean_dec_ref(v_ys_1498_);
lean_dec_ref(v_xs_1497_);
v_r_1501_ = lean_box(v_res_1500_);
return v_r_1501_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg(lean_object* v_xs_1502_, lean_object* v_ys_1503_, lean_object* v_x_1504_){
_start:
{
lean_object* v_zero_1505_; uint8_t v_isZero_1506_; 
v_zero_1505_ = lean_unsigned_to_nat(0u);
v_isZero_1506_ = lean_nat_dec_eq(v_x_1504_, v_zero_1505_);
if (v_isZero_1506_ == 1)
{
lean_dec(v_x_1504_);
return v_isZero_1506_;
}
else
{
lean_object* v_one_1507_; lean_object* v_n_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; uint8_t v___x_1511_; 
v_one_1507_ = lean_unsigned_to_nat(1u);
v_n_1508_ = lean_nat_sub(v_x_1504_, v_one_1507_);
lean_dec(v_x_1504_);
v___x_1509_ = lean_array_fget_borrowed(v_xs_1502_, v_n_1508_);
v___x_1510_ = lean_array_fget_borrowed(v_ys_1503_, v_n_1508_);
v___x_1511_ = lean_name_eq(v___x_1509_, v___x_1510_);
if (v___x_1511_ == 0)
{
lean_dec(v_n_1508_);
return v___x_1511_;
}
else
{
v_x_1504_ = v_n_1508_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg___boxed(lean_object* v_xs_1513_, lean_object* v_ys_1514_, lean_object* v_x_1515_){
_start:
{
uint8_t v_res_1516_; lean_object* v_r_1517_; 
v_res_1516_ = lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg(v_xs_1513_, v_ys_1514_, v_x_1515_);
lean_dec_ref(v_ys_1514_);
lean_dec_ref(v_xs_1513_);
v_r_1517_ = lean_box(v_res_1516_);
return v_r_1517_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7(lean_object* v_xs_1518_, lean_object* v_ys_1519_){
_start:
{
lean_object* v___x_1520_; lean_object* v___x_1521_; uint8_t v___x_1522_; 
v___x_1520_ = lean_array_get_size(v_xs_1518_);
v___x_1521_ = lean_array_get_size(v_ys_1519_);
v___x_1522_ = lean_nat_dec_eq(v___x_1520_, v___x_1521_);
if (v___x_1522_ == 0)
{
return v___x_1522_;
}
else
{
uint8_t v___x_1523_; 
v___x_1523_ = lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg(v_xs_1518_, v_ys_1519_, v___x_1520_);
return v___x_1523_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7___boxed(lean_object* v_xs_1524_, lean_object* v_ys_1525_){
_start:
{
uint8_t v_res_1526_; lean_object* v_r_1527_; 
v_res_1526_ = lp_mathlib_Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7(v_xs_1524_, v_ys_1525_);
lean_dec_ref(v_ys_1525_);
lean_dec_ref(v_xs_1524_);
v_r_1527_ = lean_box(v_res_1526_);
return v_r_1527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__1(lean_object* v___x_1528_, lean_object* v_x_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_){
_start:
{
lean_object* v___x_1533_; lean_object* v___x_1534_; 
v___x_1533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1533_, 0, v___x_1528_);
v___x_1534_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1534_, 0, v___x_1533_);
return v___x_1534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__1___boxed(lean_object* v___x_1535_, lean_object* v_x_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_){
_start:
{
lean_object* v_res_1540_; 
v_res_1540_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__1(v___x_1535_, v_x_1536_, v___y_1537_, v___y_1538_);
lean_dec(v___y_1538_);
lean_dec_ref(v___y_1537_);
return v_res_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12(lean_object* v_ref_1541_, lean_object* v_msgData_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_){
_start:
{
uint8_t v___x_1546_; uint8_t v___x_1547_; lean_object* v___x_1548_; 
v___x_1546_ = 1;
v___x_1547_ = 0;
v___x_1548_ = lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18(v_ref_1541_, v_msgData_1542_, v___x_1546_, v___x_1547_, v___y_1543_, v___y_1544_);
return v___x_1548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12___boxed(lean_object* v_ref_1549_, lean_object* v_msgData_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_){
_start:
{
lean_object* v_res_1554_; 
v_res_1554_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12(v_ref_1549_, v_msgData_1550_, v___y_1551_, v___y_1552_);
lean_dec(v___y_1552_);
lean_dec_ref(v___y_1551_);
lean_dec(v_ref_1549_);
return v_res_1554_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__0(lean_object* v_k_1555_, lean_object* v_x_1556_){
_start:
{
lean_object* v___x_1557_; uint8_t v___x_1558_; 
v___x_1557_ = l_Lean_Syntax_getId(v_x_1556_);
v___x_1558_ = lean_name_eq(v___x_1557_, v_k_1555_);
lean_dec(v___x_1557_);
return v___x_1558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__0___boxed(lean_object* v_k_1559_, lean_object* v_x_1560_){
_start:
{
uint8_t v_res_1561_; lean_object* v_r_1562_; 
v_res_1561_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__0(v_k_1559_, v_x_1560_);
lean_dec(v_x_1560_);
lean_dec(v_k_1559_);
v_r_1562_ = lean_box(v_res_1561_);
return v_r_1562_;
}
}
static lean_object* _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__1(void){
_start:
{
lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___x_1564_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__0));
v___x_1565_ = l_Lean_stringToMessageData(v___x_1564_);
return v___x_1565_;
}
}
static lean_object* _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__3(void){
_start:
{
lean_object* v___x_1567_; lean_object* v___x_1568_; 
v___x_1567_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__2));
v___x_1568_ = l_Lean_stringToMessageData(v___x_1567_);
return v___x_1568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13(lean_object* v_fst_1576_, uint8_t v___x_1577_, lean_object* v_init_1578_, lean_object* v_x_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_){
_start:
{
lean_object* v_d_1584_; 
if (lean_obj_tag(v_x_1579_) == 0)
{
lean_object* v_k_1587_; lean_object* v_l_1588_; lean_object* v_r_1589_; lean_object* v___x_1590_; 
v_k_1587_ = lean_ctor_get(v_x_1579_, 1);
lean_inc(v_k_1587_);
v_l_1588_ = lean_ctor_get(v_x_1579_, 3);
lean_inc(v_l_1588_);
v_r_1589_ = lean_ctor_get(v_x_1579_, 4);
lean_inc(v_r_1589_);
lean_dec_ref_known(v_x_1579_, 5);
lean_inc(v_fst_1576_);
v___x_1590_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13(v_fst_1576_, v___x_1577_, v_init_1578_, v_l_1588_, v___y_1580_, v___y_1581_);
if (lean_obj_tag(v___x_1590_) == 0)
{
lean_object* v_a_1591_; 
v_a_1591_ = lean_ctor_get(v___x_1590_, 0);
lean_inc(v_a_1591_);
lean_dec_ref_known(v___x_1590_, 1);
if (lean_obj_tag(v_a_1591_) == 0)
{
lean_object* v_a_1592_; 
lean_dec(v_r_1589_);
lean_dec(v_k_1587_);
lean_dec(v_fst_1576_);
v_a_1592_ = lean_ctor_get(v_a_1591_, 0);
lean_inc(v_a_1592_);
lean_dec_ref_known(v_a_1591_, 1);
v_d_1584_ = v_a_1592_;
goto v___jp_1583_;
}
else
{
lean_object* v___x_1594_; uint8_t v_isShared_1595_; uint8_t v_isSharedCheck_1633_; 
v_isSharedCheck_1633_ = !lean_is_exclusive(v_a_1591_);
if (v_isSharedCheck_1633_ == 0)
{
lean_object* v_unused_1634_; 
v_unused_1634_ = lean_ctor_get(v_a_1591_, 0);
lean_dec(v_unused_1634_);
v___x_1594_ = v_a_1591_;
v_isShared_1595_ = v_isSharedCheck_1633_;
goto v_resetjp_1593_;
}
else
{
lean_dec(v_a_1591_);
v___x_1594_ = lean_box(0);
v_isShared_1595_ = v_isSharedCheck_1633_;
goto v_resetjp_1593_;
}
v_resetjp_1593_:
{
lean_object* v___x_1596_; lean_object* v___f_1597_; lean_object* v___x_1598_; 
v___x_1596_ = lean_box(0);
lean_inc(v_k_1587_);
v___f_1597_ = lean_alloc_closure((void*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1597_, 0, v_k_1587_);
lean_inc(v_fst_1576_);
v___x_1598_ = l_Lean_Syntax_find_x3f(v_fst_1576_, v___f_1597_);
if (lean_obj_tag(v___x_1598_) == 1)
{
lean_object* v_val_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; 
lean_del_object(v___x_1594_);
v_val_1599_ = lean_ctor_get(v___x_1598_, 0);
lean_inc(v_val_1599_);
lean_dec_ref_known(v___x_1598_, 1);
v___x_1600_ = lean_obj_once(&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__1, &lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__1_once, _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__1);
v___x_1601_ = l_Lean_MessageData_ofName(v_k_1587_);
v___x_1602_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1602_, 0, v___x_1600_);
lean_ctor_set(v___x_1602_, 1, v___x_1601_);
v___x_1603_ = lean_obj_once(&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__3, &lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__3_once, _init_lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__3);
v___x_1604_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1604_, 0, v___x_1602_);
lean_ctor_set(v___x_1604_, 1, v___x_1603_);
v___x_1605_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12(v_val_1599_, v___x_1604_, v___y_1580_, v___y_1581_);
lean_dec(v_val_1599_);
if (lean_obj_tag(v___x_1605_) == 0)
{
lean_dec_ref_known(v___x_1605_, 1);
v_init_1578_ = v___x_1596_;
v_x_1579_ = v_r_1589_;
goto _start;
}
else
{
lean_object* v_a_1607_; lean_object* v___x_1609_; uint8_t v_isShared_1610_; uint8_t v_isSharedCheck_1614_; 
lean_dec(v_r_1589_);
lean_dec(v_fst_1576_);
v_a_1607_ = lean_ctor_get(v___x_1605_, 0);
v_isSharedCheck_1614_ = !lean_is_exclusive(v___x_1605_);
if (v_isSharedCheck_1614_ == 0)
{
v___x_1609_ = v___x_1605_;
v_isShared_1610_ = v_isSharedCheck_1614_;
goto v_resetjp_1608_;
}
else
{
lean_inc(v_a_1607_);
lean_dec(v___x_1605_);
v___x_1609_ = lean_box(0);
v_isShared_1610_ = v_isSharedCheck_1614_;
goto v_resetjp_1608_;
}
v_resetjp_1608_:
{
lean_object* v___x_1612_; 
if (v_isShared_1610_ == 0)
{
v___x_1612_ = v___x_1609_;
goto v_reusejp_1611_;
}
else
{
lean_object* v_reuseFailAlloc_1613_; 
v_reuseFailAlloc_1613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1613_, 0, v_a_1607_);
v___x_1612_ = v_reuseFailAlloc_1613_;
goto v_reusejp_1611_;
}
v_reusejp_1611_:
{
return v___x_1612_;
}
}
}
}
else
{
lean_object* v___f_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1619_; 
lean_dec(v___x_1598_);
v___f_1615_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__4));
v___x_1616_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__5));
v___x_1617_ = l_Lean_Name_toString(v_k_1587_, v___x_1577_);
if (v_isShared_1595_ == 0)
{
lean_ctor_set_tag(v___x_1594_, 3);
lean_ctor_set(v___x_1594_, 0, v___x_1617_);
v___x_1619_ = v___x_1594_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1632_; 
v_reuseFailAlloc_1632_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1632_, 0, v___x_1617_);
v___x_1619_ = v_reuseFailAlloc_1632_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_37373__overap_1626_; lean_object* v___x_1627_; 
v___x_1620_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1620_, 0, v___x_1616_);
lean_ctor_set(v___x_1620_, 1, v___x_1619_);
v___x_1621_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___closed__7));
v___x_1622_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1622_, 0, v___x_1620_);
lean_ctor_set(v___x_1622_, 1, v___x_1621_);
v___x_1623_ = l_Std_Format_defWidth;
v___x_1624_ = lean_unsigned_to_nat(0u);
v___x_1625_ = l_Std_Format_pretty(v___x_1622_, v___x_1623_, v___x_1624_, v___x_1624_);
v___x_37373__overap_1626_ = lean_dbg_trace(v___x_1625_, v___f_1615_);
lean_inc(v___y_1581_);
lean_inc_ref(v___y_1580_);
v___x_1627_ = lean_apply_3(v___x_37373__overap_1626_, v___y_1580_, v___y_1581_, lean_box(0));
if (lean_obj_tag(v___x_1627_) == 0)
{
lean_object* v_a_1628_; 
v_a_1628_ = lean_ctor_get(v___x_1627_, 0);
lean_inc(v_a_1628_);
lean_dec_ref_known(v___x_1627_, 1);
if (lean_obj_tag(v_a_1628_) == 0)
{
lean_object* v_a_1629_; 
lean_dec(v_r_1589_);
lean_dec(v_fst_1576_);
v_a_1629_ = lean_ctor_get(v_a_1628_, 0);
lean_inc(v_a_1629_);
lean_dec_ref_known(v_a_1628_, 1);
v_d_1584_ = v_a_1629_;
goto v___jp_1583_;
}
else
{
lean_object* v_a_1630_; 
v_a_1630_ = lean_ctor_get(v_a_1628_, 0);
lean_inc(v_a_1630_);
lean_dec_ref_known(v_a_1628_, 1);
v_init_1578_ = v_a_1630_;
v_x_1579_ = v_r_1589_;
goto _start;
}
}
else
{
lean_dec(v_r_1589_);
lean_dec(v_fst_1576_);
return v___x_1627_;
}
}
}
}
}
}
else
{
lean_dec(v_r_1589_);
lean_dec(v_k_1587_);
lean_dec(v_fst_1576_);
return v___x_1590_;
}
}
else
{
lean_object* v___x_1635_; lean_object* v___x_1636_; 
lean_dec(v_fst_1576_);
v___x_1635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1635_, 0, v_init_1578_);
v___x_1636_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1636_, 0, v___x_1635_);
return v___x_1636_;
}
v___jp_1583_:
{
lean_object* v___x_1585_; lean_object* v___x_1586_; 
v___x_1585_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1585_, 0, v_d_1584_);
v___x_1586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1586_, 0, v___x_1585_);
return v___x_1586_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13___boxed(lean_object* v_fst_1637_, lean_object* v___x_1638_, lean_object* v_init_1639_, lean_object* v_x_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_){
_start:
{
uint8_t v___x_41554__boxed_1644_; lean_object* v_res_1645_; 
v___x_41554__boxed_1644_ = lean_unbox(v___x_1638_);
v_res_1645_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13(v_fst_1637_, v___x_41554__boxed_1644_, v_init_1639_, v_x_1640_, v___y_1641_, v___y_1642_);
lean_dec(v___y_1642_);
lean_dec_ref(v___y_1641_);
return v_res_1645_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10_spec__15(lean_object* v_a_1646_, lean_object* v_as_1647_, size_t v_i_1648_, size_t v_stop_1649_){
_start:
{
uint8_t v___x_1650_; 
v___x_1650_ = lean_usize_dec_eq(v_i_1648_, v_stop_1649_);
if (v___x_1650_ == 0)
{
lean_object* v___x_1651_; uint8_t v___x_1652_; 
v___x_1651_ = lean_array_uget_borrowed(v_as_1647_, v_i_1648_);
v___x_1652_ = lean_name_eq(v_a_1646_, v___x_1651_);
if (v___x_1652_ == 0)
{
size_t v___x_1653_; size_t v___x_1654_; 
v___x_1653_ = ((size_t)1ULL);
v___x_1654_ = lean_usize_add(v_i_1648_, v___x_1653_);
v_i_1648_ = v___x_1654_;
goto _start;
}
else
{
return v___x_1652_;
}
}
else
{
uint8_t v___x_1656_; 
v___x_1656_ = 0;
return v___x_1656_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10_spec__15___boxed(lean_object* v_a_1657_, lean_object* v_as_1658_, lean_object* v_i_1659_, lean_object* v_stop_1660_){
_start:
{
size_t v_i_boxed_1661_; size_t v_stop_boxed_1662_; uint8_t v_res_1663_; lean_object* v_r_1664_; 
v_i_boxed_1661_ = lean_unbox_usize(v_i_1659_);
lean_dec(v_i_1659_);
v_stop_boxed_1662_ = lean_unbox_usize(v_stop_1660_);
lean_dec(v_stop_1660_);
v_res_1663_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10_spec__15(v_a_1657_, v_as_1658_, v_i_boxed_1661_, v_stop_boxed_1662_);
lean_dec_ref(v_as_1658_);
lean_dec(v_a_1657_);
v_r_1664_ = lean_box(v_res_1663_);
return v_r_1664_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10(lean_object* v_as_1665_, lean_object* v_a_1666_){
_start:
{
lean_object* v___x_1667_; lean_object* v___x_1668_; uint8_t v___x_1669_; 
v___x_1667_ = lean_unsigned_to_nat(0u);
v___x_1668_ = lean_array_get_size(v_as_1665_);
v___x_1669_ = lean_nat_dec_lt(v___x_1667_, v___x_1668_);
if (v___x_1669_ == 0)
{
return v___x_1669_;
}
else
{
if (v___x_1669_ == 0)
{
return v___x_1669_;
}
else
{
size_t v___x_1670_; size_t v___x_1671_; uint8_t v___x_1672_; 
v___x_1670_ = ((size_t)0ULL);
v___x_1671_ = lean_usize_of_nat(v___x_1668_);
v___x_1672_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10_spec__15(v_a_1666_, v_as_1665_, v___x_1670_, v___x_1671_);
return v___x_1672_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10___boxed(lean_object* v_as_1673_, lean_object* v_a_1674_){
_start:
{
uint8_t v_res_1675_; lean_object* v_r_1676_; 
v_res_1675_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10(v_as_1673_, v_a_1674_);
lean_dec(v_a_1674_);
lean_dec_ref(v_as_1673_);
v_r_1676_ = lean_box(v_res_1675_);
return v_r_1676_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__1(void){
_start:
{
lean_object* v___x_1678_; lean_object* v___x_1679_; 
v___x_1678_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__0));
v___x_1679_ = l_Lean_stringToMessageData(v___x_1678_);
return v___x_1679_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__3(void){
_start:
{
lean_object* v___x_1681_; lean_object* v___x_1682_; 
v___x_1681_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__2));
v___x_1682_ = l_Lean_stringToMessageData(v___x_1681_);
return v___x_1682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5(lean_object* v_linterOption_1683_, lean_object* v_stx_1684_, lean_object* v_msg_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_){
_start:
{
lean_object* v_name_1689_; lean_object* v___x_1691_; uint8_t v_isShared_1692_; uint8_t v_isSharedCheck_1707_; 
v_name_1689_ = lean_ctor_get(v_linterOption_1683_, 0);
v_isSharedCheck_1707_ = !lean_is_exclusive(v_linterOption_1683_);
if (v_isSharedCheck_1707_ == 0)
{
lean_object* v_unused_1708_; 
v_unused_1708_ = lean_ctor_get(v_linterOption_1683_, 1);
lean_dec(v_unused_1708_);
v___x_1691_ = v_linterOption_1683_;
v_isShared_1692_ = v_isSharedCheck_1707_;
goto v_resetjp_1690_;
}
else
{
lean_inc(v_name_1689_);
lean_dec(v_linterOption_1683_);
v___x_1691_ = lean_box(0);
v_isShared_1692_ = v_isSharedCheck_1707_;
goto v_resetjp_1690_;
}
v_resetjp_1690_:
{
lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1696_; 
v___x_1693_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__1);
lean_inc(v_name_1689_);
v___x_1694_ = l_Lean_MessageData_ofName(v_name_1689_);
if (v_isShared_1692_ == 0)
{
lean_ctor_set_tag(v___x_1691_, 7);
lean_ctor_set(v___x_1691_, 1, v___x_1694_);
lean_ctor_set(v___x_1691_, 0, v___x_1693_);
v___x_1696_ = v___x_1691_;
goto v_reusejp_1695_;
}
else
{
lean_object* v_reuseFailAlloc_1706_; 
v_reuseFailAlloc_1706_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1706_, 0, v___x_1693_);
lean_ctor_set(v_reuseFailAlloc_1706_, 1, v___x_1694_);
v___x_1696_ = v_reuseFailAlloc_1706_;
goto v_reusejp_1695_;
}
v_reusejp_1695_:
{
lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v_disable_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; 
v___x_1697_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___closed__3);
v___x_1698_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1698_, 0, v___x_1696_);
lean_ctor_set(v___x_1698_, 1, v___x_1697_);
v_disable_1699_ = l_Lean_MessageData_note(v___x_1698_);
v___x_1700_ = l_Lean_Linter_linterMessageTag;
v___x_1701_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1701_, 0, v_msg_1685_);
lean_ctor_set(v___x_1701_, 1, v_disable_1699_);
v___x_1702_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1702_, 0, v___x_1700_);
lean_ctor_set(v___x_1702_, 1, v___x_1701_);
v___x_1703_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1703_, 0, v_name_1689_);
lean_ctor_set(v___x_1703_, 1, v___x_1702_);
lean_inc(v_stx_1684_);
v___x_1704_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_1704_, 0, v_stx_1684_);
lean_ctor_set(v___x_1704_, 1, v___x_1703_);
v___x_1705_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12(v_stx_1684_, v___x_1704_, v___y_1686_, v___y_1687_);
lean_dec(v_stx_1684_);
return v___x_1705_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5___boxed(lean_object* v_linterOption_1709_, lean_object* v_stx_1710_, lean_object* v_msg_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_){
_start:
{
lean_object* v_res_1715_; 
v_res_1715_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5(v_linterOption_1709_, v_stx_1710_, v_msg_1711_, v___y_1712_, v___y_1713_);
lean_dec(v___y_1713_);
lean_dec_ref(v___y_1712_);
return v_res_1715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(lean_object* v_init_1716_, lean_object* v_x_1717_){
_start:
{
if (lean_obj_tag(v_x_1717_) == 0)
{
lean_object* v_k_1718_; lean_object* v_l_1719_; lean_object* v_r_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; 
v_k_1718_ = lean_ctor_get(v_x_1717_, 1);
lean_inc(v_k_1718_);
v_l_1719_ = lean_ctor_get(v_x_1717_, 3);
lean_inc(v_l_1719_);
v_r_1720_ = lean_ctor_get(v_x_1717_, 4);
lean_inc(v_r_1720_);
lean_dec_ref_known(v_x_1717_, 5);
v___x_1721_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v_init_1716_, v_l_1719_);
v___x_1722_ = lean_array_push(v___x_1721_, v_k_1718_);
v_init_1716_ = v___x_1722_;
v_x_1717_ = v_r_1720_;
goto _start;
}
else
{
return v_init_1716_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__11(size_t v_sz_1724_, size_t v_i_1725_, lean_object* v_bs_1726_){
_start:
{
uint8_t v___x_1727_; 
v___x_1727_ = lean_usize_dec_lt(v_i_1725_, v_sz_1724_);
if (v___x_1727_ == 0)
{
return v_bs_1726_;
}
else
{
lean_object* v_v_1728_; lean_object* v_module_1729_; lean_object* v___x_1730_; lean_object* v_bs_x27_1731_; size_t v___x_1732_; size_t v___x_1733_; lean_object* v___x_1734_; 
v_v_1728_ = lean_array_uget_borrowed(v_bs_1726_, v_i_1725_);
v_module_1729_ = lean_ctor_get(v_v_1728_, 0);
lean_inc(v_module_1729_);
v___x_1730_ = lean_unsigned_to_nat(0u);
v_bs_x27_1731_ = lean_array_uset(v_bs_1726_, v_i_1725_, v___x_1730_);
v___x_1732_ = ((size_t)1ULL);
v___x_1733_ = lean_usize_add(v_i_1725_, v___x_1732_);
v___x_1734_ = lean_array_uset(v_bs_x27_1731_, v_i_1725_, v_module_1729_);
v_i_1725_ = v___x_1733_;
v_bs_1726_ = v___x_1734_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__11___boxed(lean_object* v_sz_1736_, lean_object* v_i_1737_, lean_object* v_bs_1738_){
_start:
{
size_t v_sz_boxed_1739_; size_t v_i_boxed_1740_; lean_object* v_res_1741_; 
v_sz_boxed_1739_ = lean_unbox_usize(v_sz_1736_);
lean_dec(v_sz_1736_);
v_i_boxed_1740_ = lean_unbox_usize(v_i_1737_);
lean_dec(v_i_1737_);
v_res_1741_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__11(v_sz_boxed_1739_, v_i_boxed_1740_, v_bs_1738_);
return v_res_1741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg(lean_object* v_o_1742_, lean_object* v___y_1743_){
_start:
{
lean_object* v___x_1745_; lean_object* v_env_1746_; lean_object* v___x_1747_; lean_object* v_toEnvExtension_1748_; lean_object* v_asyncMode_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v_merged_1753_; lean_object* v___x_1755_; uint8_t v_isShared_1756_; uint8_t v_isSharedCheck_1761_; 
v___x_1745_ = lean_st_ref_get(v___y_1743_);
v_env_1746_ = lean_ctor_get(v___x_1745_, 0);
lean_inc_ref(v_env_1746_);
lean_dec(v___x_1745_);
v___x_1747_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_1748_ = lean_ctor_get(v___x_1747_, 0);
v_asyncMode_1749_ = lean_ctor_get(v_toEnvExtension_1748_, 2);
v___x_1750_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_1751_ = lean_box(0);
v___x_1752_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_1750_, v___x_1747_, v_env_1746_, v_asyncMode_1749_, v___x_1751_);
v_merged_1753_ = lean_ctor_get(v___x_1752_, 0);
v_isSharedCheck_1761_ = !lean_is_exclusive(v___x_1752_);
if (v_isSharedCheck_1761_ == 0)
{
lean_object* v_unused_1762_; 
v_unused_1762_ = lean_ctor_get(v___x_1752_, 1);
lean_dec(v_unused_1762_);
v___x_1755_ = v___x_1752_;
v_isShared_1756_ = v_isSharedCheck_1761_;
goto v_resetjp_1754_;
}
else
{
lean_inc(v_merged_1753_);
lean_dec(v___x_1752_);
v___x_1755_ = lean_box(0);
v_isShared_1756_ = v_isSharedCheck_1761_;
goto v_resetjp_1754_;
}
v_resetjp_1754_:
{
lean_object* v___x_1758_; 
if (v_isShared_1756_ == 0)
{
lean_ctor_set(v___x_1755_, 1, v_merged_1753_);
lean_ctor_set(v___x_1755_, 0, v_o_1742_);
v___x_1758_ = v___x_1755_;
goto v_reusejp_1757_;
}
else
{
lean_object* v_reuseFailAlloc_1760_; 
v_reuseFailAlloc_1760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1760_, 0, v_o_1742_);
lean_ctor_set(v_reuseFailAlloc_1760_, 1, v_merged_1753_);
v___x_1758_ = v_reuseFailAlloc_1760_;
goto v_reusejp_1757_;
}
v_reusejp_1757_:
{
lean_object* v___x_1759_; 
v___x_1759_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1759_, 0, v___x_1758_);
return v___x_1759_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg___boxed(lean_object* v_o_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_){
_start:
{
lean_object* v_res_1766_; 
v_res_1766_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg(v_o_1763_, v___y_1764_);
lean_dec(v___y_1764_);
return v_res_1766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0(lean_object* v___y_1767_, lean_object* v___y_1768_){
_start:
{
lean_object* v___x_1770_; lean_object* v_scopes_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v_opts_1774_; lean_object* v___x_1775_; 
v___x_1770_ = lean_st_ref_get(v___y_1768_);
v_scopes_1771_ = lean_ctor_get(v___x_1770_, 2);
lean_inc(v_scopes_1771_);
lean_dec(v___x_1770_);
v___x_1772_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1773_ = l_List_head_x21___redArg(v___x_1772_, v_scopes_1771_);
lean_dec(v_scopes_1771_);
v_opts_1774_ = lean_ctor_get(v___x_1773_, 1);
lean_inc_ref(v_opts_1774_);
lean_dec(v___x_1773_);
v___x_1775_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg(v_opts_1774_, v___y_1768_);
return v___x_1775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0___boxed(lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_){
_start:
{
lean_object* v_res_1779_; 
v_res_1779_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0(v___y_1776_, v___y_1777_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
return v_res_1779_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1781_; lean_object* v___x_1782_; 
v___x_1781_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__0));
v___x_1782_ = l_Lean_stringToMessageData(v___x_1781_);
return v___x_1782_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__3(void){
_start:
{
lean_object* v___x_1784_; lean_object* v___x_1785_; 
v___x_1784_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__2));
v___x_1785_ = l_Lean_stringToMessageData(v___x_1784_);
return v___x_1785_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__5(void){
_start:
{
lean_object* v___x_1787_; lean_object* v___x_1788_; 
v___x_1787_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__4));
v___x_1788_ = l_Lean_stringToMessageData(v___x_1787_);
return v___x_1788_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__7(void){
_start:
{
lean_object* v___x_1790_; lean_object* v___x_1791_; 
v___x_1790_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__6));
v___x_1791_ = l_Lean_stringToMessageData(v___x_1790_);
return v___x_1791_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__9(void){
_start:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; 
v___x_1793_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__8));
v___x_1794_ = l_Lean_stringToMessageData(v___x_1793_);
return v___x_1794_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10(void){
_start:
{
lean_object* v___x_1795_; lean_object* v___x_1796_; 
v___x_1795_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18___closed__0));
v___x_1796_ = l_Lean_stringToMessageData(v___x_1795_);
return v___x_1796_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__12(void){
_start:
{
lean_object* v___x_1798_; lean_object* v___x_1799_; 
v___x_1798_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__11));
v___x_1799_ = l_Lean_stringToMessageData(v___x_1798_);
return v___x_1799_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__14(void){
_start:
{
lean_object* v___x_1801_; lean_object* v___x_1802_; 
v___x_1801_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__13));
v___x_1802_ = l_Lean_stringToMessageData(v___x_1801_);
return v___x_1802_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__16(void){
_start:
{
lean_object* v___x_1804_; lean_object* v___x_1805_; 
v___x_1804_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__15));
v___x_1805_ = l_Lean_stringToMessageData(v___x_1804_);
return v___x_1805_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__20(void){
_start:
{
lean_object* v___x_1810_; lean_object* v___x_1811_; 
v___x_1810_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn___closed__2_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_));
v___x_1811_ = l_Lean_mkIdent(v___x_1810_);
return v___x_1811_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__23(void){
_start:
{
lean_object* v___x_1815_; lean_object* v___x_1816_; 
v___x_1815_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__22));
v___x_1816_ = l_Lean_MessageData_ofFormat(v___x_1815_);
return v___x_1816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2(lean_object* v___f_1817_, lean_object* v_stx_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_){
_start:
{
lean_object* v___x_1822_; lean_object* v_a_1823_; lean_object* v___x_1825_; uint8_t v_isShared_1826_; uint8_t v_isSharedCheck_2574_; 
v___x_1822_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0(v___y_1819_, v___y_1820_);
v_a_1823_ = lean_ctor_get(v___x_1822_, 0);
v_isSharedCheck_2574_ = !lean_is_exclusive(v___x_1822_);
if (v_isSharedCheck_2574_ == 0)
{
v___x_1825_ = v___x_1822_;
v_isShared_1826_ = v_isSharedCheck_2574_;
goto v_resetjp_1824_;
}
else
{
lean_inc(v_a_1823_);
lean_dec(v___x_1822_);
v___x_1825_ = lean_box(0);
v_isShared_1826_ = v_isSharedCheck_2574_;
goto v_resetjp_1824_;
}
v_resetjp_1824_:
{
lean_object* v___x_1827_; lean_object* v___y_1829_; lean_object* v___y_1830_; lean_object* v___y_1831_; lean_object* v___y_1832_; lean_object* v___y_1836_; lean_object* v___y_1837_; lean_object* v___y_1838_; lean_object* v___y_1839_; lean_object* v___y_1840_; lean_object* v___y_1841_; lean_object* v___y_1842_; lean_object* v___y_1870_; lean_object* v___y_1871_; lean_object* v___y_1872_; lean_object* v___y_1873_; lean_object* v___y_1874_; lean_object* v___y_1875_; lean_object* v___y_1876_; lean_object* v___y_1877_; lean_object* v___y_1898_; lean_object* v___y_1899_; lean_object* v___y_1900_; lean_object* v___y_1901_; lean_object* v___y_1902_; lean_object* v___y_1903_; lean_object* v___y_1904_; lean_object* v___y_1905_; lean_object* v___y_1906_; lean_object* v___y_1907_; lean_object* v___y_1921_; lean_object* v___y_1922_; lean_object* v___y_1923_; lean_object* v___y_1924_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v___y_1927_; lean_object* v___y_1928_; lean_object* v___y_1929_; lean_object* v___y_1932_; lean_object* v___y_1933_; lean_object* v___y_1934_; lean_object* v___y_1935_; lean_object* v___y_1936_; lean_object* v___y_1937_; lean_object* v___y_1938_; lean_object* v___y_1939_; lean_object* v___y_1940_; lean_object* v___y_1964_; lean_object* v___y_1965_; lean_object* v___y_1966_; lean_object* v___y_1967_; lean_object* v___y_1968_; lean_object* v___y_1969_; lean_object* v___y_1970_; lean_object* v___y_1971_; lean_object* v___y_1972_; lean_object* v___y_1973_; lean_object* v___y_1977_; lean_object* v___y_1978_; lean_object* v___y_1979_; lean_object* v___y_1980_; lean_object* v___y_1981_; lean_object* v___y_1982_; lean_object* v___y_1983_; lean_object* v___y_1984_; lean_object* v___y_1985_; lean_object* v___y_1986_; lean_object* v___y_1987_; lean_object* v___y_1992_; lean_object* v___y_1993_; lean_object* v___y_1994_; lean_object* v___y_1995_; lean_object* v___y_1996_; lean_object* v___y_1997_; lean_object* v___y_1998_; lean_object* v___y_1999_; lean_object* v___y_2000_; lean_object* v___y_2001_; lean_object* v___y_2002_; uint8_t v___y_2003_; lean_object* v___y_2009_; uint8_t v___y_2010_; lean_object* v___y_2011_; lean_object* v___y_2012_; lean_object* v___y_2013_; lean_object* v___y_2014_; lean_object* v___y_2015_; lean_object* v___y_2016_; lean_object* v___y_2017_; lean_object* v___y_2018_; lean_object* v___y_2019_; lean_object* v___y_2020_; lean_object* v___y_2021_; lean_object* v___y_2024_; uint8_t v___y_2025_; lean_object* v___y_2026_; lean_object* v___y_2027_; lean_object* v___y_2028_; lean_object* v___y_2029_; lean_object* v___y_2030_; lean_object* v___y_2031_; lean_object* v___y_2032_; lean_object* v___y_2033_; lean_object* v___y_2034_; lean_object* v___y_2035_; lean_object* v___y_2036_; lean_object* v___y_2037_; lean_object* v___y_2038_; lean_object* v___y_2039_; lean_object* v___y_2042_; uint8_t v___y_2043_; lean_object* v___y_2044_; lean_object* v___y_2045_; lean_object* v___y_2046_; lean_object* v___y_2047_; lean_object* v___y_2048_; lean_object* v___y_2049_; lean_object* v___y_2050_; lean_object* v___y_2051_; lean_object* v___y_2052_; lean_object* v___y_2053_; lean_object* v___y_2054_; lean_object* v___y_2055_; lean_object* v___y_2056_; lean_object* v___y_2057_; lean_object* v___y_2060_; lean_object* v___y_2061_; uint8_t v___y_2062_; lean_object* v___y_2063_; lean_object* v___y_2064_; lean_object* v___y_2065_; lean_object* v___y_2066_; lean_object* v___y_2067_; lean_object* v___y_2068_; lean_object* v___y_2069_; lean_object* v___y_2070_; lean_object* v___y_2071_; lean_object* v___y_2072_; lean_object* v___y_2073_; lean_object* v___y_2081_; lean_object* v___y_2082_; uint8_t v___y_2083_; lean_object* v___y_2084_; lean_object* v___y_2085_; lean_object* v___y_2086_; lean_object* v___y_2087_; lean_object* v___y_2088_; lean_object* v___y_2089_; lean_object* v___y_2090_; lean_object* v___y_2091_; lean_object* v___y_2092_; lean_object* v___y_2093_; uint8_t v___y_2094_; uint8_t v___x_2096_; lean_object* v___y_2098_; lean_object* v___y_2099_; uint8_t v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v___y_2104_; lean_object* v___y_2105_; lean_object* v___y_2106_; lean_object* v___y_2107_; lean_object* v___y_2108_; lean_object* v___y_2109_; lean_object* v___y_2110_; lean_object* v___y_2117_; lean_object* v___y_2118_; lean_object* v___y_2119_; uint8_t v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2122_; lean_object* v___y_2123_; lean_object* v___y_2124_; lean_object* v___y_2125_; lean_object* v___y_2126_; lean_object* v___y_2127_; lean_object* v___y_2128_; lean_object* v___y_2129_; lean_object* v___y_2130_; lean_object* v___y_2131_; lean_object* v___y_2132_; lean_object* v___y_2135_; lean_object* v___y_2136_; lean_object* v___y_2137_; uint8_t v___y_2138_; lean_object* v___y_2139_; lean_object* v___y_2140_; lean_object* v___y_2141_; lean_object* v___y_2142_; lean_object* v___y_2143_; lean_object* v___y_2144_; lean_object* v___y_2145_; lean_object* v___y_2146_; lean_object* v___y_2147_; lean_object* v___y_2148_; lean_object* v___y_2149_; lean_object* v___y_2150_; lean_object* v___y_2153_; lean_object* v___y_2154_; lean_object* v___y_2155_; lean_object* v___y_2156_; lean_object* v___y_2157_; lean_object* v___y_2158_; lean_object* v___y_2159_; lean_object* v___y_2160_; uint8_t v___y_2161_; lean_object* v___y_2162_; lean_object* v___y_2163_; lean_object* v___y_2173_; lean_object* v___y_2174_; lean_object* v___y_2175_; lean_object* v___y_2176_; lean_object* v___y_2177_; lean_object* v___y_2178_; lean_object* v___y_2179_; uint8_t v___y_2180_; lean_object* v___y_2181_; lean_object* v___y_2182_; lean_object* v___y_2190_; lean_object* v___y_2191_; lean_object* v___y_2192_; lean_object* v___y_2193_; lean_object* v___y_2194_; lean_object* v___y_2195_; uint8_t v___y_2196_; lean_object* v___y_2197_; lean_object* v___y_2198_; lean_object* v___y_2225_; lean_object* v___y_2226_; lean_object* v___y_2227_; lean_object* v___y_2228_; lean_object* v___y_2229_; lean_object* v___y_2230_; lean_object* v___y_2231_; lean_object* v___y_2232_; uint8_t v___y_2233_; lean_object* v___y_2234_; lean_object* v___y_2235_; lean_object* v___y_2244_; lean_object* v___y_2245_; lean_object* v___y_2246_; uint8_t v___y_2247_; lean_object* v___y_2248_; uint8_t v___y_2249_; lean_object* v___y_2250_; lean_object* v___y_2251_; lean_object* v___y_2252_; lean_object* v___y_2253_; size_t v___y_2254_; lean_object* v___y_2255_; lean_object* v___y_2256_; lean_object* v___y_2263_; lean_object* v___y_2264_; lean_object* v___y_2265_; uint8_t v___y_2266_; lean_object* v___y_2267_; lean_object* v___y_2268_; uint8_t v___y_2269_; lean_object* v___y_2270_; lean_object* v___y_2271_; lean_object* v___y_2272_; lean_object* v___y_2273_; size_t v___y_2274_; lean_object* v___y_2275_; lean_object* v___y_2276_; lean_object* v___y_2277_; lean_object* v___y_2278_; lean_object* v___y_2281_; lean_object* v___y_2282_; lean_object* v___y_2283_; lean_object* v___y_2284_; uint8_t v___y_2285_; lean_object* v___y_2286_; lean_object* v___y_2287_; uint8_t v___y_2288_; lean_object* v___y_2289_; lean_object* v___y_2290_; lean_object* v___y_2291_; lean_object* v___y_2292_; size_t v___y_2293_; lean_object* v___y_2294_; lean_object* v___y_2295_; lean_object* v___y_2296_; lean_object* v___y_2299_; lean_object* v___y_2300_; lean_object* v___y_2301_; lean_object* v___y_2302_; uint8_t v___y_2303_; lean_object* v___y_2304_; uint8_t v___y_2305_; lean_object* v___y_2306_; lean_object* v___y_2307_; lean_object* v___y_2308_; lean_object* v___y_2309_; size_t v___y_2310_; lean_object* v___y_2311_; lean_object* v___y_2312_; lean_object* v___y_2313_; lean_object* v___y_2322_; lean_object* v___y_2323_; lean_object* v___y_2324_; uint8_t v___y_2325_; lean_object* v___y_2326_; lean_object* v___y_2327_; lean_object* v___y_2328_; lean_object* v___y_2329_; lean_object* v___y_2330_; lean_object* v___y_2331_; size_t v___y_2332_; lean_object* v___y_2333_; lean_object* v___y_2334_; uint8_t v___y_2335_; lean_object* v___y_2338_; lean_object* v___y_2339_; lean_object* v___y_2340_; uint8_t v___y_2341_; lean_object* v___y_2342_; lean_object* v___y_2343_; lean_object* v___y_2344_; lean_object* v___y_2345_; lean_object* v___y_2346_; lean_object* v___y_2347_; size_t v___y_2348_; lean_object* v___y_2349_; uint8_t v___y_2350_; lean_object* v___y_2351_; uint8_t v___y_2352_; lean_object* v___y_2354_; lean_object* v___y_2355_; uint8_t v___y_2356_; lean_object* v___y_2357_; lean_object* v___y_2358_; lean_object* v___y_2359_; lean_object* v___y_2360_; lean_object* v___y_2361_; size_t v___y_2362_; lean_object* v___y_2363_; uint8_t v___y_2364_; lean_object* v___y_2365_; lean_object* v___y_2366_; lean_object* v___y_2414_; lean_object* v___y_2415_; lean_object* v___y_2416_; lean_object* v___y_2417_; lean_object* v___y_2418_; uint8_t v___y_2419_; lean_object* v___y_2420_; lean_object* v___y_2421_; lean_object* v___y_2422_; lean_object* v___y_2451_; lean_object* v___y_2452_; lean_object* v___y_2453_; lean_object* v___y_2454_; lean_object* v___y_2455_; uint8_t v___y_2456_; lean_object* v___y_2457_; uint8_t v___y_2458_; lean_object* v___y_2475_; 
v___x_1827_ = lp_mathlib_Mathlib_Linter_linter_minImports;
v___x_2096_ = l_Lean_Linter_getLinterValue(v___x_1827_, v_a_1823_);
lean_dec(v_a_1823_);
if (v___x_2096_ == 0)
{
lean_object* v___x_2509_; lean_object* v___x_2510_; 
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v___x_2509_ = lean_box(0);
v___x_2510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2510_, 0, v___x_2509_);
return v___x_2510_;
}
else
{
lean_object* v___x_2511_; lean_object* v_messages_2512_; uint8_t v___x_2513_; 
v___x_2511_ = lean_st_ref_get(v___y_1820_);
v_messages_2512_ = lean_ctor_get(v___x_2511_, 1);
lean_inc_ref(v_messages_2512_);
lean_dec(v___x_2511_);
v___x_2513_ = l_Lean_MessageLog_hasErrors(v_messages_2512_);
lean_dec_ref(v_messages_2512_);
if (v___x_2513_ == 0)
{
lean_object* v___x_2514_; 
v___x_2514_ = l_Lean_Elab_Command_getRef___redArg(v___y_1819_);
if (lean_obj_tag(v___x_2514_) == 0)
{
lean_object* v_a_2515_; lean_object* v___x_2516_; 
v_a_2515_ = lean_ctor_get(v___x_2514_, 0);
lean_inc(v_a_2515_);
lean_dec_ref_known(v___x_2514_, 1);
v___x_2516_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_1819_);
if (lean_obj_tag(v___x_2516_) == 0)
{
lean_object* v___x_2518_; uint8_t v_isShared_2519_; uint8_t v_isSharedCheck_2554_; 
v_isSharedCheck_2554_ = !lean_is_exclusive(v___x_2516_);
if (v_isSharedCheck_2554_ == 0)
{
lean_object* v_unused_2555_; 
v_unused_2555_ = lean_ctor_get(v___x_2516_, 0);
lean_dec(v_unused_2555_);
v___x_2518_ = v___x_2516_;
v_isShared_2519_ = v_isSharedCheck_2554_;
goto v_resetjp_2517_;
}
else
{
lean_dec(v___x_2516_);
v___x_2518_ = lean_box(0);
v_isShared_2519_ = v_isSharedCheck_2554_;
goto v_resetjp_2517_;
}
v_resetjp_2517_:
{
lean_object* v_quotContext_x3f_2520_; lean_object* v___x_2521_; 
v_quotContext_x3f_2520_ = lean_ctor_get(v___y_1819_, 5);
v___x_2521_ = l_Lean_SourceInfo_fromRef(v_a_2515_, v___x_2513_);
lean_dec(v_a_2515_);
if (lean_obj_tag(v_quotContext_x3f_2520_) == 0)
{
lean_object* v___x_2553_; 
v___x_2553_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg(v___y_1820_);
lean_dec_ref(v___x_2553_);
goto v___jp_2522_;
}
else
{
goto v___jp_2522_;
}
v___jp_2522_:
{
lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; uint8_t v___x_2527_; 
v___x_2523_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__2));
v___x_2524_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports_command_x23import__bumps___closed__3));
lean_inc(v___x_2521_);
v___x_2525_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2525_, 0, v___x_2521_);
lean_ctor_set(v___x_2525_, 1, v___x_2524_);
v___x_2526_ = l_Lean_Syntax_node1(v___x_2521_, v___x_2523_, v___x_2525_);
v___x_2527_ = l_Lean_Syntax_structEq(v_stx_1818_, v___x_2526_);
lean_dec(v___x_2526_);
if (v___x_2527_ == 0)
{
lean_object* v___x_2528_; 
lean_del_object(v___x_2518_);
v___x_2528_ = l_Lean_Elab_Command_getRef___redArg(v___y_1819_);
if (lean_obj_tag(v___x_2528_) == 0)
{
lean_object* v_a_2529_; lean_object* v___x_2530_; 
v_a_2529_ = lean_ctor_get(v___x_2528_, 0);
lean_inc(v_a_2529_);
lean_dec_ref_known(v___x_2528_, 1);
v___x_2530_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_1819_);
if (lean_obj_tag(v___x_2530_) == 0)
{
lean_object* v___x_2531_; 
lean_dec_ref_known(v___x_2530_, 1);
v___x_2531_ = l_Lean_SourceInfo_fromRef(v_a_2529_, v___x_2527_);
lean_dec(v_a_2529_);
if (lean_obj_tag(v_quotContext_x3f_2520_) == 0)
{
lean_object* v___x_2532_; 
v___x_2532_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__17___redArg(v___y_1820_);
lean_dec_ref(v___x_2532_);
v___y_2475_ = v___x_2531_;
goto v___jp_2474_;
}
else
{
v___y_2475_ = v___x_2531_;
goto v___jp_2474_;
}
}
else
{
lean_object* v_a_2533_; lean_object* v___x_2535_; uint8_t v_isShared_2536_; uint8_t v_isSharedCheck_2540_; 
lean_dec(v_a_2529_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2533_ = lean_ctor_get(v___x_2530_, 0);
v_isSharedCheck_2540_ = !lean_is_exclusive(v___x_2530_);
if (v_isSharedCheck_2540_ == 0)
{
v___x_2535_ = v___x_2530_;
v_isShared_2536_ = v_isSharedCheck_2540_;
goto v_resetjp_2534_;
}
else
{
lean_inc(v_a_2533_);
lean_dec(v___x_2530_);
v___x_2535_ = lean_box(0);
v_isShared_2536_ = v_isSharedCheck_2540_;
goto v_resetjp_2534_;
}
v_resetjp_2534_:
{
lean_object* v___x_2538_; 
if (v_isShared_2536_ == 0)
{
v___x_2538_ = v___x_2535_;
goto v_reusejp_2537_;
}
else
{
lean_object* v_reuseFailAlloc_2539_; 
v_reuseFailAlloc_2539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2539_, 0, v_a_2533_);
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
else
{
lean_object* v_a_2541_; lean_object* v___x_2543_; uint8_t v_isShared_2544_; uint8_t v_isSharedCheck_2548_; 
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2541_ = lean_ctor_get(v___x_2528_, 0);
v_isSharedCheck_2548_ = !lean_is_exclusive(v___x_2528_);
if (v_isSharedCheck_2548_ == 0)
{
v___x_2543_ = v___x_2528_;
v_isShared_2544_ = v_isSharedCheck_2548_;
goto v_resetjp_2542_;
}
else
{
lean_inc(v_a_2541_);
lean_dec(v___x_2528_);
v___x_2543_ = lean_box(0);
v_isShared_2544_ = v_isSharedCheck_2548_;
goto v_resetjp_2542_;
}
v_resetjp_2542_:
{
lean_object* v___x_2546_; 
if (v_isShared_2544_ == 0)
{
v___x_2546_ = v___x_2543_;
goto v_reusejp_2545_;
}
else
{
lean_object* v_reuseFailAlloc_2547_; 
v_reuseFailAlloc_2547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2547_, 0, v_a_2541_);
v___x_2546_ = v_reuseFailAlloc_2547_;
goto v_reusejp_2545_;
}
v_reusejp_2545_:
{
return v___x_2546_;
}
}
}
}
else
{
lean_object* v___x_2549_; lean_object* v___x_2551_; 
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v___x_2549_ = lean_box(0);
if (v_isShared_2519_ == 0)
{
lean_ctor_set(v___x_2518_, 0, v___x_2549_);
v___x_2551_ = v___x_2518_;
goto v_reusejp_2550_;
}
else
{
lean_object* v_reuseFailAlloc_2552_; 
v_reuseFailAlloc_2552_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2552_, 0, v___x_2549_);
v___x_2551_ = v_reuseFailAlloc_2552_;
goto v_reusejp_2550_;
}
v_reusejp_2550_:
{
return v___x_2551_;
}
}
}
}
}
else
{
lean_object* v_a_2556_; lean_object* v___x_2558_; uint8_t v_isShared_2559_; uint8_t v_isSharedCheck_2563_; 
lean_dec(v_a_2515_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2556_ = lean_ctor_get(v___x_2516_, 0);
v_isSharedCheck_2563_ = !lean_is_exclusive(v___x_2516_);
if (v_isSharedCheck_2563_ == 0)
{
v___x_2558_ = v___x_2516_;
v_isShared_2559_ = v_isSharedCheck_2563_;
goto v_resetjp_2557_;
}
else
{
lean_inc(v_a_2556_);
lean_dec(v___x_2516_);
v___x_2558_ = lean_box(0);
v_isShared_2559_ = v_isSharedCheck_2563_;
goto v_resetjp_2557_;
}
v_resetjp_2557_:
{
lean_object* v___x_2561_; 
if (v_isShared_2559_ == 0)
{
v___x_2561_ = v___x_2558_;
goto v_reusejp_2560_;
}
else
{
lean_object* v_reuseFailAlloc_2562_; 
v_reuseFailAlloc_2562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2562_, 0, v_a_2556_);
v___x_2561_ = v_reuseFailAlloc_2562_;
goto v_reusejp_2560_;
}
v_reusejp_2560_:
{
return v___x_2561_;
}
}
}
}
else
{
lean_object* v_a_2564_; lean_object* v___x_2566_; uint8_t v_isShared_2567_; uint8_t v_isSharedCheck_2571_; 
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2564_ = lean_ctor_get(v___x_2514_, 0);
v_isSharedCheck_2571_ = !lean_is_exclusive(v___x_2514_);
if (v_isSharedCheck_2571_ == 0)
{
v___x_2566_ = v___x_2514_;
v_isShared_2567_ = v_isSharedCheck_2571_;
goto v_resetjp_2565_;
}
else
{
lean_inc(v_a_2564_);
lean_dec(v___x_2514_);
v___x_2566_ = lean_box(0);
v_isShared_2567_ = v_isSharedCheck_2571_;
goto v_resetjp_2565_;
}
v_resetjp_2565_:
{
lean_object* v___x_2569_; 
if (v_isShared_2567_ == 0)
{
v___x_2569_ = v___x_2566_;
goto v_reusejp_2568_;
}
else
{
lean_object* v_reuseFailAlloc_2570_; 
v_reuseFailAlloc_2570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2570_, 0, v_a_2564_);
v___x_2569_ = v_reuseFailAlloc_2570_;
goto v_reusejp_2568_;
}
v_reusejp_2568_:
{
return v___x_2569_;
}
}
}
}
else
{
lean_object* v___x_2572_; lean_object* v___x_2573_; 
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v___x_2572_ = lean_box(0);
v___x_2573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2573_, 0, v___x_2572_);
return v___x_2573_;
}
}
v___jp_1828_:
{
lean_object* v___x_1833_; lean_object* v___x_1834_; 
v___x_1833_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1833_, 0, v___y_1829_);
lean_ctor_set(v___x_1833_, 1, v___y_1832_);
v___x_1834_ = lp_mathlib_Lean_Linter_logLint___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__5(v___x_1827_, v_stx_1818_, v___x_1833_, v___y_1831_, v___y_1830_);
return v___x_1834_;
}
v___jp_1835_:
{
lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; uint8_t v___x_1861_; 
v___x_1843_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__1, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__1);
v___x_1844_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1844_, 0, v___x_1843_);
lean_ctor_set(v___x_1844_, 1, v___y_1842_);
v___x_1845_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__3, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__3);
v___x_1846_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1846_, 0, v___x_1844_);
lean_ctor_set(v___x_1846_, 1, v___x_1845_);
v___x_1847_ = lean_array_to_list(v___y_1836_);
v___x_1848_ = lean_box(0);
v___x_1849_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__4(v___x_1847_, v___x_1848_);
v___x_1850_ = l_Lean_MessageData_ofList(v___x_1849_);
v___x_1851_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1851_, 0, v___x_1846_);
lean_ctor_set(v___x_1851_, 1, v___x_1850_);
v___x_1852_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__5, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__5);
v___x_1853_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1853_, 0, v___x_1851_);
lean_ctor_set(v___x_1853_, 1, v___x_1852_);
v___x_1854_ = lean_array_to_list(v___y_1841_);
v___x_1855_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__4(v___x_1854_, v___x_1848_);
v___x_1856_ = l_Lean_MessageData_ofList(v___x_1855_);
v___x_1857_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1857_, 0, v___x_1853_);
lean_ctor_set(v___x_1857_, 1, v___x_1856_);
v___x_1858_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__7, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__7);
v___x_1859_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1859_, 0, v___x_1857_);
lean_ctor_set(v___x_1859_, 1, v___x_1858_);
v___x_1860_ = lean_array_get_size(v___y_1840_);
v___x_1861_ = lean_nat_dec_eq(v___x_1860_, v___y_1837_);
lean_dec(v___y_1837_);
if (v___x_1861_ == 0)
{
lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; 
v___x_1862_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__9, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__9);
v___x_1863_ = lean_array_to_list(v___y_1840_);
v___x_1864_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__4(v___x_1863_, v___x_1848_);
v___x_1865_ = l_Lean_MessageData_ofList(v___x_1864_);
v___x_1866_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1866_, 0, v___x_1862_);
lean_ctor_set(v___x_1866_, 1, v___x_1865_);
v___x_1867_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1867_, 0, v___x_1866_);
lean_ctor_set(v___x_1867_, 1, v___x_1858_);
v___y_1829_ = v___x_1859_;
v___y_1830_ = v___y_1838_;
v___y_1831_ = v___y_1839_;
v___y_1832_ = v___x_1867_;
goto v___jp_1828_;
}
else
{
lean_object* v___x_1868_; 
lean_dec_ref(v___y_1840_);
v___x_1868_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10);
v___y_1829_ = v___x_1859_;
v___y_1830_ = v___y_1838_;
v___y_1831_ = v___y_1839_;
v___y_1832_ = v___x_1868_;
goto v___jp_1828_;
}
}
v___jp_1869_:
{
lean_object* v___x_1878_; lean_object* v_a_1879_; lean_object* v___x_1881_; uint8_t v_isShared_1882_; uint8_t v_isSharedCheck_1896_; 
v___x_1878_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0(v___y_1874_, v___y_1872_);
v_a_1879_ = lean_ctor_get(v___x_1878_, 0);
v_isSharedCheck_1896_ = !lean_is_exclusive(v___x_1878_);
if (v_isSharedCheck_1896_ == 0)
{
v___x_1881_ = v___x_1878_;
v_isShared_1882_ = v_isSharedCheck_1896_;
goto v_resetjp_1880_;
}
else
{
lean_inc(v_a_1879_);
lean_dec(v___x_1878_);
v___x_1881_ = lean_box(0);
v_isShared_1882_ = v_isSharedCheck_1896_;
goto v_resetjp_1880_;
}
v_resetjp_1880_:
{
lean_object* v___x_1883_; uint8_t v___x_1884_; 
v___x_1883_ = lp_mathlib_Mathlib_Linter_linter_minImports_increases;
v___x_1884_ = l_Lean_Linter_getLinterValue(v___x_1883_, v_a_1879_);
lean_dec(v_a_1879_);
if (v___x_1884_ == 0)
{
lean_object* v___x_1885_; 
lean_del_object(v___x_1881_);
lean_dec(v___y_1875_);
lean_dec(v___y_1873_);
v___x_1885_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__10);
v___y_1836_ = v___y_1870_;
v___y_1837_ = v___y_1871_;
v___y_1838_ = v___y_1872_;
v___y_1839_ = v___y_1874_;
v___y_1840_ = v___y_1877_;
v___y_1841_ = v___y_1876_;
v___y_1842_ = v___x_1885_;
goto v___jp_1835_;
}
else
{
lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1890_; 
v___x_1886_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__12, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__12);
v___x_1887_ = lean_nat_sub(v___y_1875_, v___y_1873_);
lean_dec(v___y_1873_);
lean_dec(v___y_1875_);
v___x_1888_ = l_Nat_reprFast(v___x_1887_);
if (v_isShared_1882_ == 0)
{
lean_ctor_set_tag(v___x_1881_, 3);
lean_ctor_set(v___x_1881_, 0, v___x_1888_);
v___x_1890_ = v___x_1881_;
goto v_reusejp_1889_;
}
else
{
lean_object* v_reuseFailAlloc_1895_; 
v_reuseFailAlloc_1895_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1895_, 0, v___x_1888_);
v___x_1890_ = v_reuseFailAlloc_1895_;
goto v_reusejp_1889_;
}
v_reusejp_1889_:
{
lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; 
v___x_1891_ = l_Lean_MessageData_ofFormat(v___x_1890_);
v___x_1892_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1892_, 0, v___x_1886_);
lean_ctor_set(v___x_1892_, 1, v___x_1891_);
v___x_1893_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__14, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__14);
v___x_1894_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1894_, 0, v___x_1892_);
lean_ctor_set(v___x_1894_, 1, v___x_1893_);
v___y_1836_ = v___y_1870_;
v___y_1837_ = v___y_1871_;
v___y_1838_ = v___y_1872_;
v___y_1839_ = v___y_1874_;
v___y_1840_ = v___y_1877_;
v___y_1841_ = v___y_1876_;
v___y_1842_ = v___x_1894_;
goto v___jp_1835_;
}
}
}
}
v___jp_1897_:
{
lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; uint8_t v___x_1912_; 
v___x_1908_ = lean_mk_empty_array_with_capacity(v___y_1907_);
lean_dec(v___y_1907_);
v___x_1909_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v___x_1908_, v___y_1904_);
v___x_1910_ = lean_array_get_size(v___x_1909_);
v___x_1911_ = lean_mk_empty_array_with_capacity(v___y_1899_);
v___x_1912_ = lean_nat_dec_lt(v___y_1899_, v___x_1910_);
if (v___x_1912_ == 0)
{
lean_dec_ref(v___x_1909_);
lean_dec(v___y_1902_);
v___y_1870_ = v___y_1898_;
v___y_1871_ = v___y_1899_;
v___y_1872_ = v___y_1900_;
v___y_1873_ = v___y_1901_;
v___y_1874_ = v___y_1903_;
v___y_1875_ = v___y_1905_;
v___y_1876_ = v___y_1906_;
v___y_1877_ = v___x_1911_;
goto v___jp_1869_;
}
else
{
uint8_t v___x_1913_; 
v___x_1913_ = lean_nat_dec_le(v___x_1910_, v___x_1910_);
if (v___x_1913_ == 0)
{
if (v___x_1912_ == 0)
{
lean_dec_ref(v___x_1909_);
lean_dec(v___y_1902_);
v___y_1870_ = v___y_1898_;
v___y_1871_ = v___y_1899_;
v___y_1872_ = v___y_1900_;
v___y_1873_ = v___y_1901_;
v___y_1874_ = v___y_1903_;
v___y_1875_ = v___y_1905_;
v___y_1876_ = v___y_1906_;
v___y_1877_ = v___x_1911_;
goto v___jp_1869_;
}
else
{
size_t v___x_1914_; size_t v___x_1915_; lean_object* v___x_1916_; 
v___x_1914_ = ((size_t)0ULL);
v___x_1915_ = lean_usize_of_nat(v___x_1910_);
v___x_1916_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(v___y_1902_, v___x_1909_, v___x_1914_, v___x_1915_, v___x_1911_);
lean_dec_ref(v___x_1909_);
lean_dec(v___y_1902_);
v___y_1870_ = v___y_1898_;
v___y_1871_ = v___y_1899_;
v___y_1872_ = v___y_1900_;
v___y_1873_ = v___y_1901_;
v___y_1874_ = v___y_1903_;
v___y_1875_ = v___y_1905_;
v___y_1876_ = v___y_1906_;
v___y_1877_ = v___x_1916_;
goto v___jp_1869_;
}
}
else
{
size_t v___x_1917_; size_t v___x_1918_; lean_object* v___x_1919_; 
v___x_1917_ = ((size_t)0ULL);
v___x_1918_ = lean_usize_of_nat(v___x_1910_);
v___x_1919_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(v___y_1902_, v___x_1909_, v___x_1917_, v___x_1918_, v___x_1911_);
lean_dec_ref(v___x_1909_);
lean_dec(v___y_1902_);
v___y_1870_ = v___y_1898_;
v___y_1871_ = v___y_1899_;
v___y_1872_ = v___y_1900_;
v___y_1873_ = v___y_1901_;
v___y_1874_ = v___y_1903_;
v___y_1875_ = v___y_1905_;
v___y_1876_ = v___y_1906_;
v___y_1877_ = v___x_1919_;
goto v___jp_1869_;
}
}
}
v___jp_1920_:
{
if (lean_obj_tag(v___y_1927_) == 0)
{
lean_object* v_size_1930_; 
v_size_1930_ = lean_ctor_get(v___y_1927_, 0);
lean_inc(v_size_1930_);
v___y_1898_ = v___y_1921_;
v___y_1899_ = v___y_1922_;
v___y_1900_ = v___y_1923_;
v___y_1901_ = v___y_1924_;
v___y_1902_ = v___y_1925_;
v___y_1903_ = v___y_1926_;
v___y_1904_ = v___y_1927_;
v___y_1905_ = v___y_1928_;
v___y_1906_ = v___y_1929_;
v___y_1907_ = v_size_1930_;
goto v___jp_1897_;
}
else
{
lean_inc(v___y_1922_);
v___y_1898_ = v___y_1921_;
v___y_1899_ = v___y_1922_;
v___y_1900_ = v___y_1923_;
v___y_1901_ = v___y_1924_;
v___y_1902_ = v___y_1925_;
v___y_1903_ = v___y_1926_;
v___y_1904_ = v___y_1927_;
v___y_1905_ = v___y_1928_;
v___y_1906_ = v___y_1929_;
v___y_1907_ = v___y_1922_;
goto v___jp_1897_;
}
}
v___jp_1931_:
{
lean_object* v___x_1941_; lean_object* v_transClosure_1942_; lean_object* v___x_1944_; uint8_t v_isShared_1945_; uint8_t v_isSharedCheck_1960_; 
v___x_1941_ = lean_st_ref_take(v___y_1934_);
v_transClosure_1942_ = lean_ctor_get(v___x_1941_, 0);
v_isSharedCheck_1960_ = !lean_is_exclusive(v___x_1941_);
if (v_isSharedCheck_1960_ == 0)
{
lean_object* v_unused_1961_; lean_object* v_unused_1962_; 
v_unused_1961_ = lean_ctor_get(v___x_1941_, 2);
lean_dec(v_unused_1961_);
v_unused_1962_ = lean_ctor_get(v___x_1941_, 1);
lean_dec(v_unused_1962_);
v___x_1944_ = v___x_1941_;
v_isShared_1945_ = v_isSharedCheck_1960_;
goto v_resetjp_1943_;
}
else
{
lean_inc(v_transClosure_1942_);
lean_dec(v___x_1941_);
v___x_1944_ = lean_box(0);
v_isShared_1945_ = v_isSharedCheck_1960_;
goto v_resetjp_1943_;
}
v_resetjp_1943_:
{
lean_object* v___x_1947_; 
lean_inc(v___y_1940_);
lean_inc(v___y_1938_);
if (v_isShared_1945_ == 0)
{
lean_ctor_set(v___x_1944_, 2, v___y_1940_);
lean_ctor_set(v___x_1944_, 1, v___y_1938_);
v___x_1947_ = v___x_1944_;
goto v_reusejp_1946_;
}
else
{
lean_object* v_reuseFailAlloc_1959_; 
v_reuseFailAlloc_1959_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1959_, 0, v_transClosure_1942_);
lean_ctor_set(v_reuseFailAlloc_1959_, 1, v___y_1938_);
lean_ctor_set(v_reuseFailAlloc_1959_, 2, v___y_1940_);
v___x_1947_ = v_reuseFailAlloc_1959_;
goto v_reusejp_1946_;
}
v_reusejp_1946_:
{
lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; uint8_t v___x_1951_; 
v___x_1948_ = lean_st_ref_set(v___y_1934_, v___x_1947_);
v___x_1949_ = lean_array_get_size(v___y_1932_);
v___x_1950_ = lean_mk_empty_array_with_capacity(v___y_1933_);
v___x_1951_ = lean_nat_dec_lt(v___y_1933_, v___x_1949_);
if (v___x_1951_ == 0)
{
v___y_1921_ = v___y_1932_;
v___y_1922_ = v___y_1933_;
v___y_1923_ = v___y_1935_;
v___y_1924_ = v___y_1936_;
v___y_1925_ = v___y_1938_;
v___y_1926_ = v___y_1937_;
v___y_1927_ = v___y_1939_;
v___y_1928_ = v___y_1940_;
v___y_1929_ = v___x_1950_;
goto v___jp_1920_;
}
else
{
uint8_t v___x_1952_; 
v___x_1952_ = lean_nat_dec_le(v___x_1949_, v___x_1949_);
if (v___x_1952_ == 0)
{
if (v___x_1951_ == 0)
{
v___y_1921_ = v___y_1932_;
v___y_1922_ = v___y_1933_;
v___y_1923_ = v___y_1935_;
v___y_1924_ = v___y_1936_;
v___y_1925_ = v___y_1938_;
v___y_1926_ = v___y_1937_;
v___y_1927_ = v___y_1939_;
v___y_1928_ = v___y_1940_;
v___y_1929_ = v___x_1950_;
goto v___jp_1920_;
}
else
{
size_t v___x_1953_; size_t v___x_1954_; lean_object* v___x_1955_; 
v___x_1953_ = ((size_t)0ULL);
v___x_1954_ = lean_usize_of_nat(v___x_1949_);
v___x_1955_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(v___y_1939_, v___y_1932_, v___x_1953_, v___x_1954_, v___x_1950_);
v___y_1921_ = v___y_1932_;
v___y_1922_ = v___y_1933_;
v___y_1923_ = v___y_1935_;
v___y_1924_ = v___y_1936_;
v___y_1925_ = v___y_1938_;
v___y_1926_ = v___y_1937_;
v___y_1927_ = v___y_1939_;
v___y_1928_ = v___y_1940_;
v___y_1929_ = v___x_1955_;
goto v___jp_1920_;
}
}
else
{
size_t v___x_1956_; size_t v___x_1957_; lean_object* v___x_1958_; 
v___x_1956_ = ((size_t)0ULL);
v___x_1957_ = lean_usize_of_nat(v___x_1949_);
v___x_1958_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__6(v___y_1939_, v___y_1932_, v___x_1956_, v___x_1957_, v___x_1950_);
v___y_1921_ = v___y_1932_;
v___y_1922_ = v___y_1933_;
v___y_1923_ = v___y_1935_;
v___y_1924_ = v___y_1936_;
v___y_1925_ = v___y_1938_;
v___y_1926_ = v___y_1937_;
v___y_1927_ = v___y_1939_;
v___y_1928_ = v___y_1940_;
v___y_1929_ = v___x_1958_;
goto v___jp_1920_;
}
}
}
}
}
v___jp_1963_:
{
lean_object* v___x_1974_; 
lean_inc(v___y_1972_);
v___x_1974_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_importsBelow_spec__1_spec__1(v___y_1973_, v___y_1972_, v___y_1972_);
lean_dec(v___y_1972_);
lean_dec(v___y_1973_);
if (lean_obj_tag(v___x_1974_) == 0)
{
lean_object* v_size_1975_; 
v_size_1975_ = lean_ctor_get(v___x_1974_, 0);
lean_inc(v_size_1975_);
lean_dec_ref_known(v___x_1974_, 5);
v___y_1932_ = v___y_1964_;
v___y_1933_ = v___y_1965_;
v___y_1934_ = v___y_1966_;
v___y_1935_ = v___y_1967_;
v___y_1936_ = v___y_1968_;
v___y_1937_ = v___y_1970_;
v___y_1938_ = v___y_1969_;
v___y_1939_ = v___y_1971_;
v___y_1940_ = v_size_1975_;
goto v___jp_1931_;
}
else
{
lean_inc(v___y_1965_);
v___y_1932_ = v___y_1964_;
v___y_1933_ = v___y_1965_;
v___y_1934_ = v___y_1966_;
v___y_1935_ = v___y_1967_;
v___y_1936_ = v___y_1968_;
v___y_1937_ = v___y_1970_;
v___y_1938_ = v___y_1969_;
v___y_1939_ = v___y_1971_;
v___y_1940_ = v___y_1965_;
goto v___jp_1931_;
}
}
v___jp_1976_:
{
if (lean_obj_tag(v___y_1979_) == 0)
{
lean_object* v___x_1988_; lean_object* v___x_1989_; 
v___x_1988_ = lp_importGraph_Lean_Environment_importGraph(v___y_1982_);
lean_dec_ref(v___y_1982_);
v___x_1989_ = lp_importGraph_Lean_NameMap_transitiveClosure(v___x_1988_);
v___y_1964_ = v___y_1977_;
v___y_1965_ = v___y_1978_;
v___y_1966_ = v___y_1980_;
v___y_1967_ = v___y_1981_;
v___y_1968_ = v___y_1983_;
v___y_1969_ = v___y_1985_;
v___y_1970_ = v___y_1984_;
v___y_1971_ = v___y_1986_;
v___y_1972_ = v___y_1987_;
v___y_1973_ = v___x_1989_;
goto v___jp_1963_;
}
else
{
lean_object* v_val_1990_; 
lean_dec_ref(v___y_1982_);
v_val_1990_ = lean_ctor_get(v___y_1979_, 0);
lean_inc(v_val_1990_);
lean_dec_ref_known(v___y_1979_, 1);
v___y_1964_ = v___y_1977_;
v___y_1965_ = v___y_1978_;
v___y_1966_ = v___y_1980_;
v___y_1967_ = v___y_1981_;
v___y_1968_ = v___y_1983_;
v___y_1969_ = v___y_1985_;
v___y_1970_ = v___y_1984_;
v___y_1971_ = v___y_1986_;
v___y_1972_ = v___y_1987_;
v___y_1973_ = v_val_1990_;
goto v___jp_1963_;
}
}
v___jp_1991_:
{
if (v___y_2003_ == 0)
{
lean_object* v___x_2004_; lean_object* v___x_2006_; 
lean_dec(v___y_2002_);
lean_dec(v___y_2001_);
lean_dec(v___y_1999_);
lean_dec(v___y_1998_);
lean_dec_ref(v___y_1997_);
lean_dec(v___y_1994_);
lean_dec(v___y_1993_);
lean_dec_ref(v___y_1992_);
lean_dec(v_stx_1818_);
v___x_2004_ = lean_box(0);
if (v_isShared_1826_ == 0)
{
lean_ctor_set(v___x_1825_, 0, v___x_2004_);
v___x_2006_ = v___x_1825_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v___x_2004_);
v___x_2006_ = v_reuseFailAlloc_2007_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
return v___x_2006_;
}
}
else
{
lean_del_object(v___x_1825_);
v___y_1977_ = v___y_1992_;
v___y_1978_ = v___y_1994_;
v___y_1979_ = v___y_1993_;
v___y_1980_ = v___y_1995_;
v___y_1981_ = v___y_1996_;
v___y_1982_ = v___y_1997_;
v___y_1983_ = v___y_1998_;
v___y_1984_ = v___y_2000_;
v___y_1985_ = v___y_1999_;
v___y_1986_ = v___y_2001_;
v___y_1987_ = v___y_2002_;
goto v___jp_1976_;
}
}
v___jp_2008_:
{
uint8_t v___x_2022_; 
v___x_2022_ = lp_mathlib_Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7(v___y_2012_, v___y_2021_);
lean_dec_ref(v___y_2021_);
if (v___x_2022_ == 0)
{
lean_del_object(v___x_1825_);
v___y_1977_ = v___y_2012_;
v___y_1978_ = v___y_2013_;
v___y_1979_ = v___y_2014_;
v___y_1980_ = v___y_2015_;
v___y_1981_ = v___y_2016_;
v___y_1982_ = v___y_2017_;
v___y_1983_ = v___y_2018_;
v___y_1984_ = v___y_2019_;
v___y_1985_ = v___y_2020_;
v___y_1986_ = v___y_2009_;
v___y_1987_ = v___y_2011_;
goto v___jp_1976_;
}
else
{
v___y_1992_ = v___y_2012_;
v___y_1993_ = v___y_2014_;
v___y_1994_ = v___y_2013_;
v___y_1995_ = v___y_2015_;
v___y_1996_ = v___y_2016_;
v___y_1997_ = v___y_2017_;
v___y_1998_ = v___y_2018_;
v___y_1999_ = v___y_2020_;
v___y_2000_ = v___y_2019_;
v___y_2001_ = v___y_2009_;
v___y_2002_ = v___y_2011_;
v___y_2003_ = v___y_2010_;
goto v___jp_1991_;
}
}
v___jp_2023_:
{
lean_object* v___x_2040_; 
v___x_2040_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(v___y_2038_, v___y_2032_, v___y_2026_, v___y_2039_);
lean_dec(v___y_2039_);
lean_dec(v___y_2038_);
v___y_2009_ = v___y_2024_;
v___y_2010_ = v___y_2025_;
v___y_2011_ = v___y_2027_;
v___y_2012_ = v___y_2028_;
v___y_2013_ = v___y_2029_;
v___y_2014_ = v___y_2030_;
v___y_2015_ = v___y_2031_;
v___y_2016_ = v___y_2033_;
v___y_2017_ = v___y_2034_;
v___y_2018_ = v___y_2035_;
v___y_2019_ = v___y_2037_;
v___y_2020_ = v___y_2036_;
v___y_2021_ = v___x_2040_;
goto v___jp_2008_;
}
v___jp_2041_:
{
uint8_t v___x_2058_; 
v___x_2058_ = lean_nat_dec_le(v___y_2057_, v___y_2056_);
if (v___x_2058_ == 0)
{
lean_dec(v___y_2056_);
lean_inc(v___y_2057_);
v___y_2024_ = v___y_2042_;
v___y_2025_ = v___y_2043_;
v___y_2026_ = v___y_2057_;
v___y_2027_ = v___y_2044_;
v___y_2028_ = v___y_2045_;
v___y_2029_ = v___y_2046_;
v___y_2030_ = v___y_2047_;
v___y_2031_ = v___y_2048_;
v___y_2032_ = v___y_2049_;
v___y_2033_ = v___y_2050_;
v___y_2034_ = v___y_2051_;
v___y_2035_ = v___y_2052_;
v___y_2036_ = v___y_2054_;
v___y_2037_ = v___y_2053_;
v___y_2038_ = v___y_2055_;
v___y_2039_ = v___y_2057_;
goto v___jp_2023_;
}
else
{
v___y_2024_ = v___y_2042_;
v___y_2025_ = v___y_2043_;
v___y_2026_ = v___y_2057_;
v___y_2027_ = v___y_2044_;
v___y_2028_ = v___y_2045_;
v___y_2029_ = v___y_2046_;
v___y_2030_ = v___y_2047_;
v___y_2031_ = v___y_2048_;
v___y_2032_ = v___y_2049_;
v___y_2033_ = v___y_2050_;
v___y_2034_ = v___y_2051_;
v___y_2035_ = v___y_2052_;
v___y_2036_ = v___y_2054_;
v___y_2037_ = v___y_2053_;
v___y_2038_ = v___y_2055_;
v___y_2039_ = v___y_2056_;
goto v___jp_2023_;
}
}
v___jp_2059_:
{
lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; uint8_t v___x_2077_; 
v___x_2074_ = lean_mk_empty_array_with_capacity(v___y_2073_);
lean_dec(v___y_2073_);
lean_inc(v___y_2060_);
v___x_2075_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v___x_2074_, v___y_2060_);
v___x_2076_ = lean_array_get_size(v___x_2075_);
v___x_2077_ = lean_nat_dec_eq(v___x_2076_, v___y_2065_);
if (v___x_2077_ == 0)
{
lean_object* v___x_2078_; uint8_t v___x_2079_; 
v___x_2078_ = lean_nat_sub(v___x_2076_, v___y_2061_);
v___x_2079_ = lean_nat_dec_le(v___y_2065_, v___x_2078_);
if (v___x_2079_ == 0)
{
lean_inc(v___x_2078_);
v___y_2042_ = v___y_2060_;
v___y_2043_ = v___y_2062_;
v___y_2044_ = v___y_2063_;
v___y_2045_ = v___y_2064_;
v___y_2046_ = v___y_2065_;
v___y_2047_ = v___y_2066_;
v___y_2048_ = v___y_2067_;
v___y_2049_ = v___x_2075_;
v___y_2050_ = v___y_2068_;
v___y_2051_ = v___y_2069_;
v___y_2052_ = v___y_2070_;
v___y_2053_ = v___y_2072_;
v___y_2054_ = v___y_2071_;
v___y_2055_ = v___x_2076_;
v___y_2056_ = v___x_2078_;
v___y_2057_ = v___x_2078_;
goto v___jp_2041_;
}
else
{
lean_inc(v___y_2065_);
v___y_2042_ = v___y_2060_;
v___y_2043_ = v___y_2062_;
v___y_2044_ = v___y_2063_;
v___y_2045_ = v___y_2064_;
v___y_2046_ = v___y_2065_;
v___y_2047_ = v___y_2066_;
v___y_2048_ = v___y_2067_;
v___y_2049_ = v___x_2075_;
v___y_2050_ = v___y_2068_;
v___y_2051_ = v___y_2069_;
v___y_2052_ = v___y_2070_;
v___y_2053_ = v___y_2072_;
v___y_2054_ = v___y_2071_;
v___y_2055_ = v___x_2076_;
v___y_2056_ = v___x_2078_;
v___y_2057_ = v___y_2065_;
goto v___jp_2041_;
}
}
else
{
v___y_2009_ = v___y_2060_;
v___y_2010_ = v___y_2062_;
v___y_2011_ = v___y_2063_;
v___y_2012_ = v___y_2064_;
v___y_2013_ = v___y_2065_;
v___y_2014_ = v___y_2066_;
v___y_2015_ = v___y_2067_;
v___y_2016_ = v___y_2068_;
v___y_2017_ = v___y_2069_;
v___y_2018_ = v___y_2070_;
v___y_2019_ = v___y_2072_;
v___y_2020_ = v___y_2071_;
v___y_2021_ = v___x_2075_;
goto v___jp_2008_;
}
}
v___jp_2080_:
{
if (v___y_2094_ == 0)
{
v___y_1992_ = v___y_2085_;
v___y_1993_ = v___y_2086_;
v___y_1994_ = v___y_2087_;
v___y_1995_ = v___y_2088_;
v___y_1996_ = v___y_2089_;
v___y_1997_ = v___y_2090_;
v___y_1998_ = v___y_2091_;
v___y_1999_ = v___y_2092_;
v___y_2000_ = v___y_2093_;
v___y_2001_ = v___y_2082_;
v___y_2002_ = v___y_2084_;
v___y_2003_ = v___y_2083_;
goto v___jp_1991_;
}
else
{
if (lean_obj_tag(v___y_2082_) == 0)
{
lean_object* v_size_2095_; 
v_size_2095_ = lean_ctor_get(v___y_2082_, 0);
lean_inc(v_size_2095_);
v___y_2060_ = v___y_2082_;
v___y_2061_ = v___y_2081_;
v___y_2062_ = v___y_2083_;
v___y_2063_ = v___y_2084_;
v___y_2064_ = v___y_2085_;
v___y_2065_ = v___y_2087_;
v___y_2066_ = v___y_2086_;
v___y_2067_ = v___y_2088_;
v___y_2068_ = v___y_2089_;
v___y_2069_ = v___y_2090_;
v___y_2070_ = v___y_2091_;
v___y_2071_ = v___y_2092_;
v___y_2072_ = v___y_2093_;
v___y_2073_ = v_size_2095_;
goto v___jp_2059_;
}
else
{
lean_inc(v___y_2087_);
v___y_2060_ = v___y_2082_;
v___y_2061_ = v___y_2081_;
v___y_2062_ = v___y_2083_;
v___y_2063_ = v___y_2084_;
v___y_2064_ = v___y_2085_;
v___y_2065_ = v___y_2087_;
v___y_2066_ = v___y_2086_;
v___y_2067_ = v___y_2088_;
v___y_2068_ = v___y_2089_;
v___y_2069_ = v___y_2090_;
v___y_2070_ = v___y_2091_;
v___y_2071_ = v___y_2092_;
v___y_2072_ = v___y_2093_;
v___y_2073_ = v___y_2087_;
goto v___jp_2059_;
}
}
}
v___jp_2097_:
{
lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; uint8_t v___x_2114_; 
v___x_2111_ = lean_mk_empty_array_with_capacity(v___y_2102_);
v___x_2112_ = lean_array_get_size(v___y_2110_);
v___x_2113_ = lean_array_get_size(v___x_2111_);
v___x_2114_ = lean_nat_dec_eq(v___x_2112_, v___x_2113_);
if (v___x_2114_ == 0)
{
lean_dec_ref(v___x_2111_);
v___y_2081_ = v___y_2099_;
v___y_2082_ = v___y_2098_;
v___y_2083_ = v___y_2100_;
v___y_2084_ = v___y_2101_;
v___y_2085_ = v___y_2110_;
v___y_2086_ = v___y_2103_;
v___y_2087_ = v___y_2102_;
v___y_2088_ = v___y_2104_;
v___y_2089_ = v___y_2105_;
v___y_2090_ = v___y_2106_;
v___y_2091_ = v___y_2107_;
v___y_2092_ = v___y_2109_;
v___y_2093_ = v___y_2108_;
v___y_2094_ = v___x_2096_;
goto v___jp_2080_;
}
else
{
uint8_t v___x_2115_; 
v___x_2115_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg(v___y_2110_, v___x_2111_, v___x_2112_);
lean_dec_ref(v___x_2111_);
if (v___x_2115_ == 0)
{
v___y_2081_ = v___y_2099_;
v___y_2082_ = v___y_2098_;
v___y_2083_ = v___y_2100_;
v___y_2084_ = v___y_2101_;
v___y_2085_ = v___y_2110_;
v___y_2086_ = v___y_2103_;
v___y_2087_ = v___y_2102_;
v___y_2088_ = v___y_2104_;
v___y_2089_ = v___y_2105_;
v___y_2090_ = v___y_2106_;
v___y_2091_ = v___y_2107_;
v___y_2092_ = v___y_2109_;
v___y_2093_ = v___y_2108_;
v___y_2094_ = v___x_2096_;
goto v___jp_2080_;
}
else
{
v___y_2081_ = v___y_2099_;
v___y_2082_ = v___y_2098_;
v___y_2083_ = v___y_2100_;
v___y_2084_ = v___y_2101_;
v___y_2085_ = v___y_2110_;
v___y_2086_ = v___y_2103_;
v___y_2087_ = v___y_2102_;
v___y_2088_ = v___y_2104_;
v___y_2089_ = v___y_2105_;
v___y_2090_ = v___y_2106_;
v___y_2091_ = v___y_2107_;
v___y_2092_ = v___y_2109_;
v___y_2093_ = v___y_2108_;
v___y_2094_ = v___y_2100_;
goto v___jp_2080_;
}
}
}
v___jp_2116_:
{
lean_object* v___x_2133_; 
v___x_2133_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(v___y_2117_, v___y_2128_, v___y_2121_, v___y_2132_);
lean_dec(v___y_2132_);
lean_dec(v___y_2117_);
v___y_2098_ = v___y_2118_;
v___y_2099_ = v___y_2119_;
v___y_2100_ = v___y_2120_;
v___y_2101_ = v___y_2122_;
v___y_2102_ = v___y_2123_;
v___y_2103_ = v___y_2124_;
v___y_2104_ = v___y_2125_;
v___y_2105_ = v___y_2126_;
v___y_2106_ = v___y_2127_;
v___y_2107_ = v___y_2129_;
v___y_2108_ = v___y_2131_;
v___y_2109_ = v___y_2130_;
v___y_2110_ = v___x_2133_;
goto v___jp_2097_;
}
v___jp_2134_:
{
uint8_t v___x_2151_; 
v___x_2151_ = lean_nat_dec_le(v___y_2150_, v___y_2139_);
if (v___x_2151_ == 0)
{
lean_dec(v___y_2139_);
lean_inc(v___y_2150_);
v___y_2117_ = v___y_2135_;
v___y_2118_ = v___y_2136_;
v___y_2119_ = v___y_2137_;
v___y_2120_ = v___y_2138_;
v___y_2121_ = v___y_2150_;
v___y_2122_ = v___y_2140_;
v___y_2123_ = v___y_2141_;
v___y_2124_ = v___y_2142_;
v___y_2125_ = v___y_2143_;
v___y_2126_ = v___y_2144_;
v___y_2127_ = v___y_2145_;
v___y_2128_ = v___y_2146_;
v___y_2129_ = v___y_2147_;
v___y_2130_ = v___y_2149_;
v___y_2131_ = v___y_2148_;
v___y_2132_ = v___y_2150_;
goto v___jp_2116_;
}
else
{
v___y_2117_ = v___y_2135_;
v___y_2118_ = v___y_2136_;
v___y_2119_ = v___y_2137_;
v___y_2120_ = v___y_2138_;
v___y_2121_ = v___y_2150_;
v___y_2122_ = v___y_2140_;
v___y_2123_ = v___y_2141_;
v___y_2124_ = v___y_2142_;
v___y_2125_ = v___y_2143_;
v___y_2126_ = v___y_2144_;
v___y_2127_ = v___y_2145_;
v___y_2128_ = v___y_2146_;
v___y_2129_ = v___y_2147_;
v___y_2130_ = v___y_2149_;
v___y_2131_ = v___y_2148_;
v___y_2132_ = v___y_2139_;
goto v___jp_2116_;
}
}
v___jp_2152_:
{
lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; uint8_t v___x_2169_; 
v___x_2164_ = lean_mk_empty_array_with_capacity(v___y_2163_);
lean_dec(v___y_2163_);
lean_inc(v___y_2159_);
v___x_2165_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v___x_2164_, v___y_2159_);
v___x_2166_ = lean_unsigned_to_nat(0u);
v___x_2167_ = lean_unsigned_to_nat(1u);
v___x_2168_ = lean_array_get_size(v___x_2165_);
v___x_2169_ = lean_nat_dec_eq(v___x_2168_, v___x_2166_);
if (v___x_2169_ == 0)
{
lean_object* v___x_2170_; uint8_t v___x_2171_; 
v___x_2170_ = lean_nat_sub(v___x_2168_, v___x_2167_);
v___x_2171_ = lean_nat_dec_le(v___x_2166_, v___x_2170_);
if (v___x_2171_ == 0)
{
lean_inc(v___x_2170_);
v___y_2135_ = v___x_2168_;
v___y_2136_ = v___y_2160_;
v___y_2137_ = v___x_2167_;
v___y_2138_ = v___y_2161_;
v___y_2139_ = v___x_2170_;
v___y_2140_ = v___y_2162_;
v___y_2141_ = v___x_2166_;
v___y_2142_ = v___y_2153_;
v___y_2143_ = v___y_2154_;
v___y_2144_ = v___y_2155_;
v___y_2145_ = v___y_2156_;
v___y_2146_ = v___x_2165_;
v___y_2147_ = v___y_2157_;
v___y_2148_ = v___y_2158_;
v___y_2149_ = v___y_2159_;
v___y_2150_ = v___x_2170_;
goto v___jp_2134_;
}
else
{
v___y_2135_ = v___x_2168_;
v___y_2136_ = v___y_2160_;
v___y_2137_ = v___x_2167_;
v___y_2138_ = v___y_2161_;
v___y_2139_ = v___x_2170_;
v___y_2140_ = v___y_2162_;
v___y_2141_ = v___x_2166_;
v___y_2142_ = v___y_2153_;
v___y_2143_ = v___y_2154_;
v___y_2144_ = v___y_2155_;
v___y_2145_ = v___y_2156_;
v___y_2146_ = v___x_2165_;
v___y_2147_ = v___y_2157_;
v___y_2148_ = v___y_2158_;
v___y_2149_ = v___y_2159_;
v___y_2150_ = v___x_2166_;
goto v___jp_2134_;
}
}
else
{
v___y_2098_ = v___y_2160_;
v___y_2099_ = v___x_2167_;
v___y_2100_ = v___y_2161_;
v___y_2101_ = v___y_2162_;
v___y_2102_ = v___x_2166_;
v___y_2103_ = v___y_2153_;
v___y_2104_ = v___y_2154_;
v___y_2105_ = v___y_2155_;
v___y_2106_ = v___y_2156_;
v___y_2107_ = v___y_2157_;
v___y_2108_ = v___y_2158_;
v___y_2109_ = v___y_2159_;
v___y_2110_ = v___x_2165_;
goto v___jp_2097_;
}
}
v___jp_2172_:
{
lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; 
v___x_2183_ = lean_mk_empty_array_with_capacity(v___y_2182_);
lean_dec(v___y_2182_);
lean_inc_n(v___y_2181_, 2);
v___x_2184_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v___x_2183_, v___y_2181_);
v___x_2185_ = lp_importGraph_Lean_Environment_findRedundantImports(v___y_2176_, v___x_2184_);
lean_dec_ref(v___x_2184_);
v___x_2186_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(v___y_2181_, v___x_2185_);
lean_dec(v___x_2185_);
if (lean_obj_tag(v___x_2186_) == 0)
{
lean_object* v_size_2187_; 
v_size_2187_ = lean_ctor_get(v___x_2186_, 0);
lean_inc(v_size_2187_);
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2175_;
v___y_2156_ = v___y_2176_;
v___y_2157_ = v___y_2177_;
v___y_2158_ = v___y_2178_;
v___y_2159_ = v___x_2186_;
v___y_2160_ = v___y_2179_;
v___y_2161_ = v___y_2180_;
v___y_2162_ = v___y_2181_;
v___y_2163_ = v_size_2187_;
goto v___jp_2152_;
}
else
{
lean_object* v___x_2188_; 
v___x_2188_ = lean_unsigned_to_nat(0u);
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2175_;
v___y_2156_ = v___y_2176_;
v___y_2157_ = v___y_2177_;
v___y_2158_ = v___y_2178_;
v___y_2159_ = v___x_2186_;
v___y_2160_ = v___y_2179_;
v___y_2161_ = v___y_2180_;
v___y_2162_ = v___y_2181_;
v___y_2163_ = v___x_2188_;
goto v___jp_2152_;
}
}
v___jp_2189_:
{
lean_object* v___x_2199_; 
lean_inc(v_stx_1818_);
v___x_2199_ = lp_mathlib_Mathlib_Command_MinImports_getId(v_stx_1818_, v___y_2197_, v___y_2198_);
if (lean_obj_tag(v___x_2199_) == 0)
{
lean_object* v_a_2200_; lean_object* v___x_2201_; 
v_a_2200_ = lean_ctor_get(v___x_2199_, 0);
lean_inc(v_a_2200_);
lean_dec_ref_known(v___x_2199_, 1);
lean_inc(v_stx_1818_);
v___x_2201_ = lp_mathlib_Mathlib_Command_MinImports_getAllImports(v_stx_1818_, v_a_2200_, v___y_2196_, v___y_2197_, v___y_2198_);
lean_dec(v_a_2200_);
if (lean_obj_tag(v___x_2201_) == 0)
{
lean_object* v_a_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; 
v_a_2202_ = lean_ctor_get(v___x_2201_, 0);
lean_inc(v_a_2202_);
lean_dec_ref_known(v___x_2201_, 1);
v___x_2203_ = lp_mathlib_Mathlib_Command_MinImports_getIrredundantImports(v___y_2193_, v_a_2202_);
v___x_2204_ = l_Lean_NameSet_filter(v___y_2190_, v___x_2203_);
lean_inc(v___y_2195_);
v___x_2205_ = l_Lean_NameSet_append(v___x_2204_, v___y_2195_);
if (lean_obj_tag(v___x_2205_) == 0)
{
lean_object* v_size_2206_; 
v_size_2206_ = lean_ctor_get(v___x_2205_, 0);
lean_inc(v_size_2206_);
v___y_2173_ = v___y_2191_;
v___y_2174_ = v___y_2192_;
v___y_2175_ = v___y_2198_;
v___y_2176_ = v___y_2193_;
v___y_2177_ = v___y_2194_;
v___y_2178_ = v___y_2197_;
v___y_2179_ = v___y_2195_;
v___y_2180_ = v___y_2196_;
v___y_2181_ = v___x_2205_;
v___y_2182_ = v_size_2206_;
goto v___jp_2172_;
}
else
{
lean_object* v___x_2207_; 
v___x_2207_ = lean_unsigned_to_nat(0u);
v___y_2173_ = v___y_2191_;
v___y_2174_ = v___y_2192_;
v___y_2175_ = v___y_2198_;
v___y_2176_ = v___y_2193_;
v___y_2177_ = v___y_2194_;
v___y_2178_ = v___y_2197_;
v___y_2179_ = v___y_2195_;
v___y_2180_ = v___y_2196_;
v___y_2181_ = v___x_2205_;
v___y_2182_ = v___x_2207_;
goto v___jp_2172_;
}
}
else
{
lean_object* v_a_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2215_; 
lean_dec(v___y_2195_);
lean_dec(v___y_2194_);
lean_dec_ref(v___y_2193_);
lean_dec(v___y_2191_);
lean_dec_ref(v___y_2190_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
v_a_2208_ = lean_ctor_get(v___x_2201_, 0);
v_isSharedCheck_2215_ = !lean_is_exclusive(v___x_2201_);
if (v_isSharedCheck_2215_ == 0)
{
v___x_2210_ = v___x_2201_;
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_a_2208_);
lean_dec(v___x_2201_);
v___x_2210_ = lean_box(0);
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
v_resetjp_2209_:
{
lean_object* v___x_2213_; 
if (v_isShared_2211_ == 0)
{
v___x_2213_ = v___x_2210_;
goto v_reusejp_2212_;
}
else
{
lean_object* v_reuseFailAlloc_2214_; 
v_reuseFailAlloc_2214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2214_, 0, v_a_2208_);
v___x_2213_ = v_reuseFailAlloc_2214_;
goto v_reusejp_2212_;
}
v_reusejp_2212_:
{
return v___x_2213_;
}
}
}
}
else
{
lean_object* v_a_2216_; lean_object* v___x_2218_; uint8_t v_isShared_2219_; uint8_t v_isSharedCheck_2223_; 
lean_dec(v___y_2195_);
lean_dec(v___y_2194_);
lean_dec_ref(v___y_2193_);
lean_dec(v___y_2191_);
lean_dec_ref(v___y_2190_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
v_a_2216_ = lean_ctor_get(v___x_2199_, 0);
v_isSharedCheck_2223_ = !lean_is_exclusive(v___x_2199_);
if (v_isSharedCheck_2223_ == 0)
{
v___x_2218_ = v___x_2199_;
v_isShared_2219_ = v_isSharedCheck_2223_;
goto v_resetjp_2217_;
}
else
{
lean_inc(v_a_2216_);
lean_dec(v___x_2199_);
v___x_2218_ = lean_box(0);
v_isShared_2219_ = v_isSharedCheck_2223_;
goto v_resetjp_2217_;
}
v_resetjp_2217_:
{
lean_object* v___x_2221_; 
if (v_isShared_2219_ == 0)
{
v___x_2221_ = v___x_2218_;
goto v_reusejp_2220_;
}
else
{
lean_object* v_reuseFailAlloc_2222_; 
v_reuseFailAlloc_2222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2222_, 0, v_a_2216_);
v___x_2221_ = v_reuseFailAlloc_2222_;
goto v_reusejp_2220_;
}
v_reusejp_2220_:
{
return v___x_2221_;
}
}
}
}
v___jp_2224_:
{
lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; 
v___x_2236_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__16, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__16_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__16);
v___x_2237_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__6));
v___x_2238_ = lean_array_to_list(v___y_2234_);
v___x_2239_ = l_String_intercalate(v___x_2237_, v___x_2238_);
v___x_2240_ = l_Lean_stringToMessageData(v___x_2239_);
v___x_2241_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2241_, 0, v___x_2236_);
lean_ctor_set(v___x_2241_, 1, v___x_2240_);
v___x_2242_ = lp_mathlib_Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12(v___y_2235_, v___x_2241_, v___y_2231_, v___y_2228_);
lean_dec(v___y_2235_);
if (lean_obj_tag(v___x_2242_) == 0)
{
lean_dec_ref_known(v___x_2242_, 1);
v___y_2190_ = v___y_2225_;
v___y_2191_ = v___y_2226_;
v___y_2192_ = v___y_2227_;
v___y_2193_ = v___y_2229_;
v___y_2194_ = v___y_2230_;
v___y_2195_ = v___y_2232_;
v___y_2196_ = v___y_2233_;
v___y_2197_ = v___y_2231_;
v___y_2198_ = v___y_2228_;
goto v___jp_2189_;
}
else
{
lean_dec(v___y_2232_);
lean_dec(v___y_2230_);
lean_dec_ref(v___y_2229_);
lean_dec(v___y_2226_);
lean_dec_ref(v___y_2225_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
return v___x_2242_;
}
}
v___jp_2243_:
{
size_t v_sz_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; 
v_sz_2257_ = lean_array_size(v___y_2256_);
v___x_2258_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__14(v___y_2249_, v_sz_2257_, v___y_2254_, v___y_2256_);
v___x_2259_ = l_Lean_Syntax_find_x3f(v___y_2244_, v___f_1817_);
if (lean_obj_tag(v___x_2259_) == 0)
{
lean_object* v___x_2260_; 
v___x_2260_ = lean_box(0);
v___y_2225_ = v___y_2248_;
v___y_2226_ = v___y_2250_;
v___y_2227_ = v___y_2251_;
v___y_2228_ = v___y_2252_;
v___y_2229_ = v___y_2253_;
v___y_2230_ = v___y_2255_;
v___y_2231_ = v___y_2245_;
v___y_2232_ = v___y_2246_;
v___y_2233_ = v___y_2247_;
v___y_2234_ = v___x_2258_;
v___y_2235_ = v___x_2260_;
goto v___jp_2224_;
}
else
{
lean_object* v_val_2261_; 
v_val_2261_ = lean_ctor_get(v___x_2259_, 0);
lean_inc(v_val_2261_);
lean_dec_ref_known(v___x_2259_, 1);
v___y_2225_ = v___y_2248_;
v___y_2226_ = v___y_2250_;
v___y_2227_ = v___y_2251_;
v___y_2228_ = v___y_2252_;
v___y_2229_ = v___y_2253_;
v___y_2230_ = v___y_2255_;
v___y_2231_ = v___y_2245_;
v___y_2232_ = v___y_2246_;
v___y_2233_ = v___y_2247_;
v___y_2234_ = v___x_2258_;
v___y_2235_ = v_val_2261_;
goto v___jp_2224_;
}
}
v___jp_2262_:
{
lean_object* v___x_2279_; 
v___x_2279_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(v___y_2267_, v___y_2276_, v___y_2277_, v___y_2278_);
lean_dec(v___y_2278_);
lean_dec(v___y_2267_);
v___y_2244_ = v___y_2263_;
v___y_2245_ = v___y_2264_;
v___y_2246_ = v___y_2265_;
v___y_2247_ = v___y_2266_;
v___y_2248_ = v___y_2268_;
v___y_2249_ = v___y_2269_;
v___y_2250_ = v___y_2270_;
v___y_2251_ = v___y_2271_;
v___y_2252_ = v___y_2272_;
v___y_2253_ = v___y_2273_;
v___y_2254_ = v___y_2274_;
v___y_2255_ = v___y_2275_;
v___y_2256_ = v___x_2279_;
goto v___jp_2243_;
}
v___jp_2280_:
{
uint8_t v___x_2297_; 
v___x_2297_ = lean_nat_dec_le(v___y_2296_, v___y_2281_);
if (v___x_2297_ == 0)
{
lean_dec(v___y_2281_);
lean_inc(v___y_2296_);
v___y_2263_ = v___y_2282_;
v___y_2264_ = v___y_2283_;
v___y_2265_ = v___y_2284_;
v___y_2266_ = v___y_2285_;
v___y_2267_ = v___y_2286_;
v___y_2268_ = v___y_2287_;
v___y_2269_ = v___y_2288_;
v___y_2270_ = v___y_2289_;
v___y_2271_ = v___y_2290_;
v___y_2272_ = v___y_2291_;
v___y_2273_ = v___y_2292_;
v___y_2274_ = v___y_2293_;
v___y_2275_ = v___y_2295_;
v___y_2276_ = v___y_2294_;
v___y_2277_ = v___y_2296_;
v___y_2278_ = v___y_2296_;
goto v___jp_2262_;
}
else
{
v___y_2263_ = v___y_2282_;
v___y_2264_ = v___y_2283_;
v___y_2265_ = v___y_2284_;
v___y_2266_ = v___y_2285_;
v___y_2267_ = v___y_2286_;
v___y_2268_ = v___y_2287_;
v___y_2269_ = v___y_2288_;
v___y_2270_ = v___y_2289_;
v___y_2271_ = v___y_2290_;
v___y_2272_ = v___y_2291_;
v___y_2273_ = v___y_2292_;
v___y_2274_ = v___y_2293_;
v___y_2275_ = v___y_2295_;
v___y_2276_ = v___y_2294_;
v___y_2277_ = v___y_2296_;
v___y_2278_ = v___y_2281_;
goto v___jp_2262_;
}
}
v___jp_2298_:
{
lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; uint8_t v___x_2317_; 
v___x_2314_ = lean_mk_empty_array_with_capacity(v___y_2313_);
lean_dec(v___y_2313_);
v___x_2315_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v___x_2314_, v___y_2302_);
v___x_2316_ = lean_array_get_size(v___x_2315_);
v___x_2317_ = lean_nat_dec_eq(v___x_2316_, v___y_2312_);
if (v___x_2317_ == 0)
{
lean_object* v___x_2318_; lean_object* v___x_2319_; uint8_t v___x_2320_; 
v___x_2318_ = lean_unsigned_to_nat(1u);
v___x_2319_ = lean_nat_sub(v___x_2316_, v___x_2318_);
v___x_2320_ = lean_nat_dec_le(v___y_2312_, v___x_2319_);
if (v___x_2320_ == 0)
{
lean_dec(v___y_2312_);
lean_inc(v___x_2319_);
v___y_2281_ = v___x_2319_;
v___y_2282_ = v___y_2299_;
v___y_2283_ = v___y_2300_;
v___y_2284_ = v___y_2301_;
v___y_2285_ = v___y_2303_;
v___y_2286_ = v___x_2316_;
v___y_2287_ = v___y_2304_;
v___y_2288_ = v___y_2305_;
v___y_2289_ = v___y_2306_;
v___y_2290_ = v___y_2307_;
v___y_2291_ = v___y_2308_;
v___y_2292_ = v___y_2309_;
v___y_2293_ = v___y_2310_;
v___y_2294_ = v___x_2315_;
v___y_2295_ = v___y_2311_;
v___y_2296_ = v___x_2319_;
goto v___jp_2280_;
}
else
{
v___y_2281_ = v___x_2319_;
v___y_2282_ = v___y_2299_;
v___y_2283_ = v___y_2300_;
v___y_2284_ = v___y_2301_;
v___y_2285_ = v___y_2303_;
v___y_2286_ = v___x_2316_;
v___y_2287_ = v___y_2304_;
v___y_2288_ = v___y_2305_;
v___y_2289_ = v___y_2306_;
v___y_2290_ = v___y_2307_;
v___y_2291_ = v___y_2308_;
v___y_2292_ = v___y_2309_;
v___y_2293_ = v___y_2310_;
v___y_2294_ = v___x_2315_;
v___y_2295_ = v___y_2311_;
v___y_2296_ = v___y_2312_;
goto v___jp_2280_;
}
}
else
{
lean_dec(v___y_2312_);
v___y_2244_ = v___y_2299_;
v___y_2245_ = v___y_2300_;
v___y_2246_ = v___y_2301_;
v___y_2247_ = v___y_2303_;
v___y_2248_ = v___y_2304_;
v___y_2249_ = v___y_2305_;
v___y_2250_ = v___y_2306_;
v___y_2251_ = v___y_2307_;
v___y_2252_ = v___y_2308_;
v___y_2253_ = v___y_2309_;
v___y_2254_ = v___y_2310_;
v___y_2255_ = v___y_2311_;
v___y_2256_ = v___x_2315_;
goto v___jp_2243_;
}
}
v___jp_2321_:
{
if (v___y_2335_ == 0)
{
lean_dec(v___y_2334_);
lean_dec(v___y_2326_);
lean_dec(v___y_2322_);
lean_dec_ref(v___f_1817_);
v___y_2190_ = v___y_2327_;
v___y_2191_ = v___y_2328_;
v___y_2192_ = v___y_2329_;
v___y_2193_ = v___y_2331_;
v___y_2194_ = v___y_2333_;
v___y_2195_ = v___y_2324_;
v___y_2196_ = v___y_2325_;
v___y_2197_ = v___y_2323_;
v___y_2198_ = v___y_2330_;
goto v___jp_2189_;
}
else
{
if (lean_obj_tag(v___y_2326_) == 0)
{
lean_object* v_size_2336_; 
v_size_2336_ = lean_ctor_get(v___y_2326_, 0);
lean_inc(v_size_2336_);
v___y_2299_ = v___y_2322_;
v___y_2300_ = v___y_2323_;
v___y_2301_ = v___y_2324_;
v___y_2302_ = v___y_2326_;
v___y_2303_ = v___y_2325_;
v___y_2304_ = v___y_2327_;
v___y_2305_ = v___y_2335_;
v___y_2306_ = v___y_2328_;
v___y_2307_ = v___y_2329_;
v___y_2308_ = v___y_2330_;
v___y_2309_ = v___y_2331_;
v___y_2310_ = v___y_2332_;
v___y_2311_ = v___y_2333_;
v___y_2312_ = v___y_2334_;
v___y_2313_ = v_size_2336_;
goto v___jp_2298_;
}
else
{
lean_inc(v___y_2334_);
v___y_2299_ = v___y_2322_;
v___y_2300_ = v___y_2323_;
v___y_2301_ = v___y_2324_;
v___y_2302_ = v___y_2326_;
v___y_2303_ = v___y_2325_;
v___y_2304_ = v___y_2327_;
v___y_2305_ = v___y_2335_;
v___y_2306_ = v___y_2328_;
v___y_2307_ = v___y_2329_;
v___y_2308_ = v___y_2330_;
v___y_2309_ = v___y_2331_;
v___y_2310_ = v___y_2332_;
v___y_2311_ = v___y_2333_;
v___y_2312_ = v___y_2334_;
v___y_2313_ = v___y_2334_;
goto v___jp_2298_;
}
}
}
v___jp_2337_:
{
if (v___y_2352_ == 0)
{
v___y_2322_ = v___y_2338_;
v___y_2323_ = v___y_2339_;
v___y_2324_ = v___y_2340_;
v___y_2325_ = v___y_2341_;
v___y_2326_ = v___y_2342_;
v___y_2327_ = v___y_2343_;
v___y_2328_ = v___y_2344_;
v___y_2329_ = v___y_2345_;
v___y_2330_ = v___y_2346_;
v___y_2331_ = v___y_2347_;
v___y_2332_ = v___y_2348_;
v___y_2333_ = v___y_2349_;
v___y_2334_ = v___y_2351_;
v___y_2335_ = v___y_2350_;
goto v___jp_2321_;
}
else
{
v___y_2322_ = v___y_2338_;
v___y_2323_ = v___y_2339_;
v___y_2324_ = v___y_2340_;
v___y_2325_ = v___y_2341_;
v___y_2326_ = v___y_2342_;
v___y_2327_ = v___y_2343_;
v___y_2328_ = v___y_2344_;
v___y_2329_ = v___y_2345_;
v___y_2330_ = v___y_2346_;
v___y_2331_ = v___y_2347_;
v___y_2332_ = v___y_2348_;
v___y_2333_ = v___y_2349_;
v___y_2334_ = v___y_2351_;
v___y_2335_ = v___y_2341_;
goto v___jp_2321_;
}
}
v___jp_2353_:
{
lean_object* v_fileName_2367_; lean_object* v_ref_2368_; lean_object* v___x_2369_; 
v_fileName_2367_ = lean_ctor_get(v___y_2354_, 0);
v_ref_2368_ = lean_ctor_get(v___y_2354_, 7);
v___x_2369_ = l_IO_FS_readFile(v_fileName_2367_);
if (lean_obj_tag(v___x_2369_) == 0)
{
lean_object* v_a_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; 
v_a_2370_ = lean_ctor_get(v___x_2369_, 0);
lean_inc(v_a_2370_);
lean_dec_ref_known(v___x_2369_, 1);
v___x_2371_ = lean_string_utf8_byte_size(v_a_2370_);
lean_inc_ref(v_fileName_2367_);
v___x_2372_ = l_Lean_Parser_mkInputContext___redArg(v_a_2370_, v_fileName_2367_, v___x_2096_, v___x_2371_);
v___x_2373_ = l_Lean_Parser_parseHeader(v___x_2372_);
if (lean_obj_tag(v___x_2373_) == 0)
{
lean_object* v_a_2374_; lean_object* v_fst_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; 
v_a_2374_ = lean_ctor_get(v___x_2373_, 0);
lean_inc(v_a_2374_);
lean_dec_ref_known(v___x_2373_, 1);
v_fst_2375_ = lean_ctor_get(v_a_2374_, 0);
lean_inc_n(v_fst_2375_, 2);
lean_dec(v_a_2374_);
v___x_2376_ = l_Lean_NameSet_ofArray(v___y_2366_);
lean_dec_ref(v___y_2366_);
lean_inc(v___x_2376_);
v___x_2377_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(v___x_2376_, v___y_2355_);
v___x_2378_ = lean_box(0);
v___x_2379_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__13(v_fst_2375_, v___y_2364_, v___x_2378_, v___x_2377_, v___y_2354_, v___y_2359_);
if (lean_obj_tag(v___x_2379_) == 0)
{
lean_object* v___x_2380_; 
lean_dec_ref_known(v___x_2379_, 1);
lean_inc(v___y_2355_);
v___x_2380_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(v___y_2355_, v___x_2376_);
lean_dec(v___x_2376_);
if (lean_obj_tag(v___x_2380_) == 0)
{
v___y_2338_ = v_fst_2375_;
v___y_2339_ = v___y_2354_;
v___y_2340_ = v___y_2355_;
v___y_2341_ = v___y_2356_;
v___y_2342_ = v___x_2380_;
v___y_2343_ = v___y_2357_;
v___y_2344_ = v___y_2358_;
v___y_2345_ = v___y_2360_;
v___y_2346_ = v___y_2359_;
v___y_2347_ = v___y_2361_;
v___y_2348_ = v___y_2362_;
v___y_2349_ = v___y_2363_;
v___y_2350_ = v___y_2364_;
v___y_2351_ = v___y_2365_;
v___y_2352_ = v___y_2356_;
goto v___jp_2337_;
}
else
{
v___y_2338_ = v_fst_2375_;
v___y_2339_ = v___y_2354_;
v___y_2340_ = v___y_2355_;
v___y_2341_ = v___y_2356_;
v___y_2342_ = v___x_2380_;
v___y_2343_ = v___y_2357_;
v___y_2344_ = v___y_2358_;
v___y_2345_ = v___y_2360_;
v___y_2346_ = v___y_2359_;
v___y_2347_ = v___y_2361_;
v___y_2348_ = v___y_2362_;
v___y_2349_ = v___y_2363_;
v___y_2350_ = v___y_2364_;
v___y_2351_ = v___y_2365_;
v___y_2352_ = v___y_2364_;
goto v___jp_2337_;
}
}
else
{
lean_object* v_a_2381_; lean_object* v___x_2383_; uint8_t v_isShared_2384_; uint8_t v_isSharedCheck_2388_; 
lean_dec(v___x_2376_);
lean_dec(v_fst_2375_);
lean_dec(v___y_2365_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2361_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
lean_dec(v___y_2355_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2381_ = lean_ctor_get(v___x_2379_, 0);
v_isSharedCheck_2388_ = !lean_is_exclusive(v___x_2379_);
if (v_isSharedCheck_2388_ == 0)
{
v___x_2383_ = v___x_2379_;
v_isShared_2384_ = v_isSharedCheck_2388_;
goto v_resetjp_2382_;
}
else
{
lean_inc(v_a_2381_);
lean_dec(v___x_2379_);
v___x_2383_ = lean_box(0);
v_isShared_2384_ = v_isSharedCheck_2388_;
goto v_resetjp_2382_;
}
v_resetjp_2382_:
{
lean_object* v___x_2386_; 
if (v_isShared_2384_ == 0)
{
v___x_2386_ = v___x_2383_;
goto v_reusejp_2385_;
}
else
{
lean_object* v_reuseFailAlloc_2387_; 
v_reuseFailAlloc_2387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2387_, 0, v_a_2381_);
v___x_2386_ = v_reuseFailAlloc_2387_;
goto v_reusejp_2385_;
}
v_reusejp_2385_:
{
return v___x_2386_;
}
}
}
}
else
{
lean_object* v_a_2389_; lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2400_; 
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2361_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
lean_dec(v___y_2355_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2389_ = lean_ctor_get(v___x_2373_, 0);
v_isSharedCheck_2400_ = !lean_is_exclusive(v___x_2373_);
if (v_isSharedCheck_2400_ == 0)
{
v___x_2391_ = v___x_2373_;
v_isShared_2392_ = v_isSharedCheck_2400_;
goto v_resetjp_2390_;
}
else
{
lean_inc(v_a_2389_);
lean_dec(v___x_2373_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2400_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
lean_object* v___x_2393_; lean_object* v___x_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2398_; 
v___x_2393_ = lean_io_error_to_string(v_a_2389_);
v___x_2394_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2394_, 0, v___x_2393_);
v___x_2395_ = l_Lean_MessageData_ofFormat(v___x_2394_);
lean_inc(v_ref_2368_);
v___x_2396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2396_, 0, v_ref_2368_);
lean_ctor_set(v___x_2396_, 1, v___x_2395_);
if (v_isShared_2392_ == 0)
{
lean_ctor_set(v___x_2391_, 0, v___x_2396_);
v___x_2398_ = v___x_2391_;
goto v_reusejp_2397_;
}
else
{
lean_object* v_reuseFailAlloc_2399_; 
v_reuseFailAlloc_2399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2399_, 0, v___x_2396_);
v___x_2398_ = v_reuseFailAlloc_2399_;
goto v_reusejp_2397_;
}
v_reusejp_2397_:
{
return v___x_2398_;
}
}
}
}
else
{
lean_object* v_a_2401_; lean_object* v___x_2403_; uint8_t v_isShared_2404_; uint8_t v_isSharedCheck_2412_; 
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2361_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
lean_dec(v___y_2355_);
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v_a_2401_ = lean_ctor_get(v___x_2369_, 0);
v_isSharedCheck_2412_ = !lean_is_exclusive(v___x_2369_);
if (v_isSharedCheck_2412_ == 0)
{
v___x_2403_ = v___x_2369_;
v_isShared_2404_ = v_isSharedCheck_2412_;
goto v_resetjp_2402_;
}
else
{
lean_inc(v_a_2401_);
lean_dec(v___x_2369_);
v___x_2403_ = lean_box(0);
v_isShared_2404_ = v_isSharedCheck_2412_;
goto v_resetjp_2402_;
}
v_resetjp_2402_:
{
lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v___x_2410_; 
v___x_2405_ = lean_io_error_to_string(v_a_2401_);
v___x_2406_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2406_, 0, v___x_2405_);
v___x_2407_ = l_Lean_MessageData_ofFormat(v___x_2406_);
lean_inc(v_ref_2368_);
v___x_2408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2408_, 0, v_ref_2368_);
lean_ctor_set(v___x_2408_, 1, v___x_2407_);
if (v_isShared_2404_ == 0)
{
lean_ctor_set(v___x_2403_, 0, v___x_2408_);
v___x_2410_ = v___x_2403_;
goto v_reusejp_2409_;
}
else
{
lean_object* v_reuseFailAlloc_2411_; 
v_reuseFailAlloc_2411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2411_, 0, v___x_2408_);
v___x_2410_ = v_reuseFailAlloc_2411_;
goto v_reusejp_2409_;
}
v_reusejp_2409_:
{
return v___x_2410_;
}
}
}
}
v___jp_2413_:
{
lean_object* v___x_2423_; lean_object* v_transClosure_2424_; lean_object* v_minImports_2425_; lean_object* v_importSize_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; uint8_t v___x_2436_; 
v___x_2423_ = lean_st_ref_get(v___y_2416_);
v_transClosure_2424_ = lean_ctor_get(v___x_2423_, 0);
lean_inc(v_transClosure_2424_);
v_minImports_2425_ = lean_ctor_get(v___x_2423_, 1);
lean_inc(v_minImports_2425_);
v_importSize_2426_ = lean_ctor_get(v___x_2423_, 2);
lean_inc(v_importSize_2426_);
lean_dec(v___x_2423_);
v___x_2427_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__17));
lean_inc_ref_n(v___y_2415_, 2);
lean_inc_ref_n(v___y_2420_, 2);
lean_inc_ref_n(v___y_2417_, 2);
v___x_2428_ = l_Lean_Name_mkStr4(v___y_2417_, v___y_2420_, v___y_2415_, v___x_2427_);
v___x_2429_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__18));
v___x_2430_ = l_Lean_Name_mkStr4(v___y_2417_, v___y_2420_, v___y_2415_, v___x_2429_);
v___x_2431_ = lean_unsigned_to_nat(2u);
v___x_2432_ = lean_mk_empty_array_with_capacity(v___x_2431_);
v___x_2433_ = lean_array_push(v___x_2432_, v___x_2428_);
v___x_2434_ = lean_array_push(v___x_2433_, v___x_2430_);
lean_inc(v_stx_1818_);
v___x_2435_ = l_Lean_Syntax_getKind(v_stx_1818_);
v___x_2436_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__10(v___x_2434_, v___x_2435_);
lean_dec(v___x_2435_);
lean_dec_ref(v___x_2434_);
if (v___x_2436_ == 0)
{
lean_dec_ref(v___f_1817_);
v___y_2190_ = v___y_2414_;
v___y_2191_ = v_transClosure_2424_;
v___y_2192_ = v___y_2416_;
v___y_2193_ = v___y_2418_;
v___y_2194_ = v_importSize_2426_;
v___y_2195_ = v_minImports_2425_;
v___y_2196_ = v___y_2419_;
v___y_2197_ = v___y_2421_;
v___y_2198_ = v___y_2422_;
goto v___jp_2189_;
}
else
{
lean_object* v___x_2437_; size_t v_sz_2438_; size_t v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; uint8_t v___x_2444_; 
v___x_2437_ = l_Lean_Environment_imports(v___y_2418_);
v_sz_2438_ = lean_array_size(v___x_2437_);
v___x_2439_ = ((size_t)0ULL);
v___x_2440_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__11(v_sz_2438_, v___x_2439_, v___x_2437_);
v___x_2441_ = lean_unsigned_to_nat(0u);
v___x_2442_ = lean_array_get_size(v___x_2440_);
v___x_2443_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__19));
v___x_2444_ = lean_nat_dec_lt(v___x_2441_, v___x_2442_);
if (v___x_2444_ == 0)
{
lean_dec_ref(v___x_2440_);
v___y_2354_ = v___y_2421_;
v___y_2355_ = v_minImports_2425_;
v___y_2356_ = v___y_2419_;
v___y_2357_ = v___y_2414_;
v___y_2358_ = v_transClosure_2424_;
v___y_2359_ = v___y_2422_;
v___y_2360_ = v___y_2416_;
v___y_2361_ = v___y_2418_;
v___y_2362_ = v___x_2439_;
v___y_2363_ = v_importSize_2426_;
v___y_2364_ = v___x_2436_;
v___y_2365_ = v___x_2441_;
v___y_2366_ = v___x_2443_;
goto v___jp_2353_;
}
else
{
uint8_t v___x_2445_; 
v___x_2445_ = lean_nat_dec_le(v___x_2442_, v___x_2442_);
if (v___x_2445_ == 0)
{
if (v___x_2444_ == 0)
{
lean_dec_ref(v___x_2440_);
v___y_2354_ = v___y_2421_;
v___y_2355_ = v_minImports_2425_;
v___y_2356_ = v___y_2419_;
v___y_2357_ = v___y_2414_;
v___y_2358_ = v_transClosure_2424_;
v___y_2359_ = v___y_2422_;
v___y_2360_ = v___y_2416_;
v___y_2361_ = v___y_2418_;
v___y_2362_ = v___x_2439_;
v___y_2363_ = v_importSize_2426_;
v___y_2364_ = v___x_2436_;
v___y_2365_ = v___x_2441_;
v___y_2366_ = v___x_2443_;
goto v___jp_2353_;
}
else
{
size_t v___x_2446_; lean_object* v___x_2447_; 
v___x_2446_ = lean_usize_of_nat(v___x_2442_);
v___x_2447_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15(v___x_2440_, v___x_2439_, v___x_2446_, v___x_2443_);
lean_dec_ref(v___x_2440_);
v___y_2354_ = v___y_2421_;
v___y_2355_ = v_minImports_2425_;
v___y_2356_ = v___y_2419_;
v___y_2357_ = v___y_2414_;
v___y_2358_ = v_transClosure_2424_;
v___y_2359_ = v___y_2422_;
v___y_2360_ = v___y_2416_;
v___y_2361_ = v___y_2418_;
v___y_2362_ = v___x_2439_;
v___y_2363_ = v_importSize_2426_;
v___y_2364_ = v___x_2436_;
v___y_2365_ = v___x_2441_;
v___y_2366_ = v___x_2447_;
goto v___jp_2353_;
}
}
else
{
size_t v___x_2448_; lean_object* v___x_2449_; 
v___x_2448_ = lean_usize_of_nat(v___x_2442_);
v___x_2449_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__15(v___x_2440_, v___x_2439_, v___x_2448_, v___x_2443_);
lean_dec_ref(v___x_2440_);
v___y_2354_ = v___y_2421_;
v___y_2355_ = v_minImports_2425_;
v___y_2356_ = v___y_2419_;
v___y_2357_ = v___y_2414_;
v___y_2358_ = v_transClosure_2424_;
v___y_2359_ = v___y_2422_;
v___y_2360_ = v___y_2416_;
v___y_2361_ = v___y_2418_;
v___y_2362_ = v___x_2439_;
v___y_2363_ = v_importSize_2426_;
v___y_2364_ = v___x_2436_;
v___y_2365_ = v___x_2441_;
v___y_2366_ = v___x_2449_;
goto v___jp_2353_;
}
}
}
}
v___jp_2450_:
{
if (v___y_2458_ == 0)
{
v___y_2414_ = v___y_2451_;
v___y_2415_ = v___y_2452_;
v___y_2416_ = v___y_2453_;
v___y_2417_ = v___y_2454_;
v___y_2418_ = v___y_2455_;
v___y_2419_ = v___y_2456_;
v___y_2420_ = v___y_2457_;
v___y_2421_ = v___y_1819_;
v___y_2422_ = v___y_1820_;
goto v___jp_2413_;
}
else
{
lean_object* v___x_2459_; lean_object* v_minImports_2460_; lean_object* v_importSize_2461_; lean_object* v___x_2463_; uint8_t v_isShared_2464_; uint8_t v_isSharedCheck_2472_; 
v___x_2459_ = lean_st_ref_take(v___y_2453_);
v_minImports_2460_ = lean_ctor_get(v___x_2459_, 1);
v_importSize_2461_ = lean_ctor_get(v___x_2459_, 2);
v_isSharedCheck_2472_ = !lean_is_exclusive(v___x_2459_);
if (v_isSharedCheck_2472_ == 0)
{
lean_object* v_unused_2473_; 
v_unused_2473_ = lean_ctor_get(v___x_2459_, 0);
lean_dec(v_unused_2473_);
v___x_2463_ = v___x_2459_;
v_isShared_2464_ = v_isSharedCheck_2472_;
goto v_resetjp_2462_;
}
else
{
lean_inc(v_importSize_2461_);
lean_inc(v_minImports_2460_);
lean_dec(v___x_2459_);
v___x_2463_ = lean_box(0);
v_isShared_2464_ = v_isSharedCheck_2472_;
goto v_resetjp_2462_;
}
v_resetjp_2462_:
{
lean_object* v___x_2465_; lean_object* v___x_2466_; lean_object* v___x_2467_; lean_object* v___x_2469_; 
v___x_2465_ = lp_importGraph_Lean_Environment_importGraph(v___y_2455_);
v___x_2466_ = lp_importGraph_Lean_NameMap_transitiveClosure(v___x_2465_);
v___x_2467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2467_, 0, v___x_2466_);
if (v_isShared_2464_ == 0)
{
lean_ctor_set(v___x_2463_, 0, v___x_2467_);
v___x_2469_ = v___x_2463_;
goto v_reusejp_2468_;
}
else
{
lean_object* v_reuseFailAlloc_2471_; 
v_reuseFailAlloc_2471_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2471_, 0, v___x_2467_);
lean_ctor_set(v_reuseFailAlloc_2471_, 1, v_minImports_2460_);
lean_ctor_set(v_reuseFailAlloc_2471_, 2, v_importSize_2461_);
v___x_2469_ = v_reuseFailAlloc_2471_;
goto v_reusejp_2468_;
}
v_reusejp_2468_:
{
lean_object* v___x_2470_; 
v___x_2470_ = lean_st_ref_set(v___y_2453_, v___x_2469_);
v___y_2414_ = v___y_2451_;
v___y_2415_ = v___y_2452_;
v___y_2416_ = v___y_2453_;
v___y_2417_ = v___y_2454_;
v___y_2418_ = v___y_2455_;
v___y_2419_ = v___y_2456_;
v___y_2420_ = v___y_2457_;
v___y_2421_ = v___y_1819_;
v___y_2422_ = v___y_1820_;
goto v___jp_2413_;
}
}
}
}
v___jp_2474_:
{
lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; uint8_t v___x_2489_; 
v___x_2476_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__2));
v___x_2477_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__6));
v___x_2478_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__26));
v___x_2479_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__27));
v___x_2480_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__28));
lean_inc_n(v___y_2475_, 3);
v___x_2481_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2481_, 0, v___y_2475_);
lean_ctor_set(v___x_2481_, 1, v___x_2479_);
v___x_2482_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__20, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__20_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__20);
v___x_2483_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__1));
v___x_2484_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25, &lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25_once, _init_lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__25);
v___x_2485_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2485_, 0, v___y_2475_);
lean_ctor_set(v___x_2485_, 1, v___x_2483_);
lean_ctor_set(v___x_2485_, 2, v___x_2484_);
v___x_2486_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_MinImports___aux__Mathlib__Tactic__Linter__MinImports______macroRules__Mathlib__Linter__MinImports__command_x23import__bumps__1___closed__44));
v___x_2487_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2487_, 0, v___y_2475_);
lean_ctor_set(v___x_2487_, 1, v___x_2486_);
v___x_2488_ = l_Lean_Syntax_node4(v___y_2475_, v___x_2480_, v___x_2481_, v___x_2482_, v___x_2485_, v___x_2487_);
v___x_2489_ = l_Lean_Syntax_structEq(v_stx_1818_, v___x_2488_);
lean_dec(v___x_2488_);
if (v___x_2489_ == 0)
{
lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v_env_2493_; lean_object* v_transClosure_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___f_2497_; 
v___x_2490_ = lean_st_ref_get(v___y_1820_);
v___x_2491_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_minImportsRef;
v___x_2492_ = lean_st_ref_get(v___x_2491_);
v_env_2493_ = lean_ctor_get(v___x_2490_, 0);
lean_inc_ref(v_env_2493_);
lean_dec(v___x_2490_);
v_transClosure_2494_ = lean_ctor_get(v___x_2492_, 0);
lean_inc(v_transClosure_2494_);
lean_dec(v___x_2492_);
v___x_2495_ = lean_box(v___x_2096_);
v___x_2496_ = lean_box(v___x_2489_);
v___f_2497_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__1___boxed), 3, 2);
lean_closure_set(v___f_2497_, 0, v___x_2495_);
lean_closure_set(v___f_2497_, 1, v___x_2496_);
if (lean_obj_tag(v_transClosure_2494_) == 0)
{
v___y_2451_ = v___f_2497_;
v___y_2452_ = v___x_2478_;
v___y_2453_ = v___x_2491_;
v___y_2454_ = v___x_2476_;
v___y_2455_ = v_env_2493_;
v___y_2456_ = v___x_2489_;
v___y_2457_ = v___x_2477_;
v___y_2458_ = v___x_2096_;
goto v___jp_2450_;
}
else
{
lean_dec_ref_known(v_transClosure_2494_, 1);
v___y_2451_ = v___f_2497_;
v___y_2452_ = v___x_2478_;
v___y_2453_ = v___x_2491_;
v___y_2454_ = v___x_2476_;
v___y_2455_ = v_env_2493_;
v___y_2456_ = v___x_2489_;
v___y_2457_ = v___x_2477_;
v___y_2458_ = v___x_2489_;
goto v___jp_2450_;
}
}
else
{
lean_object* v___x_2498_; lean_object* v___x_2499_; 
lean_del_object(v___x_1825_);
lean_dec(v_stx_1818_);
lean_dec_ref(v___f_1817_);
v___x_2498_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__23, &lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__23_once, _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___closed__23);
v___x_2499_ = lp_mathlib_Lean_logInfo___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__16(v___x_2498_, v___y_1819_, v___y_1820_);
if (lean_obj_tag(v___x_2499_) == 0)
{
lean_object* v___x_2501_; uint8_t v_isShared_2502_; uint8_t v_isSharedCheck_2507_; 
v_isSharedCheck_2507_ = !lean_is_exclusive(v___x_2499_);
if (v_isSharedCheck_2507_ == 0)
{
lean_object* v_unused_2508_; 
v_unused_2508_ = lean_ctor_get(v___x_2499_, 0);
lean_dec(v_unused_2508_);
v___x_2501_ = v___x_2499_;
v_isShared_2502_ = v_isSharedCheck_2507_;
goto v_resetjp_2500_;
}
else
{
lean_dec(v___x_2499_);
v___x_2501_ = lean_box(0);
v_isShared_2502_ = v_isSharedCheck_2507_;
goto v_resetjp_2500_;
}
v_resetjp_2500_:
{
lean_object* v___x_2503_; lean_object* v___x_2505_; 
v___x_2503_ = lean_box(0);
if (v_isShared_2502_ == 0)
{
lean_ctor_set(v___x_2501_, 0, v___x_2503_);
v___x_2505_ = v___x_2501_;
goto v_reusejp_2504_;
}
else
{
lean_object* v_reuseFailAlloc_2506_; 
v_reuseFailAlloc_2506_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2506_, 0, v___x_2503_);
v___x_2505_ = v_reuseFailAlloc_2506_;
goto v_reusejp_2504_;
}
v_reusejp_2504_:
{
return v___x_2505_;
}
}
}
else
{
return v___x_2499_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2___boxed(lean_object* v___f_2575_, lean_object* v_stx_2576_, lean_object* v___y_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_){
_start:
{
lean_object* v_res_2580_; 
v_res_2580_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter___lam__2(v___f_2575_, v_stx_2576_, v___y_2577_, v___y_2578_);
lean_dec(v___y_2578_);
lean_dec_ref(v___y_2577_);
return v_res_2580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0(lean_object* v_o_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_){
_start:
{
lean_object* v___x_2627_; 
v___x_2627_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___redArg(v_o_2623_, v___y_2625_);
return v___x_2627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0___boxed(lean_object* v_o_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_){
_start:
{
lean_object* v_res_2632_; 
v_res_2632_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__0_spec__0(v_o_2628_, v___y_2629_, v___y_2630_);
lean_dec(v___y_2630_);
lean_dec_ref(v___y_2629_);
return v_res_2632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1(lean_object* v_init_2633_, lean_object* v_t_2634_){
_start:
{
lean_object* v___x_2635_; 
v___x_2635_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__1_spec__2(v_init_2633_, v_t_2634_);
return v___x_2635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2(lean_object* v_00_u03b2_2636_, lean_object* v_k_2637_, lean_object* v_t_2638_, lean_object* v_h_2639_){
_start:
{
lean_object* v___x_2640_; 
v___x_2640_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___redArg(v_k_2637_, v_t_2638_);
return v___x_2640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2___boxed(lean_object* v_00_u03b2_2641_, lean_object* v_k_2642_, lean_object* v_t_2643_, lean_object* v_h_2644_){
_start:
{
lean_object* v_res_2645_; 
v_res_2645_ = lp_mathlib_Std_DTreeMap_Internal_Impl_erase___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__2(v_00_u03b2_2641_, v_k_2642_, v_t_2643_, v_h_2644_);
lean_dec(v_k_2642_);
return v_res_2645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3(lean_object* v_init_2646_, lean_object* v_t_2647_){
_start:
{
lean_object* v___x_2648_; 
v___x_2648_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3_spec__5(v_init_2646_, v_t_2647_);
return v___x_2648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3___boxed(lean_object* v_init_2649_, lean_object* v_t_2650_){
_start:
{
lean_object* v_res_2651_; 
v_res_2651_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__3(v_init_2649_, v_t_2650_);
lean_dec(v_t_2650_);
return v_res_2651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8(lean_object* v_n_2652_, lean_object* v_as_2653_, lean_object* v_lo_2654_, lean_object* v_hi_2655_, lean_object* v_w_2656_, lean_object* v_hlo_2657_, lean_object* v_hhi_2658_){
_start:
{
lean_object* v___x_2659_; 
v___x_2659_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___redArg(v_n_2652_, v_as_2653_, v_lo_2654_, v_hi_2655_);
return v___x_2659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8___boxed(lean_object* v_n_2660_, lean_object* v_as_2661_, lean_object* v_lo_2662_, lean_object* v_hi_2663_, lean_object* v_w_2664_, lean_object* v_hlo_2665_, lean_object* v_hhi_2666_){
_start:
{
lean_object* v_res_2667_; 
v_res_2667_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8(v_n_2660_, v_as_2661_, v_lo_2662_, v_hi_2663_, v_w_2664_, v_hlo_2665_, v_hhi_2666_);
lean_dec(v_hi_2663_);
lean_dec(v_n_2660_);
return v_res_2667_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9(lean_object* v_xs_2668_, lean_object* v_ys_2669_, lean_object* v_hsz_2670_, lean_object* v_x_2671_, lean_object* v_x_2672_){
_start:
{
uint8_t v___x_2673_; 
v___x_2673_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___redArg(v_xs_2668_, v_ys_2669_, v_x_2671_);
return v___x_2673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9___boxed(lean_object* v_xs_2674_, lean_object* v_ys_2675_, lean_object* v_hsz_2676_, lean_object* v_x_2677_, lean_object* v_x_2678_){
_start:
{
uint8_t v_res_2679_; lean_object* v_r_2680_; 
v_res_2679_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__9(v_xs_2674_, v_ys_2675_, v_hsz_2676_, v_x_2677_, v_x_2678_);
lean_dec_ref(v_ys_2675_);
lean_dec_ref(v_xs_2674_);
v_r_2680_ = lean_box(v_res_2679_);
return v_r_2680_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10(lean_object* v_xs_2681_, lean_object* v_ys_2682_, lean_object* v_hsz_2683_, lean_object* v_x_2684_, lean_object* v_x_2685_){
_start:
{
uint8_t v___x_2686_; 
v___x_2686_ = lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___redArg(v_xs_2681_, v_ys_2682_, v_x_2684_);
return v___x_2686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10___boxed(lean_object* v_xs_2687_, lean_object* v_ys_2688_, lean_object* v_hsz_2689_, lean_object* v_x_2690_, lean_object* v_x_2691_){
_start:
{
uint8_t v_res_2692_; lean_object* v_r_2693_; 
v_res_2692_ = lp_mathlib_Array_isEqvAux___at___00Array_instDecidableEqImpl___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__7_spec__10(v_xs_2687_, v_ys_2688_, v_hsz_2689_, v_x_2690_, v_x_2691_);
lean_dec_ref(v_ys_2688_);
lean_dec_ref(v_xs_2687_);
v_r_2693_ = lean_box(v_res_2692_);
return v_r_2693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12(lean_object* v_n_2694_, lean_object* v_lo_2695_, lean_object* v_hi_2696_, lean_object* v_hhi_2697_, lean_object* v_pivot_2698_, lean_object* v_as_2699_, lean_object* v_i_2700_, lean_object* v_k_2701_, lean_object* v_ilo_2702_, lean_object* v_ik_2703_, lean_object* v_w_2704_){
_start:
{
lean_object* v___x_2705_; 
v___x_2705_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___redArg(v_hi_2696_, v_pivot_2698_, v_as_2699_, v_i_2700_, v_k_2701_);
return v___x_2705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12___boxed(lean_object* v_n_2706_, lean_object* v_lo_2707_, lean_object* v_hi_2708_, lean_object* v_hhi_2709_, lean_object* v_pivot_2710_, lean_object* v_as_2711_, lean_object* v_i_2712_, lean_object* v_k_2713_, lean_object* v_ilo_2714_, lean_object* v_ik_2715_, lean_object* v_w_2716_){
_start:
{
lean_object* v_res_2717_; 
v_res_2717_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__8_spec__12(v_n_2706_, v_lo_2707_, v_hi_2708_, v_hhi_2709_, v_pivot_2710_, v_as_2711_, v_i_2712_, v_k_2713_, v_ilo_2714_, v_ik_2715_, v_w_2716_);
lean_dec(v_pivot_2710_);
lean_dec(v_hi_2708_);
lean_dec(v_lo_2707_);
lean_dec(v_n_2706_);
return v_res_2717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20(lean_object* v_msgData_2718_, lean_object* v___y_2719_, lean_object* v___y_2720_){
_start:
{
lean_object* v___x_2722_; 
v___x_2722_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___redArg(v_msgData_2718_, v___y_2720_);
return v___x_2722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20___boxed(lean_object* v_msgData_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_){
_start:
{
lean_object* v_res_2727_; 
v_res_2727_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logWarningAt___at___00__private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter_spec__12_spec__18_spec__20(v_msgData_2723_, v___y_2724_, v___y_2725_);
lean_dec(v___y_2725_);
lean_dec_ref(v___y_2724_);
return v_res_2727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_1947129272____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2729_; lean_object* v___x_2730_; 
v___x_2729_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_minImportsLinter));
v___x_2730_ = l_Lean_Elab_Command_addLinter(v___x_2729_);
return v___x_2730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_1947129272____hygCtx___hyg_2____boxed(lean_object* v_a_2731_){
_start:
{
lean_object* v_res_2732_; 
v_res_2732_ = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_1947129272____hygCtx___hyg_2_();
return v_res_2732_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_MinImports(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_MinImports(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Linter_instInhabitedImportState_default = _init_lp_mathlib_Mathlib_Linter_instInhabitedImportState_default();
lean_mark_persistent(lp_mathlib_Mathlib_Linter_instInhabitedImportState_default);
lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_instInhabitedImportState = _init_lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_instInhabitedImportState();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_instInhabitedImportState);
res = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2382540021____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_minImportsRef = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_minImportsRef);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_3746520658____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_minImports = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_minImports);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_2495106528____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Linter_linter_minImports_increases = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Linter_linter_minImports_increases);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Linter_MinImports_0__Mathlib_Linter_MinImports_initFn_00___x40_Mathlib_Tactic_Linter_MinImports_1947129272____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_MinImports(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_MinImports(builtin);
}
#ifdef __cplusplus
}
#endif
