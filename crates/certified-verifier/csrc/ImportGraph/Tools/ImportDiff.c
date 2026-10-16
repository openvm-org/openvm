// Lean compiler output
// Module: ImportGraph.Tools.ImportDiff
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Lean.Widget.UserWidget public meta import ImportGraph.Imports.ImportGraph
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Array_contains___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_SearchPath_findWithExt(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_array_to_list(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_searchPathRef;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_importModules(lean_object*, lean_object*, uint32_t, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Environment_allImportedModuleNames(lean_object*);
lean_object* l_Lean_Environment_imports(lean_object*);
static const lean_string_object lp_importGraph_command_x23import__diff___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "command#import_diff_"};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__0 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__0_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 49, 209, 78, 36, 207, 131, 32)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__1 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__1_value;
static const lean_string_object lp_importGraph_command_x23import__diff___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__2 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__2_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__3 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__3_value;
static const lean_string_object lp_importGraph_command_x23import__diff___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "#import_diff"};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__4 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__4_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__4_value)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__5 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__5_value;
static const lean_string_object lp_importGraph_command_x23import__diff___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__6 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__6_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__7 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__7_value;
static const lean_string_object lp_importGraph_command_x23import__diff___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__8 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__8_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__9 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__9_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__9_value)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__10 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__10_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__7_value),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__10_value)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__11 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__11_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__3_value),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__5_value),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__11_value)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__12 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__12_value;
static const lean_ctor_object lp_importGraph_command_x23import__diff___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23import__diff___00__closed__12_value)}};
static const lean_object* lp_importGraph_command_x23import__diff___00__closed__13 = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__13_value;
LEAN_EXPORT const lean_object* lp_importGraph_command_x23import__diff__ = (const lean_object*)&lp_importGraph_command_x23import__diff___00__closed__13_value;
static lean_once_cell_t lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__5(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4_spec__6(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9___closed__0 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__0;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__2;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__3;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__4;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__5;
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___closed__0 = (const lean_object*)&lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___closed__0 = (const lean_object*)&lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0;
static const lean_string_object lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__1 = (const lean_object*)&lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__1_value;
static const lean_ctor_object lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__1_value)}};
static const lean_object* lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__2 = (const lean_object*)&lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__2_value;
static lean_once_cell_t lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5(lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__0 = (const lean_object*)&lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__0_value)}};
static const lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__1 = (const lean_object*)&lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__1_value;
static lean_once_cell_t lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "olean"};
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__0 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__0_value;
static const lean_string_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "File "};
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__1 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__1_value;
static lean_once_cell_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__2;
static const lean_string_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = " cannot be found."};
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__3 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__3_value;
static lean_once_cell_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__4;
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Found "};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__0 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__0_value;
static const lean_string_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " additional imports:\n"};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__1 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__1_value;
static const lean_string_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__2 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__2_value;
static const lean_array_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__3 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__3_value;
static const lean_array_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__4 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__4_value;
static const lean_string_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "The following are already imported (possibly transitively):\n"};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__5 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__5_value;
static lean_once_cell_t lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__6;
static const lean_array_object lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__7 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__7_value;
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_30_ = lean_box(0);
v___x_31_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_32_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_32_, 0, v___x_31_);
lean_ctor_set(v___x_32_, 1, v___x_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg(){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_34_ = lean_obj_once(&lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___closed__0, &lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___closed__0_once, _init_lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___closed__0);
v___x_35_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_35_, 0, v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg___boxed(lean_object* v___y_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg();
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0(lean_object* v_00_u03b1_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg();
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___boxed(lean_object* v_00_u03b1_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0(v_00_u03b1_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__5(uint8_t v___x_48_, size_t v_sz_49_, size_t v_i_50_, lean_object* v_bs_51_){
_start:
{
uint8_t v___x_52_; 
v___x_52_ = lean_usize_dec_lt(v_i_50_, v_sz_49_);
if (v___x_52_ == 0)
{
return v_bs_51_;
}
else
{
lean_object* v_v_53_; lean_object* v___x_54_; lean_object* v_bs_x27_55_; uint8_t v___x_56_; lean_object* v___x_57_; size_t v___x_58_; size_t v___x_59_; lean_object* v___x_60_; 
v_v_53_ = lean_array_uget(v_bs_51_, v_i_50_);
v___x_54_ = lean_unsigned_to_nat(0u);
v_bs_x27_55_ = lean_array_uset(v_bs_51_, v_i_50_, v___x_54_);
v___x_56_ = 0;
v___x_57_ = lean_alloc_ctor(0, 1, 3);
lean_ctor_set(v___x_57_, 0, v_v_53_);
lean_ctor_set_uint8(v___x_57_, sizeof(void*)*1, v___x_56_);
lean_ctor_set_uint8(v___x_57_, sizeof(void*)*1 + 1, v___x_48_);
lean_ctor_set_uint8(v___x_57_, sizeof(void*)*1 + 2, v___x_56_);
v___x_58_ = ((size_t)1ULL);
v___x_59_ = lean_usize_add(v_i_50_, v___x_58_);
v___x_60_ = lean_array_uset(v_bs_x27_55_, v_i_50_, v___x_57_);
v_i_50_ = v___x_59_;
v_bs_51_ = v___x_60_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__5___boxed(lean_object* v___x_62_, lean_object* v_sz_63_, lean_object* v_i_64_, lean_object* v_bs_65_){
_start:
{
uint8_t v___x_9822__boxed_66_; size_t v_sz_boxed_67_; size_t v_i_boxed_68_; lean_object* v_res_69_; 
v___x_9822__boxed_66_ = lean_unbox(v___x_62_);
v_sz_boxed_67_ = lean_unbox_usize(v_sz_63_);
lean_dec(v_sz_63_);
v_i_boxed_68_ = lean_unbox_usize(v_i_64_);
lean_dec(v_i_64_);
v_res_69_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__5(v___x_9822__boxed_66_, v_sz_boxed_67_, v_i_boxed_68_, v_bs_65_);
return v_res_69_;
}
}
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4_spec__6(lean_object* v_a_70_, lean_object* v_as_71_, size_t v_i_72_, size_t v_stop_73_){
_start:
{
uint8_t v___x_74_; 
v___x_74_ = lean_usize_dec_eq(v_i_72_, v_stop_73_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_75_ = lean_array_uget_borrowed(v_as_71_, v_i_72_);
v___x_76_ = lean_name_eq(v_a_70_, v___x_75_);
if (v___x_76_ == 0)
{
size_t v___x_77_; size_t v___x_78_; 
v___x_77_ = ((size_t)1ULL);
v___x_78_ = lean_usize_add(v_i_72_, v___x_77_);
v_i_72_ = v___x_78_;
goto _start;
}
else
{
return v___x_76_;
}
}
else
{
uint8_t v___x_80_; 
v___x_80_ = 0;
return v___x_80_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4_spec__6___boxed(lean_object* v_a_81_, lean_object* v_as_82_, lean_object* v_i_83_, lean_object* v_stop_84_){
_start:
{
size_t v_i_boxed_85_; size_t v_stop_boxed_86_; uint8_t v_res_87_; lean_object* v_r_88_; 
v_i_boxed_85_ = lean_unbox_usize(v_i_83_);
lean_dec(v_i_83_);
v_stop_boxed_86_ = lean_unbox_usize(v_stop_84_);
lean_dec(v_stop_84_);
v_res_87_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4_spec__6(v_a_81_, v_as_82_, v_i_boxed_85_, v_stop_boxed_86_);
lean_dec_ref(v_as_82_);
lean_dec(v_a_81_);
v_r_88_ = lean_box(v_res_87_);
return v_r_88_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4(lean_object* v_as_89_, lean_object* v_a_90_){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_91_ = lean_unsigned_to_nat(0u);
v___x_92_ = lean_array_get_size(v_as_89_);
v___x_93_ = lean_nat_dec_lt(v___x_91_, v___x_92_);
if (v___x_93_ == 0)
{
return v___x_93_;
}
else
{
if (v___x_93_ == 0)
{
return v___x_93_;
}
else
{
size_t v___x_94_; size_t v___x_95_; uint8_t v___x_96_; 
v___x_94_ = ((size_t)0ULL);
v___x_95_ = lean_usize_of_nat(v___x_92_);
v___x_96_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4_spec__6(v_a_90_, v_as_89_, v___x_94_, v___x_95_);
return v___x_96_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4___boxed(lean_object* v_as_97_, lean_object* v_a_98_){
_start:
{
uint8_t v_res_99_; lean_object* v_r_100_; 
v_res_99_ = lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4(v_as_97_, v_a_98_);
lean_dec(v_a_98_);
lean_dec_ref(v_as_97_);
v_r_100_ = lean_box(v_res_99_);
return v_r_100_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11(lean_object* v___x_101_, lean_object* v_as_102_, size_t v_i_103_, size_t v_stop_104_, lean_object* v_b_105_){
_start:
{
lean_object* v___y_107_; uint8_t v___x_111_; 
v___x_111_ = lean_usize_dec_eq(v_i_103_, v_stop_104_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; uint8_t v___x_113_; 
v___x_112_ = lean_array_uget_borrowed(v_as_102_, v_i_103_);
v___x_113_ = lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4(v___x_101_, v___x_112_);
if (v___x_113_ == 0)
{
v___y_107_ = v_b_105_;
goto v___jp_106_;
}
else
{
lean_object* v___x_114_; 
lean_inc(v___x_112_);
v___x_114_ = lean_array_push(v_b_105_, v___x_112_);
v___y_107_ = v___x_114_;
goto v___jp_106_;
}
}
else
{
return v_b_105_;
}
v___jp_106_:
{
size_t v___x_108_; size_t v___x_109_; 
v___x_108_ = ((size_t)1ULL);
v___x_109_ = lean_usize_add(v_i_103_, v___x_108_);
v_i_103_ = v___x_109_;
v_b_105_ = v___y_107_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11___boxed(lean_object* v___x_115_, lean_object* v_as_116_, lean_object* v_i_117_, lean_object* v_stop_118_, lean_object* v_b_119_){
_start:
{
size_t v_i_boxed_120_; size_t v_stop_boxed_121_; lean_object* v_res_122_; 
v_i_boxed_120_ = lean_unbox_usize(v_i_117_);
lean_dec(v_i_117_);
v_stop_boxed_121_ = lean_unbox_usize(v_stop_118_);
lean_dec(v_stop_118_);
v_res_122_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11(v___x_115_, v_as_116_, v_i_boxed_120_, v_stop_boxed_121_, v_b_119_);
lean_dec_ref(v_as_116_);
lean_dec_ref(v___x_115_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9(lean_object* v___x_124_, lean_object* v_as_125_, size_t v_i_126_, size_t v_stop_127_, lean_object* v_b_128_){
_start:
{
lean_object* v___y_130_; uint8_t v___x_134_; 
v___x_134_ = lean_usize_dec_eq(v_i_126_, v_stop_127_);
if (v___x_134_ == 0)
{
lean_object* v___x_135_; lean_object* v___x_136_; uint8_t v___x_139_; 
v___x_135_ = ((lean_object*)(lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9___closed__0));
v___x_136_ = lean_array_uget_borrowed(v_as_125_, v_i_126_);
lean_inc(v___x_136_);
lean_inc_ref(v___x_124_);
v___x_139_ = l_Array_contains___redArg(v___x_135_, v___x_124_, v___x_136_);
if (v___x_139_ == 0)
{
goto v___jp_137_;
}
else
{
if (v___x_134_ == 0)
{
v___y_130_ = v_b_128_;
goto v___jp_129_;
}
else
{
goto v___jp_137_;
}
}
v___jp_137_:
{
lean_object* v___x_138_; 
lean_inc(v___x_136_);
v___x_138_ = lean_array_push(v_b_128_, v___x_136_);
v___y_130_ = v___x_138_;
goto v___jp_129_;
}
}
else
{
lean_dec_ref(v___x_124_);
return v_b_128_;
}
v___jp_129_:
{
size_t v___x_131_; size_t v___x_132_; 
v___x_131_ = ((size_t)1ULL);
v___x_132_ = lean_usize_add(v_i_126_, v___x_131_);
v_i_126_ = v___x_132_;
v_b_128_ = v___y_130_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9___boxed(lean_object* v___x_140_, lean_object* v_as_141_, lean_object* v_i_142_, lean_object* v_stop_143_, lean_object* v_b_144_){
_start:
{
size_t v_i_boxed_145_; size_t v_stop_boxed_146_; lean_object* v_res_147_; 
v_i_boxed_145_ = lean_unbox_usize(v_i_142_);
lean_dec(v_i_142_);
v_stop_boxed_146_ = lean_unbox_usize(v_stop_143_);
lean_dec(v_stop_143_);
v_res_147_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9(v___x_140_, v_as_141_, v_i_boxed_145_, v_stop_boxed_146_, v_b_144_);
lean_dec_ref(v_as_141_);
return v_res_147_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_148_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__0, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__0_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__0);
v___x_150_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_150_, 0, v___x_149_);
return v___x_150_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_151_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1);
v___x_152_ = lean_unsigned_to_nat(0u);
v___x_153_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
lean_ctor_set(v___x_153_, 2, v___x_152_);
lean_ctor_set(v___x_153_, 3, v___x_152_);
lean_ctor_set(v___x_153_, 4, v___x_151_);
lean_ctor_set(v___x_153_, 5, v___x_151_);
lean_ctor_set(v___x_153_, 6, v___x_151_);
lean_ctor_set(v___x_153_, 7, v___x_151_);
lean_ctor_set(v___x_153_, 8, v___x_151_);
lean_ctor_set(v___x_153_, 9, v___x_151_);
return v___x_153_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = lean_unsigned_to_nat(32u);
v___x_155_ = lean_mk_empty_array_with_capacity(v___x_154_);
v___x_156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
return v___x_156_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__4(void){
_start:
{
size_t v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_157_ = ((size_t)5ULL);
v___x_158_ = lean_unsigned_to_nat(0u);
v___x_159_ = lean_unsigned_to_nat(32u);
v___x_160_ = lean_mk_empty_array_with_capacity(v___x_159_);
v___x_161_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__3, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__3_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__3);
v___x_162_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_162_, 0, v___x_161_);
lean_ctor_set(v___x_162_, 1, v___x_160_);
lean_ctor_set(v___x_162_, 2, v___x_158_);
lean_ctor_set(v___x_162_, 3, v___x_158_);
lean_ctor_set_usize(v___x_162_, 4, v___x_157_);
return v___x_162_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_163_ = lean_box(1);
v___x_164_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__4, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__4_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__4);
v___x_165_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__1);
v___x_166_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v___x_164_);
lean_ctor_set(v___x_166_, 2, v___x_163_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg(lean_object* v_msgData_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_170_; lean_object* v_env_171_; lean_object* v___x_172_; lean_object* v_scopes_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v_opts_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_170_ = lean_st_ref_get(v___y_168_);
v_env_171_ = lean_ctor_get(v___x_170_, 0);
lean_inc_ref(v_env_171_);
lean_dec(v___x_170_);
v___x_172_ = lean_st_ref_get(v___y_168_);
v_scopes_173_ = lean_ctor_get(v___x_172_, 2);
lean_inc(v_scopes_173_);
lean_dec(v___x_172_);
v___x_174_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_175_ = l_List_head_x21___redArg(v___x_174_, v_scopes_173_);
lean_dec(v_scopes_173_);
v_opts_176_ = lean_ctor_get(v___x_175_, 1);
lean_inc_ref(v_opts_176_);
lean_dec(v___x_175_);
v___x_177_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__2, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__2_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__2);
v___x_178_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__5, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__5_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___closed__5);
v___x_179_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_179_, 0, v_env_171_);
lean_ctor_set(v___x_179_, 1, v___x_177_);
lean_ctor_set(v___x_179_, 2, v___x_178_);
lean_ctor_set(v___x_179_, 3, v_opts_176_);
v___x_180_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
lean_ctor_set(v___x_180_, 1, v_msgData_167_);
v___x_181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg___boxed(lean_object* v_msgData_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg(v_msgData_182_, v___y_183_);
lean_dec(v___y_183_);
return v_res_185_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4(lean_object* v_opts_186_, lean_object* v_opt_187_){
_start:
{
lean_object* v_name_188_; lean_object* v_defValue_189_; lean_object* v_map_190_; lean_object* v___x_191_; 
v_name_188_ = lean_ctor_get(v_opt_187_, 0);
v_defValue_189_ = lean_ctor_get(v_opt_187_, 1);
v_map_190_ = lean_ctor_get(v_opts_186_, 0);
v___x_191_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_190_, v_name_188_);
if (lean_obj_tag(v___x_191_) == 0)
{
uint8_t v___x_192_; 
v___x_192_ = lean_unbox(v_defValue_189_);
return v___x_192_;
}
else
{
lean_object* v_val_193_; 
v_val_193_ = lean_ctor_get(v___x_191_, 0);
lean_inc(v_val_193_);
lean_dec_ref_known(v___x_191_, 1);
if (lean_obj_tag(v_val_193_) == 1)
{
uint8_t v_v_194_; 
v_v_194_ = lean_ctor_get_uint8(v_val_193_, 0);
lean_dec_ref_known(v_val_193_, 0);
return v_v_194_;
}
else
{
uint8_t v___x_195_; 
lean_dec(v_val_193_);
v___x_195_ = lean_unbox(v_defValue_189_);
return v___x_195_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4___boxed(lean_object* v_opts_196_, lean_object* v_opt_197_){
_start:
{
uint8_t v_res_198_; lean_object* v_r_199_; 
v_res_198_ = lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4(v_opts_196_, v_opt_197_);
lean_dec_ref(v_opt_197_);
lean_dec_ref(v_opts_196_);
v_r_199_ = lean_box(v_res_198_);
return v_r_199_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0(uint8_t v___y_201_, uint8_t v_suppressElabErrors_202_, lean_object* v_x_203_){
_start:
{
if (lean_obj_tag(v_x_203_) == 1)
{
lean_object* v_pre_204_; 
v_pre_204_ = lean_ctor_get(v_x_203_, 0);
if (lean_obj_tag(v_pre_204_) == 0)
{
lean_object* v_str_205_; lean_object* v___x_206_; uint8_t v___x_207_; 
v_str_205_ = lean_ctor_get(v_x_203_, 1);
v___x_206_ = ((lean_object*)(lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___closed__0));
v___x_207_ = lean_string_dec_eq(v_str_205_, v___x_206_);
if (v___x_207_ == 0)
{
return v___y_201_;
}
else
{
return v_suppressElabErrors_202_;
}
}
else
{
return v___y_201_;
}
}
else
{
return v___y_201_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___boxed(lean_object* v___y_208_, lean_object* v_suppressElabErrors_209_, lean_object* v_x_210_){
_start:
{
uint8_t v___y_10027__boxed_211_; uint8_t v_suppressElabErrors_boxed_212_; uint8_t v_res_213_; lean_object* v_r_214_; 
v___y_10027__boxed_211_ = lean_unbox(v___y_208_);
v_suppressElabErrors_boxed_212_ = lean_unbox(v_suppressElabErrors_209_);
v_res_213_ = lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0(v___y_10027__boxed_211_, v_suppressElabErrors_boxed_212_, v_x_210_);
lean_dec(v_x_210_);
v_r_214_ = lean_box(v_res_213_);
return v_r_214_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12(lean_object* v_ref_216_, lean_object* v_msgData_217_, uint8_t v_severity_218_, uint8_t v_isSilent_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v___y_224_; uint8_t v___y_225_; lean_object* v___y_226_; uint8_t v___y_227_; lean_object* v___y_228_; lean_object* v___y_229_; lean_object* v___y_230_; lean_object* v___y_231_; uint8_t v___y_288_; uint8_t v___y_289_; uint8_t v___y_290_; lean_object* v___y_291_; lean_object* v___y_292_; uint8_t v___y_316_; uint8_t v___y_317_; uint8_t v___y_318_; lean_object* v___y_319_; lean_object* v___y_320_; uint8_t v___y_324_; uint8_t v___y_325_; uint8_t v___y_326_; uint8_t v___x_341_; uint8_t v___y_343_; uint8_t v___y_344_; uint8_t v___y_345_; uint8_t v___y_347_; uint8_t v___x_359_; 
v___x_341_ = 2;
v___x_359_ = l_Lean_instBEqMessageSeverity_beq(v_severity_218_, v___x_341_);
if (v___x_359_ == 0)
{
v___y_347_ = v___x_359_;
goto v___jp_346_;
}
else
{
uint8_t v___x_360_; 
lean_inc_ref(v_msgData_217_);
v___x_360_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_217_);
v___y_347_ = v___x_360_;
goto v___jp_346_;
}
v___jp_223_:
{
lean_object* v___x_232_; 
v___x_232_ = l_Lean_Elab_Command_getScope___redArg(v___y_231_);
if (lean_obj_tag(v___x_232_) == 0)
{
lean_object* v_a_233_; lean_object* v___x_234_; 
v_a_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_a_233_);
lean_dec_ref_known(v___x_232_, 1);
v___x_234_ = l_Lean_Elab_Command_getScope___redArg(v___y_231_);
if (lean_obj_tag(v___x_234_) == 0)
{
lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_270_; 
v_a_235_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_270_ == 0)
{
v___x_237_ = v___x_234_;
v_isShared_238_ = v_isSharedCheck_270_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_dec(v___x_234_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_270_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_239_; lean_object* v_currNamespace_240_; lean_object* v_openDecls_241_; lean_object* v_env_242_; lean_object* v_messages_243_; lean_object* v_scopes_244_; lean_object* v_usedQuotCtxts_245_; lean_object* v_nextMacroScope_246_; lean_object* v_maxRecDepth_247_; lean_object* v_ngen_248_; lean_object* v_auxDeclNGen_249_; lean_object* v_infoState_250_; lean_object* v_traceState_251_; lean_object* v_snapshotTasks_252_; lean_object* v_prevLinterStates_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_269_; 
v___x_239_ = lean_st_ref_take(v___y_231_);
v_currNamespace_240_ = lean_ctor_get(v_a_233_, 2);
lean_inc(v_currNamespace_240_);
lean_dec(v_a_233_);
v_openDecls_241_ = lean_ctor_get(v_a_235_, 3);
lean_inc(v_openDecls_241_);
lean_dec(v_a_235_);
v_env_242_ = lean_ctor_get(v___x_239_, 0);
v_messages_243_ = lean_ctor_get(v___x_239_, 1);
v_scopes_244_ = lean_ctor_get(v___x_239_, 2);
v_usedQuotCtxts_245_ = lean_ctor_get(v___x_239_, 3);
v_nextMacroScope_246_ = lean_ctor_get(v___x_239_, 4);
v_maxRecDepth_247_ = lean_ctor_get(v___x_239_, 5);
v_ngen_248_ = lean_ctor_get(v___x_239_, 6);
v_auxDeclNGen_249_ = lean_ctor_get(v___x_239_, 7);
v_infoState_250_ = lean_ctor_get(v___x_239_, 8);
v_traceState_251_ = lean_ctor_get(v___x_239_, 9);
v_snapshotTasks_252_ = lean_ctor_get(v___x_239_, 10);
v_prevLinterStates_253_ = lean_ctor_get(v___x_239_, 11);
v_isSharedCheck_269_ = !lean_is_exclusive(v___x_239_);
if (v_isSharedCheck_269_ == 0)
{
v___x_255_ = v___x_239_;
v_isShared_256_ = v_isSharedCheck_269_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_prevLinterStates_253_);
lean_inc(v_snapshotTasks_252_);
lean_inc(v_traceState_251_);
lean_inc(v_infoState_250_);
lean_inc(v_auxDeclNGen_249_);
lean_inc(v_ngen_248_);
lean_inc(v_maxRecDepth_247_);
lean_inc(v_nextMacroScope_246_);
lean_inc(v_usedQuotCtxts_245_);
lean_inc(v_scopes_244_);
lean_inc(v_messages_243_);
lean_inc(v_env_242_);
lean_dec(v___x_239_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_269_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_262_; 
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v_currNamespace_240_);
lean_ctor_set(v___x_257_, 1, v_openDecls_241_);
v___x_258_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v___y_224_);
lean_inc_ref(v___y_226_);
lean_inc_ref(v___y_228_);
v___x_259_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_259_, 0, v___y_228_);
lean_ctor_set(v___x_259_, 1, v___y_230_);
lean_ctor_set(v___x_259_, 2, v___y_229_);
lean_ctor_set(v___x_259_, 3, v___y_226_);
lean_ctor_set(v___x_259_, 4, v___x_258_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*5, v___y_225_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*5 + 1, v___y_227_);
lean_ctor_set_uint8(v___x_259_, sizeof(void*)*5 + 2, v_isSilent_219_);
v___x_260_ = l_Lean_MessageLog_add(v___x_259_, v_messages_243_);
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 1, v___x_260_);
v___x_262_ = v___x_255_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v_env_242_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v___x_260_);
lean_ctor_set(v_reuseFailAlloc_268_, 2, v_scopes_244_);
lean_ctor_set(v_reuseFailAlloc_268_, 3, v_usedQuotCtxts_245_);
lean_ctor_set(v_reuseFailAlloc_268_, 4, v_nextMacroScope_246_);
lean_ctor_set(v_reuseFailAlloc_268_, 5, v_maxRecDepth_247_);
lean_ctor_set(v_reuseFailAlloc_268_, 6, v_ngen_248_);
lean_ctor_set(v_reuseFailAlloc_268_, 7, v_auxDeclNGen_249_);
lean_ctor_set(v_reuseFailAlloc_268_, 8, v_infoState_250_);
lean_ctor_set(v_reuseFailAlloc_268_, 9, v_traceState_251_);
lean_ctor_set(v_reuseFailAlloc_268_, 10, v_snapshotTasks_252_);
lean_ctor_set(v_reuseFailAlloc_268_, 11, v_prevLinterStates_253_);
v___x_262_ = v_reuseFailAlloc_268_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_266_; 
v___x_263_ = lean_st_ref_set(v___y_231_, v___x_262_);
v___x_264_ = lean_box(0);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 0, v___x_264_);
v___x_266_ = v___x_237_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v___x_264_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
return v___x_266_;
}
}
}
}
}
else
{
lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_278_; 
lean_dec(v_a_233_);
lean_dec_ref(v___y_230_);
lean_dec(v___y_229_);
lean_dec_ref(v___y_224_);
v_a_271_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_278_ == 0)
{
v___x_273_ = v___x_234_;
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_234_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_276_; 
if (v_isShared_274_ == 0)
{
v___x_276_ = v___x_273_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v_a_271_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
}
else
{
lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_286_; 
lean_dec_ref(v___y_230_);
lean_dec(v___y_229_);
lean_dec_ref(v___y_224_);
v_a_279_ = lean_ctor_get(v___x_232_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v___x_232_);
if (v_isSharedCheck_286_ == 0)
{
v___x_281_ = v___x_232_;
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_232_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_284_; 
if (v_isShared_282_ == 0)
{
v___x_284_ = v___x_281_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_a_279_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
}
}
v___jp_287_:
{
lean_object* v_fileName_293_; lean_object* v_fileMap_294_; uint8_t v_suppressElabErrors_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_314_; 
v_fileName_293_ = lean_ctor_get(v___y_220_, 0);
v_fileMap_294_ = lean_ctor_get(v___y_220_, 1);
v_suppressElabErrors_295_ = lean_ctor_get_uint8(v___y_220_, sizeof(void*)*10);
v___x_296_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_217_);
v___x_297_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg(v___x_296_, v___y_221_);
v_a_298_ = lean_ctor_get(v___x_297_, 0);
v_isSharedCheck_314_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_314_ == 0)
{
v___x_300_ = v___x_297_;
v_isShared_301_ = v_isSharedCheck_314_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_297_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_314_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
lean_inc_ref_n(v_fileMap_294_, 2);
v___x_302_ = l_Lean_FileMap_toPosition(v_fileMap_294_, v___y_291_);
lean_dec(v___y_291_);
v___x_303_ = l_Lean_FileMap_toPosition(v_fileMap_294_, v___y_292_);
lean_dec(v___y_292_);
v___x_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_304_, 0, v___x_303_);
v___x_305_ = ((lean_object*)(lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___closed__0));
if (v_suppressElabErrors_295_ == 0)
{
lean_del_object(v___x_300_);
v___y_224_ = v_a_298_;
v___y_225_ = v___y_289_;
v___y_226_ = v___x_305_;
v___y_227_ = v___y_290_;
v___y_228_ = v_fileName_293_;
v___y_229_ = v___x_304_;
v___y_230_ = v___x_302_;
v___y_231_ = v___y_221_;
goto v___jp_223_;
}
else
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___f_308_; uint8_t v___x_309_; 
v___x_306_ = lean_box(v___y_288_);
v___x_307_ = lean_box(v_suppressElabErrors_295_);
v___f_308_ = lean_alloc_closure((void*)(lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___lam__0___boxed), 3, 2);
lean_closure_set(v___f_308_, 0, v___x_306_);
lean_closure_set(v___f_308_, 1, v___x_307_);
lean_inc(v_a_298_);
v___x_309_ = l_Lean_MessageData_hasTag(v___f_308_, v_a_298_);
if (v___x_309_ == 0)
{
lean_object* v___x_310_; lean_object* v___x_312_; 
lean_dec_ref_known(v___x_304_, 1);
lean_dec_ref(v___x_302_);
lean_dec(v_a_298_);
v___x_310_ = lean_box(0);
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v___x_310_);
v___x_312_ = v___x_300_;
goto v_reusejp_311_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_310_);
v___x_312_ = v_reuseFailAlloc_313_;
goto v_reusejp_311_;
}
v_reusejp_311_:
{
return v___x_312_;
}
}
else
{
lean_del_object(v___x_300_);
v___y_224_ = v_a_298_;
v___y_225_ = v___y_289_;
v___y_226_ = v___x_305_;
v___y_227_ = v___y_290_;
v___y_228_ = v_fileName_293_;
v___y_229_ = v___x_304_;
v___y_230_ = v___x_302_;
v___y_231_ = v___y_221_;
goto v___jp_223_;
}
}
}
}
v___jp_315_:
{
lean_object* v___x_321_; 
v___x_321_ = l_Lean_Syntax_getTailPos_x3f(v___y_319_, v___y_317_);
lean_dec(v___y_319_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_inc(v___y_320_);
v___y_288_ = v___y_316_;
v___y_289_ = v___y_317_;
v___y_290_ = v___y_318_;
v___y_291_ = v___y_320_;
v___y_292_ = v___y_320_;
goto v___jp_287_;
}
else
{
lean_object* v_val_322_; 
v_val_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc(v_val_322_);
lean_dec_ref_known(v___x_321_, 1);
v___y_288_ = v___y_316_;
v___y_289_ = v___y_317_;
v___y_290_ = v___y_318_;
v___y_291_ = v___y_320_;
v___y_292_ = v_val_322_;
goto v___jp_287_;
}
}
v___jp_323_:
{
lean_object* v___x_327_; 
v___x_327_ = l_Lean_Elab_Command_getRef___redArg(v___y_220_);
if (lean_obj_tag(v___x_327_) == 0)
{
lean_object* v_a_328_; lean_object* v_ref_329_; lean_object* v___x_330_; 
v_a_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_a_328_);
lean_dec_ref_known(v___x_327_, 1);
v_ref_329_ = l_Lean_replaceRef(v_ref_216_, v_a_328_);
lean_dec(v_a_328_);
v___x_330_ = l_Lean_Syntax_getPos_x3f(v_ref_329_, v___y_325_);
if (lean_obj_tag(v___x_330_) == 0)
{
lean_object* v___x_331_; 
v___x_331_ = lean_unsigned_to_nat(0u);
v___y_316_ = v___y_324_;
v___y_317_ = v___y_325_;
v___y_318_ = v___y_326_;
v___y_319_ = v_ref_329_;
v___y_320_ = v___x_331_;
goto v___jp_315_;
}
else
{
lean_object* v_val_332_; 
v_val_332_ = lean_ctor_get(v___x_330_, 0);
lean_inc(v_val_332_);
lean_dec_ref_known(v___x_330_, 1);
v___y_316_ = v___y_324_;
v___y_317_ = v___y_325_;
v___y_318_ = v___y_326_;
v___y_319_ = v_ref_329_;
v___y_320_ = v_val_332_;
goto v___jp_315_;
}
}
else
{
lean_object* v_a_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_340_; 
lean_dec_ref(v_msgData_217_);
v_a_333_ = lean_ctor_get(v___x_327_, 0);
v_isSharedCheck_340_ = !lean_is_exclusive(v___x_327_);
if (v_isSharedCheck_340_ == 0)
{
v___x_335_ = v___x_327_;
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_a_333_);
lean_dec(v___x_327_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_338_; 
if (v_isShared_336_ == 0)
{
v___x_338_ = v___x_335_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v_a_333_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
}
v___jp_342_:
{
if (v___y_345_ == 0)
{
v___y_324_ = v___y_343_;
v___y_325_ = v___y_344_;
v___y_326_ = v_severity_218_;
goto v___jp_323_;
}
else
{
v___y_324_ = v___y_343_;
v___y_325_ = v___y_344_;
v___y_326_ = v___x_341_;
goto v___jp_323_;
}
}
v___jp_346_:
{
if (v___y_347_ == 0)
{
lean_object* v___x_348_; lean_object* v_scopes_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v_opts_352_; uint8_t v___x_353_; uint8_t v___x_354_; 
v___x_348_ = lean_st_ref_get(v___y_221_);
v_scopes_349_ = lean_ctor_get(v___x_348_, 2);
lean_inc(v_scopes_349_);
lean_dec(v___x_348_);
v___x_350_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_351_ = l_List_head_x21___redArg(v___x_350_, v_scopes_349_);
lean_dec(v_scopes_349_);
v_opts_352_ = lean_ctor_get(v___x_351_, 1);
lean_inc_ref(v_opts_352_);
lean_dec(v___x_351_);
v___x_353_ = 1;
v___x_354_ = l_Lean_instBEqMessageSeverity_beq(v_severity_218_, v___x_353_);
if (v___x_354_ == 0)
{
lean_dec_ref(v_opts_352_);
v___y_343_ = v___y_347_;
v___y_344_ = v___y_347_;
v___y_345_ = v___x_354_;
goto v___jp_342_;
}
else
{
lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_355_ = l_Lean_warningAsError;
v___x_356_ = lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4(v_opts_352_, v___x_355_);
lean_dec_ref(v_opts_352_);
v___y_343_ = v___y_347_;
v___y_344_ = v___y_347_;
v___y_345_ = v___x_356_;
goto v___jp_342_;
}
}
else
{
lean_object* v___x_357_; lean_object* v___x_358_; 
lean_dec_ref(v_msgData_217_);
v___x_357_ = lean_box(0);
v___x_358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
return v___x_358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12___boxed(lean_object* v_ref_361_, lean_object* v_msgData_362_, lean_object* v_severity_363_, lean_object* v_isSilent_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
uint8_t v_severity_boxed_368_; uint8_t v_isSilent_boxed_369_; lean_object* v_res_370_; 
v_severity_boxed_368_ = lean_unbox(v_severity_363_);
v_isSilent_boxed_369_ = lean_unbox(v_isSilent_364_);
v_res_370_ = lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12(v_ref_361_, v_msgData_362_, v_severity_boxed_368_, v_isSilent_boxed_369_, v___y_365_, v___y_366_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
lean_dec(v_ref_361_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9(lean_object* v_msgData_371_, uint8_t v_severity_372_, uint8_t v_isSilent_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = l_Lean_Elab_Command_getRef___redArg(v___y_374_);
if (lean_obj_tag(v___x_377_) == 0)
{
lean_object* v_a_378_; lean_object* v___x_379_; 
v_a_378_ = lean_ctor_get(v___x_377_, 0);
lean_inc(v_a_378_);
lean_dec_ref_known(v___x_377_, 1);
v___x_379_ = lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9_spec__12(v_a_378_, v_msgData_371_, v_severity_372_, v_isSilent_373_, v___y_374_, v___y_375_);
lean_dec(v_a_378_);
return v___x_379_;
}
else
{
lean_object* v_a_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_387_; 
lean_dec_ref(v_msgData_371_);
v_a_380_ = lean_ctor_get(v___x_377_, 0);
v_isSharedCheck_387_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_387_ == 0)
{
v___x_382_ = v___x_377_;
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_a_380_);
lean_dec(v___x_377_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_385_; 
if (v_isShared_383_ == 0)
{
v___x_385_ = v___x_382_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_a_380_);
v___x_385_ = v_reuseFailAlloc_386_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
return v___x_385_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9___boxed(lean_object* v_msgData_388_, lean_object* v_severity_389_, lean_object* v_isSilent_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
uint8_t v_severity_boxed_394_; uint8_t v_isSilent_boxed_395_; lean_object* v_res_396_; 
v_severity_boxed_394_ = lean_unbox(v_severity_389_);
v_isSilent_boxed_395_ = lean_unbox(v_isSilent_390_);
v_res_396_ = lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9(v_msgData_388_, v_severity_boxed_394_, v_isSilent_boxed_395_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
return v_res_396_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6(lean_object* v_msgData_397_, lean_object* v___y_398_, lean_object* v___y_399_){
_start:
{
uint8_t v___x_401_; uint8_t v___x_402_; lean_object* v___x_403_; 
v___x_401_ = 0;
v___x_402_ = 0;
v___x_403_ = lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6_spec__9(v_msgData_397_, v___x_401_, v___x_402_, v___y_398_, v___y_399_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6___boxed(lean_object* v_msgData_404_, lean_object* v___y_405_, lean_object* v___y_406_, lean_object* v___y_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6(v_msgData_404_, v___y_405_, v___y_406_);
lean_dec(v___y_406_);
lean_dec_ref(v___y_405_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7(uint8_t v___x_409_, size_t v_sz_410_, size_t v_i_411_, lean_object* v_bs_412_){
_start:
{
uint8_t v___x_413_; 
v___x_413_ = lean_usize_dec_lt(v_i_411_, v_sz_410_);
if (v___x_413_ == 0)
{
return v_bs_412_;
}
else
{
lean_object* v_v_414_; lean_object* v___x_415_; lean_object* v_bs_x27_416_; lean_object* v___x_417_; size_t v___x_418_; size_t v___x_419_; lean_object* v___x_420_; 
v_v_414_ = lean_array_uget(v_bs_412_, v_i_411_);
v___x_415_ = lean_unsigned_to_nat(0u);
v_bs_x27_416_ = lean_array_uset(v_bs_412_, v_i_411_, v___x_415_);
v___x_417_ = l_Lean_Name_toString(v_v_414_, v___x_409_);
v___x_418_ = ((size_t)1ULL);
v___x_419_ = lean_usize_add(v_i_411_, v___x_418_);
v___x_420_ = lean_array_uset(v_bs_x27_416_, v_i_411_, v___x_417_);
v_i_411_ = v___x_419_;
v_bs_412_ = v___x_420_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7___boxed(lean_object* v___x_422_, lean_object* v_sz_423_, lean_object* v_i_424_, lean_object* v_bs_425_){
_start:
{
uint8_t v___x_10341__boxed_426_; size_t v_sz_boxed_427_; size_t v_i_boxed_428_; lean_object* v_res_429_; 
v___x_10341__boxed_426_ = lean_unbox(v___x_422_);
v_sz_boxed_427_ = lean_unbox_usize(v_sz_423_);
lean_dec(v_sz_423_);
v_i_boxed_428_ = lean_unbox_usize(v_i_424_);
lean_dec(v_i_424_);
v_res_429_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7(v___x_10341__boxed_426_, v_sz_boxed_427_, v_i_boxed_428_, v_bs_425_);
return v_res_429_;
}
}
static lean_object* _init_lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0(void){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_430_ = lean_box(1);
v___x_431_ = l_Lean_MessageData_ofFormat(v___x_430_);
return v___x_431_;
}
}
static lean_object* _init_lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__3(void){
_start:
{
lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_435_ = ((lean_object*)(lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__2));
v___x_436_ = l_Lean_MessageData_ofFormat(v___x_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5(lean_object* v_x_437_, lean_object* v_x_438_){
_start:
{
if (lean_obj_tag(v_x_438_) == 0)
{
return v_x_437_;
}
else
{
lean_object* v_head_439_; lean_object* v_tail_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_462_; 
v_head_439_ = lean_ctor_get(v_x_438_, 0);
v_tail_440_ = lean_ctor_get(v_x_438_, 1);
v_isSharedCheck_462_ = !lean_is_exclusive(v_x_438_);
if (v_isSharedCheck_462_ == 0)
{
v___x_442_ = v_x_438_;
v_isShared_443_ = v_isSharedCheck_462_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_tail_440_);
lean_inc(v_head_439_);
lean_dec(v_x_438_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_462_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v_before_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_460_; 
v_before_444_ = lean_ctor_get(v_head_439_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v_head_439_);
if (v_isSharedCheck_460_ == 0)
{
lean_object* v_unused_461_; 
v_unused_461_ = lean_ctor_get(v_head_439_, 1);
lean_dec(v_unused_461_);
v___x_446_ = v_head_439_;
v_isShared_447_ = v_isSharedCheck_460_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_before_444_);
lean_dec(v_head_439_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_460_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
lean_object* v___x_448_; lean_object* v___x_450_; 
v___x_448_ = lean_obj_once(&lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0, &lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0_once, _init_lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0);
if (v_isShared_447_ == 0)
{
lean_ctor_set_tag(v___x_446_, 7);
lean_ctor_set(v___x_446_, 1, v___x_448_);
lean_ctor_set(v___x_446_, 0, v_x_437_);
v___x_450_ = v___x_446_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_x_437_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v___x_448_);
v___x_450_ = v_reuseFailAlloc_459_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
lean_object* v___x_451_; lean_object* v___x_453_; 
v___x_451_ = lean_obj_once(&lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__3, &lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__3_once, _init_lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__3);
if (v_isShared_443_ == 0)
{
lean_ctor_set_tag(v___x_442_, 7);
lean_ctor_set(v___x_442_, 1, v___x_451_);
lean_ctor_set(v___x_442_, 0, v___x_450_);
v___x_453_ = v___x_442_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_450_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v___x_451_);
v___x_453_ = v_reuseFailAlloc_458_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_454_ = l_Lean_MessageData_ofSyntax(v_before_444_);
v___x_455_ = l_Lean_indentD(v___x_454_);
v___x_456_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_453_);
lean_ctor_set(v___x_456_, 1, v___x_455_);
v_x_437_ = v___x_456_;
v_x_438_ = v_tail_440_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_466_ = ((lean_object*)(lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__1));
v___x_467_ = l_Lean_MessageData_ofFormat(v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg(lean_object* v_msgData_468_, lean_object* v_macroStack_469_, lean_object* v___y_470_){
_start:
{
lean_object* v___x_472_; lean_object* v_scopes_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v_opts_476_; lean_object* v___x_477_; uint8_t v___x_478_; 
v___x_472_ = lean_st_ref_get(v___y_470_);
v_scopes_473_ = lean_ctor_get(v___x_472_, 2);
lean_inc(v_scopes_473_);
lean_dec(v___x_472_);
v___x_474_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_475_ = l_List_head_x21___redArg(v___x_474_, v_scopes_473_);
lean_dec(v_scopes_473_);
v_opts_476_ = lean_ctor_get(v___x_475_, 1);
lean_inc_ref(v_opts_476_);
lean_dec(v___x_475_);
v___x_477_ = l_Lean_Elab_pp_macroStack;
v___x_478_ = lp_importGraph_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__4(v_opts_476_, v___x_477_);
lean_dec_ref(v_opts_476_);
if (v___x_478_ == 0)
{
lean_object* v___x_479_; 
lean_dec(v_macroStack_469_);
v___x_479_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_479_, 0, v_msgData_468_);
return v___x_479_;
}
else
{
if (lean_obj_tag(v_macroStack_469_) == 0)
{
lean_object* v___x_480_; 
v___x_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_480_, 0, v_msgData_468_);
return v___x_480_;
}
else
{
lean_object* v_head_481_; lean_object* v_after_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_497_; 
v_head_481_ = lean_ctor_get(v_macroStack_469_, 0);
lean_inc(v_head_481_);
v_after_482_ = lean_ctor_get(v_head_481_, 1);
v_isSharedCheck_497_ = !lean_is_exclusive(v_head_481_);
if (v_isSharedCheck_497_ == 0)
{
lean_object* v_unused_498_; 
v_unused_498_ = lean_ctor_get(v_head_481_, 0);
lean_dec(v_unused_498_);
v___x_484_ = v_head_481_;
v_isShared_485_ = v_isSharedCheck_497_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_after_482_);
lean_dec(v_head_481_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_497_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; lean_object* v___x_488_; 
v___x_486_ = lean_obj_once(&lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0, &lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0_once, _init_lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5___closed__0);
if (v_isShared_485_ == 0)
{
lean_ctor_set_tag(v___x_484_, 7);
lean_ctor_set(v___x_484_, 1, v___x_486_);
lean_ctor_set(v___x_484_, 0, v_msgData_468_);
v___x_488_ = v___x_484_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_msgData_468_);
lean_ctor_set(v_reuseFailAlloc_496_, 1, v___x_486_);
v___x_488_ = v_reuseFailAlloc_496_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v_msgData_493_; lean_object* v___x_494_; lean_object* v___x_495_; 
v___x_489_ = lean_obj_once(&lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__2, &lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__2_once, _init_lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___closed__2);
v___x_490_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_488_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = l_Lean_MessageData_ofSyntax(v_after_482_);
v___x_492_ = l_Lean_indentD(v___x_491_);
v_msgData_493_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_493_, 0, v___x_490_);
lean_ctor_set(v_msgData_493_, 1, v___x_492_);
v___x_494_ = lp_importGraph_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3_spec__5(v_msgData_493_, v_macroStack_469_);
v___x_495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_495_, 0, v___x_494_);
return v___x_495_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_499_, lean_object* v_macroStack_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg(v_msgData_499_, v_macroStack_500_, v___y_501_);
lean_dec(v___y_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg(lean_object* v_msg_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = l_Lean_Elab_Command_getRef___redArg(v___y_505_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v_a_509_; lean_object* v_macroStack_510_; lean_object* v___x_511_; lean_object* v_a_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v_a_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_523_; 
v_a_509_ = lean_ctor_get(v___x_508_, 0);
lean_inc(v_a_509_);
lean_dec_ref_known(v___x_508_, 1);
v_macroStack_510_ = lean_ctor_get(v___y_505_, 4);
v___x_511_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg(v_msg_504_, v___y_506_);
v_a_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_a_512_);
lean_dec_ref(v___x_511_);
v___x_513_ = l_Lean_Elab_getBetterRef(v_a_509_, v_macroStack_510_);
lean_dec(v_a_509_);
lean_inc(v_macroStack_510_);
v___x_514_ = lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg(v_a_512_, v_macroStack_510_, v___y_506_);
v_a_515_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_523_ == 0)
{
v___x_517_ = v___x_514_;
v_isShared_518_ = v_isSharedCheck_523_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_a_515_);
lean_dec(v___x_514_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_523_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v___x_519_; lean_object* v___x_521_; 
v___x_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_519_, 0, v___x_513_);
lean_ctor_set(v___x_519_, 1, v_a_515_);
if (v_isShared_518_ == 0)
{
lean_ctor_set_tag(v___x_517_, 1);
lean_ctor_set(v___x_517_, 0, v___x_519_);
v___x_521_ = v___x_517_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_519_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
else
{
lean_object* v_a_524_; lean_object* v___x_526_; uint8_t v_isShared_527_; uint8_t v_isSharedCheck_531_; 
lean_dec_ref(v_msg_504_);
v_a_524_ = lean_ctor_get(v___x_508_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_531_ == 0)
{
v___x_526_ = v___x_508_;
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
else
{
lean_inc(v_a_524_);
lean_dec(v___x_508_);
v___x_526_ = lean_box(0);
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
v_resetjp_525_:
{
lean_object* v___x_529_; 
if (v_isShared_527_ == 0)
{
v___x_529_ = v___x_526_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_a_524_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg___boxed(lean_object* v_msg_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_){
_start:
{
lean_object* v_res_536_; 
v_res_536_ = lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg(v_msg_532_, v___y_533_, v___y_534_);
lean_dec(v___y_534_);
lean_dec_ref(v___y_533_);
return v_res_536_;
}
}
static lean_object* _init_lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__2(void){
_start:
{
lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_539_ = ((lean_object*)(lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__1));
v___x_540_ = l_Lean_stringToMessageData(v___x_539_);
return v___x_540_;
}
}
static lean_object* _init_lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__4(void){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; 
v___x_542_ = ((lean_object*)(lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__3));
v___x_543_ = l_Lean_stringToMessageData(v___x_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3(lean_object* v_val_544_, uint8_t v___x_545_, lean_object* v_as_546_, size_t v_sz_547_, size_t v_i_548_, lean_object* v_b_549_, lean_object* v___y_550_, lean_object* v___y_551_){
_start:
{
lean_object* v_a_554_; uint8_t v___x_558_; 
v___x_558_ = lean_usize_dec_lt(v_i_548_, v_sz_547_);
if (v___x_558_ == 0)
{
lean_object* v___x_559_; 
lean_dec(v_val_544_);
v___x_559_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_559_, 0, v_b_549_);
return v___x_559_;
}
else
{
lean_object* v_a_560_; lean_object* v___x_561_; lean_object* v___x_562_; 
v_a_560_ = lean_array_uget_borrowed(v_as_546_, v_i_548_);
v___x_561_ = ((lean_object*)(lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__0));
lean_inc(v_a_560_);
lean_inc(v_val_544_);
v___x_562_ = l_Lean_SearchPath_findWithExt(v_val_544_, v___x_561_, v_a_560_);
if (lean_obj_tag(v___x_562_) == 0)
{
lean_object* v_a_563_; lean_object* v___x_564_; 
v_a_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc(v_a_563_);
lean_dec_ref_known(v___x_562_, 1);
v___x_564_ = lean_box(0);
if (lean_obj_tag(v_a_563_) == 0)
{
goto v___jp_565_;
}
else
{
lean_dec_ref_known(v_a_563_, 1);
if (v___x_545_ == 0)
{
goto v___jp_565_;
}
else
{
v_a_554_ = v___x_564_;
goto v___jp_553_;
}
}
v___jp_565_:
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; 
v___x_566_ = lean_obj_once(&lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__2, &lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__2_once, _init_lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__2);
lean_inc(v_a_560_);
v___x_567_ = l_Lean_MessageData_ofName(v_a_560_);
v___x_568_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_568_, 0, v___x_566_);
lean_ctor_set(v___x_568_, 1, v___x_567_);
v___x_569_ = lean_obj_once(&lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__4, &lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__4_once, _init_lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___closed__4);
v___x_570_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_568_);
lean_ctor_set(v___x_570_, 1, v___x_569_);
v___x_571_ = lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg(v___x_570_, v___y_550_, v___y_551_);
if (lean_obj_tag(v___x_571_) == 0)
{
lean_dec_ref_known(v___x_571_, 1);
v_a_554_ = v___x_564_;
goto v___jp_553_;
}
else
{
lean_dec(v_val_544_);
return v___x_571_;
}
}
}
else
{
lean_object* v_a_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_584_; 
lean_dec(v_val_544_);
v_a_572_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_584_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_584_ == 0)
{
v___x_574_ = v___x_562_;
v_isShared_575_ = v_isSharedCheck_584_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_a_572_);
lean_dec(v___x_562_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_584_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v_ref_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_582_; 
v_ref_576_ = lean_ctor_get(v___y_550_, 7);
v___x_577_ = lean_io_error_to_string(v_a_572_);
v___x_578_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_578_, 0, v___x_577_);
v___x_579_ = l_Lean_MessageData_ofFormat(v___x_578_);
lean_inc(v_ref_576_);
v___x_580_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_580_, 0, v_ref_576_);
lean_ctor_set(v___x_580_, 1, v___x_579_);
if (v_isShared_575_ == 0)
{
lean_ctor_set(v___x_574_, 0, v___x_580_);
v___x_582_ = v___x_574_;
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
v___jp_553_:
{
size_t v___x_555_; size_t v___x_556_; 
v___x_555_ = ((size_t)1ULL);
v___x_556_ = lean_usize_add(v_i_548_, v___x_555_);
v_i_548_ = v___x_556_;
v_b_549_ = v_a_554_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3___boxed(lean_object* v_val_585_, lean_object* v___x_586_, lean_object* v_as_587_, lean_object* v_sz_588_, lean_object* v_i_589_, lean_object* v_b_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_){
_start:
{
uint8_t v___x_10570__boxed_594_; size_t v_sz_boxed_595_; size_t v_i_boxed_596_; lean_object* v_res_597_; 
v___x_10570__boxed_594_ = lean_unbox(v___x_586_);
v_sz_boxed_595_ = lean_unbox_usize(v_sz_588_);
lean_dec(v_sz_588_);
v_i_boxed_596_ = lean_unbox_usize(v_i_589_);
lean_dec(v_i_589_);
v_res_597_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3(v_val_585_, v___x_10570__boxed_594_, v_as_587_, v_sz_boxed_595_, v_i_boxed_596_, v_b_590_, v___y_591_, v___y_592_);
lean_dec(v___y_592_);
lean_dec_ref(v___y_591_);
lean_dec_ref(v_as_587_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__1(size_t v_sz_598_, size_t v_i_599_, lean_object* v_bs_600_){
_start:
{
uint8_t v___x_601_; 
v___x_601_ = lean_usize_dec_lt(v_i_599_, v_sz_598_);
if (v___x_601_ == 0)
{
return v_bs_600_;
}
else
{
lean_object* v_v_602_; lean_object* v___x_603_; lean_object* v_bs_x27_604_; lean_object* v___x_605_; size_t v___x_606_; size_t v___x_607_; lean_object* v___x_608_; 
v_v_602_ = lean_array_uget(v_bs_600_, v_i_599_);
v___x_603_ = lean_unsigned_to_nat(0u);
v_bs_x27_604_ = lean_array_uset(v_bs_600_, v_i_599_, v___x_603_);
v___x_605_ = l_Lean_TSyntax_getId(v_v_602_);
lean_dec(v_v_602_);
v___x_606_ = ((size_t)1ULL);
v___x_607_ = lean_usize_add(v_i_599_, v___x_606_);
v___x_608_ = lean_array_uset(v_bs_x27_604_, v_i_599_, v___x_605_);
v_i_599_ = v___x_607_;
v_bs_600_ = v___x_608_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__1___boxed(lean_object* v_sz_610_, lean_object* v_i_611_, lean_object* v_bs_612_){
_start:
{
size_t v_sz_boxed_613_; size_t v_i_boxed_614_; lean_object* v_res_615_; 
v_sz_boxed_613_ = lean_unbox_usize(v_sz_610_);
lean_dec(v_sz_610_);
v_i_boxed_614_ = lean_unbox_usize(v_i_611_);
lean_dec(v_i_611_);
v_res_615_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__1(v_sz_boxed_613_, v_i_boxed_614_, v_bs_612_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10(lean_object* v_name__arr_616_, lean_object* v_as_617_, size_t v_i_618_, size_t v_stop_619_, lean_object* v_b_620_){
_start:
{
lean_object* v___y_622_; uint8_t v___x_626_; 
v___x_626_ = lean_usize_dec_eq(v_i_618_, v_stop_619_);
if (v___x_626_ == 0)
{
lean_object* v___x_627_; lean_object* v_module_628_; uint8_t v___x_629_; 
v___x_627_ = lean_array_uget_borrowed(v_as_617_, v_i_618_);
v_module_628_ = lean_ctor_get(v___x_627_, 0);
v___x_629_ = lp_importGraph_Array_contains___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__4(v_name__arr_616_, v_module_628_);
if (v___x_629_ == 0)
{
lean_object* v___x_630_; 
lean_inc(v___x_627_);
v___x_630_ = lean_array_push(v_b_620_, v___x_627_);
v___y_622_ = v___x_630_;
goto v___jp_621_;
}
else
{
v___y_622_ = v_b_620_;
goto v___jp_621_;
}
}
else
{
return v_b_620_;
}
v___jp_621_:
{
size_t v___x_623_; size_t v___x_624_; 
v___x_623_ = ((size_t)1ULL);
v___x_624_ = lean_usize_add(v_i_618_, v___x_623_);
v_i_618_ = v___x_624_;
v_b_620_ = v___y_622_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10___boxed(lean_object* v_name__arr_631_, lean_object* v_as_632_, lean_object* v_i_633_, lean_object* v_stop_634_, lean_object* v_b_635_){
_start:
{
size_t v_i_boxed_636_; size_t v_stop_boxed_637_; lean_object* v_res_638_; 
v_i_boxed_636_ = lean_unbox_usize(v_i_633_);
lean_dec(v_i_633_);
v_stop_boxed_637_ = lean_unbox_usize(v_stop_634_);
lean_dec(v_stop_634_);
v_res_638_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10(v_name__arr_631_, v_as_632_, v_i_boxed_636_, v_stop_boxed_637_, v_b_635_);
lean_dec_ref(v_as_632_);
lean_dec_ref(v_name__arr_631_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg(lean_object* v_hi_639_, lean_object* v_pivot_640_, lean_object* v_as_641_, lean_object* v_i_642_, lean_object* v_k_643_){
_start:
{
uint8_t v___x_644_; 
v___x_644_ = lean_nat_dec_lt(v_k_643_, v_hi_639_);
if (v___x_644_ == 0)
{
lean_object* v___x_645_; lean_object* v___x_646_; 
lean_dec(v_k_643_);
v___x_645_ = lean_array_fswap(v_as_641_, v_i_642_, v_hi_639_);
v___x_646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_646_, 0, v_i_642_);
lean_ctor_set(v___x_646_, 1, v___x_645_);
return v___x_646_;
}
else
{
lean_object* v___x_647_; uint8_t v___x_648_; 
v___x_647_ = lean_array_fget_borrowed(v_as_641_, v_k_643_);
v___x_648_ = lean_string_dec_lt(v___x_647_, v_pivot_640_);
if (v___x_648_ == 0)
{
lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_649_ = lean_unsigned_to_nat(1u);
v___x_650_ = lean_nat_add(v_k_643_, v___x_649_);
lean_dec(v_k_643_);
v_k_643_ = v___x_650_;
goto _start;
}
else
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___x_652_ = lean_array_fswap(v_as_641_, v_i_642_, v_k_643_);
v___x_653_ = lean_unsigned_to_nat(1u);
v___x_654_ = lean_nat_add(v_i_642_, v___x_653_);
lean_dec(v_i_642_);
v___x_655_ = lean_nat_add(v_k_643_, v___x_653_);
lean_dec(v_k_643_);
v_as_641_ = v___x_652_;
v_i_642_ = v___x_654_;
v_k_643_ = v___x_655_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg___boxed(lean_object* v_hi_657_, lean_object* v_pivot_658_, lean_object* v_as_659_, lean_object* v_i_660_, lean_object* v_k_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg(v_hi_657_, v_pivot_658_, v_as_659_, v_i_660_, v_k_661_);
lean_dec_ref(v_pivot_658_);
lean_dec(v_hi_657_);
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(lean_object* v_n_663_, lean_object* v_as_664_, lean_object* v_lo_665_, lean_object* v_hi_666_){
_start:
{
lean_object* v___y_668_; uint8_t v___x_678_; 
v___x_678_ = lean_nat_dec_lt(v_lo_665_, v_hi_666_);
if (v___x_678_ == 0)
{
lean_dec(v_lo_665_);
return v_as_664_;
}
else
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v_mid_681_; lean_object* v___y_683_; lean_object* v___y_689_; lean_object* v___x_694_; lean_object* v___x_695_; uint8_t v___x_696_; 
v___x_679_ = lean_nat_add(v_lo_665_, v_hi_666_);
v___x_680_ = lean_unsigned_to_nat(1u);
v_mid_681_ = lean_nat_shiftr(v___x_679_, v___x_680_);
lean_dec(v___x_679_);
v___x_694_ = lean_array_fget_borrowed(v_as_664_, v_mid_681_);
v___x_695_ = lean_array_fget_borrowed(v_as_664_, v_lo_665_);
v___x_696_ = lean_string_dec_lt(v___x_694_, v___x_695_);
if (v___x_696_ == 0)
{
v___y_689_ = v_as_664_;
goto v___jp_688_;
}
else
{
lean_object* v___x_697_; 
v___x_697_ = lean_array_fswap(v_as_664_, v_lo_665_, v_mid_681_);
v___y_689_ = v___x_697_;
goto v___jp_688_;
}
v___jp_682_:
{
lean_object* v___x_684_; lean_object* v___x_685_; uint8_t v___x_686_; 
v___x_684_ = lean_array_fget_borrowed(v___y_683_, v_mid_681_);
v___x_685_ = lean_array_fget_borrowed(v___y_683_, v_hi_666_);
v___x_686_ = lean_string_dec_lt(v___x_684_, v___x_685_);
if (v___x_686_ == 0)
{
lean_dec(v_mid_681_);
v___y_668_ = v___y_683_;
goto v___jp_667_;
}
else
{
lean_object* v___x_687_; 
v___x_687_ = lean_array_fswap(v___y_683_, v_mid_681_, v_hi_666_);
lean_dec(v_mid_681_);
v___y_668_ = v___x_687_;
goto v___jp_667_;
}
}
v___jp_688_:
{
lean_object* v___x_690_; lean_object* v___x_691_; uint8_t v___x_692_; 
v___x_690_ = lean_array_fget_borrowed(v___y_689_, v_hi_666_);
v___x_691_ = lean_array_fget_borrowed(v___y_689_, v_lo_665_);
v___x_692_ = lean_string_dec_lt(v___x_690_, v___x_691_);
if (v___x_692_ == 0)
{
v___y_683_ = v___y_689_;
goto v___jp_682_;
}
else
{
lean_object* v___x_693_; 
v___x_693_ = lean_array_fswap(v___y_689_, v_lo_665_, v_hi_666_);
v___y_683_ = v___x_693_;
goto v___jp_682_;
}
}
}
v___jp_667_:
{
lean_object* v_pivot_669_; lean_object* v___x_670_; lean_object* v_fst_671_; lean_object* v_snd_672_; uint8_t v___x_673_; 
v_pivot_669_ = lean_array_fget(v___y_668_, v_hi_666_);
lean_inc_n(v_lo_665_, 2);
v___x_670_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg(v_hi_666_, v_pivot_669_, v___y_668_, v_lo_665_, v_lo_665_);
lean_dec(v_pivot_669_);
v_fst_671_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_fst_671_);
v_snd_672_ = lean_ctor_get(v___x_670_, 1);
lean_inc(v_snd_672_);
lean_dec_ref(v___x_670_);
v___x_673_ = lean_nat_dec_le(v_hi_666_, v_fst_671_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v___x_674_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(v_n_663_, v_snd_672_, v_lo_665_, v_fst_671_);
v___x_675_ = lean_unsigned_to_nat(1u);
v___x_676_ = lean_nat_add(v_fst_671_, v___x_675_);
lean_dec(v_fst_671_);
v_as_664_ = v___x_674_;
v_lo_665_ = v___x_676_;
goto _start;
}
else
{
lean_dec(v_fst_671_);
lean_dec(v_lo_665_);
return v_snd_672_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg___boxed(lean_object* v_n_698_, lean_object* v_as_699_, lean_object* v_lo_700_, lean_object* v_hi_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(v_n_698_, v_as_699_, v_lo_700_, v_hi_701_);
lean_dec(v_hi_701_);
lean_dec(v_n_698_);
return v_res_702_;
}
}
static lean_object* _init_lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__6(void){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_711_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__5));
v___x_712_ = l_Lean_stringToMessageData(v___x_711_);
return v___x_712_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1(lean_object* v_x_715_, lean_object* v_a_716_, lean_object* v_a_717_){
_start:
{
lean_object* v___y_720_; lean_object* v___y_721_; lean_object* v___y_722_; lean_object* v___y_723_; lean_object* v___y_724_; lean_object* v___y_738_; lean_object* v___y_739_; lean_object* v___y_740_; lean_object* v___y_741_; lean_object* v___y_742_; lean_object* v___y_743_; lean_object* v___y_744_; lean_object* v___y_745_; lean_object* v___y_748_; lean_object* v___y_749_; lean_object* v___y_750_; lean_object* v___y_751_; lean_object* v___y_752_; lean_object* v___y_753_; lean_object* v___y_754_; lean_object* v___y_755_; lean_object* v___x_757_; uint8_t v___x_758_; 
v___x_757_ = ((lean_object*)(lp_importGraph_command_x23import__diff___00__closed__1));
lean_inc(v_x_715_);
v___x_758_ = l_Lean_Syntax_isOfKind(v_x_715_, v___x_757_);
if (v___x_758_ == 0)
{
lean_object* v___x_759_; 
lean_dec(v_x_715_);
v___x_759_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__0___redArg();
return v___x_759_;
}
else
{
lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v_n_764_; size_t v_sz_765_; size_t v___x_766_; lean_object* v_name__arr_767_; lean_object* v___x_768_; size_t v_sz_769_; lean_object* v___x_770_; 
v___x_760_ = l_Lean_searchPathRef;
v___x_761_ = lean_st_ref_get(v___x_760_);
v___x_762_ = lean_unsigned_to_nat(1u);
v___x_763_ = l_Lean_Syntax_getArg(v_x_715_, v___x_762_);
lean_dec(v_x_715_);
v_n_764_ = l_Lean_Syntax_getArgs(v___x_763_);
lean_dec(v___x_763_);
v_sz_765_ = lean_array_size(v_n_764_);
v___x_766_ = ((size_t)0ULL);
v_name__arr_767_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__1(v_sz_765_, v___x_766_, v_n_764_);
v___x_768_ = lean_box(0);
v_sz_769_ = lean_array_size(v_name__arr_767_);
v___x_770_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__3(v___x_761_, v___x_758_, v_name__arr_767_, v_sz_769_, v___x_766_, v___x_768_, v_a_716_, v_a_717_);
if (lean_obj_tag(v___x_770_) == 0)
{
lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_899_; 
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_770_);
if (v_isSharedCheck_899_ == 0)
{
lean_object* v_unused_900_; 
v_unused_900_ = lean_ctor_get(v___x_770_, 0);
lean_dec(v_unused_900_);
v___x_772_ = v___x_770_;
v_isShared_773_ = v_isSharedCheck_899_;
goto v_resetjp_771_;
}
else
{
lean_dec(v___x_770_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_899_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_774_; lean_object* v_env_775_; lean_object* v___x_776_; lean_object* v___y_778_; lean_object* v___y_779_; lean_object* v___y_780_; lean_object* v___y_789_; lean_object* v___y_790_; lean_object* v___y_791_; lean_object* v___y_792_; lean_object* v___y_845_; lean_object* v___y_846_; lean_object* v___y_857_; lean_object* v___y_858_; lean_object* v___y_866_; lean_object* v___y_867_; lean_object* v___y_868_; lean_object* v___y_869_; lean_object* v___y_870_; lean_object* v___y_873_; lean_object* v___y_874_; lean_object* v___y_875_; lean_object* v___y_876_; lean_object* v___y_877_; lean_object* v___y_880_; lean_object* v___x_890_; lean_object* v___x_891_; uint8_t v___x_892_; 
v___x_774_ = lean_st_ref_get(v_a_717_);
v_env_775_ = lean_ctor_get(v___x_774_, 0);
lean_inc_ref(v_env_775_);
lean_dec(v___x_774_);
v___x_776_ = lean_unsigned_to_nat(0u);
v___x_890_ = lean_array_get_size(v_name__arr_767_);
v___x_891_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__7));
v___x_892_ = lean_nat_dec_lt(v___x_776_, v___x_890_);
if (v___x_892_ == 0)
{
v___y_880_ = v___x_891_;
goto v___jp_879_;
}
else
{
lean_object* v___x_893_; uint8_t v___x_894_; 
v___x_893_ = l_Lean_Environment_allImportedModuleNames(v_env_775_);
v___x_894_ = lean_nat_dec_le(v___x_890_, v___x_890_);
if (v___x_894_ == 0)
{
if (v___x_892_ == 0)
{
lean_dec_ref(v___x_893_);
v___y_880_ = v___x_891_;
goto v___jp_879_;
}
else
{
size_t v___x_895_; lean_object* v___x_896_; 
v___x_895_ = lean_usize_of_nat(v___x_890_);
v___x_896_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11(v___x_893_, v_name__arr_767_, v___x_766_, v___x_895_, v___x_891_);
lean_dec_ref(v___x_893_);
v___y_880_ = v___x_896_;
goto v___jp_879_;
}
}
else
{
size_t v___x_897_; lean_object* v___x_898_; 
v___x_897_ = lean_usize_of_nat(v___x_890_);
v___x_898_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__11(v___x_893_, v_name__arr_767_, v___x_766_, v___x_897_, v___x_891_);
lean_dec_ref(v___x_893_);
v___y_880_ = v___x_898_;
goto v___jp_879_;
}
}
v___jp_777_:
{
lean_object* v___x_781_; size_t v_sz_782_; lean_object* v___x_783_; lean_object* v___x_784_; uint8_t v___x_785_; 
v___x_781_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__2));
v_sz_782_ = lean_array_size(v___y_780_);
lean_inc_ref(v___y_780_);
v___x_783_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7(v___x_758_, v_sz_782_, v___x_766_, v___y_780_);
v___x_784_ = lean_array_get_size(v___x_783_);
v___x_785_ = lean_nat_dec_eq(v___x_784_, v___x_776_);
if (v___x_785_ == 0)
{
lean_object* v___x_786_; uint8_t v___x_787_; 
v___x_786_ = lean_nat_sub(v___x_784_, v___x_762_);
v___x_787_ = lean_nat_dec_le(v___x_776_, v___x_786_);
if (v___x_787_ == 0)
{
lean_inc(v___x_786_);
v___y_748_ = v___x_784_;
v___y_749_ = v___y_778_;
v___y_750_ = v___x_783_;
v___y_751_ = v___x_786_;
v___y_752_ = v___y_780_;
v___y_753_ = v___y_779_;
v___y_754_ = v___x_781_;
v___y_755_ = v___x_786_;
goto v___jp_747_;
}
else
{
v___y_748_ = v___x_784_;
v___y_749_ = v___y_778_;
v___y_750_ = v___x_783_;
v___y_751_ = v___x_786_;
v___y_752_ = v___y_780_;
v___y_753_ = v___y_779_;
v___y_754_ = v___x_781_;
v___y_755_ = v___x_776_;
goto v___jp_747_;
}
}
else
{
v___y_720_ = v___y_778_;
v___y_721_ = v___y_780_;
v___y_722_ = v___y_779_;
v___y_723_ = v___x_781_;
v___y_724_ = v___x_783_;
goto v___jp_719_;
}
}
v___jp_788_:
{
lean_object* v___x_793_; uint32_t v___x_794_; lean_object* v___x_795_; uint8_t v___x_796_; uint8_t v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; 
v___x_793_ = l_Lean_Options_empty;
v___x_794_ = 0;
v___x_795_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__3));
v___x_796_ = 0;
v___x_797_ = 2;
v___x_798_ = lean_box(1);
v___x_799_ = l_Lean_importModules(v___y_792_, v___x_793_, v___x_794_, v___x_795_, v___x_796_, v___x_796_, v___x_797_, v___x_798_);
if (lean_obj_tag(v___x_799_) == 0)
{
lean_object* v_a_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; 
v_a_800_ = lean_ctor_get(v___x_799_, 0);
lean_inc(v_a_800_);
lean_dec_ref_known(v___x_799_, 1);
v___x_801_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__5(v___x_758_, v_sz_769_, v___x_766_, v_name__arr_767_);
v___x_802_ = l_Array_append___redArg(v___y_790_, v___x_801_);
lean_dec_ref(v___x_801_);
v___x_803_ = l_Lean_importModules(v___x_802_, v___x_793_, v___x_794_, v___x_795_, v___x_796_, v___x_796_, v___x_797_, v___x_798_);
if (lean_obj_tag(v___x_803_) == 0)
{
lean_object* v_a_804_; lean_object* v___x_805_; lean_object* v___x_806_; uint8_t v___x_807_; 
lean_del_object(v___x_772_);
v_a_804_ = lean_ctor_get(v___x_803_, 0);
lean_inc(v_a_804_);
lean_dec_ref_known(v___x_803_, 1);
v___x_805_ = l_Lean_Environment_allImportedModuleNames(v_a_804_);
lean_dec(v_a_804_);
v___x_806_ = lean_array_get_size(v___x_805_);
v___x_807_ = lean_nat_dec_lt(v___x_776_, v___x_806_);
if (v___x_807_ == 0)
{
lean_dec_ref(v___x_805_);
lean_dec(v_a_800_);
v___y_778_ = v___y_789_;
v___y_779_ = v___y_791_;
v___y_780_ = v___x_795_;
goto v___jp_777_;
}
else
{
lean_object* v___x_808_; uint8_t v___x_809_; 
v___x_808_ = l_Lean_Environment_allImportedModuleNames(v_a_800_);
lean_dec(v_a_800_);
v___x_809_ = lean_nat_dec_le(v___x_806_, v___x_806_);
if (v___x_809_ == 0)
{
if (v___x_807_ == 0)
{
lean_dec_ref(v___x_808_);
lean_dec_ref(v___x_805_);
v___y_778_ = v___y_789_;
v___y_779_ = v___y_791_;
v___y_780_ = v___x_795_;
goto v___jp_777_;
}
else
{
size_t v___x_810_; lean_object* v___x_811_; 
v___x_810_ = lean_usize_of_nat(v___x_806_);
v___x_811_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9(v___x_808_, v___x_805_, v___x_766_, v___x_810_, v___x_795_);
lean_dec_ref(v___x_805_);
v___y_778_ = v___y_789_;
v___y_779_ = v___y_791_;
v___y_780_ = v___x_811_;
goto v___jp_777_;
}
}
else
{
size_t v___x_812_; lean_object* v___x_813_; 
v___x_812_ = lean_usize_of_nat(v___x_806_);
v___x_813_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__9(v___x_808_, v___x_805_, v___x_766_, v___x_812_, v___x_795_);
lean_dec_ref(v___x_805_);
v___y_778_ = v___y_789_;
v___y_779_ = v___y_791_;
v___y_780_ = v___x_813_;
goto v___jp_777_;
}
}
}
else
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_828_; 
lean_dec(v_a_800_);
v_a_814_ = lean_ctor_get(v___x_803_, 0);
v_isSharedCheck_828_ = !lean_is_exclusive(v___x_803_);
if (v_isSharedCheck_828_ == 0)
{
v___x_816_ = v___x_803_;
v_isShared_817_ = v_isSharedCheck_828_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_803_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_828_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v_ref_818_; lean_object* v___x_819_; lean_object* v___x_821_; 
v_ref_818_ = lean_ctor_get(v___y_791_, 7);
v___x_819_ = lean_io_error_to_string(v_a_814_);
if (v_isShared_773_ == 0)
{
lean_ctor_set_tag(v___x_772_, 3);
lean_ctor_set(v___x_772_, 0, v___x_819_);
v___x_821_ = v___x_772_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v___x_819_);
v___x_821_ = v_reuseFailAlloc_827_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_825_; 
v___x_822_ = l_Lean_MessageData_ofFormat(v___x_821_);
lean_inc(v_ref_818_);
v___x_823_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_823_, 0, v_ref_818_);
lean_ctor_set(v___x_823_, 1, v___x_822_);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v___x_823_);
v___x_825_ = v___x_816_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_826_; 
v_reuseFailAlloc_826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_826_, 0, v___x_823_);
v___x_825_ = v_reuseFailAlloc_826_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
return v___x_825_;
}
}
}
}
}
else
{
lean_object* v_a_829_; lean_object* v___x_831_; uint8_t v_isShared_832_; uint8_t v_isSharedCheck_843_; 
lean_dec_ref(v___y_790_);
lean_dec_ref(v_name__arr_767_);
v_a_829_ = lean_ctor_get(v___x_799_, 0);
v_isSharedCheck_843_ = !lean_is_exclusive(v___x_799_);
if (v_isSharedCheck_843_ == 0)
{
v___x_831_ = v___x_799_;
v_isShared_832_ = v_isSharedCheck_843_;
goto v_resetjp_830_;
}
else
{
lean_inc(v_a_829_);
lean_dec(v___x_799_);
v___x_831_ = lean_box(0);
v_isShared_832_ = v_isSharedCheck_843_;
goto v_resetjp_830_;
}
v_resetjp_830_:
{
lean_object* v_ref_833_; lean_object* v___x_834_; lean_object* v___x_836_; 
v_ref_833_ = lean_ctor_get(v___y_791_, 7);
v___x_834_ = lean_io_error_to_string(v_a_829_);
if (v_isShared_773_ == 0)
{
lean_ctor_set_tag(v___x_772_, 3);
lean_ctor_set(v___x_772_, 0, v___x_834_);
v___x_836_ = v___x_772_;
goto v_reusejp_835_;
}
else
{
lean_object* v_reuseFailAlloc_842_; 
v_reuseFailAlloc_842_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_842_, 0, v___x_834_);
v___x_836_ = v_reuseFailAlloc_842_;
goto v_reusejp_835_;
}
v_reusejp_835_:
{
lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_840_; 
v___x_837_ = l_Lean_MessageData_ofFormat(v___x_836_);
lean_inc(v_ref_833_);
v___x_838_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_838_, 0, v_ref_833_);
lean_ctor_set(v___x_838_, 1, v___x_837_);
if (v_isShared_832_ == 0)
{
lean_ctor_set(v___x_831_, 0, v___x_838_);
v___x_840_ = v___x_831_;
goto v_reusejp_839_;
}
else
{
lean_object* v_reuseFailAlloc_841_; 
v_reuseFailAlloc_841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_841_, 0, v___x_838_);
v___x_840_ = v_reuseFailAlloc_841_;
goto v_reusejp_839_;
}
v_reusejp_839_:
{
return v___x_840_;
}
}
}
}
}
v___jp_844_:
{
lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; uint8_t v___x_850_; 
v___x_847_ = l_Lean_Environment_imports(v_env_775_);
lean_dec_ref(v_env_775_);
v___x_848_ = lean_array_get_size(v___x_847_);
v___x_849_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__4));
v___x_850_ = lean_nat_dec_lt(v___x_776_, v___x_848_);
if (v___x_850_ == 0)
{
v___y_789_ = v___y_846_;
v___y_790_ = v___x_847_;
v___y_791_ = v___y_845_;
v___y_792_ = v___x_849_;
goto v___jp_788_;
}
else
{
uint8_t v___x_851_; 
v___x_851_ = lean_nat_dec_le(v___x_848_, v___x_848_);
if (v___x_851_ == 0)
{
if (v___x_850_ == 0)
{
v___y_789_ = v___y_846_;
v___y_790_ = v___x_847_;
v___y_791_ = v___y_845_;
v___y_792_ = v___x_849_;
goto v___jp_788_;
}
else
{
size_t v___x_852_; lean_object* v___x_853_; 
v___x_852_ = lean_usize_of_nat(v___x_848_);
v___x_853_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10(v_name__arr_767_, v___x_847_, v___x_766_, v___x_852_, v___x_849_);
v___y_789_ = v___y_846_;
v___y_790_ = v___x_847_;
v___y_791_ = v___y_845_;
v___y_792_ = v___x_853_;
goto v___jp_788_;
}
}
else
{
size_t v___x_854_; lean_object* v___x_855_; 
v___x_854_ = lean_usize_of_nat(v___x_848_);
v___x_855_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__10(v_name__arr_767_, v___x_847_, v___x_766_, v___x_854_, v___x_849_);
v___y_789_ = v___y_846_;
v___y_790_ = v___x_847_;
v___y_791_ = v___y_845_;
v___y_792_ = v___x_855_;
goto v___jp_788_;
}
}
}
v___jp_856_:
{
lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
v___x_859_ = lean_array_to_list(v___y_858_);
v___x_860_ = l_String_intercalate(v___y_857_, v___x_859_);
v___x_861_ = lean_obj_once(&lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__6, &lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__6_once, _init_lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__6);
v___x_862_ = l_Lean_stringToMessageData(v___x_860_);
v___x_863_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_863_, 0, v___x_861_);
lean_ctor_set(v___x_863_, 1, v___x_862_);
v___x_864_ = lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6(v___x_863_, v_a_716_, v_a_717_);
if (lean_obj_tag(v___x_864_) == 0)
{
lean_dec_ref_known(v___x_864_, 1);
v___y_845_ = v_a_716_;
v___y_846_ = v_a_717_;
goto v___jp_844_;
}
else
{
lean_dec_ref(v_env_775_);
lean_del_object(v___x_772_);
lean_dec_ref(v_name__arr_767_);
return v___x_864_;
}
}
v___jp_865_:
{
lean_object* v___x_871_; 
v___x_871_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(v___y_867_, v___y_869_, v___y_866_, v___y_870_);
lean_dec(v___y_870_);
lean_dec(v___y_867_);
v___y_857_ = v___y_868_;
v___y_858_ = v___x_871_;
goto v___jp_856_;
}
v___jp_872_:
{
uint8_t v___x_878_; 
v___x_878_ = lean_nat_dec_le(v___y_877_, v___y_873_);
if (v___x_878_ == 0)
{
lean_dec(v___y_873_);
lean_inc(v___y_877_);
v___y_866_ = v___y_877_;
v___y_867_ = v___y_874_;
v___y_868_ = v___y_875_;
v___y_869_ = v___y_876_;
v___y_870_ = v___y_877_;
goto v___jp_865_;
}
else
{
v___y_866_ = v___y_877_;
v___y_867_ = v___y_874_;
v___y_868_ = v___y_875_;
v___y_869_ = v___y_876_;
v___y_870_ = v___y_873_;
goto v___jp_865_;
}
}
v___jp_879_:
{
lean_object* v___x_881_; uint8_t v___x_882_; 
v___x_881_ = lean_array_get_size(v___y_880_);
v___x_882_ = lean_nat_dec_eq(v___x_881_, v___x_776_);
if (v___x_882_ == 0)
{
lean_object* v___x_883_; size_t v_sz_884_; lean_object* v___x_885_; lean_object* v___x_886_; uint8_t v___x_887_; 
v___x_883_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__2));
v_sz_884_ = lean_array_size(v___y_880_);
v___x_885_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__7(v___x_758_, v_sz_884_, v___x_766_, v___y_880_);
v___x_886_ = lean_array_get_size(v___x_885_);
v___x_887_ = lean_nat_dec_eq(v___x_886_, v___x_776_);
if (v___x_887_ == 0)
{
lean_object* v___x_888_; uint8_t v___x_889_; 
v___x_888_ = lean_nat_sub(v___x_886_, v___x_762_);
v___x_889_ = lean_nat_dec_le(v___x_776_, v___x_888_);
if (v___x_889_ == 0)
{
lean_inc(v___x_888_);
v___y_873_ = v___x_888_;
v___y_874_ = v___x_886_;
v___y_875_ = v___x_883_;
v___y_876_ = v___x_885_;
v___y_877_ = v___x_888_;
goto v___jp_872_;
}
else
{
v___y_873_ = v___x_888_;
v___y_874_ = v___x_886_;
v___y_875_ = v___x_883_;
v___y_876_ = v___x_885_;
v___y_877_ = v___x_776_;
goto v___jp_872_;
}
}
else
{
v___y_857_ = v___x_883_;
v___y_858_ = v___x_885_;
goto v___jp_856_;
}
}
else
{
lean_dec_ref(v___y_880_);
v___y_845_ = v_a_716_;
v___y_846_ = v_a_717_;
goto v___jp_844_;
}
}
}
}
else
{
lean_dec_ref(v_name__arr_767_);
return v___x_770_;
}
}
v___jp_719_:
{
lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; 
v___x_725_ = lean_array_to_list(v___y_724_);
v___x_726_ = l_String_intercalate(v___y_723_, v___x_725_);
v___x_727_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__0));
v___x_728_ = lean_array_get_size(v___y_721_);
lean_dec_ref(v___y_721_);
v___x_729_ = l_Nat_reprFast(v___x_728_);
v___x_730_ = lean_string_append(v___x_727_, v___x_729_);
lean_dec_ref(v___x_729_);
v___x_731_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___closed__1));
v___x_732_ = lean_string_append(v___x_730_, v___x_731_);
v___x_733_ = lean_string_append(v___x_732_, v___x_726_);
lean_dec_ref(v___x_726_);
v___x_734_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_734_, 0, v___x_733_);
v___x_735_ = l_Lean_MessageData_ofFormat(v___x_734_);
v___x_736_ = lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__6(v___x_735_, v___y_722_, v___y_720_);
return v___x_736_;
}
v___jp_737_:
{
lean_object* v___x_746_; 
v___x_746_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(v___y_739_, v___y_740_, v___y_744_, v___y_745_);
lean_dec(v___y_745_);
lean_dec(v___y_739_);
v___y_720_ = v___y_738_;
v___y_721_ = v___y_741_;
v___y_722_ = v___y_742_;
v___y_723_ = v___y_743_;
v___y_724_ = v___x_746_;
goto v___jp_719_;
}
v___jp_747_:
{
uint8_t v___x_756_; 
v___x_756_ = lean_nat_dec_le(v___y_755_, v___y_751_);
if (v___x_756_ == 0)
{
lean_dec(v___y_751_);
lean_inc(v___y_755_);
v___y_738_ = v___y_749_;
v___y_739_ = v___y_748_;
v___y_740_ = v___y_750_;
v___y_741_ = v___y_752_;
v___y_742_ = v___y_753_;
v___y_743_ = v___y_754_;
v___y_744_ = v___y_755_;
v___y_745_ = v___y_755_;
goto v___jp_737_;
}
else
{
v___y_738_ = v___y_749_;
v___y_739_ = v___y_748_;
v___y_740_ = v___y_750_;
v___y_741_ = v___y_752_;
v___y_742_ = v___y_753_;
v___y_743_ = v___y_754_;
v___y_744_ = v___y_755_;
v___y_745_ = v___y_751_;
goto v___jp_737_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1___boxed(lean_object* v_x_901_, lean_object* v_a_902_, lean_object* v_a_903_, lean_object* v_a_904_){
_start:
{
lean_object* v_res_905_; 
v_res_905_ = lp_importGraph___aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1(v_x_901_, v_a_902_, v_a_903_);
lean_dec(v_a_903_);
lean_dec_ref(v_a_902_);
return v_res_905_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2(lean_object* v_msgData_906_, lean_object* v___y_907_, lean_object* v___y_908_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___redArg(v_msgData_906_, v___y_908_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2___boxed(lean_object* v_msgData_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__2(v_msgData_911_, v___y_912_, v___y_913_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2(lean_object* v_00_u03b1_916_, lean_object* v_msg_917_, lean_object* v___y_918_, lean_object* v___y_919_){
_start:
{
lean_object* v___x_921_; 
v___x_921_ = lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___redArg(v_msg_917_, v___y_918_, v___y_919_);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2___boxed(lean_object* v_00_u03b1_922_, lean_object* v_msg_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_){
_start:
{
lean_object* v_res_927_; 
v_res_927_ = lp_importGraph_Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2(v_00_u03b1_922_, v_msg_923_, v___y_924_, v___y_925_);
lean_dec(v___y_925_);
lean_dec_ref(v___y_924_);
return v_res_927_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8(lean_object* v_n_928_, lean_object* v_as_929_, lean_object* v_lo_930_, lean_object* v_hi_931_, lean_object* v_w_932_, lean_object* v_hlo_933_, lean_object* v_hhi_934_){
_start:
{
lean_object* v___x_935_; 
v___x_935_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___redArg(v_n_928_, v_as_929_, v_lo_930_, v_hi_931_);
return v___x_935_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8___boxed(lean_object* v_n_936_, lean_object* v_as_937_, lean_object* v_lo_938_, lean_object* v_hi_939_, lean_object* v_w_940_, lean_object* v_hlo_941_, lean_object* v_hhi_942_){
_start:
{
lean_object* v_res_943_; 
v_res_943_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8(v_n_936_, v_as_937_, v_lo_938_, v_hi_939_, v_w_940_, v_hlo_941_, v_hhi_942_);
lean_dec(v_hi_939_);
lean_dec(v_n_936_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3(lean_object* v_msgData_944_, lean_object* v_macroStack_945_, lean_object* v___y_946_, lean_object* v___y_947_){
_start:
{
lean_object* v___x_949_; 
v___x_949_ = lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___redArg(v_msgData_944_, v_macroStack_945_, v___y_947_);
return v___x_949_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3___boxed(lean_object* v_msgData_950_, lean_object* v_macroStack_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_){
_start:
{
lean_object* v_res_955_; 
v_res_955_ = lp_importGraph_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__2_spec__3(v_msgData_950_, v_macroStack_951_, v___y_952_, v___y_953_);
lean_dec(v___y_953_);
lean_dec_ref(v___y_952_);
return v_res_955_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12(lean_object* v_n_956_, lean_object* v_lo_957_, lean_object* v_hi_958_, lean_object* v_hhi_959_, lean_object* v_pivot_960_, lean_object* v_as_961_, lean_object* v_i_962_, lean_object* v_k_963_, lean_object* v_ilo_964_, lean_object* v_ik_965_, lean_object* v_w_966_){
_start:
{
lean_object* v___x_967_; 
v___x_967_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___redArg(v_hi_958_, v_pivot_960_, v_as_961_, v_i_962_, v_k_963_);
return v___x_967_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12___boxed(lean_object* v_n_968_, lean_object* v_lo_969_, lean_object* v_hi_970_, lean_object* v_hhi_971_, lean_object* v_pivot_972_, lean_object* v_as_973_, lean_object* v_i_974_, lean_object* v_k_975_, lean_object* v_ilo_976_, lean_object* v_ik_977_, lean_object* v_w_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__ImportDiff______elabRules__command_x23import__diff____1_spec__8_spec__12(v_n_968_, v_lo_969_, v_hi_970_, v_hhi_971_, v_pivot_972_, v_as_973_, v_i_974_, v_k_975_, v_ilo_976_, v_ik_977_, v_w_978_);
lean_dec_ref(v_pivot_972_);
lean_dec(v_hi_970_);
lean_dec(v_lo_969_);
lean_dec(v_n_968_);
return v_res_979_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Tools_ImportDiff(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Widget_UserWidget(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Tools_ImportDiff(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Widget_UserWidget(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Widget_UserWidget(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Tools_ImportDiff(uint8_t builtin) {
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
res = initialize_Lean_Widget_UserWidget(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Tools_ImportDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Tools_ImportDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Tools_ImportDiff(builtin);
}
#ifdef __cplusplus
}
#endif
