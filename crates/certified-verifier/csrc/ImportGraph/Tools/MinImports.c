// Lean compiler output
// Module: ImportGraph.Tools.MinImports
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Lean.Widget.UserWidget public meta import ImportGraph.Imports.RequiredModules public meta import ImportGraph.Imports.Redundant
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lp_importGraph_Lean_Environment_requiredModules(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_importGraph_Lean_Environment_findRedundantImports(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1___boxed(lean_object*, lean_object*);
static const lean_array_object lp_importGraph_Lean_Environment_minimalRequiredModules___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph_Lean_Environment_minimalRequiredModules___closed__0 = (const lean_object*)&lp_importGraph_Lean_Environment_minimalRequiredModules___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_minimalRequiredModules(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_importGraph_command_x23min__imports___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "command#min_imports"};
static const lean_object* lp_importGraph_command_x23min__imports___closed__0 = (const lean_object*)&lp_importGraph_command_x23min__imports___closed__0_value;
static const lean_ctor_object lp_importGraph_command_x23min__imports___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23min__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(210, 137, 164, 30, 65, 150, 203, 181)}};
static const lean_object* lp_importGraph_command_x23min__imports___closed__1 = (const lean_object*)&lp_importGraph_command_x23min__imports___closed__1_value;
static const lean_string_object lp_importGraph_command_x23min__imports___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "#min_imports"};
static const lean_object* lp_importGraph_command_x23min__imports___closed__2 = (const lean_object*)&lp_importGraph_command_x23min__imports___closed__2_value;
static const lean_ctor_object lp_importGraph_command_x23min__imports___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_importGraph_command_x23min__imports___closed__2_value)}};
static const lean_object* lp_importGraph_command_x23min__imports___closed__3 = (const lean_object*)&lp_importGraph_command_x23min__imports___closed__3_value;
static const lean_ctor_object lp_importGraph_command_x23min__imports___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph_command_x23min__imports___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23min__imports___closed__3_value)}};
static const lean_object* lp_importGraph_command_x23min__imports___closed__4 = (const lean_object*)&lp_importGraph_command_x23min__imports___closed__4_value;
LEAN_EXPORT const lean_object* lp_importGraph_command_x23min__imports = (const lean_object*)&lp_importGraph_command_x23min__imports___closed__4_value;
static lean_once_cell_t lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__8___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__0;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__2;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__3;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__4;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__5;
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___closed__0 = (const lean_object*)&lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___closed__0 = (const lean_object*)&lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_List_foldl___at___00Std_Format_joinSep___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_Format_joinSep___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "public import "};
static const lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1___closed__0 = (const lean_object*)&lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__0 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__0_value;
static const lean_ctor_object lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__0_value)}};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__1 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__1_value;
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_command_x23minimize__imports___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "command#minimize_imports"};
static const lean_object* lp_importGraph_command_x23minimize__imports___closed__0 = (const lean_object*)&lp_importGraph_command_x23minimize__imports___closed__0_value;
static const lean_ctor_object lp_importGraph_command_x23minimize__imports___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23minimize__imports___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 216, 99, 224, 69, 18, 237, 213)}};
static const lean_object* lp_importGraph_command_x23minimize__imports___closed__1 = (const lean_object*)&lp_importGraph_command_x23minimize__imports___closed__1_value;
static const lean_string_object lp_importGraph_command_x23minimize__imports___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "#minimize_imports"};
static const lean_object* lp_importGraph_command_x23minimize__imports___closed__2 = (const lean_object*)&lp_importGraph_command_x23minimize__imports___closed__2_value;
static const lean_ctor_object lp_importGraph_command_x23minimize__imports___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_importGraph_command_x23minimize__imports___closed__2_value)}};
static const lean_object* lp_importGraph_command_x23minimize__imports___closed__3 = (const lean_object*)&lp_importGraph_command_x23minimize__imports___closed__3_value;
static const lean_ctor_object lp_importGraph_command_x23minimize__imports___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph_command_x23minimize__imports___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23minimize__imports___closed__3_value)}};
static const lean_object* lp_importGraph_command_x23minimize__imports___closed__4 = (const lean_object*)&lp_importGraph_command_x23minimize__imports___closed__4_value;
LEAN_EXPORT const lean_object* lp_importGraph_command_x23minimize__imports = (const lean_object*)&lp_importGraph_command_x23minimize__imports___closed__4_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logWarning___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logWarning___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "'#minimize_imports' is deprecated: please use '#min_imports'"};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__0 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__0_value;
static lean_once_cell_t lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__1;
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0_spec__0(lean_object* v_init_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v_k_3_; lean_object* v_l_4_; lean_object* v_r_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v_k_3_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_k_3_);
v_l_4_ = lean_ctor_get(v_x_2_, 3);
lean_inc(v_l_4_);
v_r_5_ = lean_ctor_get(v_x_2_, 4);
lean_inc(v_r_5_);
lean_dec_ref_known(v_x_2_, 5);
v___x_6_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0_spec__0(v_init_1_, v_l_4_);
v___x_7_ = lean_array_push(v___x_6_, v_k_3_);
v_init_1_ = v___x_7_;
v_x_2_ = v_r_5_;
goto _start;
}
else
{
return v_init_1_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2(lean_object* v_redundant_9_, lean_object* v_as_10_, size_t v_i_11_, size_t v_stop_12_, lean_object* v_b_13_){
_start:
{
lean_object* v___y_15_; uint8_t v___x_19_; 
v___x_19_ = lean_usize_dec_eq(v_i_11_, v_stop_12_);
if (v___x_19_ == 0)
{
lean_object* v___x_20_; uint8_t v___x_23_; 
v___x_20_ = lean_array_uget_borrowed(v_as_10_, v_i_11_);
v___x_23_ = l_Lean_NameSet_contains(v_redundant_9_, v___x_20_);
if (v___x_23_ == 0)
{
goto v___jp_21_;
}
else
{
if (v___x_19_ == 0)
{
v___y_15_ = v_b_13_;
goto v___jp_14_;
}
else
{
goto v___jp_21_;
}
}
v___jp_21_:
{
lean_object* v___x_22_; 
lean_inc(v___x_20_);
v___x_22_ = lean_array_push(v_b_13_, v___x_20_);
v___y_15_ = v___x_22_;
goto v___jp_14_;
}
}
else
{
return v_b_13_;
}
v___jp_14_:
{
size_t v___x_16_; size_t v___x_17_; 
v___x_16_ = ((size_t)1ULL);
v___x_17_ = lean_usize_add(v_i_11_, v___x_16_);
v_i_11_ = v___x_17_;
v_b_13_ = v___y_15_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2___boxed(lean_object* v_redundant_24_, lean_object* v_as_25_, lean_object* v_i_26_, lean_object* v_stop_27_, lean_object* v_b_28_){
_start:
{
size_t v_i_boxed_29_; size_t v_stop_boxed_30_; lean_object* v_res_31_; 
v_i_boxed_29_ = lean_unbox_usize(v_i_26_);
lean_dec(v_i_26_);
v_stop_boxed_30_ = lean_unbox_usize(v_stop_27_);
lean_dec(v_stop_27_);
v_res_31_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2(v_redundant_24_, v_as_25_, v_i_boxed_29_, v_stop_boxed_30_, v_b_28_);
lean_dec_ref(v_as_25_);
lean_dec(v_redundant_24_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2_spec__3(lean_object* v_xs_32_, lean_object* v_v_33_, lean_object* v_i_34_){
_start:
{
lean_object* v___x_35_; uint8_t v___x_36_; 
v___x_35_ = lean_array_get_size(v_xs_32_);
v___x_36_ = lean_nat_dec_lt(v_i_34_, v___x_35_);
if (v___x_36_ == 0)
{
lean_object* v___x_37_; 
lean_dec(v_i_34_);
v___x_37_ = lean_box(0);
return v___x_37_;
}
else
{
lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_38_ = lean_array_fget_borrowed(v_xs_32_, v_i_34_);
v___x_39_ = lean_name_eq(v___x_38_, v_v_33_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = lean_unsigned_to_nat(1u);
v___x_41_ = lean_nat_add(v_i_34_, v___x_40_);
lean_dec(v_i_34_);
v_i_34_ = v___x_41_;
goto _start;
}
else
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_43_, 0, v_i_34_);
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2_spec__3___boxed(lean_object* v_xs_44_, lean_object* v_v_45_, lean_object* v_i_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2_spec__3(v_xs_44_, v_v_45_, v_i_46_);
lean_dec(v_v_45_);
lean_dec_ref(v_xs_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2(lean_object* v_xs_48_, lean_object* v_v_49_){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lean_unsigned_to_nat(0u);
v___x_51_ = lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2_spec__3(v_xs_48_, v_v_49_, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2___boxed(lean_object* v_xs_52_, lean_object* v_v_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2(v_xs_52_, v_v_53_);
lean_dec(v_v_53_);
lean_dec_ref(v_xs_52_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1(lean_object* v_as_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1_spec__2(v_as_55_, v_a_56_);
if (lean_obj_tag(v___x_57_) == 0)
{
return v_as_55_;
}
else
{
lean_object* v_val_58_; lean_object* v___x_59_; 
v_val_58_ = lean_ctor_get(v___x_57_, 0);
lean_inc(v_val_58_);
lean_dec_ref_known(v___x_57_, 1);
v___x_59_ = l_Array_eraseIdx___redArg(v_as_55_, v_val_58_);
return v___x_59_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1___boxed(lean_object* v_as_60_, lean_object* v_a_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_importGraph_Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1(v_as_60_, v_a_61_);
lean_dec(v_a_61_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_minimalRequiredModules(lean_object* v_env_65_){
_start:
{
lean_object* v___x_66_; lean_object* v___y_68_; 
lean_inc_ref(v_env_65_);
v___x_66_ = lp_importGraph_Lean_Environment_requiredModules(v_env_65_);
if (lean_obj_tag(v___x_66_) == 0)
{
lean_object* v_size_86_; 
v_size_86_ = lean_ctor_get(v___x_66_, 0);
lean_inc(v_size_86_);
v___y_68_ = v_size_86_;
goto v___jp_67_;
}
else
{
lean_object* v___x_87_; 
v___x_87_ = lean_unsigned_to_nat(0u);
v___y_68_ = v___x_87_;
goto v___jp_67_;
}
v___jp_67_:
{
lean_object* v___x_69_; lean_object* v_mainModule_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v_required_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; uint8_t v___x_77_; 
v___x_69_ = l_Lean_Environment_header(v_env_65_);
v_mainModule_70_ = lean_ctor_get(v___x_69_, 0);
lean_inc(v_mainModule_70_);
lean_dec_ref(v___x_69_);
v___x_71_ = lean_mk_empty_array_with_capacity(v___y_68_);
lean_dec(v___y_68_);
v___x_72_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0_spec__0(v___x_71_, v___x_66_);
v_required_73_ = lp_importGraph_Array_erase___at___00Lean_Environment_minimalRequiredModules_spec__1(v___x_72_, v_mainModule_70_);
lean_dec(v_mainModule_70_);
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = lean_array_get_size(v_required_73_);
v___x_76_ = ((lean_object*)(lp_importGraph_Lean_Environment_minimalRequiredModules___closed__0));
v___x_77_ = lean_nat_dec_lt(v___x_74_, v___x_75_);
if (v___x_77_ == 0)
{
lean_dec_ref(v_required_73_);
lean_dec_ref(v_env_65_);
return v___x_76_;
}
else
{
lean_object* v_redundant_78_; uint8_t v___x_79_; 
v_redundant_78_ = lp_importGraph_Lean_Environment_findRedundantImports(v_env_65_, v_required_73_);
lean_dec_ref(v_env_65_);
v___x_79_ = lean_nat_dec_le(v___x_75_, v___x_75_);
if (v___x_79_ == 0)
{
if (v___x_77_ == 0)
{
lean_dec(v_redundant_78_);
lean_dec_ref(v_required_73_);
return v___x_76_;
}
else
{
size_t v___x_80_; size_t v___x_81_; lean_object* v___x_82_; 
v___x_80_ = ((size_t)0ULL);
v___x_81_ = lean_usize_of_nat(v___x_75_);
v___x_82_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2(v_redundant_78_, v_required_73_, v___x_80_, v___x_81_, v___x_76_);
lean_dec_ref(v_required_73_);
lean_dec(v_redundant_78_);
return v___x_82_;
}
}
else
{
size_t v___x_83_; size_t v___x_84_; lean_object* v___x_85_; 
v___x_83_ = ((size_t)0ULL);
v___x_84_ = lean_usize_of_nat(v___x_75_);
v___x_85_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_minimalRequiredModules_spec__2(v_redundant_78_, v_required_73_, v___x_83_, v___x_84_, v___x_76_);
lean_dec_ref(v_required_73_);
lean_dec(v_redundant_78_);
return v___x_85_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0(lean_object* v_init_88_, lean_object* v_t_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Environment_minimalRequiredModules_spec__0_spec__0(v_init_88_, v_t_89_);
return v___x_90_;
}
}
static lean_object* _init_lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_102_ = lean_box(0);
v___x_103_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v___x_102_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg(){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = lean_obj_once(&lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___closed__0, &lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___closed__0_once, _init_lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___closed__0);
v___x_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg___boxed(lean_object* v___y_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg();
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0(lean_object* v_00_u03b1_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg();
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___boxed(lean_object* v_00_u03b1_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0(v_00_u03b1_115_, v___y_116_, v___y_117_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
return v_res_119_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__8(lean_object* v_opts_120_, lean_object* v_opt_121_){
_start:
{
lean_object* v_name_122_; lean_object* v_defValue_123_; lean_object* v_map_124_; lean_object* v___x_125_; 
v_name_122_ = lean_ctor_get(v_opt_121_, 0);
v_defValue_123_ = lean_ctor_get(v_opt_121_, 1);
v_map_124_ = lean_ctor_get(v_opts_120_, 0);
v___x_125_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_124_, v_name_122_);
if (lean_obj_tag(v___x_125_) == 0)
{
uint8_t v___x_126_; 
v___x_126_ = lean_unbox(v_defValue_123_);
return v___x_126_;
}
else
{
lean_object* v_val_127_; 
v_val_127_ = lean_ctor_get(v___x_125_, 0);
lean_inc(v_val_127_);
lean_dec_ref_known(v___x_125_, 1);
if (lean_obj_tag(v_val_127_) == 1)
{
uint8_t v_v_128_; 
v_v_128_ = lean_ctor_get_uint8(v_val_127_, 0);
lean_dec_ref_known(v_val_127_, 0);
return v_v_128_;
}
else
{
uint8_t v___x_129_; 
lean_dec(v_val_127_);
v___x_129_ = lean_unbox(v_defValue_123_);
return v___x_129_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__8___boxed(lean_object* v_opts_130_, lean_object* v_opt_131_){
_start:
{
uint8_t v_res_132_; lean_object* v_r_133_; 
v_res_132_ = lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__8(v_opts_130_, v_opt_131_);
lean_dec_ref(v_opt_131_);
lean_dec_ref(v_opts_130_);
v_r_133_ = lean_box(v_res_132_);
return v_r_133_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_134_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__0, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__0_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__0);
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__2(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_137_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1);
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
lean_ctor_set(v___x_139_, 2, v___x_138_);
lean_ctor_set(v___x_139_, 3, v___x_138_);
lean_ctor_set(v___x_139_, 4, v___x_137_);
lean_ctor_set(v___x_139_, 5, v___x_137_);
lean_ctor_set(v___x_139_, 6, v___x_137_);
lean_ctor_set(v___x_139_, 7, v___x_137_);
lean_ctor_set(v___x_139_, 8, v___x_137_);
lean_ctor_set(v___x_139_, 9, v___x_137_);
return v___x_139_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__3(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_140_ = lean_unsigned_to_nat(32u);
v___x_141_ = lean_mk_empty_array_with_capacity(v___x_140_);
v___x_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
return v___x_142_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__4(void){
_start:
{
size_t v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_143_ = ((size_t)5ULL);
v___x_144_ = lean_unsigned_to_nat(0u);
v___x_145_ = lean_unsigned_to_nat(32u);
v___x_146_ = lean_mk_empty_array_with_capacity(v___x_145_);
v___x_147_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__3, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__3_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__3);
v___x_148_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v___x_146_);
lean_ctor_set(v___x_148_, 2, v___x_144_);
lean_ctor_set(v___x_148_, 3, v___x_144_);
lean_ctor_set_usize(v___x_148_, 4, v___x_143_);
return v___x_148_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__5(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_149_ = lean_box(1);
v___x_150_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__4, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__4_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__4);
v___x_151_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__1);
v___x_152_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v___x_150_);
lean_ctor_set(v___x_152_, 2, v___x_149_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg(lean_object* v_msgData_153_, lean_object* v___y_154_){
_start:
{
lean_object* v___x_156_; lean_object* v_env_157_; lean_object* v___x_158_; lean_object* v_scopes_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v_opts_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v___x_156_ = lean_st_ref_get(v___y_154_);
v_env_157_ = lean_ctor_get(v___x_156_, 0);
lean_inc_ref(v_env_157_);
lean_dec(v___x_156_);
v___x_158_ = lean_st_ref_get(v___y_154_);
v_scopes_159_ = lean_ctor_get(v___x_158_, 2);
lean_inc(v_scopes_159_);
lean_dec(v___x_158_);
v___x_160_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_161_ = l_List_head_x21___redArg(v___x_160_, v_scopes_159_);
lean_dec(v_scopes_159_);
v_opts_162_ = lean_ctor_get(v___x_161_, 1);
lean_inc_ref(v_opts_162_);
lean_dec(v___x_161_);
v___x_163_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__2, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__2_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__2);
v___x_164_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__5, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__5_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___closed__5);
v___x_165_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_165_, 0, v_env_157_);
lean_ctor_set(v___x_165_, 1, v___x_163_);
lean_ctor_set(v___x_165_, 2, v___x_164_);
lean_ctor_set(v___x_165_, 3, v_opts_162_);
v___x_166_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v_msgData_153_);
v___x_167_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_167_, 0, v___x_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg___boxed(lean_object* v_msgData_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg(v_msgData_168_, v___y_169_);
lean_dec(v___y_169_);
return v_res_171_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0(uint8_t v___y_173_, uint8_t v_suppressElabErrors_174_, lean_object* v_x_175_){
_start:
{
if (lean_obj_tag(v_x_175_) == 1)
{
lean_object* v_pre_176_; 
v_pre_176_ = lean_ctor_get(v_x_175_, 0);
if (lean_obj_tag(v_pre_176_) == 0)
{
lean_object* v_str_177_; lean_object* v___x_178_; uint8_t v___x_179_; 
v_str_177_ = lean_ctor_get(v_x_175_, 1);
v___x_178_ = ((lean_object*)(lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___closed__0));
v___x_179_ = lean_string_dec_eq(v_str_177_, v___x_178_);
if (v___x_179_ == 0)
{
return v___y_173_;
}
else
{
return v_suppressElabErrors_174_;
}
}
else
{
return v___y_173_;
}
}
else
{
return v___y_173_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___boxed(lean_object* v___y_180_, lean_object* v_suppressElabErrors_181_, lean_object* v_x_182_){
_start:
{
uint8_t v___y_3230__boxed_183_; uint8_t v_suppressElabErrors_boxed_184_; uint8_t v_res_185_; lean_object* v_r_186_; 
v___y_3230__boxed_183_ = lean_unbox(v___y_180_);
v_suppressElabErrors_boxed_184_ = lean_unbox(v_suppressElabErrors_181_);
v_res_185_ = lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0(v___y_3230__boxed_183_, v_suppressElabErrors_boxed_184_, v_x_182_);
lean_dec(v_x_182_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5(lean_object* v_ref_188_, lean_object* v_msgData_189_, uint8_t v_severity_190_, uint8_t v_isSilent_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___y_198_; lean_object* v___y_199_; uint8_t v___y_200_; lean_object* v___y_201_; uint8_t v___y_202_; lean_object* v___y_203_; uint8_t v___y_260_; lean_object* v___y_261_; uint8_t v___y_262_; uint8_t v___y_263_; lean_object* v___y_264_; uint8_t v___y_288_; uint8_t v___y_289_; lean_object* v___y_290_; uint8_t v___y_291_; lean_object* v___y_292_; uint8_t v___y_296_; uint8_t v___y_297_; uint8_t v___y_298_; uint8_t v___x_313_; uint8_t v___y_315_; uint8_t v___y_316_; uint8_t v___y_317_; uint8_t v___y_319_; uint8_t v___x_331_; 
v___x_313_ = 2;
v___x_331_ = l_Lean_instBEqMessageSeverity_beq(v_severity_190_, v___x_313_);
if (v___x_331_ == 0)
{
v___y_319_ = v___x_331_;
goto v___jp_318_;
}
else
{
uint8_t v___x_332_; 
lean_inc_ref(v_msgData_189_);
v___x_332_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_189_);
v___y_319_ = v___x_332_;
goto v___jp_318_;
}
v___jp_195_:
{
lean_object* v___x_204_; 
v___x_204_ = l_Lean_Elab_Command_getScope___redArg(v___y_203_);
if (lean_obj_tag(v___x_204_) == 0)
{
lean_object* v_a_205_; lean_object* v___x_206_; 
v_a_205_ = lean_ctor_get(v___x_204_, 0);
lean_inc(v_a_205_);
lean_dec_ref_known(v___x_204_, 1);
v___x_206_ = l_Lean_Elab_Command_getScope___redArg(v___y_203_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_209_; uint8_t v_isShared_210_; uint8_t v_isSharedCheck_242_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_242_ == 0)
{
v___x_209_ = v___x_206_;
v_isShared_210_ = v_isSharedCheck_242_;
goto v_resetjp_208_;
}
else
{
lean_inc(v_a_207_);
lean_dec(v___x_206_);
v___x_209_ = lean_box(0);
v_isShared_210_ = v_isSharedCheck_242_;
goto v_resetjp_208_;
}
v_resetjp_208_:
{
lean_object* v___x_211_; lean_object* v_currNamespace_212_; lean_object* v_openDecls_213_; lean_object* v_env_214_; lean_object* v_messages_215_; lean_object* v_scopes_216_; lean_object* v_usedQuotCtxts_217_; lean_object* v_nextMacroScope_218_; lean_object* v_maxRecDepth_219_; lean_object* v_ngen_220_; lean_object* v_auxDeclNGen_221_; lean_object* v_infoState_222_; lean_object* v_traceState_223_; lean_object* v_snapshotTasks_224_; lean_object* v_prevLinterStates_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_241_; 
v___x_211_ = lean_st_ref_take(v___y_203_);
v_currNamespace_212_ = lean_ctor_get(v_a_205_, 2);
lean_inc(v_currNamespace_212_);
lean_dec(v_a_205_);
v_openDecls_213_ = lean_ctor_get(v_a_207_, 3);
lean_inc(v_openDecls_213_);
lean_dec(v_a_207_);
v_env_214_ = lean_ctor_get(v___x_211_, 0);
v_messages_215_ = lean_ctor_get(v___x_211_, 1);
v_scopes_216_ = lean_ctor_get(v___x_211_, 2);
v_usedQuotCtxts_217_ = lean_ctor_get(v___x_211_, 3);
v_nextMacroScope_218_ = lean_ctor_get(v___x_211_, 4);
v_maxRecDepth_219_ = lean_ctor_get(v___x_211_, 5);
v_ngen_220_ = lean_ctor_get(v___x_211_, 6);
v_auxDeclNGen_221_ = lean_ctor_get(v___x_211_, 7);
v_infoState_222_ = lean_ctor_get(v___x_211_, 8);
v_traceState_223_ = lean_ctor_get(v___x_211_, 9);
v_snapshotTasks_224_ = lean_ctor_get(v___x_211_, 10);
v_prevLinterStates_225_ = lean_ctor_get(v___x_211_, 11);
v_isSharedCheck_241_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_241_ == 0)
{
v___x_227_ = v___x_211_;
v_isShared_228_ = v_isSharedCheck_241_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_prevLinterStates_225_);
lean_inc(v_snapshotTasks_224_);
lean_inc(v_traceState_223_);
lean_inc(v_infoState_222_);
lean_inc(v_auxDeclNGen_221_);
lean_inc(v_ngen_220_);
lean_inc(v_maxRecDepth_219_);
lean_inc(v_nextMacroScope_218_);
lean_inc(v_usedQuotCtxts_217_);
lean_inc(v_scopes_216_);
lean_inc(v_messages_215_);
lean_inc(v_env_214_);
lean_dec(v___x_211_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_241_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_234_; 
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v_currNamespace_212_);
lean_ctor_set(v___x_229_, 1, v_openDecls_213_);
v___x_230_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
lean_ctor_set(v___x_230_, 1, v___y_196_);
lean_inc_ref(v___y_199_);
lean_inc_ref(v___y_198_);
v___x_231_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_231_, 0, v___y_198_);
lean_ctor_set(v___x_231_, 1, v___y_197_);
lean_ctor_set(v___x_231_, 2, v___y_201_);
lean_ctor_set(v___x_231_, 3, v___y_199_);
lean_ctor_set(v___x_231_, 4, v___x_230_);
lean_ctor_set_uint8(v___x_231_, sizeof(void*)*5, v___y_200_);
lean_ctor_set_uint8(v___x_231_, sizeof(void*)*5 + 1, v___y_202_);
lean_ctor_set_uint8(v___x_231_, sizeof(void*)*5 + 2, v_isSilent_191_);
v___x_232_ = l_Lean_MessageLog_add(v___x_231_, v_messages_215_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 1, v___x_232_);
v___x_234_ = v___x_227_;
goto v_reusejp_233_;
}
else
{
lean_object* v_reuseFailAlloc_240_; 
v_reuseFailAlloc_240_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_240_, 0, v_env_214_);
lean_ctor_set(v_reuseFailAlloc_240_, 1, v___x_232_);
lean_ctor_set(v_reuseFailAlloc_240_, 2, v_scopes_216_);
lean_ctor_set(v_reuseFailAlloc_240_, 3, v_usedQuotCtxts_217_);
lean_ctor_set(v_reuseFailAlloc_240_, 4, v_nextMacroScope_218_);
lean_ctor_set(v_reuseFailAlloc_240_, 5, v_maxRecDepth_219_);
lean_ctor_set(v_reuseFailAlloc_240_, 6, v_ngen_220_);
lean_ctor_set(v_reuseFailAlloc_240_, 7, v_auxDeclNGen_221_);
lean_ctor_set(v_reuseFailAlloc_240_, 8, v_infoState_222_);
lean_ctor_set(v_reuseFailAlloc_240_, 9, v_traceState_223_);
lean_ctor_set(v_reuseFailAlloc_240_, 10, v_snapshotTasks_224_);
lean_ctor_set(v_reuseFailAlloc_240_, 11, v_prevLinterStates_225_);
v___x_234_ = v_reuseFailAlloc_240_;
goto v_reusejp_233_;
}
v_reusejp_233_:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_238_; 
v___x_235_ = lean_st_ref_set(v___y_203_, v___x_234_);
v___x_236_ = lean_box(0);
if (v_isShared_210_ == 0)
{
lean_ctor_set(v___x_209_, 0, v___x_236_);
v___x_238_ = v___x_209_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v___x_236_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
}
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
lean_dec(v_a_205_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_197_);
lean_dec_ref(v___y_196_);
v_a_243_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_206_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_206_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_248_; 
if (v_isShared_246_ == 0)
{
v___x_248_ = v___x_245_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_a_243_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
else
{
lean_object* v_a_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_258_; 
lean_dec(v___y_201_);
lean_dec_ref(v___y_197_);
lean_dec_ref(v___y_196_);
v_a_251_ = lean_ctor_get(v___x_204_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_204_);
if (v_isSharedCheck_258_ == 0)
{
v___x_253_ = v___x_204_;
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_a_251_);
lean_dec(v___x_204_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v___x_256_; 
if (v_isShared_254_ == 0)
{
v___x_256_ = v___x_253_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_a_251_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
v___jp_259_:
{
lean_object* v_fileName_265_; lean_object* v_fileMap_266_; uint8_t v_suppressElabErrors_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_286_; 
v_fileName_265_ = lean_ctor_get(v___y_192_, 0);
v_fileMap_266_ = lean_ctor_get(v___y_192_, 1);
v_suppressElabErrors_267_ = lean_ctor_get_uint8(v___y_192_, sizeof(void*)*10);
v___x_268_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_189_);
v___x_269_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg(v___x_268_, v___y_193_);
v_a_270_ = lean_ctor_get(v___x_269_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v___x_269_);
if (v_isSharedCheck_286_ == 0)
{
v___x_272_ = v___x_269_;
v_isShared_273_ = v_isSharedCheck_286_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___x_269_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_286_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
lean_inc_ref_n(v_fileMap_266_, 2);
v___x_274_ = l_Lean_FileMap_toPosition(v_fileMap_266_, v___y_261_);
lean_dec(v___y_261_);
v___x_275_ = l_Lean_FileMap_toPosition(v_fileMap_266_, v___y_264_);
lean_dec(v___y_264_);
v___x_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
v___x_277_ = ((lean_object*)(lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___closed__0));
if (v_suppressElabErrors_267_ == 0)
{
lean_del_object(v___x_272_);
v___y_196_ = v_a_270_;
v___y_197_ = v___x_274_;
v___y_198_ = v_fileName_265_;
v___y_199_ = v___x_277_;
v___y_200_ = v___y_262_;
v___y_201_ = v___x_276_;
v___y_202_ = v___y_263_;
v___y_203_ = v___y_193_;
goto v___jp_195_;
}
else
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___f_280_; uint8_t v___x_281_; 
v___x_278_ = lean_box(v___y_260_);
v___x_279_ = lean_box(v_suppressElabErrors_267_);
v___f_280_ = lean_alloc_closure((void*)(lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___lam__0___boxed), 3, 2);
lean_closure_set(v___f_280_, 0, v___x_278_);
lean_closure_set(v___f_280_, 1, v___x_279_);
lean_inc(v_a_270_);
v___x_281_ = l_Lean_MessageData_hasTag(v___f_280_, v_a_270_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_284_; 
lean_dec_ref_known(v___x_276_, 1);
lean_dec_ref(v___x_274_);
lean_dec(v_a_270_);
v___x_282_ = lean_box(0);
if (v_isShared_273_ == 0)
{
lean_ctor_set(v___x_272_, 0, v___x_282_);
v___x_284_ = v___x_272_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v___x_282_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
else
{
lean_del_object(v___x_272_);
v___y_196_ = v_a_270_;
v___y_197_ = v___x_274_;
v___y_198_ = v_fileName_265_;
v___y_199_ = v___x_277_;
v___y_200_ = v___y_262_;
v___y_201_ = v___x_276_;
v___y_202_ = v___y_263_;
v___y_203_ = v___y_193_;
goto v___jp_195_;
}
}
}
}
v___jp_287_:
{
lean_object* v___x_293_; 
v___x_293_ = l_Lean_Syntax_getTailPos_x3f(v___y_290_, v___y_289_);
lean_dec(v___y_290_);
if (lean_obj_tag(v___x_293_) == 0)
{
lean_inc(v___y_292_);
v___y_260_ = v___y_288_;
v___y_261_ = v___y_292_;
v___y_262_ = v___y_289_;
v___y_263_ = v___y_291_;
v___y_264_ = v___y_292_;
goto v___jp_259_;
}
else
{
lean_object* v_val_294_; 
v_val_294_ = lean_ctor_get(v___x_293_, 0);
lean_inc(v_val_294_);
lean_dec_ref_known(v___x_293_, 1);
v___y_260_ = v___y_288_;
v___y_261_ = v___y_292_;
v___y_262_ = v___y_289_;
v___y_263_ = v___y_291_;
v___y_264_ = v_val_294_;
goto v___jp_259_;
}
}
v___jp_295_:
{
lean_object* v___x_299_; 
v___x_299_ = l_Lean_Elab_Command_getRef___redArg(v___y_192_);
if (lean_obj_tag(v___x_299_) == 0)
{
lean_object* v_a_300_; lean_object* v_ref_301_; lean_object* v___x_302_; 
v_a_300_ = lean_ctor_get(v___x_299_, 0);
lean_inc(v_a_300_);
lean_dec_ref_known(v___x_299_, 1);
v_ref_301_ = l_Lean_replaceRef(v_ref_188_, v_a_300_);
lean_dec(v_a_300_);
v___x_302_ = l_Lean_Syntax_getPos_x3f(v_ref_301_, v___y_297_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v___x_303_; 
v___x_303_ = lean_unsigned_to_nat(0u);
v___y_288_ = v___y_296_;
v___y_289_ = v___y_297_;
v___y_290_ = v_ref_301_;
v___y_291_ = v___y_298_;
v___y_292_ = v___x_303_;
goto v___jp_287_;
}
else
{
lean_object* v_val_304_; 
v_val_304_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_val_304_);
lean_dec_ref_known(v___x_302_, 1);
v___y_288_ = v___y_296_;
v___y_289_ = v___y_297_;
v___y_290_ = v_ref_301_;
v___y_291_ = v___y_298_;
v___y_292_ = v_val_304_;
goto v___jp_287_;
}
}
else
{
lean_object* v_a_305_; lean_object* v___x_307_; uint8_t v_isShared_308_; uint8_t v_isSharedCheck_312_; 
lean_dec_ref(v_msgData_189_);
v_a_305_ = lean_ctor_get(v___x_299_, 0);
v_isSharedCheck_312_ = !lean_is_exclusive(v___x_299_);
if (v_isSharedCheck_312_ == 0)
{
v___x_307_ = v___x_299_;
v_isShared_308_ = v_isSharedCheck_312_;
goto v_resetjp_306_;
}
else
{
lean_inc(v_a_305_);
lean_dec(v___x_299_);
v___x_307_ = lean_box(0);
v_isShared_308_ = v_isSharedCheck_312_;
goto v_resetjp_306_;
}
v_resetjp_306_:
{
lean_object* v___x_310_; 
if (v_isShared_308_ == 0)
{
v___x_310_ = v___x_307_;
goto v_reusejp_309_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v_a_305_);
v___x_310_ = v_reuseFailAlloc_311_;
goto v_reusejp_309_;
}
v_reusejp_309_:
{
return v___x_310_;
}
}
}
}
v___jp_314_:
{
if (v___y_317_ == 0)
{
v___y_296_ = v___y_315_;
v___y_297_ = v___y_316_;
v___y_298_ = v_severity_190_;
goto v___jp_295_;
}
else
{
v___y_296_ = v___y_315_;
v___y_297_ = v___y_316_;
v___y_298_ = v___x_313_;
goto v___jp_295_;
}
}
v___jp_318_:
{
if (v___y_319_ == 0)
{
lean_object* v___x_320_; lean_object* v_scopes_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v_opts_324_; uint8_t v___x_325_; uint8_t v___x_326_; 
v___x_320_ = lean_st_ref_get(v___y_193_);
v_scopes_321_ = lean_ctor_get(v___x_320_, 2);
lean_inc(v_scopes_321_);
lean_dec(v___x_320_);
v___x_322_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_323_ = l_List_head_x21___redArg(v___x_322_, v_scopes_321_);
lean_dec(v_scopes_321_);
v_opts_324_ = lean_ctor_get(v___x_323_, 1);
lean_inc_ref(v_opts_324_);
lean_dec(v___x_323_);
v___x_325_ = 1;
v___x_326_ = l_Lean_instBEqMessageSeverity_beq(v_severity_190_, v___x_325_);
if (v___x_326_ == 0)
{
lean_dec_ref(v_opts_324_);
v___y_315_ = v___y_319_;
v___y_316_ = v___y_319_;
v___y_317_ = v___x_326_;
goto v___jp_314_;
}
else
{
lean_object* v___x_327_; uint8_t v___x_328_; 
v___x_327_ = l_Lean_warningAsError;
v___x_328_ = lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__8(v_opts_324_, v___x_327_);
lean_dec_ref(v_opts_324_);
v___y_315_ = v___y_319_;
v___y_316_ = v___y_319_;
v___y_317_ = v___x_328_;
goto v___jp_314_;
}
}
else
{
lean_object* v___x_329_; lean_object* v___x_330_; 
lean_dec_ref(v_msgData_189_);
v___x_329_ = lean_box(0);
v___x_330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
return v___x_330_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5___boxed(lean_object* v_ref_333_, lean_object* v_msgData_334_, lean_object* v_severity_335_, lean_object* v_isSilent_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_){
_start:
{
uint8_t v_severity_boxed_340_; uint8_t v_isSilent_boxed_341_; lean_object* v_res_342_; 
v_severity_boxed_340_ = lean_unbox(v_severity_335_);
v_isSilent_boxed_341_ = lean_unbox(v_isSilent_336_);
v_res_342_ = lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5(v_ref_333_, v_msgData_334_, v_severity_boxed_340_, v_isSilent_boxed_341_, v___y_337_, v___y_338_);
lean_dec(v___y_338_);
lean_dec_ref(v___y_337_);
lean_dec(v_ref_333_);
return v_res_342_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4(lean_object* v_msgData_343_, uint8_t v_severity_344_, uint8_t v_isSilent_345_, lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = l_Lean_Elab_Command_getRef___redArg(v___y_346_);
if (lean_obj_tag(v___x_349_) == 0)
{
lean_object* v_a_350_; lean_object* v___x_351_; 
v_a_350_ = lean_ctor_get(v___x_349_, 0);
lean_inc(v_a_350_);
lean_dec_ref_known(v___x_349_, 1);
v___x_351_ = lp_importGraph_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5(v_a_350_, v_msgData_343_, v_severity_344_, v_isSilent_345_, v___y_346_, v___y_347_);
lean_dec(v_a_350_);
return v___x_351_;
}
else
{
lean_object* v_a_352_; lean_object* v___x_354_; uint8_t v_isShared_355_; uint8_t v_isSharedCheck_359_; 
lean_dec_ref(v_msgData_343_);
v_a_352_ = lean_ctor_get(v___x_349_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v___x_349_);
if (v_isSharedCheck_359_ == 0)
{
v___x_354_ = v___x_349_;
v_isShared_355_ = v_isSharedCheck_359_;
goto v_resetjp_353_;
}
else
{
lean_inc(v_a_352_);
lean_dec(v___x_349_);
v___x_354_ = lean_box(0);
v_isShared_355_ = v_isSharedCheck_359_;
goto v_resetjp_353_;
}
v_resetjp_353_:
{
lean_object* v___x_357_; 
if (v_isShared_355_ == 0)
{
v___x_357_ = v___x_354_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v_a_352_);
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
LEAN_EXPORT lean_object* lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4___boxed(lean_object* v_msgData_360_, lean_object* v_severity_361_, lean_object* v_isSilent_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_){
_start:
{
uint8_t v_severity_boxed_366_; uint8_t v_isSilent_boxed_367_; lean_object* v_res_368_; 
v_severity_boxed_366_ = lean_unbox(v_severity_361_);
v_isSilent_boxed_367_ = lean_unbox(v_isSilent_362_);
v_res_368_ = lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4(v_msgData_360_, v_severity_boxed_366_, v_isSilent_boxed_367_, v___y_363_, v___y_364_);
lean_dec(v___y_364_);
lean_dec_ref(v___y_363_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3(lean_object* v_msgData_369_, lean_object* v___y_370_, lean_object* v___y_371_){
_start:
{
uint8_t v___x_373_; uint8_t v___x_374_; lean_object* v___x_375_; 
v___x_373_ = 0;
v___x_374_ = 0;
v___x_375_ = lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4(v_msgData_369_, v___x_373_, v___x_374_, v___y_370_, v___y_371_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3___boxed(lean_object* v_msgData_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3(v_msgData_376_, v___y_377_, v___y_378_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg(uint8_t v___x_381_, lean_object* v_hi_382_, lean_object* v_pivot_383_, lean_object* v_as_384_, lean_object* v_i_385_, lean_object* v_k_386_){
_start:
{
uint8_t v___x_387_; 
v___x_387_ = lean_nat_dec_lt(v_k_386_, v_hi_382_);
if (v___x_387_ == 0)
{
lean_object* v___x_388_; lean_object* v___x_389_; 
lean_dec(v_k_386_);
lean_dec(v_pivot_383_);
v___x_388_ = lean_array_fswap(v_as_384_, v_i_385_, v_hi_382_);
v___x_389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_389_, 0, v_i_385_);
lean_ctor_set(v___x_389_, 1, v___x_388_);
return v___x_389_;
}
else
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; uint8_t v___x_393_; 
v___x_390_ = lean_array_fget_borrowed(v_as_384_, v_k_386_);
lean_inc(v___x_390_);
v___x_391_ = l_Lean_Name_toString(v___x_390_, v___x_381_);
lean_inc(v_pivot_383_);
v___x_392_ = l_Lean_Name_toString(v_pivot_383_, v___x_381_);
v___x_393_ = lean_string_dec_lt(v___x_391_, v___x_392_);
lean_dec_ref(v___x_392_);
lean_dec_ref(v___x_391_);
if (v___x_393_ == 0)
{
lean_object* v___x_394_; lean_object* v___x_395_; 
v___x_394_ = lean_unsigned_to_nat(1u);
v___x_395_ = lean_nat_add(v_k_386_, v___x_394_);
lean_dec(v_k_386_);
v_k_386_ = v___x_395_;
goto _start;
}
else
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_397_ = lean_array_fswap(v_as_384_, v_i_385_, v_k_386_);
v___x_398_ = lean_unsigned_to_nat(1u);
v___x_399_ = lean_nat_add(v_i_385_, v___x_398_);
lean_dec(v_i_385_);
v___x_400_ = lean_nat_add(v_k_386_, v___x_398_);
lean_dec(v_k_386_);
v_as_384_ = v___x_397_;
v_i_385_ = v___x_399_;
v_k_386_ = v___x_400_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg___boxed(lean_object* v___x_402_, lean_object* v_hi_403_, lean_object* v_pivot_404_, lean_object* v_as_405_, lean_object* v_i_406_, lean_object* v_k_407_){
_start:
{
uint8_t v___x_3544__boxed_408_; lean_object* v_res_409_; 
v___x_3544__boxed_408_ = lean_unbox(v___x_402_);
v_res_409_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg(v___x_3544__boxed_408_, v_hi_403_, v_pivot_404_, v_as_405_, v_i_406_, v_k_407_);
lean_dec(v_hi_403_);
return v_res_409_;
}
}
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0(uint8_t v___x_410_, lean_object* v_x1_411_, lean_object* v_x2_412_){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; uint8_t v___x_415_; 
v___x_413_ = l_Lean_Name_toString(v_x1_411_, v___x_410_);
v___x_414_ = l_Lean_Name_toString(v_x2_412_, v___x_410_);
v___x_415_ = lean_string_dec_lt(v___x_413_, v___x_414_);
lean_dec_ref(v___x_414_);
lean_dec_ref(v___x_413_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0___boxed(lean_object* v___x_416_, lean_object* v_x1_417_, lean_object* v_x2_418_){
_start:
{
uint8_t v___x_3577__boxed_419_; uint8_t v_res_420_; lean_object* v_r_421_; 
v___x_3577__boxed_419_ = lean_unbox(v___x_416_);
v_res_420_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0(v___x_3577__boxed_419_, v_x1_417_, v_x2_418_);
v_r_421_ = lean_box(v_res_420_);
return v_r_421_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg(uint8_t v___x_422_, lean_object* v_n_423_, lean_object* v_as_424_, lean_object* v_lo_425_, lean_object* v_hi_426_){
_start:
{
lean_object* v___y_428_; uint8_t v___x_438_; 
v___x_438_ = lean_nat_dec_lt(v_lo_425_, v_hi_426_);
if (v___x_438_ == 0)
{
lean_dec(v_lo_425_);
return v_as_424_;
}
else
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v_mid_441_; lean_object* v___y_443_; lean_object* v___y_449_; lean_object* v___x_454_; lean_object* v___x_455_; uint8_t v___x_456_; 
v___x_439_ = lean_nat_add(v_lo_425_, v_hi_426_);
v___x_440_ = lean_unsigned_to_nat(1u);
v_mid_441_ = lean_nat_shiftr(v___x_439_, v___x_440_);
lean_dec(v___x_439_);
v___x_454_ = lean_array_fget_borrowed(v_as_424_, v_mid_441_);
v___x_455_ = lean_array_fget_borrowed(v_as_424_, v_lo_425_);
lean_inc(v___x_455_);
lean_inc(v___x_454_);
v___x_456_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0(v___x_422_, v___x_454_, v___x_455_);
if (v___x_456_ == 0)
{
v___y_449_ = v_as_424_;
goto v___jp_448_;
}
else
{
lean_object* v___x_457_; 
v___x_457_ = lean_array_fswap(v_as_424_, v_lo_425_, v_mid_441_);
v___y_449_ = v___x_457_;
goto v___jp_448_;
}
v___jp_442_:
{
lean_object* v___x_444_; lean_object* v___x_445_; uint8_t v___x_446_; 
v___x_444_ = lean_array_fget_borrowed(v___y_443_, v_mid_441_);
v___x_445_ = lean_array_fget_borrowed(v___y_443_, v_hi_426_);
lean_inc(v___x_445_);
lean_inc(v___x_444_);
v___x_446_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0(v___x_422_, v___x_444_, v___x_445_);
if (v___x_446_ == 0)
{
lean_dec(v_mid_441_);
v___y_428_ = v___y_443_;
goto v___jp_427_;
}
else
{
lean_object* v___x_447_; 
v___x_447_ = lean_array_fswap(v___y_443_, v_mid_441_, v_hi_426_);
lean_dec(v_mid_441_);
v___y_428_ = v___x_447_;
goto v___jp_427_;
}
}
v___jp_448_:
{
lean_object* v___x_450_; lean_object* v___x_451_; uint8_t v___x_452_; 
v___x_450_ = lean_array_fget_borrowed(v___y_449_, v_hi_426_);
v___x_451_ = lean_array_fget_borrowed(v___y_449_, v_lo_425_);
lean_inc(v___x_451_);
lean_inc(v___x_450_);
v___x_452_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___lam__0(v___x_422_, v___x_450_, v___x_451_);
if (v___x_452_ == 0)
{
v___y_443_ = v___y_449_;
goto v___jp_442_;
}
else
{
lean_object* v___x_453_; 
v___x_453_ = lean_array_fswap(v___y_449_, v_lo_425_, v_hi_426_);
v___y_443_ = v___x_453_;
goto v___jp_442_;
}
}
}
v___jp_427_:
{
lean_object* v_pivot_429_; lean_object* v___x_430_; lean_object* v_fst_431_; lean_object* v_snd_432_; uint8_t v___x_433_; 
v_pivot_429_ = lean_array_fget(v___y_428_, v_hi_426_);
lean_inc_n(v_lo_425_, 2);
v___x_430_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg(v___x_422_, v_hi_426_, v_pivot_429_, v___y_428_, v_lo_425_, v_lo_425_);
v_fst_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc(v_fst_431_);
v_snd_432_ = lean_ctor_get(v___x_430_, 1);
lean_inc(v_snd_432_);
lean_dec_ref(v___x_430_);
v___x_433_ = lean_nat_dec_le(v_hi_426_, v_fst_431_);
if (v___x_433_ == 0)
{
lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; 
v___x_434_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg(v___x_422_, v_n_423_, v_snd_432_, v_lo_425_, v_fst_431_);
v___x_435_ = lean_unsigned_to_nat(1u);
v___x_436_ = lean_nat_add(v_fst_431_, v___x_435_);
lean_dec(v_fst_431_);
v_as_424_ = v___x_434_;
v_lo_425_ = v___x_436_;
goto _start;
}
else
{
lean_dec(v_fst_431_);
lean_dec(v_lo_425_);
return v_snd_432_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg___boxed(lean_object* v___x_458_, lean_object* v_n_459_, lean_object* v_as_460_, lean_object* v_lo_461_, lean_object* v_hi_462_){
_start:
{
uint8_t v___x_3592__boxed_463_; lean_object* v_res_464_; 
v___x_3592__boxed_463_ = lean_unbox(v___x_458_);
v_res_464_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg(v___x_3592__boxed_463_, v_n_459_, v_as_460_, v_lo_461_, v_hi_462_);
lean_dec(v_hi_462_);
lean_dec(v_n_459_);
return v_res_464_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_List_foldl___at___00Std_Format_joinSep___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__2_spec__2(lean_object* v_x_465_, lean_object* v_x_466_, lean_object* v_x_467_){
_start:
{
if (lean_obj_tag(v_x_467_) == 0)
{
lean_dec(v_x_465_);
return v_x_466_;
}
else
{
lean_object* v_head_468_; lean_object* v_tail_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_479_; 
v_head_468_ = lean_ctor_get(v_x_467_, 0);
v_tail_469_ = lean_ctor_get(v_x_467_, 1);
v_isSharedCheck_479_ = !lean_is_exclusive(v_x_467_);
if (v_isSharedCheck_479_ == 0)
{
v___x_471_ = v_x_467_;
v_isShared_472_ = v_isSharedCheck_479_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_tail_469_);
lean_inc(v_head_468_);
lean_dec(v_x_467_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_479_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_474_; 
lean_inc(v_x_465_);
if (v_isShared_472_ == 0)
{
lean_ctor_set_tag(v___x_471_, 5);
lean_ctor_set(v___x_471_, 1, v_x_465_);
lean_ctor_set(v___x_471_, 0, v_x_466_);
v___x_474_ = v___x_471_;
goto v_reusejp_473_;
}
else
{
lean_object* v_reuseFailAlloc_478_; 
v_reuseFailAlloc_478_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_478_, 0, v_x_466_);
lean_ctor_set(v_reuseFailAlloc_478_, 1, v_x_465_);
v___x_474_ = v_reuseFailAlloc_478_;
goto v_reusejp_473_;
}
v_reusejp_473_:
{
lean_object* v___x_475_; lean_object* v___x_476_; 
v___x_475_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_475_, 0, v_head_468_);
v___x_476_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_476_, 0, v___x_474_);
lean_ctor_set(v___x_476_, 1, v___x_475_);
v_x_466_ = v___x_476_;
v_x_467_ = v_tail_469_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_Format_joinSep___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__2(lean_object* v_x_480_, lean_object* v_x_481_){
_start:
{
if (lean_obj_tag(v_x_480_) == 0)
{
lean_object* v___x_482_; 
lean_dec(v_x_481_);
v___x_482_ = lean_box(0);
return v___x_482_;
}
else
{
lean_object* v_tail_483_; 
v_tail_483_ = lean_ctor_get(v_x_480_, 1);
if (lean_obj_tag(v_tail_483_) == 0)
{
lean_object* v_head_484_; lean_object* v___x_485_; 
lean_dec(v_x_481_);
v_head_484_ = lean_ctor_get(v_x_480_, 0);
lean_inc(v_head_484_);
lean_dec_ref_known(v_x_480_, 2);
v___x_485_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_485_, 0, v_head_484_);
return v___x_485_;
}
else
{
lean_object* v_head_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
lean_inc(v_tail_483_);
v_head_486_ = lean_ctor_get(v_x_480_, 0);
lean_inc(v_head_486_);
lean_dec_ref_known(v_x_480_, 2);
v___x_487_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_487_, 0, v_head_486_);
v___x_488_ = lp_importGraph_List_foldl___at___00Std_Format_joinSep___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__2_spec__2(v_x_481_, v___x_487_, v_tail_483_);
return v___x_488_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1(uint8_t v___x_490_, lean_object* v_a_491_, lean_object* v_a_492_){
_start:
{
if (lean_obj_tag(v_a_491_) == 0)
{
lean_object* v___x_493_; 
v___x_493_ = l_List_reverse___redArg(v_a_492_);
return v___x_493_;
}
else
{
lean_object* v_head_494_; lean_object* v_tail_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_506_; 
v_head_494_ = lean_ctor_get(v_a_491_, 0);
v_tail_495_ = lean_ctor_get(v_a_491_, 1);
v_isSharedCheck_506_ = !lean_is_exclusive(v_a_491_);
if (v_isSharedCheck_506_ == 0)
{
v___x_497_ = v_a_491_;
v_isShared_498_ = v_isSharedCheck_506_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_tail_495_);
lean_inc(v_head_494_);
lean_dec(v_a_491_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_506_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_503_; 
v___x_499_ = ((lean_object*)(lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1___closed__0));
v___x_500_ = l_Lean_Name_toString(v_head_494_, v___x_490_);
v___x_501_ = lean_string_append(v___x_499_, v___x_500_);
lean_dec_ref(v___x_500_);
if (v_isShared_498_ == 0)
{
lean_ctor_set(v___x_497_, 1, v_a_492_);
lean_ctor_set(v___x_497_, 0, v___x_501_);
v___x_503_ = v___x_497_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v___x_501_);
lean_ctor_set(v_reuseFailAlloc_505_, 1, v_a_492_);
v___x_503_ = v_reuseFailAlloc_505_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
v_a_491_ = v_tail_495_;
v_a_492_ = v___x_503_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1___boxed(lean_object* v___x_507_, lean_object* v_a_508_, lean_object* v_a_509_){
_start:
{
uint8_t v___x_3703__boxed_510_; lean_object* v_res_511_; 
v___x_3703__boxed_510_ = lean_unbox(v___x_507_);
v_res_511_ = lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1(v___x_3703__boxed_510_, v_a_508_, v_a_509_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1(lean_object* v_x_515_, lean_object* v_a_516_, lean_object* v_a_517_){
_start:
{
lean_object* v___x_519_; uint8_t v___x_520_; 
v___x_519_ = ((lean_object*)(lp_importGraph_command_x23min__imports___closed__1));
v___x_520_ = l_Lean_Syntax_isOfKind(v_x_515_, v___x_519_);
if (v___x_520_ == 0)
{
lean_object* v___x_521_; 
v___x_521_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg();
return v___x_521_;
}
else
{
lean_object* v___x_522_; lean_object* v___y_524_; lean_object* v_env_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___y_536_; lean_object* v___y_537_; lean_object* v___x_539_; uint8_t v___x_540_; 
v___x_522_ = lean_st_ref_get(v_a_517_);
v_env_532_ = lean_ctor_get(v___x_522_, 0);
lean_inc_ref(v_env_532_);
lean_dec(v___x_522_);
v___x_533_ = lp_importGraph_Lean_Environment_minimalRequiredModules(v_env_532_);
v___x_534_ = lean_array_get_size(v___x_533_);
v___x_539_ = lean_unsigned_to_nat(0u);
v___x_540_ = lean_nat_dec_eq(v___x_534_, v___x_539_);
if (v___x_540_ == 0)
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___y_544_; uint8_t v___x_546_; 
v___x_541_ = lean_unsigned_to_nat(1u);
v___x_542_ = lean_nat_sub(v___x_534_, v___x_541_);
v___x_546_ = lean_nat_dec_le(v___x_539_, v___x_542_);
if (v___x_546_ == 0)
{
lean_inc(v___x_542_);
v___y_544_ = v___x_542_;
goto v___jp_543_;
}
else
{
v___y_544_ = v___x_539_;
goto v___jp_543_;
}
v___jp_543_:
{
uint8_t v___x_545_; 
v___x_545_ = lean_nat_dec_le(v___y_544_, v___x_542_);
if (v___x_545_ == 0)
{
lean_dec(v___x_542_);
lean_inc(v___y_544_);
v___y_536_ = v___y_544_;
v___y_537_ = v___y_544_;
goto v___jp_535_;
}
else
{
v___y_536_ = v___y_544_;
v___y_537_ = v___x_542_;
goto v___jp_535_;
}
}
}
else
{
v___y_524_ = v___x_533_;
goto v___jp_523_;
}
v___jp_523_:
{
lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; 
v___x_525_ = lean_array_to_list(v___y_524_);
v___x_526_ = lean_box(0);
v___x_527_ = lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__1(v___x_520_, v___x_525_, v___x_526_);
v___x_528_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___closed__1));
v___x_529_ = lp_importGraph_Std_Format_joinSep___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__2(v___x_527_, v___x_528_);
v___x_530_ = l_Lean_MessageData_ofFormat(v___x_529_);
v___x_531_ = lp_importGraph_Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3(v___x_530_, v_a_516_, v_a_517_);
return v___x_531_;
}
v___jp_535_:
{
lean_object* v___x_538_; 
v___x_538_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg(v___x_520_, v___x_534_, v___x_533_, v___y_536_, v___y_537_);
lean_dec(v___y_537_);
v___y_524_ = v___x_538_;
goto v___jp_523_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1___boxed(lean_object* v_x_547_, lean_object* v_a_548_, lean_object* v_a_549_, lean_object* v_a_550_){
_start:
{
lean_object* v_res_551_; 
v_res_551_ = lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1(v_x_547_, v_a_548_, v_a_549_);
lean_dec(v_a_549_);
lean_dec_ref(v_a_548_);
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4(uint8_t v___x_552_, lean_object* v_n_553_, lean_object* v_as_554_, lean_object* v_lo_555_, lean_object* v_hi_556_, lean_object* v_w_557_, lean_object* v_hlo_558_, lean_object* v_hhi_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___redArg(v___x_552_, v_n_553_, v_as_554_, v_lo_555_, v_hi_556_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4___boxed(lean_object* v___x_561_, lean_object* v_n_562_, lean_object* v_as_563_, lean_object* v_lo_564_, lean_object* v_hi_565_, lean_object* v_w_566_, lean_object* v_hlo_567_, lean_object* v_hhi_568_){
_start:
{
uint8_t v___x_3813__boxed_569_; lean_object* v_res_570_; 
v___x_3813__boxed_569_ = lean_unbox(v___x_561_);
v_res_570_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4(v___x_3813__boxed_569_, v_n_562_, v_as_563_, v_lo_564_, v_hi_565_, v_w_566_, v_hlo_567_, v_hhi_568_);
lean_dec(v_hi_565_);
lean_dec(v_n_562_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6(uint8_t v___x_571_, lean_object* v_n_572_, lean_object* v_lo_573_, lean_object* v_hi_574_, lean_object* v_hhi_575_, lean_object* v_pivot_576_, lean_object* v_as_577_, lean_object* v_i_578_, lean_object* v_k_579_, lean_object* v_ilo_580_, lean_object* v_ik_581_, lean_object* v_w_582_){
_start:
{
lean_object* v___x_583_; 
v___x_583_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___redArg(v___x_571_, v_hi_574_, v_pivot_576_, v_as_577_, v_i_578_, v_k_579_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6___boxed(lean_object* v___x_584_, lean_object* v_n_585_, lean_object* v_lo_586_, lean_object* v_hi_587_, lean_object* v_hhi_588_, lean_object* v_pivot_589_, lean_object* v_as_590_, lean_object* v_i_591_, lean_object* v_k_592_, lean_object* v_ilo_593_, lean_object* v_ik_594_, lean_object* v_w_595_){
_start:
{
uint8_t v___x_3818__boxed_596_; lean_object* v_res_597_; 
v___x_3818__boxed_596_ = lean_unbox(v___x_584_);
v_res_597_ = lp_importGraph___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__4_spec__6(v___x_3818__boxed_596_, v_n_585_, v_lo_586_, v_hi_587_, v_hhi_588_, v_pivot_589_, v_as_590_, v_i_591_, v_k_592_, v_ilo_593_, v_ik_594_, v_w_595_);
lean_dec(v_hi_587_);
lean_dec(v_lo_586_);
lean_dec(v_n_585_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7(lean_object* v_msgData_598_, lean_object* v___y_599_, lean_object* v___y_600_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___redArg(v_msgData_598_, v___y_600_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7___boxed(lean_object* v_msgData_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_){
_start:
{
lean_object* v_res_607_; 
v_res_607_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4_spec__5_spec__7(v_msgData_603_, v___y_604_, v___y_605_);
lean_dec(v___y_605_);
lean_dec_ref(v___y_604_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg(lean_object* v___y_619_){
_start:
{
lean_object* v___x_621_; lean_object* v_env_622_; lean_object* v___x_623_; lean_object* v_mainModule_624_; lean_object* v___x_625_; 
v___x_621_ = lean_st_ref_get(v___y_619_);
v_env_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc_ref(v_env_622_);
lean_dec(v___x_621_);
v___x_623_ = l_Lean_Environment_header(v_env_622_);
lean_dec_ref(v_env_622_);
v_mainModule_624_ = lean_ctor_get(v___x_623_, 0);
lean_inc(v_mainModule_624_);
lean_dec_ref(v___x_623_);
v___x_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_625_, 0, v_mainModule_624_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg___boxed(lean_object* v___y_626_, lean_object* v___y_627_){
_start:
{
lean_object* v_res_628_; 
v_res_628_ = lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg(v___y_626_);
lean_dec(v___y_626_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1(lean_object* v___y_629_, lean_object* v___y_630_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg(v___y_630_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___boxed(lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1(v___y_633_, v___y_634_);
lean_dec(v___y_634_);
lean_dec_ref(v___y_633_);
return v_res_636_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logWarning___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__0(lean_object* v_msgData_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
uint8_t v___x_641_; uint8_t v___x_642_; lean_object* v___x_643_; 
v___x_641_ = 1;
v___x_642_ = 0;
v___x_643_ = lp_importGraph_Lean_log___at___00Lean_logInfo___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__3_spec__4(v_msgData_637_, v___x_641_, v___x_642_, v___y_638_, v___y_639_);
return v___x_643_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logWarning___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__0___boxed(lean_object* v_msgData_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_importGraph_Lean_logWarning___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__0(v_msgData_644_, v___y_645_, v___y_646_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
return v_res_648_;
}
}
static lean_object* _init_lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__1(void){
_start:
{
lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_650_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__0));
v___x_651_ = l_Lean_stringToMessageData(v___x_650_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1(lean_object* v_x_652_, lean_object* v_a_653_, lean_object* v_a_654_){
_start:
{
lean_object* v___x_656_; uint8_t v___x_657_; 
v___x_656_ = ((lean_object*)(lp_importGraph_command_x23minimize__imports___closed__1));
v___x_657_ = l_Lean_Syntax_isOfKind(v_x_652_, v___x_656_);
if (v___x_657_ == 0)
{
lean_object* v___x_658_; 
v___x_658_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23min__imports__1_spec__0___redArg();
return v___x_658_;
}
else
{
lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_659_ = lean_obj_once(&lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__1, &lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__1_once, _init_lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___closed__1);
v___x_660_ = lp_importGraph_Lean_logWarning___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__0(v___x_659_, v_a_653_, v_a_654_);
if (lean_obj_tag(v___x_660_) == 0)
{
lean_object* v___x_661_; 
lean_dec_ref_known(v___x_660_, 1);
v___x_661_ = l_Lean_Elab_Command_getRef___redArg(v_a_653_);
if (lean_obj_tag(v___x_661_) == 0)
{
lean_object* v_a_662_; lean_object* v___x_663_; 
v_a_662_ = lean_ctor_get(v___x_661_, 0);
lean_inc(v_a_662_);
lean_dec_ref_known(v___x_661_, 1);
v___x_663_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_653_);
if (lean_obj_tag(v___x_663_) == 0)
{
lean_object* v_quotContext_x3f_664_; uint8_t v___x_665_; lean_object* v___x_666_; 
lean_dec_ref_known(v___x_663_, 1);
v_quotContext_x3f_664_ = lean_ctor_get(v_a_653_, 5);
v___x_665_ = 0;
v___x_666_ = l_Lean_SourceInfo_fromRef(v_a_662_, v___x_665_);
lean_dec(v_a_662_);
if (lean_obj_tag(v_quotContext_x3f_664_) == 0)
{
lean_object* v___x_673_; 
v___x_673_ = lp_importGraph_Lean_getMainModule___at___00__aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1_spec__1___redArg(v_a_654_);
lean_dec_ref(v___x_673_);
goto v___jp_667_;
}
else
{
goto v___jp_667_;
}
v___jp_667_:
{
lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v___x_668_ = ((lean_object*)(lp_importGraph_command_x23min__imports___closed__1));
v___x_669_ = ((lean_object*)(lp_importGraph_command_x23min__imports___closed__2));
lean_inc(v___x_666_);
v___x_670_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_670_, 0, v___x_666_);
lean_ctor_set(v___x_670_, 1, v___x_669_);
v___x_671_ = l_Lean_Syntax_node1(v___x_666_, v___x_668_, v___x_670_);
v___x_672_ = l_Lean_Elab_Command_elabCommand(v___x_671_, v_a_653_, v_a_654_);
return v___x_672_;
}
}
else
{
lean_object* v_a_674_; lean_object* v___x_676_; uint8_t v_isShared_677_; uint8_t v_isSharedCheck_681_; 
lean_dec(v_a_662_);
v_a_674_ = lean_ctor_get(v___x_663_, 0);
v_isSharedCheck_681_ = !lean_is_exclusive(v___x_663_);
if (v_isSharedCheck_681_ == 0)
{
v___x_676_ = v___x_663_;
v_isShared_677_ = v_isSharedCheck_681_;
goto v_resetjp_675_;
}
else
{
lean_inc(v_a_674_);
lean_dec(v___x_663_);
v___x_676_ = lean_box(0);
v_isShared_677_ = v_isSharedCheck_681_;
goto v_resetjp_675_;
}
v_resetjp_675_:
{
lean_object* v___x_679_; 
if (v_isShared_677_ == 0)
{
v___x_679_ = v___x_676_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_680_; 
v_reuseFailAlloc_680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_680_, 0, v_a_674_);
v___x_679_ = v_reuseFailAlloc_680_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
return v___x_679_;
}
}
}
}
else
{
lean_object* v_a_682_; lean_object* v___x_684_; uint8_t v_isShared_685_; uint8_t v_isSharedCheck_689_; 
v_a_682_ = lean_ctor_get(v___x_661_, 0);
v_isSharedCheck_689_ = !lean_is_exclusive(v___x_661_);
if (v_isSharedCheck_689_ == 0)
{
v___x_684_ = v___x_661_;
v_isShared_685_ = v_isSharedCheck_689_;
goto v_resetjp_683_;
}
else
{
lean_inc(v_a_682_);
lean_dec(v___x_661_);
v___x_684_ = lean_box(0);
v_isShared_685_ = v_isSharedCheck_689_;
goto v_resetjp_683_;
}
v_resetjp_683_:
{
lean_object* v___x_687_; 
if (v_isShared_685_ == 0)
{
v___x_687_ = v___x_684_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v_a_682_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
}
}
else
{
return v___x_660_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1___boxed(lean_object* v_x_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_importGraph___aux__ImportGraph__Tools__MinImports______elabRules__command_x23minimize__imports__1(v_x_690_, v_a_691_, v_a_692_);
lean_dec(v_a_692_);
lean_dec_ref(v_a_691_);
return v_res_694_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Tools_MinImports(uint8_t builtin) {
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
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_RequiredModules(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Tools_MinImports(uint8_t builtin) {
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
res = runtime_initialize_importGraph_ImportGraph_Imports_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Widget_UserWidget(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_RequiredModules(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Tools_MinImports(uint8_t builtin) {
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
res = initialize_importGraph_ImportGraph_Imports_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Tools_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Tools_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Tools_MinImports(builtin);
}
#ifdef __cplusplus
}
#endif
