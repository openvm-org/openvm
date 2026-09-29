// Lean compiler output
// Module: Mathlib.Util.WhatsNew
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_privateToUserName_x3f(lean_object*);
uint8_t l_Lean_isProtected(lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Environment_constants(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
extern lean_object* l_Lean_persistentEnvExtensionsRef;
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
extern lean_object* l_Lean_instInhabitedEnvExtensionState;
lean_object* l_Lean_instInhabitedPersistentEnvExtensionState___redArg(lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_ptr_addr(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "unknown identifier '"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ".{"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__5_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__3_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "private "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "protected "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__9_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "unsafe "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__12_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__14;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "partial "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__17;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " :="};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "inductive"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "constructors:"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__1_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "axiom"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "theorem"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "constant"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Quotient primitive"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constructor"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "recursor"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "-- "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " extension: "};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " new entries"};
static const lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_WhatsNew_diffExtension_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__3;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "no new constants"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "WhatsNew"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "command#whats_newIn__"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(209, 241, 72, 170, 222, 214, 76, 134)}};
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 19, 36, 27, 203, 85, 95, 242)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "#whats_new "};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "in"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ppLine"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(117, 61, 38, 245, 158, 59, 171, 58)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__12_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__20 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__17_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__21 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__22 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn____ = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__22_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "oldStx"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(209, 241, 72, 170, 222, 214, 76, 134)}};
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(117, 188, 255, 197, 237, 102, 126, 115)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "whatsnew "};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_WhatsNew_oldStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_WhatsNew_oldStx = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew_oldStx___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "#whats_new"};
static const lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lean_box(1);
v___x_2_ = l_Lean_MessageData_ofFormat(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__3(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_6_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__2));
v___x_7_ = l_Lean_MessageData_ofFormat(v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3(lean_object* v_x_8_, lean_object* v_x_9_){
_start:
{
if (lean_obj_tag(v_x_9_) == 0)
{
return v_x_8_;
}
else
{
lean_object* v_head_10_; lean_object* v_tail_11_; lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_33_; 
v_head_10_ = lean_ctor_get(v_x_9_, 0);
v_tail_11_ = lean_ctor_get(v_x_9_, 1);
v_isSharedCheck_33_ = !lean_is_exclusive(v_x_9_);
if (v_isSharedCheck_33_ == 0)
{
v___x_13_ = v_x_9_;
v_isShared_14_ = v_isSharedCheck_33_;
goto v_resetjp_12_;
}
else
{
lean_inc(v_tail_11_);
lean_inc(v_head_10_);
lean_dec(v_x_9_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_33_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v_before_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_31_; 
v_before_15_ = lean_ctor_get(v_head_10_, 0);
v_isSharedCheck_31_ = !lean_is_exclusive(v_head_10_);
if (v_isSharedCheck_31_ == 0)
{
lean_object* v_unused_32_; 
v_unused_32_ = lean_ctor_get(v_head_10_, 1);
lean_dec(v_unused_32_);
v___x_17_ = v_head_10_;
v_isShared_18_ = v_isSharedCheck_31_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_before_15_);
lean_dec(v_head_10_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_31_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; lean_object* v___x_21_; 
v___x_19_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_18_ == 0)
{
lean_ctor_set_tag(v___x_17_, 7);
lean_ctor_set(v___x_17_, 1, v___x_19_);
lean_ctor_set(v___x_17_, 0, v_x_8_);
v___x_21_ = v___x_17_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_30_; 
v_reuseFailAlloc_30_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_30_, 0, v_x_8_);
lean_ctor_set(v_reuseFailAlloc_30_, 1, v___x_19_);
v___x_21_ = v_reuseFailAlloc_30_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
lean_object* v___x_22_; lean_object* v___x_24_; 
v___x_22_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__3);
if (v_isShared_14_ == 0)
{
lean_ctor_set_tag(v___x_13_, 7);
lean_ctor_set(v___x_13_, 1, v___x_22_);
lean_ctor_set(v___x_13_, 0, v___x_21_);
v___x_24_ = v___x_13_;
goto v_reusejp_23_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v___x_21_);
lean_ctor_set(v_reuseFailAlloc_29_, 1, v___x_22_);
v___x_24_ = v_reuseFailAlloc_29_;
goto v_reusejp_23_;
}
v_reusejp_23_:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = l_Lean_MessageData_ofSyntax(v_before_15_);
v___x_26_ = l_Lean_indentD(v___x_25_);
v___x_27_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_27_, 0, v___x_24_);
lean_ctor_set(v___x_27_, 1, v___x_26_);
v_x_8_ = v___x_27_;
v_x_9_ = v_tail_11_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2(lean_object* v_opts_34_, lean_object* v_opt_35_){
_start:
{
lean_object* v_name_36_; lean_object* v_defValue_37_; lean_object* v_map_38_; lean_object* v___x_39_; 
v_name_36_ = lean_ctor_get(v_opt_35_, 0);
v_defValue_37_ = lean_ctor_get(v_opt_35_, 1);
v_map_38_ = lean_ctor_get(v_opts_34_, 0);
v___x_39_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_38_, v_name_36_);
if (lean_obj_tag(v___x_39_) == 0)
{
uint8_t v___x_40_; 
v___x_40_ = lean_unbox(v_defValue_37_);
return v___x_40_;
}
else
{
lean_object* v_val_41_; 
v_val_41_ = lean_ctor_get(v___x_39_, 0);
lean_inc(v_val_41_);
lean_dec_ref_known(v___x_39_, 1);
if (lean_obj_tag(v_val_41_) == 1)
{
uint8_t v_v_42_; 
v_v_42_ = lean_ctor_get_uint8(v_val_41_, 0);
lean_dec_ref_known(v_val_41_, 0);
return v_v_42_;
}
else
{
uint8_t v___x_43_; 
lean_dec(v_val_41_);
v___x_43_ = lean_unbox(v_defValue_37_);
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2___boxed(lean_object* v_opts_44_, lean_object* v_opt_45_){
_start:
{
uint8_t v_res_46_; lean_object* v_r_47_; 
v_res_46_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2(v_opts_44_, v_opt_45_);
lean_dec_ref(v_opt_45_);
lean_dec_ref(v_opts_44_);
v_r_47_ = lean_box(v_res_46_);
return v_r_47_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__1));
v___x_52_ = l_Lean_MessageData_ofFormat(v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg(lean_object* v_msgData_53_, lean_object* v_macroStack_54_, lean_object* v___y_55_){
_start:
{
lean_object* v___x_57_; lean_object* v_scopes_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v_opts_61_; lean_object* v___x_62_; uint8_t v___x_63_; 
v___x_57_ = lean_st_ref_get(v___y_55_);
v_scopes_58_ = lean_ctor_get(v___x_57_, 2);
lean_inc(v_scopes_58_);
lean_dec(v___x_57_);
v___x_59_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_60_ = l_List_head_x21___redArg(v___x_59_, v_scopes_58_);
lean_dec(v_scopes_58_);
v_opts_61_ = lean_ctor_get(v___x_60_, 1);
lean_inc_ref(v_opts_61_);
lean_dec(v___x_60_);
v___x_62_ = l_Lean_Elab_pp_macroStack;
v___x_63_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2(v_opts_61_, v___x_62_);
lean_dec_ref(v_opts_61_);
if (v___x_63_ == 0)
{
lean_object* v___x_64_; 
lean_dec(v_macroStack_54_);
v___x_64_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_64_, 0, v_msgData_53_);
return v___x_64_;
}
else
{
if (lean_obj_tag(v_macroStack_54_) == 0)
{
lean_object* v___x_65_; 
v___x_65_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_65_, 0, v_msgData_53_);
return v___x_65_;
}
else
{
lean_object* v_head_66_; lean_object* v_after_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_82_; 
v_head_66_ = lean_ctor_get(v_macroStack_54_, 0);
lean_inc(v_head_66_);
v_after_67_ = lean_ctor_get(v_head_66_, 1);
v_isSharedCheck_82_ = !lean_is_exclusive(v_head_66_);
if (v_isSharedCheck_82_ == 0)
{
lean_object* v_unused_83_; 
v_unused_83_ = lean_ctor_get(v_head_66_, 0);
lean_dec(v_unused_83_);
v___x_69_ = v_head_66_;
v_isShared_70_ = v_isSharedCheck_82_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_after_67_);
lean_dec(v_head_66_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_82_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v___x_71_; lean_object* v___x_73_; 
v___x_71_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_70_ == 0)
{
lean_ctor_set_tag(v___x_69_, 7);
lean_ctor_set(v___x_69_, 1, v___x_71_);
lean_ctor_set(v___x_69_, 0, v_msgData_53_);
v___x_73_ = v___x_69_;
goto v_reusejp_72_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v_msgData_53_);
lean_ctor_set(v_reuseFailAlloc_81_, 1, v___x_71_);
v___x_73_ = v_reuseFailAlloc_81_;
goto v_reusejp_72_;
}
v_reusejp_72_:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v_msgData_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_74_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___closed__2);
v___x_75_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_73_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
v___x_76_ = l_Lean_MessageData_ofSyntax(v_after_67_);
v___x_77_ = l_Lean_indentD(v___x_76_);
v_msgData_78_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_78_, 0, v___x_75_);
lean_ctor_set(v_msgData_78_, 1, v___x_77_);
v___x_79_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3(v_msgData_78_, v_macroStack_54_);
v___x_80_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
return v___x_80_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg___boxed(lean_object* v_msgData_84_, lean_object* v_macroStack_85_, lean_object* v___y_86_, lean_object* v___y_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg(v_msgData_84_, v_macroStack_85_, v___y_86_);
lean_dec(v___y_86_);
return v_res_88_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_89_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__0);
v___x_91_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1);
v___x_93_ = lean_unsigned_to_nat(0u);
v___x_94_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
lean_ctor_set(v___x_94_, 2, v___x_93_);
lean_ctor_set(v___x_94_, 3, v___x_93_);
lean_ctor_set(v___x_94_, 4, v___x_92_);
lean_ctor_set(v___x_94_, 5, v___x_92_);
lean_ctor_set(v___x_94_, 6, v___x_92_);
lean_ctor_set(v___x_94_, 7, v___x_92_);
lean_ctor_set(v___x_94_, 8, v___x_92_);
lean_ctor_set(v___x_94_, 9, v___x_92_);
return v___x_94_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_unsigned_to_nat(32u);
v___x_96_ = lean_mk_empty_array_with_capacity(v___x_95_);
v___x_97_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__4(void){
_start:
{
size_t v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_98_ = ((size_t)5ULL);
v___x_99_ = lean_unsigned_to_nat(0u);
v___x_100_ = lean_unsigned_to_nat(32u);
v___x_101_ = lean_mk_empty_array_with_capacity(v___x_100_);
v___x_102_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__3);
v___x_103_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v___x_101_);
lean_ctor_set(v___x_103_, 2, v___x_99_);
lean_ctor_set(v___x_103_, 3, v___x_99_);
lean_ctor_set_usize(v___x_103_, 4, v___x_98_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_104_ = lean_box(1);
v___x_105_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__4);
v___x_106_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__1);
v___x_107_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v___x_105_);
lean_ctor_set(v___x_107_, 2, v___x_104_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg(lean_object* v_msgData_108_, lean_object* v___y_109_){
_start:
{
lean_object* v___x_111_; lean_object* v_env_112_; lean_object* v___x_113_; lean_object* v_scopes_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v_opts_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_111_ = lean_st_ref_get(v___y_109_);
v_env_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc_ref(v_env_112_);
lean_dec(v___x_111_);
v___x_113_ = lean_st_ref_get(v___y_109_);
v_scopes_114_ = lean_ctor_get(v___x_113_, 2);
lean_inc(v_scopes_114_);
lean_dec(v___x_113_);
v___x_115_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_116_ = l_List_head_x21___redArg(v___x_115_, v_scopes_114_);
lean_dec(v_scopes_114_);
v_opts_117_ = lean_ctor_get(v___x_116_, 1);
lean_inc_ref(v_opts_117_);
lean_dec(v___x_116_);
v___x_118_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2);
v___x_119_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5);
v___x_120_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_120_, 0, v_env_112_);
lean_ctor_set(v___x_120_, 1, v___x_118_);
lean_ctor_set(v___x_120_, 2, v___x_119_);
lean_ctor_set(v___x_120_, 3, v_opts_117_);
v___x_121_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v_msgData_108_);
v___x_122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___boxed(lean_object* v_msgData_123_, lean_object* v___y_124_, lean_object* v___y_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg(v_msgData_123_, v___y_124_);
lean_dec(v___y_124_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg(lean_object* v_msg_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = l_Lean_Elab_Command_getRef___redArg(v___y_128_);
if (lean_obj_tag(v___x_131_) == 0)
{
lean_object* v_a_132_; lean_object* v_macroStack_133_; lean_object* v___x_134_; lean_object* v_a_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_146_; 
v_a_132_ = lean_ctor_get(v___x_131_, 0);
lean_inc(v_a_132_);
lean_dec_ref_known(v___x_131_, 1);
v_macroStack_133_ = lean_ctor_get(v___y_128_, 4);
v___x_134_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg(v_msg_127_, v___y_129_);
v_a_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_a_135_);
lean_dec_ref(v___x_134_);
v___x_136_ = l_Lean_Elab_getBetterRef(v_a_132_, v_macroStack_133_);
lean_dec(v_a_132_);
lean_inc(v_macroStack_133_);
v___x_137_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg(v_a_135_, v_macroStack_133_, v___y_129_);
v_a_138_ = lean_ctor_get(v___x_137_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_137_);
if (v_isSharedCheck_146_ == 0)
{
v___x_140_ = v___x_137_;
v_isShared_141_ = v_isSharedCheck_146_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_137_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_146_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_142_; lean_object* v___x_144_; 
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_136_);
lean_ctor_set(v___x_142_, 1, v_a_138_);
if (v_isShared_141_ == 0)
{
lean_ctor_set_tag(v___x_140_, 1);
lean_ctor_set(v___x_140_, 0, v___x_142_);
v___x_144_ = v___x_140_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v___x_142_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
else
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
lean_dec_ref(v_msg_127_);
v_a_147_ = lean_ctor_get(v___x_131_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_131_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_131_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_131_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_a_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg___boxed(lean_object* v_msg_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg(v_msg_155_, v___y_156_, v___y_157_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
return v_res_159_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__1(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__0));
v___x_162_ = l_Lean_stringToMessageData(v___x_161_);
return v___x_162_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__3(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__2));
v___x_165_ = l_Lean_stringToMessageData(v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId(lean_object* v_id_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_170_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__1, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__1_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__1);
v___x_171_ = lean_box(0);
v___x_172_ = l_Lean_mkConst(v_id_166_, v___x_171_);
v___x_173_ = l_Lean_MessageData_ofExpr(v___x_172_);
v___x_174_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_170_);
lean_ctor_set(v___x_174_, 1, v___x_173_);
v___x_175_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__3, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__3_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___closed__3);
v___x_176_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_174_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
v___x_177_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg(v___x_176_, v_a_167_, v_a_168_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId___boxed(lean_object* v_id_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId(v_id_178_, v_a_179_, v_a_180_);
lean_dec(v_a_180_);
lean_dec_ref(v_a_179_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0(lean_object* v_msgData_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg(v_msgData_183_, v___y_185_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___boxed(lean_object* v_msgData_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0(v_msgData_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0(lean_object* v_00_u03b1_193_, lean_object* v_msg_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___redArg(v_msg_194_, v___y_195_, v___y_196_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0___boxed(lean_object* v_00_u03b1_199_, lean_object* v_msg_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0(v_00_u03b1_199_, v_msg_200_, v___y_201_, v___y_202_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1(lean_object* v_msgData_205_, lean_object* v_macroStack_206_, lean_object* v___y_207_, lean_object* v___y_208_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___redArg(v_msgData_205_, v_macroStack_206_, v___y_208_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1___boxed(lean_object* v_msgData_211_, lean_object* v_macroStack_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1(v_msgData_211_, v_macroStack_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
return v_res_216_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_220_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__1));
v___x_221_ = l_Lean_MessageData_ofFormat(v___x_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg(lean_object* v_as_x27_222_, lean_object* v_b_223_){
_start:
{
if (lean_obj_tag(v_as_x27_222_) == 0)
{
return v_b_223_;
}
else
{
lean_object* v_head_224_; lean_object* v_tail_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v_head_224_ = lean_ctor_get(v_as_x27_222_, 0);
v_tail_225_ = lean_ctor_get(v_as_x27_222_, 1);
v___x_226_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__2, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__2_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___closed__2);
v___x_227_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_227_, 0, v_b_223_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
lean_inc(v_head_224_);
v___x_228_ = l_Lean_MessageData_ofName(v_head_224_);
v___x_229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_227_);
lean_ctor_set(v___x_229_, 1, v___x_228_);
v_as_x27_222_ = v_tail_225_;
v_b_223_ = v___x_229_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg___boxed(lean_object* v_as_x27_231_, lean_object* v_b_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg(v_as_x27_231_, v_b_232_);
lean_dec(v_as_x27_231_);
return v_res_233_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_237_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__1));
v___x_238_ = l_Lean_MessageData_ofFormat(v___x_237_);
return v___x_238_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__4(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_240_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__3));
v___x_241_ = l_Lean_stringToMessageData(v___x_240_);
return v___x_241_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__7(void){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_245_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__6));
v___x_246_ = l_Lean_MessageData_ofFormat(v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData(lean_object* v_levelParams_247_){
_start:
{
if (lean_obj_tag(v_levelParams_247_) == 0)
{
lean_object* v___x_248_; 
v___x_248_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2);
return v___x_248_;
}
else
{
lean_object* v_head_249_; lean_object* v_tail_250_; lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_262_; 
v_head_249_ = lean_ctor_get(v_levelParams_247_, 0);
v_tail_250_ = lean_ctor_get(v_levelParams_247_, 1);
v_isSharedCheck_262_ = !lean_is_exclusive(v_levelParams_247_);
if (v_isSharedCheck_262_ == 0)
{
v___x_252_ = v_levelParams_247_;
v_isShared_253_ = v_isSharedCheck_262_;
goto v_resetjp_251_;
}
else
{
lean_inc(v_tail_250_);
lean_inc(v_head_249_);
lean_dec(v_levelParams_247_);
v___x_252_ = lean_box(0);
v_isShared_253_ = v_isSharedCheck_262_;
goto v_resetjp_251_;
}
v_resetjp_251_:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v_m_257_; 
v___x_254_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__4, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__4_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__4);
v___x_255_ = l_Lean_MessageData_ofName(v_head_249_);
if (v_isShared_253_ == 0)
{
lean_ctor_set_tag(v___x_252_, 7);
lean_ctor_set(v___x_252_, 1, v___x_255_);
lean_ctor_set(v___x_252_, 0, v___x_254_);
v_m_257_ = v___x_252_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_261_; 
v_reuseFailAlloc_261_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_261_, 0, v___x_254_);
lean_ctor_set(v_reuseFailAlloc_261_, 1, v___x_255_);
v_m_257_ = v_reuseFailAlloc_261_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
v___x_258_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg(v_tail_250_, v_m_257_);
lean_dec(v_tail_250_);
v___x_259_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__7, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__7_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__7);
v___x_260_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_258_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
return v___x_260_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0(lean_object* v_as_263_, lean_object* v_as_x27_264_, lean_object* v_b_265_, lean_object* v_a_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___redArg(v_as_x27_264_, v_b_265_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0___boxed(lean_object* v_as_268_, lean_object* v_as_x27_269_, lean_object* v_b_270_, lean_object* v_a_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData_spec__0(v_as_268_, v_as_x27_269_, v_b_270_, v_a_271_);
lean_dec(v_as_x27_269_);
lean_dec(v_as_268_);
return v_res_272_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__2(void){
_start:
{
lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_276_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__1));
v___x_277_ = l_Lean_MessageData_ofFormat(v___x_276_);
return v___x_277_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5(void){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_281_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__4));
v___x_282_ = l_Lean_MessageData_ofFormat(v___x_281_);
return v___x_282_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__8(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_286_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__7));
v___x_287_ = l_Lean_MessageData_ofFormat(v___x_286_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__11(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__10));
v___x_292_ = l_Lean_MessageData_ofFormat(v___x_291_);
return v___x_292_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__14(void){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_296_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__13));
v___x_297_ = l_Lean_MessageData_ofFormat(v___x_296_);
return v___x_297_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__17(void){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_301_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__16));
v___x_302_ = l_Lean_MessageData_ofFormat(v___x_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(lean_object* v_kind_303_, lean_object* v_id_304_, lean_object* v_levelParams_305_, lean_object* v_type_306_, uint8_t v_safety_307_, lean_object* v_a_308_){
_start:
{
lean_object* v_fst_311_; lean_object* v_snd_312_; lean_object* v___y_328_; lean_object* v___y_334_; 
switch(v_safety_307_)
{
case 0:
{
lean_object* v___x_340_; 
v___x_340_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__14, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__14);
v___y_334_ = v___x_340_;
goto v___jp_333_;
}
case 1:
{
lean_object* v___x_341_; 
v___x_341_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__2);
v___y_334_ = v___x_341_;
goto v___jp_333_;
}
default: 
{
lean_object* v___x_342_; 
v___x_342_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__17, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__17_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__17);
v___y_334_ = v___x_342_;
goto v___jp_333_;
}
}
v___jp_310_:
{
lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_313_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_313_, 0, v_kind_303_);
v___x_314_ = l_Lean_MessageData_ofFormat(v___x_313_);
v___x_315_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_315_, 0, v_fst_311_);
lean_ctor_set(v___x_315_, 1, v___x_314_);
v___x_316_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__2);
v___x_317_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_317_, 0, v___x_315_);
lean_ctor_set(v___x_317_, 1, v___x_316_);
v___x_318_ = l_Lean_MessageData_ofName(v_snd_312_);
v___x_319_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_319_, 0, v___x_317_);
lean_ctor_set(v___x_319_, 1, v___x_318_);
v___x_320_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData(v_levelParams_305_);
v___x_321_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_319_);
lean_ctor_set(v___x_321_, 1, v___x_320_);
v___x_322_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5);
v___x_323_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_323_, 0, v___x_321_);
lean_ctor_set(v___x_323_, 1, v___x_322_);
v___x_324_ = l_Lean_MessageData_ofExpr(v_type_306_);
v___x_325_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_323_);
lean_ctor_set(v___x_325_, 1, v___x_324_);
v___x_326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_326_, 0, v___x_325_);
return v___x_326_;
}
v___jp_327_:
{
lean_object* v___x_329_; 
lean_inc(v_id_304_);
v___x_329_ = l_Lean_privateToUserName_x3f(v_id_304_);
if (lean_obj_tag(v___x_329_) == 0)
{
v_fst_311_ = v___y_328_;
v_snd_312_ = v_id_304_;
goto v___jp_310_;
}
else
{
lean_object* v_val_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
lean_dec(v_id_304_);
v_val_330_ = lean_ctor_get(v___x_329_, 0);
lean_inc(v_val_330_);
lean_dec_ref_known(v___x_329_, 1);
v___x_331_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__8, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__8_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__8);
v___x_332_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_332_, 0, v___y_328_);
lean_ctor_set(v___x_332_, 1, v___x_331_);
v_fst_311_ = v___x_332_;
v_snd_312_ = v_val_330_;
goto v___jp_310_;
}
}
v___jp_333_:
{
lean_object* v___x_335_; lean_object* v_env_336_; uint8_t v___x_337_; 
v___x_335_ = lean_st_ref_get(v_a_308_);
v_env_336_ = lean_ctor_get(v___x_335_, 0);
lean_inc_ref(v_env_336_);
lean_dec(v___x_335_);
lean_inc(v_id_304_);
v___x_337_ = l_Lean_isProtected(v_env_336_, v_id_304_);
if (v___x_337_ == 0)
{
lean_inc_ref(v___y_334_);
v___y_328_ = v___y_334_;
goto v___jp_327_;
}
else
{
lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_338_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__11, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__11);
lean_inc_ref(v___y_334_);
v___x_339_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_339_, 0, v___y_334_);
lean_ctor_set(v___x_339_, 1, v___x_338_);
v___y_328_ = v___x_339_;
goto v___jp_327_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___boxed(lean_object* v_kind_343_, lean_object* v_id_344_, lean_object* v_levelParams_345_, lean_object* v_type_346_, lean_object* v_safety_347_, lean_object* v_a_348_, lean_object* v_a_349_){
_start:
{
uint8_t v_safety_boxed_350_; lean_object* v_res_351_; 
v_safety_boxed_350_ = lean_unbox(v_safety_347_);
v_res_351_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(v_kind_343_, v_id_344_, v_levelParams_345_, v_type_346_, v_safety_boxed_350_, v_a_348_);
lean_dec(v_a_348_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader(lean_object* v_kind_352_, lean_object* v_id_353_, lean_object* v_levelParams_354_, lean_object* v_type_355_, uint8_t v_safety_356_, lean_object* v_a_357_, lean_object* v_a_358_){
_start:
{
lean_object* v___x_360_; 
v___x_360_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(v_kind_352_, v_id_353_, v_levelParams_354_, v_type_355_, v_safety_356_, v_a_358_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___boxed(lean_object* v_kind_361_, lean_object* v_id_362_, lean_object* v_levelParams_363_, lean_object* v_type_364_, lean_object* v_safety_365_, lean_object* v_a_366_, lean_object* v_a_367_, lean_object* v_a_368_){
_start:
{
uint8_t v_safety_boxed_369_; lean_object* v_res_370_; 
v_safety_boxed_369_ = lean_unbox(v_safety_365_);
v_res_370_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader(v_kind_361_, v_id_362_, v_levelParams_363_, v_type_364_, v_safety_boxed_369_, v_a_366_, v_a_367_);
lean_dec(v_a_367_);
lean_dec_ref(v_a_366_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(lean_object* v_kind_371_, lean_object* v_id_372_, lean_object* v_levelParams_373_, lean_object* v_type_374_, uint8_t v_isUnsafe_375_, lean_object* v_a_376_){
_start:
{
if (v_isUnsafe_375_ == 0)
{
uint8_t v___x_378_; lean_object* v___x_379_; 
v___x_378_ = 1;
v___x_379_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(v_kind_371_, v_id_372_, v_levelParams_373_, v_type_374_, v___x_378_, v_a_376_);
return v___x_379_;
}
else
{
uint8_t v___x_380_; lean_object* v___x_381_; 
v___x_380_ = 0;
v___x_381_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(v_kind_371_, v_id_372_, v_levelParams_373_, v_type_374_, v___x_380_, v_a_376_);
return v___x_381_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg___boxed(lean_object* v_kind_382_, lean_object* v_id_383_, lean_object* v_levelParams_384_, lean_object* v_type_385_, lean_object* v_isUnsafe_386_, lean_object* v_a_387_, lean_object* v_a_388_){
_start:
{
uint8_t v_isUnsafe_boxed_389_; lean_object* v_res_390_; 
v_isUnsafe_boxed_389_ = lean_unbox(v_isUnsafe_386_);
v_res_390_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v_kind_382_, v_id_383_, v_levelParams_384_, v_type_385_, v_isUnsafe_boxed_389_, v_a_387_);
lean_dec(v_a_387_);
return v_res_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27(lean_object* v_kind_391_, lean_object* v_id_392_, lean_object* v_levelParams_393_, lean_object* v_type_394_, uint8_t v_isUnsafe_395_, lean_object* v_a_396_, lean_object* v_a_397_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v_kind_391_, v_id_392_, v_levelParams_393_, v_type_394_, v_isUnsafe_395_, v_a_397_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___boxed(lean_object* v_kind_400_, lean_object* v_id_401_, lean_object* v_levelParams_402_, lean_object* v_type_403_, lean_object* v_isUnsafe_404_, lean_object* v_a_405_, lean_object* v_a_406_, lean_object* v_a_407_){
_start:
{
uint8_t v_isUnsafe_boxed_408_; lean_object* v_res_409_; 
v_isUnsafe_boxed_408_ = lean_unbox(v_isUnsafe_404_);
v_res_409_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27(v_kind_400_, v_id_401_, v_levelParams_402_, v_type_403_, v_isUnsafe_boxed_408_, v_a_405_, v_a_406_);
lean_dec(v_a_406_);
lean_dec_ref(v_a_405_);
return v_res_409_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__2(void){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_413_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__1));
v___x_414_ = l_Lean_MessageData_ofFormat(v___x_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg(lean_object* v_kind_415_, lean_object* v_id_416_, lean_object* v_levelParams_417_, lean_object* v_type_418_, lean_object* v_value_419_, uint8_t v_safety_420_, lean_object* v_a_421_){
_start:
{
lean_object* v___x_423_; lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_437_; 
v___x_423_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg(v_kind_415_, v_id_416_, v_levelParams_417_, v_type_418_, v_safety_420_, v_a_421_);
v_a_424_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_437_ == 0)
{
v___x_426_ = v___x_423_;
v_isShared_427_ = v_isSharedCheck_437_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_423_);
v___x_426_ = lean_box(0);
v_isShared_427_ = v_isSharedCheck_437_;
goto v_resetjp_425_;
}
v_resetjp_425_:
{
lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_435_; 
v___x_428_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___closed__2);
v___x_429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_429_, 0, v_a_424_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
v___x_430_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0);
v___x_431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_431_, 0, v___x_429_);
lean_ctor_set(v___x_431_, 1, v___x_430_);
v___x_432_ = l_Lean_MessageData_ofExpr(v_value_419_);
v___x_433_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_433_, 0, v___x_431_);
lean_ctor_set(v___x_433_, 1, v___x_432_);
if (v_isShared_427_ == 0)
{
lean_ctor_set(v___x_426_, 0, v___x_433_);
v___x_435_ = v___x_426_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v___x_433_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg___boxed(lean_object* v_kind_438_, lean_object* v_id_439_, lean_object* v_levelParams_440_, lean_object* v_type_441_, lean_object* v_value_442_, lean_object* v_safety_443_, lean_object* v_a_444_, lean_object* v_a_445_){
_start:
{
uint8_t v_safety_boxed_446_; lean_object* v_res_447_; 
v_safety_boxed_446_ = lean_unbox(v_safety_443_);
v_res_447_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg(v_kind_438_, v_id_439_, v_levelParams_440_, v_type_441_, v_value_442_, v_safety_boxed_446_, v_a_444_);
lean_dec(v_a_444_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike(lean_object* v_kind_448_, lean_object* v_id_449_, lean_object* v_levelParams_450_, lean_object* v_type_451_, lean_object* v_value_452_, uint8_t v_safety_453_, lean_object* v_a_454_, lean_object* v_a_455_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg(v_kind_448_, v_id_449_, v_levelParams_450_, v_type_451_, v_value_452_, v_safety_453_, v_a_455_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___boxed(lean_object* v_kind_458_, lean_object* v_id_459_, lean_object* v_levelParams_460_, lean_object* v_type_461_, lean_object* v_value_462_, lean_object* v_safety_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_){
_start:
{
uint8_t v_safety_boxed_467_; lean_object* v_res_468_; 
v_safety_boxed_467_ = lean_unbox(v_safety_463_);
v_res_468_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike(v_kind_458_, v_id_459_, v_levelParams_460_, v_type_461_, v_value_462_, v_safety_boxed_467_, v_a_464_, v_a_465_);
lean_dec(v_a_465_);
lean_dec_ref(v_a_464_);
return v_res_468_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_470_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__0));
v___x_471_ = l_Lean_stringToMessageData(v___x_470_);
return v___x_471_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3(void){
_start:
{
lean_object* v___x_473_; lean_object* v___x_474_; 
v___x_473_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__2));
v___x_474_ = l_Lean_stringToMessageData(v___x_473_);
return v___x_474_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_476_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__4));
v___x_477_ = l_Lean_stringToMessageData(v___x_476_);
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__6));
v___x_480_ = l_Lean_stringToMessageData(v___x_479_);
return v___x_480_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9(void){
_start:
{
lean_object* v___x_482_; lean_object* v___x_483_; 
v___x_482_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__8));
v___x_483_ = l_Lean_stringToMessageData(v___x_482_);
return v___x_483_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11(void){
_start:
{
lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_485_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__10));
v___x_486_ = l_Lean_stringToMessageData(v___x_485_);
return v___x_486_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13(void){
_start:
{
lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_488_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__12));
v___x_489_ = l_Lean_stringToMessageData(v___x_488_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(lean_object* v_msg_490_, lean_object* v_declHint_491_, lean_object* v___y_492_){
_start:
{
lean_object* v___x_494_; lean_object* v_env_495_; uint8_t v___x_496_; 
v___x_494_ = lean_st_ref_get(v___y_492_);
v_env_495_ = lean_ctor_get(v___x_494_, 0);
lean_inc_ref(v_env_495_);
lean_dec(v___x_494_);
v___x_496_ = l_Lean_Name_isAnonymous(v_declHint_491_);
if (v___x_496_ == 0)
{
uint8_t v_isExporting_497_; 
v_isExporting_497_ = lean_ctor_get_uint8(v_env_495_, sizeof(void*)*8);
if (v_isExporting_497_ == 0)
{
lean_object* v___x_498_; 
lean_dec_ref(v_env_495_);
lean_dec(v_declHint_491_);
v___x_498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_498_, 0, v_msg_490_);
return v___x_498_;
}
else
{
lean_object* v___x_499_; uint8_t v___x_500_; 
lean_inc_ref(v_env_495_);
v___x_499_ = l_Lean_Environment_setExporting(v_env_495_, v___x_496_);
lean_inc(v_declHint_491_);
lean_inc_ref(v___x_499_);
v___x_500_ = l_Lean_Environment_contains(v___x_499_, v_declHint_491_, v_isExporting_497_);
if (v___x_500_ == 0)
{
lean_object* v___x_501_; 
lean_dec_ref(v___x_499_);
lean_dec_ref(v_env_495_);
lean_dec(v_declHint_491_);
v___x_501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_501_, 0, v_msg_490_);
return v___x_501_;
}
else
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v_c_507_; lean_object* v___x_508_; 
v___x_502_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2);
v___x_503_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5);
v___x_504_ = l_Lean_Options_empty;
v___x_505_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_505_, 0, v___x_499_);
lean_ctor_set(v___x_505_, 1, v___x_502_);
lean_ctor_set(v___x_505_, 2, v___x_503_);
lean_ctor_set(v___x_505_, 3, v___x_504_);
lean_inc(v_declHint_491_);
v___x_506_ = l_Lean_MessageData_ofConstName(v_declHint_491_, v___x_496_);
v_c_507_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_507_, 0, v___x_505_);
lean_ctor_set(v_c_507_, 1, v___x_506_);
v___x_508_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_495_, v_declHint_491_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; 
lean_dec_ref(v_env_495_);
lean_dec(v_declHint_491_);
v___x_509_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_510_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
lean_ctor_set(v___x_510_, 1, v_c_507_);
v___x_511_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__3);
v___x_512_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_512_, 0, v___x_510_);
lean_ctor_set(v___x_512_, 1, v___x_511_);
v___x_513_ = l_Lean_MessageData_note(v___x_512_);
v___x_514_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_514_, 0, v_msg_490_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
v___x_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_515_, 0, v___x_514_);
return v___x_515_;
}
else
{
lean_object* v_val_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_551_; 
v_val_516_ = lean_ctor_get(v___x_508_, 0);
v_isSharedCheck_551_ = !lean_is_exclusive(v___x_508_);
if (v_isSharedCheck_551_ == 0)
{
v___x_518_ = v___x_508_;
v_isShared_519_ = v_isSharedCheck_551_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_val_516_);
lean_dec(v___x_508_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_551_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v_mod_523_; uint8_t v___x_524_; 
v___x_520_ = lean_box(0);
v___x_521_ = l_Lean_Environment_header(v_env_495_);
lean_dec_ref(v_env_495_);
v___x_522_ = l_Lean_EnvironmentHeader_moduleNames(v___x_521_);
v_mod_523_ = lean_array_get(v___x_520_, v___x_522_, v_val_516_);
lean_dec(v_val_516_);
lean_dec_ref(v___x_522_);
v___x_524_ = l_Lean_isPrivateName(v_declHint_491_);
lean_dec(v_declHint_491_);
if (v___x_524_ == 0)
{
lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_536_; 
v___x_525_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__5);
v___x_526_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_526_, 0, v___x_525_);
lean_ctor_set(v___x_526_, 1, v_c_507_);
v___x_527_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__7);
v___x_528_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_528_, 0, v___x_526_);
lean_ctor_set(v___x_528_, 1, v___x_527_);
v___x_529_ = l_Lean_MessageData_ofName(v_mod_523_);
v___x_530_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_530_, 0, v___x_528_);
lean_ctor_set(v___x_530_, 1, v___x_529_);
v___x_531_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__9);
v___x_532_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_532_, 0, v___x_530_);
lean_ctor_set(v___x_532_, 1, v___x_531_);
v___x_533_ = l_Lean_MessageData_note(v___x_532_);
v___x_534_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_534_, 0, v_msg_490_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
if (v_isShared_519_ == 0)
{
lean_ctor_set_tag(v___x_518_, 0);
lean_ctor_set(v___x_518_, 0, v___x_534_);
v___x_536_ = v___x_518_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v___x_534_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
else
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_549_; 
v___x_538_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__1);
v___x_539_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_538_);
lean_ctor_set(v___x_539_, 1, v_c_507_);
v___x_540_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__11);
v___x_541_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_541_, 0, v___x_539_);
lean_ctor_set(v___x_541_, 1, v___x_540_);
v___x_542_ = l_Lean_MessageData_ofName(v_mod_523_);
v___x_543_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_541_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___closed__13);
v___x_545_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_545_, 0, v___x_543_);
lean_ctor_set(v___x_545_, 1, v___x_544_);
v___x_546_ = l_Lean_MessageData_note(v___x_545_);
v___x_547_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_547_, 0, v_msg_490_);
lean_ctor_set(v___x_547_, 1, v___x_546_);
if (v_isShared_519_ == 0)
{
lean_ctor_set_tag(v___x_518_, 0);
lean_ctor_set(v___x_518_, 0, v___x_547_);
v___x_549_ = v___x_518_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v___x_547_);
v___x_549_ = v_reuseFailAlloc_550_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
return v___x_549_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_552_; 
lean_dec_ref(v_env_495_);
lean_dec(v_declHint_491_);
v___x_552_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_552_, 0, v_msg_490_);
return v___x_552_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg___boxed(lean_object* v_msg_553_, lean_object* v_declHint_554_, lean_object* v___y_555_, lean_object* v___y_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(v_msg_553_, v_declHint_554_, v___y_555_);
lean_dec(v___y_555_);
return v_res_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4(lean_object* v_msg_558_, lean_object* v_declHint_559_, lean_object* v___y_560_, lean_object* v___y_561_){
_start:
{
lean_object* v___x_563_; lean_object* v_a_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_573_; 
v___x_563_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(v_msg_558_, v_declHint_559_, v___y_561_);
v_a_564_ = lean_ctor_get(v___x_563_, 0);
v_isSharedCheck_573_ = !lean_is_exclusive(v___x_563_);
if (v_isSharedCheck_573_ == 0)
{
v___x_566_ = v___x_563_;
v_isShared_567_ = v_isSharedCheck_573_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_a_564_);
lean_dec(v___x_563_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_573_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_571_; 
v___x_568_ = l_Lean_unknownIdentifierMessageTag;
v___x_569_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_569_, 0, v___x_568_);
lean_ctor_set(v___x_569_, 1, v_a_564_);
if (v_isShared_567_ == 0)
{
lean_ctor_set(v___x_566_, 0, v___x_569_);
v___x_571_ = v___x_566_;
goto v_reusejp_570_;
}
else
{
lean_object* v_reuseFailAlloc_572_; 
v_reuseFailAlloc_572_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_572_, 0, v___x_569_);
v___x_571_ = v_reuseFailAlloc_572_;
goto v_reusejp_570_;
}
v_reusejp_570_:
{
return v___x_571_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4___boxed(lean_object* v_msg_574_, lean_object* v_declHint_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_){
_start:
{
lean_object* v_res_579_; 
v_res_579_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4(v_msg_574_, v_declHint_575_, v___y_576_, v___y_577_);
lean_dec(v___y_577_);
lean_dec_ref(v___y_576_);
return v_res_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(lean_object* v_msgData_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v___x_584_; lean_object* v_env_585_; lean_object* v_options_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_584_ = lean_st_ref_get(v___y_582_);
v_env_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc_ref(v_env_585_);
lean_dec(v___x_584_);
v_options_586_ = lean_ctor_get(v___y_581_, 2);
v___x_587_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__2);
v___x_588_ = lean_unsigned_to_nat(32u);
v___x_589_ = lean_mk_empty_array_with_capacity(v___x_588_);
lean_dec_ref(v___x_589_);
v___x_590_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg___closed__5);
lean_inc_ref(v_options_586_);
v___x_591_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_591_, 0, v_env_585_);
lean_ctor_set(v___x_591_, 1, v___x_587_);
lean_ctor_set(v___x_591_, 2, v___x_590_);
lean_ctor_set(v___x_591_, 3, v_options_586_);
v___x_592_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_592_, 0, v___x_591_);
lean_ctor_set(v___x_592_, 1, v_msgData_580_);
v___x_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8___boxed(lean_object* v_msgData_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(v_msgData_594_, v___y_595_, v___y_596_);
lean_dec(v___y_596_);
lean_dec_ref(v___y_595_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(lean_object* v_msg_599_, lean_object* v___y_600_, lean_object* v___y_601_){
_start:
{
lean_object* v_ref_603_; lean_object* v___x_604_; lean_object* v_a_605_; lean_object* v___x_607_; uint8_t v_isShared_608_; uint8_t v_isSharedCheck_613_; 
v_ref_603_ = lean_ctor_get(v___y_600_, 5);
v___x_604_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7_spec__8(v_msg_599_, v___y_600_, v___y_601_);
v_a_605_ = lean_ctor_get(v___x_604_, 0);
v_isSharedCheck_613_ = !lean_is_exclusive(v___x_604_);
if (v_isSharedCheck_613_ == 0)
{
v___x_607_ = v___x_604_;
v_isShared_608_ = v_isSharedCheck_613_;
goto v_resetjp_606_;
}
else
{
lean_inc(v_a_605_);
lean_dec(v___x_604_);
v___x_607_ = lean_box(0);
v_isShared_608_ = v_isSharedCheck_613_;
goto v_resetjp_606_;
}
v_resetjp_606_:
{
lean_object* v___x_609_; lean_object* v___x_611_; 
lean_inc(v_ref_603_);
v___x_609_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_609_, 0, v_ref_603_);
lean_ctor_set(v___x_609_, 1, v_a_605_);
if (v_isShared_608_ == 0)
{
lean_ctor_set_tag(v___x_607_, 1);
lean_ctor_set(v___x_607_, 0, v___x_609_);
v___x_611_ = v___x_607_;
goto v_reusejp_610_;
}
else
{
lean_object* v_reuseFailAlloc_612_; 
v_reuseFailAlloc_612_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_612_, 0, v___x_609_);
v___x_611_ = v_reuseFailAlloc_612_;
goto v_reusejp_610_;
}
v_reusejp_610_:
{
return v___x_611_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg___boxed(lean_object* v_msg_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_){
_start:
{
lean_object* v_res_618_; 
v_res_618_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v_msg_614_, v___y_615_, v___y_616_);
lean_dec(v___y_616_);
lean_dec_ref(v___y_615_);
return v_res_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(lean_object* v_ref_619_, lean_object* v_msg_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
lean_object* v_fileName_624_; lean_object* v_fileMap_625_; lean_object* v_options_626_; lean_object* v_currRecDepth_627_; lean_object* v_maxRecDepth_628_; lean_object* v_ref_629_; lean_object* v_currNamespace_630_; lean_object* v_openDecls_631_; lean_object* v_initHeartbeats_632_; lean_object* v_maxHeartbeats_633_; lean_object* v_quotContext_634_; lean_object* v_currMacroScope_635_; uint8_t v_diag_636_; lean_object* v_cancelTk_x3f_637_; uint8_t v_suppressElabErrors_638_; lean_object* v_inheritedTraceOptions_639_; lean_object* v_ref_640_; lean_object* v___x_641_; lean_object* v___x_642_; 
v_fileName_624_ = lean_ctor_get(v___y_621_, 0);
v_fileMap_625_ = lean_ctor_get(v___y_621_, 1);
v_options_626_ = lean_ctor_get(v___y_621_, 2);
v_currRecDepth_627_ = lean_ctor_get(v___y_621_, 3);
v_maxRecDepth_628_ = lean_ctor_get(v___y_621_, 4);
v_ref_629_ = lean_ctor_get(v___y_621_, 5);
v_currNamespace_630_ = lean_ctor_get(v___y_621_, 6);
v_openDecls_631_ = lean_ctor_get(v___y_621_, 7);
v_initHeartbeats_632_ = lean_ctor_get(v___y_621_, 8);
v_maxHeartbeats_633_ = lean_ctor_get(v___y_621_, 9);
v_quotContext_634_ = lean_ctor_get(v___y_621_, 10);
v_currMacroScope_635_ = lean_ctor_get(v___y_621_, 11);
v_diag_636_ = lean_ctor_get_uint8(v___y_621_, sizeof(void*)*14);
v_cancelTk_x3f_637_ = lean_ctor_get(v___y_621_, 12);
v_suppressElabErrors_638_ = lean_ctor_get_uint8(v___y_621_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_639_ = lean_ctor_get(v___y_621_, 13);
v_ref_640_ = l_Lean_replaceRef(v_ref_619_, v_ref_629_);
lean_inc_ref(v_inheritedTraceOptions_639_);
lean_inc(v_cancelTk_x3f_637_);
lean_inc(v_currMacroScope_635_);
lean_inc(v_quotContext_634_);
lean_inc(v_maxHeartbeats_633_);
lean_inc(v_initHeartbeats_632_);
lean_inc(v_openDecls_631_);
lean_inc(v_currNamespace_630_);
lean_inc(v_maxRecDepth_628_);
lean_inc(v_currRecDepth_627_);
lean_inc_ref(v_options_626_);
lean_inc_ref(v_fileMap_625_);
lean_inc_ref(v_fileName_624_);
v___x_641_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_641_, 0, v_fileName_624_);
lean_ctor_set(v___x_641_, 1, v_fileMap_625_);
lean_ctor_set(v___x_641_, 2, v_options_626_);
lean_ctor_set(v___x_641_, 3, v_currRecDepth_627_);
lean_ctor_set(v___x_641_, 4, v_maxRecDepth_628_);
lean_ctor_set(v___x_641_, 5, v_ref_640_);
lean_ctor_set(v___x_641_, 6, v_currNamespace_630_);
lean_ctor_set(v___x_641_, 7, v_openDecls_631_);
lean_ctor_set(v___x_641_, 8, v_initHeartbeats_632_);
lean_ctor_set(v___x_641_, 9, v_maxHeartbeats_633_);
lean_ctor_set(v___x_641_, 10, v_quotContext_634_);
lean_ctor_set(v___x_641_, 11, v_currMacroScope_635_);
lean_ctor_set(v___x_641_, 12, v_cancelTk_x3f_637_);
lean_ctor_set(v___x_641_, 13, v_inheritedTraceOptions_639_);
lean_ctor_set_uint8(v___x_641_, sizeof(void*)*14, v_diag_636_);
lean_ctor_set_uint8(v___x_641_, sizeof(void*)*14 + 1, v_suppressElabErrors_638_);
v___x_642_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v_msg_620_, v___x_641_, v___y_622_);
lean_dec_ref_known(v___x_641_, 14);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_ref_643_, lean_object* v_msg_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(v_ref_643_, v_msg_644_, v___y_645_, v___y_646_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
lean_dec(v_ref_643_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_ref_649_, lean_object* v_msg_650_, lean_object* v_declHint_651_, lean_object* v___y_652_, lean_object* v___y_653_){
_start:
{
lean_object* v___x_655_; lean_object* v_a_656_; lean_object* v___x_657_; 
v___x_655_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4(v_msg_650_, v_declHint_651_, v___y_652_, v___y_653_);
v_a_656_ = lean_ctor_get(v___x_655_, 0);
lean_inc(v_a_656_);
lean_dec_ref(v___x_655_);
v___x_657_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(v_ref_649_, v_a_656_, v___y_652_, v___y_653_);
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_ref_658_, lean_object* v_msg_659_, lean_object* v_declHint_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg(v_ref_658_, v_msg_659_, v_declHint_660_, v___y_661_, v___y_662_);
lean_dec(v___y_662_);
lean_dec_ref(v___y_661_);
lean_dec(v_ref_658_);
return v_res_664_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_666_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__0));
v___x_667_ = l_Lean_stringToMessageData(v___x_666_);
return v___x_667_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_669_; lean_object* v___x_670_; 
v___x_669_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__2));
v___x_670_ = l_Lean_stringToMessageData(v___x_669_);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg(lean_object* v_ref_671_, lean_object* v_constName_672_, lean_object* v___y_673_, lean_object* v___y_674_){
_start:
{
lean_object* v___x_676_; uint8_t v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v___x_676_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__1);
v___x_677_ = 0;
lean_inc(v_constName_672_);
v___x_678_ = l_Lean_MessageData_ofConstName(v_constName_672_, v___x_677_);
v___x_679_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_679_, 0, v___x_676_);
lean_ctor_set(v___x_679_, 1, v___x_678_);
v___x_680_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___closed__3);
v___x_681_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_679_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
v___x_682_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg(v_ref_671_, v___x_681_, v_constName_672_, v___y_673_, v___y_674_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_ref_683_, lean_object* v_constName_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg(v_ref_683_, v_constName_684_, v___y_685_, v___y_686_);
lean_dec(v___y_686_);
lean_dec_ref(v___y_685_);
lean_dec(v_ref_683_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg(lean_object* v_constName_689_, lean_object* v___y_690_, lean_object* v___y_691_){
_start:
{
lean_object* v_ref_693_; lean_object* v___x_694_; 
v_ref_693_ = lean_ctor_get(v___y_690_, 5);
v___x_694_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg(v_ref_693_, v_constName_689_, v___y_690_, v___y_691_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg___boxed(lean_object* v_constName_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_){
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg(v_constName_695_, v___y_696_, v___y_697_);
lean_dec(v___y_697_);
lean_dec_ref(v___y_696_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0(lean_object* v_constName_700_, lean_object* v___y_701_, lean_object* v___y_702_){
_start:
{
lean_object* v___x_704_; lean_object* v_env_705_; uint8_t v___x_706_; lean_object* v___x_707_; 
v___x_704_ = lean_st_ref_get(v___y_702_);
v_env_705_ = lean_ctor_get(v___x_704_, 0);
lean_inc_ref(v_env_705_);
lean_dec(v___x_704_);
v___x_706_ = 0;
lean_inc(v_constName_700_);
v___x_707_ = l_Lean_Environment_find_x3f(v_env_705_, v_constName_700_, v___x_706_);
if (lean_obj_tag(v___x_707_) == 0)
{
lean_object* v___x_708_; 
v___x_708_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg(v_constName_700_, v___y_701_, v___y_702_);
return v___x_708_;
}
else
{
lean_object* v_val_709_; lean_object* v___x_711_; uint8_t v_isShared_712_; uint8_t v_isSharedCheck_716_; 
lean_dec(v_constName_700_);
v_val_709_ = lean_ctor_get(v___x_707_, 0);
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_716_ == 0)
{
v___x_711_ = v___x_707_;
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
else
{
lean_inc(v_val_709_);
lean_dec(v___x_707_);
v___x_711_ = lean_box(0);
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
v_resetjp_710_:
{
lean_object* v___x_714_; 
if (v_isShared_712_ == 0)
{
lean_ctor_set_tag(v___x_711_, 0);
v___x_714_ = v___x_711_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_val_709_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0___boxed(lean_object* v_constName_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_){
_start:
{
lean_object* v_res_721_; 
v_res_721_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0(v_constName_717_, v___y_718_, v___y_719_);
lean_dec(v___y_719_);
lean_dec_ref(v___y_718_);
return v_res_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg(lean_object* v_as_x27_722_, lean_object* v_b_723_, lean_object* v___y_724_, lean_object* v___y_725_){
_start:
{
if (lean_obj_tag(v_as_x27_722_) == 0)
{
lean_object* v___x_727_; 
v___x_727_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_727_, 0, v_b_723_);
return v___x_727_;
}
else
{
lean_object* v_head_728_; lean_object* v_tail_729_; lean_object* v___x_730_; 
v_head_728_ = lean_ctor_get(v_as_x27_722_, 0);
v_tail_729_ = lean_ctor_get(v_as_x27_722_, 1);
lean_inc(v_head_728_);
v___x_730_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0(v_head_728_, v___y_724_, v___y_725_);
if (lean_obj_tag(v___x_730_) == 0)
{
lean_object* v_a_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; 
v_a_731_ = lean_ctor_get(v___x_730_, 0);
lean_inc(v_a_731_);
lean_dec_ref_known(v___x_730_, 1);
v___x_732_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0);
v___x_733_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_733_, 0, v_b_723_);
lean_ctor_set(v___x_733_, 1, v___x_732_);
lean_inc(v_head_728_);
v___x_734_ = l_Lean_MessageData_ofName(v_head_728_);
v___x_735_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_735_, 0, v___x_733_);
lean_ctor_set(v___x_735_, 1, v___x_734_);
v___x_736_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader___redArg___closed__5);
v___x_737_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_737_, 0, v___x_735_);
lean_ctor_set(v___x_737_, 1, v___x_736_);
v___x_738_ = l_Lean_ConstantInfo_type(v_a_731_);
lean_dec(v_a_731_);
v___x_739_ = l_Lean_MessageData_ofExpr(v___x_738_);
v___x_740_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_737_);
lean_ctor_set(v___x_740_, 1, v___x_739_);
v_as_x27_722_ = v_tail_729_;
v_b_723_ = v___x_740_;
goto _start;
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
lean_dec_ref(v_b_723_);
v_a_742_ = lean_ctor_get(v___x_730_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_730_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_730_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_730_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_a_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg___boxed(lean_object* v_as_x27_750_, lean_object* v_b_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg(v_as_x27_750_, v_b_751_, v___y_752_, v___y_753_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v_as_x27_750_);
return v_res_755_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__3(void){
_start:
{
lean_object* v___x_760_; lean_object* v___x_761_; 
v___x_760_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__2));
v___x_761_ = l_Lean_MessageData_ofFormat(v___x_760_);
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg(lean_object* v_id_762_, lean_object* v_levelParams_763_, lean_object* v_type_764_, lean_object* v_ctors_765_, uint8_t v_isUnsafe_766_, lean_object* v_a_767_, lean_object* v_a_768_){
_start:
{
lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v_a_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v___x_770_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__0));
v___x_771_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v___x_770_, v_id_762_, v_levelParams_763_, v_type_764_, v_isUnsafe_766_, v_a_768_);
v_a_772_ = lean_ctor_get(v___x_771_, 0);
lean_inc(v_a_772_);
lean_dec_ref(v___x_771_);
v___x_773_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__3___closed__0);
v___x_774_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_774_, 0, v_a_772_);
lean_ctor_set(v___x_774_, 1, v___x_773_);
v___x_775_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__3, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___closed__3);
v___x_776_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_776_, 0, v___x_774_);
lean_ctor_set(v___x_776_, 1, v___x_775_);
v___x_777_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg(v_ctors_765_, v___x_776_, v_a_767_, v_a_768_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg___boxed(lean_object* v_id_778_, lean_object* v_levelParams_779_, lean_object* v_type_780_, lean_object* v_ctors_781_, lean_object* v_isUnsafe_782_, lean_object* v_a_783_, lean_object* v_a_784_, lean_object* v_a_785_){
_start:
{
uint8_t v_isUnsafe_boxed_786_; lean_object* v_res_787_; 
v_isUnsafe_boxed_786_ = lean_unbox(v_isUnsafe_782_);
v_res_787_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg(v_id_778_, v_levelParams_779_, v_type_780_, v_ctors_781_, v_isUnsafe_boxed_786_, v_a_783_, v_a_784_);
lean_dec(v_a_784_);
lean_dec_ref(v_a_783_);
lean_dec(v_ctors_781_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct(lean_object* v_id_788_, lean_object* v_levelParams_789_, lean_object* v___numParams_790_, lean_object* v___numIndices_791_, lean_object* v_type_792_, lean_object* v_ctors_793_, uint8_t v_isUnsafe_794_, lean_object* v_a_795_, lean_object* v_a_796_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg(v_id_788_, v_levelParams_789_, v_type_792_, v_ctors_793_, v_isUnsafe_794_, v_a_795_, v_a_796_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___boxed(lean_object* v_id_799_, lean_object* v_levelParams_800_, lean_object* v___numParams_801_, lean_object* v___numIndices_802_, lean_object* v_type_803_, lean_object* v_ctors_804_, lean_object* v_isUnsafe_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_){
_start:
{
uint8_t v_isUnsafe_boxed_809_; lean_object* v_res_810_; 
v_isUnsafe_boxed_809_ = lean_unbox(v_isUnsafe_805_);
v_res_810_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct(v_id_799_, v_levelParams_800_, v___numParams_801_, v___numIndices_802_, v_type_803_, v_ctors_804_, v_isUnsafe_boxed_809_, v_a_806_, v_a_807_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
lean_dec(v_ctors_804_);
lean_dec(v___numIndices_802_);
lean_dec(v___numParams_801_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1(lean_object* v_as_811_, lean_object* v_as_x27_812_, lean_object* v_b_813_, lean_object* v_a_814_, lean_object* v___y_815_, lean_object* v___y_816_){
_start:
{
lean_object* v___x_818_; 
v___x_818_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___redArg(v_as_x27_812_, v_b_813_, v___y_815_, v___y_816_);
return v___x_818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1___boxed(lean_object* v_as_819_, lean_object* v_as_x27_820_, lean_object* v_b_821_, lean_object* v_a_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v_res_826_; 
v_res_826_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__1(v_as_819_, v_as_x27_820_, v_b_821_, v_a_822_, v___y_823_, v___y_824_);
lean_dec(v___y_824_);
lean_dec_ref(v___y_823_);
lean_dec(v_as_x27_820_);
lean_dec(v_as_819_);
return v_res_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0(lean_object* v_00_u03b1_827_, lean_object* v_constName_828_, lean_object* v___y_829_, lean_object* v___y_830_){
_start:
{
lean_object* v___x_832_; 
v___x_832_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___redArg(v_constName_828_, v___y_829_, v___y_830_);
return v___x_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0___boxed(lean_object* v_00_u03b1_833_, lean_object* v_constName_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_){
_start:
{
lean_object* v_res_838_; 
v_res_838_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0(v_00_u03b1_833_, v_constName_834_, v___y_835_, v___y_836_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_839_, lean_object* v_ref_840_, lean_object* v_constName_841_, lean_object* v___y_842_, lean_object* v___y_843_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___redArg(v_ref_840_, v_constName_841_, v___y_842_, v___y_843_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_846_, lean_object* v_ref_847_, lean_object* v_constName_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1(v_00_u03b1_846_, v_ref_847_, v_constName_848_, v___y_849_, v___y_850_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v_ref_847_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b1_853_, lean_object* v_ref_854_, lean_object* v_msg_855_, lean_object* v_declHint_856_, lean_object* v___y_857_, lean_object* v___y_858_){
_start:
{
lean_object* v___x_860_; 
v___x_860_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___redArg(v_ref_854_, v_msg_855_, v_declHint_856_, v___y_857_, v___y_858_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b1_861_, lean_object* v_ref_862_, lean_object* v_msg_863_, lean_object* v_declHint_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3(v_00_u03b1_861_, v_ref_862_, v_msg_863_, v_declHint_864_, v___y_865_, v___y_866_);
lean_dec(v___y_866_);
lean_dec_ref(v___y_865_);
lean_dec(v_ref_862_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5(lean_object* v_msg_869_, lean_object* v_declHint_870_, lean_object* v___y_871_, lean_object* v___y_872_){
_start:
{
lean_object* v___x_874_; 
v___x_874_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___redArg(v_msg_869_, v_declHint_870_, v___y_872_);
return v___x_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5___boxed(lean_object* v_msg_875_, lean_object* v_declHint_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_){
_start:
{
lean_object* v_res_880_; 
v_res_880_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__4_spec__5(v_msg_875_, v_declHint_876_, v___y_877_, v___y_878_);
lean_dec(v___y_878_);
lean_dec_ref(v___y_877_);
return v_res_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5(lean_object* v_00_u03b1_881_, lean_object* v_ref_882_, lean_object* v_msg_883_, lean_object* v___y_884_, lean_object* v___y_885_){
_start:
{
lean_object* v___x_887_; 
v___x_887_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___redArg(v_ref_882_, v_msg_883_, v___y_884_, v___y_885_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03b1_888_, lean_object* v_ref_889_, lean_object* v_msg_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_){
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5(v_00_u03b1_888_, v_ref_889_, v_msg_890_, v___y_891_, v___y_892_);
lean_dec(v___y_892_);
lean_dec_ref(v___y_891_);
lean_dec(v_ref_889_);
return v_res_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7(lean_object* v_00_u03b1_895_, lean_object* v_msg_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v___x_900_; 
v___x_900_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___redArg(v_msg_896_, v___y_897_, v___y_898_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7___boxed(lean_object* v_00_u03b1_901_, lean_object* v_msg_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_){
_start:
{
lean_object* v_res_906_; 
v_res_906_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct_spec__0_spec__0_spec__1_spec__3_spec__5_spec__7(v_00_u03b1_901_, v_msg_902_, v___y_903_, v___y_904_);
lean_dec(v___y_904_);
lean_dec_ref(v___y_903_);
return v_res_906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore(lean_object* v_id_914_, lean_object* v_x_915_, lean_object* v_a_916_, lean_object* v_a_917_){
_start:
{
switch(lean_obj_tag(v_x_915_))
{
case 0:
{
lean_object* v_val_919_; lean_object* v_toConstantVal_920_; uint8_t v_isUnsafe_921_; lean_object* v_levelParams_922_; lean_object* v_type_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v_val_919_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_919_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_920_ = lean_ctor_get(v_val_919_, 0);
lean_inc_ref(v_toConstantVal_920_);
v_isUnsafe_921_ = lean_ctor_get_uint8(v_val_919_, sizeof(void*)*1);
lean_dec_ref(v_val_919_);
v_levelParams_922_ = lean_ctor_get(v_toConstantVal_920_, 1);
lean_inc(v_levelParams_922_);
v_type_923_ = lean_ctor_get(v_toConstantVal_920_, 2);
lean_inc_ref(v_type_923_);
lean_dec_ref(v_toConstantVal_920_);
v___x_924_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__0));
v___x_925_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v___x_924_, v_id_914_, v_levelParams_922_, v_type_923_, v_isUnsafe_921_, v_a_917_);
return v___x_925_;
}
case 1:
{
lean_object* v_val_926_; lean_object* v_toConstantVal_927_; lean_object* v_value_928_; uint8_t v_safety_929_; lean_object* v_levelParams_930_; lean_object* v_type_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v_val_926_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_926_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_927_ = lean_ctor_get(v_val_926_, 0);
lean_inc_ref(v_toConstantVal_927_);
v_value_928_ = lean_ctor_get(v_val_926_, 1);
lean_inc_ref(v_value_928_);
v_safety_929_ = lean_ctor_get_uint8(v_val_926_, sizeof(void*)*4);
lean_dec_ref(v_val_926_);
v_levelParams_930_ = lean_ctor_get(v_toConstantVal_927_, 1);
lean_inc(v_levelParams_930_);
v_type_931_ = lean_ctor_get(v_toConstantVal_927_, 2);
lean_inc_ref(v_type_931_);
lean_dec_ref(v_toConstantVal_927_);
v___x_932_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__1));
v___x_933_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg(v___x_932_, v_id_914_, v_levelParams_930_, v_type_931_, v_value_928_, v_safety_929_, v_a_917_);
return v___x_933_;
}
case 2:
{
lean_object* v_val_934_; lean_object* v_toConstantVal_935_; lean_object* v_value_936_; lean_object* v_levelParams_937_; lean_object* v_type_938_; lean_object* v___x_939_; uint8_t v___x_940_; lean_object* v___x_941_; 
v_val_934_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_934_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_935_ = lean_ctor_get(v_val_934_, 0);
lean_inc_ref(v_toConstantVal_935_);
v_value_936_ = lean_ctor_get(v_val_934_, 1);
lean_inc_ref(v_value_936_);
lean_dec_ref(v_val_934_);
v_levelParams_937_ = lean_ctor_get(v_toConstantVal_935_, 1);
lean_inc(v_levelParams_937_);
v_type_938_ = lean_ctor_get(v_toConstantVal_935_, 2);
lean_inc_ref(v_type_938_);
lean_dec_ref(v_toConstantVal_935_);
v___x_939_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__2));
v___x_940_ = 1;
v___x_941_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printDefLike___redArg(v___x_939_, v_id_914_, v_levelParams_937_, v_type_938_, v_value_936_, v___x_940_, v_a_917_);
return v___x_941_;
}
case 3:
{
lean_object* v_val_942_; lean_object* v_toConstantVal_943_; uint8_t v_isUnsafe_944_; lean_object* v_levelParams_945_; lean_object* v_type_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v_val_942_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_942_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_943_ = lean_ctor_get(v_val_942_, 0);
lean_inc_ref(v_toConstantVal_943_);
v_isUnsafe_944_ = lean_ctor_get_uint8(v_val_942_, sizeof(void*)*3);
lean_dec_ref(v_val_942_);
v_levelParams_945_ = lean_ctor_get(v_toConstantVal_943_, 1);
lean_inc(v_levelParams_945_);
v_type_946_ = lean_ctor_get(v_toConstantVal_943_, 2);
lean_inc_ref(v_type_946_);
lean_dec_ref(v_toConstantVal_943_);
v___x_947_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__3));
v___x_948_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v___x_947_, v_id_914_, v_levelParams_945_, v_type_946_, v_isUnsafe_944_, v_a_917_);
return v___x_948_;
}
case 4:
{
lean_object* v_val_949_; lean_object* v_toConstantVal_950_; lean_object* v_levelParams_951_; lean_object* v_type_952_; lean_object* v___x_953_; uint8_t v___x_954_; lean_object* v___x_955_; 
v_val_949_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_949_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_950_ = lean_ctor_get(v_val_949_, 0);
lean_inc_ref(v_toConstantVal_950_);
lean_dec_ref(v_val_949_);
v_levelParams_951_ = lean_ctor_get(v_toConstantVal_950_, 1);
lean_inc(v_levelParams_951_);
v_type_952_ = lean_ctor_get(v_toConstantVal_950_, 2);
lean_inc_ref(v_type_952_);
lean_dec_ref(v_toConstantVal_950_);
v___x_953_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__4));
v___x_954_ = 0;
v___x_955_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v___x_953_, v_id_914_, v_levelParams_951_, v_type_952_, v___x_954_, v_a_917_);
return v___x_955_;
}
case 5:
{
lean_object* v_val_956_; lean_object* v_toConstantVal_957_; lean_object* v_ctors_958_; uint8_t v_isUnsafe_959_; lean_object* v_levelParams_960_; lean_object* v_type_961_; lean_object* v___x_962_; 
v_val_956_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_956_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_957_ = lean_ctor_get(v_val_956_, 0);
lean_inc_ref(v_toConstantVal_957_);
v_ctors_958_ = lean_ctor_get(v_val_956_, 4);
lean_inc(v_ctors_958_);
v_isUnsafe_959_ = lean_ctor_get_uint8(v_val_956_, sizeof(void*)*6 + 1);
lean_dec_ref(v_val_956_);
v_levelParams_960_ = lean_ctor_get(v_toConstantVal_957_, 1);
lean_inc(v_levelParams_960_);
v_type_961_ = lean_ctor_get(v_toConstantVal_957_, 2);
lean_inc_ref(v_type_961_);
lean_dec_ref(v_toConstantVal_957_);
v___x_962_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printInduct___redArg(v_id_914_, v_levelParams_960_, v_type_961_, v_ctors_958_, v_isUnsafe_959_, v_a_916_, v_a_917_);
lean_dec(v_ctors_958_);
return v___x_962_;
}
case 6:
{
lean_object* v_val_963_; lean_object* v_toConstantVal_964_; uint8_t v_isUnsafe_965_; lean_object* v_levelParams_966_; lean_object* v_type_967_; lean_object* v___x_968_; lean_object* v___x_969_; 
v_val_963_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_963_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_964_ = lean_ctor_get(v_val_963_, 0);
lean_inc_ref(v_toConstantVal_964_);
v_isUnsafe_965_ = lean_ctor_get_uint8(v_val_963_, sizeof(void*)*5);
lean_dec_ref(v_val_963_);
v_levelParams_966_ = lean_ctor_get(v_toConstantVal_964_, 1);
lean_inc(v_levelParams_966_);
v_type_967_ = lean_ctor_get(v_toConstantVal_964_, 2);
lean_inc_ref(v_type_967_);
lean_dec_ref(v_toConstantVal_964_);
v___x_968_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__5));
v___x_969_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v___x_968_, v_id_914_, v_levelParams_966_, v_type_967_, v_isUnsafe_965_, v_a_917_);
return v___x_969_;
}
default: 
{
lean_object* v_val_970_; lean_object* v_toConstantVal_971_; uint8_t v_isUnsafe_972_; lean_object* v_levelParams_973_; lean_object* v_type_974_; lean_object* v___x_975_; lean_object* v___x_976_; 
v_val_970_ = lean_ctor_get(v_x_915_, 0);
lean_inc_ref(v_val_970_);
lean_dec_ref_known(v_x_915_, 1);
v_toConstantVal_971_ = lean_ctor_get(v_val_970_, 0);
lean_inc_ref(v_toConstantVal_971_);
v_isUnsafe_972_ = lean_ctor_get_uint8(v_val_970_, sizeof(void*)*7 + 1);
lean_dec_ref(v_val_970_);
v_levelParams_973_ = lean_ctor_get(v_toConstantVal_971_, 1);
lean_inc(v_levelParams_973_);
v_type_974_ = lean_ctor_get(v_toConstantVal_971_, 2);
lean_inc_ref(v_type_974_);
lean_dec_ref(v_toConstantVal_971_);
v___x_975_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___closed__6));
v___x_976_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_mkHeader_x27___redArg(v___x_975_, v_id_914_, v_levelParams_973_, v_type_974_, v_isUnsafe_972_, v_a_917_);
return v___x_976_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore___boxed(lean_object* v_id_977_, lean_object* v_x_978_, lean_object* v_a_979_, lean_object* v_a_980_, lean_object* v_a_981_){
_start:
{
lean_object* v_res_982_; 
v_res_982_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore(v_id_977_, v_x_978_, v_a_979_, v_a_980_);
lean_dec(v_a_980_);
lean_dec_ref(v_a_979_);
return v_res_982_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_983_; lean_object* v___x_984_; 
v___x_983_ = l_Lean_instInhabitedEnvExtensionState;
v___x_984_ = l_Lean_instInhabitedPersistentEnvExtensionState___redArg(v___x_983_);
return v___x_984_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_986_; lean_object* v___x_987_; 
v___x_986_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__1));
v___x_987_ = l_Lean_stringToMessageData(v___x_986_);
return v___x_987_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4(void){
_start:
{
lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_989_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__3));
v___x_990_ = l_Lean_stringToMessageData(v___x_989_);
return v___x_990_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6(void){
_start:
{
lean_object* v___x_992_; lean_object* v___x_993_; 
v___x_992_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__5));
v___x_993_ = l_Lean_stringToMessageData(v___x_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg(lean_object* v_old_994_, lean_object* v_new_995_, lean_object* v_ext_996_, lean_object* v_a_997_){
_start:
{
lean_object* v_toEnvExtension_999_; lean_object* v_name_1000_; lean_object* v_exportEntriesFn_1001_; lean_object* v_asyncMode_1002_; lean_object* v___x_1003_; lean_object* v_asyncMode_1005_; lean_object* v___y_1006_; 
v_toEnvExtension_999_ = lean_ctor_get(v_ext_996_, 0);
lean_inc_ref(v_toEnvExtension_999_);
v_name_1000_ = lean_ctor_get(v_ext_996_, 1);
lean_inc(v_name_1000_);
v_exportEntriesFn_1001_ = lean_ctor_get(v_ext_996_, 4);
lean_inc_ref(v_exportEntriesFn_1001_);
lean_dec_ref(v_ext_996_);
v_asyncMode_1002_ = lean_ctor_get(v_toEnvExtension_999_, 2);
v___x_1003_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0);
if (lean_obj_tag(v_asyncMode_1002_) == 3)
{
lean_object* v_asyncMode_1057_; 
v_asyncMode_1057_ = lean_box(0);
v_asyncMode_1005_ = v_asyncMode_1057_;
v___y_1006_ = v_a_997_;
goto v___jp_1004_;
}
else
{
lean_inc(v_asyncMode_1002_);
v_asyncMode_1005_ = v_asyncMode_1002_;
v___y_1006_ = v_a_997_;
goto v___jp_1004_;
}
v___jp_1004_:
{
lean_object* v___x_1007_; lean_object* v_oldSt_1008_; lean_object* v_newSt_1009_; size_t v___x_1010_; size_t v___x_1011_; uint8_t v___x_1012_; 
v___x_1007_ = lean_box(0);
v_oldSt_1008_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_1003_, v_toEnvExtension_999_, v_old_994_, v_asyncMode_1005_, v___x_1007_);
v_newSt_1009_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_1003_, v_toEnvExtension_999_, v_new_995_, v_asyncMode_1005_, v___x_1007_);
lean_dec(v_asyncMode_1005_);
lean_dec_ref(v_toEnvExtension_999_);
v___x_1010_ = lean_ptr_addr(v_oldSt_1008_);
v___x_1011_ = lean_ptr_addr(v_newSt_1009_);
v___x_1012_ = lean_usize_dec_eq(v___x_1010_, v___x_1011_);
if (v___x_1012_ == 0)
{
lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v_env_1015_; lean_object* v_state_1016_; lean_object* v___x_1018_; uint8_t v_isShared_1019_; uint8_t v_isSharedCheck_1053_; 
v___x_1013_ = lean_st_ref_get(v___y_1006_);
v___x_1014_ = lean_st_ref_get(v___y_1006_);
v_env_1015_ = lean_ctor_get(v___x_1013_, 0);
lean_inc_ref(v_env_1015_);
lean_dec(v___x_1013_);
v_state_1016_ = lean_ctor_get(v_oldSt_1008_, 1);
v_isSharedCheck_1053_ = !lean_is_exclusive(v_oldSt_1008_);
if (v_isSharedCheck_1053_ == 0)
{
lean_object* v_unused_1054_; 
v_unused_1054_ = lean_ctor_get(v_oldSt_1008_, 0);
lean_dec(v_unused_1054_);
v___x_1018_ = v_oldSt_1008_;
v_isShared_1019_ = v_isSharedCheck_1053_;
goto v_resetjp_1017_;
}
else
{
lean_inc(v_state_1016_);
lean_dec(v_oldSt_1008_);
v___x_1018_ = lean_box(0);
v_isShared_1019_ = v_isSharedCheck_1053_;
goto v_resetjp_1017_;
}
v_resetjp_1017_:
{
lean_object* v___x_1020_; lean_object* v_private_1021_; lean_object* v_env_1022_; lean_object* v_state_1023_; lean_object* v___x_1025_; uint8_t v_isShared_1026_; uint8_t v_isSharedCheck_1051_; 
lean_inc_ref(v_exportEntriesFn_1001_);
v___x_1020_ = lean_apply_2(v_exportEntriesFn_1001_, v_env_1015_, v_state_1016_);
v_private_1021_ = lean_ctor_get(v___x_1020_, 2);
lean_inc(v_private_1021_);
lean_dec_ref(v___x_1020_);
v_env_1022_ = lean_ctor_get(v___x_1014_, 0);
lean_inc_ref(v_env_1022_);
lean_dec(v___x_1014_);
v_state_1023_ = lean_ctor_get(v_newSt_1009_, 1);
v_isSharedCheck_1051_ = !lean_is_exclusive(v_newSt_1009_);
if (v_isSharedCheck_1051_ == 0)
{
lean_object* v_unused_1052_; 
v_unused_1052_ = lean_ctor_get(v_newSt_1009_, 0);
lean_dec(v_unused_1052_);
v___x_1025_ = v_newSt_1009_;
v_isShared_1026_ = v_isSharedCheck_1051_;
goto v_resetjp_1024_;
}
else
{
lean_inc(v_state_1023_);
lean_dec(v_newSt_1009_);
v___x_1025_ = lean_box(0);
v_isShared_1026_ = v_isSharedCheck_1051_;
goto v_resetjp_1024_;
}
v_resetjp_1024_:
{
lean_object* v___x_1027_; lean_object* v_private_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1032_; 
v___x_1027_ = lean_apply_2(v_exportEntriesFn_1001_, v_env_1022_, v_state_1023_);
v_private_1028_ = lean_ctor_get(v___x_1027_, 2);
lean_inc(v_private_1028_);
lean_dec_ref(v___x_1027_);
v___x_1029_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2);
v___x_1030_ = l_Lean_MessageData_ofName(v_name_1000_);
if (v_isShared_1026_ == 0)
{
lean_ctor_set_tag(v___x_1025_, 7);
lean_ctor_set(v___x_1025_, 1, v___x_1030_);
lean_ctor_set(v___x_1025_, 0, v___x_1029_);
v___x_1032_ = v___x_1025_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1050_; 
v_reuseFailAlloc_1050_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1050_, 0, v___x_1029_);
lean_ctor_set(v_reuseFailAlloc_1050_, 1, v___x_1030_);
v___x_1032_ = v_reuseFailAlloc_1050_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
lean_object* v___x_1033_; lean_object* v___x_1035_; 
v___x_1033_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4);
if (v_isShared_1019_ == 0)
{
lean_ctor_set_tag(v___x_1018_, 7);
lean_ctor_set(v___x_1018_, 1, v___x_1033_);
lean_ctor_set(v___x_1018_, 0, v___x_1032_);
v___x_1035_ = v___x_1018_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v___x_1032_);
lean_ctor_set(v_reuseFailAlloc_1049_, 1, v___x_1033_);
v___x_1035_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v___x_1036_ = lean_array_get_size(v_private_1028_);
lean_dec(v_private_1028_);
v___x_1037_ = lean_nat_to_int(v___x_1036_);
v___x_1038_ = lean_array_get_size(v_private_1021_);
lean_dec(v_private_1021_);
v___x_1039_ = lean_nat_to_int(v___x_1038_);
v___x_1040_ = lean_int_sub(v___x_1037_, v___x_1039_);
lean_dec(v___x_1039_);
lean_dec(v___x_1037_);
v___x_1041_ = l_Int_repr(v___x_1040_);
lean_dec(v___x_1040_);
v___x_1042_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1041_);
v___x_1043_ = l_Lean_MessageData_ofFormat(v___x_1042_);
v___x_1044_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1044_, 0, v___x_1035_);
lean_ctor_set(v___x_1044_, 1, v___x_1043_);
v___x_1045_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6);
v___x_1046_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1044_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
v___x_1047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1047_, 0, v___x_1046_);
v___x_1048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1048_, 0, v___x_1047_);
return v___x_1048_;
}
}
}
}
}
else
{
lean_object* v___x_1055_; lean_object* v___x_1056_; 
lean_dec(v_newSt_1009_);
lean_dec(v_oldSt_1008_);
lean_dec_ref(v_exportEntriesFn_1001_);
lean_dec(v_name_1000_);
v___x_1055_ = lean_box(0);
v___x_1056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1055_);
return v___x_1056_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___boxed(lean_object* v_old_1058_, lean_object* v_new_1059_, lean_object* v_ext_1060_, lean_object* v_a_1061_, lean_object* v_a_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg(v_old_1058_, v_new_1059_, v_ext_1060_, v_a_1061_);
lean_dec(v_a_1061_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4(lean_object* v_old_1064_, lean_object* v_new_1065_, lean_object* v_ext_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_){
_start:
{
lean_object* v_toEnvExtension_1070_; lean_object* v_name_1071_; lean_object* v_exportEntriesFn_1072_; lean_object* v_asyncMode_1073_; lean_object* v___x_1074_; lean_object* v_asyncMode_1076_; lean_object* v___y_1077_; 
v_toEnvExtension_1070_ = lean_ctor_get(v_ext_1066_, 0);
lean_inc_ref(v_toEnvExtension_1070_);
v_name_1071_ = lean_ctor_get(v_ext_1066_, 1);
lean_inc(v_name_1071_);
v_exportEntriesFn_1072_ = lean_ctor_get(v_ext_1066_, 4);
lean_inc_ref(v_exportEntriesFn_1072_);
lean_dec_ref(v_ext_1066_);
v_asyncMode_1073_ = lean_ctor_get(v_toEnvExtension_1070_, 2);
v___x_1074_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0);
if (lean_obj_tag(v_asyncMode_1073_) == 3)
{
lean_object* v_asyncMode_1128_; 
v_asyncMode_1128_ = lean_box(0);
v_asyncMode_1076_ = v_asyncMode_1128_;
v___y_1077_ = v_a_1068_;
goto v___jp_1075_;
}
else
{
lean_inc(v_asyncMode_1073_);
v_asyncMode_1076_ = v_asyncMode_1073_;
v___y_1077_ = v_a_1068_;
goto v___jp_1075_;
}
v___jp_1075_:
{
lean_object* v___x_1078_; lean_object* v_oldSt_1079_; lean_object* v_newSt_1080_; size_t v___x_1081_; size_t v___x_1082_; uint8_t v___x_1083_; 
v___x_1078_ = lean_box(0);
v_oldSt_1079_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_1074_, v_toEnvExtension_1070_, v_old_1064_, v_asyncMode_1076_, v___x_1078_);
v_newSt_1080_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_1074_, v_toEnvExtension_1070_, v_new_1065_, v_asyncMode_1076_, v___x_1078_);
lean_dec(v_asyncMode_1076_);
lean_dec_ref(v_toEnvExtension_1070_);
v___x_1081_ = lean_ptr_addr(v_oldSt_1079_);
v___x_1082_ = lean_ptr_addr(v_newSt_1080_);
v___x_1083_ = lean_usize_dec_eq(v___x_1081_, v___x_1082_);
if (v___x_1083_ == 0)
{
lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v_env_1086_; lean_object* v_state_1087_; lean_object* v___x_1089_; uint8_t v_isShared_1090_; uint8_t v_isSharedCheck_1124_; 
v___x_1084_ = lean_st_ref_get(v___y_1077_);
v___x_1085_ = lean_st_ref_get(v___y_1077_);
v_env_1086_ = lean_ctor_get(v___x_1084_, 0);
lean_inc_ref(v_env_1086_);
lean_dec(v___x_1084_);
v_state_1087_ = lean_ctor_get(v_oldSt_1079_, 1);
v_isSharedCheck_1124_ = !lean_is_exclusive(v_oldSt_1079_);
if (v_isSharedCheck_1124_ == 0)
{
lean_object* v_unused_1125_; 
v_unused_1125_ = lean_ctor_get(v_oldSt_1079_, 0);
lean_dec(v_unused_1125_);
v___x_1089_ = v_oldSt_1079_;
v_isShared_1090_ = v_isSharedCheck_1124_;
goto v_resetjp_1088_;
}
else
{
lean_inc(v_state_1087_);
lean_dec(v_oldSt_1079_);
v___x_1089_ = lean_box(0);
v_isShared_1090_ = v_isSharedCheck_1124_;
goto v_resetjp_1088_;
}
v_resetjp_1088_:
{
lean_object* v___x_1091_; lean_object* v_private_1092_; lean_object* v_env_1093_; lean_object* v_state_1094_; lean_object* v___x_1096_; uint8_t v_isShared_1097_; uint8_t v_isSharedCheck_1122_; 
lean_inc_ref(v_exportEntriesFn_1072_);
v___x_1091_ = lean_apply_2(v_exportEntriesFn_1072_, v_env_1086_, v_state_1087_);
v_private_1092_ = lean_ctor_get(v___x_1091_, 2);
lean_inc(v_private_1092_);
lean_dec_ref(v___x_1091_);
v_env_1093_ = lean_ctor_get(v___x_1085_, 0);
lean_inc_ref(v_env_1093_);
lean_dec(v___x_1085_);
v_state_1094_ = lean_ctor_get(v_newSt_1080_, 1);
v_isSharedCheck_1122_ = !lean_is_exclusive(v_newSt_1080_);
if (v_isSharedCheck_1122_ == 0)
{
lean_object* v_unused_1123_; 
v_unused_1123_ = lean_ctor_get(v_newSt_1080_, 0);
lean_dec(v_unused_1123_);
v___x_1096_ = v_newSt_1080_;
v_isShared_1097_ = v_isSharedCheck_1122_;
goto v_resetjp_1095_;
}
else
{
lean_inc(v_state_1094_);
lean_dec(v_newSt_1080_);
v___x_1096_ = lean_box(0);
v_isShared_1097_ = v_isSharedCheck_1122_;
goto v_resetjp_1095_;
}
v_resetjp_1095_:
{
lean_object* v___x_1098_; lean_object* v_private_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1103_; 
v___x_1098_ = lean_apply_2(v_exportEntriesFn_1072_, v_env_1093_, v_state_1094_);
v_private_1099_ = lean_ctor_get(v___x_1098_, 2);
lean_inc(v_private_1099_);
lean_dec_ref(v___x_1098_);
v___x_1100_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2);
v___x_1101_ = l_Lean_MessageData_ofName(v_name_1071_);
if (v_isShared_1097_ == 0)
{
lean_ctor_set_tag(v___x_1096_, 7);
lean_ctor_set(v___x_1096_, 1, v___x_1101_);
lean_ctor_set(v___x_1096_, 0, v___x_1100_);
v___x_1103_ = v___x_1096_;
goto v_reusejp_1102_;
}
else
{
lean_object* v_reuseFailAlloc_1121_; 
v_reuseFailAlloc_1121_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1121_, 0, v___x_1100_);
lean_ctor_set(v_reuseFailAlloc_1121_, 1, v___x_1101_);
v___x_1103_ = v_reuseFailAlloc_1121_;
goto v_reusejp_1102_;
}
v_reusejp_1102_:
{
lean_object* v___x_1104_; lean_object* v___x_1106_; 
v___x_1104_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4);
if (v_isShared_1090_ == 0)
{
lean_ctor_set_tag(v___x_1089_, 7);
lean_ctor_set(v___x_1089_, 1, v___x_1104_);
lean_ctor_set(v___x_1089_, 0, v___x_1103_);
v___x_1106_ = v___x_1089_;
goto v_reusejp_1105_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v___x_1103_);
lean_ctor_set(v_reuseFailAlloc_1120_, 1, v___x_1104_);
v___x_1106_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1105_;
}
v_reusejp_1105_:
{
lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; 
v___x_1107_ = lean_array_get_size(v_private_1099_);
lean_dec(v_private_1099_);
v___x_1108_ = lean_nat_to_int(v___x_1107_);
v___x_1109_ = lean_array_get_size(v_private_1092_);
lean_dec(v_private_1092_);
v___x_1110_ = lean_nat_to_int(v___x_1109_);
v___x_1111_ = lean_int_sub(v___x_1108_, v___x_1110_);
lean_dec(v___x_1110_);
lean_dec(v___x_1108_);
v___x_1112_ = l_Int_repr(v___x_1111_);
lean_dec(v___x_1111_);
v___x_1113_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1113_, 0, v___x_1112_);
v___x_1114_ = l_Lean_MessageData_ofFormat(v___x_1113_);
v___x_1115_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1115_, 0, v___x_1106_);
lean_ctor_set(v___x_1115_, 1, v___x_1114_);
v___x_1116_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6);
v___x_1117_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1117_, 0, v___x_1115_);
lean_ctor_set(v___x_1117_, 1, v___x_1116_);
v___x_1118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1118_, 0, v___x_1117_);
v___x_1119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1119_, 0, v___x_1118_);
return v___x_1119_;
}
}
}
}
}
else
{
lean_object* v___x_1126_; lean_object* v___x_1127_; 
lean_dec(v_newSt_1080_);
lean_dec(v_oldSt_1079_);
lean_dec_ref(v_exportEntriesFn_1072_);
lean_dec(v_name_1071_);
v___x_1126_ = lean_box(0);
v___x_1127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1127_, 0, v___x_1126_);
return v___x_1127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___boxed(lean_object* v_old_1129_, lean_object* v_new_1130_, lean_object* v_ext_1131_, lean_object* v_a_1132_, lean_object* v_a_1133_, lean_object* v_a_1134_){
_start:
{
lean_object* v_res_1135_; 
v_res_1135_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4(v_old_1129_, v_new_1130_, v_ext_1131_, v_a_1132_, v_a_1133_);
lean_dec(v_a_1133_);
lean_dec_ref(v_a_1132_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_WhatsNew_diffExtension_spec__0(lean_object* v_a_1136_){
_start:
{
lean_object* v___x_1137_; 
v___x_1137_ = lean_nat_to_int(v_a_1136_);
return v___x_1137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg(lean_object* v_old_1138_, lean_object* v_new_1139_, lean_object* v_ext_1140_, lean_object* v_a_1141_){
_start:
{
lean_object* v_toEnvExtension_1143_; lean_object* v_name_1144_; lean_object* v_exportEntriesFn_1145_; lean_object* v_asyncMode_1146_; lean_object* v___x_1147_; lean_object* v_asyncMode_1149_; lean_object* v___y_1150_; 
v_toEnvExtension_1143_ = lean_ctor_get(v_ext_1140_, 0);
lean_inc_ref(v_toEnvExtension_1143_);
v_name_1144_ = lean_ctor_get(v_ext_1140_, 1);
lean_inc(v_name_1144_);
v_exportEntriesFn_1145_ = lean_ctor_get(v_ext_1140_, 4);
lean_inc_ref(v_exportEntriesFn_1145_);
lean_dec_ref(v_ext_1140_);
v_asyncMode_1146_ = lean_ctor_get(v_toEnvExtension_1143_, 2);
v___x_1147_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__0);
if (lean_obj_tag(v_asyncMode_1146_) == 3)
{
lean_object* v_asyncMode_1201_; 
v_asyncMode_1201_ = lean_box(0);
v_asyncMode_1149_ = v_asyncMode_1201_;
v___y_1150_ = v_a_1141_;
goto v___jp_1148_;
}
else
{
lean_inc(v_asyncMode_1146_);
v_asyncMode_1149_ = v_asyncMode_1146_;
v___y_1150_ = v_a_1141_;
goto v___jp_1148_;
}
v___jp_1148_:
{
lean_object* v___x_1151_; lean_object* v_oldSt_1152_; lean_object* v_newSt_1153_; size_t v___x_1154_; size_t v___x_1155_; uint8_t v___x_1156_; 
v___x_1151_ = lean_box(0);
v_oldSt_1152_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_1147_, v_toEnvExtension_1143_, v_old_1138_, v_asyncMode_1149_, v___x_1151_);
v_newSt_1153_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_1147_, v_toEnvExtension_1143_, v_new_1139_, v_asyncMode_1149_, v___x_1151_);
lean_dec(v_asyncMode_1149_);
lean_dec_ref(v_toEnvExtension_1143_);
v___x_1154_ = lean_ptr_addr(v_oldSt_1152_);
v___x_1155_ = lean_ptr_addr(v_newSt_1153_);
v___x_1156_ = lean_usize_dec_eq(v___x_1154_, v___x_1155_);
if (v___x_1156_ == 0)
{
lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v_env_1159_; lean_object* v_state_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1197_; 
v___x_1157_ = lean_st_ref_get(v___y_1150_);
v___x_1158_ = lean_st_ref_get(v___y_1150_);
v_env_1159_ = lean_ctor_get(v___x_1157_, 0);
lean_inc_ref(v_env_1159_);
lean_dec(v___x_1157_);
v_state_1160_ = lean_ctor_get(v_oldSt_1152_, 1);
v_isSharedCheck_1197_ = !lean_is_exclusive(v_oldSt_1152_);
if (v_isSharedCheck_1197_ == 0)
{
lean_object* v_unused_1198_; 
v_unused_1198_ = lean_ctor_get(v_oldSt_1152_, 0);
lean_dec(v_unused_1198_);
v___x_1162_ = v_oldSt_1152_;
v_isShared_1163_ = v_isSharedCheck_1197_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_state_1160_);
lean_dec(v_oldSt_1152_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1197_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___x_1164_; lean_object* v_private_1165_; lean_object* v_env_1166_; lean_object* v_state_1167_; lean_object* v___x_1169_; uint8_t v_isShared_1170_; uint8_t v_isSharedCheck_1195_; 
lean_inc_ref(v_exportEntriesFn_1145_);
v___x_1164_ = lean_apply_2(v_exportEntriesFn_1145_, v_env_1159_, v_state_1160_);
v_private_1165_ = lean_ctor_get(v___x_1164_, 2);
lean_inc(v_private_1165_);
lean_dec_ref(v___x_1164_);
v_env_1166_ = lean_ctor_get(v___x_1158_, 0);
lean_inc_ref(v_env_1166_);
lean_dec(v___x_1158_);
v_state_1167_ = lean_ctor_get(v_newSt_1153_, 1);
v_isSharedCheck_1195_ = !lean_is_exclusive(v_newSt_1153_);
if (v_isSharedCheck_1195_ == 0)
{
lean_object* v_unused_1196_; 
v_unused_1196_ = lean_ctor_get(v_newSt_1153_, 0);
lean_dec(v_unused_1196_);
v___x_1169_ = v_newSt_1153_;
v_isShared_1170_ = v_isSharedCheck_1195_;
goto v_resetjp_1168_;
}
else
{
lean_inc(v_state_1167_);
lean_dec(v_newSt_1153_);
v___x_1169_ = lean_box(0);
v_isShared_1170_ = v_isSharedCheck_1195_;
goto v_resetjp_1168_;
}
v_resetjp_1168_:
{
lean_object* v___x_1171_; lean_object* v_private_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1176_; 
v___x_1171_ = lean_apply_2(v_exportEntriesFn_1145_, v_env_1166_, v_state_1167_);
v_private_1172_ = lean_ctor_get(v___x_1171_, 2);
lean_inc(v_private_1172_);
lean_dec_ref(v___x_1171_);
v___x_1173_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__2);
v___x_1174_ = l_Lean_MessageData_ofName(v_name_1144_);
if (v_isShared_1170_ == 0)
{
lean_ctor_set_tag(v___x_1169_, 7);
lean_ctor_set(v___x_1169_, 1, v___x_1174_);
lean_ctor_set(v___x_1169_, 0, v___x_1173_);
v___x_1176_ = v___x_1169_;
goto v_reusejp_1175_;
}
else
{
lean_object* v_reuseFailAlloc_1194_; 
v_reuseFailAlloc_1194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1194_, 0, v___x_1173_);
lean_ctor_set(v_reuseFailAlloc_1194_, 1, v___x_1174_);
v___x_1176_ = v_reuseFailAlloc_1194_;
goto v_reusejp_1175_;
}
v_reusejp_1175_:
{
lean_object* v___x_1177_; lean_object* v___x_1179_; 
v___x_1177_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__4);
if (v_isShared_1163_ == 0)
{
lean_ctor_set_tag(v___x_1162_, 7);
lean_ctor_set(v___x_1162_, 1, v___x_1177_);
lean_ctor_set(v___x_1162_, 0, v___x_1176_);
v___x_1179_ = v___x_1162_;
goto v_reusejp_1178_;
}
else
{
lean_object* v_reuseFailAlloc_1193_; 
v_reuseFailAlloc_1193_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1193_, 0, v___x_1176_);
lean_ctor_set(v_reuseFailAlloc_1193_, 1, v___x_1177_);
v___x_1179_ = v_reuseFailAlloc_1193_;
goto v_reusejp_1178_;
}
v_reusejp_1178_:
{
lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1180_ = lean_array_get_size(v_private_1172_);
lean_dec(v_private_1172_);
v___x_1181_ = lean_nat_to_int(v___x_1180_);
v___x_1182_ = lean_array_get_size(v_private_1165_);
lean_dec(v_private_1165_);
v___x_1183_ = lean_nat_to_int(v___x_1182_);
v___x_1184_ = lean_int_sub(v___x_1181_, v___x_1183_);
lean_dec(v___x_1183_);
lean_dec(v___x_1181_);
v___x_1185_ = l_Int_repr(v___x_1184_);
lean_dec(v___x_1184_);
v___x_1186_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1185_);
v___x_1187_ = l_Lean_MessageData_ofFormat(v___x_1186_);
v___x_1188_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1188_, 0, v___x_1179_);
lean_ctor_set(v___x_1188_, 1, v___x_1187_);
v___x_1189_ = lean_obj_once(&lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6, &lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_diffExtension_unsafe__4___redArg___closed__6);
v___x_1190_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1188_);
lean_ctor_set(v___x_1190_, 1, v___x_1189_);
v___x_1191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1190_);
v___x_1192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1192_, 0, v___x_1191_);
return v___x_1192_;
}
}
}
}
}
else
{
lean_object* v___x_1199_; lean_object* v___x_1200_; 
lean_dec(v_newSt_1153_);
lean_dec(v_oldSt_1152_);
lean_dec_ref(v_exportEntriesFn_1145_);
lean_dec(v_name_1144_);
v___x_1199_ = lean_box(0);
v___x_1200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1200_, 0, v___x_1199_);
return v___x_1200_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg___boxed(lean_object* v_old_1202_, lean_object* v_new_1203_, lean_object* v_ext_1204_, lean_object* v_a_1205_, lean_object* v_a_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg(v_old_1202_, v_new_1203_, v_ext_1204_, v_a_1205_);
lean_dec(v_a_1205_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension(lean_object* v_old_1208_, lean_object* v_new_1209_, lean_object* v_ext_1210_, lean_object* v_a_1211_, lean_object* v_a_1212_){
_start:
{
lean_object* v___x_1214_; 
v___x_1214_ = lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg(v_old_1208_, v_new_1209_, v_ext_1210_, v_a_1212_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_diffExtension___boxed(lean_object* v_old_1215_, lean_object* v_new_1216_, lean_object* v_ext_1217_, lean_object* v_a_1218_, lean_object* v_a_1219_, lean_object* v_a_1220_){
_start:
{
lean_object* v_res_1221_; 
v_res_1221_ = lp_mathlib_Mathlib_WhatsNew_diffExtension(v_old_1215_, v_new_1216_, v_ext_1217_, v_a_1218_, v_a_1219_);
lean_dec(v_a_1219_);
lean_dec_ref(v_a_1218_);
return v_res_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___lam__0(lean_object* v_ps_1222_, lean_object* v_k_1223_, lean_object* v_v_1224_){
_start:
{
lean_object* v___x_1225_; lean_object* v___x_1226_; 
v___x_1225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1225_, 0, v_k_1223_);
lean_ctor_set(v___x_1225_, 1, v_v_1224_);
v___x_1226_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1226_, 0, v___x_1225_);
lean_ctor_set(v___x_1226_, 1, v_ps_1222_);
return v___x_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(lean_object* v_f_1227_, lean_object* v_keys_1228_, lean_object* v_vals_1229_, lean_object* v_i_1230_, lean_object* v_acc_1231_){
_start:
{
lean_object* v___x_1232_; uint8_t v___x_1233_; 
v___x_1232_ = lean_array_get_size(v_keys_1228_);
v___x_1233_ = lean_nat_dec_lt(v_i_1230_, v___x_1232_);
if (v___x_1233_ == 0)
{
lean_dec(v_i_1230_);
lean_dec(v_f_1227_);
return v_acc_1231_;
}
else
{
lean_object* v_k_1234_; lean_object* v_v_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; 
v_k_1234_ = lean_array_fget_borrowed(v_keys_1228_, v_i_1230_);
v_v_1235_ = lean_array_fget_borrowed(v_vals_1229_, v_i_1230_);
lean_inc(v_f_1227_);
lean_inc(v_v_1235_);
lean_inc(v_k_1234_);
v___x_1236_ = lean_apply_3(v_f_1227_, v_acc_1231_, v_k_1234_, v_v_1235_);
v___x_1237_ = lean_unsigned_to_nat(1u);
v___x_1238_ = lean_nat_add(v_i_1230_, v___x_1237_);
lean_dec(v_i_1230_);
v_i_1230_ = v___x_1238_;
v_acc_1231_ = v___x_1236_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg___boxed(lean_object* v_f_1240_, lean_object* v_keys_1241_, lean_object* v_vals_1242_, lean_object* v_i_1243_, lean_object* v_acc_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_1240_, v_keys_1241_, v_vals_1242_, v_i_1243_, v_acc_1244_);
lean_dec_ref(v_vals_1242_);
lean_dec_ref(v_keys_1241_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(lean_object* v_f_1246_, lean_object* v_x_1247_, lean_object* v_x_1248_){
_start:
{
if (lean_obj_tag(v_x_1247_) == 0)
{
lean_object* v_es_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; uint8_t v___x_1252_; 
v_es_1249_ = lean_ctor_get(v_x_1247_, 0);
v___x_1250_ = lean_unsigned_to_nat(0u);
v___x_1251_ = lean_array_get_size(v_es_1249_);
v___x_1252_ = lean_nat_dec_lt(v___x_1250_, v___x_1251_);
if (v___x_1252_ == 0)
{
lean_dec(v_f_1246_);
return v_x_1248_;
}
else
{
uint8_t v___x_1253_; 
v___x_1253_ = lean_nat_dec_le(v___x_1251_, v___x_1251_);
if (v___x_1253_ == 0)
{
if (v___x_1252_ == 0)
{
lean_dec(v_f_1246_);
return v_x_1248_;
}
else
{
size_t v___x_1254_; size_t v___x_1255_; lean_object* v___x_1256_; 
v___x_1254_ = ((size_t)0ULL);
v___x_1255_ = lean_usize_of_nat(v___x_1251_);
v___x_1256_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg(v_f_1246_, v_es_1249_, v___x_1254_, v___x_1255_, v_x_1248_);
return v___x_1256_;
}
}
else
{
size_t v___x_1257_; size_t v___x_1258_; lean_object* v___x_1259_; 
v___x_1257_ = ((size_t)0ULL);
v___x_1258_ = lean_usize_of_nat(v___x_1251_);
v___x_1259_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg(v_f_1246_, v_es_1249_, v___x_1257_, v___x_1258_, v_x_1248_);
return v___x_1259_;
}
}
}
else
{
lean_object* v_ks_1260_; lean_object* v_vs_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
v_ks_1260_ = lean_ctor_get(v_x_1247_, 0);
v_vs_1261_ = lean_ctor_get(v_x_1247_, 1);
v___x_1262_ = lean_unsigned_to_nat(0u);
v___x_1263_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_1246_, v_ks_1260_, v_vs_1261_, v___x_1262_, v_x_1248_);
return v___x_1263_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg(lean_object* v_f_1264_, lean_object* v_as_1265_, size_t v_i_1266_, size_t v_stop_1267_, lean_object* v_b_1268_){
_start:
{
lean_object* v___y_1270_; uint8_t v___x_1274_; 
v___x_1274_ = lean_usize_dec_eq(v_i_1266_, v_stop_1267_);
if (v___x_1274_ == 0)
{
lean_object* v___x_1275_; 
v___x_1275_ = lean_array_uget_borrowed(v_as_1265_, v_i_1266_);
switch(lean_obj_tag(v___x_1275_))
{
case 0:
{
lean_object* v_key_1276_; lean_object* v_val_1277_; lean_object* v___x_1278_; 
v_key_1276_ = lean_ctor_get(v___x_1275_, 0);
v_val_1277_ = lean_ctor_get(v___x_1275_, 1);
lean_inc(v_f_1264_);
lean_inc(v_val_1277_);
lean_inc(v_key_1276_);
v___x_1278_ = lean_apply_3(v_f_1264_, v_b_1268_, v_key_1276_, v_val_1277_);
v___y_1270_ = v___x_1278_;
goto v___jp_1269_;
}
case 1:
{
lean_object* v_node_1279_; lean_object* v___x_1280_; 
v_node_1279_ = lean_ctor_get(v___x_1275_, 0);
lean_inc(v_f_1264_);
v___x_1280_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(v_f_1264_, v_node_1279_, v_b_1268_);
v___y_1270_ = v___x_1280_;
goto v___jp_1269_;
}
default: 
{
v___y_1270_ = v_b_1268_;
goto v___jp_1269_;
}
}
}
else
{
lean_dec(v_f_1264_);
return v_b_1268_;
}
v___jp_1269_:
{
size_t v___x_1271_; size_t v___x_1272_; 
v___x_1271_ = ((size_t)1ULL);
v___x_1272_ = lean_usize_add(v_i_1266_, v___x_1271_);
v_i_1266_ = v___x_1272_;
v_b_1268_ = v___y_1270_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg___boxed(lean_object* v_f_1281_, lean_object* v_as_1282_, lean_object* v_i_1283_, lean_object* v_stop_1284_, lean_object* v_b_1285_){
_start:
{
size_t v_i_boxed_1286_; size_t v_stop_boxed_1287_; lean_object* v_res_1288_; 
v_i_boxed_1286_ = lean_unbox_usize(v_i_1283_);
lean_dec(v_i_1283_);
v_stop_boxed_1287_ = lean_unbox_usize(v_stop_1284_);
lean_dec(v_stop_1284_);
v_res_1288_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg(v_f_1281_, v_as_1282_, v_i_boxed_1286_, v_stop_boxed_1287_, v_b_1285_);
lean_dec_ref(v_as_1282_);
return v_res_1288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg___boxed(lean_object* v_f_1289_, lean_object* v_x_1290_, lean_object* v_x_1291_){
_start:
{
lean_object* v_res_1292_; 
v_res_1292_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(v_f_1289_, v_x_1290_, v_x_1291_);
lean_dec_ref(v_x_1290_);
return v_res_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg___lam__0(lean_object* v_f_1293_, lean_object* v_x1_1294_, lean_object* v_x2_1295_, lean_object* v_x3_1296_){
_start:
{
lean_object* v___x_1297_; 
v___x_1297_ = lean_apply_3(v_f_1293_, v_x1_1294_, v_x2_1295_, v_x3_1296_);
return v___x_1297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg(lean_object* v_map_1298_, lean_object* v_f_1299_, lean_object* v_init_1300_){
_start:
{
lean_object* v___f_1301_; lean_object* v___x_1302_; 
v___f_1301_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1301_, 0, v_f_1299_);
v___x_1302_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(v___f_1301_, v_map_1298_, v_init_1300_);
return v___x_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg___boxed(lean_object* v_map_1303_, lean_object* v_f_1304_, lean_object* v_init_1305_){
_start:
{
lean_object* v_res_1306_; 
v_res_1306_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg(v_map_1303_, v_f_1304_, v_init_1305_);
lean_dec_ref(v_map_1303_);
return v_res_1306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg(lean_object* v_m_1308_){
_start:
{
lean_object* v___f_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; 
v___f_1309_ = ((lean_object*)(lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___closed__0));
v___x_1310_ = lean_box(0);
v___x_1311_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg(v_m_1308_, v___f_1309_, v___x_1310_);
return v___x_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg___boxed(lean_object* v_m_1312_){
_start:
{
lean_object* v_res_1313_; 
v_res_1313_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg(v_m_1312_);
lean_dec_ref(v_m_1312_);
return v_res_1313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg(lean_object* v_old_1314_, lean_object* v_new_1315_, lean_object* v_as_1316_, size_t v_sz_1317_, size_t v_i_1318_, lean_object* v_b_1319_, lean_object* v___y_1320_){
_start:
{
uint8_t v___x_1322_; 
v___x_1322_ = lean_usize_dec_lt(v_i_1318_, v_sz_1317_);
if (v___x_1322_ == 0)
{
lean_object* v___x_1323_; 
lean_dec_ref(v_new_1315_);
lean_dec_ref(v_old_1314_);
v___x_1323_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1323_, 0, v_b_1319_);
return v___x_1323_;
}
else
{
lean_object* v_a_1324_; lean_object* v___x_1325_; 
v_a_1324_ = lean_array_uget_borrowed(v_as_1316_, v_i_1318_);
lean_inc(v_a_1324_);
lean_inc_ref(v_new_1315_);
lean_inc_ref(v_old_1314_);
v___x_1325_ = lp_mathlib_Mathlib_WhatsNew_diffExtension___redArg(v_old_1314_, v_new_1315_, v_a_1324_, v___y_1320_);
if (lean_obj_tag(v___x_1325_) == 0)
{
lean_object* v_a_1326_; lean_object* v_a_1328_; 
v_a_1326_ = lean_ctor_get(v___x_1325_, 0);
lean_inc(v_a_1326_);
lean_dec_ref_known(v___x_1325_, 1);
if (lean_obj_tag(v_a_1326_) == 1)
{
lean_object* v_val_1332_; lean_object* v___x_1333_; 
v_val_1332_ = lean_ctor_get(v_a_1326_, 0);
lean_inc(v_val_1332_);
lean_dec_ref_known(v_a_1326_, 1);
v___x_1333_ = lean_array_push(v_b_1319_, v_val_1332_);
v_a_1328_ = v___x_1333_;
goto v___jp_1327_;
}
else
{
lean_dec(v_a_1326_);
v_a_1328_ = v_b_1319_;
goto v___jp_1327_;
}
v___jp_1327_:
{
size_t v___x_1329_; size_t v___x_1330_; 
v___x_1329_ = ((size_t)1ULL);
v___x_1330_ = lean_usize_add(v_i_1318_, v___x_1329_);
v_i_1318_ = v___x_1330_;
v_b_1319_ = v_a_1328_;
goto _start;
}
}
else
{
lean_object* v_a_1334_; lean_object* v___x_1336_; uint8_t v_isShared_1337_; uint8_t v_isSharedCheck_1341_; 
lean_dec_ref(v_b_1319_);
lean_dec_ref(v_new_1315_);
lean_dec_ref(v_old_1314_);
v_a_1334_ = lean_ctor_get(v___x_1325_, 0);
v_isSharedCheck_1341_ = !lean_is_exclusive(v___x_1325_);
if (v_isSharedCheck_1341_ == 0)
{
v___x_1336_ = v___x_1325_;
v_isShared_1337_ = v_isSharedCheck_1341_;
goto v_resetjp_1335_;
}
else
{
lean_inc(v_a_1334_);
lean_dec(v___x_1325_);
v___x_1336_ = lean_box(0);
v_isShared_1337_ = v_isSharedCheck_1341_;
goto v_resetjp_1335_;
}
v_resetjp_1335_:
{
lean_object* v___x_1339_; 
if (v_isShared_1337_ == 0)
{
v___x_1339_ = v___x_1336_;
goto v_reusejp_1338_;
}
else
{
lean_object* v_reuseFailAlloc_1340_; 
v_reuseFailAlloc_1340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1340_, 0, v_a_1334_);
v___x_1339_ = v_reuseFailAlloc_1340_;
goto v_reusejp_1338_;
}
v_reusejp_1338_:
{
return v___x_1339_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg___boxed(lean_object* v_old_1342_, lean_object* v_new_1343_, lean_object* v_as_1344_, lean_object* v_sz_1345_, lean_object* v_i_1346_, lean_object* v_b_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_){
_start:
{
size_t v_sz_boxed_1350_; size_t v_i_boxed_1351_; lean_object* v_res_1352_; 
v_sz_boxed_1350_ = lean_unbox_usize(v_sz_1345_);
lean_dec(v_sz_1345_);
v_i_boxed_1351_ = lean_unbox_usize(v_i_1346_);
lean_dec(v_i_1346_);
v_res_1352_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg(v_old_1342_, v_new_1343_, v_as_1344_, v_sz_boxed_1350_, v_i_boxed_1351_, v_b_1347_, v___y_1348_);
lean_dec(v___y_1348_);
lean_dec_ref(v_as_1344_);
return v_res_1352_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_1353_, lean_object* v_i_1354_, lean_object* v_k_1355_){
_start:
{
lean_object* v___x_1356_; uint8_t v___x_1357_; 
v___x_1356_ = lean_array_get_size(v_keys_1353_);
v___x_1357_ = lean_nat_dec_lt(v_i_1354_, v___x_1356_);
if (v___x_1357_ == 0)
{
lean_dec(v_i_1354_);
return v___x_1357_;
}
else
{
lean_object* v_k_x27_1358_; uint8_t v___x_1359_; 
v_k_x27_1358_ = lean_array_fget_borrowed(v_keys_1353_, v_i_1354_);
v___x_1359_ = lean_name_eq(v_k_1355_, v_k_x27_1358_);
if (v___x_1359_ == 0)
{
lean_object* v___x_1360_; lean_object* v___x_1361_; 
v___x_1360_ = lean_unsigned_to_nat(1u);
v___x_1361_ = lean_nat_add(v_i_1354_, v___x_1360_);
lean_dec(v_i_1354_);
v_i_1354_ = v___x_1361_;
goto _start;
}
else
{
lean_dec(v_i_1354_);
return v___x_1359_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_1363_, lean_object* v_i_1364_, lean_object* v_k_1365_){
_start:
{
uint8_t v_res_1366_; lean_object* v_r_1367_; 
v_res_1366_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg(v_keys_1363_, v_i_1364_, v_k_1365_);
lean_dec(v_k_1365_);
lean_dec_ref(v_keys_1363_);
v_r_1367_ = lean_box(v_res_1366_);
return v_r_1367_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg(lean_object* v_x_1368_, size_t v_x_1369_, lean_object* v_x_1370_){
_start:
{
if (lean_obj_tag(v_x_1368_) == 0)
{
lean_object* v_es_1371_; lean_object* v___x_1372_; size_t v___x_1373_; size_t v___x_1374_; lean_object* v_j_1375_; lean_object* v___x_1376_; 
v_es_1371_ = lean_ctor_get(v_x_1368_, 0);
v___x_1372_ = lean_box(2);
v___x_1373_ = ((size_t)31ULL);
v___x_1374_ = lean_usize_land(v_x_1369_, v___x_1373_);
v_j_1375_ = lean_usize_to_nat(v___x_1374_);
v___x_1376_ = lean_array_get_borrowed(v___x_1372_, v_es_1371_, v_j_1375_);
lean_dec(v_j_1375_);
switch(lean_obj_tag(v___x_1376_))
{
case 0:
{
lean_object* v_key_1377_; uint8_t v___x_1378_; 
v_key_1377_ = lean_ctor_get(v___x_1376_, 0);
v___x_1378_ = lean_name_eq(v_x_1370_, v_key_1377_);
return v___x_1378_;
}
case 1:
{
lean_object* v_node_1379_; size_t v___x_1380_; size_t v___x_1381_; 
v_node_1379_ = lean_ctor_get(v___x_1376_, 0);
v___x_1380_ = ((size_t)5ULL);
v___x_1381_ = lean_usize_shift_right(v_x_1369_, v___x_1380_);
v_x_1368_ = v_node_1379_;
v_x_1369_ = v___x_1381_;
goto _start;
}
default: 
{
uint8_t v___x_1383_; 
v___x_1383_ = 0;
return v___x_1383_;
}
}
}
else
{
lean_object* v_ks_1384_; lean_object* v___x_1385_; uint8_t v___x_1386_; 
v_ks_1384_ = lean_ctor_get(v_x_1368_, 0);
v___x_1385_ = lean_unsigned_to_nat(0u);
v___x_1386_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg(v_ks_1384_, v___x_1385_, v_x_1370_);
return v___x_1386_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg___boxed(lean_object* v_x_1387_, lean_object* v_x_1388_, lean_object* v_x_1389_){
_start:
{
size_t v_x_2709__boxed_1390_; uint8_t v_res_1391_; lean_object* v_r_1392_; 
v_x_2709__boxed_1390_ = lean_unbox_usize(v_x_1388_);
lean_dec(v_x_1388_);
v_res_1391_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg(v_x_1387_, v_x_2709__boxed_1390_, v_x_1389_);
lean_dec(v_x_1389_);
lean_dec_ref(v_x_1387_);
v_r_1392_ = lean_box(v_res_1391_);
return v_r_1392_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg(lean_object* v_x_1393_, lean_object* v_x_1394_){
_start:
{
uint64_t v___y_1396_; 
if (lean_obj_tag(v_x_1394_) == 0)
{
uint64_t v___x_1399_; 
v___x_1399_ = 1723ULL;
v___y_1396_ = v___x_1399_;
goto v___jp_1395_;
}
else
{
uint64_t v_hash_1400_; 
v_hash_1400_ = lean_ctor_get_uint64(v_x_1394_, sizeof(void*)*2);
v___y_1396_ = v_hash_1400_;
goto v___jp_1395_;
}
v___jp_1395_:
{
size_t v___x_1397_; uint8_t v___x_1398_; 
v___x_1397_ = lean_uint64_to_usize(v___y_1396_);
v___x_1398_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg(v_x_1393_, v___x_1397_, v_x_1394_);
return v___x_1398_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg___boxed(lean_object* v_x_1401_, lean_object* v_x_1402_){
_start:
{
uint8_t v_res_1403_; lean_object* v_r_1404_; 
v_res_1403_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg(v_x_1401_, v_x_1402_);
lean_dec(v_x_1402_);
lean_dec_ref(v_x_1401_);
v_r_1404_ = lean_box(v_res_1403_);
return v_r_1404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg(lean_object* v_old_1405_, lean_object* v_as_x27_1406_, lean_object* v_b_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_){
_start:
{
if (lean_obj_tag(v_as_x27_1406_) == 0)
{
lean_object* v___x_1411_; 
lean_dec_ref(v_old_1405_);
v___x_1411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1411_, 0, v_b_1407_);
return v___x_1411_;
}
else
{
lean_object* v_head_1412_; lean_object* v_tail_1413_; lean_object* v_fst_1414_; lean_object* v_snd_1415_; lean_object* v___x_1416_; lean_object* v_map_u2082_1417_; uint8_t v___x_1418_; 
v_head_1412_ = lean_ctor_get(v_as_x27_1406_, 0);
v_tail_1413_ = lean_ctor_get(v_as_x27_1406_, 1);
v_fst_1414_ = lean_ctor_get(v_head_1412_, 0);
v_snd_1415_ = lean_ctor_get(v_head_1412_, 1);
lean_inc_ref(v_old_1405_);
v___x_1416_ = l_Lean_Environment_constants(v_old_1405_);
v_map_u2082_1417_ = lean_ctor_get(v___x_1416_, 1);
lean_inc_ref(v_map_u2082_1417_);
lean_dec_ref(v___x_1416_);
v___x_1418_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg(v_map_u2082_1417_, v_fst_1414_);
lean_dec_ref(v_map_u2082_1417_);
if (v___x_1418_ == 0)
{
lean_object* v___x_1419_; 
lean_inc(v_snd_1415_);
lean_inc(v_fst_1414_);
v___x_1419_ = lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_printIdCore(v_fst_1414_, v_snd_1415_, v___y_1408_, v___y_1409_);
if (lean_obj_tag(v___x_1419_) == 0)
{
lean_object* v_a_1420_; lean_object* v___x_1421_; 
v_a_1420_ = lean_ctor_get(v___x_1419_, 0);
lean_inc(v_a_1420_);
lean_dec_ref_known(v___x_1419_, 1);
v___x_1421_ = lean_array_push(v_b_1407_, v_a_1420_);
v_as_x27_1406_ = v_tail_1413_;
v_b_1407_ = v___x_1421_;
goto _start;
}
else
{
lean_object* v_a_1423_; lean_object* v___x_1425_; uint8_t v_isShared_1426_; uint8_t v_isSharedCheck_1430_; 
lean_dec_ref(v_b_1407_);
lean_dec_ref(v_old_1405_);
v_a_1423_ = lean_ctor_get(v___x_1419_, 0);
v_isSharedCheck_1430_ = !lean_is_exclusive(v___x_1419_);
if (v_isSharedCheck_1430_ == 0)
{
v___x_1425_ = v___x_1419_;
v_isShared_1426_ = v_isSharedCheck_1430_;
goto v_resetjp_1424_;
}
else
{
lean_inc(v_a_1423_);
lean_dec(v___x_1419_);
v___x_1425_ = lean_box(0);
v_isShared_1426_ = v_isSharedCheck_1430_;
goto v_resetjp_1424_;
}
v_resetjp_1424_:
{
lean_object* v___x_1428_; 
if (v_isShared_1426_ == 0)
{
v___x_1428_ = v___x_1425_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1429_; 
v_reuseFailAlloc_1429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1429_, 0, v_a_1423_);
v___x_1428_ = v_reuseFailAlloc_1429_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
return v___x_1428_;
}
}
}
}
else
{
v_as_x27_1406_ = v_tail_1413_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg___boxed(lean_object* v_old_1432_, lean_object* v_as_x27_1433_, lean_object* v_b_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_){
_start:
{
lean_object* v_res_1438_; 
v_res_1438_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg(v_old_1432_, v_as_x27_1433_, v_b_1434_, v___y_1435_, v___y_1436_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec(v_as_x27_1433_);
return v_res_1438_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__3(void){
_start:
{
lean_object* v___x_1444_; lean_object* v___x_1445_; 
v___x_1444_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__2));
v___x_1445_ = l_Lean_MessageData_ofFormat(v___x_1444_);
return v___x_1445_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__6(void){
_start:
{
lean_object* v___x_1449_; lean_object* v___x_1450_; 
v___x_1449_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__5));
v___x_1450_ = l_Lean_MessageData_ofFormat(v___x_1449_);
return v___x_1450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew(lean_object* v_old_1451_, lean_object* v_new_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v___x_1456_; lean_object* v_map_u2082_1457_; lean_object* v___x_1458_; lean_object* v_diffs_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; 
lean_inc_ref(v_new_1452_);
v___x_1456_ = l_Lean_Environment_constants(v_new_1452_);
v_map_u2082_1457_ = lean_ctor_get(v___x_1456_, 1);
lean_inc_ref(v_map_u2082_1457_);
lean_dec_ref(v___x_1456_);
v___x_1458_ = lean_unsigned_to_nat(0u);
v_diffs_1459_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__0));
v___x_1460_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg(v_map_u2082_1457_);
lean_dec_ref(v_map_u2082_1457_);
lean_inc_ref(v_old_1451_);
v___x_1461_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg(v_old_1451_, v___x_1460_, v_diffs_1459_, v_a_1453_, v_a_1454_);
lean_dec(v___x_1460_);
if (lean_obj_tag(v___x_1461_) == 0)
{
lean_object* v_a_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; size_t v_sz_1465_; size_t v___x_1466_; lean_object* v___x_1467_; 
v_a_1462_ = lean_ctor_get(v___x_1461_, 0);
lean_inc(v_a_1462_);
lean_dec_ref_known(v___x_1461_, 1);
v___x_1463_ = l_Lean_persistentEnvExtensionsRef;
v___x_1464_ = lean_st_ref_get(v___x_1463_);
v_sz_1465_ = lean_array_size(v___x_1464_);
v___x_1466_ = ((size_t)0ULL);
v___x_1467_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg(v_old_1451_, v_new_1452_, v___x_1464_, v_sz_1465_, v___x_1466_, v_a_1462_, v_a_1454_);
lean_dec(v___x_1464_);
if (lean_obj_tag(v___x_1467_) == 0)
{
lean_object* v_a_1468_; lean_object* v___x_1470_; uint8_t v_isShared_1471_; uint8_t v_isSharedCheck_1484_; 
v_a_1468_ = lean_ctor_get(v___x_1467_, 0);
v_isSharedCheck_1484_ = !lean_is_exclusive(v___x_1467_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1470_ = v___x_1467_;
v_isShared_1471_ = v_isSharedCheck_1484_;
goto v_resetjp_1469_;
}
else
{
lean_inc(v_a_1468_);
lean_dec(v___x_1467_);
v___x_1470_ = lean_box(0);
v_isShared_1471_ = v_isSharedCheck_1484_;
goto v_resetjp_1469_;
}
v_resetjp_1469_:
{
lean_object* v___x_1472_; uint8_t v___x_1473_; 
v___x_1472_ = lean_array_get_size(v_a_1468_);
v___x_1473_ = lean_nat_dec_eq(v___x_1472_, v___x_1458_);
if (v___x_1473_ == 0)
{
lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1478_; 
v___x_1474_ = lean_array_to_list(v_a_1468_);
v___x_1475_ = lean_obj_once(&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__3, &lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__3_once, _init_lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__3);
v___x_1476_ = l_Lean_MessageData_joinSep(v___x_1474_, v___x_1475_);
if (v_isShared_1471_ == 0)
{
lean_ctor_set(v___x_1470_, 0, v___x_1476_);
v___x_1478_ = v___x_1470_;
goto v_reusejp_1477_;
}
else
{
lean_object* v_reuseFailAlloc_1479_; 
v_reuseFailAlloc_1479_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1479_, 0, v___x_1476_);
v___x_1478_ = v_reuseFailAlloc_1479_;
goto v_reusejp_1477_;
}
v_reusejp_1477_:
{
return v___x_1478_;
}
}
else
{
lean_object* v___x_1480_; lean_object* v___x_1482_; 
lean_dec(v_a_1468_);
v___x_1480_ = lean_obj_once(&lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__6, &lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__6_once, _init_lp_mathlib_Mathlib_WhatsNew_whatsNew___closed__6);
if (v_isShared_1471_ == 0)
{
lean_ctor_set(v___x_1470_, 0, v___x_1480_);
v___x_1482_ = v___x_1470_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v___x_1480_);
v___x_1482_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1481_;
}
v_reusejp_1481_:
{
return v___x_1482_;
}
}
}
}
else
{
lean_object* v_a_1485_; lean_object* v___x_1487_; uint8_t v_isShared_1488_; uint8_t v_isSharedCheck_1492_; 
v_a_1485_ = lean_ctor_get(v___x_1467_, 0);
v_isSharedCheck_1492_ = !lean_is_exclusive(v___x_1467_);
if (v_isSharedCheck_1492_ == 0)
{
v___x_1487_ = v___x_1467_;
v_isShared_1488_ = v_isSharedCheck_1492_;
goto v_resetjp_1486_;
}
else
{
lean_inc(v_a_1485_);
lean_dec(v___x_1467_);
v___x_1487_ = lean_box(0);
v_isShared_1488_ = v_isSharedCheck_1492_;
goto v_resetjp_1486_;
}
v_resetjp_1486_:
{
lean_object* v___x_1490_; 
if (v_isShared_1488_ == 0)
{
v___x_1490_ = v___x_1487_;
goto v_reusejp_1489_;
}
else
{
lean_object* v_reuseFailAlloc_1491_; 
v_reuseFailAlloc_1491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1491_, 0, v_a_1485_);
v___x_1490_ = v_reuseFailAlloc_1491_;
goto v_reusejp_1489_;
}
v_reusejp_1489_:
{
return v___x_1490_;
}
}
}
}
else
{
lean_object* v_a_1493_; lean_object* v___x_1495_; uint8_t v_isShared_1496_; uint8_t v_isSharedCheck_1500_; 
lean_dec_ref(v_new_1452_);
lean_dec_ref(v_old_1451_);
v_a_1493_ = lean_ctor_get(v___x_1461_, 0);
v_isSharedCheck_1500_ = !lean_is_exclusive(v___x_1461_);
if (v_isSharedCheck_1500_ == 0)
{
v___x_1495_ = v___x_1461_;
v_isShared_1496_ = v_isSharedCheck_1500_;
goto v_resetjp_1494_;
}
else
{
lean_inc(v_a_1493_);
lean_dec(v___x_1461_);
v___x_1495_ = lean_box(0);
v_isShared_1496_ = v_isSharedCheck_1500_;
goto v_resetjp_1494_;
}
v_resetjp_1494_:
{
lean_object* v___x_1498_; 
if (v_isShared_1496_ == 0)
{
v___x_1498_ = v___x_1495_;
goto v_reusejp_1497_;
}
else
{
lean_object* v_reuseFailAlloc_1499_; 
v_reuseFailAlloc_1499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1499_, 0, v_a_1493_);
v___x_1498_ = v_reuseFailAlloc_1499_;
goto v_reusejp_1497_;
}
v_reusejp_1497_:
{
return v___x_1498_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew_whatsNew___boxed(lean_object* v_old_1501_, lean_object* v_new_1502_, lean_object* v_a_1503_, lean_object* v_a_1504_, lean_object* v_a_1505_){
_start:
{
lean_object* v_res_1506_; 
v_res_1506_ = lp_mathlib_Mathlib_WhatsNew_whatsNew(v_old_1501_, v_new_1502_, v_a_1503_, v_a_1504_);
lean_dec(v_a_1504_);
lean_dec_ref(v_a_1503_);
return v_res_1506_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0(lean_object* v_00_u03b2_1507_, lean_object* v_x_1508_, lean_object* v_x_1509_){
_start:
{
uint8_t v___x_1510_; 
v___x_1510_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___redArg(v_x_1508_, v_x_1509_);
return v___x_1510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0___boxed(lean_object* v_00_u03b2_1511_, lean_object* v_x_1512_, lean_object* v_x_1513_){
_start:
{
uint8_t v_res_1514_; lean_object* v_r_1515_; 
v_res_1514_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0(v_00_u03b2_1511_, v_x_1512_, v_x_1513_);
lean_dec(v_x_1513_);
lean_dec_ref(v_x_1512_);
v_r_1515_ = lean_box(v_res_1514_);
return v_r_1515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1(lean_object* v_00_u03b2_1516_, lean_object* v_m_1517_){
_start:
{
lean_object* v___x_1518_; 
v___x_1518_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___redArg(v_m_1517_);
return v___x_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1___boxed(lean_object* v_00_u03b2_1519_, lean_object* v_m_1520_){
_start:
{
lean_object* v_res_1521_; 
v_res_1521_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1(v_00_u03b2_1519_, v_m_1520_);
lean_dec_ref(v_m_1520_);
return v_res_1521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2(lean_object* v_old_1522_, lean_object* v_as_1523_, lean_object* v_as_x27_1524_, lean_object* v_b_1525_, lean_object* v_a_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_){
_start:
{
lean_object* v___x_1530_; 
v___x_1530_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___redArg(v_old_1522_, v_as_x27_1524_, v_b_1525_, v___y_1527_, v___y_1528_);
return v___x_1530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2___boxed(lean_object* v_old_1531_, lean_object* v_as_1532_, lean_object* v_as_x27_1533_, lean_object* v_b_1534_, lean_object* v_a_1535_, lean_object* v___y_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_){
_start:
{
lean_object* v_res_1539_; 
v_res_1539_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_WhatsNew_whatsNew_spec__2(v_old_1531_, v_as_1532_, v_as_x27_1533_, v_b_1534_, v_a_1535_, v___y_1536_, v___y_1537_);
lean_dec(v___y_1537_);
lean_dec_ref(v___y_1536_);
lean_dec(v_as_x27_1533_);
lean_dec(v_as_1532_);
return v_res_1539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3(lean_object* v_old_1540_, lean_object* v_new_1541_, lean_object* v_as_1542_, size_t v_sz_1543_, size_t v_i_1544_, lean_object* v_b_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v___x_1549_; 
v___x_1549_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___redArg(v_old_1540_, v_new_1541_, v_as_1542_, v_sz_1543_, v_i_1544_, v_b_1545_, v___y_1547_);
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3___boxed(lean_object* v_old_1550_, lean_object* v_new_1551_, lean_object* v_as_1552_, lean_object* v_sz_1553_, lean_object* v_i_1554_, lean_object* v_b_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_){
_start:
{
size_t v_sz_boxed_1559_; size_t v_i_boxed_1560_; lean_object* v_res_1561_; 
v_sz_boxed_1559_ = lean_unbox_usize(v_sz_1553_);
lean_dec(v_sz_1553_);
v_i_boxed_1560_ = lean_unbox_usize(v_i_1554_);
lean_dec(v_i_1554_);
v_res_1561_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_WhatsNew_whatsNew_spec__3(v_old_1550_, v_new_1551_, v_as_1552_, v_sz_boxed_1559_, v_i_boxed_1560_, v_b_1555_, v___y_1556_, v___y_1557_);
lean_dec(v___y_1557_);
lean_dec_ref(v___y_1556_);
lean_dec_ref(v_as_1552_);
return v_res_1561_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0(lean_object* v_00_u03b2_1562_, lean_object* v_x_1563_, size_t v_x_1564_, lean_object* v_x_1565_){
_start:
{
uint8_t v___x_1566_; 
v___x_1566_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___redArg(v_x_1563_, v_x_1564_, v_x_1565_);
return v___x_1566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1567_, lean_object* v_x_1568_, lean_object* v_x_1569_, lean_object* v_x_1570_){
_start:
{
size_t v_x_2970__boxed_1571_; uint8_t v_res_1572_; lean_object* v_r_1573_; 
v_x_2970__boxed_1571_ = lean_unbox_usize(v_x_1569_);
lean_dec(v_x_1569_);
v_res_1572_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0(v_00_u03b2_1567_, v_x_1568_, v_x_2970__boxed_1571_, v_x_1570_);
lean_dec(v_x_1570_);
lean_dec_ref(v_x_1568_);
v_r_1573_ = lean_box(v_res_1572_);
return v_r_1573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2(lean_object* v_00_u03c3_1574_, lean_object* v_00_u03b2_1575_, lean_object* v_map_1576_, lean_object* v_f_1577_, lean_object* v_init_1578_){
_start:
{
lean_object* v___x_1579_; 
v___x_1579_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___redArg(v_map_1576_, v_f_1577_, v_init_1578_);
return v___x_1579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2___boxed(lean_object* v_00_u03c3_1580_, lean_object* v_00_u03b2_1581_, lean_object* v_map_1582_, lean_object* v_f_1583_, lean_object* v_init_1584_){
_start:
{
lean_object* v_res_1585_; 
v_res_1585_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2(v_00_u03c3_1580_, v_00_u03b2_1581_, v_map_1582_, v_f_1583_, v_init_1584_);
lean_dec_ref(v_map_1582_);
return v_res_1585_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1586_, lean_object* v_keys_1587_, lean_object* v_vals_1588_, lean_object* v_heq_1589_, lean_object* v_i_1590_, lean_object* v_k_1591_){
_start:
{
uint8_t v___x_1592_; 
v___x_1592_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___redArg(v_keys_1587_, v_i_1590_, v_k_1591_);
return v___x_1592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1593_, lean_object* v_keys_1594_, lean_object* v_vals_1595_, lean_object* v_heq_1596_, lean_object* v_i_1597_, lean_object* v_k_1598_){
_start:
{
uint8_t v_res_1599_; lean_object* v_r_1600_; 
v_res_1599_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_WhatsNew_whatsNew_spec__0_spec__0_spec__1(v_00_u03b2_1593_, v_keys_1594_, v_vals_1595_, v_heq_1596_, v_i_1597_, v_k_1598_);
lean_dec(v_k_1598_);
lean_dec_ref(v_vals_1595_);
lean_dec_ref(v_keys_1594_);
v_r_1600_ = lean_box(v_res_1599_);
return v_r_1600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___redArg(lean_object* v_map_1601_, lean_object* v_f_1602_, lean_object* v_init_1603_){
_start:
{
lean_object* v___x_1604_; 
v___x_1604_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(v_f_1602_, v_map_1601_, v_init_1603_);
return v___x_1604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_map_1605_, lean_object* v_f_1606_, lean_object* v_init_1607_){
_start:
{
lean_object* v_res_1608_; 
v_res_1608_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___redArg(v_map_1605_, v_f_1606_, v_init_1607_);
lean_dec_ref(v_map_1605_);
return v_res_1608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4(lean_object* v_00_u03c3_1609_, lean_object* v_00_u03b2_1610_, lean_object* v_map_1611_, lean_object* v_f_1612_, lean_object* v_init_1613_){
_start:
{
lean_object* v___x_1614_; 
v___x_1614_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(v_f_1612_, v_map_1611_, v_init_1613_);
return v___x_1614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03c3_1615_, lean_object* v_00_u03b2_1616_, lean_object* v_map_1617_, lean_object* v_f_1618_, lean_object* v_init_1619_){
_start:
{
lean_object* v_res_1620_; 
v_res_1620_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4(v_00_u03c3_1615_, v_00_u03b2_1616_, v_map_1617_, v_f_1618_, v_init_1619_);
lean_dec_ref(v_map_1617_);
return v_res_1620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7(lean_object* v_00_u03c3_1621_, lean_object* v_00_u03b1_1622_, lean_object* v_00_u03b2_1623_, lean_object* v_f_1624_, lean_object* v_x_1625_, lean_object* v_x_1626_){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___redArg(v_f_1624_, v_x_1625_, v_x_1626_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7___boxed(lean_object* v_00_u03c3_1628_, lean_object* v_00_u03b1_1629_, lean_object* v_00_u03b2_1630_, lean_object* v_f_1631_, lean_object* v_x_1632_, lean_object* v_x_1633_){
_start:
{
lean_object* v_res_1634_; 
v_res_1634_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7(v_00_u03c3_1628_, v_00_u03b1_1629_, v_00_u03b2_1630_, v_f_1631_, v_x_1632_, v_x_1633_);
lean_dec_ref(v_x_1632_);
return v_res_1634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8(lean_object* v_00_u03b1_1635_, lean_object* v_00_u03b2_1636_, lean_object* v_00_u03c3_1637_, lean_object* v_f_1638_, lean_object* v_as_1639_, size_t v_i_1640_, size_t v_stop_1641_, lean_object* v_b_1642_){
_start:
{
lean_object* v___x_1643_; 
v___x_1643_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___redArg(v_f_1638_, v_as_1639_, v_i_1640_, v_stop_1641_, v_b_1642_);
return v___x_1643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8___boxed(lean_object* v_00_u03b1_1644_, lean_object* v_00_u03b2_1645_, lean_object* v_00_u03c3_1646_, lean_object* v_f_1647_, lean_object* v_as_1648_, lean_object* v_i_1649_, lean_object* v_stop_1650_, lean_object* v_b_1651_){
_start:
{
size_t v_i_boxed_1652_; size_t v_stop_boxed_1653_; lean_object* v_res_1654_; 
v_i_boxed_1652_ = lean_unbox_usize(v_i_1649_);
lean_dec(v_i_1649_);
v_stop_boxed_1653_ = lean_unbox_usize(v_stop_1650_);
lean_dec(v_stop_1650_);
v_res_1654_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__8(v_00_u03b1_1644_, v_00_u03b2_1645_, v_00_u03c3_1646_, v_f_1647_, v_as_1648_, v_i_boxed_1652_, v_stop_boxed_1653_, v_b_1651_);
lean_dec_ref(v_as_1648_);
return v_res_1654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9(lean_object* v_00_u03c3_1655_, lean_object* v_00_u03b1_1656_, lean_object* v_00_u03b2_1657_, lean_object* v_f_1658_, lean_object* v_keys_1659_, lean_object* v_vals_1660_, lean_object* v_heq_1661_, lean_object* v_i_1662_, lean_object* v_acc_1663_){
_start:
{
lean_object* v___x_1664_; 
v___x_1664_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___redArg(v_f_1658_, v_keys_1659_, v_vals_1660_, v_i_1662_, v_acc_1663_);
return v___x_1664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9___boxed(lean_object* v_00_u03c3_1665_, lean_object* v_00_u03b1_1666_, lean_object* v_00_u03b2_1667_, lean_object* v_f_1668_, lean_object* v_keys_1669_, lean_object* v_vals_1670_, lean_object* v_heq_1671_, lean_object* v_i_1672_, lean_object* v_acc_1673_){
_start:
{
lean_object* v_res_1674_; 
v_res_1674_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Mathlib_WhatsNew_whatsNew_spec__1_spec__2_spec__4_spec__7_spec__9(v_00_u03c3_1665_, v_00_u03b1_1666_, v_00_u03b2_1667_, v_f_1668_, v_keys_1669_, v_vals_1670_, v_heq_1671_, v_i_1672_, v_acc_1673_);
lean_dec_ref(v_vals_1670_);
lean_dec_ref(v_keys_1669_);
return v_res_1674_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1725_ = lean_box(0);
v___x_1726_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1727_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1727_, 0, v___x_1726_);
lean_ctor_set(v___x_1727_, 1, v___x_1725_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg(){
_start:
{
lean_object* v___x_1729_; lean_object* v___x_1730_; 
v___x_1729_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___closed__0);
v___x_1730_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1730_, 0, v___x_1729_);
return v___x_1730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg___boxed(lean_object* v___y_1731_){
_start:
{
lean_object* v_res_1732_; 
v_res_1732_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg();
return v_res_1732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0(lean_object* v_00_u03b1_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_){
_start:
{
lean_object* v___x_1737_; 
v___x_1737_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg();
return v___x_1737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___boxed(lean_object* v_00_u03b1_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_){
_start:
{
lean_object* v_res_1742_; 
v_res_1742_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0(v_00_u03b1_1738_, v___y_1739_, v___y_1740_);
lean_dec(v___y_1740_);
lean_dec_ref(v___y_1739_);
return v_res_1742_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0(uint8_t v___y_1744_, uint8_t v_suppressElabErrors_1745_, lean_object* v_x_1746_){
_start:
{
if (lean_obj_tag(v_x_1746_) == 1)
{
lean_object* v_pre_1747_; 
v_pre_1747_ = lean_ctor_get(v_x_1746_, 0);
if (lean_obj_tag(v_pre_1747_) == 0)
{
lean_object* v_str_1748_; lean_object* v___x_1749_; uint8_t v___x_1750_; 
v_str_1748_ = lean_ctor_get(v_x_1746_, 1);
v___x_1749_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___closed__0));
v___x_1750_ = lean_string_dec_eq(v_str_1748_, v___x_1749_);
if (v___x_1750_ == 0)
{
return v___y_1744_;
}
else
{
return v_suppressElabErrors_1745_;
}
}
else
{
return v___y_1744_;
}
}
else
{
return v___y_1744_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___boxed(lean_object* v___y_1751_, lean_object* v_suppressElabErrors_1752_, lean_object* v_x_1753_){
_start:
{
uint8_t v___y_2921__boxed_1754_; uint8_t v_suppressElabErrors_boxed_1755_; uint8_t v_res_1756_; lean_object* v_r_1757_; 
v___y_2921__boxed_1754_ = lean_unbox(v___y_1751_);
v_suppressElabErrors_boxed_1755_ = lean_unbox(v_suppressElabErrors_1752_);
v_res_1756_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0(v___y_2921__boxed_1754_, v_suppressElabErrors_boxed_1755_, v_x_1753_);
lean_dec(v_x_1753_);
v_r_1757_ = lean_box(v_res_1756_);
return v_r_1757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2(lean_object* v_ref_1758_, lean_object* v_msgData_1759_, uint8_t v_severity_1760_, uint8_t v_isSilent_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
lean_object* v___y_1766_; uint8_t v___y_1767_; uint8_t v___y_1768_; lean_object* v___y_1769_; lean_object* v___y_1770_; lean_object* v___y_1771_; lean_object* v___y_1772_; lean_object* v___y_1773_; uint8_t v___y_1830_; uint8_t v___y_1831_; uint8_t v___y_1832_; lean_object* v___y_1833_; lean_object* v___y_1834_; uint8_t v___y_1858_; lean_object* v___y_1859_; uint8_t v___y_1860_; uint8_t v___y_1861_; lean_object* v___y_1862_; uint8_t v___y_1866_; uint8_t v___y_1867_; uint8_t v___y_1868_; uint8_t v___x_1883_; uint8_t v___y_1885_; uint8_t v___y_1886_; uint8_t v___y_1887_; uint8_t v___y_1889_; uint8_t v___x_1901_; 
v___x_1883_ = 2;
v___x_1901_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1760_, v___x_1883_);
if (v___x_1901_ == 0)
{
v___y_1889_ = v___x_1901_;
goto v___jp_1888_;
}
else
{
uint8_t v___x_1902_; 
lean_inc_ref(v_msgData_1759_);
v___x_1902_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1759_);
v___y_1889_ = v___x_1902_;
goto v___jp_1888_;
}
v___jp_1765_:
{
lean_object* v___x_1774_; 
v___x_1774_ = l_Lean_Elab_Command_getScope___redArg(v___y_1773_);
if (lean_obj_tag(v___x_1774_) == 0)
{
lean_object* v_a_1775_; lean_object* v___x_1776_; 
v_a_1775_ = lean_ctor_get(v___x_1774_, 0);
lean_inc(v_a_1775_);
lean_dec_ref_known(v___x_1774_, 1);
v___x_1776_ = l_Lean_Elab_Command_getScope___redArg(v___y_1773_);
if (lean_obj_tag(v___x_1776_) == 0)
{
lean_object* v_a_1777_; lean_object* v___x_1779_; uint8_t v_isShared_1780_; uint8_t v_isSharedCheck_1812_; 
v_a_1777_ = lean_ctor_get(v___x_1776_, 0);
v_isSharedCheck_1812_ = !lean_is_exclusive(v___x_1776_);
if (v_isSharedCheck_1812_ == 0)
{
v___x_1779_ = v___x_1776_;
v_isShared_1780_ = v_isSharedCheck_1812_;
goto v_resetjp_1778_;
}
else
{
lean_inc(v_a_1777_);
lean_dec(v___x_1776_);
v___x_1779_ = lean_box(0);
v_isShared_1780_ = v_isSharedCheck_1812_;
goto v_resetjp_1778_;
}
v_resetjp_1778_:
{
lean_object* v___x_1781_; lean_object* v_currNamespace_1782_; lean_object* v_openDecls_1783_; lean_object* v_env_1784_; lean_object* v_messages_1785_; lean_object* v_scopes_1786_; lean_object* v_usedQuotCtxts_1787_; lean_object* v_nextMacroScope_1788_; lean_object* v_maxRecDepth_1789_; lean_object* v_ngen_1790_; lean_object* v_auxDeclNGen_1791_; lean_object* v_infoState_1792_; lean_object* v_traceState_1793_; lean_object* v_snapshotTasks_1794_; lean_object* v_prevLinterStates_1795_; lean_object* v___x_1797_; uint8_t v_isShared_1798_; uint8_t v_isSharedCheck_1811_; 
v___x_1781_ = lean_st_ref_take(v___y_1773_);
v_currNamespace_1782_ = lean_ctor_get(v_a_1775_, 2);
lean_inc(v_currNamespace_1782_);
lean_dec(v_a_1775_);
v_openDecls_1783_ = lean_ctor_get(v_a_1777_, 3);
lean_inc(v_openDecls_1783_);
lean_dec(v_a_1777_);
v_env_1784_ = lean_ctor_get(v___x_1781_, 0);
v_messages_1785_ = lean_ctor_get(v___x_1781_, 1);
v_scopes_1786_ = lean_ctor_get(v___x_1781_, 2);
v_usedQuotCtxts_1787_ = lean_ctor_get(v___x_1781_, 3);
v_nextMacroScope_1788_ = lean_ctor_get(v___x_1781_, 4);
v_maxRecDepth_1789_ = lean_ctor_get(v___x_1781_, 5);
v_ngen_1790_ = lean_ctor_get(v___x_1781_, 6);
v_auxDeclNGen_1791_ = lean_ctor_get(v___x_1781_, 7);
v_infoState_1792_ = lean_ctor_get(v___x_1781_, 8);
v_traceState_1793_ = lean_ctor_get(v___x_1781_, 9);
v_snapshotTasks_1794_ = lean_ctor_get(v___x_1781_, 10);
v_prevLinterStates_1795_ = lean_ctor_get(v___x_1781_, 11);
v_isSharedCheck_1811_ = !lean_is_exclusive(v___x_1781_);
if (v_isSharedCheck_1811_ == 0)
{
v___x_1797_ = v___x_1781_;
v_isShared_1798_ = v_isSharedCheck_1811_;
goto v_resetjp_1796_;
}
else
{
lean_inc(v_prevLinterStates_1795_);
lean_inc(v_snapshotTasks_1794_);
lean_inc(v_traceState_1793_);
lean_inc(v_infoState_1792_);
lean_inc(v_auxDeclNGen_1791_);
lean_inc(v_ngen_1790_);
lean_inc(v_maxRecDepth_1789_);
lean_inc(v_nextMacroScope_1788_);
lean_inc(v_usedQuotCtxts_1787_);
lean_inc(v_scopes_1786_);
lean_inc(v_messages_1785_);
lean_inc(v_env_1784_);
lean_dec(v___x_1781_);
v___x_1797_ = lean_box(0);
v_isShared_1798_ = v_isSharedCheck_1811_;
goto v_resetjp_1796_;
}
v_resetjp_1796_:
{
lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1804_; 
v___x_1799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1799_, 0, v_currNamespace_1782_);
lean_ctor_set(v___x_1799_, 1, v_openDecls_1783_);
v___x_1800_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1800_, 0, v___x_1799_);
lean_ctor_set(v___x_1800_, 1, v___y_1766_);
lean_inc_ref(v___y_1769_);
lean_inc_ref(v___y_1770_);
v___x_1801_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1801_, 0, v___y_1770_);
lean_ctor_set(v___x_1801_, 1, v___y_1771_);
lean_ctor_set(v___x_1801_, 2, v___y_1772_);
lean_ctor_set(v___x_1801_, 3, v___y_1769_);
lean_ctor_set(v___x_1801_, 4, v___x_1800_);
lean_ctor_set_uint8(v___x_1801_, sizeof(void*)*5, v___y_1768_);
lean_ctor_set_uint8(v___x_1801_, sizeof(void*)*5 + 1, v___y_1767_);
lean_ctor_set_uint8(v___x_1801_, sizeof(void*)*5 + 2, v_isSilent_1761_);
v___x_1802_ = l_Lean_MessageLog_add(v___x_1801_, v_messages_1785_);
if (v_isShared_1798_ == 0)
{
lean_ctor_set(v___x_1797_, 1, v___x_1802_);
v___x_1804_ = v___x_1797_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1810_; 
v_reuseFailAlloc_1810_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1810_, 0, v_env_1784_);
lean_ctor_set(v_reuseFailAlloc_1810_, 1, v___x_1802_);
lean_ctor_set(v_reuseFailAlloc_1810_, 2, v_scopes_1786_);
lean_ctor_set(v_reuseFailAlloc_1810_, 3, v_usedQuotCtxts_1787_);
lean_ctor_set(v_reuseFailAlloc_1810_, 4, v_nextMacroScope_1788_);
lean_ctor_set(v_reuseFailAlloc_1810_, 5, v_maxRecDepth_1789_);
lean_ctor_set(v_reuseFailAlloc_1810_, 6, v_ngen_1790_);
lean_ctor_set(v_reuseFailAlloc_1810_, 7, v_auxDeclNGen_1791_);
lean_ctor_set(v_reuseFailAlloc_1810_, 8, v_infoState_1792_);
lean_ctor_set(v_reuseFailAlloc_1810_, 9, v_traceState_1793_);
lean_ctor_set(v_reuseFailAlloc_1810_, 10, v_snapshotTasks_1794_);
lean_ctor_set(v_reuseFailAlloc_1810_, 11, v_prevLinterStates_1795_);
v___x_1804_ = v_reuseFailAlloc_1810_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1808_; 
v___x_1805_ = lean_st_ref_set(v___y_1773_, v___x_1804_);
v___x_1806_ = lean_box(0);
if (v_isShared_1780_ == 0)
{
lean_ctor_set(v___x_1779_, 0, v___x_1806_);
v___x_1808_ = v___x_1779_;
goto v_reusejp_1807_;
}
else
{
lean_object* v_reuseFailAlloc_1809_; 
v_reuseFailAlloc_1809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1809_, 0, v___x_1806_);
v___x_1808_ = v_reuseFailAlloc_1809_;
goto v_reusejp_1807_;
}
v_reusejp_1807_:
{
return v___x_1808_;
}
}
}
}
}
else
{
lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1820_; 
lean_dec(v_a_1775_);
lean_dec(v___y_1772_);
lean_dec_ref(v___y_1771_);
lean_dec_ref(v___y_1766_);
v_a_1813_ = lean_ctor_get(v___x_1776_, 0);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1776_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1815_ = v___x_1776_;
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1776_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1818_; 
if (v_isShared_1816_ == 0)
{
v___x_1818_ = v___x_1815_;
goto v_reusejp_1817_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v_a_1813_);
v___x_1818_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1817_;
}
v_reusejp_1817_:
{
return v___x_1818_;
}
}
}
}
else
{
lean_object* v_a_1821_; lean_object* v___x_1823_; uint8_t v_isShared_1824_; uint8_t v_isSharedCheck_1828_; 
lean_dec(v___y_1772_);
lean_dec_ref(v___y_1771_);
lean_dec_ref(v___y_1766_);
v_a_1821_ = lean_ctor_get(v___x_1774_, 0);
v_isSharedCheck_1828_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1828_ == 0)
{
v___x_1823_ = v___x_1774_;
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
else
{
lean_inc(v_a_1821_);
lean_dec(v___x_1774_);
v___x_1823_ = lean_box(0);
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
v_resetjp_1822_:
{
lean_object* v___x_1826_; 
if (v_isShared_1824_ == 0)
{
v___x_1826_ = v___x_1823_;
goto v_reusejp_1825_;
}
else
{
lean_object* v_reuseFailAlloc_1827_; 
v_reuseFailAlloc_1827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1827_, 0, v_a_1821_);
v___x_1826_ = v_reuseFailAlloc_1827_;
goto v_reusejp_1825_;
}
v_reusejp_1825_:
{
return v___x_1826_;
}
}
}
}
v___jp_1829_:
{
lean_object* v_fileName_1835_; lean_object* v_fileMap_1836_; uint8_t v_suppressElabErrors_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v_a_1840_; lean_object* v___x_1842_; uint8_t v_isShared_1843_; uint8_t v_isSharedCheck_1856_; 
v_fileName_1835_ = lean_ctor_get(v___y_1762_, 0);
v_fileMap_1836_ = lean_ctor_get(v___y_1762_, 1);
v_suppressElabErrors_1837_ = lean_ctor_get_uint8(v___y_1762_, sizeof(void*)*10);
v___x_1838_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1759_);
v___x_1839_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__0___redArg(v___x_1838_, v___y_1763_);
v_a_1840_ = lean_ctor_get(v___x_1839_, 0);
v_isSharedCheck_1856_ = !lean_is_exclusive(v___x_1839_);
if (v_isSharedCheck_1856_ == 0)
{
v___x_1842_ = v___x_1839_;
v_isShared_1843_ = v_isSharedCheck_1856_;
goto v_resetjp_1841_;
}
else
{
lean_inc(v_a_1840_);
lean_dec(v___x_1839_);
v___x_1842_ = lean_box(0);
v_isShared_1843_ = v_isSharedCheck_1856_;
goto v_resetjp_1841_;
}
v_resetjp_1841_:
{
lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; 
lean_inc_ref_n(v_fileMap_1836_, 2);
v___x_1844_ = l_Lean_FileMap_toPosition(v_fileMap_1836_, v___y_1833_);
lean_dec(v___y_1833_);
v___x_1845_ = l_Lean_FileMap_toPosition(v_fileMap_1836_, v___y_1834_);
lean_dec(v___y_1834_);
v___x_1846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1846_, 0, v___x_1845_);
v___x_1847_ = ((lean_object*)(lp_mathlib___private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_levelParamsToMessageData___closed__0));
if (v_suppressElabErrors_1837_ == 0)
{
lean_del_object(v___x_1842_);
v___y_1766_ = v_a_1840_;
v___y_1767_ = v___y_1832_;
v___y_1768_ = v___y_1831_;
v___y_1769_ = v___x_1847_;
v___y_1770_ = v_fileName_1835_;
v___y_1771_ = v___x_1844_;
v___y_1772_ = v___x_1846_;
v___y_1773_ = v___y_1763_;
goto v___jp_1765_;
}
else
{
lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___f_1850_; uint8_t v___x_1851_; 
v___x_1848_ = lean_box(v___y_1830_);
v___x_1849_ = lean_box(v_suppressElabErrors_1837_);
v___f_1850_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1850_, 0, v___x_1848_);
lean_closure_set(v___f_1850_, 1, v___x_1849_);
lean_inc(v_a_1840_);
v___x_1851_ = l_Lean_MessageData_hasTag(v___f_1850_, v_a_1840_);
if (v___x_1851_ == 0)
{
lean_object* v___x_1852_; lean_object* v___x_1854_; 
lean_dec_ref_known(v___x_1846_, 1);
lean_dec_ref(v___x_1844_);
lean_dec(v_a_1840_);
v___x_1852_ = lean_box(0);
if (v_isShared_1843_ == 0)
{
lean_ctor_set(v___x_1842_, 0, v___x_1852_);
v___x_1854_ = v___x_1842_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1855_; 
v_reuseFailAlloc_1855_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1855_, 0, v___x_1852_);
v___x_1854_ = v_reuseFailAlloc_1855_;
goto v_reusejp_1853_;
}
v_reusejp_1853_:
{
return v___x_1854_;
}
}
else
{
lean_del_object(v___x_1842_);
v___y_1766_ = v_a_1840_;
v___y_1767_ = v___y_1832_;
v___y_1768_ = v___y_1831_;
v___y_1769_ = v___x_1847_;
v___y_1770_ = v_fileName_1835_;
v___y_1771_ = v___x_1844_;
v___y_1772_ = v___x_1846_;
v___y_1773_ = v___y_1763_;
goto v___jp_1765_;
}
}
}
}
v___jp_1857_:
{
lean_object* v___x_1863_; 
v___x_1863_ = l_Lean_Syntax_getTailPos_x3f(v___y_1859_, v___y_1861_);
lean_dec(v___y_1859_);
if (lean_obj_tag(v___x_1863_) == 0)
{
lean_inc(v___y_1862_);
v___y_1830_ = v___y_1858_;
v___y_1831_ = v___y_1861_;
v___y_1832_ = v___y_1860_;
v___y_1833_ = v___y_1862_;
v___y_1834_ = v___y_1862_;
goto v___jp_1829_;
}
else
{
lean_object* v_val_1864_; 
v_val_1864_ = lean_ctor_get(v___x_1863_, 0);
lean_inc(v_val_1864_);
lean_dec_ref_known(v___x_1863_, 1);
v___y_1830_ = v___y_1858_;
v___y_1831_ = v___y_1861_;
v___y_1832_ = v___y_1860_;
v___y_1833_ = v___y_1862_;
v___y_1834_ = v_val_1864_;
goto v___jp_1829_;
}
}
v___jp_1865_:
{
lean_object* v___x_1869_; 
v___x_1869_ = l_Lean_Elab_Command_getRef___redArg(v___y_1762_);
if (lean_obj_tag(v___x_1869_) == 0)
{
lean_object* v_a_1870_; lean_object* v_ref_1871_; lean_object* v___x_1872_; 
v_a_1870_ = lean_ctor_get(v___x_1869_, 0);
lean_inc(v_a_1870_);
lean_dec_ref_known(v___x_1869_, 1);
v_ref_1871_ = l_Lean_replaceRef(v_ref_1758_, v_a_1870_);
lean_dec(v_a_1870_);
v___x_1872_ = l_Lean_Syntax_getPos_x3f(v_ref_1871_, v___y_1867_);
if (lean_obj_tag(v___x_1872_) == 0)
{
lean_object* v___x_1873_; 
v___x_1873_ = lean_unsigned_to_nat(0u);
v___y_1858_ = v___y_1866_;
v___y_1859_ = v_ref_1871_;
v___y_1860_ = v___y_1868_;
v___y_1861_ = v___y_1867_;
v___y_1862_ = v___x_1873_;
goto v___jp_1857_;
}
else
{
lean_object* v_val_1874_; 
v_val_1874_ = lean_ctor_get(v___x_1872_, 0);
lean_inc(v_val_1874_);
lean_dec_ref_known(v___x_1872_, 1);
v___y_1858_ = v___y_1866_;
v___y_1859_ = v_ref_1871_;
v___y_1860_ = v___y_1868_;
v___y_1861_ = v___y_1867_;
v___y_1862_ = v_val_1874_;
goto v___jp_1857_;
}
}
else
{
lean_object* v_a_1875_; lean_object* v___x_1877_; uint8_t v_isShared_1878_; uint8_t v_isSharedCheck_1882_; 
lean_dec_ref(v_msgData_1759_);
v_a_1875_ = lean_ctor_get(v___x_1869_, 0);
v_isSharedCheck_1882_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1882_ == 0)
{
v___x_1877_ = v___x_1869_;
v_isShared_1878_ = v_isSharedCheck_1882_;
goto v_resetjp_1876_;
}
else
{
lean_inc(v_a_1875_);
lean_dec(v___x_1869_);
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
v___jp_1884_:
{
if (v___y_1887_ == 0)
{
v___y_1866_ = v___y_1885_;
v___y_1867_ = v___y_1886_;
v___y_1868_ = v_severity_1760_;
goto v___jp_1865_;
}
else
{
v___y_1866_ = v___y_1885_;
v___y_1867_ = v___y_1886_;
v___y_1868_ = v___x_1883_;
goto v___jp_1865_;
}
}
v___jp_1888_:
{
if (v___y_1889_ == 0)
{
lean_object* v___x_1890_; lean_object* v_scopes_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v_opts_1894_; uint8_t v___x_1895_; uint8_t v___x_1896_; 
v___x_1890_ = lean_st_ref_get(v___y_1763_);
v_scopes_1891_ = lean_ctor_get(v___x_1890_, 2);
lean_inc(v_scopes_1891_);
lean_dec(v___x_1890_);
v___x_1892_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1893_ = l_List_head_x21___redArg(v___x_1892_, v_scopes_1891_);
lean_dec(v_scopes_1891_);
v_opts_1894_ = lean_ctor_get(v___x_1893_, 1);
lean_inc_ref(v_opts_1894_);
lean_dec(v___x_1893_);
v___x_1895_ = 1;
v___x_1896_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1760_, v___x_1895_);
if (v___x_1896_ == 0)
{
lean_dec_ref(v_opts_1894_);
v___y_1885_ = v___y_1889_;
v___y_1886_ = v___y_1889_;
v___y_1887_ = v___x_1896_;
goto v___jp_1884_;
}
else
{
lean_object* v___x_1897_; uint8_t v___x_1898_; 
v___x_1897_ = l_Lean_warningAsError;
v___x_1898_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Util_WhatsNew_0__Mathlib_WhatsNew_throwUnknownId_spec__0_spec__1_spec__2(v_opts_1894_, v___x_1897_);
lean_dec_ref(v_opts_1894_);
v___y_1885_ = v___y_1889_;
v___y_1886_ = v___y_1889_;
v___y_1887_ = v___x_1898_;
goto v___jp_1884_;
}
}
else
{
lean_object* v___x_1899_; lean_object* v___x_1900_; 
lean_dec_ref(v_msgData_1759_);
v___x_1899_ = lean_box(0);
v___x_1900_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1900_, 0, v___x_1899_);
return v___x_1900_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_1903_, lean_object* v_msgData_1904_, lean_object* v_severity_1905_, lean_object* v_isSilent_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_){
_start:
{
uint8_t v_severity_boxed_1910_; uint8_t v_isSilent_boxed_1911_; lean_object* v_res_1912_; 
v_severity_boxed_1910_ = lean_unbox(v_severity_1905_);
v_isSilent_boxed_1911_ = lean_unbox(v_isSilent_1906_);
v_res_1912_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2(v_ref_1903_, v_msgData_1904_, v_severity_boxed_1910_, v_isSilent_boxed_1911_, v___y_1907_, v___y_1908_);
lean_dec(v___y_1908_);
lean_dec_ref(v___y_1907_);
lean_dec(v_ref_1903_);
return v_res_1912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1(lean_object* v_msgData_1913_, uint8_t v_severity_1914_, uint8_t v_isSilent_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_){
_start:
{
lean_object* v___x_1919_; 
v___x_1919_ = l_Lean_Elab_Command_getRef___redArg(v___y_1916_);
if (lean_obj_tag(v___x_1919_) == 0)
{
lean_object* v_a_1920_; lean_object* v___x_1921_; 
v_a_1920_ = lean_ctor_get(v___x_1919_, 0);
lean_inc(v_a_1920_);
lean_dec_ref_known(v___x_1919_, 1);
v___x_1921_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1_spec__2(v_a_1920_, v_msgData_1913_, v_severity_1914_, v_isSilent_1915_, v___y_1916_, v___y_1917_);
lean_dec(v_a_1920_);
return v___x_1921_;
}
else
{
lean_object* v_a_1922_; lean_object* v___x_1924_; uint8_t v_isShared_1925_; uint8_t v_isSharedCheck_1929_; 
lean_dec_ref(v_msgData_1913_);
v_a_1922_ = lean_ctor_get(v___x_1919_, 0);
v_isSharedCheck_1929_ = !lean_is_exclusive(v___x_1919_);
if (v_isSharedCheck_1929_ == 0)
{
v___x_1924_ = v___x_1919_;
v_isShared_1925_ = v_isSharedCheck_1929_;
goto v_resetjp_1923_;
}
else
{
lean_inc(v_a_1922_);
lean_dec(v___x_1919_);
v___x_1924_ = lean_box(0);
v_isShared_1925_ = v_isSharedCheck_1929_;
goto v_resetjp_1923_;
}
v_resetjp_1923_:
{
lean_object* v___x_1927_; 
if (v_isShared_1925_ == 0)
{
v___x_1927_ = v___x_1924_;
goto v_reusejp_1926_;
}
else
{
lean_object* v_reuseFailAlloc_1928_; 
v_reuseFailAlloc_1928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1928_, 0, v_a_1922_);
v___x_1927_ = v_reuseFailAlloc_1928_;
goto v_reusejp_1926_;
}
v_reusejp_1926_:
{
return v___x_1927_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1___boxed(lean_object* v_msgData_1930_, lean_object* v_severity_1931_, lean_object* v_isSilent_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_){
_start:
{
uint8_t v_severity_boxed_1936_; uint8_t v_isSilent_boxed_1937_; lean_object* v_res_1938_; 
v_severity_boxed_1936_ = lean_unbox(v_severity_1931_);
v_isSilent_boxed_1937_ = lean_unbox(v_isSilent_1932_);
v_res_1938_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1(v_msgData_1930_, v_severity_boxed_1936_, v_isSilent_boxed_1937_, v___y_1933_, v___y_1934_);
lean_dec(v___y_1934_);
lean_dec_ref(v___y_1933_);
return v_res_1938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1(lean_object* v_msgData_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_){
_start:
{
uint8_t v___x_1943_; uint8_t v___x_1944_; lean_object* v___x_1945_; 
v___x_1943_ = 0;
v___x_1944_ = 0;
v___x_1945_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1_spec__1(v_msgData_1939_, v___x_1943_, v___x_1944_, v___y_1940_, v___y_1941_);
return v___x_1945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1___boxed(lean_object* v_msgData_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_){
_start:
{
lean_object* v_res_1950_; 
v_res_1950_ = lp_mathlib_Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1(v_msgData_1946_, v___y_1947_, v___y_1948_);
lean_dec(v___y_1948_);
lean_dec_ref(v___y_1947_);
return v_res_1950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0(lean_object* v_a_1951_, lean_object* v_env_1952_, lean_object* v_a_1953_, lean_object* v_a_x3f_1954_){
_start:
{
lean_object* v___x_1956_; lean_object* v_env_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; 
v___x_1956_ = lean_st_ref_get(v_a_1951_);
v_env_1957_ = lean_ctor_get(v___x_1956_, 0);
lean_inc_ref(v_env_1957_);
lean_dec(v___x_1956_);
v___x_1958_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_WhatsNew_whatsNew___boxed), 5, 2);
lean_closure_set(v___x_1958_, 0, v_env_1952_);
lean_closure_set(v___x_1958_, 1, v_env_1957_);
v___x_1959_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_1958_, v_a_1953_, v_a_1951_);
if (lean_obj_tag(v___x_1959_) == 0)
{
lean_object* v_a_1960_; lean_object* v___x_1961_; 
v_a_1960_ = lean_ctor_get(v___x_1959_, 0);
lean_inc(v_a_1960_);
lean_dec_ref_known(v___x_1959_, 1);
v___x_1961_ = lp_mathlib_Lean_logInfo___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__1(v_a_1960_, v_a_1953_, v_a_1951_);
return v___x_1961_;
}
else
{
lean_object* v_a_1962_; lean_object* v___x_1964_; uint8_t v_isShared_1965_; uint8_t v_isSharedCheck_1969_; 
v_a_1962_ = lean_ctor_get(v___x_1959_, 0);
v_isSharedCheck_1969_ = !lean_is_exclusive(v___x_1959_);
if (v_isSharedCheck_1969_ == 0)
{
v___x_1964_ = v___x_1959_;
v_isShared_1965_ = v_isSharedCheck_1969_;
goto v_resetjp_1963_;
}
else
{
lean_inc(v_a_1962_);
lean_dec(v___x_1959_);
v___x_1964_ = lean_box(0);
v_isShared_1965_ = v_isSharedCheck_1969_;
goto v_resetjp_1963_;
}
v_resetjp_1963_:
{
lean_object* v___x_1967_; 
if (v_isShared_1965_ == 0)
{
v___x_1967_ = v___x_1964_;
goto v_reusejp_1966_;
}
else
{
lean_object* v_reuseFailAlloc_1968_; 
v_reuseFailAlloc_1968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1968_, 0, v_a_1962_);
v___x_1967_ = v_reuseFailAlloc_1968_;
goto v_reusejp_1966_;
}
v_reusejp_1966_:
{
return v___x_1967_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0___boxed(lean_object* v_a_1970_, lean_object* v_env_1971_, lean_object* v_a_1972_, lean_object* v_a_x3f_1973_, lean_object* v___y_1974_){
_start:
{
lean_object* v_res_1975_; 
v_res_1975_ = lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0(v_a_1970_, v_env_1971_, v_a_1972_, v_a_x3f_1973_);
lean_dec(v_a_x3f_1973_);
lean_dec_ref(v_a_1972_);
lean_dec(v_a_1970_);
return v_res_1975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1(lean_object* v_x_1976_, lean_object* v_a_1977_, lean_object* v_a_1978_){
_start:
{
lean_object* v___x_1980_; uint8_t v___x_1981_; 
v___x_1980_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3));
lean_inc(v_x_1976_);
v___x_1981_ = l_Lean_Syntax_isOfKind(v_x_1976_, v___x_1980_);
if (v___x_1981_ == 0)
{
lean_object* v___x_1982_; 
lean_dec(v_x_1976_);
v___x_1982_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1_spec__0___redArg();
return v___x_1982_;
}
else
{
lean_object* v___x_1983_; lean_object* v_env_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v_r_1987_; 
v___x_1983_ = lean_st_ref_get(v_a_1978_);
v_env_1984_ = lean_ctor_get(v___x_1983_, 0);
lean_inc_ref(v_env_1984_);
lean_dec(v___x_1983_);
v___x_1985_ = lean_unsigned_to_nat(3u);
v___x_1986_ = l_Lean_Syntax_getArg(v_x_1976_, v___x_1985_);
lean_dec(v_x_1976_);
v_r_1987_ = l_Lean_Elab_Command_elabCommand(v___x_1986_, v_a_1977_, v_a_1978_);
if (lean_obj_tag(v_r_1987_) == 0)
{
lean_object* v_a_1988_; lean_object* v___x_1990_; uint8_t v_isShared_1991_; uint8_t v_isSharedCheck_2004_; 
v_a_1988_ = lean_ctor_get(v_r_1987_, 0);
v_isSharedCheck_2004_ = !lean_is_exclusive(v_r_1987_);
if (v_isSharedCheck_2004_ == 0)
{
v___x_1990_ = v_r_1987_;
v_isShared_1991_ = v_isSharedCheck_2004_;
goto v_resetjp_1989_;
}
else
{
lean_inc(v_a_1988_);
lean_dec(v_r_1987_);
v___x_1990_ = lean_box(0);
v_isShared_1991_ = v_isSharedCheck_2004_;
goto v_resetjp_1989_;
}
v_resetjp_1989_:
{
lean_object* v___x_1993_; 
lean_inc(v_a_1988_);
if (v_isShared_1991_ == 0)
{
lean_ctor_set_tag(v___x_1990_, 1);
v___x_1993_ = v___x_1990_;
goto v_reusejp_1992_;
}
else
{
lean_object* v_reuseFailAlloc_2003_; 
v_reuseFailAlloc_2003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2003_, 0, v_a_1988_);
v___x_1993_ = v_reuseFailAlloc_2003_;
goto v_reusejp_1992_;
}
v_reusejp_1992_:
{
lean_object* v___x_1994_; 
v___x_1994_ = lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0(v_a_1978_, v_env_1984_, v_a_1977_, v___x_1993_);
lean_dec_ref(v___x_1993_);
if (lean_obj_tag(v___x_1994_) == 0)
{
lean_object* v___x_1996_; uint8_t v_isShared_1997_; uint8_t v_isSharedCheck_2001_; 
v_isSharedCheck_2001_ = !lean_is_exclusive(v___x_1994_);
if (v_isSharedCheck_2001_ == 0)
{
lean_object* v_unused_2002_; 
v_unused_2002_ = lean_ctor_get(v___x_1994_, 0);
lean_dec(v_unused_2002_);
v___x_1996_ = v___x_1994_;
v_isShared_1997_ = v_isSharedCheck_2001_;
goto v_resetjp_1995_;
}
else
{
lean_dec(v___x_1994_);
v___x_1996_ = lean_box(0);
v_isShared_1997_ = v_isSharedCheck_2001_;
goto v_resetjp_1995_;
}
v_resetjp_1995_:
{
lean_object* v___x_1999_; 
if (v_isShared_1997_ == 0)
{
lean_ctor_set(v___x_1996_, 0, v_a_1988_);
v___x_1999_ = v___x_1996_;
goto v_reusejp_1998_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v_a_1988_);
v___x_1999_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1998_;
}
v_reusejp_1998_:
{
return v___x_1999_;
}
}
}
else
{
lean_dec(v_a_1988_);
return v___x_1994_;
}
}
}
}
else
{
lean_object* v_a_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; 
v_a_2005_ = lean_ctor_get(v_r_1987_, 0);
lean_inc(v_a_2005_);
lean_dec_ref_known(v_r_1987_, 1);
v___x_2006_ = lean_box(0);
v___x_2007_ = lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___lam__0(v_a_1978_, v_env_1984_, v_a_1977_, v___x_2006_);
if (lean_obj_tag(v___x_2007_) == 0)
{
lean_object* v___x_2009_; uint8_t v_isShared_2010_; uint8_t v_isSharedCheck_2014_; 
v_isSharedCheck_2014_ = !lean_is_exclusive(v___x_2007_);
if (v_isSharedCheck_2014_ == 0)
{
lean_object* v_unused_2015_; 
v_unused_2015_ = lean_ctor_get(v___x_2007_, 0);
lean_dec(v_unused_2015_);
v___x_2009_ = v___x_2007_;
v_isShared_2010_ = v_isSharedCheck_2014_;
goto v_resetjp_2008_;
}
else
{
lean_dec(v___x_2007_);
v___x_2009_ = lean_box(0);
v_isShared_2010_ = v_isSharedCheck_2014_;
goto v_resetjp_2008_;
}
v_resetjp_2008_:
{
lean_object* v___x_2012_; 
if (v_isShared_2010_ == 0)
{
lean_ctor_set_tag(v___x_2009_, 1);
lean_ctor_set(v___x_2009_, 0, v_a_2005_);
v___x_2012_ = v___x_2009_;
goto v_reusejp_2011_;
}
else
{
lean_object* v_reuseFailAlloc_2013_; 
v_reuseFailAlloc_2013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2013_, 0, v_a_2005_);
v___x_2012_ = v_reuseFailAlloc_2013_;
goto v_reusejp_2011_;
}
v_reusejp_2011_:
{
return v___x_2012_;
}
}
}
else
{
lean_dec(v_a_2005_);
return v___x_2007_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1___boxed(lean_object* v_x_2016_, lean_object* v_a_2017_, lean_object* v_a_2018_, lean_object* v_a_2019_){
_start:
{
lean_object* v_res_2020_; 
v_res_2020_ = lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______elabRules__Mathlib__WhatsNew__command_x23whats__newIn______1(v_x_2016_, v_a_2017_, v_a_2018_);
lean_dec(v_a_2018_);
lean_dec_ref(v_a_2017_);
return v_res_2020_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__1(void){
_start:
{
lean_object* v___x_2047_; 
v___x_2047_ = l_Array_mkArray0(lean_box(0));
return v___x_2047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1(lean_object* v_x_2048_, lean_object* v_a_2049_, lean_object* v_a_2050_){
_start:
{
lean_object* v___x_2051_; uint8_t v___x_2052_; 
v___x_2051_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_oldStx___closed__1));
lean_inc(v_x_2048_);
v___x_2052_ = l_Lean_Syntax_isOfKind(v_x_2048_, v___x_2051_);
if (v___x_2052_ == 0)
{
lean_object* v___x_2053_; lean_object* v___x_2054_; 
lean_dec(v_x_2048_);
v___x_2053_ = lean_box(1);
v___x_2054_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2054_, 0, v___x_2053_);
lean_ctor_set(v___x_2054_, 1, v_a_2050_);
return v___x_2054_;
}
else
{
lean_object* v_ref_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; uint8_t v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; 
v_ref_2055_ = lean_ctor_get(v_a_2049_, 5);
v___x_2056_ = lean_unsigned_to_nat(3u);
v___x_2057_ = l_Lean_Syntax_getArg(v_x_2048_, v___x_2056_);
lean_dec(v_x_2048_);
v___x_2058_ = 0;
v___x_2059_ = l_Lean_SourceInfo_fromRef(v_ref_2055_, v___x_2058_);
v___x_2060_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__3));
v___x_2061_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__0));
lean_inc_n(v___x_2059_, 3);
v___x_2062_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2059_);
lean_ctor_set(v___x_2062_, 1, v___x_2061_);
v___x_2063_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__8));
v___x_2064_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2064_, 0, v___x_2059_);
lean_ctor_set(v___x_2064_, 1, v___x_2063_);
v___x_2065_ = ((lean_object*)(lp_mathlib_Mathlib_WhatsNew_command_x23whats__newIn_____00__closed__12));
v___x_2066_ = lean_obj_once(&lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__1, &lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__1_once, _init_lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___closed__1);
v___x_2067_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2067_, 0, v___x_2059_);
lean_ctor_set(v___x_2067_, 1, v___x_2065_);
lean_ctor_set(v___x_2067_, 2, v___x_2066_);
v___x_2068_ = l_Lean_Syntax_node4(v___x_2059_, v___x_2060_, v___x_2062_, v___x_2064_, v___x_2067_, v___x_2057_);
v___x_2069_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2069_, 0, v___x_2068_);
lean_ctor_set(v___x_2069_, 1, v_a_2050_);
return v___x_2069_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1___boxed(lean_object* v_x_2070_, lean_object* v_a_2071_, lean_object* v_a_2072_){
_start:
{
lean_object* v_res_2073_; 
v_res_2073_ = lp_mathlib_Mathlib_WhatsNew___aux__Mathlib__Util__WhatsNew______macroRules__Mathlib__WhatsNew__oldStx__1(v_x_2070_, v_a_2071_, v_a_2072_);
lean_dec_ref(v_a_2071_);
return v_res_2073_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_WhatsNew(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_WhatsNew(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_WhatsNew(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_WhatsNew(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_WhatsNew(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_WhatsNew(builtin);
}
#ifdef __cplusplus
}
#endif
