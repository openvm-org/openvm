// Lean compiler output
// Module: Batteries.Tactic.Lint.Frontend
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Batteries.Tactic.Lint.Basic
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
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_task_get_own(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lean_get_stdout();
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
extern lean_object* l_Lean_Elab_Command_mkMetaContext;
size_t lean_array_size(lean_object*);
lean_object* l_Lean_MessageData_format(lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Environment_allImportedModuleNames(lean_object*);
lean_object* l_Lean_SearchPath_findWithExt(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_modToFilePath(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_mainModule(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Name_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Array_instInhabited(lean_object*);
extern lean_object* lp_batteries_Batteries_Tactic_Lint_nolintAttr;
lean_object* l_Lean_ParametricAttribute_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_builtinDeclRanges;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_isRecCore(lean_object*, lean_object*);
lean_object* l_Lean_Name_getPrefix(lean_object*);
extern lean_object* l_Lean_declRangeExt;
extern lean_object* l_Lean_instInhabitedDeclarationRanges_default;
lean_object* l_Lean_MapDeclarationExtension_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_isAuxRecursor(lean_object*, lean_object*);
uint8_t l_Lean_isNoConfusion(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_Tactic_Lint_getLinter(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_const2ModIdx(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Environment_constants(lean_object*);
uint8_t lp_batteries_Lean_Environment_isAutoDecl(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_getSrcSearchPath();
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
extern lean_object* l_Lean_LocalContext_empty;
lean_object* lean_string_append(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_Core_wrapAsync___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_as_task(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity_default;
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity;
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "Batteries.Tactic.Lint.LintVerbosity.low"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__0_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Batteries.Tactic.Lint.LintVerbosity.medium"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Batteries.Tactic.Lint.LintVerbosity.high"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity___closed__0_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity___closed__0_value;
LEAN_EXPORT uint8_t lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_Tactic_Lint_getChecks___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_getChecks___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_getChecks___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getChecks(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getChecks___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14_spec__18(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__0;
static const lean_array_object lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__0;
static const lean_string_object lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1 = (const lean_object*)&lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1_value;
static const lean_array_object lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__2 = (const lean_object*)&lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__21(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__0;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__1;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6(lean_object*, uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lint"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2_value_aux_0),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 145, 97, 148, 215, 213, 156, 19)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__3 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__3_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__4_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "- "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "(0/2) Starting..."};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__8 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__8_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11_value_aux_0),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(85, 241, 153, 159, 236, 59, 207, 249)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 160, 226, 71, 160, 244, 82, 120)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3_spec__16___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "LINTER FAILED:\n"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "(2/2) "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Failed with "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__1_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " messages"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__2_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__3 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__3_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Passed!"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__4_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = ", but these may include declarations in `nolints.json`"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__5 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__5_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "(1/2) Getting..."};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__8 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__8_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_lintCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Completed linting!"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_lintCore___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_lintCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Running linters:\n  "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_lintCore___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_lintCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\n  "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_lintCore___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13;
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__1;
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_printWarning___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#check "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarning___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__1;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_printWarning___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " /- "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarning___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_printWarning___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " -/"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_printWarning___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__6_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarning___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__7;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_printWarning___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = ": error: "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarning___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__9;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_printWarning___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__10_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarning___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___closed__11;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_printWarnings_spec__0(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_printWarnings_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarnings(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarnings___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "-- "};
static const lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__0 = (const lean_object*)&lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__0_value;
static lean_once_cell_t lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__1;
static const lean_string_object lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__2 = (const lean_object*)&lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__2_value;
static lean_once_cell_t lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___closed__0 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_groupedByFilename___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_groupedByFilename___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_groupedByFilename(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_groupedByFilename___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "/- The `"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__1;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "` linter reports:\n"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__3;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "\nThis linter can be disabled with `@[nolint "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__4_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__5;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "]`. -/\n"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__6 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__6_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__7;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "/- OK: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__8 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__8_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__9;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0(uint8_t, uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0(uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_formatLinterResults_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_formatLinterResults_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "-- (slow linters skipped)\n"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__1;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = " declarations (plus "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__5;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = " automatically generated ones) "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__6_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__7;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__9;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " linters\n\n"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__10_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__11;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "-- Found "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__12_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__13;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " error"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__14_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__15;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "s"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__16_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getAllDecls_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__0 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__0_value;
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__1 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__1_value;
static const lean_string_object lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__2 = (const lean_object*)&lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__2_value;
static lean_once_cell_t lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "inProject"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__0_value),LEAN_SCALAR_PTR_LITERAL(137, 32, 238, 10, 188, 210, 254, 16)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__6_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_inProject___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__0_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__2_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__10_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Lint_inProject = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__10_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "command#lint+-*Only___"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 86, 230, 46, 156, 83, 223, 163)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#lint"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "+"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "token"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__10_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(207, 29, 99, 203, 176, 199, 84, 49)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__10_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__12_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__14_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(208, 60, 148, 146, 72, 117, 42, 6)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__14_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__12_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__17_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__19_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__20_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__21 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__21_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__21_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__22 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__22_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__19_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__22_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__23 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__23_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " only"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__24 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__24_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__24_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__25 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__25_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__25_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__26 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__26_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__23_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__26_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__27 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__27_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__28 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__28_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__28_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__29 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__29_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__30 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__30_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__30_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__31 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__31_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__31_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__32 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__32_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__32_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__33 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__33_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__29_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__33_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__34 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__34_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__27_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__34_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__35 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__35_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__36 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__36_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__35_value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__36_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__37 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__37_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__37_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__38 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__38_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly______ = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__38_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__0;
static lean_once_cell_t lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "not a linter: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__5(uint8_t, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "\n-- All linting checks passed!"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__1;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "in the current file"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "all"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(135, 186, 94, 176, 136, 38, 52, 11)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "in "};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "in all files"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__6_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "command#list_linters"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value_aux_1),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(175, 64, 250, 82, 196, 254, 167, 174)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__0_value),LEAN_SCALAR_PTR_LITERAL(68, 158, 25, 32, 158, 126, 206, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "#list_linters"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__4_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Lint_command_x23list__linters = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__4_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " (*)"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "Available linters (linters marked with (*) are in the default lint set):"};
static const lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__0_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__1_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 175, 18, 163, 178, 203, 59, 243)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__2_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value),LEAN_SCALAR_PTR_LITERAL(47, 92, 120, 195, 85, 23, 119, 138)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__3_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(211, 182, 232, 123, 14, 144, 43, 55)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__4_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(110, 214, 239, 54, 226, 242, 192, 165)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__6_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(23, 176, 97, 178, 35, 194, 224, 179)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__7_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(42, 160, 215, 230, 106, 94, 174, 244)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__8_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value),LEAN_SCALAR_PTR_LITERAL(127, 222, 37, 189, 92, 195, 191, 19)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__9_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(227, 148, 230, 216, 194, 224, 9, 164)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__10_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__11_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(66, 36, 110, 40, 195, 123, 154, 100)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__12_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__13_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 31, 12, 10, 80, 138, 155, 17)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__14_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(182, 242, 78, 145, 63, 185, 209, 4)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__15_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries_Batteries_Tactic_Lint_inProject___closed__1_value),LEAN_SCALAR_PTR_LITERAL(75, 6, 191, 127, 161, 67, 124, 163)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__16_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__1_value),LEAN_SCALAR_PTR_LITERAL(223, 159, 78, 199, 153, 223, 210, 200)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__17_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__5_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 223, 41, 251, 113, 127, 162, 91)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__18_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)(((size_t)(971841226) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(11, 176, 219, 228, 142, 149, 195, 38)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__19_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__20_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(112, 204, 226, 41, 208, 193, 254, 80)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__21_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__22_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 169, 197, 88, 108, 73, 163, 223)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__23_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(77, 188, 221, 254, 209, 123, 206, 102)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx(uint8_t v_x_1_){
_start:
{
switch(v_x_1_)
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
uint8_t v_x_boxed_6_; lean_object* v_res_7_; 
v_x_boxed_6_ = lean_unbox(v_x_5_);
v_res_7_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx(v_x_boxed_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___redArg(lean_object* v_k_8_){
_start:
{
lean_inc(v_k_8_);
return v_k_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___redArg___boxed(lean_object* v_k_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___redArg(v_k_9_);
lean_dec(v_k_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, uint8_t v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_inc(v_k_15_);
return v_k_15_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
uint8_t v_t_boxed_21_; lean_object* v_res_22_; 
v_t_boxed_21_ = lean_unbox(v_t_18_);
v_res_22_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_boxed_21_, v_h_19_, v_k_20_);
lean_dec(v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___redArg(lean_object* v_low_23_){
_start:
{
lean_inc(v_low_23_);
return v_low_23_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___redArg___boxed(lean_object* v_low_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___redArg(v_low_24_);
lean_dec(v_low_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim(lean_object* v_motive_26_, uint8_t v_t_27_, lean_object* v_h_28_, lean_object* v_low_29_){
_start:
{
lean_inc(v_low_29_);
return v_low_29_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim___boxed(lean_object* v_motive_30_, lean_object* v_t_31_, lean_object* v_h_32_, lean_object* v_low_33_){
_start:
{
uint8_t v_t_boxed_34_; lean_object* v_res_35_; 
v_t_boxed_34_ = lean_unbox(v_t_31_);
v_res_35_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_low_elim(v_motive_30_, v_t_boxed_34_, v_h_32_, v_low_33_);
lean_dec(v_low_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___redArg(lean_object* v_medium_36_){
_start:
{
lean_inc(v_medium_36_);
return v_medium_36_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___redArg___boxed(lean_object* v_medium_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___redArg(v_medium_37_);
lean_dec(v_medium_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim(lean_object* v_motive_39_, uint8_t v_t_40_, lean_object* v_h_41_, lean_object* v_medium_42_){
_start:
{
lean_inc(v_medium_42_);
return v_medium_42_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim___boxed(lean_object* v_motive_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_medium_46_){
_start:
{
uint8_t v_t_boxed_47_; lean_object* v_res_48_; 
v_t_boxed_47_ = lean_unbox(v_t_44_);
v_res_48_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_medium_elim(v_motive_43_, v_t_boxed_47_, v_h_45_, v_medium_46_);
lean_dec(v_medium_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___redArg(lean_object* v_high_49_){
_start:
{
lean_inc(v_high_49_);
return v_high_49_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___redArg___boxed(lean_object* v_high_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___redArg(v_high_50_);
lean_dec(v_high_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim(lean_object* v_motive_52_, uint8_t v_t_53_, lean_object* v_h_54_, lean_object* v_high_55_){
_start:
{
lean_inc(v_high_55_);
return v_high_55_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim___boxed(lean_object* v_motive_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_high_59_){
_start:
{
uint8_t v_t_boxed_60_; lean_object* v_res_61_; 
v_t_boxed_60_ = lean_unbox(v_t_57_);
v_res_61_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_high_elim(v_motive_56_, v_t_boxed_60_, v_h_58_, v_high_59_);
lean_dec(v_high_59_);
return v_res_61_;
}
}
static uint8_t _init_lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity_default(void){
_start:
{
uint8_t v___x_62_; 
v___x_62_ = 0;
return v___x_62_;
}
}
static uint8_t _init_lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity(void){
_start:
{
uint8_t v___x_63_; 
v___x_63_ = 0;
return v___x_63_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ofNat(lean_object* v_n_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = lean_nat_dec_le(v_n_64_, v___x_65_);
if (v___x_66_ == 0)
{
lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = lean_nat_dec_le(v_n_64_, v___x_67_);
if (v___x_68_ == 0)
{
uint8_t v___x_69_; 
v___x_69_ = 2;
return v___x_69_;
}
else
{
uint8_t v___x_70_; 
v___x_70_ = 1;
return v___x_70_;
}
}
else
{
uint8_t v___x_71_; 
v___x_71_ = 0;
return v___x_71_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ofNat___boxed(lean_object* v_n_72_){
_start:
{
uint8_t v_res_73_; lean_object* v_r_74_; 
v_res_73_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ofNat(v_n_72_);
lean_dec(v_n_72_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity(uint8_t v_x_75_, uint8_t v_y_76_){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; uint8_t v___x_79_; 
v___x_77_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx(v_x_75_);
v___x_78_ = lp_batteries_Batteries_Tactic_Lint_LintVerbosity_ctorIdx(v_y_76_);
v___x_79_ = lean_nat_dec_eq(v___x_77_, v___x_78_);
lean_dec(v___x_78_);
lean_dec(v___x_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity___boxed(lean_object* v_x_80_, lean_object* v_y_81_){
_start:
{
uint8_t v_x_13__boxed_82_; uint8_t v_y_14__boxed_83_; uint8_t v_res_84_; lean_object* v_r_85_; 
v_x_13__boxed_82_ = lean_unbox(v_x_80_);
v_y_14__boxed_83_ = lean_unbox(v_y_81_);
v_res_84_ = lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity(v_x_13__boxed_82_, v_y_14__boxed_83_);
v_r_85_ = lean_box(v_res_84_);
return v_r_85_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = lean_unsigned_to_nat(2u);
v___x_96_ = lean_nat_to_int(v___x_95_);
return v___x_96_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_97_ = lean_unsigned_to_nat(1u);
v___x_98_ = lean_nat_to_int(v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr(uint8_t v_x_99_, lean_object* v_prec_100_){
_start:
{
lean_object* v___y_102_; lean_object* v___y_109_; lean_object* v___y_116_; 
switch(v_x_99_)
{
case 0:
{
lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_122_ = lean_unsigned_to_nat(1024u);
v___x_123_ = lean_nat_dec_le(v___x_122_, v_prec_100_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; 
v___x_124_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6, &lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6);
v___y_102_ = v___x_124_;
goto v___jp_101_;
}
else
{
lean_object* v___x_125_; 
v___x_125_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7, &lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7);
v___y_102_ = v___x_125_;
goto v___jp_101_;
}
}
case 1:
{
lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_126_ = lean_unsigned_to_nat(1024u);
v___x_127_ = lean_nat_dec_le(v___x_126_, v_prec_100_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; 
v___x_128_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6, &lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6);
v___y_109_ = v___x_128_;
goto v___jp_108_;
}
else
{
lean_object* v___x_129_; 
v___x_129_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7, &lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7);
v___y_109_ = v___x_129_;
goto v___jp_108_;
}
}
default: 
{
lean_object* v___x_130_; uint8_t v___x_131_; 
v___x_130_ = lean_unsigned_to_nat(1024u);
v___x_131_ = lean_nat_dec_le(v___x_130_, v_prec_100_);
if (v___x_131_ == 0)
{
lean_object* v___x_132_; 
v___x_132_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6, &lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6_once, _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__6);
v___y_116_ = v___x_132_;
goto v___jp_115_;
}
else
{
lean_object* v___x_133_; 
v___x_133_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7, &lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__7);
v___y_116_ = v___x_133_;
goto v___jp_115_;
}
}
}
v___jp_101_:
{
lean_object* v___x_103_; lean_object* v___x_104_; uint8_t v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_103_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__1));
lean_inc(v___y_102_);
v___x_104_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_104_, 0, v___y_102_);
lean_ctor_set(v___x_104_, 1, v___x_103_);
v___x_105_ = 0;
v___x_106_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_106_, 0, v___x_104_);
lean_ctor_set_uint8(v___x_106_, sizeof(void*)*1, v___x_105_);
v___x_107_ = l_Repr_addAppParen(v___x_106_, v_prec_100_);
return v___x_107_;
}
v___jp_108_:
{
lean_object* v___x_110_; lean_object* v___x_111_; uint8_t v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_110_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__3));
lean_inc(v___y_109_);
v___x_111_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_111_, 0, v___y_109_);
lean_ctor_set(v___x_111_, 1, v___x_110_);
v___x_112_ = 0;
v___x_113_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_113_, 0, v___x_111_);
lean_ctor_set_uint8(v___x_113_, sizeof(void*)*1, v___x_112_);
v___x_114_ = l_Repr_addAppParen(v___x_113_, v_prec_100_);
return v___x_114_;
}
v___jp_115_:
{
lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_117_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___closed__5));
lean_inc(v___y_116_);
v___x_118_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_118_, 0, v___y_116_);
lean_ctor_set(v___x_118_, 1, v___x_117_);
v___x_119_ = 0;
v___x_120_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_120_, 0, v___x_118_);
lean_ctor_set_uint8(v___x_120_, sizeof(void*)*1, v___x_119_);
v___x_121_ = l_Repr_addAppParen(v___x_120_, v_prec_100_);
return v___x_121_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr___boxed(lean_object* v_x_134_, lean_object* v_prec_135_){
_start:
{
uint8_t v_x_177__boxed_136_; lean_object* v_res_137_; 
v_x_177__boxed_136_ = lean_unbox(v_x_134_);
v_res_137_ = lp_batteries_Batteries_Tactic_Lint_instReprLintVerbosity_repr(v_x_177__boxed_136_, v_prec_135_);
lean_dec(v_prec_135_);
return v_res_137_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(lean_object* v_a_140_, lean_object* v_x_141_){
_start:
{
if (lean_obj_tag(v_x_141_) == 0)
{
uint8_t v___x_142_; 
v___x_142_ = 0;
return v___x_142_;
}
else
{
lean_object* v_head_143_; lean_object* v_tail_144_; uint8_t v___x_145_; 
v_head_143_ = lean_ctor_get(v_x_141_, 0);
v_tail_144_ = lean_ctor_get(v_x_141_, 1);
v___x_145_ = lean_name_eq(v_a_140_, v_head_143_);
if (v___x_145_ == 0)
{
v_x_141_ = v_tail_144_;
goto _start;
}
else
{
return v___x_145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1___boxed(lean_object* v_a_147_, lean_object* v_x_148_){
_start:
{
uint8_t v_res_149_; lean_object* v_r_150_; 
v_res_149_ = lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(v_a_147_, v_x_148_);
lean_dec(v_x_148_);
lean_dec(v_a_147_);
v_r_150_ = lean_box(v_res_149_);
return v_r_150_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(lean_object* v_x1_151_, lean_object* v_x2_152_){
_start:
{
lean_object* v_name_153_; lean_object* v_name_154_; uint8_t v___x_155_; 
v_name_153_ = lean_ctor_get(v_x1_151_, 1);
v_name_154_ = lean_ctor_get(v_x2_152_, 1);
v___x_155_ = l_Lean_Name_lt(v_name_153_, v_name_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0___boxed(lean_object* v_x1_156_, lean_object* v_x2_157_){
_start:
{
uint8_t v_res_158_; lean_object* v_r_159_; 
v_res_158_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v_x1_156_, v_x2_157_);
lean_dec_ref(v_x2_157_);
lean_dec_ref(v_x1_156_);
v_r_159_ = lean_box(v_res_158_);
return v_r_159_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg(lean_object* v_a_160_, lean_object* v_as_161_, lean_object* v_k_162_, lean_object* v_x_163_, lean_object* v_x_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v_mid_167_; lean_object* v_midVal_168_; uint8_t v___x_169_; 
v___x_165_ = lean_nat_add(v_x_163_, v_x_164_);
v___x_166_ = lean_unsigned_to_nat(1u);
v_mid_167_ = lean_nat_shiftr(v___x_165_, v___x_166_);
lean_dec(v___x_165_);
v_midVal_168_ = lean_array_fget_borrowed(v_as_161_, v_mid_167_);
v___x_169_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v_midVal_168_, v_k_162_);
if (v___x_169_ == 0)
{
uint8_t v___x_170_; 
lean_dec(v_x_164_);
v___x_170_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v_k_162_, v_midVal_168_);
if (v___x_170_ == 0)
{
lean_object* v___x_171_; uint8_t v___x_172_; 
lean_dec(v_x_163_);
v___x_171_ = lean_array_get_size(v_as_161_);
v___x_172_ = lean_nat_dec_lt(v_mid_167_, v___x_171_);
if (v___x_172_ == 0)
{
lean_dec(v_mid_167_);
lean_dec_ref(v_a_160_);
return v_as_161_;
}
else
{
lean_object* v___x_173_; lean_object* v_xs_x27_174_; lean_object* v___x_175_; 
v___x_173_ = lean_box(0);
v_xs_x27_174_ = lean_array_fset(v_as_161_, v_mid_167_, v___x_173_);
v___x_175_ = lean_array_fset(v_xs_x27_174_, v_mid_167_, v_a_160_);
lean_dec(v_mid_167_);
return v___x_175_;
}
}
else
{
v_x_164_ = v_mid_167_;
goto _start;
}
}
else
{
uint8_t v___x_177_; 
v___x_177_ = lean_nat_dec_eq(v_mid_167_, v_x_163_);
if (v___x_177_ == 0)
{
lean_dec(v_x_163_);
v_x_163_ = v_mid_167_;
goto _start;
}
else
{
lean_object* v___x_179_; lean_object* v_j_180_; lean_object* v_as_181_; lean_object* v___x_182_; 
lean_dec(v_mid_167_);
lean_dec(v_x_164_);
v___x_179_ = lean_nat_add(v_x_163_, v___x_166_);
lean_dec(v_x_163_);
v_j_180_ = lean_array_get_size(v_as_161_);
v_as_181_ = lean_array_push(v_as_161_, v_a_160_);
v___x_182_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_179_, v_as_181_, v_j_180_);
lean_dec(v___x_179_);
return v___x_182_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg___boxed(lean_object* v_a_183_, lean_object* v_as_184_, lean_object* v_k_185_, lean_object* v_x_186_, lean_object* v_x_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg(v_a_183_, v_as_184_, v_k_185_, v_x_186_, v_x_187_);
lean_dec_ref(v_k_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0(lean_object* v_a_189_, lean_object* v_as_190_, lean_object* v_k_191_){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; uint8_t v___x_194_; 
v___x_192_ = lean_array_get_size(v_as_190_);
v___x_193_ = lean_unsigned_to_nat(0u);
v___x_194_ = lean_nat_dec_eq(v___x_192_, v___x_193_);
if (v___x_194_ == 0)
{
lean_object* v___x_195_; uint8_t v___x_196_; 
v___x_195_ = lean_array_fget_borrowed(v_as_190_, v___x_193_);
v___x_196_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v_k_191_, v___x_195_);
if (v___x_196_ == 0)
{
uint8_t v___x_197_; 
v___x_197_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v___x_195_, v_k_191_);
if (v___x_197_ == 0)
{
uint8_t v___x_198_; 
v___x_198_ = lean_nat_dec_lt(v___x_193_, v___x_192_);
if (v___x_198_ == 0)
{
lean_dec_ref(v_a_189_);
return v_as_190_;
}
else
{
lean_object* v___x_199_; lean_object* v_xs_x27_200_; lean_object* v___x_201_; 
v___x_199_ = lean_box(0);
v_xs_x27_200_ = lean_array_fset(v_as_190_, v___x_193_, v___x_199_);
v___x_201_ = lean_array_fset(v_xs_x27_200_, v___x_193_, v_a_189_);
return v___x_201_;
}
}
else
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; uint8_t v___x_205_; 
v___x_202_ = lean_unsigned_to_nat(1u);
v___x_203_ = lean_nat_sub(v___x_192_, v___x_202_);
v___x_204_ = lean_array_fget_borrowed(v_as_190_, v___x_203_);
v___x_205_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v___x_204_, v_k_191_);
if (v___x_205_ == 0)
{
uint8_t v___x_206_; 
v___x_206_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___lam__0(v_k_191_, v___x_204_);
if (v___x_206_ == 0)
{
uint8_t v___x_207_; 
v___x_207_ = lean_nat_dec_lt(v___x_203_, v___x_192_);
if (v___x_207_ == 0)
{
lean_dec(v___x_203_);
lean_dec_ref(v_a_189_);
return v_as_190_;
}
else
{
lean_object* v___x_208_; lean_object* v_xs_x27_209_; lean_object* v___x_210_; 
v___x_208_ = lean_box(0);
v_xs_x27_209_ = lean_array_fset(v_as_190_, v___x_203_, v___x_208_);
v___x_210_ = lean_array_fset(v_xs_x27_209_, v___x_203_, v_a_189_);
lean_dec(v___x_203_);
return v___x_210_;
}
}
else
{
lean_object* v___x_211_; 
v___x_211_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg(v_a_189_, v_as_190_, v_k_191_, v___x_193_, v___x_203_);
return v___x_211_;
}
}
else
{
lean_object* v___x_212_; 
lean_dec(v___x_203_);
v___x_212_ = lean_array_push(v_as_190_, v_a_189_);
return v___x_212_;
}
}
}
else
{
lean_object* v_as_213_; lean_object* v___x_214_; 
v_as_213_ = lean_array_push(v_as_190_, v_a_189_);
v___x_214_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_193_, v_as_213_, v___x_192_);
return v___x_214_;
}
}
else
{
lean_object* v___x_215_; 
v___x_215_ = lean_array_push(v_as_190_, v_a_189_);
return v___x_215_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0___boxed(lean_object* v_a_216_, lean_object* v_as_217_, lean_object* v_k_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0(v_a_216_, v_as_217_, v_k_218_);
lean_dec_ref(v_k_218_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2(uint8_t v_slow_220_, lean_object* v_runOnly_221_, lean_object* v_runAlways_222_, lean_object* v_init_223_, lean_object* v_x_224_, lean_object* v___y_225_, lean_object* v___y_226_){
_start:
{
lean_object* v_d_229_; 
if (lean_obj_tag(v_x_224_) == 0)
{
lean_object* v_k_232_; lean_object* v_v_233_; lean_object* v_l_234_; lean_object* v_r_235_; lean_object* v___x_236_; 
v_k_232_ = lean_ctor_get(v_x_224_, 1);
lean_inc(v_k_232_);
v_v_233_ = lean_ctor_get(v_x_224_, 2);
lean_inc(v_v_233_);
v_l_234_ = lean_ctor_get(v_x_224_, 3);
lean_inc(v_l_234_);
v_r_235_ = lean_ctor_get(v_x_224_, 4);
lean_inc(v_r_235_);
lean_dec_ref_known(v_x_224_, 5);
v___x_236_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2(v_slow_220_, v_runOnly_221_, v_runAlways_222_, v_init_223_, v_l_234_, v___y_225_, v___y_226_);
if (lean_obj_tag(v___x_236_) == 0)
{
lean_object* v_a_237_; 
v_a_237_ = lean_ctor_get(v___x_236_, 0);
lean_inc(v_a_237_);
if (lean_obj_tag(v_a_237_) == 0)
{
lean_object* v_a_238_; 
lean_dec_ref_known(v___x_236_, 1);
lean_dec(v_r_235_);
lean_dec(v_v_233_);
lean_dec(v_k_232_);
v_a_238_ = lean_ctor_get(v_a_237_, 0);
lean_inc(v_a_238_);
lean_dec_ref_known(v_a_237_, 1);
v_d_229_ = v_a_238_;
goto v___jp_228_;
}
else
{
lean_object* v_a_239_; lean_object* v___y_241_; lean_object* v_fst_244_; lean_object* v_snd_245_; uint8_t v___y_262_; 
v_a_239_ = lean_ctor_get(v_a_237_, 0);
lean_inc(v_a_239_);
lean_dec_ref_known(v_a_237_, 1);
v_fst_244_ = lean_ctor_get(v_v_233_, 0);
lean_inc(v_fst_244_);
v_snd_245_ = lean_ctor_get(v_v_233_, 1);
lean_inc(v_snd_245_);
lean_dec(v_v_233_);
if (lean_obj_tag(v_runOnly_221_) == 0)
{
if (lean_obj_tag(v_runAlways_222_) == 1)
{
uint8_t v___x_267_; 
v___x_267_ = lean_unbox(v_snd_245_);
lean_dec(v_snd_245_);
if (v___x_267_ == 0)
{
lean_object* v_val_268_; uint8_t v___x_269_; 
v_val_268_ = lean_ctor_get(v_runAlways_222_, 0);
v___x_269_ = lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(v_k_232_, v_val_268_);
v___y_262_ = v___x_269_;
goto v___jp_261_;
}
else
{
lean_dec_ref_known(v___x_236_, 1);
goto v___jp_246_;
}
}
else
{
uint8_t v___x_270_; 
v___x_270_ = lean_unbox(v_snd_245_);
lean_dec(v_snd_245_);
v___y_262_ = v___x_270_;
goto v___jp_261_;
}
}
else
{
if (lean_obj_tag(v_runAlways_222_) == 0)
{
lean_object* v_val_271_; uint8_t v___x_272_; 
lean_dec(v_snd_245_);
v_val_271_ = lean_ctor_get(v_runOnly_221_, 0);
v___x_272_ = lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(v_k_232_, v_val_271_);
v___y_262_ = v___x_272_;
goto v___jp_261_;
}
else
{
lean_object* v_val_273_; lean_object* v_val_274_; uint8_t v___x_275_; 
v_val_273_ = lean_ctor_get(v_runOnly_221_, 0);
v_val_274_ = lean_ctor_get(v_runAlways_222_, 0);
v___x_275_ = lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(v_k_232_, v_val_273_);
if (v___x_275_ == 0)
{
lean_dec(v_snd_245_);
v___y_262_ = v___x_275_;
goto v___jp_261_;
}
else
{
uint8_t v___x_276_; 
v___x_276_ = lp_batteries_List_elem___at___00Batteries_Tactic_Lint_getChecks_spec__1(v_k_232_, v_val_274_);
if (v___x_276_ == 0)
{
uint8_t v___x_277_; 
v___x_277_ = lean_unbox(v_snd_245_);
lean_dec(v_snd_245_);
v___y_262_ = v___x_277_;
goto v___jp_261_;
}
else
{
lean_dec(v_snd_245_);
v___y_262_ = v___x_276_;
goto v___jp_261_;
}
}
}
}
v___jp_240_:
{
lean_object* v___x_242_; 
lean_inc_ref(v___y_241_);
v___x_242_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0(v___y_241_, v_a_239_, v___y_241_);
lean_dec_ref(v___y_241_);
v_init_223_ = v___x_242_;
v_x_224_ = v_r_235_;
goto _start;
}
v___jp_246_:
{
lean_object* v___x_247_; 
v___x_247_ = lp_batteries_Batteries_Tactic_Lint_getLinter(v_k_232_, v_fst_244_, v___y_225_, v___y_226_);
if (lean_obj_tag(v___x_247_) == 0)
{
if (v_slow_220_ == 0)
{
lean_object* v_a_248_; lean_object* v_toLinter_249_; uint8_t v_isFast_250_; 
v_a_248_ = lean_ctor_get(v___x_247_, 0);
lean_inc(v_a_248_);
lean_dec_ref_known(v___x_247_, 1);
v_toLinter_249_ = lean_ctor_get(v_a_248_, 0);
v_isFast_250_ = lean_ctor_get_uint8(v_toLinter_249_, sizeof(void*)*3);
if (v_isFast_250_ == 0)
{
lean_dec(v_a_248_);
v_init_223_ = v_a_239_;
v_x_224_ = v_r_235_;
goto _start;
}
else
{
v___y_241_ = v_a_248_;
goto v___jp_240_;
}
}
else
{
lean_object* v_a_252_; 
v_a_252_ = lean_ctor_get(v___x_247_, 0);
lean_inc(v_a_252_);
lean_dec_ref_known(v___x_247_, 1);
v___y_241_ = v_a_252_;
goto v___jp_240_;
}
}
else
{
lean_object* v_a_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_260_; 
lean_dec(v_a_239_);
lean_dec(v_r_235_);
v_a_253_ = lean_ctor_get(v___x_247_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v___x_247_);
if (v_isSharedCheck_260_ == 0)
{
v___x_255_ = v___x_247_;
v_isShared_256_ = v_isSharedCheck_260_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_a_253_);
lean_dec(v___x_247_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_260_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_258_; 
if (v_isShared_256_ == 0)
{
v___x_258_ = v___x_255_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v_a_253_);
v___x_258_ = v_reuseFailAlloc_259_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
return v___x_258_;
}
}
}
}
v___jp_261_:
{
if (v___y_262_ == 0)
{
lean_dec(v_fst_244_);
lean_dec(v_a_239_);
lean_dec(v_k_232_);
if (lean_obj_tag(v___x_236_) == 0)
{
lean_object* v_a_263_; 
v_a_263_ = lean_ctor_get(v___x_236_, 0);
lean_inc(v_a_263_);
lean_dec_ref_known(v___x_236_, 1);
if (lean_obj_tag(v_a_263_) == 0)
{
lean_object* v_a_264_; 
lean_dec(v_r_235_);
v_a_264_ = lean_ctor_get(v_a_263_, 0);
lean_inc(v_a_264_);
lean_dec_ref_known(v_a_263_, 1);
v_d_229_ = v_a_264_;
goto v___jp_228_;
}
else
{
lean_object* v_a_265_; 
v_a_265_ = lean_ctor_get(v_a_263_, 0);
lean_inc(v_a_265_);
lean_dec_ref_known(v_a_263_, 1);
v_init_223_ = v_a_265_;
v_x_224_ = v_r_235_;
goto _start;
}
}
else
{
lean_dec(v_r_235_);
return v___x_236_;
}
}
else
{
lean_dec_ref_known(v___x_236_, 1);
goto v___jp_246_;
}
}
}
}
else
{
lean_dec(v_r_235_);
lean_dec(v_v_233_);
lean_dec(v_k_232_);
return v___x_236_;
}
}
else
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_278_, 0, v_init_223_);
v___x_279_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
return v___x_279_;
}
v___jp_228_:
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_230_, 0, v_d_229_);
v___x_231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_231_, 0, v___x_230_);
return v___x_231_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2___boxed(lean_object* v_slow_280_, lean_object* v_runOnly_281_, lean_object* v_runAlways_282_, lean_object* v_init_283_, lean_object* v_x_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_){
_start:
{
uint8_t v_slow_boxed_288_; lean_object* v_res_289_; 
v_slow_boxed_288_ = lean_unbox(v_slow_280_);
v_res_289_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2(v_slow_boxed_288_, v_runOnly_281_, v_runAlways_282_, v_init_283_, v_x_284_, v___y_285_, v___y_286_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v_runAlways_282_);
lean_dec(v_runOnly_281_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getChecks(uint8_t v_slow_292_, lean_object* v_runOnly_293_, lean_object* v_runAlways_294_, lean_object* v_a_295_, lean_object* v_a_296_){
_start:
{
lean_object* v___x_298_; lean_object* v_env_299_; lean_object* v___x_300_; lean_object* v_toEnvExtension_301_; lean_object* v_asyncMode_302_; lean_object* v___x_303_; lean_object* v_result_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_298_ = lean_st_ref_get(v_a_296_);
v_env_299_ = lean_ctor_get(v___x_298_, 0);
lean_inc_ref(v_env_299_);
lean_dec(v___x_298_);
v___x_300_ = lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
v_toEnvExtension_301_ = lean_ctor_get(v___x_300_, 0);
v_asyncMode_302_ = lean_ctor_get(v_toEnvExtension_301_, 2);
v___x_303_ = lean_box(1);
v_result_304_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_getChecks___closed__0));
v___x_305_ = lean_box(0);
v___x_306_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_303_, v___x_300_, v_env_299_, v_asyncMode_302_, v___x_305_);
v___x_307_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint_getChecks_spec__2(v_slow_292_, v_runOnly_293_, v_runAlways_294_, v_result_304_, v___x_306_, v_a_295_, v_a_296_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_316_; 
v_a_308_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_316_ == 0)
{
v___x_310_ = v___x_307_;
v_isShared_311_ = v_isSharedCheck_316_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_307_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_316_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v_a_312_; lean_object* v___x_314_; 
v_a_312_ = lean_ctor_get(v_a_308_, 0);
lean_inc(v_a_312_);
lean_dec(v_a_308_);
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 0, v_a_312_);
v___x_314_ = v___x_310_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v_a_312_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
else
{
lean_object* v_a_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_324_; 
v_a_317_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_324_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_324_ == 0)
{
v___x_319_ = v___x_307_;
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_a_317_);
lean_dec(v___x_307_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___x_322_; 
if (v_isShared_320_ == 0)
{
v___x_322_ = v___x_319_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v_a_317_);
v___x_322_ = v_reuseFailAlloc_323_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
return v___x_322_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getChecks___boxed(lean_object* v_slow_325_, lean_object* v_runOnly_326_, lean_object* v_runAlways_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_){
_start:
{
uint8_t v_slow_boxed_331_; lean_object* v_res_332_; 
v_slow_boxed_331_ = lean_unbox(v_slow_325_);
v_res_332_ = lp_batteries_Batteries_Tactic_Lint_getChecks(v_slow_boxed_331_, v_runOnly_326_, v_runAlways_327_, v_a_328_, v_a_329_);
lean_dec(v_a_329_);
lean_dec_ref(v_a_328_);
lean_dec(v_runAlways_327_);
lean_dec(v_runOnly_326_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0(lean_object* v_a_333_, lean_object* v_as_334_, lean_object* v_k_335_, lean_object* v_x_336_, lean_object* v_x_337_, lean_object* v_x_338_, lean_object* v_x_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___redArg(v_a_333_, v_as_334_, v_k_335_, v_x_336_, v_x_337_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0___boxed(lean_object* v_a_341_, lean_object* v_as_342_, lean_object* v_k_343_, lean_object* v_x_344_, lean_object* v_x_345_, lean_object* v_x_346_, lean_object* v_x_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0_spec__0(v_a_341_, v_as_342_, v_k_343_, v_x_344_, v_x_345_, v_x_346_, v_x_347_);
lean_dec_ref(v_k_343_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(lean_object* v_k_349_, uint8_t v_defValue_350_, lean_object* v___y_351_){
_start:
{
lean_object* v_options_353_; lean_object* v_map_354_; lean_object* v___x_355_; 
v_options_353_ = lean_ctor_get(v___y_351_, 2);
v_map_354_ = lean_ctor_get(v_options_353_, 0);
v___x_355_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_354_, v_k_349_);
if (lean_obj_tag(v___x_355_) == 0)
{
lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_356_ = lean_box(v_defValue_350_);
v___x_357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
return v___x_357_;
}
else
{
lean_object* v_val_358_; lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_371_; 
v_val_358_ = lean_ctor_get(v___x_355_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_371_ == 0)
{
v___x_360_ = v___x_355_;
v_isShared_361_ = v_isSharedCheck_371_;
goto v_resetjp_359_;
}
else
{
lean_inc(v_val_358_);
lean_dec(v___x_355_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_371_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
if (lean_obj_tag(v_val_358_) == 1)
{
uint8_t v_v_362_; lean_object* v___x_363_; lean_object* v___x_365_; 
v_v_362_ = lean_ctor_get_uint8(v_val_358_, 0);
lean_dec_ref_known(v_val_358_, 0);
v___x_363_ = lean_box(v_v_362_);
if (v_isShared_361_ == 0)
{
lean_ctor_set_tag(v___x_360_, 0);
lean_ctor_set(v___x_360_, 0, v___x_363_);
v___x_365_ = v___x_360_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v___x_363_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
else
{
lean_object* v___x_367_; lean_object* v___x_369_; 
lean_dec(v_val_358_);
v___x_367_ = lean_box(v_defValue_350_);
if (v_isShared_361_ == 0)
{
lean_ctor_set_tag(v___x_360_, 0);
lean_ctor_set(v___x_360_, 0, v___x_367_);
v___x_369_ = v___x_360_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v___x_367_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg___boxed(lean_object* v_k_372_, lean_object* v_defValue_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
uint8_t v_defValue_boxed_376_; lean_object* v_res_377_; 
v_defValue_boxed_376_ = lean_unbox(v_defValue_373_);
v_res_377_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v_k_372_, v_defValue_boxed_376_, v___y_374_);
lean_dec_ref(v___y_374_);
lean_dec(v_k_372_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3(lean_object* v_k_378_, uint8_t v_defValue_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v___x_383_; 
v___x_383_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v_k_378_, v_defValue_379_, v___y_380_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___boxed(lean_object* v_k_384_, lean_object* v_defValue_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
uint8_t v_defValue_boxed_389_; lean_object* v_res_390_; 
v_defValue_boxed_389_ = lean_unbox(v_defValue_385_);
v_res_390_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3(v_k_384_, v_defValue_boxed_389_, v___y_386_, v___y_387_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v_k_384_);
return v_res_390_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14_spec__18(lean_object* v_a_391_, lean_object* v_as_392_, size_t v_i_393_, size_t v_stop_394_){
_start:
{
uint8_t v___x_395_; 
v___x_395_ = lean_usize_dec_eq(v_i_393_, v_stop_394_);
if (v___x_395_ == 0)
{
lean_object* v___x_396_; uint8_t v___x_397_; 
v___x_396_ = lean_array_uget_borrowed(v_as_392_, v_i_393_);
v___x_397_ = lean_name_eq(v_a_391_, v___x_396_);
if (v___x_397_ == 0)
{
size_t v___x_398_; size_t v___x_399_; 
v___x_398_ = ((size_t)1ULL);
v___x_399_ = lean_usize_add(v_i_393_, v___x_398_);
v_i_393_ = v___x_399_;
goto _start;
}
else
{
return v___x_397_;
}
}
else
{
uint8_t v___x_401_; 
v___x_401_ = 0;
return v___x_401_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14_spec__18___boxed(lean_object* v_a_402_, lean_object* v_as_403_, lean_object* v_i_404_, lean_object* v_stop_405_){
_start:
{
size_t v_i_boxed_406_; size_t v_stop_boxed_407_; uint8_t v_res_408_; lean_object* v_r_409_; 
v_i_boxed_406_ = lean_unbox_usize(v_i_404_);
lean_dec(v_i_404_);
v_stop_boxed_407_ = lean_unbox_usize(v_stop_405_);
lean_dec(v_stop_405_);
v_res_408_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14_spec__18(v_a_402_, v_as_403_, v_i_boxed_406_, v_stop_boxed_407_);
lean_dec_ref(v_as_403_);
lean_dec(v_a_402_);
v_r_409_ = lean_box(v_res_408_);
return v_r_409_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14(lean_object* v_as_410_, lean_object* v_a_411_){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; uint8_t v___x_414_; 
v___x_412_ = lean_unsigned_to_nat(0u);
v___x_413_ = lean_array_get_size(v_as_410_);
v___x_414_ = lean_nat_dec_lt(v___x_412_, v___x_413_);
if (v___x_414_ == 0)
{
return v___x_414_;
}
else
{
if (v___x_414_ == 0)
{
return v___x_414_;
}
else
{
size_t v___x_415_; size_t v___x_416_; uint8_t v___x_417_; 
v___x_415_ = ((size_t)0ULL);
v___x_416_ = lean_usize_of_nat(v___x_413_);
v___x_417_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14_spec__18(v_a_411_, v_as_410_, v___x_415_, v___x_416_);
return v___x_417_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14___boxed(lean_object* v_as_418_, lean_object* v_a_419_){
_start:
{
uint8_t v_res_420_; lean_object* v_r_421_; 
v_res_420_ = lp_batteries_Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14(v_as_418_, v_a_419_);
lean_dec(v_a_419_);
lean_dec_ref(v_as_418_);
v_r_421_ = lean_box(v_res_420_);
return v_r_421_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = l_Array_instInhabited(lean_box(0));
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg(lean_object* v_linter_425_, lean_object* v_decl_426_, lean_object* v___y_427_){
_start:
{
lean_object* v___x_429_; lean_object* v___y_431_; lean_object* v_env_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_429_ = lean_st_ref_get(v___y_427_);
v_env_439_ = lean_ctor_get(v___x_429_, 0);
lean_inc_ref(v_env_439_);
lean_dec(v___x_429_);
v___x_440_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__0, &lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__0);
v___x_441_ = lp_batteries_Batteries_Tactic_Lint_nolintAttr;
v___x_442_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v___x_440_, v___x_441_, v_env_439_, v_decl_426_);
if (lean_obj_tag(v___x_442_) == 0)
{
lean_object* v___x_443_; 
v___x_443_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1));
v___y_431_ = v___x_443_;
goto v___jp_430_;
}
else
{
lean_object* v_val_444_; 
v_val_444_ = lean_ctor_get(v___x_442_, 0);
lean_inc(v_val_444_);
lean_dec_ref_known(v___x_442_, 1);
v___y_431_ = v_val_444_;
goto v___jp_430_;
}
v___jp_430_:
{
uint8_t v___x_432_; 
v___x_432_ = lp_batteries_Array_contains___at___00Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7_spec__14(v___y_431_, v_linter_425_);
lean_dec_ref(v___y_431_);
if (v___x_432_ == 0)
{
uint8_t v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_433_ = 1;
v___x_434_ = lean_box(v___x_433_);
v___x_435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_435_, 0, v___x_434_);
return v___x_435_;
}
else
{
uint8_t v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; 
v___x_436_ = 0;
v___x_437_ = lean_box(v___x_436_);
v___x_438_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_438_, 0, v___x_437_);
return v___x_438_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___boxed(lean_object* v_linter_445_, lean_object* v_decl_446_, lean_object* v___y_447_, lean_object* v___y_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg(v_linter_445_, v_decl_446_, v___y_447_);
lean_dec(v___y_447_);
lean_dec(v_linter_445_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8(lean_object* v___x_450_, lean_object* v_as_451_, size_t v_i_452_, size_t v_stop_453_, lean_object* v_b_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
uint8_t v___x_458_; 
v___x_458_ = lean_usize_dec_eq(v_i_452_, v_stop_453_);
if (v___x_458_ == 0)
{
lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_459_ = lean_array_uget_borrowed(v_as_451_, v_i_452_);
lean_inc(v___x_459_);
v___x_460_ = lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg(v___x_450_, v___x_459_, v___y_456_);
if (lean_obj_tag(v___x_460_) == 0)
{
lean_object* v_a_461_; lean_object* v_a_463_; uint8_t v___x_467_; 
v_a_461_ = lean_ctor_get(v___x_460_, 0);
lean_inc(v_a_461_);
lean_dec_ref_known(v___x_460_, 1);
v___x_467_ = lean_unbox(v_a_461_);
lean_dec(v_a_461_);
if (v___x_467_ == 0)
{
v_a_463_ = v_b_454_;
goto v___jp_462_;
}
else
{
lean_object* v___x_468_; 
lean_inc(v___x_459_);
v___x_468_ = lean_array_push(v_b_454_, v___x_459_);
v_a_463_ = v___x_468_;
goto v___jp_462_;
}
v___jp_462_:
{
size_t v___x_464_; size_t v___x_465_; 
v___x_464_ = ((size_t)1ULL);
v___x_465_ = lean_usize_add(v_i_452_, v___x_464_);
v_i_452_ = v___x_465_;
v_b_454_ = v_a_463_;
goto _start;
}
}
else
{
lean_object* v_a_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_476_; 
lean_dec_ref(v_b_454_);
v_a_469_ = lean_ctor_get(v___x_460_, 0);
v_isSharedCheck_476_ = !lean_is_exclusive(v___x_460_);
if (v_isSharedCheck_476_ == 0)
{
v___x_471_ = v___x_460_;
v_isShared_472_ = v_isSharedCheck_476_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_a_469_);
lean_dec(v___x_460_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_476_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_474_; 
if (v_isShared_472_ == 0)
{
v___x_474_ = v___x_471_;
goto v_reusejp_473_;
}
else
{
lean_object* v_reuseFailAlloc_475_; 
v_reuseFailAlloc_475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_475_, 0, v_a_469_);
v___x_474_ = v_reuseFailAlloc_475_;
goto v_reusejp_473_;
}
v_reusejp_473_:
{
return v___x_474_;
}
}
}
}
else
{
lean_object* v___x_477_; 
v___x_477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_477_, 0, v_b_454_);
return v___x_477_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8___boxed(lean_object* v___x_478_, lean_object* v_as_479_, lean_object* v_i_480_, lean_object* v_stop_481_, lean_object* v_b_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
size_t v_i_boxed_486_; size_t v_stop_boxed_487_; lean_object* v_res_488_; 
v_i_boxed_486_ = lean_unbox_usize(v_i_480_);
lean_dec(v_i_480_);
v_stop_boxed_487_ = lean_unbox_usize(v_stop_481_);
lean_dec(v_stop_481_);
v_res_488_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8(v___x_478_, v_as_479_, v_i_boxed_486_, v_stop_boxed_487_, v_b_482_, v___y_483_, v___y_484_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec_ref(v_as_479_);
lean_dec(v___x_478_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8(lean_object* v_s_489_){
_start:
{
lean_object* v___x_491_; lean_object* v_putStr_492_; lean_object* v___x_493_; 
v___x_491_ = lean_get_stdout();
v_putStr_492_ = lean_ctor_get(v___x_491_, 4);
lean_inc_ref(v_putStr_492_);
lean_dec_ref(v___x_491_);
v___x_493_ = lean_apply_2(v_putStr_492_, v_s_489_, lean_box(0));
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8___boxed(lean_object* v_s_494_, lean_object* v_a_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8(v_s_494_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(lean_object* v_s_497_){
_start:
{
uint32_t v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_499_ = 10;
v___x_500_ = lean_string_push(v_s_497_, v___x_499_);
v___x_501_ = lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8(v___x_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4___boxed(lean_object* v_s_502_, lean_object* v_a_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v_s_502_);
return v_res_504_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__0(void){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_505_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1(void){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; 
v___x_506_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__0);
v___x_507_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_507_, 0, v___x_506_);
return v___x_507_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2(void){
_start:
{
lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_508_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1);
v___x_509_ = lean_unsigned_to_nat(0u);
v___x_510_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
lean_ctor_set(v___x_510_, 1, v___x_509_);
lean_ctor_set(v___x_510_, 2, v___x_509_);
lean_ctor_set(v___x_510_, 3, v___x_509_);
lean_ctor_set(v___x_510_, 4, v___x_508_);
lean_ctor_set(v___x_510_, 5, v___x_508_);
lean_ctor_set(v___x_510_, 6, v___x_508_);
lean_ctor_set(v___x_510_, 7, v___x_508_);
lean_ctor_set(v___x_510_, 8, v___x_508_);
lean_ctor_set(v___x_510_, 9, v___x_508_);
return v___x_510_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lean_unsigned_to_nat(32u);
v___x_512_ = lean_mk_empty_array_with_capacity(v___x_511_);
v___x_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
return v___x_513_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__4(void){
_start:
{
size_t v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; 
v___x_514_ = ((size_t)5ULL);
v___x_515_ = lean_unsigned_to_nat(0u);
v___x_516_ = lean_unsigned_to_nat(32u);
v___x_517_ = lean_mk_empty_array_with_capacity(v___x_516_);
v___x_518_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__3);
v___x_519_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_519_, 0, v___x_518_);
lean_ctor_set(v___x_519_, 1, v___x_517_);
lean_ctor_set(v___x_519_, 2, v___x_515_);
lean_ctor_set(v___x_519_, 3, v___x_515_);
lean_ctor_set_usize(v___x_519_, 4, v___x_514_);
return v___x_519_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5(void){
_start:
{
lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; 
v___x_520_ = lean_box(1);
v___x_521_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__4);
v___x_522_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1);
v___x_523_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_523_, 0, v___x_522_);
lean_ctor_set(v___x_523_, 1, v___x_521_);
lean_ctor_set(v___x_523_, 2, v___x_520_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(lean_object* v_msgData_524_, lean_object* v___y_525_, lean_object* v___y_526_){
_start:
{
lean_object* v___x_528_; lean_object* v_env_529_; lean_object* v_options_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_528_ = lean_st_ref_get(v___y_526_);
v_env_529_ = lean_ctor_get(v___x_528_, 0);
lean_inc_ref(v_env_529_);
lean_dec(v___x_528_);
v_options_530_ = lean_ctor_get(v___y_525_, 2);
v___x_531_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2);
v___x_532_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5);
lean_inc_ref(v_options_530_);
v___x_533_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_533_, 0, v_env_529_);
lean_ctor_set(v___x_533_, 1, v___x_531_);
lean_ctor_set(v___x_533_, 2, v___x_532_);
lean_ctor_set(v___x_533_, 3, v_options_530_);
v___x_534_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
lean_ctor_set(v___x_534_, 1, v_msgData_524_);
v___x_535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_535_, 0, v___x_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___boxed(lean_object* v_msgData_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(v_msgData_536_, v___y_537_, v___y_538_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
return v_res_540_;
}
}
static double _init_lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__0(void){
_start:
{
lean_object* v___x_541_; double v___x_542_; 
v___x_541_ = lean_unsigned_to_nat(0u);
v___x_542_ = lean_float_of_nat(v___x_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(lean_object* v_cls_546_, lean_object* v_msg_547_, lean_object* v___y_548_, lean_object* v___y_549_){
_start:
{
lean_object* v_ref_551_; lean_object* v___x_552_; lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_597_; 
v_ref_551_ = lean_ctor_get(v___y_548_, 5);
v___x_552_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(v_msg_547_, v___y_548_, v___y_549_);
v_a_553_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_597_ == 0)
{
v___x_555_ = v___x_552_;
v_isShared_556_ = v_isSharedCheck_597_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_552_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_597_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; lean_object* v_traceState_558_; lean_object* v_env_559_; lean_object* v_nextMacroScope_560_; lean_object* v_ngen_561_; lean_object* v_auxDeclNGen_562_; lean_object* v_cache_563_; lean_object* v_messages_564_; lean_object* v_infoState_565_; lean_object* v_snapshotTasks_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_596_; 
v___x_557_ = lean_st_ref_take(v___y_549_);
v_traceState_558_ = lean_ctor_get(v___x_557_, 4);
v_env_559_ = lean_ctor_get(v___x_557_, 0);
v_nextMacroScope_560_ = lean_ctor_get(v___x_557_, 1);
v_ngen_561_ = lean_ctor_get(v___x_557_, 2);
v_auxDeclNGen_562_ = lean_ctor_get(v___x_557_, 3);
v_cache_563_ = lean_ctor_get(v___x_557_, 5);
v_messages_564_ = lean_ctor_get(v___x_557_, 6);
v_infoState_565_ = lean_ctor_get(v___x_557_, 7);
v_snapshotTasks_566_ = lean_ctor_get(v___x_557_, 8);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_557_);
if (v_isSharedCheck_596_ == 0)
{
v___x_568_ = v___x_557_;
v_isShared_569_ = v_isSharedCheck_596_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_snapshotTasks_566_);
lean_inc(v_infoState_565_);
lean_inc(v_messages_564_);
lean_inc(v_cache_563_);
lean_inc(v_traceState_558_);
lean_inc(v_auxDeclNGen_562_);
lean_inc(v_ngen_561_);
lean_inc(v_nextMacroScope_560_);
lean_inc(v_env_559_);
lean_dec(v___x_557_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_596_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
uint64_t v_tid_570_; lean_object* v_traces_571_; lean_object* v___x_573_; uint8_t v_isShared_574_; uint8_t v_isSharedCheck_595_; 
v_tid_570_ = lean_ctor_get_uint64(v_traceState_558_, sizeof(void*)*1);
v_traces_571_ = lean_ctor_get(v_traceState_558_, 0);
v_isSharedCheck_595_ = !lean_is_exclusive(v_traceState_558_);
if (v_isSharedCheck_595_ == 0)
{
v___x_573_ = v_traceState_558_;
v_isShared_574_ = v_isSharedCheck_595_;
goto v_resetjp_572_;
}
else
{
lean_inc(v_traces_571_);
lean_dec(v_traceState_558_);
v___x_573_ = lean_box(0);
v_isShared_574_ = v_isSharedCheck_595_;
goto v_resetjp_572_;
}
v_resetjp_572_:
{
lean_object* v___x_575_; double v___x_576_; uint8_t v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_585_; 
v___x_575_ = lean_box(0);
v___x_576_ = lean_float_once(&lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__0, &lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__0_once, _init_lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__0);
v___x_577_ = 0;
v___x_578_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___x_579_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_579_, 0, v_cls_546_);
lean_ctor_set(v___x_579_, 1, v___x_575_);
lean_ctor_set(v___x_579_, 2, v___x_578_);
lean_ctor_set_float(v___x_579_, sizeof(void*)*3, v___x_576_);
lean_ctor_set_float(v___x_579_, sizeof(void*)*3 + 8, v___x_576_);
lean_ctor_set_uint8(v___x_579_, sizeof(void*)*3 + 16, v___x_577_);
v___x_580_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__2));
v___x_581_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_581_, 0, v___x_579_);
lean_ctor_set(v___x_581_, 1, v_a_553_);
lean_ctor_set(v___x_581_, 2, v___x_580_);
lean_inc(v_ref_551_);
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v_ref_551_);
lean_ctor_set(v___x_582_, 1, v___x_581_);
v___x_583_ = l_Lean_PersistentArray_push___redArg(v_traces_571_, v___x_582_);
if (v_isShared_574_ == 0)
{
lean_ctor_set(v___x_573_, 0, v___x_583_);
v___x_585_ = v___x_573_;
goto v_reusejp_584_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v___x_583_);
lean_ctor_set_uint64(v_reuseFailAlloc_594_, sizeof(void*)*1, v_tid_570_);
v___x_585_ = v_reuseFailAlloc_594_;
goto v_reusejp_584_;
}
v_reusejp_584_:
{
lean_object* v___x_587_; 
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 4, v___x_585_);
v___x_587_ = v___x_568_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v_env_559_);
lean_ctor_set(v_reuseFailAlloc_593_, 1, v_nextMacroScope_560_);
lean_ctor_set(v_reuseFailAlloc_593_, 2, v_ngen_561_);
lean_ctor_set(v_reuseFailAlloc_593_, 3, v_auxDeclNGen_562_);
lean_ctor_set(v_reuseFailAlloc_593_, 4, v___x_585_);
lean_ctor_set(v_reuseFailAlloc_593_, 5, v_cache_563_);
lean_ctor_set(v_reuseFailAlloc_593_, 6, v_messages_564_);
lean_ctor_set(v_reuseFailAlloc_593_, 7, v_infoState_565_);
lean_ctor_set(v_reuseFailAlloc_593_, 8, v_snapshotTasks_566_);
v___x_587_ = v_reuseFailAlloc_593_;
goto v_reusejp_586_;
}
v_reusejp_586_:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_591_; 
v___x_588_ = lean_st_ref_set(v___y_549_, v___x_587_);
v___x_589_ = lean_box(0);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v___x_589_);
v___x_591_ = v___x_555_;
goto v_reusejp_590_;
}
else
{
lean_object* v_reuseFailAlloc_592_; 
v_reuseFailAlloc_592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_592_, 0, v___x_589_);
v___x_591_ = v_reuseFailAlloc_592_;
goto v_reusejp_590_;
}
v_reusejp_590_:
{
return v___x_591_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___boxed(lean_object* v_cls_598_, lean_object* v_msg_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(v_cls_598_, v_msg_599_, v___y_600_, v___y_601_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(lean_object* v_s_604_){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; uint32_t v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_606_ = l_Std_Format_defWidth;
v___x_607_ = lean_unsigned_to_nat(0u);
v___x_608_ = l_Std_Format_pretty(v_s_604_, v___x_606_, v___x_607_, v___x_607_);
v___x_609_ = 10;
v___x_610_ = lean_string_push(v___x_608_, v___x_609_);
v___x_611_ = lp_batteries_IO_print___at___00IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4_spec__8(v___x_610_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10___boxed(lean_object* v_s_612_, lean_object* v_a_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(v_s_612_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg(lean_object* v_as_618_, size_t v_sz_619_, size_t v_i_620_, lean_object* v_b_621_, lean_object* v___y_622_){
_start:
{
uint8_t v___x_624_; 
v___x_624_ = lean_usize_dec_lt(v_i_620_, v_sz_619_);
if (v___x_624_ == 0)
{
lean_object* v___x_625_; 
v___x_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_625_, 0, v_b_621_);
return v___x_625_;
}
else
{
lean_object* v_a_626_; lean_object* v_msg_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_653_; 
lean_dec_ref(v_b_621_);
v_a_626_ = lean_array_uget(v_as_618_, v_i_620_);
v_msg_627_ = lean_ctor_get(v_a_626_, 1);
v_isSharedCheck_653_ = !lean_is_exclusive(v_a_626_);
if (v_isSharedCheck_653_ == 0)
{
lean_object* v_unused_654_; 
v_unused_654_ = lean_ctor_get(v_a_626_, 0);
lean_dec(v_unused_654_);
v___x_629_ = v_a_626_;
v_isShared_630_ = v_isSharedCheck_653_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_msg_627_);
lean_dec(v_a_626_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_653_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v___x_631_ = lean_box(0);
v___x_632_ = l_Lean_MessageData_format(v_msg_627_, v___x_631_);
v___x_633_ = lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(v___x_632_);
if (lean_obj_tag(v___x_633_) == 0)
{
lean_object* v___x_634_; size_t v___x_635_; size_t v___x_636_; 
lean_dec_ref_known(v___x_633_, 1);
lean_del_object(v___x_629_);
v___x_634_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___closed__0));
v___x_635_ = ((size_t)1ULL);
v___x_636_ = lean_usize_add(v_i_620_, v___x_635_);
v_i_620_ = v___x_636_;
v_b_621_ = v___x_634_;
goto _start;
}
else
{
lean_object* v_a_638_; lean_object* v___x_640_; uint8_t v_isShared_641_; uint8_t v_isSharedCheck_652_; 
v_a_638_ = lean_ctor_get(v___x_633_, 0);
v_isSharedCheck_652_ = !lean_is_exclusive(v___x_633_);
if (v_isSharedCheck_652_ == 0)
{
v___x_640_ = v___x_633_;
v_isShared_641_ = v_isSharedCheck_652_;
goto v_resetjp_639_;
}
else
{
lean_inc(v_a_638_);
lean_dec(v___x_633_);
v___x_640_ = lean_box(0);
v_isShared_641_ = v_isSharedCheck_652_;
goto v_resetjp_639_;
}
v_resetjp_639_:
{
lean_object* v_ref_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_647_; 
v_ref_642_ = lean_ctor_get(v___y_622_, 5);
v___x_643_ = lean_io_error_to_string(v_a_638_);
v___x_644_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_644_, 0, v___x_643_);
v___x_645_ = l_Lean_MessageData_ofFormat(v___x_644_);
lean_inc(v_ref_642_);
if (v_isShared_630_ == 0)
{
lean_ctor_set(v___x_629_, 1, v___x_645_);
lean_ctor_set(v___x_629_, 0, v_ref_642_);
v___x_647_ = v___x_629_;
goto v_reusejp_646_;
}
else
{
lean_object* v_reuseFailAlloc_651_; 
v_reuseFailAlloc_651_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_651_, 0, v_ref_642_);
lean_ctor_set(v_reuseFailAlloc_651_, 1, v___x_645_);
v___x_647_ = v_reuseFailAlloc_651_;
goto v_reusejp_646_;
}
v_reusejp_646_:
{
lean_object* v___x_649_; 
if (v_isShared_641_ == 0)
{
lean_ctor_set(v___x_640_, 0, v___x_647_);
v___x_649_ = v___x_640_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_647_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___boxed(lean_object* v_as_655_, lean_object* v_sz_656_, lean_object* v_i_657_, lean_object* v_b_658_, lean_object* v___y_659_, lean_object* v___y_660_){
_start:
{
size_t v_sz_boxed_661_; size_t v_i_boxed_662_; lean_object* v_res_663_; 
v_sz_boxed_661_ = lean_unbox_usize(v_sz_656_);
lean_dec(v_sz_656_);
v_i_boxed_662_ = lean_unbox_usize(v_i_657_);
lean_dec(v_i_657_);
v_res_663_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg(v_as_655_, v_sz_boxed_661_, v_i_boxed_662_, v_b_658_, v___y_659_);
lean_dec_ref(v___y_659_);
lean_dec_ref(v_as_655_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14(lean_object* v_as_664_, size_t v_sz_665_, size_t v_i_666_, lean_object* v_b_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
uint8_t v___x_673_; 
v___x_673_ = lean_usize_dec_lt(v_i_666_, v_sz_665_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; 
v___x_674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_674_, 0, v_b_667_);
return v___x_674_;
}
else
{
lean_object* v_a_675_; lean_object* v_msg_676_; lean_object* v___x_678_; uint8_t v_isShared_679_; uint8_t v_isSharedCheck_702_; 
lean_dec_ref(v_b_667_);
v_a_675_ = lean_array_uget(v_as_664_, v_i_666_);
v_msg_676_ = lean_ctor_get(v_a_675_, 1);
v_isSharedCheck_702_ = !lean_is_exclusive(v_a_675_);
if (v_isSharedCheck_702_ == 0)
{
lean_object* v_unused_703_; 
v_unused_703_ = lean_ctor_get(v_a_675_, 0);
lean_dec(v_unused_703_);
v___x_678_ = v_a_675_;
v_isShared_679_ = v_isSharedCheck_702_;
goto v_resetjp_677_;
}
else
{
lean_inc(v_msg_676_);
lean_dec(v_a_675_);
v___x_678_ = lean_box(0);
v_isShared_679_ = v_isSharedCheck_702_;
goto v_resetjp_677_;
}
v_resetjp_677_:
{
lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v___x_680_ = lean_box(0);
v___x_681_ = l_Lean_MessageData_format(v_msg_676_, v___x_680_);
v___x_682_ = lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(v___x_681_);
if (lean_obj_tag(v___x_682_) == 0)
{
lean_object* v___x_683_; size_t v___x_684_; size_t v___x_685_; lean_object* v___x_686_; 
lean_dec_ref_known(v___x_682_, 1);
lean_del_object(v___x_678_);
v___x_683_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg___closed__0));
v___x_684_ = ((size_t)1ULL);
v___x_685_ = lean_usize_add(v_i_666_, v___x_684_);
v___x_686_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg(v_as_664_, v_sz_665_, v___x_685_, v___x_683_, v___y_670_);
return v___x_686_;
}
else
{
lean_object* v_a_687_; lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_701_; 
v_a_687_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_701_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_701_ == 0)
{
v___x_689_ = v___x_682_;
v_isShared_690_ = v_isSharedCheck_701_;
goto v_resetjp_688_;
}
else
{
lean_inc(v_a_687_);
lean_dec(v___x_682_);
v___x_689_ = lean_box(0);
v_isShared_690_ = v_isSharedCheck_701_;
goto v_resetjp_688_;
}
v_resetjp_688_:
{
lean_object* v_ref_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_696_; 
v_ref_691_ = lean_ctor_get(v___y_670_, 5);
v___x_692_ = lean_io_error_to_string(v_a_687_);
v___x_693_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_693_, 0, v___x_692_);
v___x_694_ = l_Lean_MessageData_ofFormat(v___x_693_);
lean_inc(v_ref_691_);
if (v_isShared_679_ == 0)
{
lean_ctor_set(v___x_678_, 1, v___x_694_);
lean_ctor_set(v___x_678_, 0, v_ref_691_);
v___x_696_ = v___x_678_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v_ref_691_);
lean_ctor_set(v_reuseFailAlloc_700_, 1, v___x_694_);
v___x_696_ = v_reuseFailAlloc_700_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
lean_object* v___x_698_; 
if (v_isShared_690_ == 0)
{
lean_ctor_set(v___x_689_, 0, v___x_696_);
v___x_698_ = v___x_689_;
goto v_reusejp_697_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v___x_696_);
v___x_698_ = v_reuseFailAlloc_699_;
goto v_reusejp_697_;
}
v_reusejp_697_:
{
return v___x_698_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14___boxed(lean_object* v_as_704_, lean_object* v_sz_705_, lean_object* v_i_706_, lean_object* v_b_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_){
_start:
{
size_t v_sz_boxed_713_; size_t v_i_boxed_714_; lean_object* v_res_715_; 
v_sz_boxed_713_ = lean_unbox_usize(v_sz_705_);
lean_dec(v_sz_705_);
v_i_boxed_714_ = lean_unbox_usize(v_i_706_);
lean_dec(v_i_706_);
v_res_715_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14(v_as_704_, v_sz_boxed_713_, v_i_boxed_714_, v_b_707_, v___y_708_, v___y_709_, v___y_710_, v___y_711_);
lean_dec(v___y_711_);
lean_dec_ref(v___y_710_);
lean_dec(v___y_709_);
lean_dec_ref(v___y_708_);
lean_dec_ref(v_as_704_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg(lean_object* v_as_719_, size_t v_sz_720_, size_t v_i_721_, lean_object* v_b_722_, lean_object* v___y_723_){
_start:
{
uint8_t v___x_725_; 
v___x_725_ = lean_usize_dec_lt(v_i_721_, v_sz_720_);
if (v___x_725_ == 0)
{
lean_object* v___x_726_; 
v___x_726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_726_, 0, v_b_722_);
return v___x_726_;
}
else
{
lean_object* v_a_727_; lean_object* v_msg_728_; lean_object* v___x_730_; uint8_t v_isShared_731_; uint8_t v_isSharedCheck_754_; 
lean_dec_ref(v_b_722_);
v_a_727_ = lean_array_uget(v_as_719_, v_i_721_);
v_msg_728_ = lean_ctor_get(v_a_727_, 1);
v_isSharedCheck_754_ = !lean_is_exclusive(v_a_727_);
if (v_isSharedCheck_754_ == 0)
{
lean_object* v_unused_755_; 
v_unused_755_ = lean_ctor_get(v_a_727_, 0);
lean_dec(v_unused_755_);
v___x_730_ = v_a_727_;
v_isShared_731_ = v_isSharedCheck_754_;
goto v_resetjp_729_;
}
else
{
lean_inc(v_msg_728_);
lean_dec(v_a_727_);
v___x_730_ = lean_box(0);
v_isShared_731_ = v_isSharedCheck_754_;
goto v_resetjp_729_;
}
v_resetjp_729_:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; 
v___x_732_ = lean_box(0);
v___x_733_ = l_Lean_MessageData_format(v_msg_728_, v___x_732_);
v___x_734_ = lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(v___x_733_);
if (lean_obj_tag(v___x_734_) == 0)
{
lean_object* v___x_735_; size_t v___x_736_; size_t v___x_737_; 
lean_dec_ref_known(v___x_734_, 1);
lean_del_object(v___x_730_);
v___x_735_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___closed__0));
v___x_736_ = ((size_t)1ULL);
v___x_737_ = lean_usize_add(v_i_721_, v___x_736_);
v_i_721_ = v___x_737_;
v_b_722_ = v___x_735_;
goto _start;
}
else
{
lean_object* v_a_739_; lean_object* v___x_741_; uint8_t v_isShared_742_; uint8_t v_isSharedCheck_753_; 
v_a_739_ = lean_ctor_get(v___x_734_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v___x_734_);
if (v_isSharedCheck_753_ == 0)
{
v___x_741_ = v___x_734_;
v_isShared_742_ = v_isSharedCheck_753_;
goto v_resetjp_740_;
}
else
{
lean_inc(v_a_739_);
lean_dec(v___x_734_);
v___x_741_ = lean_box(0);
v_isShared_742_ = v_isSharedCheck_753_;
goto v_resetjp_740_;
}
v_resetjp_740_:
{
lean_object* v_ref_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_748_; 
v_ref_743_ = lean_ctor_get(v___y_723_, 5);
v___x_744_ = lean_io_error_to_string(v_a_739_);
v___x_745_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
v___x_746_ = l_Lean_MessageData_ofFormat(v___x_745_);
lean_inc(v_ref_743_);
if (v_isShared_731_ == 0)
{
lean_ctor_set(v___x_730_, 1, v___x_746_);
lean_ctor_set(v___x_730_, 0, v_ref_743_);
v___x_748_ = v___x_730_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_ref_743_);
lean_ctor_set(v_reuseFailAlloc_752_, 1, v___x_746_);
v___x_748_ = v_reuseFailAlloc_752_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
lean_object* v___x_750_; 
if (v_isShared_742_ == 0)
{
lean_ctor_set(v___x_741_, 0, v___x_748_);
v___x_750_ = v___x_741_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v___x_748_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___boxed(lean_object* v_as_756_, lean_object* v_sz_757_, lean_object* v_i_758_, lean_object* v_b_759_, lean_object* v___y_760_, lean_object* v___y_761_){
_start:
{
size_t v_sz_boxed_762_; size_t v_i_boxed_763_; lean_object* v_res_764_; 
v_sz_boxed_762_ = lean_unbox_usize(v_sz_757_);
lean_dec(v_sz_757_);
v_i_boxed_763_ = lean_unbox_usize(v_i_758_);
lean_dec(v_i_758_);
v_res_764_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg(v_as_756_, v_sz_boxed_762_, v_i_boxed_763_, v_b_759_, v___y_760_);
lean_dec_ref(v___y_760_);
lean_dec_ref(v_as_756_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22(lean_object* v_as_765_, size_t v_sz_766_, size_t v_i_767_, lean_object* v_b_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_){
_start:
{
uint8_t v___x_774_; 
v___x_774_ = lean_usize_dec_lt(v_i_767_, v_sz_766_);
if (v___x_774_ == 0)
{
lean_object* v___x_775_; 
v___x_775_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_775_, 0, v_b_768_);
return v___x_775_;
}
else
{
lean_object* v_a_776_; lean_object* v_msg_777_; lean_object* v___x_779_; uint8_t v_isShared_780_; uint8_t v_isSharedCheck_803_; 
lean_dec_ref(v_b_768_);
v_a_776_ = lean_array_uget(v_as_765_, v_i_767_);
v_msg_777_ = lean_ctor_get(v_a_776_, 1);
v_isSharedCheck_803_ = !lean_is_exclusive(v_a_776_);
if (v_isSharedCheck_803_ == 0)
{
lean_object* v_unused_804_; 
v_unused_804_ = lean_ctor_get(v_a_776_, 0);
lean_dec(v_unused_804_);
v___x_779_ = v_a_776_;
v_isShared_780_ = v_isSharedCheck_803_;
goto v_resetjp_778_;
}
else
{
lean_inc(v_msg_777_);
lean_dec(v_a_776_);
v___x_779_ = lean_box(0);
v_isShared_780_ = v_isSharedCheck_803_;
goto v_resetjp_778_;
}
v_resetjp_778_:
{
lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v___x_781_ = lean_box(0);
v___x_782_ = l_Lean_MessageData_format(v_msg_777_, v___x_781_);
v___x_783_ = lp_batteries_IO_println___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__10(v___x_782_);
if (lean_obj_tag(v___x_783_) == 0)
{
lean_object* v___x_784_; size_t v___x_785_; size_t v___x_786_; lean_object* v___x_787_; 
lean_dec_ref_known(v___x_783_, 1);
lean_del_object(v___x_779_);
v___x_784_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg___closed__0));
v___x_785_ = ((size_t)1ULL);
v___x_786_ = lean_usize_add(v_i_767_, v___x_785_);
v___x_787_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg(v_as_765_, v_sz_766_, v___x_786_, v___x_784_, v___y_771_);
return v___x_787_;
}
else
{
lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_802_; 
v_a_788_ = lean_ctor_get(v___x_783_, 0);
v_isSharedCheck_802_ = !lean_is_exclusive(v___x_783_);
if (v_isSharedCheck_802_ == 0)
{
v___x_790_ = v___x_783_;
v_isShared_791_ = v_isSharedCheck_802_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_783_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_802_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v_ref_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_797_; 
v_ref_792_ = lean_ctor_get(v___y_771_, 5);
v___x_793_ = lean_io_error_to_string(v_a_788_);
v___x_794_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_794_, 0, v___x_793_);
v___x_795_ = l_Lean_MessageData_ofFormat(v___x_794_);
lean_inc(v_ref_792_);
if (v_isShared_780_ == 0)
{
lean_ctor_set(v___x_779_, 1, v___x_795_);
lean_ctor_set(v___x_779_, 0, v_ref_792_);
v___x_797_ = v___x_779_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_801_; 
v_reuseFailAlloc_801_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_801_, 0, v_ref_792_);
lean_ctor_set(v_reuseFailAlloc_801_, 1, v___x_795_);
v___x_797_ = v_reuseFailAlloc_801_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
lean_object* v___x_799_; 
if (v_isShared_791_ == 0)
{
lean_ctor_set(v___x_790_, 0, v___x_797_);
v___x_799_ = v___x_790_;
goto v_reusejp_798_;
}
else
{
lean_object* v_reuseFailAlloc_800_; 
v_reuseFailAlloc_800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_800_, 0, v___x_797_);
v___x_799_ = v_reuseFailAlloc_800_;
goto v_reusejp_798_;
}
v_reusejp_798_:
{
return v___x_799_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22___boxed(lean_object* v_as_805_, lean_object* v_sz_806_, lean_object* v_i_807_, lean_object* v_b_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_){
_start:
{
size_t v_sz_boxed_814_; size_t v_i_boxed_815_; lean_object* v_res_816_; 
v_sz_boxed_814_ = lean_unbox_usize(v_sz_806_);
lean_dec(v_sz_806_);
v_i_boxed_815_ = lean_unbox_usize(v_i_807_);
lean_dec(v_i_807_);
v_res_816_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22(v_as_805_, v_sz_boxed_814_, v_i_boxed_815_, v_b_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
lean_dec(v___y_810_);
lean_dec_ref(v___y_809_);
lean_dec_ref(v_as_805_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13(lean_object* v_init_817_, lean_object* v_n_818_, lean_object* v_b_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_){
_start:
{
if (lean_obj_tag(v_n_818_) == 0)
{
lean_object* v_cs_825_; lean_object* v___x_826_; lean_object* v___x_827_; size_t v_sz_828_; size_t v___x_829_; lean_object* v___x_830_; 
v_cs_825_ = lean_ctor_get(v_n_818_, 0);
v___x_826_ = lean_box(0);
v___x_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_827_, 0, v___x_826_);
lean_ctor_set(v___x_827_, 1, v_b_819_);
v_sz_828_ = lean_array_size(v_cs_825_);
v___x_829_ = ((size_t)0ULL);
v___x_830_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__21(v_init_817_, v_cs_825_, v_sz_828_, v___x_829_, v___x_827_, v___y_820_, v___y_821_, v___y_822_, v___y_823_);
if (lean_obj_tag(v___x_830_) == 0)
{
lean_object* v_a_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_845_; 
v_a_831_ = lean_ctor_get(v___x_830_, 0);
v_isSharedCheck_845_ = !lean_is_exclusive(v___x_830_);
if (v_isSharedCheck_845_ == 0)
{
v___x_833_ = v___x_830_;
v_isShared_834_ = v_isSharedCheck_845_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_a_831_);
lean_dec(v___x_830_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_845_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v_fst_835_; 
v_fst_835_ = lean_ctor_get(v_a_831_, 0);
if (lean_obj_tag(v_fst_835_) == 0)
{
lean_object* v_snd_836_; lean_object* v___x_837_; lean_object* v___x_839_; 
v_snd_836_ = lean_ctor_get(v_a_831_, 1);
lean_inc(v_snd_836_);
lean_dec(v_a_831_);
v___x_837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_837_, 0, v_snd_836_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 0, v___x_837_);
v___x_839_ = v___x_833_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_837_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
else
{
lean_object* v_val_841_; lean_object* v___x_843_; 
lean_inc_ref(v_fst_835_);
lean_dec(v_a_831_);
v_val_841_ = lean_ctor_get(v_fst_835_, 0);
lean_inc(v_val_841_);
lean_dec_ref_known(v_fst_835_, 1);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 0, v_val_841_);
v___x_843_ = v___x_833_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_844_; 
v_reuseFailAlloc_844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_844_, 0, v_val_841_);
v___x_843_ = v_reuseFailAlloc_844_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
return v___x_843_;
}
}
}
}
else
{
lean_object* v_a_846_; lean_object* v___x_848_; uint8_t v_isShared_849_; uint8_t v_isSharedCheck_853_; 
v_a_846_ = lean_ctor_get(v___x_830_, 0);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_830_);
if (v_isSharedCheck_853_ == 0)
{
v___x_848_ = v___x_830_;
v_isShared_849_ = v_isSharedCheck_853_;
goto v_resetjp_847_;
}
else
{
lean_inc(v_a_846_);
lean_dec(v___x_830_);
v___x_848_ = lean_box(0);
v_isShared_849_ = v_isSharedCheck_853_;
goto v_resetjp_847_;
}
v_resetjp_847_:
{
lean_object* v___x_851_; 
if (v_isShared_849_ == 0)
{
v___x_851_ = v___x_848_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v_a_846_);
v___x_851_ = v_reuseFailAlloc_852_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
return v___x_851_;
}
}
}
}
else
{
lean_object* v_vs_854_; lean_object* v___x_855_; lean_object* v___x_856_; size_t v_sz_857_; size_t v___x_858_; lean_object* v___x_859_; 
v_vs_854_ = lean_ctor_get(v_n_818_, 0);
v___x_855_ = lean_box(0);
v___x_856_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_856_, 0, v___x_855_);
lean_ctor_set(v___x_856_, 1, v_b_819_);
v_sz_857_ = lean_array_size(v_vs_854_);
v___x_858_ = ((size_t)0ULL);
v___x_859_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22(v_vs_854_, v_sz_857_, v___x_858_, v___x_856_, v___y_820_, v___y_821_, v___y_822_, v___y_823_);
if (lean_obj_tag(v___x_859_) == 0)
{
lean_object* v_a_860_; lean_object* v___x_862_; uint8_t v_isShared_863_; uint8_t v_isSharedCheck_874_; 
v_a_860_ = lean_ctor_get(v___x_859_, 0);
v_isSharedCheck_874_ = !lean_is_exclusive(v___x_859_);
if (v_isSharedCheck_874_ == 0)
{
v___x_862_ = v___x_859_;
v_isShared_863_ = v_isSharedCheck_874_;
goto v_resetjp_861_;
}
else
{
lean_inc(v_a_860_);
lean_dec(v___x_859_);
v___x_862_ = lean_box(0);
v_isShared_863_ = v_isSharedCheck_874_;
goto v_resetjp_861_;
}
v_resetjp_861_:
{
lean_object* v_fst_864_; 
v_fst_864_ = lean_ctor_get(v_a_860_, 0);
if (lean_obj_tag(v_fst_864_) == 0)
{
lean_object* v_snd_865_; lean_object* v___x_866_; lean_object* v___x_868_; 
v_snd_865_ = lean_ctor_get(v_a_860_, 1);
lean_inc(v_snd_865_);
lean_dec(v_a_860_);
v___x_866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_866_, 0, v_snd_865_);
if (v_isShared_863_ == 0)
{
lean_ctor_set(v___x_862_, 0, v___x_866_);
v___x_868_ = v___x_862_;
goto v_reusejp_867_;
}
else
{
lean_object* v_reuseFailAlloc_869_; 
v_reuseFailAlloc_869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_869_, 0, v___x_866_);
v___x_868_ = v_reuseFailAlloc_869_;
goto v_reusejp_867_;
}
v_reusejp_867_:
{
return v___x_868_;
}
}
else
{
lean_object* v_val_870_; lean_object* v___x_872_; 
lean_inc_ref(v_fst_864_);
lean_dec(v_a_860_);
v_val_870_ = lean_ctor_get(v_fst_864_, 0);
lean_inc(v_val_870_);
lean_dec_ref_known(v_fst_864_, 1);
if (v_isShared_863_ == 0)
{
lean_ctor_set(v___x_862_, 0, v_val_870_);
v___x_872_ = v___x_862_;
goto v_reusejp_871_;
}
else
{
lean_object* v_reuseFailAlloc_873_; 
v_reuseFailAlloc_873_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_873_, 0, v_val_870_);
v___x_872_ = v_reuseFailAlloc_873_;
goto v_reusejp_871_;
}
v_reusejp_871_:
{
return v___x_872_;
}
}
}
}
else
{
lean_object* v_a_875_; lean_object* v___x_877_; uint8_t v_isShared_878_; uint8_t v_isSharedCheck_882_; 
v_a_875_ = lean_ctor_get(v___x_859_, 0);
v_isSharedCheck_882_ = !lean_is_exclusive(v___x_859_);
if (v_isSharedCheck_882_ == 0)
{
v___x_877_ = v___x_859_;
v_isShared_878_ = v_isSharedCheck_882_;
goto v_resetjp_876_;
}
else
{
lean_inc(v_a_875_);
lean_dec(v___x_859_);
v___x_877_ = lean_box(0);
v_isShared_878_ = v_isSharedCheck_882_;
goto v_resetjp_876_;
}
v_resetjp_876_:
{
lean_object* v___x_880_; 
if (v_isShared_878_ == 0)
{
v___x_880_ = v___x_877_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v_a_875_);
v___x_880_ = v_reuseFailAlloc_881_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
return v___x_880_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__21(lean_object* v_init_883_, lean_object* v_as_884_, size_t v_sz_885_, size_t v_i_886_, lean_object* v_b_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_){
_start:
{
uint8_t v___x_893_; 
v___x_893_ = lean_usize_dec_lt(v_i_886_, v_sz_885_);
if (v___x_893_ == 0)
{
lean_object* v___x_894_; 
v___x_894_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_894_, 0, v_b_887_);
return v___x_894_;
}
else
{
lean_object* v_snd_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_929_; 
v_snd_895_ = lean_ctor_get(v_b_887_, 1);
v_isSharedCheck_929_ = !lean_is_exclusive(v_b_887_);
if (v_isSharedCheck_929_ == 0)
{
lean_object* v_unused_930_; 
v_unused_930_ = lean_ctor_get(v_b_887_, 0);
lean_dec(v_unused_930_);
v___x_897_ = v_b_887_;
v_isShared_898_ = v_isSharedCheck_929_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_snd_895_);
lean_dec(v_b_887_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_929_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v_a_899_; lean_object* v___x_900_; 
v_a_899_ = lean_array_uget_borrowed(v_as_884_, v_i_886_);
lean_inc(v_snd_895_);
v___x_900_ = lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13(v_init_883_, v_a_899_, v_snd_895_, v___y_888_, v___y_889_, v___y_890_, v___y_891_);
if (lean_obj_tag(v___x_900_) == 0)
{
lean_object* v_a_901_; lean_object* v___x_903_; uint8_t v_isShared_904_; uint8_t v_isSharedCheck_920_; 
v_a_901_ = lean_ctor_get(v___x_900_, 0);
v_isSharedCheck_920_ = !lean_is_exclusive(v___x_900_);
if (v_isSharedCheck_920_ == 0)
{
v___x_903_ = v___x_900_;
v_isShared_904_ = v_isSharedCheck_920_;
goto v_resetjp_902_;
}
else
{
lean_inc(v_a_901_);
lean_dec(v___x_900_);
v___x_903_ = lean_box(0);
v_isShared_904_ = v_isSharedCheck_920_;
goto v_resetjp_902_;
}
v_resetjp_902_:
{
if (lean_obj_tag(v_a_901_) == 0)
{
lean_object* v___x_905_; lean_object* v___x_907_; 
v___x_905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_905_, 0, v_a_901_);
if (v_isShared_898_ == 0)
{
lean_ctor_set(v___x_897_, 0, v___x_905_);
v___x_907_ = v___x_897_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_911_; 
v_reuseFailAlloc_911_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_911_, 0, v___x_905_);
lean_ctor_set(v_reuseFailAlloc_911_, 1, v_snd_895_);
v___x_907_ = v_reuseFailAlloc_911_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
lean_object* v___x_909_; 
if (v_isShared_904_ == 0)
{
lean_ctor_set(v___x_903_, 0, v___x_907_);
v___x_909_ = v___x_903_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_910_; 
v_reuseFailAlloc_910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_910_, 0, v___x_907_);
v___x_909_ = v_reuseFailAlloc_910_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
return v___x_909_;
}
}
}
else
{
lean_object* v_a_912_; lean_object* v___x_913_; lean_object* v___x_915_; 
lean_del_object(v___x_903_);
lean_dec(v_snd_895_);
v_a_912_ = lean_ctor_get(v_a_901_, 0);
lean_inc(v_a_912_);
lean_dec_ref_known(v_a_901_, 1);
v___x_913_ = lean_box(0);
if (v_isShared_898_ == 0)
{
lean_ctor_set(v___x_897_, 1, v_a_912_);
lean_ctor_set(v___x_897_, 0, v___x_913_);
v___x_915_ = v___x_897_;
goto v_reusejp_914_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v___x_913_);
lean_ctor_set(v_reuseFailAlloc_919_, 1, v_a_912_);
v___x_915_ = v_reuseFailAlloc_919_;
goto v_reusejp_914_;
}
v_reusejp_914_:
{
size_t v___x_916_; size_t v___x_917_; 
v___x_916_ = ((size_t)1ULL);
v___x_917_ = lean_usize_add(v_i_886_, v___x_916_);
v_i_886_ = v___x_917_;
v_b_887_ = v___x_915_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_928_; 
lean_del_object(v___x_897_);
lean_dec(v_snd_895_);
v_a_921_ = lean_ctor_get(v___x_900_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_900_);
if (v_isSharedCheck_928_ == 0)
{
v___x_923_ = v___x_900_;
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_900_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_926_; 
if (v_isShared_924_ == 0)
{
v___x_926_ = v___x_923_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v_a_921_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__21___boxed(lean_object* v_init_931_, lean_object* v_as_932_, lean_object* v_sz_933_, lean_object* v_i_934_, lean_object* v_b_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_){
_start:
{
size_t v_sz_boxed_941_; size_t v_i_boxed_942_; lean_object* v_res_943_; 
v_sz_boxed_941_ = lean_unbox_usize(v_sz_933_);
lean_dec(v_sz_933_);
v_i_boxed_942_ = lean_unbox_usize(v_i_934_);
lean_dec(v_i_934_);
v_res_943_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__21(v_init_931_, v_as_932_, v_sz_boxed_941_, v_i_boxed_942_, v_b_935_, v___y_936_, v___y_937_, v___y_938_, v___y_939_);
lean_dec(v___y_939_);
lean_dec_ref(v___y_938_);
lean_dec(v___y_937_);
lean_dec_ref(v___y_936_);
lean_dec_ref(v_as_932_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13___boxed(lean_object* v_init_944_, lean_object* v_n_945_, lean_object* v_b_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13(v_init_944_, v_n_945_, v_b_946_, v___y_947_, v___y_948_, v___y_949_, v___y_950_);
lean_dec(v___y_950_);
lean_dec_ref(v___y_949_);
lean_dec(v___y_948_);
lean_dec_ref(v___y_947_);
lean_dec_ref(v_n_945_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11(lean_object* v_t_953_, lean_object* v_init_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_){
_start:
{
lean_object* v_root_960_; lean_object* v_tail_961_; lean_object* v___x_962_; 
v_root_960_ = lean_ctor_get(v_t_953_, 0);
v_tail_961_ = lean_ctor_get(v_t_953_, 1);
v___x_962_ = lp_batteries_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13(v_init_954_, v_root_960_, v_init_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
if (lean_obj_tag(v___x_962_) == 0)
{
lean_object* v_a_963_; lean_object* v___x_965_; uint8_t v_isShared_966_; uint8_t v_isSharedCheck_999_; 
v_a_963_ = lean_ctor_get(v___x_962_, 0);
v_isSharedCheck_999_ = !lean_is_exclusive(v___x_962_);
if (v_isSharedCheck_999_ == 0)
{
v___x_965_ = v___x_962_;
v_isShared_966_ = v_isSharedCheck_999_;
goto v_resetjp_964_;
}
else
{
lean_inc(v_a_963_);
lean_dec(v___x_962_);
v___x_965_ = lean_box(0);
v_isShared_966_ = v_isSharedCheck_999_;
goto v_resetjp_964_;
}
v_resetjp_964_:
{
if (lean_obj_tag(v_a_963_) == 0)
{
lean_object* v_a_967_; lean_object* v___x_969_; 
v_a_967_ = lean_ctor_get(v_a_963_, 0);
lean_inc(v_a_967_);
lean_dec_ref_known(v_a_963_, 1);
if (v_isShared_966_ == 0)
{
lean_ctor_set(v___x_965_, 0, v_a_967_);
v___x_969_ = v___x_965_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v_a_967_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
else
{
lean_object* v_a_971_; lean_object* v___x_972_; lean_object* v___x_973_; size_t v_sz_974_; size_t v___x_975_; lean_object* v___x_976_; 
lean_del_object(v___x_965_);
v_a_971_ = lean_ctor_get(v_a_963_, 0);
lean_inc(v_a_971_);
lean_dec_ref_known(v_a_963_, 1);
v___x_972_ = lean_box(0);
v___x_973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_973_, 0, v___x_972_);
lean_ctor_set(v___x_973_, 1, v_a_971_);
v_sz_974_ = lean_array_size(v_tail_961_);
v___x_975_ = ((size_t)0ULL);
v___x_976_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14(v_tail_961_, v_sz_974_, v___x_975_, v___x_973_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
if (lean_obj_tag(v___x_976_) == 0)
{
lean_object* v_a_977_; lean_object* v___x_979_; uint8_t v_isShared_980_; uint8_t v_isSharedCheck_990_; 
v_a_977_ = lean_ctor_get(v___x_976_, 0);
v_isSharedCheck_990_ = !lean_is_exclusive(v___x_976_);
if (v_isSharedCheck_990_ == 0)
{
v___x_979_ = v___x_976_;
v_isShared_980_ = v_isSharedCheck_990_;
goto v_resetjp_978_;
}
else
{
lean_inc(v_a_977_);
lean_dec(v___x_976_);
v___x_979_ = lean_box(0);
v_isShared_980_ = v_isSharedCheck_990_;
goto v_resetjp_978_;
}
v_resetjp_978_:
{
lean_object* v_fst_981_; 
v_fst_981_ = lean_ctor_get(v_a_977_, 0);
if (lean_obj_tag(v_fst_981_) == 0)
{
lean_object* v_snd_982_; lean_object* v___x_984_; 
v_snd_982_ = lean_ctor_get(v_a_977_, 1);
lean_inc(v_snd_982_);
lean_dec(v_a_977_);
if (v_isShared_980_ == 0)
{
lean_ctor_set(v___x_979_, 0, v_snd_982_);
v___x_984_ = v___x_979_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_985_; 
v_reuseFailAlloc_985_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_985_, 0, v_snd_982_);
v___x_984_ = v_reuseFailAlloc_985_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
return v___x_984_;
}
}
else
{
lean_object* v_val_986_; lean_object* v___x_988_; 
lean_inc_ref(v_fst_981_);
lean_dec(v_a_977_);
v_val_986_ = lean_ctor_get(v_fst_981_, 0);
lean_inc(v_val_986_);
lean_dec_ref_known(v_fst_981_, 1);
if (v_isShared_980_ == 0)
{
lean_ctor_set(v___x_979_, 0, v_val_986_);
v___x_988_ = v___x_979_;
goto v_reusejp_987_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v_val_986_);
v___x_988_ = v_reuseFailAlloc_989_;
goto v_reusejp_987_;
}
v_reusejp_987_:
{
return v___x_988_;
}
}
}
}
else
{
lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_998_; 
v_a_991_ = lean_ctor_get(v___x_976_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v___x_976_);
if (v_isSharedCheck_998_ == 0)
{
v___x_993_ = v___x_976_;
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_a_991_);
lean_dec(v___x_976_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_996_; 
if (v_isShared_994_ == 0)
{
v___x_996_ = v___x_993_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v_a_991_);
v___x_996_ = v_reuseFailAlloc_997_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
return v___x_996_;
}
}
}
}
}
}
else
{
lean_object* v_a_1000_; lean_object* v___x_1002_; uint8_t v_isShared_1003_; uint8_t v_isSharedCheck_1007_; 
v_a_1000_ = lean_ctor_get(v___x_962_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v___x_962_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_1002_ = v___x_962_;
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
else
{
lean_inc(v_a_1000_);
lean_dec(v___x_962_);
v___x_1002_ = lean_box(0);
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
v_resetjp_1001_:
{
lean_object* v___x_1005_; 
if (v_isShared_1003_ == 0)
{
v___x_1005_ = v___x_1002_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v_a_1000_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11___boxed(lean_object* v_t_1008_, lean_object* v_init_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_){
_start:
{
lean_object* v_res_1015_; 
v_res_1015_ = lp_batteries_Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11(v_t_1008_, v_init_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
lean_dec(v___y_1011_);
lean_dec_ref(v___y_1010_);
lean_dec_ref(v_t_1008_);
return v_res_1015_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5(lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_){
_start:
{
lean_object* v___x_1021_; lean_object* v_traceState_1022_; lean_object* v_traces_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1021_ = lean_st_ref_get(v___y_1019_);
v_traceState_1022_ = lean_ctor_get(v___x_1021_, 4);
lean_inc_ref(v_traceState_1022_);
lean_dec(v___x_1021_);
v_traces_1023_ = lean_ctor_get(v_traceState_1022_, 0);
lean_inc_ref(v_traces_1023_);
lean_dec_ref(v_traceState_1022_);
v___x_1024_ = lean_box(0);
v___x_1025_ = lp_batteries_Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11(v_traces_1023_, v___x_1024_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
lean_dec_ref(v_traces_1023_);
if (lean_obj_tag(v___x_1025_) == 0)
{
lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1032_; 
v_isSharedCheck_1032_ = !lean_is_exclusive(v___x_1025_);
if (v_isSharedCheck_1032_ == 0)
{
lean_object* v_unused_1033_; 
v_unused_1033_ = lean_ctor_get(v___x_1025_, 0);
lean_dec(v_unused_1033_);
v___x_1027_ = v___x_1025_;
v_isShared_1028_ = v_isSharedCheck_1032_;
goto v_resetjp_1026_;
}
else
{
lean_dec(v___x_1025_);
v___x_1027_ = lean_box(0);
v_isShared_1028_ = v_isSharedCheck_1032_;
goto v_resetjp_1026_;
}
v_resetjp_1026_:
{
lean_object* v___x_1030_; 
if (v_isShared_1028_ == 0)
{
lean_ctor_set(v___x_1027_, 0, v___x_1024_);
v___x_1030_ = v___x_1027_;
goto v_reusejp_1029_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v___x_1024_);
v___x_1030_ = v_reuseFailAlloc_1031_;
goto v_reusejp_1029_;
}
v_reusejp_1029_:
{
return v___x_1030_;
}
}
}
else
{
return v___x_1025_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5___boxed(lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_){
_start:
{
lean_object* v_res_1039_; 
v_res_1039_ = lp_batteries_Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5(v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_);
lean_dec(v___y_1037_);
lean_dec_ref(v___y_1036_);
lean_dec(v___y_1035_);
lean_dec_ref(v___y_1034_);
return v_res_1039_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1040_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1);
v___x_1041_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
lean_ctor_set(v___x_1041_, 1, v___x_1040_);
lean_ctor_set(v___x_1041_, 2, v___x_1040_);
lean_ctor_set(v___x_1041_, 3, v___x_1040_);
lean_ctor_set(v___x_1041_, 4, v___x_1040_);
lean_ctor_set(v___x_1041_, 5, v___x_1040_);
return v___x_1041_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1042_ = lean_unsigned_to_nat(32u);
v___x_1043_ = lean_mk_empty_array_with_capacity(v___x_1042_);
v___x_1044_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1044_, 0, v___x_1043_);
return v___x_1044_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__2(void){
_start:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; 
v___x_1045_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1);
v___x_1046_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1045_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
lean_ctor_set(v___x_1046_, 2, v___x_1045_);
lean_ctor_set(v___x_1046_, 3, v___x_1045_);
lean_ctor_set(v___x_1046_, 4, v___x_1045_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0(lean_object* v___x_1047_, lean_object* v___x_1048_, lean_object* v_test_1049_, lean_object* v_v_1050_, uint8_t v___y_1051_, lean_object* v_x_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_){
_start:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; size_t v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; 
v___x_1056_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__1);
lean_inc_n(v___x_1047_, 5);
v___x_1057_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1047_);
lean_ctor_set(v___x_1057_, 1, v___x_1047_);
lean_ctor_set(v___x_1057_, 2, v___x_1047_);
lean_ctor_set(v___x_1057_, 3, v___x_1047_);
lean_ctor_set(v___x_1057_, 4, v___x_1056_);
lean_ctor_set(v___x_1057_, 5, v___x_1056_);
lean_ctor_set(v___x_1057_, 6, v___x_1056_);
lean_ctor_set(v___x_1057_, 7, v___x_1056_);
lean_ctor_set(v___x_1057_, 8, v___x_1056_);
lean_ctor_set(v___x_1057_, 9, v___x_1056_);
v___x_1058_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__0, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__0_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__0);
v___x_1059_ = lean_unsigned_to_nat(32u);
v___x_1060_ = lean_mk_empty_array_with_capacity(v___x_1059_);
v___x_1061_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__1);
v___x_1062_ = ((size_t)5ULL);
v___x_1063_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1063_, 0, v___x_1061_);
lean_ctor_set(v___x_1063_, 1, v___x_1060_);
lean_ctor_set(v___x_1063_, 2, v___x_1047_);
lean_ctor_set(v___x_1063_, 3, v___x_1047_);
lean_ctor_set_usize(v___x_1063_, 4, v___x_1062_);
v___x_1064_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__2, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__2_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___closed__2);
v___x_1065_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1057_);
lean_ctor_set(v___x_1065_, 1, v___x_1058_);
lean_ctor_set(v___x_1065_, 2, v___x_1048_);
lean_ctor_set(v___x_1065_, 3, v___x_1063_);
lean_ctor_set(v___x_1065_, 4, v___x_1064_);
v___x_1066_ = lean_st_mk_ref(v___x_1065_);
v___x_1067_ = l_Lean_Elab_Command_mkMetaContext;
lean_inc(v___y_1054_);
lean_inc_ref(v___y_1053_);
lean_inc(v___x_1066_);
v___x_1068_ = lean_apply_6(v_test_1049_, v_v_1050_, v___x_1067_, v___x_1066_, v___y_1053_, v___y_1054_, lean_box(0));
if (lean_obj_tag(v___x_1068_) == 0)
{
lean_object* v_a_1069_; lean_object* v___x_1071_; uint8_t v_isShared_1072_; uint8_t v_isSharedCheck_1087_; 
v_a_1069_ = lean_ctor_get(v___x_1068_, 0);
v_isSharedCheck_1087_ = !lean_is_exclusive(v___x_1068_);
if (v_isSharedCheck_1087_ == 0)
{
v___x_1071_ = v___x_1068_;
v_isShared_1072_ = v_isSharedCheck_1087_;
goto v_resetjp_1070_;
}
else
{
lean_inc(v_a_1069_);
lean_dec(v___x_1068_);
v___x_1071_ = lean_box(0);
v_isShared_1072_ = v_isSharedCheck_1087_;
goto v_resetjp_1070_;
}
v_resetjp_1070_:
{
if (v___y_1051_ == 0)
{
goto v___jp_1073_;
}
else
{
lean_object* v___x_1078_; 
v___x_1078_ = lp_batteries_Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5(v___x_1067_, v___x_1066_, v___y_1053_, v___y_1054_);
if (lean_obj_tag(v___x_1078_) == 0)
{
lean_dec_ref_known(v___x_1078_, 1);
goto v___jp_1073_;
}
else
{
lean_object* v_a_1079_; lean_object* v___x_1081_; uint8_t v_isShared_1082_; uint8_t v_isSharedCheck_1086_; 
lean_del_object(v___x_1071_);
lean_dec(v_a_1069_);
lean_dec(v___x_1066_);
v_a_1079_ = lean_ctor_get(v___x_1078_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1078_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1081_ = v___x_1078_;
v_isShared_1082_ = v_isSharedCheck_1086_;
goto v_resetjp_1080_;
}
else
{
lean_inc(v_a_1079_);
lean_dec(v___x_1078_);
v___x_1081_ = lean_box(0);
v_isShared_1082_ = v_isSharedCheck_1086_;
goto v_resetjp_1080_;
}
v_resetjp_1080_:
{
lean_object* v___x_1084_; 
if (v_isShared_1082_ == 0)
{
v___x_1084_ = v___x_1081_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v_a_1079_);
v___x_1084_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1083_;
}
v_reusejp_1083_:
{
return v___x_1084_;
}
}
}
}
v___jp_1073_:
{
lean_object* v___x_1074_; lean_object* v___x_1076_; 
v___x_1074_ = lean_st_ref_get(v___x_1066_);
lean_dec(v___x_1066_);
lean_dec(v___x_1074_);
if (v_isShared_1072_ == 0)
{
v___x_1076_ = v___x_1071_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_a_1069_);
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
else
{
lean_dec(v___x_1066_);
return v___x_1068_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___boxed(lean_object* v___x_1088_, lean_object* v___x_1089_, lean_object* v_test_1090_, lean_object* v_v_1091_, lean_object* v___y_1092_, lean_object* v_x_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
uint8_t v___y_24639__boxed_1097_; lean_object* v_res_1098_; 
v___y_24639__boxed_1097_ = lean_unbox(v___y_1092_);
v_res_1098_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0(v___x_1088_, v___x_1089_, v_test_1090_, v_v_1091_, v___y_24639__boxed_1097_, v_x_1093_, v___y_1094_, v___y_1095_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
return v_res_1098_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__1(lean_object* v_a_1099_, lean_object* v___x_1100_){
_start:
{
lean_object* v___x_1102_; 
v___x_1102_ = lean_apply_2(v_a_1099_, v___x_1100_, lean_box(0));
if (lean_obj_tag(v___x_1102_) == 0)
{
lean_object* v_a_1103_; lean_object* v___x_1105_; uint8_t v_isShared_1106_; uint8_t v_isSharedCheck_1110_; 
v_a_1103_ = lean_ctor_get(v___x_1102_, 0);
v_isSharedCheck_1110_ = !lean_is_exclusive(v___x_1102_);
if (v_isSharedCheck_1110_ == 0)
{
v___x_1105_ = v___x_1102_;
v_isShared_1106_ = v_isSharedCheck_1110_;
goto v_resetjp_1104_;
}
else
{
lean_inc(v_a_1103_);
lean_dec(v___x_1102_);
v___x_1105_ = lean_box(0);
v_isShared_1106_ = v_isSharedCheck_1110_;
goto v_resetjp_1104_;
}
v_resetjp_1104_:
{
lean_object* v___x_1108_; 
if (v_isShared_1106_ == 0)
{
lean_ctor_set_tag(v___x_1105_, 1);
v___x_1108_ = v___x_1105_;
goto v_reusejp_1107_;
}
else
{
lean_object* v_reuseFailAlloc_1109_; 
v_reuseFailAlloc_1109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1109_, 0, v_a_1103_);
v___x_1108_ = v_reuseFailAlloc_1109_;
goto v_reusejp_1107_;
}
v_reusejp_1107_:
{
return v___x_1108_;
}
}
}
else
{
lean_object* v_a_1111_; lean_object* v___x_1113_; uint8_t v_isShared_1114_; uint8_t v_isSharedCheck_1118_; 
v_a_1111_ = lean_ctor_get(v___x_1102_, 0);
v_isSharedCheck_1118_ = !lean_is_exclusive(v___x_1102_);
if (v_isSharedCheck_1118_ == 0)
{
v___x_1113_ = v___x_1102_;
v_isShared_1114_ = v_isSharedCheck_1118_;
goto v_resetjp_1112_;
}
else
{
lean_inc(v_a_1111_);
lean_dec(v___x_1102_);
v___x_1113_ = lean_box(0);
v_isShared_1114_ = v_isSharedCheck_1118_;
goto v_resetjp_1112_;
}
v_resetjp_1112_:
{
lean_object* v___x_1116_; 
if (v_isShared_1114_ == 0)
{
lean_ctor_set_tag(v___x_1113_, 0);
v___x_1116_ = v___x_1113_;
goto v_reusejp_1115_;
}
else
{
lean_object* v_reuseFailAlloc_1117_; 
v_reuseFailAlloc_1117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1117_, 0, v_a_1111_);
v___x_1116_ = v_reuseFailAlloc_1117_;
goto v_reusejp_1115_;
}
v_reusejp_1115_:
{
return v___x_1116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__1___boxed(lean_object* v_a_1119_, lean_object* v___x_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__1(v_a_1119_, v___x_1120_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6(lean_object* v_linter_1123_, uint8_t v___y_1124_, size_t v_sz_1125_, size_t v_i_1126_, lean_object* v_bs_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
uint8_t v___x_1131_; 
v___x_1131_ = lean_usize_dec_lt(v_i_1126_, v_sz_1125_);
if (v___x_1131_ == 0)
{
lean_object* v___x_1132_; 
lean_dec_ref(v_linter_1123_);
v___x_1132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1132_, 0, v_bs_1127_);
return v___x_1132_;
}
else
{
lean_object* v_toLinter_1133_; lean_object* v_test_1134_; lean_object* v_v_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___f_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; 
v_toLinter_1133_ = lean_ctor_get(v_linter_1123_, 0);
v_test_1134_ = lean_ctor_get(v_toLinter_1133_, 0);
v_v_1135_ = lean_array_uget(v_bs_1127_, v_i_1126_);
v___x_1136_ = lean_unsigned_to_nat(0u);
v___x_1137_ = lean_box(1);
v___x_1138_ = lean_box(v___y_1124_);
lean_inc(v_v_1135_);
lean_inc_ref(v_test_1134_);
v___f_1139_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__0___boxed), 9, 5);
lean_closure_set(v___f_1139_, 0, v___x_1136_);
lean_closure_set(v___f_1139_, 1, v___x_1137_);
lean_closure_set(v___f_1139_, 2, v_test_1134_);
lean_closure_set(v___f_1139_, 3, v_v_1135_);
lean_closure_set(v___f_1139_, 4, v___x_1138_);
v___x_1140_ = lean_box(0);
v___x_1141_ = l_Lean_Core_wrapAsync___redArg(v___f_1139_, v___x_1140_, v___y_1128_, v___y_1129_);
if (lean_obj_tag(v___x_1141_) == 0)
{
lean_object* v_a_1142_; lean_object* v___x_1143_; lean_object* v___f_1144_; lean_object* v___x_1145_; lean_object* v_bs_x27_1146_; lean_object* v___x_1147_; size_t v___x_1148_; size_t v___x_1149_; lean_object* v___x_1150_; 
v_a_1142_ = lean_ctor_get(v___x_1141_, 0);
lean_inc(v_a_1142_);
lean_dec_ref_known(v___x_1141_, 1);
v___x_1143_ = lean_box(0);
v___f_1144_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___lam__1___boxed), 3, 2);
lean_closure_set(v___f_1144_, 0, v_a_1142_);
lean_closure_set(v___f_1144_, 1, v___x_1143_);
v___x_1145_ = lean_io_as_task(v___f_1144_, v___x_1136_);
v_bs_x27_1146_ = lean_array_uset(v_bs_1127_, v_i_1126_, v___x_1136_);
v___x_1147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1147_, 0, v_v_1135_);
lean_ctor_set(v___x_1147_, 1, v___x_1145_);
v___x_1148_ = ((size_t)1ULL);
v___x_1149_ = lean_usize_add(v_i_1126_, v___x_1148_);
v___x_1150_ = lean_array_uset(v_bs_x27_1146_, v_i_1126_, v___x_1147_);
v_i_1126_ = v___x_1149_;
v_bs_1127_ = v___x_1150_;
goto _start;
}
else
{
lean_object* v_a_1152_; lean_object* v___x_1154_; uint8_t v_isShared_1155_; uint8_t v_isSharedCheck_1159_; 
lean_dec(v_v_1135_);
lean_dec_ref(v_bs_1127_);
lean_dec_ref(v_linter_1123_);
v_a_1152_ = lean_ctor_get(v___x_1141_, 0);
v_isSharedCheck_1159_ = !lean_is_exclusive(v___x_1141_);
if (v_isSharedCheck_1159_ == 0)
{
v___x_1154_ = v___x_1141_;
v_isShared_1155_ = v_isSharedCheck_1159_;
goto v_resetjp_1153_;
}
else
{
lean_inc(v_a_1152_);
lean_dec(v___x_1141_);
v___x_1154_ = lean_box(0);
v_isShared_1155_ = v_isSharedCheck_1159_;
goto v_resetjp_1153_;
}
v_resetjp_1153_:
{
lean_object* v___x_1157_; 
if (v_isShared_1155_ == 0)
{
v___x_1157_ = v___x_1154_;
goto v_reusejp_1156_;
}
else
{
lean_object* v_reuseFailAlloc_1158_; 
v_reuseFailAlloc_1158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1158_, 0, v_a_1152_);
v___x_1157_ = v_reuseFailAlloc_1158_;
goto v_reusejp_1156_;
}
v_reusejp_1156_:
{
return v___x_1157_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6___boxed(lean_object* v_linter_1160_, lean_object* v___y_1161_, lean_object* v_sz_1162_, lean_object* v_i_1163_, lean_object* v_bs_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_){
_start:
{
uint8_t v___y_24774__boxed_1168_; size_t v_sz_boxed_1169_; size_t v_i_boxed_1170_; lean_object* v_res_1171_; 
v___y_24774__boxed_1168_ = lean_unbox(v___y_1161_);
v_sz_boxed_1169_ = lean_unbox_usize(v_sz_1162_);
lean_dec(v_sz_1162_);
v_i_boxed_1170_ = lean_unbox_usize(v_i_1163_);
lean_dec(v_i_1163_);
v_res_1171_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6(v_linter_1160_, v___y_24774__boxed_1168_, v_sz_boxed_1169_, v_i_boxed_1170_, v_bs_1164_, v___y_1165_, v___y_1166_);
lean_dec(v___y_1166_);
lean_dec_ref(v___y_1165_);
return v_res_1171_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5(void){
_start:
{
lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___x_1180_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_1181_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__4));
v___x_1182_ = l_Lean_Name_append(v___x_1181_, v___x_1180_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9(uint8_t v___y_1192_, lean_object* v_decls_1193_, lean_object* v_currentModule_1194_, size_t v_sz_1195_, size_t v_i_1196_, lean_object* v_bs_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_){
_start:
{
uint8_t v___x_1201_; 
v___x_1201_ = lean_usize_dec_lt(v_i_1196_, v_sz_1195_);
if (v___x_1201_ == 0)
{
lean_object* v___x_1202_; 
lean_dec(v_currentModule_1194_);
v___x_1202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1202_, 0, v_bs_1197_);
return v___x_1202_;
}
else
{
lean_object* v_v_1203_; lean_object* v___x_1204_; lean_object* v_bs_x27_1205_; lean_object* v_a_1207_; lean_object* v___y_1218_; 
v_v_1203_ = lean_array_uget(v_bs_1197_, v_i_1196_);
v___x_1204_ = lean_unsigned_to_nat(0u);
v_bs_x27_1205_ = lean_array_uset(v_bs_1197_, v_i_1196_, v___x_1204_);
if (v___y_1192_ == 0)
{
lean_object* v_options_1240_; uint8_t v_hasTrace_1241_; 
v_options_1240_ = lean_ctor_get(v___y_1198_, 2);
v_hasTrace_1241_ = lean_ctor_get_uint8(v_options_1240_, sizeof(void*)*1);
if (v_hasTrace_1241_ == 0)
{
goto v___jp_1228_;
}
else
{
lean_object* v_inheritedTraceOptions_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; uint8_t v___x_1245_; lean_object* v___y_1247_; 
v_inheritedTraceOptions_1242_ = lean_ctor_get(v___y_1198_, 13);
v___x_1243_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_1244_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5);
v___x_1245_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1242_, v_options_1240_, v___x_1244_);
if (v___x_1245_ == 0)
{
goto v___jp_1228_;
}
else
{
if (lean_obj_tag(v_currentModule_1194_) == 1)
{
lean_object* v_val_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; 
v_val_1268_ = lean_ctor_get(v_currentModule_1194_, 0);
v___x_1269_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
lean_inc(v_val_1268_);
v___x_1270_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1268_, v___x_1245_);
v___x_1271_ = lean_string_append(v___x_1269_, v___x_1270_);
lean_dec_ref(v___x_1270_);
v___x_1272_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1273_ = lean_string_append(v___x_1271_, v___x_1272_);
v___y_1247_ = v___x_1273_;
goto v___jp_1246_;
}
else
{
lean_object* v___x_1274_; 
v___x_1274_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1247_ = v___x_1274_;
goto v___jp_1246_;
}
}
v___jp_1246_:
{
lean_object* v_name_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; 
v_name_1248_ = lean_ctor_get(v_v_1203_, 1);
v___x_1249_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6));
lean_inc(v_name_1248_);
v___x_1250_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1248_, v___x_1245_);
v___x_1251_ = lean_string_append(v___x_1249_, v___x_1250_);
lean_dec_ref(v___x_1250_);
v___x_1252_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7));
v___x_1253_ = lean_string_append(v___x_1251_, v___x_1252_);
v___x_1254_ = lean_string_append(v___y_1247_, v___x_1253_);
lean_dec_ref(v___x_1253_);
v___x_1255_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__8));
v___x_1256_ = lean_string_append(v___x_1254_, v___x_1255_);
v___x_1257_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1257_, 0, v___x_1256_);
v___x_1258_ = l_Lean_MessageData_ofFormat(v___x_1257_);
v___x_1259_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(v___x_1243_, v___x_1258_, v___y_1198_, v___y_1199_);
if (lean_obj_tag(v___x_1259_) == 0)
{
lean_dec_ref_known(v___x_1259_, 1);
goto v___jp_1228_;
}
else
{
lean_object* v_a_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1267_; 
lean_dec_ref(v_bs_x27_1205_);
lean_dec(v_v_1203_);
lean_dec(v_currentModule_1194_);
v_a_1260_ = lean_ctor_get(v___x_1259_, 0);
v_isSharedCheck_1267_ = !lean_is_exclusive(v___x_1259_);
if (v_isSharedCheck_1267_ == 0)
{
v___x_1262_ = v___x_1259_;
v_isShared_1263_ = v_isSharedCheck_1267_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_a_1260_);
lean_dec(v___x_1259_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1267_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v___x_1265_; 
if (v_isShared_1263_ == 0)
{
v___x_1265_ = v___x_1262_;
goto v_reusejp_1264_;
}
else
{
lean_object* v_reuseFailAlloc_1266_; 
v_reuseFailAlloc_1266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1266_, 0, v_a_1260_);
v___x_1265_ = v_reuseFailAlloc_1266_;
goto v_reusejp_1264_;
}
v_reusejp_1264_:
{
return v___x_1265_;
}
}
}
}
}
}
else
{
lean_object* v___x_1275_; uint8_t v___x_1276_; lean_object* v___x_1277_; 
v___x_1275_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11));
v___x_1276_ = 0;
v___x_1277_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v___x_1275_, v___x_1276_, v___y_1198_);
if (lean_obj_tag(v___x_1277_) == 0)
{
lean_object* v_a_1278_; lean_object* v___y_1280_; uint8_t v___x_1305_; 
v_a_1278_ = lean_ctor_get(v___x_1277_, 0);
lean_inc(v_a_1278_);
lean_dec_ref_known(v___x_1277_, 1);
v___x_1305_ = lean_unbox(v_a_1278_);
if (v___x_1305_ == 0)
{
lean_dec(v_a_1278_);
goto v___jp_1228_;
}
else
{
if (lean_obj_tag(v_currentModule_1194_) == 1)
{
lean_object* v_val_1306_; lean_object* v___x_1307_; uint8_t v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; 
v_val_1306_ = lean_ctor_get(v_currentModule_1194_, 0);
v___x_1307_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
v___x_1308_ = lean_unbox(v_a_1278_);
lean_inc(v_val_1306_);
v___x_1309_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1306_, v___x_1308_);
v___x_1310_ = lean_string_append(v___x_1307_, v___x_1309_);
lean_dec_ref(v___x_1309_);
v___x_1311_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1312_ = lean_string_append(v___x_1310_, v___x_1311_);
v___y_1280_ = v___x_1312_;
goto v___jp_1279_;
}
else
{
lean_object* v___x_1313_; 
v___x_1313_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1280_ = v___x_1313_;
goto v___jp_1279_;
}
}
v___jp_1279_:
{
lean_object* v_name_1281_; lean_object* v___x_1282_; uint8_t v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; 
v_name_1281_ = lean_ctor_get(v_v_1203_, 1);
v___x_1282_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6));
v___x_1283_ = lean_unbox(v_a_1278_);
lean_dec(v_a_1278_);
lean_inc(v_name_1281_);
v___x_1284_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1281_, v___x_1283_);
v___x_1285_ = lean_string_append(v___x_1282_, v___x_1284_);
lean_dec_ref(v___x_1284_);
v___x_1286_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7));
v___x_1287_ = lean_string_append(v___x_1285_, v___x_1286_);
v___x_1288_ = lean_string_append(v___y_1280_, v___x_1287_);
lean_dec_ref(v___x_1287_);
v___x_1289_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__8));
v___x_1290_ = lean_string_append(v___x_1288_, v___x_1289_);
v___x_1291_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_1290_);
if (lean_obj_tag(v___x_1291_) == 0)
{
lean_dec_ref_known(v___x_1291_, 1);
goto v___jp_1228_;
}
else
{
lean_object* v_a_1292_; lean_object* v___x_1294_; uint8_t v_isShared_1295_; uint8_t v_isSharedCheck_1304_; 
lean_dec_ref(v_bs_x27_1205_);
lean_dec(v_v_1203_);
lean_dec(v_currentModule_1194_);
v_a_1292_ = lean_ctor_get(v___x_1291_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1291_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1294_ = v___x_1291_;
v_isShared_1295_ = v_isSharedCheck_1304_;
goto v_resetjp_1293_;
}
else
{
lean_inc(v_a_1292_);
lean_dec(v___x_1291_);
v___x_1294_ = lean_box(0);
v_isShared_1295_ = v_isSharedCheck_1304_;
goto v_resetjp_1293_;
}
v_resetjp_1293_:
{
lean_object* v_ref_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1302_; 
v_ref_1296_ = lean_ctor_get(v___y_1198_, 5);
v___x_1297_ = lean_io_error_to_string(v_a_1292_);
v___x_1298_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1297_);
v___x_1299_ = l_Lean_MessageData_ofFormat(v___x_1298_);
lean_inc(v_ref_1296_);
v___x_1300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1300_, 0, v_ref_1296_);
lean_ctor_set(v___x_1300_, 1, v___x_1299_);
if (v_isShared_1295_ == 0)
{
lean_ctor_set(v___x_1294_, 0, v___x_1300_);
v___x_1302_ = v___x_1294_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v___x_1300_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
}
}
}
else
{
lean_object* v_a_1314_; lean_object* v___x_1316_; uint8_t v_isShared_1317_; uint8_t v_isSharedCheck_1321_; 
lean_dec_ref(v_bs_x27_1205_);
lean_dec(v_v_1203_);
lean_dec(v_currentModule_1194_);
v_a_1314_ = lean_ctor_get(v___x_1277_, 0);
v_isSharedCheck_1321_ = !lean_is_exclusive(v___x_1277_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1316_ = v___x_1277_;
v_isShared_1317_ = v_isSharedCheck_1321_;
goto v_resetjp_1315_;
}
else
{
lean_inc(v_a_1314_);
lean_dec(v___x_1277_);
v___x_1316_ = lean_box(0);
v_isShared_1317_ = v_isSharedCheck_1321_;
goto v_resetjp_1315_;
}
v_resetjp_1315_:
{
lean_object* v___x_1319_; 
if (v_isShared_1317_ == 0)
{
v___x_1319_ = v___x_1316_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1320_; 
v_reuseFailAlloc_1320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1320_, 0, v_a_1314_);
v___x_1319_ = v_reuseFailAlloc_1320_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
return v___x_1319_;
}
}
}
}
v___jp_1206_:
{
size_t v_sz_1208_; size_t v___x_1209_; lean_object* v___x_1210_; 
v_sz_1208_ = lean_array_size(v_a_1207_);
v___x_1209_ = ((size_t)0ULL);
lean_inc(v_v_1203_);
v___x_1210_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__6(v_v_1203_, v___y_1192_, v_sz_1208_, v___x_1209_, v_a_1207_, v___y_1198_, v___y_1199_);
if (lean_obj_tag(v___x_1210_) == 0)
{
lean_object* v_a_1211_; lean_object* v___x_1212_; size_t v___x_1213_; size_t v___x_1214_; lean_object* v___x_1215_; 
v_a_1211_ = lean_ctor_get(v___x_1210_, 0);
lean_inc(v_a_1211_);
lean_dec_ref_known(v___x_1210_, 1);
v___x_1212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1212_, 0, v_v_1203_);
lean_ctor_set(v___x_1212_, 1, v_a_1211_);
v___x_1213_ = ((size_t)1ULL);
v___x_1214_ = lean_usize_add(v_i_1196_, v___x_1213_);
v___x_1215_ = lean_array_uset(v_bs_x27_1205_, v_i_1196_, v___x_1212_);
v_i_1196_ = v___x_1214_;
v_bs_1197_ = v___x_1215_;
goto _start;
}
else
{
lean_dec_ref(v_bs_x27_1205_);
lean_dec(v_v_1203_);
lean_dec(v_currentModule_1194_);
return v___x_1210_;
}
}
v___jp_1217_:
{
if (lean_obj_tag(v___y_1218_) == 0)
{
lean_object* v_a_1219_; 
v_a_1219_ = lean_ctor_get(v___y_1218_, 0);
lean_inc(v_a_1219_);
lean_dec_ref_known(v___y_1218_, 1);
v_a_1207_ = v_a_1219_;
goto v___jp_1206_;
}
else
{
lean_object* v_a_1220_; lean_object* v___x_1222_; uint8_t v_isShared_1223_; uint8_t v_isSharedCheck_1227_; 
lean_dec_ref(v_bs_x27_1205_);
lean_dec(v_v_1203_);
lean_dec(v_currentModule_1194_);
v_a_1220_ = lean_ctor_get(v___y_1218_, 0);
v_isSharedCheck_1227_ = !lean_is_exclusive(v___y_1218_);
if (v_isSharedCheck_1227_ == 0)
{
v___x_1222_ = v___y_1218_;
v_isShared_1223_ = v_isSharedCheck_1227_;
goto v_resetjp_1221_;
}
else
{
lean_inc(v_a_1220_);
lean_dec(v___y_1218_);
v___x_1222_ = lean_box(0);
v_isShared_1223_ = v_isSharedCheck_1227_;
goto v_resetjp_1221_;
}
v_resetjp_1221_:
{
lean_object* v___x_1225_; 
if (v_isShared_1223_ == 0)
{
v___x_1225_ = v___x_1222_;
goto v_reusejp_1224_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v_a_1220_);
v___x_1225_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1224_;
}
v_reusejp_1224_:
{
return v___x_1225_;
}
}
}
}
v___jp_1228_:
{
lean_object* v___x_1229_; lean_object* v___x_1230_; uint8_t v___x_1231_; 
v___x_1229_ = lean_array_get_size(v_decls_1193_);
v___x_1230_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1));
v___x_1231_ = lean_nat_dec_lt(v___x_1204_, v___x_1229_);
if (v___x_1231_ == 0)
{
v_a_1207_ = v___x_1230_;
goto v___jp_1206_;
}
else
{
lean_object* v_name_1232_; uint8_t v___x_1233_; 
v_name_1232_ = lean_ctor_get(v_v_1203_, 1);
v___x_1233_ = lean_nat_dec_le(v___x_1229_, v___x_1229_);
if (v___x_1233_ == 0)
{
if (v___x_1231_ == 0)
{
v_a_1207_ = v___x_1230_;
goto v___jp_1206_;
}
else
{
size_t v___x_1234_; size_t v___x_1235_; lean_object* v___x_1236_; 
v___x_1234_ = ((size_t)0ULL);
v___x_1235_ = lean_usize_of_nat(v___x_1229_);
v___x_1236_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8(v_name_1232_, v_decls_1193_, v___x_1234_, v___x_1235_, v___x_1230_, v___y_1198_, v___y_1199_);
v___y_1218_ = v___x_1236_;
goto v___jp_1217_;
}
}
else
{
size_t v___x_1237_; size_t v___x_1238_; lean_object* v___x_1239_; 
v___x_1237_ = ((size_t)0ULL);
v___x_1238_ = lean_usize_of_nat(v___x_1229_);
v___x_1239_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_lintCore_spec__8(v_name_1232_, v_decls_1193_, v___x_1237_, v___x_1238_, v___x_1230_, v___y_1198_, v___y_1199_);
v___y_1218_ = v___x_1239_;
goto v___jp_1217_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___boxed(lean_object* v___y_1322_, lean_object* v_decls_1323_, lean_object* v_currentModule_1324_, lean_object* v_sz_1325_, lean_object* v_i_1326_, lean_object* v_bs_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_){
_start:
{
uint8_t v___y_24888__boxed_1331_; size_t v_sz_boxed_1332_; size_t v_i_boxed_1333_; lean_object* v_res_1334_; 
v___y_24888__boxed_1331_ = lean_unbox(v___y_1322_);
v_sz_boxed_1332_ = lean_unbox_usize(v_sz_1325_);
lean_dec(v_sz_1325_);
v_i_boxed_1333_ = lean_unbox_usize(v_i_1326_);
lean_dec(v_i_1326_);
v_res_1334_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9(v___y_24888__boxed_1331_, v_decls_1323_, v_currentModule_1324_, v_sz_boxed_1332_, v_i_boxed_1333_, v_bs_1327_, v___y_1328_, v___y_1329_);
lean_dec(v___y_1329_);
lean_dec_ref(v___y_1328_);
lean_dec_ref(v_decls_1323_);
return v_res_1334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2___redArg(lean_object* v_a_1335_, lean_object* v_b_1336_, lean_object* v_x_1337_){
_start:
{
if (lean_obj_tag(v_x_1337_) == 0)
{
lean_dec(v_b_1336_);
lean_dec(v_a_1335_);
return v_x_1337_;
}
else
{
lean_object* v_key_1338_; lean_object* v_value_1339_; lean_object* v_tail_1340_; lean_object* v___x_1342_; uint8_t v_isShared_1343_; uint8_t v_isSharedCheck_1352_; 
v_key_1338_ = lean_ctor_get(v_x_1337_, 0);
v_value_1339_ = lean_ctor_get(v_x_1337_, 1);
v_tail_1340_ = lean_ctor_get(v_x_1337_, 2);
v_isSharedCheck_1352_ = !lean_is_exclusive(v_x_1337_);
if (v_isSharedCheck_1352_ == 0)
{
v___x_1342_ = v_x_1337_;
v_isShared_1343_ = v_isSharedCheck_1352_;
goto v_resetjp_1341_;
}
else
{
lean_inc(v_tail_1340_);
lean_inc(v_value_1339_);
lean_inc(v_key_1338_);
lean_dec(v_x_1337_);
v___x_1342_ = lean_box(0);
v_isShared_1343_ = v_isSharedCheck_1352_;
goto v_resetjp_1341_;
}
v_resetjp_1341_:
{
uint8_t v___x_1344_; 
v___x_1344_ = lean_name_eq(v_key_1338_, v_a_1335_);
if (v___x_1344_ == 0)
{
lean_object* v___x_1345_; lean_object* v___x_1347_; 
v___x_1345_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2___redArg(v_a_1335_, v_b_1336_, v_tail_1340_);
if (v_isShared_1343_ == 0)
{
lean_ctor_set(v___x_1342_, 2, v___x_1345_);
v___x_1347_ = v___x_1342_;
goto v_reusejp_1346_;
}
else
{
lean_object* v_reuseFailAlloc_1348_; 
v_reuseFailAlloc_1348_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1348_, 0, v_key_1338_);
lean_ctor_set(v_reuseFailAlloc_1348_, 1, v_value_1339_);
lean_ctor_set(v_reuseFailAlloc_1348_, 2, v___x_1345_);
v___x_1347_ = v_reuseFailAlloc_1348_;
goto v_reusejp_1346_;
}
v_reusejp_1346_:
{
return v___x_1347_;
}
}
else
{
lean_object* v___x_1350_; 
lean_dec(v_value_1339_);
lean_dec(v_key_1338_);
if (v_isShared_1343_ == 0)
{
lean_ctor_set(v___x_1342_, 1, v_b_1336_);
lean_ctor_set(v___x_1342_, 0, v_a_1335_);
v___x_1350_ = v___x_1342_;
goto v_reusejp_1349_;
}
else
{
lean_object* v_reuseFailAlloc_1351_; 
v_reuseFailAlloc_1351_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1351_, 0, v_a_1335_);
lean_ctor_set(v_reuseFailAlloc_1351_, 1, v_b_1336_);
lean_ctor_set(v_reuseFailAlloc_1351_, 2, v_tail_1340_);
v___x_1350_ = v_reuseFailAlloc_1351_;
goto v_reusejp_1349_;
}
v_reusejp_1349_:
{
return v___x_1350_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3_spec__16___redArg(lean_object* v_x_1353_, lean_object* v_x_1354_){
_start:
{
if (lean_obj_tag(v_x_1354_) == 0)
{
return v_x_1353_;
}
else
{
lean_object* v_key_1355_; lean_object* v_value_1356_; lean_object* v_tail_1357_; lean_object* v___x_1359_; uint8_t v_isShared_1360_; uint8_t v_isSharedCheck_1383_; 
v_key_1355_ = lean_ctor_get(v_x_1354_, 0);
v_value_1356_ = lean_ctor_get(v_x_1354_, 1);
v_tail_1357_ = lean_ctor_get(v_x_1354_, 2);
v_isSharedCheck_1383_ = !lean_is_exclusive(v_x_1354_);
if (v_isSharedCheck_1383_ == 0)
{
v___x_1359_ = v_x_1354_;
v_isShared_1360_ = v_isSharedCheck_1383_;
goto v_resetjp_1358_;
}
else
{
lean_inc(v_tail_1357_);
lean_inc(v_value_1356_);
lean_inc(v_key_1355_);
lean_dec(v_x_1354_);
v___x_1359_ = lean_box(0);
v_isShared_1360_ = v_isSharedCheck_1383_;
goto v_resetjp_1358_;
}
v_resetjp_1358_:
{
lean_object* v___x_1361_; uint64_t v___y_1363_; 
v___x_1361_ = lean_array_get_size(v_x_1353_);
if (lean_obj_tag(v_key_1355_) == 0)
{
uint64_t v___x_1381_; 
v___x_1381_ = 1723ULL;
v___y_1363_ = v___x_1381_;
goto v___jp_1362_;
}
else
{
uint64_t v_hash_1382_; 
v_hash_1382_ = lean_ctor_get_uint64(v_key_1355_, sizeof(void*)*2);
v___y_1363_ = v_hash_1382_;
goto v___jp_1362_;
}
v___jp_1362_:
{
uint64_t v___x_1364_; uint64_t v___x_1365_; uint64_t v_fold_1366_; uint64_t v___x_1367_; uint64_t v___x_1368_; uint64_t v___x_1369_; size_t v___x_1370_; size_t v___x_1371_; size_t v___x_1372_; size_t v___x_1373_; size_t v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1377_; 
v___x_1364_ = 32ULL;
v___x_1365_ = lean_uint64_shift_right(v___y_1363_, v___x_1364_);
v_fold_1366_ = lean_uint64_xor(v___y_1363_, v___x_1365_);
v___x_1367_ = 16ULL;
v___x_1368_ = lean_uint64_shift_right(v_fold_1366_, v___x_1367_);
v___x_1369_ = lean_uint64_xor(v_fold_1366_, v___x_1368_);
v___x_1370_ = lean_uint64_to_usize(v___x_1369_);
v___x_1371_ = lean_usize_of_nat(v___x_1361_);
v___x_1372_ = ((size_t)1ULL);
v___x_1373_ = lean_usize_sub(v___x_1371_, v___x_1372_);
v___x_1374_ = lean_usize_land(v___x_1370_, v___x_1373_);
v___x_1375_ = lean_array_uget_borrowed(v_x_1353_, v___x_1374_);
lean_inc(v___x_1375_);
if (v_isShared_1360_ == 0)
{
lean_ctor_set(v___x_1359_, 2, v___x_1375_);
v___x_1377_ = v___x_1359_;
goto v_reusejp_1376_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_key_1355_);
lean_ctor_set(v_reuseFailAlloc_1380_, 1, v_value_1356_);
lean_ctor_set(v_reuseFailAlloc_1380_, 2, v___x_1375_);
v___x_1377_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1376_;
}
v_reusejp_1376_:
{
lean_object* v___x_1378_; 
v___x_1378_ = lean_array_uset(v_x_1353_, v___x_1374_, v___x_1377_);
v_x_1353_ = v___x_1378_;
v_x_1354_ = v_tail_1357_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3___redArg(lean_object* v_i_1384_, lean_object* v_source_1385_, lean_object* v_target_1386_){
_start:
{
lean_object* v___x_1387_; uint8_t v___x_1388_; 
v___x_1387_ = lean_array_get_size(v_source_1385_);
v___x_1388_ = lean_nat_dec_lt(v_i_1384_, v___x_1387_);
if (v___x_1388_ == 0)
{
lean_dec_ref(v_source_1385_);
lean_dec(v_i_1384_);
return v_target_1386_;
}
else
{
lean_object* v_es_1389_; lean_object* v___x_1390_; lean_object* v_source_1391_; lean_object* v_target_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; 
v_es_1389_ = lean_array_fget(v_source_1385_, v_i_1384_);
v___x_1390_ = lean_box(0);
v_source_1391_ = lean_array_fset(v_source_1385_, v_i_1384_, v___x_1390_);
v_target_1392_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3_spec__16___redArg(v_target_1386_, v_es_1389_);
v___x_1393_ = lean_unsigned_to_nat(1u);
v___x_1394_ = lean_nat_add(v_i_1384_, v___x_1393_);
lean_dec(v_i_1384_);
v_i_1384_ = v___x_1394_;
v_source_1385_ = v_source_1391_;
v_target_1386_ = v_target_1392_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1___redArg(lean_object* v_data_1396_){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v_nbuckets_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; 
v___x_1397_ = lean_array_get_size(v_data_1396_);
v___x_1398_ = lean_unsigned_to_nat(2u);
v_nbuckets_1399_ = lean_nat_mul(v___x_1397_, v___x_1398_);
v___x_1400_ = lean_unsigned_to_nat(0u);
v___x_1401_ = lean_box(0);
v___x_1402_ = lean_mk_array(v_nbuckets_1399_, v___x_1401_);
v___x_1403_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3___redArg(v___x_1400_, v_data_1396_, v___x_1402_);
return v___x_1403_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg(lean_object* v_a_1404_, lean_object* v_x_1405_){
_start:
{
if (lean_obj_tag(v_x_1405_) == 0)
{
uint8_t v___x_1406_; 
v___x_1406_ = 0;
return v___x_1406_;
}
else
{
lean_object* v_key_1407_; lean_object* v_tail_1408_; uint8_t v___x_1409_; 
v_key_1407_ = lean_ctor_get(v_x_1405_, 0);
v_tail_1408_ = lean_ctor_get(v_x_1405_, 2);
v___x_1409_ = lean_name_eq(v_key_1407_, v_a_1404_);
if (v___x_1409_ == 0)
{
v_x_1405_ = v_tail_1408_;
goto _start;
}
else
{
return v___x_1409_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg___boxed(lean_object* v_a_1411_, lean_object* v_x_1412_){
_start:
{
uint8_t v_res_1413_; lean_object* v_r_1414_; 
v_res_1413_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg(v_a_1411_, v_x_1412_);
lean_dec(v_x_1412_);
lean_dec(v_a_1411_);
v_r_1414_ = lean_box(v_res_1413_);
return v_r_1414_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(lean_object* v_m_1415_, lean_object* v_a_1416_, lean_object* v_b_1417_){
_start:
{
lean_object* v_size_1418_; lean_object* v_buckets_1419_; lean_object* v___x_1421_; uint8_t v_isShared_1422_; uint8_t v_isSharedCheck_1465_; 
v_size_1418_ = lean_ctor_get(v_m_1415_, 0);
v_buckets_1419_ = lean_ctor_get(v_m_1415_, 1);
v_isSharedCheck_1465_ = !lean_is_exclusive(v_m_1415_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1421_ = v_m_1415_;
v_isShared_1422_ = v_isSharedCheck_1465_;
goto v_resetjp_1420_;
}
else
{
lean_inc(v_buckets_1419_);
lean_inc(v_size_1418_);
lean_dec(v_m_1415_);
v___x_1421_ = lean_box(0);
v_isShared_1422_ = v_isSharedCheck_1465_;
goto v_resetjp_1420_;
}
v_resetjp_1420_:
{
lean_object* v___x_1423_; uint64_t v___y_1425_; 
v___x_1423_ = lean_array_get_size(v_buckets_1419_);
if (lean_obj_tag(v_a_1416_) == 0)
{
uint64_t v___x_1463_; 
v___x_1463_ = 1723ULL;
v___y_1425_ = v___x_1463_;
goto v___jp_1424_;
}
else
{
uint64_t v_hash_1464_; 
v_hash_1464_ = lean_ctor_get_uint64(v_a_1416_, sizeof(void*)*2);
v___y_1425_ = v_hash_1464_;
goto v___jp_1424_;
}
v___jp_1424_:
{
uint64_t v___x_1426_; uint64_t v___x_1427_; uint64_t v_fold_1428_; uint64_t v___x_1429_; uint64_t v___x_1430_; uint64_t v___x_1431_; size_t v___x_1432_; size_t v___x_1433_; size_t v___x_1434_; size_t v___x_1435_; size_t v___x_1436_; lean_object* v_bkt_1437_; uint8_t v___x_1438_; 
v___x_1426_ = 32ULL;
v___x_1427_ = lean_uint64_shift_right(v___y_1425_, v___x_1426_);
v_fold_1428_ = lean_uint64_xor(v___y_1425_, v___x_1427_);
v___x_1429_ = 16ULL;
v___x_1430_ = lean_uint64_shift_right(v_fold_1428_, v___x_1429_);
v___x_1431_ = lean_uint64_xor(v_fold_1428_, v___x_1430_);
v___x_1432_ = lean_uint64_to_usize(v___x_1431_);
v___x_1433_ = lean_usize_of_nat(v___x_1423_);
v___x_1434_ = ((size_t)1ULL);
v___x_1435_ = lean_usize_sub(v___x_1433_, v___x_1434_);
v___x_1436_ = lean_usize_land(v___x_1432_, v___x_1435_);
v_bkt_1437_ = lean_array_uget_borrowed(v_buckets_1419_, v___x_1436_);
v___x_1438_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg(v_a_1416_, v_bkt_1437_);
if (v___x_1438_ == 0)
{
lean_object* v___x_1439_; lean_object* v_size_x27_1440_; lean_object* v___x_1441_; lean_object* v_buckets_x27_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; uint8_t v___x_1448_; 
v___x_1439_ = lean_unsigned_to_nat(1u);
v_size_x27_1440_ = lean_nat_add(v_size_1418_, v___x_1439_);
lean_dec(v_size_1418_);
lean_inc(v_bkt_1437_);
v___x_1441_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1441_, 0, v_a_1416_);
lean_ctor_set(v___x_1441_, 1, v_b_1417_);
lean_ctor_set(v___x_1441_, 2, v_bkt_1437_);
v_buckets_x27_1442_ = lean_array_uset(v_buckets_1419_, v___x_1436_, v___x_1441_);
v___x_1443_ = lean_unsigned_to_nat(4u);
v___x_1444_ = lean_nat_mul(v_size_x27_1440_, v___x_1443_);
v___x_1445_ = lean_unsigned_to_nat(3u);
v___x_1446_ = lean_nat_div(v___x_1444_, v___x_1445_);
lean_dec(v___x_1444_);
v___x_1447_ = lean_array_get_size(v_buckets_x27_1442_);
v___x_1448_ = lean_nat_dec_le(v___x_1446_, v___x_1447_);
lean_dec(v___x_1446_);
if (v___x_1448_ == 0)
{
lean_object* v_val_1449_; lean_object* v___x_1451_; 
v_val_1449_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1___redArg(v_buckets_x27_1442_);
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 1, v_val_1449_);
lean_ctor_set(v___x_1421_, 0, v_size_x27_1440_);
v___x_1451_ = v___x_1421_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_size_x27_1440_);
lean_ctor_set(v_reuseFailAlloc_1452_, 1, v_val_1449_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
else
{
lean_object* v___x_1454_; 
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 1, v_buckets_x27_1442_);
lean_ctor_set(v___x_1421_, 0, v_size_x27_1440_);
v___x_1454_ = v___x_1421_;
goto v_reusejp_1453_;
}
else
{
lean_object* v_reuseFailAlloc_1455_; 
v_reuseFailAlloc_1455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1455_, 0, v_size_x27_1440_);
lean_ctor_set(v_reuseFailAlloc_1455_, 1, v_buckets_x27_1442_);
v___x_1454_ = v_reuseFailAlloc_1455_;
goto v_reusejp_1453_;
}
v_reusejp_1453_:
{
return v___x_1454_;
}
}
}
else
{
lean_object* v___x_1456_; lean_object* v_buckets_x27_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1461_; 
lean_inc(v_bkt_1437_);
v___x_1456_ = lean_box(0);
v_buckets_x27_1457_ = lean_array_uset(v_buckets_1419_, v___x_1436_, v___x_1456_);
v___x_1458_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2___redArg(v_a_1416_, v_b_1417_, v_bkt_1437_);
v___x_1459_ = lean_array_uset(v_buckets_x27_1457_, v___x_1436_, v___x_1458_);
if (v_isShared_1422_ == 0)
{
lean_ctor_set(v___x_1421_, 1, v___x_1459_);
v___x_1461_ = v___x_1421_;
goto v_reusejp_1460_;
}
else
{
lean_object* v_reuseFailAlloc_1462_; 
v_reuseFailAlloc_1462_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1462_, 0, v_size_1418_);
lean_ctor_set(v_reuseFailAlloc_1462_, 1, v___x_1459_);
v___x_1461_ = v_reuseFailAlloc_1462_;
goto v_reusejp_1460_;
}
v_reusejp_1460_:
{
return v___x_1461_;
}
}
}
}
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_1467_; lean_object* v___x_1468_; 
v___x_1467_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__0));
v___x_1468_ = l_Lean_stringToMessageData(v___x_1467_);
return v___x_1468_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg(lean_object* v_as_1469_, size_t v_sz_1470_, size_t v_i_1471_, lean_object* v_b_1472_){
_start:
{
lean_object* v_a_1475_; uint8_t v___x_1479_; 
v___x_1479_ = lean_usize_dec_lt(v_i_1471_, v_sz_1470_);
if (v___x_1479_ == 0)
{
lean_object* v___x_1480_; 
v___x_1480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1480_, 0, v_b_1472_);
return v___x_1480_;
}
else
{
lean_object* v_a_1481_; lean_object* v_fst_1482_; lean_object* v_snd_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1499_; 
v_a_1481_ = lean_array_uget(v_as_1469_, v_i_1471_);
v_fst_1482_ = lean_ctor_get(v_a_1481_, 0);
v_snd_1483_ = lean_ctor_get(v_a_1481_, 1);
v_isSharedCheck_1499_ = !lean_is_exclusive(v_a_1481_);
if (v_isSharedCheck_1499_ == 0)
{
v___x_1485_ = v_a_1481_;
v_isShared_1486_ = v_isSharedCheck_1499_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_snd_1483_);
lean_inc(v_fst_1482_);
lean_dec(v_a_1481_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1499_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v_val_1488_; lean_object* v___x_1490_; 
v___x_1490_ = lean_task_get_own(v_snd_1483_);
if (lean_obj_tag(v___x_1490_) == 0)
{
lean_object* v_a_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1495_; 
v_a_1491_ = lean_ctor_get(v___x_1490_, 0);
lean_inc(v_a_1491_);
lean_dec_ref_known(v___x_1490_, 1);
v___x_1492_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___closed__1);
v___x_1493_ = l_Lean_Exception_toMessageData(v_a_1491_);
if (v_isShared_1486_ == 0)
{
lean_ctor_set_tag(v___x_1485_, 7);
lean_ctor_set(v___x_1485_, 1, v___x_1493_);
lean_ctor_set(v___x_1485_, 0, v___x_1492_);
v___x_1495_ = v___x_1485_;
goto v_reusejp_1494_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v___x_1492_);
lean_ctor_set(v_reuseFailAlloc_1496_, 1, v___x_1493_);
v___x_1495_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1494_;
}
v_reusejp_1494_:
{
v_val_1488_ = v___x_1495_;
goto v___jp_1487_;
}
}
else
{
lean_object* v_a_1497_; 
lean_del_object(v___x_1485_);
v_a_1497_ = lean_ctor_get(v___x_1490_, 0);
lean_inc(v_a_1497_);
lean_dec_ref_known(v___x_1490_, 1);
if (lean_obj_tag(v_a_1497_) == 1)
{
lean_object* v_val_1498_; 
v_val_1498_ = lean_ctor_get(v_a_1497_, 0);
lean_inc(v_val_1498_);
lean_dec_ref_known(v_a_1497_, 1);
v_val_1488_ = v_val_1498_;
goto v___jp_1487_;
}
else
{
lean_dec(v_a_1497_);
lean_dec(v_fst_1482_);
v_a_1475_ = v_b_1472_;
goto v___jp_1474_;
}
}
v___jp_1487_:
{
lean_object* v___x_1489_; 
v___x_1489_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(v_b_1472_, v_fst_1482_, v_val_1488_);
v_a_1475_ = v___x_1489_;
goto v___jp_1474_;
}
}
}
v___jp_1474_:
{
size_t v___x_1476_; size_t v___x_1477_; 
v___x_1476_ = ((size_t)1ULL);
v___x_1477_ = lean_usize_add(v_i_1471_, v___x_1476_);
v_i_1471_ = v___x_1477_;
v_b_1472_ = v_a_1475_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg___boxed(lean_object* v_as_1500_, lean_object* v_sz_1501_, lean_object* v_i_1502_, lean_object* v_b_1503_, lean_object* v___y_1504_){
_start:
{
size_t v_sz_boxed_1505_; size_t v_i_boxed_1506_; lean_object* v_res_1507_; 
v_sz_boxed_1505_ = lean_unbox_usize(v_sz_1501_);
lean_dec(v_sz_1501_);
v_i_boxed_1506_ = lean_unbox_usize(v_i_1502_);
lean_dec(v_i_1502_);
v_res_1507_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg(v_as_1500_, v_sz_boxed_1505_, v_i_boxed_1506_, v_b_1503_);
lean_dec_ref(v_as_1500_);
return v_res_1507_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6(void){
_start:
{
lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1514_ = lean_box(0);
v___x_1515_ = lean_unsigned_to_nat(16u);
v___x_1516_ = lean_mk_array(v___x_1515_, v___x_1514_);
return v___x_1516_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7(void){
_start:
{
lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; 
v___x_1517_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6);
v___x_1518_ = lean_unsigned_to_nat(0u);
v___x_1519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1519_, 0, v___x_1518_);
lean_ctor_set(v___x_1519_, 1, v___x_1517_);
return v___x_1519_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10(uint8_t v___y_1521_, lean_object* v_currentModule_1522_, size_t v_sz_1523_, size_t v_i_1524_, lean_object* v_bs_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_){
_start:
{
uint8_t v___x_1529_; 
v___x_1529_ = lean_usize_dec_lt(v_i_1524_, v_sz_1523_);
if (v___x_1529_ == 0)
{
lean_object* v___x_1530_; 
lean_dec(v_currentModule_1522_);
v___x_1530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1530_, 0, v_bs_1525_);
return v___x_1530_;
}
else
{
lean_object* v_v_1531_; lean_object* v_fst_1532_; lean_object* v_snd_1533_; lean_object* v___x_1535_; uint8_t v_isShared_1536_; uint8_t v_isSharedCheck_1771_; 
v_v_1531_ = lean_array_uget(v_bs_1525_, v_i_1524_);
v_fst_1532_ = lean_ctor_get(v_v_1531_, 0);
v_snd_1533_ = lean_ctor_get(v_v_1531_, 1);
v_isSharedCheck_1771_ = !lean_is_exclusive(v_v_1531_);
if (v_isSharedCheck_1771_ == 0)
{
v___x_1535_ = v_v_1531_;
v_isShared_1536_ = v_isSharedCheck_1771_;
goto v_resetjp_1534_;
}
else
{
lean_inc(v_snd_1533_);
lean_inc(v_fst_1532_);
lean_dec(v_v_1531_);
v___x_1535_ = lean_box(0);
v_isShared_1536_ = v_isSharedCheck_1771_;
goto v_resetjp_1534_;
}
v_resetjp_1534_:
{
lean_object* v___x_1537_; lean_object* v_bs_x27_1538_; lean_object* v___y_1540_; lean_object* v___y_1549_; lean_object* v___y_1550_; lean_object* v___y_1551_; lean_object* v___y_1552_; lean_object* v___y_1553_; lean_object* v___y_1568_; lean_object* v___y_1569_; lean_object* v___y_1570_; uint8_t v___y_1571_; lean_object* v___y_1572_; lean_object* v___y_1592_; lean_object* v___y_1593_; lean_object* v___y_1594_; lean_object* v___y_1595_; lean_object* v___y_1613_; uint8_t v___y_1614_; lean_object* v___y_1615_; lean_object* v___y_1616_; 
v___x_1537_ = lean_unsigned_to_nat(0u);
v_bs_x27_1538_ = lean_array_uset(v_bs_1525_, v_i_1524_, v___x_1537_);
if (v___y_1521_ == 0)
{
lean_object* v_options_1689_; uint8_t v_hasTrace_1690_; 
v_options_1689_ = lean_ctor_get(v___y_1526_, 2);
v_hasTrace_1690_ = lean_ctor_get_uint8(v_options_1689_, sizeof(void*)*1);
if (v_hasTrace_1690_ == 0)
{
goto v___jp_1637_;
}
else
{
lean_object* v_inheritedTraceOptions_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; uint8_t v___x_1694_; lean_object* v___y_1696_; 
v_inheritedTraceOptions_1691_ = lean_ctor_get(v___y_1526_, 13);
v___x_1692_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_1693_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5);
v___x_1694_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1691_, v_options_1689_, v___x_1693_);
if (v___x_1694_ == 0)
{
goto v___jp_1637_;
}
else
{
if (lean_obj_tag(v_currentModule_1522_) == 1)
{
lean_object* v_val_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; 
v_val_1717_ = lean_ctor_get(v_currentModule_1522_, 0);
v___x_1718_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
lean_inc(v_val_1717_);
v___x_1719_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1717_, v___x_1694_);
v___x_1720_ = lean_string_append(v___x_1718_, v___x_1719_);
lean_dec_ref(v___x_1719_);
v___x_1721_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1722_ = lean_string_append(v___x_1720_, v___x_1721_);
v___y_1696_ = v___x_1722_;
goto v___jp_1695_;
}
else
{
lean_object* v___x_1723_; 
v___x_1723_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1696_ = v___x_1723_;
goto v___jp_1695_;
}
}
v___jp_1695_:
{
lean_object* v_name_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; 
v_name_1697_ = lean_ctor_get(v_fst_1532_, 1);
v___x_1698_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6));
lean_inc(v_name_1697_);
v___x_1699_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1697_, v___x_1694_);
v___x_1700_ = lean_string_append(v___x_1698_, v___x_1699_);
lean_dec_ref(v___x_1699_);
v___x_1701_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7));
v___x_1702_ = lean_string_append(v___x_1700_, v___x_1701_);
v___x_1703_ = lean_string_append(v___y_1696_, v___x_1702_);
lean_dec_ref(v___x_1702_);
v___x_1704_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__8));
v___x_1705_ = lean_string_append(v___x_1703_, v___x_1704_);
v___x_1706_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1706_, 0, v___x_1705_);
v___x_1707_ = l_Lean_MessageData_ofFormat(v___x_1706_);
v___x_1708_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(v___x_1692_, v___x_1707_, v___y_1526_, v___y_1527_);
if (lean_obj_tag(v___x_1708_) == 0)
{
lean_dec_ref_known(v___x_1708_, 1);
goto v___jp_1637_;
}
else
{
lean_object* v_a_1709_; lean_object* v___x_1711_; uint8_t v_isShared_1712_; uint8_t v_isSharedCheck_1716_; 
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_snd_1533_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1709_ = lean_ctor_get(v___x_1708_, 0);
v_isSharedCheck_1716_ = !lean_is_exclusive(v___x_1708_);
if (v_isSharedCheck_1716_ == 0)
{
v___x_1711_ = v___x_1708_;
v_isShared_1712_ = v_isSharedCheck_1716_;
goto v_resetjp_1710_;
}
else
{
lean_inc(v_a_1709_);
lean_dec(v___x_1708_);
v___x_1711_ = lean_box(0);
v_isShared_1712_ = v_isSharedCheck_1716_;
goto v_resetjp_1710_;
}
v_resetjp_1710_:
{
lean_object* v___x_1714_; 
if (v_isShared_1712_ == 0)
{
v___x_1714_ = v___x_1711_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1715_; 
v_reuseFailAlloc_1715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1715_, 0, v_a_1709_);
v___x_1714_ = v_reuseFailAlloc_1715_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
return v___x_1714_;
}
}
}
}
}
}
else
{
lean_object* v___x_1724_; uint8_t v___x_1725_; lean_object* v___x_1726_; 
v___x_1724_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11));
v___x_1725_ = 0;
v___x_1726_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v___x_1724_, v___x_1725_, v___y_1526_);
if (lean_obj_tag(v___x_1726_) == 0)
{
lean_object* v_a_1727_; lean_object* v___y_1729_; uint8_t v___x_1754_; 
v_a_1727_ = lean_ctor_get(v___x_1726_, 0);
lean_inc(v_a_1727_);
lean_dec_ref_known(v___x_1726_, 1);
v___x_1754_ = lean_unbox(v_a_1727_);
if (v___x_1754_ == 0)
{
lean_dec(v_a_1727_);
goto v___jp_1637_;
}
else
{
if (lean_obj_tag(v_currentModule_1522_) == 1)
{
lean_object* v_val_1755_; lean_object* v___x_1756_; uint8_t v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; 
v_val_1755_ = lean_ctor_get(v_currentModule_1522_, 0);
v___x_1756_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
v___x_1757_ = lean_unbox(v_a_1727_);
lean_inc(v_val_1755_);
v___x_1758_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1755_, v___x_1757_);
v___x_1759_ = lean_string_append(v___x_1756_, v___x_1758_);
lean_dec_ref(v___x_1758_);
v___x_1760_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1761_ = lean_string_append(v___x_1759_, v___x_1760_);
v___y_1729_ = v___x_1761_;
goto v___jp_1728_;
}
else
{
lean_object* v___x_1762_; 
v___x_1762_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1729_ = v___x_1762_;
goto v___jp_1728_;
}
}
v___jp_1728_:
{
lean_object* v_name_1730_; lean_object* v___x_1731_; uint8_t v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; 
v_name_1730_ = lean_ctor_get(v_fst_1532_, 1);
v___x_1731_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6));
v___x_1732_ = lean_unbox(v_a_1727_);
lean_dec(v_a_1727_);
lean_inc(v_name_1730_);
v___x_1733_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1730_, v___x_1732_);
v___x_1734_ = lean_string_append(v___x_1731_, v___x_1733_);
lean_dec_ref(v___x_1733_);
v___x_1735_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7));
v___x_1736_ = lean_string_append(v___x_1734_, v___x_1735_);
v___x_1737_ = lean_string_append(v___y_1729_, v___x_1736_);
lean_dec_ref(v___x_1736_);
v___x_1738_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__8));
v___x_1739_ = lean_string_append(v___x_1737_, v___x_1738_);
v___x_1740_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_1739_);
if (lean_obj_tag(v___x_1740_) == 0)
{
lean_dec_ref_known(v___x_1740_, 1);
goto v___jp_1637_;
}
else
{
lean_object* v_a_1741_; lean_object* v___x_1743_; uint8_t v_isShared_1744_; uint8_t v_isSharedCheck_1753_; 
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_snd_1533_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1741_ = lean_ctor_get(v___x_1740_, 0);
v_isSharedCheck_1753_ = !lean_is_exclusive(v___x_1740_);
if (v_isSharedCheck_1753_ == 0)
{
v___x_1743_ = v___x_1740_;
v_isShared_1744_ = v_isSharedCheck_1753_;
goto v_resetjp_1742_;
}
else
{
lean_inc(v_a_1741_);
lean_dec(v___x_1740_);
v___x_1743_ = lean_box(0);
v_isShared_1744_ = v_isSharedCheck_1753_;
goto v_resetjp_1742_;
}
v_resetjp_1742_:
{
lean_object* v_ref_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1751_; 
v_ref_1745_ = lean_ctor_get(v___y_1526_, 5);
v___x_1746_ = lean_io_error_to_string(v_a_1741_);
v___x_1747_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1747_, 0, v___x_1746_);
v___x_1748_ = l_Lean_MessageData_ofFormat(v___x_1747_);
lean_inc(v_ref_1745_);
v___x_1749_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1749_, 0, v_ref_1745_);
lean_ctor_set(v___x_1749_, 1, v___x_1748_);
if (v_isShared_1744_ == 0)
{
lean_ctor_set(v___x_1743_, 0, v___x_1749_);
v___x_1751_ = v___x_1743_;
goto v_reusejp_1750_;
}
else
{
lean_object* v_reuseFailAlloc_1752_; 
v_reuseFailAlloc_1752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1752_, 0, v___x_1749_);
v___x_1751_ = v_reuseFailAlloc_1752_;
goto v_reusejp_1750_;
}
v_reusejp_1750_:
{
return v___x_1751_;
}
}
}
}
}
else
{
lean_object* v_a_1763_; lean_object* v___x_1765_; uint8_t v_isShared_1766_; uint8_t v_isSharedCheck_1770_; 
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_snd_1533_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1763_ = lean_ctor_get(v___x_1726_, 0);
v_isSharedCheck_1770_ = !lean_is_exclusive(v___x_1726_);
if (v_isSharedCheck_1770_ == 0)
{
v___x_1765_ = v___x_1726_;
v_isShared_1766_ = v_isSharedCheck_1770_;
goto v_resetjp_1764_;
}
else
{
lean_inc(v_a_1763_);
lean_dec(v___x_1726_);
v___x_1765_ = lean_box(0);
v_isShared_1766_ = v_isSharedCheck_1770_;
goto v_resetjp_1764_;
}
v_resetjp_1764_:
{
lean_object* v___x_1768_; 
if (v_isShared_1766_ == 0)
{
v___x_1768_ = v___x_1765_;
goto v_reusejp_1767_;
}
else
{
lean_object* v_reuseFailAlloc_1769_; 
v_reuseFailAlloc_1769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1769_, 0, v_a_1763_);
v___x_1768_ = v_reuseFailAlloc_1769_;
goto v_reusejp_1767_;
}
v_reusejp_1767_:
{
return v___x_1768_;
}
}
}
}
v___jp_1539_:
{
lean_object* v___x_1542_; 
if (v_isShared_1536_ == 0)
{
lean_ctor_set(v___x_1535_, 1, v___y_1540_);
v___x_1542_ = v___x_1535_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1547_; 
v_reuseFailAlloc_1547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1547_, 0, v_fst_1532_);
lean_ctor_set(v_reuseFailAlloc_1547_, 1, v___y_1540_);
v___x_1542_ = v_reuseFailAlloc_1547_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
size_t v___x_1543_; size_t v___x_1544_; lean_object* v___x_1545_; 
v___x_1543_ = ((size_t)1ULL);
v___x_1544_ = lean_usize_add(v_i_1524_, v___x_1543_);
v___x_1545_ = lean_array_uset(v_bs_x27_1538_, v_i_1524_, v___x_1542_);
v_i_1524_ = v___x_1544_;
v_bs_1525_ = v___x_1545_;
goto _start;
}
}
v___jp_1548_:
{
lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; 
lean_inc_ref(v___y_1551_);
v___x_1554_ = lean_string_append(v___y_1551_, v___y_1553_);
lean_dec_ref(v___y_1553_);
v___x_1555_ = lean_string_append(v___y_1552_, v___x_1554_);
lean_dec_ref(v___x_1554_);
v___x_1556_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1556_, 0, v___x_1555_);
v___x_1557_ = l_Lean_MessageData_ofFormat(v___x_1556_);
lean_inc(v___y_1550_);
v___x_1558_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(v___y_1550_, v___x_1557_, v___y_1526_, v___y_1527_);
if (lean_obj_tag(v___x_1558_) == 0)
{
lean_dec_ref_known(v___x_1558_, 1);
v___y_1540_ = v___y_1549_;
goto v___jp_1539_;
}
else
{
lean_object* v_a_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1566_; 
lean_dec_ref(v___y_1549_);
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1559_ = lean_ctor_get(v___x_1558_, 0);
v_isSharedCheck_1566_ = !lean_is_exclusive(v___x_1558_);
if (v_isSharedCheck_1566_ == 0)
{
v___x_1561_ = v___x_1558_;
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_a_1559_);
lean_dec(v___x_1558_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
lean_object* v___x_1564_; 
if (v_isShared_1562_ == 0)
{
v___x_1564_ = v___x_1561_;
goto v_reusejp_1563_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v_a_1559_);
v___x_1564_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1563_;
}
v_reusejp_1563_:
{
return v___x_1564_;
}
}
}
}
v___jp_1567_:
{
lean_object* v_name_1573_; lean_object* v_size_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; uint8_t v___x_1582_; 
v_name_1573_ = lean_ctor_get(v_fst_1532_, 1);
v_size_1574_ = lean_ctor_get(v___y_1568_, 0);
v___x_1575_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6));
lean_inc(v_name_1573_);
v___x_1576_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1573_, v___y_1571_);
v___x_1577_ = lean_string_append(v___x_1575_, v___x_1576_);
lean_dec_ref(v___x_1576_);
v___x_1578_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7));
v___x_1579_ = lean_string_append(v___x_1577_, v___x_1578_);
v___x_1580_ = lean_string_append(v___y_1572_, v___x_1579_);
lean_dec_ref(v___x_1579_);
v___x_1581_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__0));
v___x_1582_ = lean_nat_dec_eq(v_size_1574_, v___y_1570_);
lean_dec(v___y_1570_);
if (v___x_1582_ == 0)
{
lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; 
v___x_1583_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__1));
lean_inc(v_size_1574_);
v___x_1584_ = l_Nat_reprFast(v_size_1574_);
v___x_1585_ = lean_string_append(v___x_1583_, v___x_1584_);
lean_dec_ref(v___x_1584_);
v___x_1586_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__2));
v___x_1587_ = lean_string_append(v___x_1585_, v___x_1586_);
v___x_1588_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__3));
v___x_1589_ = lean_string_append(v___x_1587_, v___x_1588_);
v___y_1549_ = v___y_1568_;
v___y_1550_ = v___y_1569_;
v___y_1551_ = v___x_1581_;
v___y_1552_ = v___x_1580_;
v___y_1553_ = v___x_1589_;
goto v___jp_1548_;
}
else
{
lean_object* v___x_1590_; 
v___x_1590_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__4));
v___y_1549_ = v___y_1568_;
v___y_1550_ = v___y_1569_;
v___y_1551_ = v___x_1581_;
v___y_1552_ = v___x_1580_;
v___y_1553_ = v___x_1590_;
goto v___jp_1548_;
}
}
v___jp_1591_:
{
lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; 
lean_inc_ref(v___y_1594_);
v___x_1596_ = lean_string_append(v___y_1594_, v___y_1595_);
lean_dec_ref(v___y_1595_);
v___x_1597_ = lean_string_append(v___y_1593_, v___x_1596_);
lean_dec_ref(v___x_1596_);
v___x_1598_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_1597_);
if (lean_obj_tag(v___x_1598_) == 0)
{
lean_dec_ref_known(v___x_1598_, 1);
v___y_1540_ = v___y_1592_;
goto v___jp_1539_;
}
else
{
lean_object* v_a_1599_; lean_object* v___x_1601_; uint8_t v_isShared_1602_; uint8_t v_isSharedCheck_1611_; 
lean_dec_ref(v___y_1592_);
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1599_ = lean_ctor_get(v___x_1598_, 0);
v_isSharedCheck_1611_ = !lean_is_exclusive(v___x_1598_);
if (v_isSharedCheck_1611_ == 0)
{
v___x_1601_ = v___x_1598_;
v_isShared_1602_ = v_isSharedCheck_1611_;
goto v_resetjp_1600_;
}
else
{
lean_inc(v_a_1599_);
lean_dec(v___x_1598_);
v___x_1601_ = lean_box(0);
v_isShared_1602_ = v_isSharedCheck_1611_;
goto v_resetjp_1600_;
}
v_resetjp_1600_:
{
lean_object* v_ref_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1609_; 
v_ref_1603_ = lean_ctor_get(v___y_1526_, 5);
v___x_1604_ = lean_io_error_to_string(v_a_1599_);
v___x_1605_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1605_, 0, v___x_1604_);
v___x_1606_ = l_Lean_MessageData_ofFormat(v___x_1605_);
lean_inc(v_ref_1603_);
v___x_1607_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1607_, 0, v_ref_1603_);
lean_ctor_set(v___x_1607_, 1, v___x_1606_);
if (v_isShared_1602_ == 0)
{
lean_ctor_set(v___x_1601_, 0, v___x_1607_);
v___x_1609_ = v___x_1601_;
goto v_reusejp_1608_;
}
else
{
lean_object* v_reuseFailAlloc_1610_; 
v_reuseFailAlloc_1610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1610_, 0, v___x_1607_);
v___x_1609_ = v_reuseFailAlloc_1610_;
goto v_reusejp_1608_;
}
v_reusejp_1608_:
{
return v___x_1609_;
}
}
}
}
v___jp_1612_:
{
lean_object* v_name_1617_; lean_object* v_size_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; uint8_t v___x_1626_; 
v_name_1617_ = lean_ctor_get(v_fst_1532_, 1);
v_size_1618_ = lean_ctor_get(v___y_1613_, 0);
v___x_1619_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__6));
lean_inc(v_name_1617_);
v___x_1620_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1617_, v___y_1614_);
v___x_1621_ = lean_string_append(v___x_1619_, v___x_1620_);
lean_dec_ref(v___x_1620_);
v___x_1622_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__7));
v___x_1623_ = lean_string_append(v___x_1621_, v___x_1622_);
v___x_1624_ = lean_string_append(v___y_1616_, v___x_1623_);
lean_dec_ref(v___x_1623_);
v___x_1625_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__0));
v___x_1626_ = lean_nat_dec_eq(v_size_1618_, v___y_1615_);
lean_dec(v___y_1615_);
if (v___x_1626_ == 0)
{
lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; 
v___x_1627_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__1));
lean_inc(v_size_1618_);
v___x_1628_ = l_Nat_reprFast(v_size_1618_);
v___x_1629_ = lean_string_append(v___x_1627_, v___x_1628_);
lean_dec_ref(v___x_1628_);
v___x_1630_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__2));
v___x_1631_ = lean_string_append(v___x_1629_, v___x_1630_);
v___x_1632_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__5));
v___x_1633_ = lean_string_append(v___x_1631_, v___x_1632_);
v___x_1634_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__3));
v___x_1635_ = lean_string_append(v___x_1633_, v___x_1634_);
v___y_1592_ = v___y_1613_;
v___y_1593_ = v___x_1624_;
v___y_1594_ = v___x_1625_;
v___y_1595_ = v___x_1635_;
goto v___jp_1591_;
}
else
{
lean_object* v___x_1636_; 
v___x_1636_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__4));
v___y_1592_ = v___y_1613_;
v___y_1593_ = v___x_1624_;
v___y_1594_ = v___x_1625_;
v___y_1595_ = v___x_1636_;
goto v___jp_1591_;
}
}
v___jp_1637_:
{
lean_object* v___x_1638_; size_t v_sz_1639_; size_t v___x_1640_; lean_object* v___x_1641_; 
v___x_1638_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7);
v_sz_1639_ = lean_array_size(v_snd_1533_);
v___x_1640_ = ((size_t)0ULL);
v___x_1641_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg(v_snd_1533_, v_sz_1639_, v___x_1640_, v___x_1638_);
lean_dec(v_snd_1533_);
if (lean_obj_tag(v___x_1641_) == 0)
{
if (v___y_1521_ == 0)
{
lean_object* v_options_1642_; uint8_t v_hasTrace_1643_; 
v_options_1642_ = lean_ctor_get(v___y_1526_, 2);
v_hasTrace_1643_ = lean_ctor_get_uint8(v_options_1642_, sizeof(void*)*1);
if (v_hasTrace_1643_ == 0)
{
lean_object* v_a_1644_; 
v_a_1644_ = lean_ctor_get(v___x_1641_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1641_, 1);
v___y_1540_ = v_a_1644_;
goto v___jp_1539_;
}
else
{
lean_object* v_a_1645_; lean_object* v_inheritedTraceOptions_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; uint8_t v___x_1649_; 
v_a_1645_ = lean_ctor_get(v___x_1641_, 0);
lean_inc(v_a_1645_);
lean_dec_ref_known(v___x_1641_, 1);
v_inheritedTraceOptions_1646_ = lean_ctor_get(v___y_1526_, 13);
v___x_1647_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_1648_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5);
v___x_1649_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1646_, v_options_1642_, v___x_1648_);
if (v___x_1649_ == 0)
{
v___y_1540_ = v_a_1645_;
goto v___jp_1539_;
}
else
{
if (lean_obj_tag(v_currentModule_1522_) == 1)
{
lean_object* v_val_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; 
v_val_1650_ = lean_ctor_get(v_currentModule_1522_, 0);
v___x_1651_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
lean_inc(v_val_1650_);
v___x_1652_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1650_, v___x_1649_);
v___x_1653_ = lean_string_append(v___x_1651_, v___x_1652_);
lean_dec_ref(v___x_1652_);
v___x_1654_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1655_ = lean_string_append(v___x_1653_, v___x_1654_);
v___y_1568_ = v_a_1645_;
v___y_1569_ = v___x_1647_;
v___y_1570_ = v___x_1537_;
v___y_1571_ = v___x_1649_;
v___y_1572_ = v___x_1655_;
goto v___jp_1567_;
}
else
{
lean_object* v___x_1656_; 
v___x_1656_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1568_ = v_a_1645_;
v___y_1569_ = v___x_1647_;
v___y_1570_ = v___x_1537_;
v___y_1571_ = v___x_1649_;
v___y_1572_ = v___x_1656_;
goto v___jp_1567_;
}
}
}
}
else
{
lean_object* v_a_1657_; lean_object* v___x_1658_; uint8_t v___x_1659_; lean_object* v___x_1660_; 
v_a_1657_ = lean_ctor_get(v___x_1641_, 0);
lean_inc(v_a_1657_);
lean_dec_ref_known(v___x_1641_, 1);
v___x_1658_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11));
v___x_1659_ = 0;
v___x_1660_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v___x_1658_, v___x_1659_, v___y_1526_);
if (lean_obj_tag(v___x_1660_) == 0)
{
lean_object* v_a_1661_; uint8_t v___x_1662_; 
v_a_1661_ = lean_ctor_get(v___x_1660_, 0);
lean_inc(v_a_1661_);
lean_dec_ref_known(v___x_1660_, 1);
v___x_1662_ = lean_unbox(v_a_1661_);
if (v___x_1662_ == 0)
{
lean_dec(v_a_1661_);
v___y_1540_ = v_a_1657_;
goto v___jp_1539_;
}
else
{
if (lean_obj_tag(v_currentModule_1522_) == 1)
{
lean_object* v_val_1663_; lean_object* v___x_1664_; uint8_t v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; uint8_t v___x_1670_; 
v_val_1663_ = lean_ctor_get(v_currentModule_1522_, 0);
v___x_1664_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
v___x_1665_ = lean_unbox(v_a_1661_);
lean_inc(v_val_1663_);
v___x_1666_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1663_, v___x_1665_);
v___x_1667_ = lean_string_append(v___x_1664_, v___x_1666_);
lean_dec_ref(v___x_1666_);
v___x_1668_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1669_ = lean_string_append(v___x_1667_, v___x_1668_);
v___x_1670_ = lean_unbox(v_a_1661_);
lean_dec(v_a_1661_);
v___y_1613_ = v_a_1657_;
v___y_1614_ = v___x_1670_;
v___y_1615_ = v___x_1537_;
v___y_1616_ = v___x_1669_;
goto v___jp_1612_;
}
else
{
lean_object* v___x_1671_; uint8_t v___x_1672_; 
v___x_1671_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___x_1672_ = lean_unbox(v_a_1661_);
lean_dec(v_a_1661_);
v___y_1613_ = v_a_1657_;
v___y_1614_ = v___x_1672_;
v___y_1615_ = v___x_1537_;
v___y_1616_ = v___x_1671_;
goto v___jp_1612_;
}
}
}
else
{
lean_object* v_a_1673_; lean_object* v___x_1675_; uint8_t v_isShared_1676_; uint8_t v_isSharedCheck_1680_; 
lean_dec(v_a_1657_);
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1673_ = lean_ctor_get(v___x_1660_, 0);
v_isSharedCheck_1680_ = !lean_is_exclusive(v___x_1660_);
if (v_isSharedCheck_1680_ == 0)
{
v___x_1675_ = v___x_1660_;
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
else
{
lean_inc(v_a_1673_);
lean_dec(v___x_1660_);
v___x_1675_ = lean_box(0);
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
v_resetjp_1674_:
{
lean_object* v___x_1678_; 
if (v_isShared_1676_ == 0)
{
v___x_1678_ = v___x_1675_;
goto v_reusejp_1677_;
}
else
{
lean_object* v_reuseFailAlloc_1679_; 
v_reuseFailAlloc_1679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1679_, 0, v_a_1673_);
v___x_1678_ = v_reuseFailAlloc_1679_;
goto v_reusejp_1677_;
}
v_reusejp_1677_:
{
return v___x_1678_;
}
}
}
}
}
else
{
lean_object* v_a_1681_; lean_object* v___x_1683_; uint8_t v_isShared_1684_; uint8_t v_isSharedCheck_1688_; 
lean_dec_ref(v_bs_x27_1538_);
lean_del_object(v___x_1535_);
lean_dec(v_fst_1532_);
lean_dec(v_currentModule_1522_);
v_a_1681_ = lean_ctor_get(v___x_1641_, 0);
v_isSharedCheck_1688_ = !lean_is_exclusive(v___x_1641_);
if (v_isSharedCheck_1688_ == 0)
{
v___x_1683_ = v___x_1641_;
v_isShared_1684_ = v_isSharedCheck_1688_;
goto v_resetjp_1682_;
}
else
{
lean_inc(v_a_1681_);
lean_dec(v___x_1641_);
v___x_1683_ = lean_box(0);
v_isShared_1684_ = v_isSharedCheck_1688_;
goto v_resetjp_1682_;
}
v_resetjp_1682_:
{
lean_object* v___x_1686_; 
if (v_isShared_1684_ == 0)
{
v___x_1686_ = v___x_1683_;
goto v_reusejp_1685_;
}
else
{
lean_object* v_reuseFailAlloc_1687_; 
v_reuseFailAlloc_1687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1687_, 0, v_a_1681_);
v___x_1686_ = v_reuseFailAlloc_1687_;
goto v_reusejp_1685_;
}
v_reusejp_1685_:
{
return v___x_1686_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___boxed(lean_object* v___y_1772_, lean_object* v_currentModule_1773_, lean_object* v_sz_1774_, lean_object* v_i_1775_, lean_object* v_bs_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_){
_start:
{
uint8_t v___y_25507__boxed_1780_; size_t v_sz_boxed_1781_; size_t v_i_boxed_1782_; lean_object* v_res_1783_; 
v___y_25507__boxed_1780_ = lean_unbox(v___y_1772_);
v_sz_boxed_1781_ = lean_unbox_usize(v_sz_1774_);
lean_dec(v_sz_1774_);
v_i_boxed_1782_ = lean_unbox_usize(v_i_1775_);
lean_dec(v_i_1775_);
v_res_1783_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10(v___y_25507__boxed_1780_, v_currentModule_1773_, v_sz_boxed_1781_, v_i_boxed_1782_, v_bs_1776_, v___y_1777_, v___y_1778_);
lean_dec(v___y_1778_);
lean_dec_ref(v___y_1777_);
return v_res_1783_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11(uint8_t v___x_1784_, size_t v_sz_1785_, size_t v_i_1786_, lean_object* v_bs_1787_){
_start:
{
uint8_t v___x_1788_; 
v___x_1788_ = lean_usize_dec_lt(v_i_1786_, v_sz_1785_);
if (v___x_1788_ == 0)
{
return v_bs_1787_;
}
else
{
lean_object* v_v_1789_; lean_object* v_name_1790_; lean_object* v___x_1791_; lean_object* v_bs_x27_1792_; lean_object* v___x_1793_; size_t v___x_1794_; size_t v___x_1795_; lean_object* v___x_1796_; 
v_v_1789_ = lean_array_uget_borrowed(v_bs_1787_, v_i_1786_);
v_name_1790_ = lean_ctor_get(v_v_1789_, 1);
lean_inc(v_name_1790_);
v___x_1791_ = lean_unsigned_to_nat(0u);
v_bs_x27_1792_ = lean_array_uset(v_bs_1787_, v_i_1786_, v___x_1791_);
v___x_1793_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1790_, v___x_1784_);
v___x_1794_ = ((size_t)1ULL);
v___x_1795_ = lean_usize_add(v_i_1786_, v___x_1794_);
v___x_1796_ = lean_array_uset(v_bs_x27_1792_, v_i_1786_, v___x_1793_);
v_i_1786_ = v___x_1795_;
v_bs_1787_ = v___x_1796_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11___boxed(lean_object* v___x_1798_, lean_object* v_sz_1799_, lean_object* v_i_1800_, lean_object* v_bs_1801_){
_start:
{
uint8_t v___x_25988__boxed_1802_; size_t v_sz_boxed_1803_; size_t v_i_boxed_1804_; lean_object* v_res_1805_; 
v___x_25988__boxed_1802_ = lean_unbox(v___x_1798_);
v_sz_boxed_1803_ = lean_unbox_usize(v_sz_1799_);
lean_dec(v_sz_1799_);
v_i_boxed_1804_ = lean_unbox_usize(v_i_1800_);
lean_dec(v_i_1800_);
v_res_1805_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11(v___x_25988__boxed_1802_, v_sz_boxed_1803_, v_i_boxed_1804_, v_bs_1801_);
return v_res_1805_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore(lean_object* v_decls_1809_, lean_object* v_linters_1810_, lean_object* v_currentModule_1811_, uint8_t v_inIO_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_){
_start:
{
lean_object* v___y_1817_; lean_object* v___y_1818_; lean_object* v___y_1819_; lean_object* v___y_1842_; lean_object* v___y_1843_; 
if (v_inIO_1812_ == 0)
{
lean_object* v_options_1918_; uint8_t v_hasTrace_1919_; 
v_options_1918_ = lean_ctor_get(v_a_1813_, 2);
v_hasTrace_1919_ = lean_ctor_get_uint8(v_options_1918_, sizeof(void*)*1);
if (v_hasTrace_1919_ == 0)
{
goto v___jp_1868_;
}
else
{
lean_object* v_inheritedTraceOptions_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; uint8_t v___x_1923_; lean_object* v___y_1925_; 
v_inheritedTraceOptions_1920_ = lean_ctor_get(v_a_1813_, 13);
v___x_1921_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_1922_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5);
v___x_1923_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1920_, v_options_1918_, v___x_1922_);
if (v___x_1923_ == 0)
{
goto v___jp_1868_;
}
else
{
if (lean_obj_tag(v_currentModule_1811_) == 1)
{
lean_object* v_val_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; 
v_val_1946_ = lean_ctor_get(v_currentModule_1811_, 0);
v___x_1947_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
lean_inc(v_val_1946_);
v___x_1948_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1946_, v___x_1923_);
v___x_1949_ = lean_string_append(v___x_1947_, v___x_1948_);
lean_dec_ref(v___x_1948_);
v___x_1950_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1951_ = lean_string_append(v___x_1949_, v___x_1950_);
v___y_1925_ = v___x_1951_;
goto v___jp_1924_;
}
else
{
lean_object* v___x_1952_; 
v___x_1952_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1925_ = v___x_1952_;
goto v___jp_1924_;
}
}
v___jp_1924_:
{
lean_object* v___x_1926_; lean_object* v___x_1927_; size_t v_sz_1928_; size_t v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; 
v___x_1926_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_lintCore___closed__1));
v___x_1927_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_lintCore___closed__2));
v_sz_1928_ = lean_array_size(v_linters_1810_);
v___x_1929_ = ((size_t)0ULL);
lean_inc_ref(v_linters_1810_);
v___x_1930_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11(v___x_1923_, v_sz_1928_, v___x_1929_, v_linters_1810_);
v___x_1931_ = lean_array_to_list(v___x_1930_);
v___x_1932_ = l_String_intercalate(v___x_1927_, v___x_1931_);
v___x_1933_ = lean_string_append(v___x_1926_, v___x_1932_);
lean_dec_ref(v___x_1932_);
v___x_1934_ = lean_string_append(v___y_1925_, v___x_1933_);
lean_dec_ref(v___x_1933_);
v___x_1935_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1935_, 0, v___x_1934_);
v___x_1936_ = l_Lean_MessageData_ofFormat(v___x_1935_);
v___x_1937_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(v___x_1921_, v___x_1936_, v_a_1813_, v_a_1814_);
if (lean_obj_tag(v___x_1937_) == 0)
{
lean_dec_ref_known(v___x_1937_, 1);
goto v___jp_1868_;
}
else
{
lean_object* v_a_1938_; lean_object* v___x_1940_; uint8_t v_isShared_1941_; uint8_t v_isSharedCheck_1945_; 
lean_dec(v_currentModule_1811_);
lean_dec_ref(v_linters_1810_);
v_a_1938_ = lean_ctor_get(v___x_1937_, 0);
v_isSharedCheck_1945_ = !lean_is_exclusive(v___x_1937_);
if (v_isSharedCheck_1945_ == 0)
{
v___x_1940_ = v___x_1937_;
v_isShared_1941_ = v_isSharedCheck_1945_;
goto v_resetjp_1939_;
}
else
{
lean_inc(v_a_1938_);
lean_dec(v___x_1937_);
v___x_1940_ = lean_box(0);
v_isShared_1941_ = v_isSharedCheck_1945_;
goto v_resetjp_1939_;
}
v_resetjp_1939_:
{
lean_object* v___x_1943_; 
if (v_isShared_1941_ == 0)
{
v___x_1943_ = v___x_1940_;
goto v_reusejp_1942_;
}
else
{
lean_object* v_reuseFailAlloc_1944_; 
v_reuseFailAlloc_1944_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1944_, 0, v_a_1938_);
v___x_1943_ = v_reuseFailAlloc_1944_;
goto v_reusejp_1942_;
}
v_reusejp_1942_:
{
return v___x_1943_;
}
}
}
}
}
}
else
{
lean_object* v___x_1953_; uint8_t v___x_1954_; lean_object* v___x_1955_; lean_object* v_a_1956_; lean_object* v___x_1958_; uint8_t v_isShared_1959_; uint8_t v_isSharedCheck_1997_; 
v___x_1953_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11));
v___x_1954_ = 0;
v___x_1955_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v___x_1953_, v___x_1954_, v_a_1813_);
v_a_1956_ = lean_ctor_get(v___x_1955_, 0);
v_isSharedCheck_1997_ = !lean_is_exclusive(v___x_1955_);
if (v_isSharedCheck_1997_ == 0)
{
v___x_1958_ = v___x_1955_;
v_isShared_1959_ = v_isSharedCheck_1997_;
goto v_resetjp_1957_;
}
else
{
lean_inc(v_a_1956_);
lean_dec(v___x_1955_);
v___x_1958_ = lean_box(0);
v_isShared_1959_ = v_isSharedCheck_1997_;
goto v_resetjp_1957_;
}
v_resetjp_1957_:
{
lean_object* v___y_1961_; uint8_t v___x_1988_; 
v___x_1988_ = lean_unbox(v_a_1956_);
if (v___x_1988_ == 0)
{
lean_del_object(v___x_1958_);
lean_dec(v_a_1956_);
goto v___jp_1868_;
}
else
{
if (lean_obj_tag(v_currentModule_1811_) == 1)
{
lean_object* v_val_1989_; lean_object* v___x_1990_; uint8_t v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; 
v_val_1989_ = lean_ctor_get(v_currentModule_1811_, 0);
v___x_1990_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
v___x_1991_ = lean_unbox(v_a_1956_);
lean_inc(v_val_1989_);
v___x_1992_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1989_, v___x_1991_);
v___x_1993_ = lean_string_append(v___x_1990_, v___x_1992_);
lean_dec_ref(v___x_1992_);
v___x_1994_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1995_ = lean_string_append(v___x_1993_, v___x_1994_);
v___y_1961_ = v___x_1995_;
goto v___jp_1960_;
}
else
{
lean_object* v___x_1996_; 
v___x_1996_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1961_ = v___x_1996_;
goto v___jp_1960_;
}
}
v___jp_1960_:
{
lean_object* v___x_1962_; lean_object* v___x_1963_; size_t v_sz_1964_; size_t v___x_1965_; uint8_t v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; 
v___x_1962_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_lintCore___closed__1));
v___x_1963_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_lintCore___closed__2));
v_sz_1964_ = lean_array_size(v_linters_1810_);
v___x_1965_ = ((size_t)0ULL);
v___x_1966_ = lean_unbox(v_a_1956_);
lean_dec(v_a_1956_);
lean_inc_ref(v_linters_1810_);
v___x_1967_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__11(v___x_1966_, v_sz_1964_, v___x_1965_, v_linters_1810_);
v___x_1968_ = lean_array_to_list(v___x_1967_);
v___x_1969_ = l_String_intercalate(v___x_1963_, v___x_1968_);
v___x_1970_ = lean_string_append(v___x_1962_, v___x_1969_);
lean_dec_ref(v___x_1969_);
v___x_1971_ = lean_string_append(v___y_1961_, v___x_1970_);
lean_dec_ref(v___x_1970_);
v___x_1972_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_1971_);
if (lean_obj_tag(v___x_1972_) == 0)
{
lean_dec_ref_known(v___x_1972_, 1);
lean_del_object(v___x_1958_);
goto v___jp_1868_;
}
else
{
lean_object* v_a_1973_; lean_object* v___x_1975_; uint8_t v_isShared_1976_; uint8_t v_isSharedCheck_1987_; 
lean_dec(v_currentModule_1811_);
lean_dec_ref(v_linters_1810_);
v_a_1973_ = lean_ctor_get(v___x_1972_, 0);
v_isSharedCheck_1987_ = !lean_is_exclusive(v___x_1972_);
if (v_isSharedCheck_1987_ == 0)
{
v___x_1975_ = v___x_1972_;
v_isShared_1976_ = v_isSharedCheck_1987_;
goto v_resetjp_1974_;
}
else
{
lean_inc(v_a_1973_);
lean_dec(v___x_1972_);
v___x_1975_ = lean_box(0);
v_isShared_1976_ = v_isSharedCheck_1987_;
goto v_resetjp_1974_;
}
v_resetjp_1974_:
{
lean_object* v_ref_1977_; lean_object* v___x_1978_; lean_object* v___x_1980_; 
v_ref_1977_ = lean_ctor_get(v_a_1813_, 5);
v___x_1978_ = lean_io_error_to_string(v_a_1973_);
if (v_isShared_1959_ == 0)
{
lean_ctor_set_tag(v___x_1958_, 3);
lean_ctor_set(v___x_1958_, 0, v___x_1978_);
v___x_1980_ = v___x_1958_;
goto v_reusejp_1979_;
}
else
{
lean_object* v_reuseFailAlloc_1986_; 
v_reuseFailAlloc_1986_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1986_, 0, v___x_1978_);
v___x_1980_ = v_reuseFailAlloc_1986_;
goto v_reusejp_1979_;
}
v_reusejp_1979_:
{
lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1984_; 
v___x_1981_ = l_Lean_MessageData_ofFormat(v___x_1980_);
lean_inc(v_ref_1977_);
v___x_1982_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1982_, 0, v_ref_1977_);
lean_ctor_set(v___x_1982_, 1, v___x_1981_);
if (v_isShared_1976_ == 0)
{
lean_ctor_set(v___x_1975_, 0, v___x_1982_);
v___x_1984_ = v___x_1975_;
goto v_reusejp_1983_;
}
else
{
lean_object* v_reuseFailAlloc_1985_; 
v_reuseFailAlloc_1985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1985_, 0, v___x_1982_);
v___x_1984_ = v_reuseFailAlloc_1985_;
goto v_reusejp_1983_;
}
v_reusejp_1983_:
{
return v___x_1984_;
}
}
}
}
}
}
}
v___jp_1816_:
{
lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; 
v___x_1820_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_lintCore___closed__0));
v___x_1821_ = lean_string_append(v___y_1819_, v___x_1820_);
v___x_1822_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1822_, 0, v___x_1821_);
v___x_1823_ = l_Lean_MessageData_ofFormat(v___x_1822_);
lean_inc(v___y_1818_);
v___x_1824_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2(v___y_1818_, v___x_1823_, v_a_1813_, v_a_1814_);
if (lean_obj_tag(v___x_1824_) == 0)
{
lean_object* v___x_1826_; uint8_t v_isShared_1827_; uint8_t v_isSharedCheck_1831_; 
v_isSharedCheck_1831_ = !lean_is_exclusive(v___x_1824_);
if (v_isSharedCheck_1831_ == 0)
{
lean_object* v_unused_1832_; 
v_unused_1832_ = lean_ctor_get(v___x_1824_, 0);
lean_dec(v_unused_1832_);
v___x_1826_ = v___x_1824_;
v_isShared_1827_ = v_isSharedCheck_1831_;
goto v_resetjp_1825_;
}
else
{
lean_dec(v___x_1824_);
v___x_1826_ = lean_box(0);
v_isShared_1827_ = v_isSharedCheck_1831_;
goto v_resetjp_1825_;
}
v_resetjp_1825_:
{
lean_object* v___x_1829_; 
if (v_isShared_1827_ == 0)
{
lean_ctor_set(v___x_1826_, 0, v___y_1817_);
v___x_1829_ = v___x_1826_;
goto v_reusejp_1828_;
}
else
{
lean_object* v_reuseFailAlloc_1830_; 
v_reuseFailAlloc_1830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1830_, 0, v___y_1817_);
v___x_1829_ = v_reuseFailAlloc_1830_;
goto v_reusejp_1828_;
}
v_reusejp_1828_:
{
return v___x_1829_;
}
}
}
else
{
lean_object* v_a_1833_; lean_object* v___x_1835_; uint8_t v_isShared_1836_; uint8_t v_isSharedCheck_1840_; 
lean_dec_ref(v___y_1817_);
v_a_1833_ = lean_ctor_get(v___x_1824_, 0);
v_isSharedCheck_1840_ = !lean_is_exclusive(v___x_1824_);
if (v_isSharedCheck_1840_ == 0)
{
v___x_1835_ = v___x_1824_;
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
else
{
lean_inc(v_a_1833_);
lean_dec(v___x_1824_);
v___x_1835_ = lean_box(0);
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
v_resetjp_1834_:
{
lean_object* v___x_1838_; 
if (v_isShared_1836_ == 0)
{
v___x_1838_ = v___x_1835_;
goto v_reusejp_1837_;
}
else
{
lean_object* v_reuseFailAlloc_1839_; 
v_reuseFailAlloc_1839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1839_, 0, v_a_1833_);
v___x_1838_ = v_reuseFailAlloc_1839_;
goto v_reusejp_1837_;
}
v_reusejp_1837_:
{
return v___x_1838_;
}
}
}
}
v___jp_1841_:
{
lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; 
v___x_1844_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_lintCore___closed__0));
v___x_1845_ = lean_string_append(v___y_1843_, v___x_1844_);
v___x_1846_ = lp_batteries_IO_println___at___00Batteries_Tactic_Lint_lintCore_spec__4(v___x_1845_);
if (lean_obj_tag(v___x_1846_) == 0)
{
lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_1853_; 
v_isSharedCheck_1853_ = !lean_is_exclusive(v___x_1846_);
if (v_isSharedCheck_1853_ == 0)
{
lean_object* v_unused_1854_; 
v_unused_1854_ = lean_ctor_get(v___x_1846_, 0);
lean_dec(v_unused_1854_);
v___x_1848_ = v___x_1846_;
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
else
{
lean_dec(v___x_1846_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
lean_object* v___x_1851_; 
if (v_isShared_1849_ == 0)
{
lean_ctor_set(v___x_1848_, 0, v___y_1842_);
v___x_1851_ = v___x_1848_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v___y_1842_);
v___x_1851_ = v_reuseFailAlloc_1852_;
goto v_reusejp_1850_;
}
v_reusejp_1850_:
{
return v___x_1851_;
}
}
}
else
{
lean_object* v_a_1855_; lean_object* v___x_1857_; uint8_t v_isShared_1858_; uint8_t v_isSharedCheck_1867_; 
lean_dec_ref(v___y_1842_);
v_a_1855_ = lean_ctor_get(v___x_1846_, 0);
v_isSharedCheck_1867_ = !lean_is_exclusive(v___x_1846_);
if (v_isSharedCheck_1867_ == 0)
{
v___x_1857_ = v___x_1846_;
v_isShared_1858_ = v_isSharedCheck_1867_;
goto v_resetjp_1856_;
}
else
{
lean_inc(v_a_1855_);
lean_dec(v___x_1846_);
v___x_1857_ = lean_box(0);
v_isShared_1858_ = v_isSharedCheck_1867_;
goto v_resetjp_1856_;
}
v_resetjp_1856_:
{
lean_object* v_ref_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1865_; 
v_ref_1859_ = lean_ctor_get(v_a_1813_, 5);
v___x_1860_ = lean_io_error_to_string(v_a_1855_);
v___x_1861_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1861_, 0, v___x_1860_);
v___x_1862_ = l_Lean_MessageData_ofFormat(v___x_1861_);
lean_inc(v_ref_1859_);
v___x_1863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1863_, 0, v_ref_1859_);
lean_ctor_set(v___x_1863_, 1, v___x_1862_);
if (v_isShared_1858_ == 0)
{
lean_ctor_set(v___x_1857_, 0, v___x_1863_);
v___x_1865_ = v___x_1857_;
goto v_reusejp_1864_;
}
else
{
lean_object* v_reuseFailAlloc_1866_; 
v_reuseFailAlloc_1866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1866_, 0, v___x_1863_);
v___x_1865_ = v_reuseFailAlloc_1866_;
goto v_reusejp_1864_;
}
v_reusejp_1864_:
{
return v___x_1865_;
}
}
}
}
v___jp_1868_:
{
size_t v_sz_1869_; size_t v___x_1870_; lean_object* v___x_1871_; 
v_sz_1869_ = lean_array_size(v_linters_1810_);
v___x_1870_ = ((size_t)0ULL);
lean_inc(v_currentModule_1811_);
v___x_1871_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9(v_inIO_1812_, v_decls_1809_, v_currentModule_1811_, v_sz_1869_, v___x_1870_, v_linters_1810_, v_a_1813_, v_a_1814_);
if (lean_obj_tag(v___x_1871_) == 0)
{
lean_object* v_a_1872_; size_t v_sz_1873_; lean_object* v___x_1874_; 
v_a_1872_ = lean_ctor_get(v___x_1871_, 0);
lean_inc(v_a_1872_);
lean_dec_ref_known(v___x_1871_, 1);
v_sz_1873_ = lean_array_size(v_a_1872_);
lean_inc(v_currentModule_1811_);
v___x_1874_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10(v_inIO_1812_, v_currentModule_1811_, v_sz_1873_, v___x_1870_, v_a_1872_, v_a_1813_, v_a_1814_);
if (lean_obj_tag(v___x_1874_) == 0)
{
if (v_inIO_1812_ == 0)
{
lean_object* v_options_1875_; uint8_t v_hasTrace_1876_; 
v_options_1875_ = lean_ctor_get(v_a_1813_, 2);
v_hasTrace_1876_ = lean_ctor_get_uint8(v_options_1875_, sizeof(void*)*1);
if (v_hasTrace_1876_ == 0)
{
lean_dec(v_currentModule_1811_);
return v___x_1874_;
}
else
{
lean_object* v_a_1877_; lean_object* v_inheritedTraceOptions_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; uint8_t v___x_1881_; 
v_a_1877_ = lean_ctor_get(v___x_1874_, 0);
lean_inc(v_a_1877_);
v_inheritedTraceOptions_1878_ = lean_ctor_get(v_a_1813_, 13);
v___x_1879_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_1880_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__5);
v___x_1881_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1878_, v_options_1875_, v___x_1880_);
if (v___x_1881_ == 0)
{
lean_dec(v_a_1877_);
lean_dec(v_currentModule_1811_);
return v___x_1874_;
}
else
{
lean_dec_ref_known(v___x_1874_, 1);
if (lean_obj_tag(v_currentModule_1811_) == 1)
{
lean_object* v_val_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; 
v_val_1882_ = lean_ctor_get(v_currentModule_1811_, 0);
lean_inc(v_val_1882_);
lean_dec_ref_known(v_currentModule_1811_, 1);
v___x_1883_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
v___x_1884_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1882_, v___x_1881_);
v___x_1885_ = lean_string_append(v___x_1883_, v___x_1884_);
lean_dec_ref(v___x_1884_);
v___x_1886_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1887_ = lean_string_append(v___x_1885_, v___x_1886_);
v___y_1817_ = v_a_1877_;
v___y_1818_ = v___x_1879_;
v___y_1819_ = v___x_1887_;
goto v___jp_1816_;
}
else
{
lean_object* v___x_1888_; 
lean_dec(v_currentModule_1811_);
v___x_1888_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1817_ = v_a_1877_;
v___y_1818_ = v___x_1879_;
v___y_1819_ = v___x_1888_;
goto v___jp_1816_;
}
}
}
}
else
{
lean_object* v_a_1889_; lean_object* v___x_1890_; uint8_t v___x_1891_; lean_object* v___x_1892_; lean_object* v_a_1893_; lean_object* v___x_1895_; uint8_t v_isShared_1896_; uint8_t v_isSharedCheck_1909_; 
v_a_1889_ = lean_ctor_get(v___x_1874_, 0);
lean_inc(v_a_1889_);
lean_dec_ref_known(v___x_1874_, 1);
v___x_1890_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__11));
v___x_1891_ = 0;
v___x_1892_ = lp_batteries_Lean_getBoolOption___at___00Batteries_Tactic_Lint_lintCore_spec__3___redArg(v___x_1890_, v___x_1891_, v_a_1813_);
v_a_1893_ = lean_ctor_get(v___x_1892_, 0);
v_isSharedCheck_1909_ = !lean_is_exclusive(v___x_1892_);
if (v_isSharedCheck_1909_ == 0)
{
v___x_1895_ = v___x_1892_;
v_isShared_1896_ = v_isSharedCheck_1909_;
goto v_resetjp_1894_;
}
else
{
lean_inc(v_a_1893_);
lean_dec(v___x_1892_);
v___x_1895_ = lean_box(0);
v_isShared_1896_ = v_isSharedCheck_1909_;
goto v_resetjp_1894_;
}
v_resetjp_1894_:
{
uint8_t v___x_1897_; 
v___x_1897_ = lean_unbox(v_a_1893_);
if (v___x_1897_ == 0)
{
lean_object* v___x_1899_; 
lean_dec(v_a_1893_);
lean_dec(v_currentModule_1811_);
if (v_isShared_1896_ == 0)
{
lean_ctor_set(v___x_1895_, 0, v_a_1889_);
v___x_1899_ = v___x_1895_;
goto v_reusejp_1898_;
}
else
{
lean_object* v_reuseFailAlloc_1900_; 
v_reuseFailAlloc_1900_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1900_, 0, v_a_1889_);
v___x_1899_ = v_reuseFailAlloc_1900_;
goto v_reusejp_1898_;
}
v_reusejp_1898_:
{
return v___x_1899_;
}
}
else
{
lean_del_object(v___x_1895_);
if (lean_obj_tag(v_currentModule_1811_) == 1)
{
lean_object* v_val_1901_; lean_object* v___x_1902_; uint8_t v___x_1903_; lean_object* v___x_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; 
v_val_1901_ = lean_ctor_get(v_currentModule_1811_, 0);
lean_inc(v_val_1901_);
lean_dec_ref_known(v_currentModule_1811_, 1);
v___x_1902_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__9));
v___x_1903_ = lean_unbox(v_a_1893_);
lean_dec(v_a_1893_);
v___x_1904_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1901_, v___x_1903_);
v___x_1905_ = lean_string_append(v___x_1902_, v___x_1904_);
lean_dec_ref(v___x_1904_);
v___x_1906_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__10));
v___x_1907_ = lean_string_append(v___x_1905_, v___x_1906_);
v___y_1842_ = v_a_1889_;
v___y_1843_ = v___x_1907_;
goto v___jp_1841_;
}
else
{
lean_object* v___x_1908_; 
lean_dec(v_a_1893_);
lean_dec(v_currentModule_1811_);
v___x_1908_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_1842_ = v_a_1889_;
v___y_1843_ = v___x_1908_;
goto v___jp_1841_;
}
}
}
}
}
else
{
lean_dec(v_currentModule_1811_);
return v___x_1874_;
}
}
else
{
lean_object* v_a_1910_; lean_object* v___x_1912_; uint8_t v_isShared_1913_; uint8_t v_isSharedCheck_1917_; 
lean_dec(v_currentModule_1811_);
v_a_1910_ = lean_ctor_get(v___x_1871_, 0);
v_isSharedCheck_1917_ = !lean_is_exclusive(v___x_1871_);
if (v_isSharedCheck_1917_ == 0)
{
v___x_1912_ = v___x_1871_;
v_isShared_1913_ = v_isSharedCheck_1917_;
goto v_resetjp_1911_;
}
else
{
lean_inc(v_a_1910_);
lean_dec(v___x_1871_);
v___x_1912_ = lean_box(0);
v_isShared_1913_ = v_isSharedCheck_1917_;
goto v_resetjp_1911_;
}
v_resetjp_1911_:
{
lean_object* v___x_1915_; 
if (v_isShared_1913_ == 0)
{
v___x_1915_ = v___x_1912_;
goto v_reusejp_1914_;
}
else
{
lean_object* v_reuseFailAlloc_1916_; 
v_reuseFailAlloc_1916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1916_, 0, v_a_1910_);
v___x_1915_ = v_reuseFailAlloc_1916_;
goto v_reusejp_1914_;
}
v_reusejp_1914_:
{
return v___x_1915_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_lintCore___boxed(lean_object* v_decls_1998_, lean_object* v_linters_1999_, lean_object* v_currentModule_2000_, lean_object* v_inIO_2001_, lean_object* v_a_2002_, lean_object* v_a_2003_, lean_object* v_a_2004_){
_start:
{
uint8_t v_inIO_boxed_2005_; lean_object* v_res_2006_; 
v_inIO_boxed_2005_ = lean_unbox(v_inIO_2001_);
v_res_2006_ = lp_batteries_Batteries_Tactic_Lint_lintCore(v_decls_1998_, v_linters_1999_, v_currentModule_2000_, v_inIO_boxed_2005_, v_a_2002_, v_a_2003_);
lean_dec(v_a_2003_);
lean_dec_ref(v_a_2002_);
lean_dec_ref(v_decls_1998_);
return v_res_2006_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0(lean_object* v_00_u03b2_2007_, lean_object* v_m_2008_, lean_object* v_a_2009_, lean_object* v_b_2010_){
_start:
{
lean_object* v___x_2011_; 
v___x_2011_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(v_m_2008_, v_a_2009_, v_b_2010_);
return v___x_2011_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1(lean_object* v_as_2012_, size_t v_sz_2013_, size_t v_i_2014_, lean_object* v_b_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_){
_start:
{
lean_object* v___x_2019_; 
v___x_2019_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___redArg(v_as_2012_, v_sz_2013_, v_i_2014_, v_b_2015_);
return v___x_2019_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1___boxed(lean_object* v_as_2020_, lean_object* v_sz_2021_, lean_object* v_i_2022_, lean_object* v_b_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_){
_start:
{
size_t v_sz_boxed_2027_; size_t v_i_boxed_2028_; lean_object* v_res_2029_; 
v_sz_boxed_2027_ = lean_unbox_usize(v_sz_2021_);
lean_dec(v_sz_2021_);
v_i_boxed_2028_ = lean_unbox_usize(v_i_2022_);
lean_dec(v_i_2022_);
v_res_2029_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_lintCore_spec__1(v_as_2020_, v_sz_boxed_2027_, v_i_boxed_2028_, v_b_2023_, v___y_2024_, v___y_2025_);
lean_dec(v___y_2025_);
lean_dec_ref(v___y_2024_);
lean_dec_ref(v_as_2020_);
return v_res_2029_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7(lean_object* v_linter_2030_, lean_object* v_decl_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_){
_start:
{
lean_object* v___x_2035_; 
v___x_2035_ = lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg(v_linter_2030_, v_decl_2031_, v___y_2033_);
return v___x_2035_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___boxed(lean_object* v_linter_2036_, lean_object* v_decl_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_){
_start:
{
lean_object* v_res_2041_; 
v_res_2041_ = lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7(v_linter_2036_, v_decl_2037_, v___y_2038_, v___y_2039_);
lean_dec(v___y_2039_);
lean_dec_ref(v___y_2038_);
lean_dec(v_linter_2036_);
return v_res_2041_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0(lean_object* v_00_u03b2_2042_, lean_object* v_a_2043_, lean_object* v_x_2044_){
_start:
{
uint8_t v___x_2045_; 
v___x_2045_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___redArg(v_a_2043_, v_x_2044_);
return v___x_2045_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0___boxed(lean_object* v_00_u03b2_2046_, lean_object* v_a_2047_, lean_object* v_x_2048_){
_start:
{
uint8_t v_res_2049_; lean_object* v_r_2050_; 
v_res_2049_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__0(v_00_u03b2_2046_, v_a_2047_, v_x_2048_);
lean_dec(v_x_2048_);
lean_dec(v_a_2047_);
v_r_2050_ = lean_box(v_res_2049_);
return v_r_2050_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1(lean_object* v_00_u03b2_2051_, lean_object* v_data_2052_){
_start:
{
lean_object* v___x_2053_; 
v___x_2053_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1___redArg(v_data_2052_);
return v___x_2053_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2(lean_object* v_00_u03b2_2054_, lean_object* v_a_2055_, lean_object* v_b_2056_, lean_object* v_x_2057_){
_start:
{
lean_object* v___x_2058_; 
v___x_2058_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__2___redArg(v_a_2055_, v_b_2056_, v_x_2057_);
return v___x_2058_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_2059_, lean_object* v_i_2060_, lean_object* v_source_2061_, lean_object* v_target_2062_){
_start:
{
lean_object* v___x_2063_; 
v___x_2063_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3___redArg(v_i_2060_, v_source_2061_, v_target_2062_);
return v___x_2063_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3_spec__16(lean_object* v_00_u03b2_2064_, lean_object* v_x_2065_, lean_object* v_x_2066_){
_start:
{
lean_object* v___x_2067_; 
v___x_2067_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0_spec__1_spec__3_spec__16___redArg(v_x_2065_, v_x_2066_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24(lean_object* v_as_2068_, size_t v_sz_2069_, size_t v_i_2070_, lean_object* v_b_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_){
_start:
{
lean_object* v___x_2077_; 
v___x_2077_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___redArg(v_as_2068_, v_sz_2069_, v_i_2070_, v_b_2071_, v___y_2074_);
return v___x_2077_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24___boxed(lean_object* v_as_2078_, lean_object* v_sz_2079_, lean_object* v_i_2080_, lean_object* v_b_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_){
_start:
{
size_t v_sz_boxed_2087_; size_t v_i_boxed_2088_; lean_object* v_res_2089_; 
v_sz_boxed_2087_ = lean_unbox_usize(v_sz_2079_);
lean_dec(v_sz_2079_);
v_i_boxed_2088_ = lean_unbox_usize(v_i_2080_);
lean_dec(v_i_2080_);
v_res_2089_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__14_spec__24(v_as_2078_, v_sz_boxed_2087_, v_i_boxed_2088_, v_b_2081_, v___y_2082_, v___y_2083_, v___y_2084_, v___y_2085_);
lean_dec(v___y_2085_);
lean_dec_ref(v___y_2084_);
lean_dec(v___y_2083_);
lean_dec_ref(v___y_2082_);
lean_dec_ref(v_as_2078_);
return v_res_2089_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24(lean_object* v_as_2090_, size_t v_sz_2091_, size_t v_i_2092_, lean_object* v_b_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_){
_start:
{
lean_object* v___x_2099_; 
v___x_2099_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___redArg(v_as_2090_, v_sz_2091_, v_i_2092_, v_b_2093_, v___y_2096_);
return v___x_2099_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24___boxed(lean_object* v_as_2100_, lean_object* v_sz_2101_, lean_object* v_i_2102_, lean_object* v_b_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_){
_start:
{
size_t v_sz_boxed_2109_; size_t v_i_boxed_2110_; lean_object* v_res_2111_; 
v_sz_boxed_2109_ = lean_unbox_usize(v_sz_2101_);
lean_dec(v_sz_2101_);
v_i_boxed_2110_ = lean_unbox_usize(v_i_2102_);
lean_dec(v_i_2102_);
v_res_2111_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_printTraces___at___00Batteries_Tactic_Lint_lintCore_spec__5_spec__11_spec__13_spec__22_spec__24(v_as_2100_, v_sz_boxed_2109_, v_i_boxed_2110_, v_b_2103_, v___y_2104_, v___y_2105_, v___y_2106_, v___y_2107_);
lean_dec(v___y_2107_);
lean_dec_ref(v___y_2106_);
lean_dec(v___y_2105_);
lean_dec_ref(v___y_2104_);
lean_dec_ref(v_as_2100_);
return v_res_2111_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg(lean_object* v_declName_2112_, lean_object* v___y_2113_){
_start:
{
lean_object* v___x_2115_; lean_object* v_env_2116_; uint8_t v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; 
v___x_2115_ = lean_st_ref_get(v___y_2113_);
v_env_2116_ = lean_ctor_get(v___x_2115_, 0);
lean_inc_ref(v_env_2116_);
lean_dec(v___x_2115_);
v___x_2117_ = l_Lean_isRecCore(v_env_2116_, v_declName_2112_);
v___x_2118_ = lean_box(v___x_2117_);
v___x_2119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2119_, 0, v___x_2118_);
return v___x_2119_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg___boxed(lean_object* v_declName_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_){
_start:
{
lean_object* v_res_2123_; 
v_res_2123_ = lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg(v_declName_2120_, v___y_2121_);
lean_dec(v___y_2121_);
return v_res_2123_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg(lean_object* v_declName_2124_, lean_object* v___y_2125_){
_start:
{
lean_object* v___x_2127_; lean_object* v_env_2128_; lean_object* v___x_2129_; lean_object* v_env_2130_; lean_object* v___x_2131_; lean_object* v_toEnvExtension_2132_; lean_object* v_asyncMode_2133_; lean_object* v___x_2134_; uint8_t v___x_2135_; lean_object* v___x_2136_; 
v___x_2127_ = lean_st_ref_get(v___y_2125_);
v_env_2128_ = lean_ctor_get(v___x_2127_, 0);
lean_inc_ref(v_env_2128_);
lean_dec(v___x_2127_);
v___x_2129_ = lean_st_ref_get(v___y_2125_);
v_env_2130_ = lean_ctor_get(v___x_2129_, 0);
lean_inc_ref(v_env_2130_);
lean_dec(v___x_2129_);
v___x_2131_ = l_Lean_declRangeExt;
v_toEnvExtension_2132_ = lean_ctor_get(v___x_2131_, 0);
v_asyncMode_2133_ = lean_ctor_get(v_toEnvExtension_2132_, 2);
v___x_2134_ = l_Lean_instInhabitedDeclarationRanges_default;
v___x_2135_ = 0;
lean_inc(v_declName_2124_);
v___x_2136_ = l_Lean_MapDeclarationExtension_find_x3f___redArg(v___x_2134_, v___x_2131_, v_env_2128_, v_declName_2124_, v_asyncMode_2133_, v___x_2135_);
if (lean_obj_tag(v___x_2136_) == 0)
{
uint8_t v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; 
v___x_2137_ = 1;
v___x_2138_ = l_Lean_MapDeclarationExtension_find_x3f___redArg(v___x_2134_, v___x_2131_, v_env_2130_, v_declName_2124_, v_asyncMode_2133_, v___x_2137_);
v___x_2139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2139_, 0, v___x_2138_);
return v___x_2139_;
}
else
{
lean_object* v___x_2140_; 
lean_dec_ref(v_env_2130_);
lean_dec(v_declName_2124_);
v___x_2140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2140_, 0, v___x_2136_);
return v___x_2140_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg___boxed(lean_object* v_declName_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_){
_start:
{
lean_object* v_res_2144_; 
v_res_2144_ = lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg(v_declName_2141_, v___y_2142_);
lean_dec(v___y_2142_);
return v_res_2144_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0(lean_object* v_declName_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_){
_start:
{
lean_object* v_ranges_2150_; lean_object* v___x_2156_; lean_object* v_env_2157_; lean_object* v___x_2158_; lean_object* v_a_2159_; uint8_t v___y_2165_; uint8_t v___x_2169_; 
v___x_2156_ = lean_st_ref_get(v___y_2147_);
v_env_2157_ = lean_ctor_get(v___x_2156_, 0);
lean_inc_ref_n(v_env_2157_, 2);
lean_dec(v___x_2156_);
lean_inc_n(v_declName_2145_, 2);
v___x_2158_ = lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg(v_declName_2145_, v___y_2147_);
v_a_2159_ = lean_ctor_get(v___x_2158_, 0);
lean_inc(v_a_2159_);
lean_dec_ref(v___x_2158_);
v___x_2169_ = l_Lean_isAuxRecursor(v_env_2157_, v_declName_2145_);
if (v___x_2169_ == 0)
{
uint8_t v___x_2170_; 
lean_inc(v_declName_2145_);
v___x_2170_ = l_Lean_isNoConfusion(v_env_2157_, v_declName_2145_);
v___y_2165_ = v___x_2170_;
goto v___jp_2164_;
}
else
{
lean_dec_ref(v_env_2157_);
v___y_2165_ = v___x_2169_;
goto v___jp_2164_;
}
v___jp_2149_:
{
if (lean_obj_tag(v_ranges_2150_) == 0)
{
lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; 
v___x_2151_ = l_Lean_builtinDeclRanges;
v___x_2152_ = lean_st_ref_get(v___x_2151_);
v___x_2153_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___x_2152_, v_declName_2145_);
lean_dec(v_declName_2145_);
lean_dec(v___x_2152_);
v___x_2154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2154_, 0, v___x_2153_);
return v___x_2154_;
}
else
{
lean_object* v___x_2155_; 
lean_dec(v_declName_2145_);
v___x_2155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2155_, 0, v_ranges_2150_);
return v___x_2155_;
}
}
v___jp_2160_:
{
lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v_a_2163_; 
v___x_2161_ = l_Lean_Name_getPrefix(v_declName_2145_);
v___x_2162_ = lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg(v___x_2161_, v___y_2147_);
v_a_2163_ = lean_ctor_get(v___x_2162_, 0);
lean_inc(v_a_2163_);
lean_dec_ref(v___x_2162_);
v_ranges_2150_ = v_a_2163_;
goto v___jp_2149_;
}
v___jp_2164_:
{
if (v___y_2165_ == 0)
{
uint8_t v___x_2166_; 
v___x_2166_ = lean_unbox(v_a_2159_);
lean_dec(v_a_2159_);
if (v___x_2166_ == 0)
{
lean_object* v___x_2167_; lean_object* v_a_2168_; 
lean_inc(v_declName_2145_);
v___x_2167_ = lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg(v_declName_2145_, v___y_2147_);
v_a_2168_ = lean_ctor_get(v___x_2167_, 0);
lean_inc(v_a_2168_);
lean_dec_ref(v___x_2167_);
v_ranges_2150_ = v_a_2168_;
goto v___jp_2149_;
}
else
{
goto v___jp_2160_;
}
}
else
{
lean_dec(v_a_2159_);
goto v___jp_2160_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0___boxed(lean_object* v_declName_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_){
_start:
{
lean_object* v_res_2175_; 
v_res_2175_ = lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0(v_declName_2171_, v___y_2172_, v___y_2173_);
lean_dec(v___y_2173_);
lean_dec_ref(v___y_2172_);
return v_res_2175_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg(lean_object* v_as_2176_, size_t v_sz_2177_, size_t v_i_2178_, lean_object* v_b_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_){
_start:
{
uint8_t v___x_2183_; 
v___x_2183_ = lean_usize_dec_lt(v_i_2178_, v_sz_2177_);
if (v___x_2183_ == 0)
{
lean_object* v___x_2184_; 
v___x_2184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2184_, 0, v_b_2179_);
return v___x_2184_;
}
else
{
lean_object* v_a_2185_; lean_object* v_fst_2186_; lean_object* v___x_2187_; 
v_a_2185_ = lean_array_uget_borrowed(v_as_2176_, v_i_2178_);
v_fst_2186_ = lean_ctor_get(v_a_2185_, 0);
lean_inc(v_fst_2186_);
v___x_2187_ = lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0(v_fst_2186_, v___y_2180_, v___y_2181_);
if (lean_obj_tag(v___x_2187_) == 0)
{
lean_object* v_a_2188_; lean_object* v_a_2190_; 
v_a_2188_ = lean_ctor_get(v___x_2187_, 0);
lean_inc(v_a_2188_);
lean_dec_ref_known(v___x_2187_, 1);
if (lean_obj_tag(v_a_2188_) == 1)
{
lean_object* v_val_2194_; lean_object* v_range_2195_; lean_object* v_pos_2196_; lean_object* v_line_2197_; lean_object* v___x_2198_; 
v_val_2194_ = lean_ctor_get(v_a_2188_, 0);
lean_inc(v_val_2194_);
lean_dec_ref_known(v_a_2188_, 1);
v_range_2195_ = lean_ctor_get(v_val_2194_, 0);
lean_inc_ref(v_range_2195_);
lean_dec(v_val_2194_);
v_pos_2196_ = lean_ctor_get(v_range_2195_, 0);
lean_inc_ref(v_pos_2196_);
lean_dec_ref(v_range_2195_);
v_line_2197_ = lean_ctor_get(v_pos_2196_, 0);
lean_inc(v_line_2197_);
lean_dec_ref(v_pos_2196_);
lean_inc(v_fst_2186_);
v___x_2198_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(v_b_2179_, v_fst_2186_, v_line_2197_);
v_a_2190_ = v___x_2198_;
goto v___jp_2189_;
}
else
{
lean_dec(v_a_2188_);
v_a_2190_ = v_b_2179_;
goto v___jp_2189_;
}
v___jp_2189_:
{
size_t v___x_2191_; size_t v___x_2192_; 
v___x_2191_ = ((size_t)1ULL);
v___x_2192_ = lean_usize_add(v_i_2178_, v___x_2191_);
v_i_2178_ = v___x_2192_;
v_b_2179_ = v_a_2190_;
goto _start;
}
}
else
{
lean_object* v_a_2199_; lean_object* v___x_2201_; uint8_t v_isShared_2202_; uint8_t v_isSharedCheck_2206_; 
lean_dec_ref(v_b_2179_);
v_a_2199_ = lean_ctor_get(v___x_2187_, 0);
v_isSharedCheck_2206_ = !lean_is_exclusive(v___x_2187_);
if (v_isSharedCheck_2206_ == 0)
{
v___x_2201_ = v___x_2187_;
v_isShared_2202_ = v_isSharedCheck_2206_;
goto v_resetjp_2200_;
}
else
{
lean_inc(v_a_2199_);
lean_dec(v___x_2187_);
v___x_2201_ = lean_box(0);
v_isShared_2202_ = v_isSharedCheck_2206_;
goto v_resetjp_2200_;
}
v_resetjp_2200_:
{
lean_object* v___x_2204_; 
if (v_isShared_2202_ == 0)
{
v___x_2204_ = v___x_2201_;
goto v_reusejp_2203_;
}
else
{
lean_object* v_reuseFailAlloc_2205_; 
v_reuseFailAlloc_2205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2205_, 0, v_a_2199_);
v___x_2204_ = v_reuseFailAlloc_2205_;
goto v_reusejp_2203_;
}
v_reusejp_2203_:
{
return v___x_2204_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg___boxed(lean_object* v_as_2207_, lean_object* v_sz_2208_, lean_object* v_i_2209_, lean_object* v_b_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_, lean_object* v___y_2213_){
_start:
{
size_t v_sz_boxed_2214_; size_t v_i_boxed_2215_; lean_object* v_res_2216_; 
v_sz_boxed_2214_ = lean_unbox_usize(v_sz_2208_);
lean_dec(v_sz_2208_);
v_i_boxed_2215_ = lean_unbox_usize(v_i_2209_);
lean_dec(v_i_2209_);
v_res_2216_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg(v_as_2207_, v_sz_boxed_2214_, v_i_boxed_2215_, v_b_2210_, v___y_2211_, v___y_2212_);
lean_dec(v___y_2212_);
lean_dec_ref(v___y_2211_);
lean_dec_ref(v_as_2207_);
return v_res_2216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg(lean_object* v_x_2217_, lean_object* v_x_2218_){
_start:
{
if (lean_obj_tag(v_x_2218_) == 0)
{
return v_x_2217_;
}
else
{
lean_object* v_key_2219_; lean_object* v_value_2220_; lean_object* v_tail_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; 
v_key_2219_ = lean_ctor_get(v_x_2218_, 0);
v_value_2220_ = lean_ctor_get(v_x_2218_, 1);
v_tail_2221_ = lean_ctor_get(v_x_2218_, 2);
lean_inc(v_value_2220_);
lean_inc(v_key_2219_);
v___x_2222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2222_, 0, v_key_2219_);
lean_ctor_set(v___x_2222_, 1, v_value_2220_);
v___x_2223_ = lean_array_push(v_x_2217_, v___x_2222_);
v_x_2217_ = v___x_2223_;
v_x_2218_ = v_tail_2221_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg___boxed(lean_object* v_x_2225_, lean_object* v_x_2226_){
_start:
{
lean_object* v_res_2227_; 
v_res_2227_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg(v_x_2225_, v_x_2226_);
lean_dec(v_x_2226_);
return v_res_2227_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg(lean_object* v_as_2228_, size_t v_i_2229_, size_t v_stop_2230_, lean_object* v_b_2231_){
_start:
{
uint8_t v___x_2232_; 
v___x_2232_ = lean_usize_dec_eq(v_i_2229_, v_stop_2230_);
if (v___x_2232_ == 0)
{
lean_object* v___x_2233_; lean_object* v___x_2234_; size_t v___x_2235_; size_t v___x_2236_; 
v___x_2233_ = lean_array_uget_borrowed(v_as_2228_, v_i_2229_);
v___x_2234_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg(v_b_2231_, v___x_2233_);
v___x_2235_ = ((size_t)1ULL);
v___x_2236_ = lean_usize_add(v_i_2229_, v___x_2235_);
v_i_2229_ = v___x_2236_;
v_b_2231_ = v___x_2234_;
goto _start;
}
else
{
return v_b_2231_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg___boxed(lean_object* v_as_2238_, lean_object* v_i_2239_, lean_object* v_stop_2240_, lean_object* v_b_2241_){
_start:
{
size_t v_i_boxed_2242_; size_t v_stop_boxed_2243_; lean_object* v_res_2244_; 
v_i_boxed_2242_ = lean_unbox_usize(v_i_2239_);
lean_dec(v_i_2239_);
v_stop_boxed_2243_ = lean_unbox_usize(v_stop_2240_);
lean_dec(v_stop_2240_);
v_res_2244_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg(v_as_2238_, v_i_boxed_2242_, v_stop_boxed_2243_, v_b_2241_);
lean_dec_ref(v_as_2238_);
return v_res_2244_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg(lean_object* v_a_2245_, lean_object* v_fallback_2246_, lean_object* v_x_2247_){
_start:
{
if (lean_obj_tag(v_x_2247_) == 0)
{
lean_inc(v_fallback_2246_);
return v_fallback_2246_;
}
else
{
lean_object* v_key_2248_; lean_object* v_value_2249_; lean_object* v_tail_2250_; uint8_t v___x_2251_; 
v_key_2248_ = lean_ctor_get(v_x_2247_, 0);
v_value_2249_ = lean_ctor_get(v_x_2247_, 1);
v_tail_2250_ = lean_ctor_get(v_x_2247_, 2);
v___x_2251_ = lean_name_eq(v_key_2248_, v_a_2245_);
if (v___x_2251_ == 0)
{
v_x_2247_ = v_tail_2250_;
goto _start;
}
else
{
lean_inc(v_value_2249_);
return v_value_2249_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg___boxed(lean_object* v_a_2253_, lean_object* v_fallback_2254_, lean_object* v_x_2255_){
_start:
{
lean_object* v_res_2256_; 
v_res_2256_ = lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg(v_a_2253_, v_fallback_2254_, v_x_2255_);
lean_dec(v_x_2255_);
lean_dec(v_fallback_2254_);
lean_dec(v_a_2253_);
return v_res_2256_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(lean_object* v_m_2257_, lean_object* v_a_2258_, lean_object* v_fallback_2259_){
_start:
{
lean_object* v_buckets_2260_; lean_object* v___x_2261_; uint64_t v___y_2263_; 
v_buckets_2260_ = lean_ctor_get(v_m_2257_, 1);
v___x_2261_ = lean_array_get_size(v_buckets_2260_);
if (lean_obj_tag(v_a_2258_) == 0)
{
uint64_t v___x_2277_; 
v___x_2277_ = 1723ULL;
v___y_2263_ = v___x_2277_;
goto v___jp_2262_;
}
else
{
uint64_t v_hash_2278_; 
v_hash_2278_ = lean_ctor_get_uint64(v_a_2258_, sizeof(void*)*2);
v___y_2263_ = v_hash_2278_;
goto v___jp_2262_;
}
v___jp_2262_:
{
uint64_t v___x_2264_; uint64_t v___x_2265_; uint64_t v_fold_2266_; uint64_t v___x_2267_; uint64_t v___x_2268_; uint64_t v___x_2269_; size_t v___x_2270_; size_t v___x_2271_; size_t v___x_2272_; size_t v___x_2273_; size_t v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; 
v___x_2264_ = 32ULL;
v___x_2265_ = lean_uint64_shift_right(v___y_2263_, v___x_2264_);
v_fold_2266_ = lean_uint64_xor(v___y_2263_, v___x_2265_);
v___x_2267_ = 16ULL;
v___x_2268_ = lean_uint64_shift_right(v_fold_2266_, v___x_2267_);
v___x_2269_ = lean_uint64_xor(v_fold_2266_, v___x_2268_);
v___x_2270_ = lean_uint64_to_usize(v___x_2269_);
v___x_2271_ = lean_usize_of_nat(v___x_2261_);
v___x_2272_ = ((size_t)1ULL);
v___x_2273_ = lean_usize_sub(v___x_2271_, v___x_2272_);
v___x_2274_ = lean_usize_land(v___x_2270_, v___x_2273_);
v___x_2275_ = lean_array_uget_borrowed(v_buckets_2260_, v___x_2274_);
v___x_2276_ = lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg(v_a_2258_, v_fallback_2259_, v___x_2275_);
return v___x_2276_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg___boxed(lean_object* v_m_2279_, lean_object* v_a_2280_, lean_object* v_fallback_2281_){
_start:
{
lean_object* v_res_2282_; 
v_res_2282_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(v_m_2279_, v_a_2280_, v_fallback_2281_);
lean_dec(v_fallback_2281_);
lean_dec(v_a_2280_);
lean_dec_ref(v_m_2279_);
return v_res_2282_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0(lean_object* v_a_2283_, lean_object* v_x_2284_, lean_object* v_x_2285_){
_start:
{
lean_object* v_fst_2286_; lean_object* v_fst_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; uint8_t v___x_2291_; 
v_fst_2286_ = lean_ctor_get(v_x_2284_, 0);
v_fst_2287_ = lean_ctor_get(v_x_2285_, 0);
v___x_2288_ = lean_unsigned_to_nat(0u);
v___x_2289_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(v_a_2283_, v_fst_2286_, v___x_2288_);
v___x_2290_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(v_a_2283_, v_fst_2287_, v___x_2288_);
v___x_2291_ = lean_nat_dec_lt(v___x_2289_, v___x_2290_);
lean_dec(v___x_2290_);
lean_dec(v___x_2289_);
return v___x_2291_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0___boxed(lean_object* v_a_2292_, lean_object* v_x_2293_, lean_object* v_x_2294_){
_start:
{
uint8_t v_res_2295_; lean_object* v_r_2296_; 
v_res_2295_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0(v_a_2292_, v_x_2293_, v_x_2294_);
lean_dec_ref(v_x_2294_);
lean_dec_ref(v_x_2293_);
lean_dec_ref(v_a_2292_);
v_r_2296_ = lean_box(v_res_2295_);
return v_r_2296_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg(lean_object* v_a_2297_, lean_object* v_hi_2298_, lean_object* v_pivot_2299_, lean_object* v_as_2300_, lean_object* v_i_2301_, lean_object* v_k_2302_){
_start:
{
uint8_t v___x_2303_; 
v___x_2303_ = lean_nat_dec_lt(v_k_2302_, v_hi_2298_);
if (v___x_2303_ == 0)
{
lean_object* v___x_2304_; lean_object* v___x_2305_; 
lean_dec(v_k_2302_);
v___x_2304_ = lean_array_fswap(v_as_2300_, v_i_2301_, v_hi_2298_);
v___x_2305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2305_, 0, v_i_2301_);
lean_ctor_set(v___x_2305_, 1, v___x_2304_);
return v___x_2305_;
}
else
{
lean_object* v___x_2306_; lean_object* v_fst_2307_; lean_object* v_fst_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; uint8_t v___x_2312_; 
v___x_2306_ = lean_array_fget_borrowed(v_as_2300_, v_k_2302_);
v_fst_2307_ = lean_ctor_get(v___x_2306_, 0);
v_fst_2308_ = lean_ctor_get(v_pivot_2299_, 0);
v___x_2309_ = lean_unsigned_to_nat(0u);
v___x_2310_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(v_a_2297_, v_fst_2307_, v___x_2309_);
v___x_2311_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(v_a_2297_, v_fst_2308_, v___x_2309_);
v___x_2312_ = lean_nat_dec_lt(v___x_2310_, v___x_2311_);
lean_dec(v___x_2311_);
lean_dec(v___x_2310_);
if (v___x_2312_ == 0)
{
lean_object* v___x_2313_; lean_object* v___x_2314_; 
v___x_2313_ = lean_unsigned_to_nat(1u);
v___x_2314_ = lean_nat_add(v_k_2302_, v___x_2313_);
lean_dec(v_k_2302_);
v_k_2302_ = v___x_2314_;
goto _start;
}
else
{
lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; 
v___x_2316_ = lean_array_fswap(v_as_2300_, v_i_2301_, v_k_2302_);
v___x_2317_ = lean_unsigned_to_nat(1u);
v___x_2318_ = lean_nat_add(v_i_2301_, v___x_2317_);
lean_dec(v_i_2301_);
v___x_2319_ = lean_nat_add(v_k_2302_, v___x_2317_);
lean_dec(v_k_2302_);
v_as_2300_ = v___x_2316_;
v_i_2301_ = v___x_2318_;
v_k_2302_ = v___x_2319_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg___boxed(lean_object* v_a_2321_, lean_object* v_hi_2322_, lean_object* v_pivot_2323_, lean_object* v_as_2324_, lean_object* v_i_2325_, lean_object* v_k_2326_){
_start:
{
lean_object* v_res_2327_; 
v_res_2327_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg(v_a_2321_, v_hi_2322_, v_pivot_2323_, v_as_2324_, v_i_2325_, v_k_2326_);
lean_dec_ref(v_pivot_2323_);
lean_dec(v_hi_2322_);
lean_dec_ref(v_a_2321_);
return v_res_2327_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg(lean_object* v_a_2328_, lean_object* v_n_2329_, lean_object* v_as_2330_, lean_object* v_lo_2331_, lean_object* v_hi_2332_){
_start:
{
lean_object* v___y_2334_; uint8_t v___x_2344_; 
v___x_2344_ = lean_nat_dec_lt(v_lo_2331_, v_hi_2332_);
if (v___x_2344_ == 0)
{
lean_dec(v_lo_2331_);
return v_as_2330_;
}
else
{
lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v_mid_2347_; lean_object* v___y_2349_; lean_object* v___y_2355_; lean_object* v___x_2360_; lean_object* v___x_2361_; uint8_t v___x_2362_; 
v___x_2345_ = lean_nat_add(v_lo_2331_, v_hi_2332_);
v___x_2346_ = lean_unsigned_to_nat(1u);
v_mid_2347_ = lean_nat_shiftr(v___x_2345_, v___x_2346_);
lean_dec(v___x_2345_);
v___x_2360_ = lean_array_fget_borrowed(v_as_2330_, v_mid_2347_);
v___x_2361_ = lean_array_fget_borrowed(v_as_2330_, v_lo_2331_);
v___x_2362_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0(v_a_2328_, v___x_2360_, v___x_2361_);
if (v___x_2362_ == 0)
{
v___y_2355_ = v_as_2330_;
goto v___jp_2354_;
}
else
{
lean_object* v___x_2363_; 
v___x_2363_ = lean_array_fswap(v_as_2330_, v_lo_2331_, v_mid_2347_);
v___y_2355_ = v___x_2363_;
goto v___jp_2354_;
}
v___jp_2348_:
{
lean_object* v___x_2350_; lean_object* v___x_2351_; uint8_t v___x_2352_; 
v___x_2350_ = lean_array_fget_borrowed(v___y_2349_, v_mid_2347_);
v___x_2351_ = lean_array_fget_borrowed(v___y_2349_, v_hi_2332_);
v___x_2352_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0(v_a_2328_, v___x_2350_, v___x_2351_);
if (v___x_2352_ == 0)
{
lean_dec(v_mid_2347_);
v___y_2334_ = v___y_2349_;
goto v___jp_2333_;
}
else
{
lean_object* v___x_2353_; 
v___x_2353_ = lean_array_fswap(v___y_2349_, v_mid_2347_, v_hi_2332_);
lean_dec(v_mid_2347_);
v___y_2334_ = v___x_2353_;
goto v___jp_2333_;
}
}
v___jp_2354_:
{
lean_object* v___x_2356_; lean_object* v___x_2357_; uint8_t v___x_2358_; 
v___x_2356_ = lean_array_fget_borrowed(v___y_2355_, v_hi_2332_);
v___x_2357_ = lean_array_fget_borrowed(v___y_2355_, v_lo_2331_);
v___x_2358_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___lam__0(v_a_2328_, v___x_2356_, v___x_2357_);
if (v___x_2358_ == 0)
{
v___y_2349_ = v___y_2355_;
goto v___jp_2348_;
}
else
{
lean_object* v___x_2359_; 
v___x_2359_ = lean_array_fswap(v___y_2355_, v_lo_2331_, v_hi_2332_);
v___y_2349_ = v___x_2359_;
goto v___jp_2348_;
}
}
}
v___jp_2333_:
{
lean_object* v_pivot_2335_; lean_object* v___x_2336_; lean_object* v_fst_2337_; lean_object* v_snd_2338_; uint8_t v___x_2339_; 
v_pivot_2335_ = lean_array_fget(v___y_2334_, v_hi_2332_);
lean_inc_n(v_lo_2331_, 2);
v___x_2336_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg(v_a_2328_, v_hi_2332_, v_pivot_2335_, v___y_2334_, v_lo_2331_, v_lo_2331_);
lean_dec(v_pivot_2335_);
v_fst_2337_ = lean_ctor_get(v___x_2336_, 0);
lean_inc(v_fst_2337_);
v_snd_2338_ = lean_ctor_get(v___x_2336_, 1);
lean_inc(v_snd_2338_);
lean_dec_ref(v___x_2336_);
v___x_2339_ = lean_nat_dec_le(v_hi_2332_, v_fst_2337_);
if (v___x_2339_ == 0)
{
lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; 
v___x_2340_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg(v_a_2328_, v_n_2329_, v_snd_2338_, v_lo_2331_, v_fst_2337_);
v___x_2341_ = lean_unsigned_to_nat(1u);
v___x_2342_ = lean_nat_add(v_fst_2337_, v___x_2341_);
lean_dec(v_fst_2337_);
v_as_2330_ = v___x_2340_;
v_lo_2331_ = v___x_2342_;
goto _start;
}
else
{
lean_dec(v_fst_2337_);
lean_dec(v_lo_2331_);
return v_snd_2338_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg___boxed(lean_object* v_a_2364_, lean_object* v_n_2365_, lean_object* v_as_2366_, lean_object* v_lo_2367_, lean_object* v_hi_2368_){
_start:
{
lean_object* v_res_2369_; 
v_res_2369_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg(v_a_2364_, v_n_2365_, v_as_2366_, v_lo_2367_, v_hi_2368_);
lean_dec(v_hi_2368_);
lean_dec(v_n_2365_);
lean_dec_ref(v_a_2364_);
return v_res_2369_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults___redArg(lean_object* v_results_2370_, lean_object* v_a_2371_, lean_object* v_a_2372_){
_start:
{
lean_object* v___y_2375_; lean_object* v___y_2376_; lean_object* v___y_2377_; lean_object* v___y_2378_; lean_object* v___y_2379_; lean_object* v___y_2383_; lean_object* v___y_2384_; lean_object* v___y_2385_; lean_object* v___y_2386_; lean_object* v___y_2387_; lean_object* v_size_2389_; lean_object* v_buckets_2390_; lean_object* v___x_2391_; lean_object* v_key_2392_; lean_object* v___y_2394_; lean_object* v___x_2419_; lean_object* v___x_2420_; uint8_t v___x_2421_; 
v_size_2389_ = lean_ctor_get(v_results_2370_, 0);
v_buckets_2390_ = lean_ctor_get(v_results_2370_, 1);
v___x_2391_ = lean_unsigned_to_nat(0u);
v_key_2392_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7);
v___x_2419_ = lean_mk_empty_array_with_capacity(v_size_2389_);
v___x_2420_ = lean_array_get_size(v_buckets_2390_);
v___x_2421_ = lean_nat_dec_lt(v___x_2391_, v___x_2420_);
if (v___x_2421_ == 0)
{
v___y_2394_ = v___x_2419_;
goto v___jp_2393_;
}
else
{
uint8_t v___x_2422_; 
v___x_2422_ = lean_nat_dec_le(v___x_2420_, v___x_2420_);
if (v___x_2422_ == 0)
{
if (v___x_2421_ == 0)
{
v___y_2394_ = v___x_2419_;
goto v___jp_2393_;
}
else
{
size_t v___x_2423_; size_t v___x_2424_; lean_object* v___x_2425_; 
v___x_2423_ = ((size_t)0ULL);
v___x_2424_ = lean_usize_of_nat(v___x_2420_);
v___x_2425_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg(v_buckets_2390_, v___x_2423_, v___x_2424_, v___x_2419_);
v___y_2394_ = v___x_2425_;
goto v___jp_2393_;
}
}
else
{
size_t v___x_2426_; size_t v___x_2427_; lean_object* v___x_2428_; 
v___x_2426_ = ((size_t)0ULL);
v___x_2427_ = lean_usize_of_nat(v___x_2420_);
v___x_2428_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg(v_buckets_2390_, v___x_2426_, v___x_2427_, v___x_2419_);
v___y_2394_ = v___x_2428_;
goto v___jp_2393_;
}
}
v___jp_2374_:
{
lean_object* v___x_2380_; lean_object* v___x_2381_; 
v___x_2380_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg(v___y_2378_, v___y_2376_, v___y_2377_, v___y_2375_, v___y_2379_);
lean_dec(v___y_2379_);
lean_dec(v___y_2376_);
lean_dec_ref(v___y_2378_);
v___x_2381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2381_, 0, v___x_2380_);
return v___x_2381_;
}
v___jp_2382_:
{
uint8_t v___x_2388_; 
v___x_2388_ = lean_nat_dec_le(v___y_2387_, v___y_2385_);
if (v___x_2388_ == 0)
{
lean_dec(v___y_2385_);
lean_inc(v___y_2387_);
v___y_2375_ = v___y_2387_;
v___y_2376_ = v___y_2383_;
v___y_2377_ = v___y_2384_;
v___y_2378_ = v___y_2386_;
v___y_2379_ = v___y_2387_;
goto v___jp_2374_;
}
else
{
v___y_2375_ = v___y_2387_;
v___y_2376_ = v___y_2383_;
v___y_2377_ = v___y_2384_;
v___y_2378_ = v___y_2386_;
v___y_2379_ = v___y_2385_;
goto v___jp_2374_;
}
}
v___jp_2393_:
{
size_t v_sz_2395_; size_t v___x_2396_; lean_object* v___x_2397_; 
v_sz_2395_ = lean_array_size(v___y_2394_);
v___x_2396_ = ((size_t)0ULL);
v___x_2397_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg(v___y_2394_, v_sz_2395_, v___x_2396_, v_key_2392_, v_a_2371_, v_a_2372_);
if (lean_obj_tag(v___x_2397_) == 0)
{
lean_object* v_a_2398_; lean_object* v___x_2400_; uint8_t v_isShared_2401_; uint8_t v_isSharedCheck_2410_; 
v_a_2398_ = lean_ctor_get(v___x_2397_, 0);
v_isSharedCheck_2410_ = !lean_is_exclusive(v___x_2397_);
if (v_isSharedCheck_2410_ == 0)
{
v___x_2400_ = v___x_2397_;
v_isShared_2401_ = v_isSharedCheck_2410_;
goto v_resetjp_2399_;
}
else
{
lean_inc(v_a_2398_);
lean_dec(v___x_2397_);
v___x_2400_ = lean_box(0);
v_isShared_2401_ = v_isSharedCheck_2410_;
goto v_resetjp_2399_;
}
v_resetjp_2399_:
{
lean_object* v___x_2402_; uint8_t v___x_2403_; 
v___x_2402_ = lean_array_get_size(v___y_2394_);
v___x_2403_ = lean_nat_dec_eq(v___x_2402_, v___x_2391_);
if (v___x_2403_ == 0)
{
lean_object* v___x_2404_; lean_object* v___x_2405_; uint8_t v___x_2406_; 
lean_del_object(v___x_2400_);
v___x_2404_ = lean_unsigned_to_nat(1u);
v___x_2405_ = lean_nat_sub(v___x_2402_, v___x_2404_);
v___x_2406_ = lean_nat_dec_le(v___x_2391_, v___x_2405_);
if (v___x_2406_ == 0)
{
lean_inc(v___x_2405_);
v___y_2383_ = v___x_2402_;
v___y_2384_ = v___y_2394_;
v___y_2385_ = v___x_2405_;
v___y_2386_ = v_a_2398_;
v___y_2387_ = v___x_2405_;
goto v___jp_2382_;
}
else
{
v___y_2383_ = v___x_2402_;
v___y_2384_ = v___y_2394_;
v___y_2385_ = v___x_2405_;
v___y_2386_ = v_a_2398_;
v___y_2387_ = v___x_2391_;
goto v___jp_2382_;
}
}
else
{
lean_object* v___x_2408_; 
lean_dec(v_a_2398_);
if (v_isShared_2401_ == 0)
{
lean_ctor_set(v___x_2400_, 0, v___y_2394_);
v___x_2408_ = v___x_2400_;
goto v_reusejp_2407_;
}
else
{
lean_object* v_reuseFailAlloc_2409_; 
v_reuseFailAlloc_2409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2409_, 0, v___y_2394_);
v___x_2408_ = v_reuseFailAlloc_2409_;
goto v_reusejp_2407_;
}
v_reusejp_2407_:
{
return v___x_2408_;
}
}
}
}
else
{
lean_object* v_a_2411_; lean_object* v___x_2413_; uint8_t v_isShared_2414_; uint8_t v_isSharedCheck_2418_; 
lean_dec_ref(v___y_2394_);
v_a_2411_ = lean_ctor_get(v___x_2397_, 0);
v_isSharedCheck_2418_ = !lean_is_exclusive(v___x_2397_);
if (v_isSharedCheck_2418_ == 0)
{
v___x_2413_ = v___x_2397_;
v_isShared_2414_ = v_isSharedCheck_2418_;
goto v_resetjp_2412_;
}
else
{
lean_inc(v_a_2411_);
lean_dec(v___x_2397_);
v___x_2413_ = lean_box(0);
v_isShared_2414_ = v_isSharedCheck_2418_;
goto v_resetjp_2412_;
}
v_resetjp_2412_:
{
lean_object* v___x_2416_; 
if (v_isShared_2414_ == 0)
{
v___x_2416_ = v___x_2413_;
goto v_reusejp_2415_;
}
else
{
lean_object* v_reuseFailAlloc_2417_; 
v_reuseFailAlloc_2417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2417_, 0, v_a_2411_);
v___x_2416_ = v_reuseFailAlloc_2417_;
goto v_reusejp_2415_;
}
v_reusejp_2415_:
{
return v___x_2416_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults___redArg___boxed(lean_object* v_results_2429_, lean_object* v_a_2430_, lean_object* v_a_2431_, lean_object* v_a_2432_){
_start:
{
lean_object* v_res_2433_; 
v_res_2433_ = lp_batteries_Batteries_Tactic_Lint_sortResults___redArg(v_results_2429_, v_a_2430_, v_a_2431_);
lean_dec(v_a_2431_);
lean_dec_ref(v_a_2430_);
lean_dec_ref(v_results_2429_);
return v_res_2433_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults(lean_object* v_00_u03b1_2434_, lean_object* v_results_2435_, lean_object* v_a_2436_, lean_object* v_a_2437_){
_start:
{
lean_object* v___x_2439_; 
v___x_2439_ = lp_batteries_Batteries_Tactic_Lint_sortResults___redArg(v_results_2435_, v_a_2436_, v_a_2437_);
return v___x_2439_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_sortResults___boxed(lean_object* v_00_u03b1_2440_, lean_object* v_results_2441_, lean_object* v_a_2442_, lean_object* v_a_2443_, lean_object* v_a_2444_){
_start:
{
lean_object* v_res_2445_; 
v_res_2445_ = lp_batteries_Batteries_Tactic_Lint_sortResults(v_00_u03b1_2440_, v_results_2441_, v_a_2442_, v_a_2443_);
lean_dec(v_a_2443_);
lean_dec_ref(v_a_2442_);
lean_dec_ref(v_results_2441_);
return v_res_2445_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0(lean_object* v_declName_2446_, lean_object* v___y_2447_, lean_object* v___y_2448_){
_start:
{
lean_object* v___x_2450_; 
v___x_2450_ = lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___redArg(v_declName_2446_, v___y_2448_);
return v___x_2450_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0___boxed(lean_object* v_declName_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_){
_start:
{
lean_object* v_res_2455_; 
v_res_2455_ = lp_batteries_Lean_isRec___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__0(v_declName_2451_, v___y_2452_, v___y_2453_);
lean_dec(v___y_2453_);
lean_dec_ref(v___y_2452_);
return v_res_2455_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1(lean_object* v_declName_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_){
_start:
{
lean_object* v___x_2460_; 
v___x_2460_ = lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___redArg(v_declName_2456_, v___y_2458_);
return v___x_2460_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1___boxed(lean_object* v_declName_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_){
_start:
{
lean_object* v_res_2465_; 
v_res_2465_ = lp_batteries_Lean_findDeclarationRangesCore_x3f___at___00Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0_spec__1(v_declName_2461_, v___y_2462_, v___y_2463_);
lean_dec(v___y_2463_);
lean_dec_ref(v___y_2462_);
return v_res_2465_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1(lean_object* v_00_u03b1_2466_, lean_object* v_as_2467_, size_t v_sz_2468_, size_t v_i_2469_, lean_object* v_b_2470_, lean_object* v___y_2471_, lean_object* v___y_2472_){
_start:
{
lean_object* v___x_2474_; 
v___x_2474_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___redArg(v_as_2467_, v_sz_2468_, v_i_2469_, v_b_2470_, v___y_2471_, v___y_2472_);
return v___x_2474_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1___boxed(lean_object* v_00_u03b1_2475_, lean_object* v_as_2476_, lean_object* v_sz_2477_, lean_object* v_i_2478_, lean_object* v_b_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_){
_start:
{
size_t v_sz_boxed_2483_; size_t v_i_boxed_2484_; lean_object* v_res_2485_; 
v_sz_boxed_2483_ = lean_unbox_usize(v_sz_2477_);
lean_dec(v_sz_2477_);
v_i_boxed_2484_ = lean_unbox_usize(v_i_2478_);
lean_dec(v_i_2478_);
v_res_2485_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint_sortResults_spec__1(v_00_u03b1_2475_, v_as_2476_, v_sz_boxed_2483_, v_i_boxed_2484_, v_b_2479_, v___y_2480_, v___y_2481_);
lean_dec(v___y_2481_);
lean_dec_ref(v___y_2480_);
lean_dec_ref(v_as_2476_);
return v_res_2485_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2(lean_object* v_00_u03b2_2486_, lean_object* v_m_2487_, lean_object* v_a_2488_, lean_object* v_fallback_2489_){
_start:
{
lean_object* v___x_2490_; 
v___x_2490_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___redArg(v_m_2487_, v_a_2488_, v_fallback_2489_);
return v___x_2490_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2___boxed(lean_object* v_00_u03b2_2491_, lean_object* v_m_2492_, lean_object* v_a_2493_, lean_object* v_fallback_2494_){
_start:
{
lean_object* v_res_2495_; 
v_res_2495_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2(v_00_u03b2_2491_, v_m_2492_, v_a_2493_, v_fallback_2494_);
lean_dec(v_fallback_2494_);
lean_dec(v_a_2493_);
lean_dec_ref(v_m_2492_);
return v_res_2495_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3(lean_object* v_00_u03b1_2496_, lean_object* v_a_2497_, lean_object* v_n_2498_, lean_object* v_as_2499_, lean_object* v_lo_2500_, lean_object* v_hi_2501_, lean_object* v_w_2502_, lean_object* v_hlo_2503_, lean_object* v_hhi_2504_){
_start:
{
lean_object* v___x_2505_; 
v___x_2505_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___redArg(v_a_2497_, v_n_2498_, v_as_2499_, v_lo_2500_, v_hi_2501_);
return v___x_2505_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3___boxed(lean_object* v_00_u03b1_2506_, lean_object* v_a_2507_, lean_object* v_n_2508_, lean_object* v_as_2509_, lean_object* v_lo_2510_, lean_object* v_hi_2511_, lean_object* v_w_2512_, lean_object* v_hlo_2513_, lean_object* v_hhi_2514_){
_start:
{
lean_object* v_res_2515_; 
v_res_2515_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3(v_00_u03b1_2506_, v_a_2507_, v_n_2508_, v_as_2509_, v_lo_2510_, v_hi_2511_, v_w_2512_, v_hlo_2513_, v_hhi_2514_);
lean_dec(v_hi_2511_);
lean_dec(v_n_2508_);
lean_dec_ref(v_a_2507_);
return v_res_2515_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4(lean_object* v_00_u03b1_2516_, lean_object* v_x_2517_, lean_object* v_x_2518_){
_start:
{
lean_object* v___x_2519_; 
v___x_2519_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___redArg(v_x_2517_, v_x_2518_);
return v___x_2519_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4___boxed(lean_object* v_00_u03b1_2520_, lean_object* v_x_2521_, lean_object* v_x_2522_){
_start:
{
lean_object* v_res_2523_; 
v_res_2523_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_sortResults_spec__4(v_00_u03b1_2520_, v_x_2521_, v_x_2522_);
lean_dec(v_x_2522_);
return v_res_2523_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5(lean_object* v_00_u03b1_2524_, lean_object* v_as_2525_, size_t v_i_2526_, size_t v_stop_2527_, lean_object* v_b_2528_){
_start:
{
lean_object* v___x_2529_; 
v___x_2529_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___redArg(v_as_2525_, v_i_2526_, v_stop_2527_, v_b_2528_);
return v___x_2529_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5___boxed(lean_object* v_00_u03b1_2530_, lean_object* v_as_2531_, lean_object* v_i_2532_, lean_object* v_stop_2533_, lean_object* v_b_2534_){
_start:
{
size_t v_i_boxed_2535_; size_t v_stop_boxed_2536_; lean_object* v_res_2537_; 
v_i_boxed_2535_ = lean_unbox_usize(v_i_2532_);
lean_dec(v_i_2532_);
v_stop_boxed_2536_ = lean_unbox_usize(v_stop_2533_);
lean_dec(v_stop_2533_);
v_res_2537_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_sortResults_spec__5(v_00_u03b1_2530_, v_as_2531_, v_i_boxed_2535_, v_stop_boxed_2536_, v_b_2534_);
lean_dec_ref(v_as_2531_);
return v_res_2537_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4(lean_object* v_00_u03b2_2538_, lean_object* v_a_2539_, lean_object* v_fallback_2540_, lean_object* v_x_2541_){
_start:
{
lean_object* v___x_2542_; 
v___x_2542_ = lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___redArg(v_a_2539_, v_fallback_2540_, v_x_2541_);
return v___x_2542_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4___boxed(lean_object* v_00_u03b2_2543_, lean_object* v_a_2544_, lean_object* v_fallback_2545_, lean_object* v_x_2546_){
_start:
{
lean_object* v_res_2547_; 
v_res_2547_ = lp_batteries_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Batteries_Tactic_Lint_sortResults_spec__2_spec__4(v_00_u03b2_2543_, v_a_2544_, v_fallback_2545_, v_x_2546_);
lean_dec(v_x_2546_);
lean_dec(v_fallback_2545_);
lean_dec(v_a_2544_);
return v_res_2547_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6(lean_object* v_00_u03b1_2548_, lean_object* v_a_2549_, lean_object* v_n_2550_, lean_object* v_lo_2551_, lean_object* v_hi_2552_, lean_object* v_hhi_2553_, lean_object* v_pivot_2554_, lean_object* v_as_2555_, lean_object* v_i_2556_, lean_object* v_k_2557_, lean_object* v_ilo_2558_, lean_object* v_ik_2559_, lean_object* v_w_2560_){
_start:
{
lean_object* v___x_2561_; 
v___x_2561_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___redArg(v_a_2549_, v_hi_2552_, v_pivot_2554_, v_as_2555_, v_i_2556_, v_k_2557_);
return v___x_2561_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6___boxed(lean_object* v_00_u03b1_2562_, lean_object* v_a_2563_, lean_object* v_n_2564_, lean_object* v_lo_2565_, lean_object* v_hi_2566_, lean_object* v_hhi_2567_, lean_object* v_pivot_2568_, lean_object* v_as_2569_, lean_object* v_i_2570_, lean_object* v_k_2571_, lean_object* v_ilo_2572_, lean_object* v_ik_2573_, lean_object* v_w_2574_){
_start:
{
lean_object* v_res_2575_; 
v_res_2575_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_sortResults_spec__3_spec__6(v_00_u03b1_2562_, v_a_2563_, v_n_2564_, v_lo_2565_, v_hi_2566_, v_hhi_2567_, v_pivot_2568_, v_as_2569_, v_i_2570_, v_k_2571_, v_ilo_2572_, v_ik_2573_, v_w_2574_);
lean_dec_ref(v_pivot_2568_);
lean_dec(v_hi_2566_);
lean_dec(v_lo_2565_);
lean_dec(v_n_2564_);
lean_dec_ref(v_a_2563_);
return v_res_2575_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__1(lean_object* v_a_2576_, lean_object* v_a_2577_){
_start:
{
if (lean_obj_tag(v_a_2576_) == 0)
{
lean_object* v___x_2578_; 
v___x_2578_ = l_List_reverse___redArg(v_a_2577_);
return v___x_2578_;
}
else
{
lean_object* v_head_2579_; lean_object* v_tail_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2589_; 
v_head_2579_ = lean_ctor_get(v_a_2576_, 0);
v_tail_2580_ = lean_ctor_get(v_a_2576_, 1);
v_isSharedCheck_2589_ = !lean_is_exclusive(v_a_2576_);
if (v_isSharedCheck_2589_ == 0)
{
v___x_2582_ = v_a_2576_;
v_isShared_2583_ = v_isSharedCheck_2589_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_tail_2580_);
lean_inc(v_head_2579_);
lean_dec(v_a_2576_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2589_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2584_; lean_object* v___x_2586_; 
v___x_2584_ = l_Lean_mkLevelParam(v_head_2579_);
if (v_isShared_2583_ == 0)
{
lean_ctor_set(v___x_2582_, 1, v_a_2577_);
lean_ctor_set(v___x_2582_, 0, v___x_2584_);
v___x_2586_ = v___x_2582_;
goto v_reusejp_2585_;
}
else
{
lean_object* v_reuseFailAlloc_2588_; 
v_reuseFailAlloc_2588_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2588_, 0, v___x_2584_);
lean_ctor_set(v_reuseFailAlloc_2588_, 1, v_a_2577_);
v___x_2586_ = v_reuseFailAlloc_2588_;
goto v_reusejp_2585_;
}
v_reusejp_2585_:
{
v_a_2576_ = v_tail_2580_;
v_a_2577_ = v___x_2586_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_2591_; lean_object* v___x_2592_; 
v___x_2591_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0));
v___x_2592_ = l_Lean_stringToMessageData(v___x_2591_);
return v___x_2592_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_2594_; lean_object* v___x_2595_; 
v___x_2594_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2));
v___x_2595_ = l_Lean_stringToMessageData(v___x_2594_);
return v___x_2595_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_2597_; lean_object* v___x_2598_; 
v___x_2597_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4));
v___x_2598_ = l_Lean_stringToMessageData(v___x_2597_);
return v___x_2598_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7(void){
_start:
{
lean_object* v___x_2600_; lean_object* v___x_2601_; 
v___x_2600_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6));
v___x_2601_ = l_Lean_stringToMessageData(v___x_2600_);
return v___x_2601_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9(void){
_start:
{
lean_object* v___x_2603_; lean_object* v___x_2604_; 
v___x_2603_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8));
v___x_2604_ = l_Lean_stringToMessageData(v___x_2603_);
return v___x_2604_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11(void){
_start:
{
lean_object* v___x_2606_; lean_object* v___x_2607_; 
v___x_2606_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10));
v___x_2607_ = l_Lean_stringToMessageData(v___x_2606_);
return v___x_2607_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13(void){
_start:
{
lean_object* v___x_2609_; lean_object* v___x_2610_; 
v___x_2609_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12));
v___x_2610_ = l_Lean_stringToMessageData(v___x_2609_);
return v___x_2610_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(lean_object* v_msg_2611_, lean_object* v_declHint_2612_, lean_object* v___y_2613_){
_start:
{
lean_object* v___x_2615_; lean_object* v_env_2616_; uint8_t v___x_2617_; 
v___x_2615_ = lean_st_ref_get(v___y_2613_);
v_env_2616_ = lean_ctor_get(v___x_2615_, 0);
lean_inc_ref(v_env_2616_);
lean_dec(v___x_2615_);
v___x_2617_ = l_Lean_Name_isAnonymous(v_declHint_2612_);
if (v___x_2617_ == 0)
{
uint8_t v_isExporting_2618_; 
v_isExporting_2618_ = lean_ctor_get_uint8(v_env_2616_, sizeof(void*)*8);
if (v_isExporting_2618_ == 0)
{
lean_object* v___x_2619_; 
lean_dec_ref(v_env_2616_);
lean_dec(v_declHint_2612_);
v___x_2619_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2619_, 0, v_msg_2611_);
return v___x_2619_;
}
else
{
lean_object* v___x_2620_; uint8_t v___x_2621_; 
lean_inc_ref(v_env_2616_);
v___x_2620_ = l_Lean_Environment_setExporting(v_env_2616_, v___x_2617_);
lean_inc(v_declHint_2612_);
lean_inc_ref(v___x_2620_);
v___x_2621_ = l_Lean_Environment_contains(v___x_2620_, v_declHint_2612_, v_isExporting_2618_);
if (v___x_2621_ == 0)
{
lean_object* v___x_2622_; 
lean_dec_ref(v___x_2620_);
lean_dec_ref(v_env_2616_);
lean_dec(v_declHint_2612_);
v___x_2622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2622_, 0, v_msg_2611_);
return v___x_2622_;
}
else
{
lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v_c_2628_; lean_object* v___x_2629_; 
v___x_2623_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2);
v___x_2624_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5);
v___x_2625_ = l_Lean_Options_empty;
v___x_2626_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2626_, 0, v___x_2620_);
lean_ctor_set(v___x_2626_, 1, v___x_2623_);
lean_ctor_set(v___x_2626_, 2, v___x_2624_);
lean_ctor_set(v___x_2626_, 3, v___x_2625_);
lean_inc(v_declHint_2612_);
v___x_2627_ = l_Lean_MessageData_ofConstName(v_declHint_2612_, v___x_2617_);
v_c_2628_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_2628_, 0, v___x_2626_);
lean_ctor_set(v_c_2628_, 1, v___x_2627_);
v___x_2629_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_2616_, v_declHint_2612_);
if (lean_obj_tag(v___x_2629_) == 0)
{
lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; 
lean_dec_ref(v_env_2616_);
lean_dec(v_declHint_2612_);
v___x_2630_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1);
v___x_2631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2631_, 0, v___x_2630_);
lean_ctor_set(v___x_2631_, 1, v_c_2628_);
v___x_2632_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3);
v___x_2633_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2633_, 0, v___x_2631_);
lean_ctor_set(v___x_2633_, 1, v___x_2632_);
v___x_2634_ = l_Lean_MessageData_note(v___x_2633_);
v___x_2635_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2635_, 0, v_msg_2611_);
lean_ctor_set(v___x_2635_, 1, v___x_2634_);
v___x_2636_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2636_, 0, v___x_2635_);
return v___x_2636_;
}
else
{
lean_object* v_val_2637_; lean_object* v___x_2639_; uint8_t v_isShared_2640_; uint8_t v_isSharedCheck_2672_; 
v_val_2637_ = lean_ctor_get(v___x_2629_, 0);
v_isSharedCheck_2672_ = !lean_is_exclusive(v___x_2629_);
if (v_isSharedCheck_2672_ == 0)
{
v___x_2639_ = v___x_2629_;
v_isShared_2640_ = v_isSharedCheck_2672_;
goto v_resetjp_2638_;
}
else
{
lean_inc(v_val_2637_);
lean_dec(v___x_2629_);
v___x_2639_ = lean_box(0);
v_isShared_2640_ = v_isSharedCheck_2672_;
goto v_resetjp_2638_;
}
v_resetjp_2638_:
{
lean_object* v___x_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v_mod_2644_; uint8_t v___x_2645_; 
v___x_2641_ = lean_box(0);
v___x_2642_ = l_Lean_Environment_header(v_env_2616_);
lean_dec_ref(v_env_2616_);
v___x_2643_ = l_Lean_EnvironmentHeader_moduleNames(v___x_2642_);
v_mod_2644_ = lean_array_get(v___x_2641_, v___x_2643_, v_val_2637_);
lean_dec(v_val_2637_);
lean_dec_ref(v___x_2643_);
v___x_2645_ = l_Lean_isPrivateName(v_declHint_2612_);
lean_dec(v_declHint_2612_);
if (v___x_2645_ == 0)
{
lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; lean_object* v___x_2657_; 
v___x_2646_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5);
v___x_2647_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2647_, 0, v___x_2646_);
lean_ctor_set(v___x_2647_, 1, v_c_2628_);
v___x_2648_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7);
v___x_2649_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2649_, 0, v___x_2647_);
lean_ctor_set(v___x_2649_, 1, v___x_2648_);
v___x_2650_ = l_Lean_MessageData_ofName(v_mod_2644_);
v___x_2651_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2651_, 0, v___x_2649_);
lean_ctor_set(v___x_2651_, 1, v___x_2650_);
v___x_2652_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9);
v___x_2653_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2653_, 0, v___x_2651_);
lean_ctor_set(v___x_2653_, 1, v___x_2652_);
v___x_2654_ = l_Lean_MessageData_note(v___x_2653_);
v___x_2655_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2655_, 0, v_msg_2611_);
lean_ctor_set(v___x_2655_, 1, v___x_2654_);
if (v_isShared_2640_ == 0)
{
lean_ctor_set_tag(v___x_2639_, 0);
lean_ctor_set(v___x_2639_, 0, v___x_2655_);
v___x_2657_ = v___x_2639_;
goto v_reusejp_2656_;
}
else
{
lean_object* v_reuseFailAlloc_2658_; 
v_reuseFailAlloc_2658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2658_, 0, v___x_2655_);
v___x_2657_ = v_reuseFailAlloc_2658_;
goto v_reusejp_2656_;
}
v_reusejp_2656_:
{
return v___x_2657_;
}
}
else
{
lean_object* v___x_2659_; lean_object* v___x_2660_; lean_object* v___x_2661_; lean_object* v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2670_; 
v___x_2659_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1);
v___x_2660_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2660_, 0, v___x_2659_);
lean_ctor_set(v___x_2660_, 1, v_c_2628_);
v___x_2661_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11);
v___x_2662_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2662_, 0, v___x_2660_);
lean_ctor_set(v___x_2662_, 1, v___x_2661_);
v___x_2663_ = l_Lean_MessageData_ofName(v_mod_2644_);
v___x_2664_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2664_, 0, v___x_2662_);
lean_ctor_set(v___x_2664_, 1, v___x_2663_);
v___x_2665_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13);
v___x_2666_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2666_, 0, v___x_2664_);
lean_ctor_set(v___x_2666_, 1, v___x_2665_);
v___x_2667_ = l_Lean_MessageData_note(v___x_2666_);
v___x_2668_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2668_, 0, v_msg_2611_);
lean_ctor_set(v___x_2668_, 1, v___x_2667_);
if (v_isShared_2640_ == 0)
{
lean_ctor_set_tag(v___x_2639_, 0);
lean_ctor_set(v___x_2639_, 0, v___x_2668_);
v___x_2670_ = v___x_2639_;
goto v_reusejp_2669_;
}
else
{
lean_object* v_reuseFailAlloc_2671_; 
v_reuseFailAlloc_2671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2671_, 0, v___x_2668_);
v___x_2670_ = v_reuseFailAlloc_2671_;
goto v_reusejp_2669_;
}
v_reusejp_2669_:
{
return v___x_2670_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_2673_; 
lean_dec_ref(v_env_2616_);
lean_dec(v_declHint_2612_);
v___x_2673_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2673_, 0, v_msg_2611_);
return v___x_2673_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___boxed(lean_object* v_msg_2674_, lean_object* v_declHint_2675_, lean_object* v___y_2676_, lean_object* v___y_2677_){
_start:
{
lean_object* v_res_2678_; 
v_res_2678_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(v_msg_2674_, v_declHint_2675_, v___y_2676_);
lean_dec(v___y_2676_);
return v_res_2678_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(lean_object* v_msg_2679_, lean_object* v_declHint_2680_, lean_object* v___y_2681_, lean_object* v___y_2682_){
_start:
{
lean_object* v___x_2684_; lean_object* v_a_2685_; lean_object* v___x_2687_; uint8_t v_isShared_2688_; uint8_t v_isSharedCheck_2694_; 
v___x_2684_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(v_msg_2679_, v_declHint_2680_, v___y_2682_);
v_a_2685_ = lean_ctor_get(v___x_2684_, 0);
v_isSharedCheck_2694_ = !lean_is_exclusive(v___x_2684_);
if (v_isSharedCheck_2694_ == 0)
{
v___x_2687_ = v___x_2684_;
v_isShared_2688_ = v_isSharedCheck_2694_;
goto v_resetjp_2686_;
}
else
{
lean_inc(v_a_2685_);
lean_dec(v___x_2684_);
v___x_2687_ = lean_box(0);
v_isShared_2688_ = v_isSharedCheck_2694_;
goto v_resetjp_2686_;
}
v_resetjp_2686_:
{
lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2692_; 
v___x_2689_ = l_Lean_unknownIdentifierMessageTag;
v___x_2690_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_2690_, 0, v___x_2689_);
lean_ctor_set(v___x_2690_, 1, v_a_2685_);
if (v_isShared_2688_ == 0)
{
lean_ctor_set(v___x_2687_, 0, v___x_2690_);
v___x_2692_ = v___x_2687_;
goto v_reusejp_2691_;
}
else
{
lean_object* v_reuseFailAlloc_2693_; 
v_reuseFailAlloc_2693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2693_, 0, v___x_2690_);
v___x_2692_ = v_reuseFailAlloc_2693_;
goto v_reusejp_2691_;
}
v_reusejp_2691_:
{
return v___x_2692_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5___boxed(lean_object* v_msg_2695_, lean_object* v_declHint_2696_, lean_object* v___y_2697_, lean_object* v___y_2698_, lean_object* v___y_2699_){
_start:
{
lean_object* v_res_2700_; 
v_res_2700_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(v_msg_2695_, v_declHint_2696_, v___y_2697_, v___y_2698_);
lean_dec(v___y_2698_);
lean_dec_ref(v___y_2697_);
return v_res_2700_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg(lean_object* v_msg_2701_, lean_object* v___y_2702_, lean_object* v___y_2703_){
_start:
{
lean_object* v_ref_2705_; lean_object* v___x_2706_; lean_object* v_a_2707_; lean_object* v___x_2709_; uint8_t v_isShared_2710_; uint8_t v_isSharedCheck_2715_; 
v_ref_2705_ = lean_ctor_get(v___y_2702_, 5);
v___x_2706_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(v_msg_2701_, v___y_2702_, v___y_2703_);
v_a_2707_ = lean_ctor_get(v___x_2706_, 0);
v_isSharedCheck_2715_ = !lean_is_exclusive(v___x_2706_);
if (v_isSharedCheck_2715_ == 0)
{
v___x_2709_ = v___x_2706_;
v_isShared_2710_ = v_isSharedCheck_2715_;
goto v_resetjp_2708_;
}
else
{
lean_inc(v_a_2707_);
lean_dec(v___x_2706_);
v___x_2709_ = lean_box(0);
v_isShared_2710_ = v_isSharedCheck_2715_;
goto v_resetjp_2708_;
}
v_resetjp_2708_:
{
lean_object* v___x_2711_; lean_object* v___x_2713_; 
lean_inc(v_ref_2705_);
v___x_2711_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2711_, 0, v_ref_2705_);
lean_ctor_set(v___x_2711_, 1, v_a_2707_);
if (v_isShared_2710_ == 0)
{
lean_ctor_set_tag(v___x_2709_, 1);
lean_ctor_set(v___x_2709_, 0, v___x_2711_);
v___x_2713_ = v___x_2709_;
goto v_reusejp_2712_;
}
else
{
lean_object* v_reuseFailAlloc_2714_; 
v_reuseFailAlloc_2714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2714_, 0, v___x_2711_);
v___x_2713_ = v_reuseFailAlloc_2714_;
goto v_reusejp_2712_;
}
v_reusejp_2712_:
{
return v___x_2713_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg___boxed(lean_object* v_msg_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_){
_start:
{
lean_object* v_res_2720_; 
v_res_2720_ = lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg(v_msg_2716_, v___y_2717_, v___y_2718_);
lean_dec(v___y_2718_);
lean_dec_ref(v___y_2717_);
return v_res_2720_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(lean_object* v_ref_2721_, lean_object* v_msg_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_){
_start:
{
lean_object* v_fileName_2726_; lean_object* v_fileMap_2727_; lean_object* v_options_2728_; lean_object* v_currRecDepth_2729_; lean_object* v_maxRecDepth_2730_; lean_object* v_ref_2731_; lean_object* v_currNamespace_2732_; lean_object* v_openDecls_2733_; lean_object* v_initHeartbeats_2734_; lean_object* v_maxHeartbeats_2735_; lean_object* v_quotContext_2736_; lean_object* v_currMacroScope_2737_; uint8_t v_diag_2738_; lean_object* v_cancelTk_x3f_2739_; uint8_t v_suppressElabErrors_2740_; lean_object* v_inheritedTraceOptions_2741_; lean_object* v_ref_2742_; lean_object* v___x_2743_; lean_object* v___x_2744_; 
v_fileName_2726_ = lean_ctor_get(v___y_2723_, 0);
v_fileMap_2727_ = lean_ctor_get(v___y_2723_, 1);
v_options_2728_ = lean_ctor_get(v___y_2723_, 2);
v_currRecDepth_2729_ = lean_ctor_get(v___y_2723_, 3);
v_maxRecDepth_2730_ = lean_ctor_get(v___y_2723_, 4);
v_ref_2731_ = lean_ctor_get(v___y_2723_, 5);
v_currNamespace_2732_ = lean_ctor_get(v___y_2723_, 6);
v_openDecls_2733_ = lean_ctor_get(v___y_2723_, 7);
v_initHeartbeats_2734_ = lean_ctor_get(v___y_2723_, 8);
v_maxHeartbeats_2735_ = lean_ctor_get(v___y_2723_, 9);
v_quotContext_2736_ = lean_ctor_get(v___y_2723_, 10);
v_currMacroScope_2737_ = lean_ctor_get(v___y_2723_, 11);
v_diag_2738_ = lean_ctor_get_uint8(v___y_2723_, sizeof(void*)*14);
v_cancelTk_x3f_2739_ = lean_ctor_get(v___y_2723_, 12);
v_suppressElabErrors_2740_ = lean_ctor_get_uint8(v___y_2723_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2741_ = lean_ctor_get(v___y_2723_, 13);
v_ref_2742_ = l_Lean_replaceRef(v_ref_2721_, v_ref_2731_);
lean_inc_ref(v_inheritedTraceOptions_2741_);
lean_inc(v_cancelTk_x3f_2739_);
lean_inc(v_currMacroScope_2737_);
lean_inc(v_quotContext_2736_);
lean_inc(v_maxHeartbeats_2735_);
lean_inc(v_initHeartbeats_2734_);
lean_inc(v_openDecls_2733_);
lean_inc(v_currNamespace_2732_);
lean_inc(v_maxRecDepth_2730_);
lean_inc(v_currRecDepth_2729_);
lean_inc_ref(v_options_2728_);
lean_inc_ref(v_fileMap_2727_);
lean_inc_ref(v_fileName_2726_);
v___x_2743_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2743_, 0, v_fileName_2726_);
lean_ctor_set(v___x_2743_, 1, v_fileMap_2727_);
lean_ctor_set(v___x_2743_, 2, v_options_2728_);
lean_ctor_set(v___x_2743_, 3, v_currRecDepth_2729_);
lean_ctor_set(v___x_2743_, 4, v_maxRecDepth_2730_);
lean_ctor_set(v___x_2743_, 5, v_ref_2742_);
lean_ctor_set(v___x_2743_, 6, v_currNamespace_2732_);
lean_ctor_set(v___x_2743_, 7, v_openDecls_2733_);
lean_ctor_set(v___x_2743_, 8, v_initHeartbeats_2734_);
lean_ctor_set(v___x_2743_, 9, v_maxHeartbeats_2735_);
lean_ctor_set(v___x_2743_, 10, v_quotContext_2736_);
lean_ctor_set(v___x_2743_, 11, v_currMacroScope_2737_);
lean_ctor_set(v___x_2743_, 12, v_cancelTk_x3f_2739_);
lean_ctor_set(v___x_2743_, 13, v_inheritedTraceOptions_2741_);
lean_ctor_set_uint8(v___x_2743_, sizeof(void*)*14, v_diag_2738_);
lean_ctor_set_uint8(v___x_2743_, sizeof(void*)*14 + 1, v_suppressElabErrors_2740_);
v___x_2744_ = lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg(v_msg_2722_, v___x_2743_, v___y_2724_);
lean_dec_ref_known(v___x_2743_, 14);
return v___x_2744_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg___boxed(lean_object* v_ref_2745_, lean_object* v_msg_2746_, lean_object* v___y_2747_, lean_object* v___y_2748_, lean_object* v___y_2749_){
_start:
{
lean_object* v_res_2750_; 
v_res_2750_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_ref_2745_, v_msg_2746_, v___y_2747_, v___y_2748_);
lean_dec(v___y_2748_);
lean_dec_ref(v___y_2747_);
lean_dec(v_ref_2745_);
return v_res_2750_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_ref_2751_, lean_object* v_msg_2752_, lean_object* v_declHint_2753_, lean_object* v___y_2754_, lean_object* v___y_2755_){
_start:
{
lean_object* v___x_2757_; lean_object* v_a_2758_; lean_object* v___x_2759_; 
v___x_2757_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(v_msg_2752_, v_declHint_2753_, v___y_2754_, v___y_2755_);
v_a_2758_ = lean_ctor_get(v___x_2757_, 0);
lean_inc(v_a_2758_);
lean_dec_ref(v___x_2757_);
v___x_2759_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_ref_2751_, v_a_2758_, v___y_2754_, v___y_2755_);
return v___x_2759_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_ref_2760_, lean_object* v_msg_2761_, lean_object* v_declHint_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_){
_start:
{
lean_object* v_res_2766_; 
v_res_2766_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_ref_2760_, v_msg_2761_, v_declHint_2762_, v___y_2763_, v___y_2764_);
lean_dec(v___y_2764_);
lean_dec_ref(v___y_2763_);
lean_dec(v_ref_2760_);
return v_res_2766_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_2768_; lean_object* v___x_2769_; 
v___x_2768_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__0));
v___x_2769_ = l_Lean_stringToMessageData(v___x_2768_);
return v___x_2769_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_2771_; lean_object* v___x_2772_; 
v___x_2771_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__2));
v___x_2772_ = l_Lean_stringToMessageData(v___x_2771_);
return v___x_2772_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_ref_2773_, lean_object* v_constName_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_){
_start:
{
lean_object* v___x_2778_; uint8_t v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; 
v___x_2778_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__1, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__1_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__1);
v___x_2779_ = 0;
lean_inc(v_constName_2774_);
v___x_2780_ = l_Lean_MessageData_ofConstName(v_constName_2774_, v___x_2779_);
v___x_2781_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2781_, 0, v___x_2778_);
lean_ctor_set(v___x_2781_, 1, v___x_2780_);
v___x_2782_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__3, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__3_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___closed__3);
v___x_2783_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2783_, 0, v___x_2781_);
lean_ctor_set(v___x_2783_, 1, v___x_2782_);
v___x_2784_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_ref_2773_, v___x_2783_, v_constName_2774_, v___y_2775_, v___y_2776_);
return v___x_2784_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_ref_2785_, lean_object* v_constName_2786_, lean_object* v___y_2787_, lean_object* v___y_2788_, lean_object* v___y_2789_){
_start:
{
lean_object* v_res_2790_; 
v_res_2790_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg(v_ref_2785_, v_constName_2786_, v___y_2787_, v___y_2788_);
lean_dec(v___y_2788_);
lean_dec_ref(v___y_2787_);
lean_dec(v_ref_2785_);
return v_res_2790_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg(lean_object* v_constName_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_){
_start:
{
lean_object* v_ref_2795_; lean_object* v___x_2796_; 
v_ref_2795_ = lean_ctor_get(v___y_2792_, 5);
v___x_2796_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg(v_ref_2795_, v_constName_2791_, v___y_2792_, v___y_2793_);
return v___x_2796_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_constName_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_){
_start:
{
lean_object* v_res_2801_; 
v_res_2801_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg(v_constName_2797_, v___y_2798_, v___y_2799_);
lean_dec(v___y_2799_);
lean_dec_ref(v___y_2798_);
return v_res_2801_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0(lean_object* v_constName_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_){
_start:
{
lean_object* v___x_2806_; lean_object* v_env_2807_; uint8_t v___x_2808_; lean_object* v___x_2809_; 
v___x_2806_ = lean_st_ref_get(v___y_2804_);
v_env_2807_ = lean_ctor_get(v___x_2806_, 0);
lean_inc_ref(v_env_2807_);
lean_dec(v___x_2806_);
v___x_2808_ = 0;
lean_inc(v_constName_2802_);
v___x_2809_ = l_Lean_Environment_findConstVal_x3f(v_env_2807_, v_constName_2802_, v___x_2808_);
if (lean_obj_tag(v___x_2809_) == 0)
{
lean_object* v___x_2810_; 
v___x_2810_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg(v_constName_2802_, v___y_2803_, v___y_2804_);
return v___x_2810_;
}
else
{
lean_object* v_val_2811_; lean_object* v___x_2813_; uint8_t v_isShared_2814_; uint8_t v_isSharedCheck_2818_; 
lean_dec(v_constName_2802_);
v_val_2811_ = lean_ctor_get(v___x_2809_, 0);
v_isSharedCheck_2818_ = !lean_is_exclusive(v___x_2809_);
if (v_isSharedCheck_2818_ == 0)
{
v___x_2813_ = v___x_2809_;
v_isShared_2814_ = v_isSharedCheck_2818_;
goto v_resetjp_2812_;
}
else
{
lean_inc(v_val_2811_);
lean_dec(v___x_2809_);
v___x_2813_ = lean_box(0);
v_isShared_2814_ = v_isSharedCheck_2818_;
goto v_resetjp_2812_;
}
v_resetjp_2812_:
{
lean_object* v___x_2816_; 
if (v_isShared_2814_ == 0)
{
lean_ctor_set_tag(v___x_2813_, 0);
v___x_2816_ = v___x_2813_;
goto v_reusejp_2815_;
}
else
{
lean_object* v_reuseFailAlloc_2817_; 
v_reuseFailAlloc_2817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2817_, 0, v_val_2811_);
v___x_2816_ = v_reuseFailAlloc_2817_;
goto v_reusejp_2815_;
}
v_reusejp_2815_:
{
return v___x_2816_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0___boxed(lean_object* v_constName_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_, lean_object* v___y_2822_){
_start:
{
lean_object* v_res_2823_; 
v_res_2823_ = lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0(v_constName_2819_, v___y_2820_, v___y_2821_);
lean_dec(v___y_2821_);
lean_dec_ref(v___y_2820_);
return v_res_2823_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0(lean_object* v_constName_2824_, lean_object* v___y_2825_, lean_object* v___y_2826_){
_start:
{
lean_object* v___x_2828_; 
lean_inc(v_constName_2824_);
v___x_2828_ = lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0(v_constName_2824_, v___y_2825_, v___y_2826_);
if (lean_obj_tag(v___x_2828_) == 0)
{
lean_object* v_a_2829_; lean_object* v___x_2831_; uint8_t v_isShared_2832_; uint8_t v_isSharedCheck_2840_; 
v_a_2829_ = lean_ctor_get(v___x_2828_, 0);
v_isSharedCheck_2840_ = !lean_is_exclusive(v___x_2828_);
if (v_isSharedCheck_2840_ == 0)
{
v___x_2831_ = v___x_2828_;
v_isShared_2832_ = v_isSharedCheck_2840_;
goto v_resetjp_2830_;
}
else
{
lean_inc(v_a_2829_);
lean_dec(v___x_2828_);
v___x_2831_ = lean_box(0);
v_isShared_2832_ = v_isSharedCheck_2840_;
goto v_resetjp_2830_;
}
v_resetjp_2830_:
{
lean_object* v_levelParams_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2838_; 
v_levelParams_2833_ = lean_ctor_get(v_a_2829_, 1);
lean_inc(v_levelParams_2833_);
lean_dec(v_a_2829_);
v___x_2834_ = lean_box(0);
v___x_2835_ = lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__1(v_levelParams_2833_, v___x_2834_);
v___x_2836_ = l_Lean_mkConst(v_constName_2824_, v___x_2835_);
if (v_isShared_2832_ == 0)
{
lean_ctor_set(v___x_2831_, 0, v___x_2836_);
v___x_2838_ = v___x_2831_;
goto v_reusejp_2837_;
}
else
{
lean_object* v_reuseFailAlloc_2839_; 
v_reuseFailAlloc_2839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2839_, 0, v___x_2836_);
v___x_2838_ = v_reuseFailAlloc_2839_;
goto v_reusejp_2837_;
}
v_reusejp_2837_:
{
return v___x_2838_;
}
}
}
else
{
lean_object* v_a_2841_; lean_object* v___x_2843_; uint8_t v_isShared_2844_; uint8_t v_isSharedCheck_2848_; 
lean_dec(v_constName_2824_);
v_a_2841_ = lean_ctor_get(v___x_2828_, 0);
v_isSharedCheck_2848_ = !lean_is_exclusive(v___x_2828_);
if (v_isSharedCheck_2848_ == 0)
{
v___x_2843_ = v___x_2828_;
v_isShared_2844_ = v_isSharedCheck_2848_;
goto v_resetjp_2842_;
}
else
{
lean_inc(v_a_2841_);
lean_dec(v___x_2828_);
v___x_2843_ = lean_box(0);
v_isShared_2844_ = v_isSharedCheck_2848_;
goto v_resetjp_2842_;
}
v_resetjp_2842_:
{
lean_object* v___x_2846_; 
if (v_isShared_2844_ == 0)
{
v___x_2846_ = v___x_2843_;
goto v_reusejp_2845_;
}
else
{
lean_object* v_reuseFailAlloc_2847_; 
v_reuseFailAlloc_2847_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2847_, 0, v_a_2841_);
v___x_2846_ = v_reuseFailAlloc_2847_;
goto v_reusejp_2845_;
}
v_reusejp_2845_:
{
return v___x_2846_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0___boxed(lean_object* v_constName_2849_, lean_object* v___y_2850_, lean_object* v___y_2851_, lean_object* v___y_2852_){
_start:
{
lean_object* v_res_2853_; 
v_res_2853_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0(v_constName_2849_, v___y_2850_, v___y_2851_);
lean_dec(v___y_2851_);
lean_dec_ref(v___y_2850_);
return v_res_2853_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__1(void){
_start:
{
lean_object* v___x_2855_; lean_object* v___x_2856_; 
v___x_2855_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_printWarning___closed__0));
v___x_2856_ = l_Lean_stringToMessageData(v___x_2855_);
return v___x_2856_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__3(void){
_start:
{
lean_object* v___x_2858_; lean_object* v___x_2859_; 
v___x_2858_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_printWarning___closed__2));
v___x_2859_ = l_Lean_stringToMessageData(v___x_2858_);
return v___x_2859_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5(void){
_start:
{
lean_object* v___x_2861_; lean_object* v___x_2862_; 
v___x_2861_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_printWarning___closed__4));
v___x_2862_ = l_Lean_stringToMessageData(v___x_2861_);
return v___x_2862_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__7(void){
_start:
{
lean_object* v___x_2864_; lean_object* v___x_2865_; 
v___x_2864_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_printWarning___closed__6));
v___x_2865_ = l_Lean_stringToMessageData(v___x_2864_);
return v___x_2865_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__9(void){
_start:
{
lean_object* v___x_2867_; lean_object* v___x_2868_; 
v___x_2867_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_printWarning___closed__8));
v___x_2868_ = l_Lean_stringToMessageData(v___x_2867_);
return v___x_2868_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__11(void){
_start:
{
lean_object* v___x_2870_; lean_object* v___x_2871_; 
v___x_2870_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_printWarning___closed__10));
v___x_2871_ = l_Lean_stringToMessageData(v___x_2870_);
return v___x_2871_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning(lean_object* v_declName_2872_, lean_object* v_warning_2873_, uint8_t v_useErrorFormat_2874_, lean_object* v_filePath_2875_, lean_object* v_a_2876_, lean_object* v_a_2877_){
_start:
{
lean_object* v___y_2880_; lean_object* v___y_2881_; 
if (v_useErrorFormat_2874_ == 0)
{
lean_dec_ref(v_filePath_2875_);
v___y_2880_ = v_a_2876_;
v___y_2881_ = v_a_2877_;
goto v___jp_2879_;
}
else
{
lean_object* v___x_2901_; 
lean_inc(v_declName_2872_);
v___x_2901_ = lp_batteries_Lean_findDeclarationRanges_x3f___at___00Batteries_Tactic_Lint_sortResults_spec__0(v_declName_2872_, v_a_2876_, v_a_2877_);
if (lean_obj_tag(v___x_2901_) == 0)
{
lean_object* v_a_2902_; 
v_a_2902_ = lean_ctor_get(v___x_2901_, 0);
lean_inc(v_a_2902_);
lean_dec_ref_known(v___x_2901_, 1);
if (lean_obj_tag(v_a_2902_) == 1)
{
lean_object* v_val_2903_; lean_object* v___x_2905_; uint8_t v_isShared_2906_; uint8_t v_isSharedCheck_2959_; 
v_val_2903_ = lean_ctor_get(v_a_2902_, 0);
v_isSharedCheck_2959_ = !lean_is_exclusive(v_a_2902_);
if (v_isSharedCheck_2959_ == 0)
{
v___x_2905_ = v_a_2902_;
v_isShared_2906_ = v_isSharedCheck_2959_;
goto v_resetjp_2904_;
}
else
{
lean_inc(v_val_2903_);
lean_dec(v_a_2902_);
v___x_2905_ = lean_box(0);
v_isShared_2906_ = v_isSharedCheck_2959_;
goto v_resetjp_2904_;
}
v_resetjp_2904_:
{
lean_object* v___x_2907_; 
v___x_2907_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0(v_declName_2872_, v_a_2876_, v_a_2877_);
if (lean_obj_tag(v___x_2907_) == 0)
{
lean_object* v_range_2908_; lean_object* v___x_2910_; uint8_t v_isShared_2911_; uint8_t v_isSharedCheck_2949_; 
v_range_2908_ = lean_ctor_get(v_val_2903_, 0);
v_isSharedCheck_2949_ = !lean_is_exclusive(v_val_2903_);
if (v_isSharedCheck_2949_ == 0)
{
lean_object* v_unused_2950_; 
v_unused_2950_ = lean_ctor_get(v_val_2903_, 1);
lean_dec(v_unused_2950_);
v___x_2910_ = v_val_2903_;
v_isShared_2911_ = v_isSharedCheck_2949_;
goto v_resetjp_2909_;
}
else
{
lean_inc(v_range_2908_);
lean_dec(v_val_2903_);
v___x_2910_ = lean_box(0);
v_isShared_2911_ = v_isSharedCheck_2949_;
goto v_resetjp_2909_;
}
v_resetjp_2909_:
{
lean_object* v_pos_2912_; lean_object* v_a_2913_; lean_object* v_line_2914_; lean_object* v_column_2915_; lean_object* v___x_2917_; uint8_t v_isShared_2918_; uint8_t v_isSharedCheck_2948_; 
v_pos_2912_ = lean_ctor_get(v_range_2908_, 0);
lean_inc_ref(v_pos_2912_);
lean_dec_ref(v_range_2908_);
v_a_2913_ = lean_ctor_get(v___x_2907_, 0);
lean_inc(v_a_2913_);
lean_dec_ref_known(v___x_2907_, 1);
v_line_2914_ = lean_ctor_get(v_pos_2912_, 0);
v_column_2915_ = lean_ctor_get(v_pos_2912_, 1);
v_isSharedCheck_2948_ = !lean_is_exclusive(v_pos_2912_);
if (v_isSharedCheck_2948_ == 0)
{
v___x_2917_ = v_pos_2912_;
v_isShared_2918_ = v_isSharedCheck_2948_;
goto v_resetjp_2916_;
}
else
{
lean_inc(v_column_2915_);
lean_inc(v_line_2914_);
lean_dec(v_pos_2912_);
v___x_2917_ = lean_box(0);
v_isShared_2918_ = v_isSharedCheck_2948_;
goto v_resetjp_2916_;
}
v_resetjp_2916_:
{
lean_object* v___x_2920_; 
if (v_isShared_2906_ == 0)
{
lean_ctor_set_tag(v___x_2905_, 3);
lean_ctor_set(v___x_2905_, 0, v_filePath_2875_);
v___x_2920_ = v___x_2905_;
goto v_reusejp_2919_;
}
else
{
lean_object* v_reuseFailAlloc_2947_; 
v_reuseFailAlloc_2947_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2947_, 0, v_filePath_2875_);
v___x_2920_ = v_reuseFailAlloc_2947_;
goto v_reusejp_2919_;
}
v_reusejp_2919_:
{
lean_object* v___x_2921_; lean_object* v___x_2922_; lean_object* v___x_2924_; 
v___x_2921_ = l_Lean_MessageData_ofFormat(v___x_2920_);
v___x_2922_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__7, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__7);
if (v_isShared_2918_ == 0)
{
lean_ctor_set_tag(v___x_2917_, 7);
lean_ctor_set(v___x_2917_, 1, v___x_2922_);
lean_ctor_set(v___x_2917_, 0, v___x_2921_);
v___x_2924_ = v___x_2917_;
goto v_reusejp_2923_;
}
else
{
lean_object* v_reuseFailAlloc_2946_; 
v_reuseFailAlloc_2946_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2946_, 0, v___x_2921_);
lean_ctor_set(v_reuseFailAlloc_2946_, 1, v___x_2922_);
v___x_2924_ = v_reuseFailAlloc_2946_;
goto v_reusejp_2923_;
}
v_reusejp_2923_:
{
lean_object* v___x_2925_; lean_object* v___x_2926_; lean_object* v___x_2927_; lean_object* v___x_2929_; 
v___x_2925_ = l_Nat_reprFast(v_line_2914_);
v___x_2926_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2926_, 0, v___x_2925_);
v___x_2927_ = l_Lean_MessageData_ofFormat(v___x_2926_);
if (v_isShared_2911_ == 0)
{
lean_ctor_set_tag(v___x_2910_, 7);
lean_ctor_set(v___x_2910_, 1, v___x_2927_);
lean_ctor_set(v___x_2910_, 0, v___x_2924_);
v___x_2929_ = v___x_2910_;
goto v_reusejp_2928_;
}
else
{
lean_object* v_reuseFailAlloc_2945_; 
v_reuseFailAlloc_2945_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2945_, 0, v___x_2924_);
lean_ctor_set(v_reuseFailAlloc_2945_, 1, v___x_2927_);
v___x_2929_ = v_reuseFailAlloc_2945_;
goto v_reusejp_2928_;
}
v_reusejp_2928_:
{
lean_object* v___x_2930_; lean_object* v___x_2931_; lean_object* v___x_2932_; lean_object* v___x_2933_; lean_object* v___x_2934_; lean_object* v___x_2935_; lean_object* v___x_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2944_; 
v___x_2930_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2930_, 0, v___x_2929_);
lean_ctor_set(v___x_2930_, 1, v___x_2922_);
v___x_2931_ = lean_unsigned_to_nat(1u);
v___x_2932_ = lean_nat_add(v_column_2915_, v___x_2931_);
lean_dec(v_column_2915_);
v___x_2933_ = l_Nat_reprFast(v___x_2932_);
v___x_2934_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2934_, 0, v___x_2933_);
v___x_2935_ = l_Lean_MessageData_ofFormat(v___x_2934_);
v___x_2936_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2936_, 0, v___x_2930_);
lean_ctor_set(v___x_2936_, 1, v___x_2935_);
v___x_2937_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__9, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__9_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__9);
v___x_2938_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2938_, 0, v___x_2936_);
lean_ctor_set(v___x_2938_, 1, v___x_2937_);
v___x_2939_ = l_Lean_MessageData_ofExpr(v_a_2913_);
v___x_2940_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2940_, 0, v___x_2938_);
lean_ctor_set(v___x_2940_, 1, v___x_2939_);
v___x_2941_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__11, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__11_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__11);
v___x_2942_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2942_, 0, v___x_2940_);
lean_ctor_set(v___x_2942_, 1, v___x_2941_);
v___x_2943_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2943_, 0, v___x_2942_);
lean_ctor_set(v___x_2943_, 1, v_warning_2873_);
v___x_2944_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(v___x_2943_, v_a_2876_, v_a_2877_);
return v___x_2944_;
}
}
}
}
}
}
else
{
lean_object* v_a_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_2958_; 
lean_del_object(v___x_2905_);
lean_dec(v_val_2903_);
lean_dec_ref(v_filePath_2875_);
lean_dec_ref(v_warning_2873_);
v_a_2951_ = lean_ctor_get(v___x_2907_, 0);
v_isSharedCheck_2958_ = !lean_is_exclusive(v___x_2907_);
if (v_isSharedCheck_2958_ == 0)
{
v___x_2953_ = v___x_2907_;
v_isShared_2954_ = v_isSharedCheck_2958_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_a_2951_);
lean_dec(v___x_2907_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_2958_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
lean_object* v___x_2956_; 
if (v_isShared_2954_ == 0)
{
v___x_2956_ = v___x_2953_;
goto v_reusejp_2955_;
}
else
{
lean_object* v_reuseFailAlloc_2957_; 
v_reuseFailAlloc_2957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2957_, 0, v_a_2951_);
v___x_2956_ = v_reuseFailAlloc_2957_;
goto v_reusejp_2955_;
}
v_reusejp_2955_:
{
return v___x_2956_;
}
}
}
}
}
else
{
lean_dec(v_a_2902_);
lean_dec_ref(v_filePath_2875_);
v___y_2880_ = v_a_2876_;
v___y_2881_ = v_a_2877_;
goto v___jp_2879_;
}
}
else
{
lean_object* v_a_2960_; lean_object* v___x_2962_; uint8_t v_isShared_2963_; uint8_t v_isSharedCheck_2967_; 
lean_dec_ref(v_filePath_2875_);
lean_dec_ref(v_warning_2873_);
lean_dec(v_declName_2872_);
v_a_2960_ = lean_ctor_get(v___x_2901_, 0);
v_isSharedCheck_2967_ = !lean_is_exclusive(v___x_2901_);
if (v_isSharedCheck_2967_ == 0)
{
v___x_2962_ = v___x_2901_;
v_isShared_2963_ = v_isSharedCheck_2967_;
goto v_resetjp_2961_;
}
else
{
lean_inc(v_a_2960_);
lean_dec(v___x_2901_);
v___x_2962_ = lean_box(0);
v_isShared_2963_ = v_isSharedCheck_2967_;
goto v_resetjp_2961_;
}
v_resetjp_2961_:
{
lean_object* v___x_2965_; 
if (v_isShared_2963_ == 0)
{
v___x_2965_ = v___x_2962_;
goto v_reusejp_2964_;
}
else
{
lean_object* v_reuseFailAlloc_2966_; 
v_reuseFailAlloc_2966_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2966_, 0, v_a_2960_);
v___x_2965_ = v_reuseFailAlloc_2966_;
goto v_reusejp_2964_;
}
v_reusejp_2964_:
{
return v___x_2965_;
}
}
}
}
v___jp_2879_:
{
lean_object* v___x_2882_; 
v___x_2882_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0(v_declName_2872_, v___y_2880_, v___y_2881_);
if (lean_obj_tag(v___x_2882_) == 0)
{
lean_object* v_a_2883_; lean_object* v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; 
v_a_2883_ = lean_ctor_get(v___x_2882_, 0);
lean_inc(v_a_2883_);
lean_dec_ref_known(v___x_2882_, 1);
v___x_2884_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__1, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__1);
v___x_2885_ = l_Lean_MessageData_ofExpr(v_a_2883_);
v___x_2886_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2886_, 0, v___x_2884_);
lean_ctor_set(v___x_2886_, 1, v___x_2885_);
v___x_2887_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__3, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__3_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__3);
v___x_2888_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2888_, 0, v___x_2886_);
lean_ctor_set(v___x_2888_, 1, v___x_2887_);
v___x_2889_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2889_, 0, v___x_2888_);
lean_ctor_set(v___x_2889_, 1, v_warning_2873_);
v___x_2890_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5);
v___x_2891_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2891_, 0, v___x_2889_);
lean_ctor_set(v___x_2891_, 1, v___x_2890_);
v___x_2892_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5(v___x_2891_, v___y_2880_, v___y_2881_);
return v___x_2892_;
}
else
{
lean_object* v_a_2893_; lean_object* v___x_2895_; uint8_t v_isShared_2896_; uint8_t v_isSharedCheck_2900_; 
lean_dec_ref(v_warning_2873_);
v_a_2893_ = lean_ctor_get(v___x_2882_, 0);
v_isSharedCheck_2900_ = !lean_is_exclusive(v___x_2882_);
if (v_isSharedCheck_2900_ == 0)
{
v___x_2895_ = v___x_2882_;
v_isShared_2896_ = v_isSharedCheck_2900_;
goto v_resetjp_2894_;
}
else
{
lean_inc(v_a_2893_);
lean_dec(v___x_2882_);
v___x_2895_ = lean_box(0);
v_isShared_2896_ = v_isSharedCheck_2900_;
goto v_resetjp_2894_;
}
v_resetjp_2894_:
{
lean_object* v___x_2898_; 
if (v_isShared_2896_ == 0)
{
v___x_2898_ = v___x_2895_;
goto v_reusejp_2897_;
}
else
{
lean_object* v_reuseFailAlloc_2899_; 
v_reuseFailAlloc_2899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2899_, 0, v_a_2893_);
v___x_2898_ = v_reuseFailAlloc_2899_;
goto v_reusejp_2897_;
}
v_reusejp_2897_:
{
return v___x_2898_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarning___boxed(lean_object* v_declName_2968_, lean_object* v_warning_2969_, lean_object* v_useErrorFormat_2970_, lean_object* v_filePath_2971_, lean_object* v_a_2972_, lean_object* v_a_2973_, lean_object* v_a_2974_){
_start:
{
uint8_t v_useErrorFormat_boxed_2975_; lean_object* v_res_2976_; 
v_useErrorFormat_boxed_2975_ = lean_unbox(v_useErrorFormat_2970_);
v_res_2976_ = lp_batteries_Batteries_Tactic_Lint_printWarning(v_declName_2968_, v_warning_2969_, v_useErrorFormat_boxed_2975_, v_filePath_2971_, v_a_2972_, v_a_2973_);
lean_dec(v_a_2973_);
lean_dec_ref(v_a_2972_);
return v_res_2976_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_2977_, lean_object* v_constName_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_){
_start:
{
lean_object* v___x_2982_; 
v___x_2982_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg(v_constName_2978_, v___y_2979_, v___y_2980_);
return v___x_2982_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_2983_, lean_object* v_constName_2984_, lean_object* v___y_2985_, lean_object* v___y_2986_, lean_object* v___y_2987_){
_start:
{
lean_object* v_res_2988_; 
v_res_2988_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1(v_00_u03b1_2983_, v_constName_2984_, v___y_2985_, v___y_2986_);
lean_dec(v___y_2986_);
lean_dec_ref(v___y_2985_);
return v_res_2988_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_2989_, lean_object* v_ref_2990_, lean_object* v_constName_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_){
_start:
{
lean_object* v___x_2995_; 
v___x_2995_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___redArg(v_ref_2990_, v_constName_2991_, v___y_2992_, v___y_2993_);
return v___x_2995_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_2996_, lean_object* v_ref_2997_, lean_object* v_constName_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_){
_start:
{
lean_object* v_res_3002_; 
v_res_3002_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_2996_, v_ref_2997_, v_constName_2998_, v___y_2999_, v___y_3000_);
lean_dec(v___y_3000_);
lean_dec_ref(v___y_2999_);
lean_dec(v_ref_2997_);
return v_res_3002_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b1_3003_, lean_object* v_ref_3004_, lean_object* v_msg_3005_, lean_object* v_declHint_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_){
_start:
{
lean_object* v___x_3010_; 
v___x_3010_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_ref_3004_, v_msg_3005_, v_declHint_3006_, v___y_3007_, v___y_3008_);
return v___x_3010_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b1_3011_, lean_object* v_ref_3012_, lean_object* v_msg_3013_, lean_object* v_declHint_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_){
_start:
{
lean_object* v_res_3018_; 
v_res_3018_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4(v_00_u03b1_3011_, v_ref_3012_, v_msg_3013_, v_declHint_3014_, v___y_3015_, v___y_3016_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
lean_dec(v_ref_3012_);
return v_res_3018_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6(lean_object* v_msg_3019_, lean_object* v_declHint_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_){
_start:
{
lean_object* v___x_3024_; 
v___x_3024_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(v_msg_3019_, v_declHint_3020_, v___y_3022_);
return v___x_3024_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___boxed(lean_object* v_msg_3025_, lean_object* v_declHint_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_){
_start:
{
lean_object* v_res_3030_; 
v_res_3030_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6(v_msg_3025_, v_declHint_3026_, v___y_3027_, v___y_3028_);
lean_dec(v___y_3028_);
lean_dec_ref(v___y_3027_);
return v_res_3030_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(lean_object* v_00_u03b1_3031_, lean_object* v_ref_3032_, lean_object* v_msg_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_){
_start:
{
lean_object* v___x_3037_; 
v___x_3037_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_ref_3032_, v_msg_3033_, v___y_3034_, v___y_3035_);
return v___x_3037_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___boxed(lean_object* v_00_u03b1_3038_, lean_object* v_ref_3039_, lean_object* v_msg_3040_, lean_object* v___y_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_){
_start:
{
lean_object* v_res_3044_; 
v_res_3044_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(v_00_u03b1_3038_, v_ref_3039_, v_msg_3040_, v___y_3041_, v___y_3042_);
lean_dec(v___y_3042_);
lean_dec_ref(v___y_3041_);
lean_dec(v_ref_3039_);
return v_res_3044_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8(lean_object* v_00_u03b1_3045_, lean_object* v_msg_3046_, lean_object* v___y_3047_, lean_object* v___y_3048_){
_start:
{
lean_object* v___x_3050_; 
v___x_3050_ = lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___redArg(v_msg_3046_, v___y_3047_, v___y_3048_);
return v___x_3050_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8___boxed(lean_object* v_00_u03b1_3051_, lean_object* v_msg_3052_, lean_object* v___y_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_){
_start:
{
lean_object* v_res_3056_; 
v_res_3056_ = lp_batteries_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6_spec__8(v_00_u03b1_3051_, v_msg_3052_, v___y_3053_, v___y_3054_);
lean_dec(v___y_3054_);
lean_dec_ref(v___y_3053_);
return v_res_3056_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_printWarnings_spec__0(uint8_t v_useErrorFormat_3057_, lean_object* v_filePath_3058_, size_t v_sz_3059_, size_t v_i_3060_, lean_object* v_bs_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_){
_start:
{
uint8_t v___x_3065_; 
v___x_3065_ = lean_usize_dec_lt(v_i_3060_, v_sz_3059_);
if (v___x_3065_ == 0)
{
lean_object* v___x_3066_; 
lean_dec_ref(v_filePath_3058_);
v___x_3066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3066_, 0, v_bs_3061_);
return v___x_3066_;
}
else
{
lean_object* v_v_3067_; lean_object* v_fst_3068_; lean_object* v_snd_3069_; lean_object* v___x_3070_; 
v_v_3067_ = lean_array_uget_borrowed(v_bs_3061_, v_i_3060_);
v_fst_3068_ = lean_ctor_get(v_v_3067_, 0);
v_snd_3069_ = lean_ctor_get(v_v_3067_, 1);
lean_inc_ref(v_filePath_3058_);
lean_inc(v_snd_3069_);
lean_inc(v_fst_3068_);
v___x_3070_ = lp_batteries_Batteries_Tactic_Lint_printWarning(v_fst_3068_, v_snd_3069_, v_useErrorFormat_3057_, v_filePath_3058_, v___y_3062_, v___y_3063_);
if (lean_obj_tag(v___x_3070_) == 0)
{
lean_object* v_a_3071_; lean_object* v___x_3072_; lean_object* v_bs_x27_3073_; size_t v___x_3074_; size_t v___x_3075_; lean_object* v___x_3076_; 
v_a_3071_ = lean_ctor_get(v___x_3070_, 0);
lean_inc(v_a_3071_);
lean_dec_ref_known(v___x_3070_, 1);
v___x_3072_ = lean_unsigned_to_nat(0u);
v_bs_x27_3073_ = lean_array_uset(v_bs_3061_, v_i_3060_, v___x_3072_);
v___x_3074_ = ((size_t)1ULL);
v___x_3075_ = lean_usize_add(v_i_3060_, v___x_3074_);
v___x_3076_ = lean_array_uset(v_bs_x27_3073_, v_i_3060_, v_a_3071_);
v_i_3060_ = v___x_3075_;
v_bs_3061_ = v___x_3076_;
goto _start;
}
else
{
lean_object* v_a_3078_; lean_object* v___x_3080_; uint8_t v_isShared_3081_; uint8_t v_isSharedCheck_3085_; 
lean_dec_ref(v_bs_3061_);
lean_dec_ref(v_filePath_3058_);
v_a_3078_ = lean_ctor_get(v___x_3070_, 0);
v_isSharedCheck_3085_ = !lean_is_exclusive(v___x_3070_);
if (v_isSharedCheck_3085_ == 0)
{
v___x_3080_ = v___x_3070_;
v_isShared_3081_ = v_isSharedCheck_3085_;
goto v_resetjp_3079_;
}
else
{
lean_inc(v_a_3078_);
lean_dec(v___x_3070_);
v___x_3080_ = lean_box(0);
v_isShared_3081_ = v_isSharedCheck_3085_;
goto v_resetjp_3079_;
}
v_resetjp_3079_:
{
lean_object* v___x_3083_; 
if (v_isShared_3081_ == 0)
{
v___x_3083_ = v___x_3080_;
goto v_reusejp_3082_;
}
else
{
lean_object* v_reuseFailAlloc_3084_; 
v_reuseFailAlloc_3084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3084_, 0, v_a_3078_);
v___x_3083_ = v_reuseFailAlloc_3084_;
goto v_reusejp_3082_;
}
v_reusejp_3082_:
{
return v___x_3083_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_printWarnings_spec__0___boxed(lean_object* v_useErrorFormat_3086_, lean_object* v_filePath_3087_, lean_object* v_sz_3088_, lean_object* v_i_3089_, lean_object* v_bs_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_){
_start:
{
uint8_t v_useErrorFormat_boxed_3094_; size_t v_sz_boxed_3095_; size_t v_i_boxed_3096_; lean_object* v_res_3097_; 
v_useErrorFormat_boxed_3094_ = lean_unbox(v_useErrorFormat_3086_);
v_sz_boxed_3095_ = lean_unbox_usize(v_sz_3088_);
lean_dec(v_sz_3088_);
v_i_boxed_3096_ = lean_unbox_usize(v_i_3089_);
lean_dec(v_i_3089_);
v_res_3097_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_printWarnings_spec__0(v_useErrorFormat_boxed_3094_, v_filePath_3087_, v_sz_boxed_3095_, v_i_boxed_3096_, v_bs_3090_, v___y_3091_, v___y_3092_);
lean_dec(v___y_3092_);
lean_dec_ref(v___y_3091_);
return v_res_3097_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0(void){
_start:
{
lean_object* v___x_3098_; lean_object* v___x_3099_; 
v___x_3098_ = lean_box(1);
v___x_3099_ = l_Lean_MessageData_ofFormat(v___x_3098_);
return v___x_3099_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarnings(lean_object* v_results_3100_, lean_object* v_filePath_3101_, uint8_t v_useErrorFormat_3102_, lean_object* v_a_3103_, lean_object* v_a_3104_){
_start:
{
lean_object* v___x_3106_; 
v___x_3106_ = lp_batteries_Batteries_Tactic_Lint_sortResults___redArg(v_results_3100_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3106_) == 0)
{
lean_object* v_a_3107_; size_t v_sz_3108_; size_t v___x_3109_; lean_object* v___x_3110_; 
v_a_3107_ = lean_ctor_get(v___x_3106_, 0);
lean_inc(v_a_3107_);
lean_dec_ref_known(v___x_3106_, 1);
v_sz_3108_ = lean_array_size(v_a_3107_);
v___x_3109_ = ((size_t)0ULL);
v___x_3110_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_printWarnings_spec__0(v_useErrorFormat_3102_, v_filePath_3101_, v_sz_3108_, v___x_3109_, v_a_3107_, v_a_3103_, v_a_3104_);
if (lean_obj_tag(v___x_3110_) == 0)
{
lean_object* v_a_3111_; lean_object* v___x_3113_; uint8_t v_isShared_3114_; uint8_t v_isSharedCheck_3121_; 
v_a_3111_ = lean_ctor_get(v___x_3110_, 0);
v_isSharedCheck_3121_ = !lean_is_exclusive(v___x_3110_);
if (v_isSharedCheck_3121_ == 0)
{
v___x_3113_ = v___x_3110_;
v_isShared_3114_ = v_isSharedCheck_3121_;
goto v_resetjp_3112_;
}
else
{
lean_inc(v_a_3111_);
lean_dec(v___x_3110_);
v___x_3113_ = lean_box(0);
v_isShared_3114_ = v_isSharedCheck_3121_;
goto v_resetjp_3112_;
}
v_resetjp_3112_:
{
lean_object* v___x_3115_; lean_object* v___x_3116_; lean_object* v___x_3117_; lean_object* v___x_3119_; 
v___x_3115_ = lean_array_to_list(v_a_3111_);
v___x_3116_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0, &lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0);
v___x_3117_ = l_Lean_MessageData_joinSep(v___x_3115_, v___x_3116_);
if (v_isShared_3114_ == 0)
{
lean_ctor_set(v___x_3113_, 0, v___x_3117_);
v___x_3119_ = v___x_3113_;
goto v_reusejp_3118_;
}
else
{
lean_object* v_reuseFailAlloc_3120_; 
v_reuseFailAlloc_3120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3120_, 0, v___x_3117_);
v___x_3119_ = v_reuseFailAlloc_3120_;
goto v_reusejp_3118_;
}
v_reusejp_3118_:
{
return v___x_3119_;
}
}
}
else
{
lean_object* v_a_3122_; lean_object* v___x_3124_; uint8_t v_isShared_3125_; uint8_t v_isSharedCheck_3129_; 
v_a_3122_ = lean_ctor_get(v___x_3110_, 0);
v_isSharedCheck_3129_ = !lean_is_exclusive(v___x_3110_);
if (v_isSharedCheck_3129_ == 0)
{
v___x_3124_ = v___x_3110_;
v_isShared_3125_ = v_isSharedCheck_3129_;
goto v_resetjp_3123_;
}
else
{
lean_inc(v_a_3122_);
lean_dec(v___x_3110_);
v___x_3124_ = lean_box(0);
v_isShared_3125_ = v_isSharedCheck_3129_;
goto v_resetjp_3123_;
}
v_resetjp_3123_:
{
lean_object* v___x_3127_; 
if (v_isShared_3125_ == 0)
{
v___x_3127_ = v___x_3124_;
goto v_reusejp_3126_;
}
else
{
lean_object* v_reuseFailAlloc_3128_; 
v_reuseFailAlloc_3128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3128_, 0, v_a_3122_);
v___x_3127_ = v_reuseFailAlloc_3128_;
goto v_reusejp_3126_;
}
v_reusejp_3126_:
{
return v___x_3127_;
}
}
}
}
else
{
lean_object* v_a_3130_; lean_object* v___x_3132_; uint8_t v_isShared_3133_; uint8_t v_isSharedCheck_3137_; 
lean_dec_ref(v_filePath_3101_);
v_a_3130_ = lean_ctor_get(v___x_3106_, 0);
v_isSharedCheck_3137_ = !lean_is_exclusive(v___x_3106_);
if (v_isSharedCheck_3137_ == 0)
{
v___x_3132_ = v___x_3106_;
v_isShared_3133_ = v_isSharedCheck_3137_;
goto v_resetjp_3131_;
}
else
{
lean_inc(v_a_3130_);
lean_dec(v___x_3106_);
v___x_3132_ = lean_box(0);
v_isShared_3133_ = v_isSharedCheck_3137_;
goto v_resetjp_3131_;
}
v_resetjp_3131_:
{
lean_object* v___x_3135_; 
if (v_isShared_3133_ == 0)
{
v___x_3135_ = v___x_3132_;
goto v_reusejp_3134_;
}
else
{
lean_object* v_reuseFailAlloc_3136_; 
v_reuseFailAlloc_3136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3136_, 0, v_a_3130_);
v___x_3135_ = v_reuseFailAlloc_3136_;
goto v_reusejp_3134_;
}
v_reusejp_3134_:
{
return v___x_3135_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_printWarnings___boxed(lean_object* v_results_3138_, lean_object* v_filePath_3139_, lean_object* v_useErrorFormat_3140_, lean_object* v_a_3141_, lean_object* v_a_3142_, lean_object* v_a_3143_){
_start:
{
uint8_t v_useErrorFormat_boxed_3144_; lean_object* v_res_3145_; 
v_useErrorFormat_boxed_3144_ = lean_unbox(v_useErrorFormat_3140_);
v_res_3145_ = lp_batteries_Batteries_Tactic_Lint_printWarnings(v_results_3138_, v_filePath_3139_, v_useErrorFormat_boxed_3144_, v_a_3141_, v_a_3142_);
lean_dec(v_a_3142_);
lean_dec_ref(v_a_3141_);
lean_dec_ref(v_results_3138_);
return v_res_3145_;
}
}
static lean_object* _init_lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__1(void){
_start:
{
lean_object* v___x_3147_; lean_object* v___x_3148_; 
v___x_3147_ = ((lean_object*)(lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__0));
v___x_3148_ = l_Lean_stringToMessageData(v___x_3147_);
return v___x_3148_;
}
}
static lean_object* _init_lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3(void){
_start:
{
lean_object* v___x_3150_; lean_object* v___x_3151_; 
v___x_3150_ = ((lean_object*)(lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__2));
v___x_3151_ = l_Lean_stringToMessageData(v___x_3150_);
return v___x_3151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0(uint8_t v_useErrorFormat_3152_, lean_object* v_x_3153_, lean_object* v_x_3154_, lean_object* v___y_3155_, lean_object* v___y_3156_){
_start:
{
if (lean_obj_tag(v_x_3153_) == 0)
{
lean_object* v___x_3158_; lean_object* v___x_3159_; 
v___x_3158_ = l_List_reverse___redArg(v_x_3154_);
v___x_3159_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3159_, 0, v___x_3158_);
return v___x_3159_;
}
else
{
lean_object* v_head_3160_; lean_object* v_tail_3161_; lean_object* v___x_3163_; uint8_t v_isShared_3164_; uint8_t v_isSharedCheck_3204_; 
v_head_3160_ = lean_ctor_get(v_x_3153_, 0);
v_tail_3161_ = lean_ctor_get(v_x_3153_, 1);
v_isSharedCheck_3204_ = !lean_is_exclusive(v_x_3153_);
if (v_isSharedCheck_3204_ == 0)
{
v___x_3163_ = v_x_3153_;
v_isShared_3164_ = v_isSharedCheck_3204_;
goto v_resetjp_3162_;
}
else
{
lean_inc(v_tail_3161_);
lean_inc(v_head_3160_);
lean_dec(v_x_3153_);
v___x_3163_ = lean_box(0);
v_isShared_3164_ = v_isSharedCheck_3204_;
goto v_resetjp_3162_;
}
v_resetjp_3162_:
{
lean_object* v_a_3166_; lean_object* v_snd_3171_; lean_object* v_fst_3172_; lean_object* v___x_3174_; uint8_t v_isShared_3175_; uint8_t v_isSharedCheck_3203_; 
v_snd_3171_ = lean_ctor_get(v_head_3160_, 1);
v_fst_3172_ = lean_ctor_get(v_head_3160_, 0);
v_isSharedCheck_3203_ = !lean_is_exclusive(v_head_3160_);
if (v_isSharedCheck_3203_ == 0)
{
v___x_3174_ = v_head_3160_;
v_isShared_3175_ = v_isSharedCheck_3203_;
goto v_resetjp_3173_;
}
else
{
lean_inc(v_snd_3171_);
lean_inc(v_fst_3172_);
lean_dec(v_head_3160_);
v___x_3174_ = lean_box(0);
v_isShared_3175_ = v_isSharedCheck_3203_;
goto v_resetjp_3173_;
}
v___jp_3165_:
{
lean_object* v___x_3168_; 
if (v_isShared_3164_ == 0)
{
lean_ctor_set(v___x_3163_, 1, v_x_3154_);
lean_ctor_set(v___x_3163_, 0, v_a_3166_);
v___x_3168_ = v___x_3163_;
goto v_reusejp_3167_;
}
else
{
lean_object* v_reuseFailAlloc_3170_; 
v_reuseFailAlloc_3170_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3170_, 0, v_a_3166_);
lean_ctor_set(v_reuseFailAlloc_3170_, 1, v_x_3154_);
v___x_3168_ = v_reuseFailAlloc_3170_;
goto v_reusejp_3167_;
}
v_reusejp_3167_:
{
v_x_3153_ = v_tail_3161_;
v_x_3154_ = v___x_3168_;
goto _start;
}
}
v_resetjp_3173_:
{
lean_object* v_fst_3176_; lean_object* v_snd_3177_; lean_object* v___x_3179_; uint8_t v_isShared_3180_; uint8_t v_isSharedCheck_3202_; 
v_fst_3176_ = lean_ctor_get(v_snd_3171_, 0);
v_snd_3177_ = lean_ctor_get(v_snd_3171_, 1);
v_isSharedCheck_3202_ = !lean_is_exclusive(v_snd_3171_);
if (v_isSharedCheck_3202_ == 0)
{
v___x_3179_ = v_snd_3171_;
v_isShared_3180_ = v_isSharedCheck_3202_;
goto v_resetjp_3178_;
}
else
{
lean_inc(v_snd_3177_);
lean_inc(v_fst_3176_);
lean_dec(v_snd_3171_);
v___x_3179_ = lean_box(0);
v_isShared_3180_ = v_isSharedCheck_3202_;
goto v_resetjp_3178_;
}
v_resetjp_3178_:
{
lean_object* v___x_3181_; 
v___x_3181_ = lp_batteries_Batteries_Tactic_Lint_printWarnings(v_snd_3177_, v_fst_3176_, v_useErrorFormat_3152_, v___y_3155_, v___y_3156_);
lean_dec(v_snd_3177_);
if (lean_obj_tag(v___x_3181_) == 0)
{
lean_object* v_a_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3186_; 
v_a_3182_ = lean_ctor_get(v___x_3181_, 0);
lean_inc(v_a_3182_);
lean_dec_ref_known(v___x_3181_, 1);
v___x_3183_ = lean_obj_once(&lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__1, &lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__1_once, _init_lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__1);
v___x_3184_ = l_Lean_MessageData_ofName(v_fst_3172_);
if (v_isShared_3180_ == 0)
{
lean_ctor_set_tag(v___x_3179_, 7);
lean_ctor_set(v___x_3179_, 1, v___x_3184_);
lean_ctor_set(v___x_3179_, 0, v___x_3183_);
v___x_3186_ = v___x_3179_;
goto v_reusejp_3185_;
}
else
{
lean_object* v_reuseFailAlloc_3192_; 
v_reuseFailAlloc_3192_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3192_, 0, v___x_3183_);
lean_ctor_set(v_reuseFailAlloc_3192_, 1, v___x_3184_);
v___x_3186_ = v_reuseFailAlloc_3192_;
goto v_reusejp_3185_;
}
v_reusejp_3185_:
{
lean_object* v___x_3187_; lean_object* v___x_3189_; 
v___x_3187_ = lean_obj_once(&lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3, &lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3_once, _init_lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3);
if (v_isShared_3175_ == 0)
{
lean_ctor_set_tag(v___x_3174_, 7);
lean_ctor_set(v___x_3174_, 1, v___x_3187_);
lean_ctor_set(v___x_3174_, 0, v___x_3186_);
v___x_3189_ = v___x_3174_;
goto v_reusejp_3188_;
}
else
{
lean_object* v_reuseFailAlloc_3191_; 
v_reuseFailAlloc_3191_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3191_, 0, v___x_3186_);
lean_ctor_set(v_reuseFailAlloc_3191_, 1, v___x_3187_);
v___x_3189_ = v_reuseFailAlloc_3191_;
goto v_reusejp_3188_;
}
v_reusejp_3188_:
{
lean_object* v___x_3190_; 
v___x_3190_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3190_, 0, v___x_3189_);
lean_ctor_set(v___x_3190_, 1, v_a_3182_);
v_a_3166_ = v___x_3190_;
goto v___jp_3165_;
}
}
}
else
{
lean_del_object(v___x_3179_);
lean_del_object(v___x_3174_);
lean_dec(v_fst_3172_);
if (lean_obj_tag(v___x_3181_) == 0)
{
lean_object* v_a_3193_; 
v_a_3193_ = lean_ctor_get(v___x_3181_, 0);
lean_inc(v_a_3193_);
lean_dec_ref_known(v___x_3181_, 1);
v_a_3166_ = v_a_3193_;
goto v___jp_3165_;
}
else
{
lean_object* v_a_3194_; lean_object* v___x_3196_; uint8_t v_isShared_3197_; uint8_t v_isSharedCheck_3201_; 
lean_del_object(v___x_3163_);
lean_dec(v_tail_3161_);
lean_dec(v_x_3154_);
v_a_3194_ = lean_ctor_get(v___x_3181_, 0);
v_isSharedCheck_3201_ = !lean_is_exclusive(v___x_3181_);
if (v_isSharedCheck_3201_ == 0)
{
v___x_3196_ = v___x_3181_;
v_isShared_3197_ = v_isSharedCheck_3201_;
goto v_resetjp_3195_;
}
else
{
lean_inc(v_a_3194_);
lean_dec(v___x_3181_);
v___x_3196_ = lean_box(0);
v_isShared_3197_ = v_isSharedCheck_3201_;
goto v_resetjp_3195_;
}
v_resetjp_3195_:
{
lean_object* v___x_3199_; 
if (v_isShared_3197_ == 0)
{
v___x_3199_ = v___x_3196_;
goto v_reusejp_3198_;
}
else
{
lean_object* v_reuseFailAlloc_3200_; 
v_reuseFailAlloc_3200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3200_, 0, v_a_3194_);
v___x_3199_ = v_reuseFailAlloc_3200_;
goto v_reusejp_3198_;
}
v_reusejp_3198_:
{
return v___x_3199_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___boxed(lean_object* v_useErrorFormat_3205_, lean_object* v_x_3206_, lean_object* v_x_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_, lean_object* v___y_3210_){
_start:
{
uint8_t v_useErrorFormat_boxed_3211_; lean_object* v_res_3212_; 
v_useErrorFormat_boxed_3211_ = lean_unbox(v_useErrorFormat_3205_);
v_res_3212_ = lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0(v_useErrorFormat_boxed_3211_, v_x_3206_, v_x_3207_, v___y_3208_, v___y_3209_);
lean_dec(v___y_3209_);
lean_dec_ref(v___y_3208_);
return v_res_3212_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0(uint8_t v___x_3213_, lean_object* v_x_3214_, lean_object* v_x_3215_){
_start:
{
lean_object* v_fst_3216_; lean_object* v_fst_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; uint8_t v___x_3220_; 
v_fst_3216_ = lean_ctor_get(v_x_3214_, 0);
lean_inc(v_fst_3216_);
lean_dec_ref(v_x_3214_);
v_fst_3217_ = lean_ctor_get(v_x_3215_, 0);
lean_inc(v_fst_3217_);
lean_dec_ref(v_x_3215_);
v___x_3218_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_3216_, v___x_3213_);
v___x_3219_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_3217_, v___x_3213_);
v___x_3220_ = lean_string_dec_lt(v___x_3218_, v___x_3219_);
lean_dec_ref(v___x_3219_);
lean_dec_ref(v___x_3218_);
return v___x_3220_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0___boxed(lean_object* v___x_3221_, lean_object* v_x_3222_, lean_object* v_x_3223_){
_start:
{
uint8_t v___x_5002__boxed_3224_; uint8_t v_res_3225_; lean_object* v_r_3226_; 
v___x_5002__boxed_3224_ = lean_unbox(v___x_3221_);
v_res_3225_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0(v___x_5002__boxed_3224_, v_x_3222_, v_x_3223_);
v_r_3226_ = lean_box(v_res_3225_);
return v_r_3226_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg(lean_object* v_hi_3227_, lean_object* v_pivot_3228_, lean_object* v_as_3229_, lean_object* v_i_3230_, lean_object* v_k_3231_){
_start:
{
uint8_t v___x_3232_; 
v___x_3232_ = lean_nat_dec_lt(v_k_3231_, v_hi_3227_);
if (v___x_3232_ == 0)
{
lean_object* v___x_3233_; lean_object* v___x_3234_; 
lean_dec(v_k_3231_);
lean_dec_ref(v_pivot_3228_);
v___x_3233_ = lean_array_fswap(v_as_3229_, v_i_3230_, v_hi_3227_);
v___x_3234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3234_, 0, v_i_3230_);
lean_ctor_set(v___x_3234_, 1, v___x_3233_);
return v___x_3234_;
}
else
{
lean_object* v___x_3235_; lean_object* v_fst_3236_; lean_object* v_fst_3237_; lean_object* v___x_3238_; lean_object* v___x_3239_; uint8_t v___x_3240_; 
v___x_3235_ = lean_array_fget_borrowed(v_as_3229_, v_k_3231_);
v_fst_3236_ = lean_ctor_get(v___x_3235_, 0);
v_fst_3237_ = lean_ctor_get(v_pivot_3228_, 0);
lean_inc(v_fst_3236_);
v___x_3238_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_3236_, v___x_3232_);
lean_inc(v_fst_3237_);
v___x_3239_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_fst_3237_, v___x_3232_);
v___x_3240_ = lean_string_dec_lt(v___x_3238_, v___x_3239_);
lean_dec_ref(v___x_3239_);
lean_dec_ref(v___x_3238_);
if (v___x_3240_ == 0)
{
lean_object* v___x_3241_; lean_object* v___x_3242_; 
v___x_3241_ = lean_unsigned_to_nat(1u);
v___x_3242_ = lean_nat_add(v_k_3231_, v___x_3241_);
lean_dec(v_k_3231_);
v_k_3231_ = v___x_3242_;
goto _start;
}
else
{
lean_object* v___x_3244_; lean_object* v___x_3245_; lean_object* v___x_3246_; lean_object* v___x_3247_; 
v___x_3244_ = lean_array_fswap(v_as_3229_, v_i_3230_, v_k_3231_);
v___x_3245_ = lean_unsigned_to_nat(1u);
v___x_3246_ = lean_nat_add(v_i_3230_, v___x_3245_);
lean_dec(v_i_3230_);
v___x_3247_ = lean_nat_add(v_k_3231_, v___x_3245_);
lean_dec(v_k_3231_);
v_as_3229_ = v___x_3244_;
v_i_3230_ = v___x_3246_;
v_k_3231_ = v___x_3247_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg___boxed(lean_object* v_hi_3249_, lean_object* v_pivot_3250_, lean_object* v_as_3251_, lean_object* v_i_3252_, lean_object* v_k_3253_){
_start:
{
lean_object* v_res_3254_; 
v_res_3254_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg(v_hi_3249_, v_pivot_3250_, v_as_3251_, v_i_3252_, v_k_3253_);
lean_dec(v_hi_3249_);
return v_res_3254_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg(lean_object* v_n_3255_, lean_object* v_as_3256_, lean_object* v_lo_3257_, lean_object* v_hi_3258_){
_start:
{
lean_object* v___y_3260_; uint8_t v___x_3270_; 
v___x_3270_ = lean_nat_dec_lt(v_lo_3257_, v_hi_3258_);
if (v___x_3270_ == 0)
{
lean_dec(v_lo_3257_);
return v_as_3256_;
}
else
{
lean_object* v___x_3271_; lean_object* v___x_3272_; lean_object* v_mid_3273_; lean_object* v___y_3275_; lean_object* v___y_3281_; lean_object* v___x_3286_; lean_object* v___x_3287_; uint8_t v___x_3288_; 
v___x_3271_ = lean_nat_add(v_lo_3257_, v_hi_3258_);
v___x_3272_ = lean_unsigned_to_nat(1u);
v_mid_3273_ = lean_nat_shiftr(v___x_3271_, v___x_3272_);
lean_dec(v___x_3271_);
v___x_3286_ = lean_array_fget_borrowed(v_as_3256_, v_mid_3273_);
v___x_3287_ = lean_array_fget_borrowed(v_as_3256_, v_lo_3257_);
lean_inc(v___x_3287_);
lean_inc(v___x_3286_);
v___x_3288_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0(v___x_3270_, v___x_3286_, v___x_3287_);
if (v___x_3288_ == 0)
{
v___y_3281_ = v_as_3256_;
goto v___jp_3280_;
}
else
{
lean_object* v___x_3289_; 
v___x_3289_ = lean_array_fswap(v_as_3256_, v_lo_3257_, v_mid_3273_);
v___y_3281_ = v___x_3289_;
goto v___jp_3280_;
}
v___jp_3274_:
{
lean_object* v___x_3276_; lean_object* v___x_3277_; uint8_t v___x_3278_; 
v___x_3276_ = lean_array_fget_borrowed(v___y_3275_, v_mid_3273_);
v___x_3277_ = lean_array_fget_borrowed(v___y_3275_, v_hi_3258_);
lean_inc(v___x_3277_);
lean_inc(v___x_3276_);
v___x_3278_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0(v___x_3270_, v___x_3276_, v___x_3277_);
if (v___x_3278_ == 0)
{
lean_dec(v_mid_3273_);
v___y_3260_ = v___y_3275_;
goto v___jp_3259_;
}
else
{
lean_object* v___x_3279_; 
v___x_3279_ = lean_array_fswap(v___y_3275_, v_mid_3273_, v_hi_3258_);
lean_dec(v_mid_3273_);
v___y_3260_ = v___x_3279_;
goto v___jp_3259_;
}
}
v___jp_3280_:
{
lean_object* v___x_3282_; lean_object* v___x_3283_; uint8_t v___x_3284_; 
v___x_3282_ = lean_array_fget_borrowed(v___y_3281_, v_hi_3258_);
v___x_3283_ = lean_array_fget_borrowed(v___y_3281_, v_lo_3257_);
lean_inc(v___x_3283_);
lean_inc(v___x_3282_);
v___x_3284_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___lam__0(v___x_3270_, v___x_3282_, v___x_3283_);
if (v___x_3284_ == 0)
{
v___y_3275_ = v___y_3281_;
goto v___jp_3274_;
}
else
{
lean_object* v___x_3285_; 
v___x_3285_ = lean_array_fswap(v___y_3281_, v_lo_3257_, v_hi_3258_);
v___y_3275_ = v___x_3285_;
goto v___jp_3274_;
}
}
}
v___jp_3259_:
{
lean_object* v_pivot_3261_; lean_object* v___x_3262_; lean_object* v_fst_3263_; lean_object* v_snd_3264_; uint8_t v___x_3265_; 
v_pivot_3261_ = lean_array_fget(v___y_3260_, v_hi_3258_);
lean_inc_n(v_lo_3257_, 2);
v___x_3262_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg(v_hi_3258_, v_pivot_3261_, v___y_3260_, v_lo_3257_, v_lo_3257_);
v_fst_3263_ = lean_ctor_get(v___x_3262_, 0);
lean_inc(v_fst_3263_);
v_snd_3264_ = lean_ctor_get(v___x_3262_, 1);
lean_inc(v_snd_3264_);
lean_dec_ref(v___x_3262_);
v___x_3265_ = lean_nat_dec_le(v_hi_3258_, v_fst_3263_);
if (v___x_3265_ == 0)
{
lean_object* v___x_3266_; lean_object* v___x_3267_; lean_object* v___x_3268_; 
v___x_3266_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg(v_n_3255_, v_snd_3264_, v_lo_3257_, v_fst_3263_);
v___x_3267_ = lean_unsigned_to_nat(1u);
v___x_3268_ = lean_nat_add(v_fst_3263_, v___x_3267_);
lean_dec(v_fst_3263_);
v_as_3256_ = v___x_3266_;
v_lo_3257_ = v___x_3268_;
goto _start;
}
else
{
lean_dec(v_fst_3263_);
lean_dec(v_lo_3257_);
return v_snd_3264_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg___boxed(lean_object* v_n_3290_, lean_object* v_as_3291_, lean_object* v_lo_3292_, lean_object* v_hi_3293_){
_start:
{
lean_object* v_res_3294_; 
v_res_3294_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg(v_n_3290_, v_as_3291_, v_lo_3292_, v_hi_3293_);
lean_dec(v_hi_3293_);
lean_dec(v_n_3290_);
return v_res_3294_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__2(lean_object* v_x_3295_, lean_object* v_x_3296_){
_start:
{
if (lean_obj_tag(v_x_3296_) == 0)
{
return v_x_3295_;
}
else
{
lean_object* v_key_3297_; lean_object* v_value_3298_; lean_object* v_tail_3299_; lean_object* v___x_3300_; lean_object* v___x_3301_; 
v_key_3297_ = lean_ctor_get(v_x_3296_, 0);
v_value_3298_ = lean_ctor_get(v_x_3296_, 1);
v_tail_3299_ = lean_ctor_get(v_x_3296_, 2);
lean_inc(v_value_3298_);
lean_inc(v_key_3297_);
v___x_3300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3300_, 0, v_key_3297_);
lean_ctor_set(v___x_3300_, 1, v_value_3298_);
v___x_3301_ = lean_array_push(v_x_3295_, v___x_3300_);
v_x_3295_ = v___x_3301_;
v_x_3296_ = v_tail_3299_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__2___boxed(lean_object* v_x_3303_, lean_object* v_x_3304_){
_start:
{
lean_object* v_res_3305_; 
v_res_3305_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__2(v_x_3303_, v_x_3304_);
lean_dec(v_x_3304_);
return v_res_3305_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3(lean_object* v_as_3306_, size_t v_i_3307_, size_t v_stop_3308_, lean_object* v_b_3309_){
_start:
{
uint8_t v___x_3310_; 
v___x_3310_ = lean_usize_dec_eq(v_i_3307_, v_stop_3308_);
if (v___x_3310_ == 0)
{
lean_object* v___x_3311_; lean_object* v___x_3312_; size_t v___x_3313_; size_t v___x_3314_; 
v___x_3311_ = lean_array_uget_borrowed(v_as_3306_, v_i_3307_);
v___x_3312_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__2(v_b_3309_, v___x_3311_);
v___x_3313_ = ((size_t)1ULL);
v___x_3314_ = lean_usize_add(v_i_3307_, v___x_3313_);
v_i_3307_ = v___x_3314_;
v_b_3309_ = v___x_3312_;
goto _start;
}
else
{
return v_b_3309_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3___boxed(lean_object* v_as_3316_, lean_object* v_i_3317_, lean_object* v_stop_3318_, lean_object* v_b_3319_){
_start:
{
size_t v_i_boxed_3320_; size_t v_stop_boxed_3321_; lean_object* v_res_3322_; 
v_i_boxed_3320_ = lean_unbox_usize(v_i_3317_);
lean_dec(v_i_3317_);
v_stop_boxed_3321_ = lean_unbox_usize(v_stop_3318_);
lean_dec(v_stop_3318_);
v_res_3322_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3(v_as_3316_, v_i_boxed_3320_, v_stop_boxed_3321_, v_b_3319_);
lean_dec_ref(v_as_3316_);
return v_res_3322_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0(lean_object* v_key_3323_, lean_object* v_value_3324_, lean_object* v_fp_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_){
_start:
{
lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; 
v___x_3329_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7);
v___x_3330_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(v___x_3329_, v_key_3323_, v_value_3324_);
v___x_3331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3331_, 0, v_fp_3325_);
lean_ctor_set(v___x_3331_, 1, v___x_3330_);
v___x_3332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3332_, 0, v___x_3331_);
return v___x_3332_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0___boxed(lean_object* v_key_3333_, lean_object* v_value_3334_, lean_object* v_fp_3335_, lean_object* v___y_3336_, lean_object* v___y_3337_, lean_object* v___y_3338_){
_start:
{
lean_object* v_res_3339_; 
v_res_3339_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0(v_key_3333_, v_value_3334_, v_fp_3335_, v___y_3336_, v___y_3337_);
lean_dec(v___y_3337_);
lean_dec_ref(v___y_3336_);
return v_res_3339_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4_spec__5(lean_object* v_constName_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_){
_start:
{
lean_object* v___x_3344_; lean_object* v_env_3345_; uint8_t v___x_3346_; lean_object* v___x_3347_; 
v___x_3344_ = lean_st_ref_get(v___y_3342_);
v_env_3345_ = lean_ctor_get(v___x_3344_, 0);
lean_inc_ref(v_env_3345_);
lean_dec(v___x_3344_);
v___x_3346_ = 0;
lean_inc(v_constName_3340_);
v___x_3347_ = l_Lean_Environment_find_x3f(v_env_3345_, v_constName_3340_, v___x_3346_);
if (lean_obj_tag(v___x_3347_) == 0)
{
lean_object* v___x_3348_; 
v___x_3348_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1___redArg(v_constName_3340_, v___y_3341_, v___y_3342_);
return v___x_3348_;
}
else
{
lean_object* v_val_3349_; lean_object* v___x_3351_; uint8_t v_isShared_3352_; uint8_t v_isSharedCheck_3356_; 
lean_dec(v_constName_3340_);
v_val_3349_ = lean_ctor_get(v___x_3347_, 0);
v_isSharedCheck_3356_ = !lean_is_exclusive(v___x_3347_);
if (v_isSharedCheck_3356_ == 0)
{
v___x_3351_ = v___x_3347_;
v_isShared_3352_ = v_isSharedCheck_3356_;
goto v_resetjp_3350_;
}
else
{
lean_inc(v_val_3349_);
lean_dec(v___x_3347_);
v___x_3351_ = lean_box(0);
v_isShared_3352_ = v_isSharedCheck_3356_;
goto v_resetjp_3350_;
}
v_resetjp_3350_:
{
lean_object* v___x_3354_; 
if (v_isShared_3352_ == 0)
{
lean_ctor_set_tag(v___x_3351_, 0);
v___x_3354_ = v___x_3351_;
goto v_reusejp_3353_;
}
else
{
lean_object* v_reuseFailAlloc_3355_; 
v_reuseFailAlloc_3355_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3355_, 0, v_val_3349_);
v___x_3354_ = v_reuseFailAlloc_3355_;
goto v_reusejp_3353_;
}
v_reusejp_3353_:
{
return v___x_3354_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4_spec__5___boxed(lean_object* v_constName_3357_, lean_object* v___y_3358_, lean_object* v___y_3359_, lean_object* v___y_3360_){
_start:
{
lean_object* v_res_3361_; 
v_res_3361_ = lp_batteries_Lean_getConstInfo___at___00Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4_spec__5(v_constName_3357_, v___y_3358_, v___y_3359_);
lean_dec(v___y_3359_);
lean_dec_ref(v___y_3358_);
return v_res_3361_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4(lean_object* v_declName_3362_, lean_object* v___y_3363_, lean_object* v___y_3364_){
_start:
{
lean_object* v___x_3366_; 
lean_inc(v_declName_3362_);
v___x_3366_ = lp_batteries_Lean_getConstInfo___at___00Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4_spec__5(v_declName_3362_, v___y_3363_, v___y_3364_);
if (lean_obj_tag(v___x_3366_) == 0)
{
lean_object* v___x_3368_; uint8_t v_isShared_3369_; uint8_t v_isSharedCheck_3393_; 
v_isSharedCheck_3393_ = !lean_is_exclusive(v___x_3366_);
if (v_isSharedCheck_3393_ == 0)
{
lean_object* v_unused_3394_; 
v_unused_3394_ = lean_ctor_get(v___x_3366_, 0);
lean_dec(v_unused_3394_);
v___x_3368_ = v___x_3366_;
v_isShared_3369_ = v_isSharedCheck_3393_;
goto v_resetjp_3367_;
}
else
{
lean_dec(v___x_3366_);
v___x_3368_ = lean_box(0);
v_isShared_3369_ = v_isSharedCheck_3393_;
goto v_resetjp_3367_;
}
v_resetjp_3367_:
{
lean_object* v___x_3370_; lean_object* v_env_3371_; lean_object* v___x_3372_; 
v___x_3370_ = lean_st_ref_get(v___y_3364_);
v_env_3371_ = lean_ctor_get(v___x_3370_, 0);
lean_inc_ref(v_env_3371_);
lean_dec(v___x_3370_);
v___x_3372_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_3371_, v_declName_3362_);
lean_dec(v_declName_3362_);
lean_dec_ref(v_env_3371_);
if (lean_obj_tag(v___x_3372_) == 0)
{
lean_object* v___x_3373_; lean_object* v___x_3375_; 
v___x_3373_ = lean_box(0);
if (v_isShared_3369_ == 0)
{
lean_ctor_set(v___x_3368_, 0, v___x_3373_);
v___x_3375_ = v___x_3368_;
goto v_reusejp_3374_;
}
else
{
lean_object* v_reuseFailAlloc_3376_; 
v_reuseFailAlloc_3376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3376_, 0, v___x_3373_);
v___x_3375_ = v_reuseFailAlloc_3376_;
goto v_reusejp_3374_;
}
v_reusejp_3374_:
{
return v___x_3375_;
}
}
else
{
lean_object* v_val_3377_; lean_object* v___x_3379_; uint8_t v_isShared_3380_; uint8_t v_isSharedCheck_3392_; 
v_val_3377_ = lean_ctor_get(v___x_3372_, 0);
v_isSharedCheck_3392_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3392_ == 0)
{
v___x_3379_ = v___x_3372_;
v_isShared_3380_ = v_isSharedCheck_3392_;
goto v_resetjp_3378_;
}
else
{
lean_inc(v_val_3377_);
lean_dec(v___x_3372_);
v___x_3379_ = lean_box(0);
v_isShared_3380_ = v_isSharedCheck_3392_;
goto v_resetjp_3378_;
}
v_resetjp_3378_:
{
lean_object* v___x_3381_; lean_object* v_env_3382_; lean_object* v___x_3383_; lean_object* v___x_3384_; lean_object* v___x_3385_; lean_object* v___x_3387_; 
v___x_3381_ = lean_st_ref_get(v___y_3364_);
v_env_3382_ = lean_ctor_get(v___x_3381_, 0);
lean_inc_ref(v_env_3382_);
lean_dec(v___x_3381_);
v___x_3383_ = lean_box(0);
v___x_3384_ = l_Lean_Environment_allImportedModuleNames(v_env_3382_);
lean_dec_ref(v_env_3382_);
v___x_3385_ = lean_array_get(v___x_3383_, v___x_3384_, v_val_3377_);
lean_dec(v_val_3377_);
lean_dec_ref(v___x_3384_);
if (v_isShared_3380_ == 0)
{
lean_ctor_set(v___x_3379_, 0, v___x_3385_);
v___x_3387_ = v___x_3379_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3391_; 
v_reuseFailAlloc_3391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3391_, 0, v___x_3385_);
v___x_3387_ = v_reuseFailAlloc_3391_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
lean_object* v___x_3389_; 
if (v_isShared_3369_ == 0)
{
lean_ctor_set(v___x_3368_, 0, v___x_3387_);
v___x_3389_ = v___x_3368_;
goto v_reusejp_3388_;
}
else
{
lean_object* v_reuseFailAlloc_3390_; 
v_reuseFailAlloc_3390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3390_, 0, v___x_3387_);
v___x_3389_ = v_reuseFailAlloc_3390_;
goto v_reusejp_3388_;
}
v_reusejp_3388_:
{
return v___x_3389_;
}
}
}
}
}
}
else
{
lean_object* v_a_3395_; lean_object* v___x_3397_; uint8_t v_isShared_3398_; uint8_t v_isSharedCheck_3402_; 
lean_dec(v_declName_3362_);
v_a_3395_ = lean_ctor_get(v___x_3366_, 0);
v_isSharedCheck_3402_ = !lean_is_exclusive(v___x_3366_);
if (v_isSharedCheck_3402_ == 0)
{
v___x_3397_ = v___x_3366_;
v_isShared_3398_ = v_isSharedCheck_3402_;
goto v_resetjp_3396_;
}
else
{
lean_inc(v_a_3395_);
lean_dec(v___x_3366_);
v___x_3397_ = lean_box(0);
v_isShared_3398_ = v_isSharedCheck_3402_;
goto v_resetjp_3396_;
}
v_resetjp_3396_:
{
lean_object* v___x_3400_; 
if (v_isShared_3398_ == 0)
{
v___x_3400_ = v___x_3397_;
goto v_reusejp_3399_;
}
else
{
lean_object* v_reuseFailAlloc_3401_; 
v_reuseFailAlloc_3401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3401_, 0, v_a_3395_);
v___x_3400_ = v_reuseFailAlloc_3401_;
goto v_reusejp_3399_;
}
v_reusejp_3399_:
{
return v___x_3400_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4___boxed(lean_object* v_declName_3403_, lean_object* v___y_3404_, lean_object* v___y_3405_, lean_object* v___y_3406_){
_start:
{
lean_object* v_res_3407_; 
v_res_3407_ = lp_batteries_Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4(v_declName_3403_, v___y_3404_, v___y_3405_);
lean_dec(v___y_3405_);
lean_dec_ref(v___y_3404_);
return v_res_3407_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg(lean_object* v_a_3408_, lean_object* v_x_3409_){
_start:
{
if (lean_obj_tag(v_x_3409_) == 0)
{
lean_object* v___x_3410_; 
v___x_3410_ = lean_box(0);
return v___x_3410_;
}
else
{
lean_object* v_key_3411_; lean_object* v_value_3412_; lean_object* v_tail_3413_; uint8_t v___x_3414_; 
v_key_3411_ = lean_ctor_get(v_x_3409_, 0);
v_value_3412_ = lean_ctor_get(v_x_3409_, 1);
v_tail_3413_ = lean_ctor_get(v_x_3409_, 2);
v___x_3414_ = lean_name_eq(v_key_3411_, v_a_3408_);
if (v___x_3414_ == 0)
{
v_x_3409_ = v_tail_3413_;
goto _start;
}
else
{
lean_object* v___x_3416_; 
lean_inc(v_value_3412_);
v___x_3416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3416_, 0, v_value_3412_);
return v___x_3416_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg___boxed(lean_object* v_a_3417_, lean_object* v_x_3418_){
_start:
{
lean_object* v_res_3419_; 
v_res_3419_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg(v_a_3417_, v_x_3418_);
lean_dec(v_x_3418_);
lean_dec(v_a_3417_);
return v_res_3419_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg(lean_object* v_m_3420_, lean_object* v_a_3421_){
_start:
{
lean_object* v_buckets_3422_; lean_object* v___x_3423_; uint64_t v___y_3425_; 
v_buckets_3422_ = lean_ctor_get(v_m_3420_, 1);
v___x_3423_ = lean_array_get_size(v_buckets_3422_);
if (lean_obj_tag(v_a_3421_) == 0)
{
uint64_t v___x_3439_; 
v___x_3439_ = 1723ULL;
v___y_3425_ = v___x_3439_;
goto v___jp_3424_;
}
else
{
uint64_t v_hash_3440_; 
v_hash_3440_ = lean_ctor_get_uint64(v_a_3421_, sizeof(void*)*2);
v___y_3425_ = v_hash_3440_;
goto v___jp_3424_;
}
v___jp_3424_:
{
uint64_t v___x_3426_; uint64_t v___x_3427_; uint64_t v_fold_3428_; uint64_t v___x_3429_; uint64_t v___x_3430_; uint64_t v___x_3431_; size_t v___x_3432_; size_t v___x_3433_; size_t v___x_3434_; size_t v___x_3435_; size_t v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; 
v___x_3426_ = 32ULL;
v___x_3427_ = lean_uint64_shift_right(v___y_3425_, v___x_3426_);
v_fold_3428_ = lean_uint64_xor(v___y_3425_, v___x_3427_);
v___x_3429_ = 16ULL;
v___x_3430_ = lean_uint64_shift_right(v_fold_3428_, v___x_3429_);
v___x_3431_ = lean_uint64_xor(v_fold_3428_, v___x_3430_);
v___x_3432_ = lean_uint64_to_usize(v___x_3431_);
v___x_3433_ = lean_usize_of_nat(v___x_3423_);
v___x_3434_ = ((size_t)1ULL);
v___x_3435_ = lean_usize_sub(v___x_3433_, v___x_3434_);
v___x_3436_ = lean_usize_land(v___x_3432_, v___x_3435_);
v___x_3437_ = lean_array_uget_borrowed(v_buckets_3422_, v___x_3436_);
v___x_3438_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg(v_a_3421_, v___x_3437_);
return v___x_3438_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg___boxed(lean_object* v_m_3441_, lean_object* v_a_3442_){
_start:
{
lean_object* v_res_3443_; 
v_res_3443_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg(v_m_3441_, v_a_3442_);
lean_dec(v_a_3442_);
lean_dec_ref(v_m_3441_);
return v_res_3443_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6(uint8_t v_useErrorFormat_3445_, lean_object* v_sp_3446_, lean_object* v_x_3447_, lean_object* v_x_3448_, lean_object* v___y_3449_, lean_object* v___y_3450_){
_start:
{
if (lean_obj_tag(v_x_3448_) == 0)
{
lean_object* v___x_3452_; 
lean_dec(v_sp_3446_);
v___x_3452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3452_, 0, v_x_3447_);
return v___x_3452_;
}
else
{
lean_object* v_key_3453_; lean_object* v_value_3454_; lean_object* v_tail_3455_; lean_object* v___y_3457_; lean_object* v_a_3458_; lean_object* v___y_3462_; lean_object* v___y_3463_; lean_object* v___x_3465_; 
v_key_3453_ = lean_ctor_get(v_x_3448_, 0);
lean_inc_n(v_key_3453_, 2);
v_value_3454_ = lean_ctor_get(v_x_3448_, 1);
lean_inc(v_value_3454_);
v_tail_3455_ = lean_ctor_get(v_x_3448_, 2);
lean_inc(v_tail_3455_);
lean_dec_ref_known(v_x_3448_, 3);
v___x_3465_ = lp_batteries_Lean_findModuleOf_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__4(v_key_3453_, v___y_3449_, v___y_3450_);
if (lean_obj_tag(v___x_3465_) == 0)
{
lean_object* v_a_3466_; lean_object* v___x_3467_; lean_object* v___y_3469_; 
v_a_3466_ = lean_ctor_get(v___x_3465_, 0);
lean_inc(v_a_3466_);
lean_dec_ref_known(v___x_3465_, 1);
v___x_3467_ = lean_st_ref_get(v___y_3450_);
if (lean_obj_tag(v_a_3466_) == 0)
{
lean_object* v_env_3505_; lean_object* v___x_3506_; 
v_env_3505_ = lean_ctor_get(v___x_3467_, 0);
lean_inc_ref(v_env_3505_);
lean_dec(v___x_3467_);
v___x_3506_ = l_Lean_Environment_mainModule(v_env_3505_);
lean_dec_ref(v_env_3505_);
v___y_3469_ = v___x_3506_;
goto v___jp_3468_;
}
else
{
lean_object* v_val_3507_; 
lean_dec(v___x_3467_);
v_val_3507_ = lean_ctor_get(v_a_3466_, 0);
lean_inc(v_val_3507_);
lean_dec_ref_known(v_a_3466_, 1);
v___y_3469_ = v_val_3507_;
goto v___jp_3468_;
}
v___jp_3468_:
{
lean_object* v___x_3470_; 
v___x_3470_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg(v_x_3447_, v___y_3469_);
if (lean_obj_tag(v___x_3470_) == 0)
{
if (v_useErrorFormat_3445_ == 0)
{
lean_object* v___x_3471_; lean_object* v___x_3472_; 
v___x_3471_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___x_3472_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0(v_key_3453_, v_value_3454_, v___x_3471_, v___y_3449_, v___y_3450_);
v___y_3462_ = v___y_3469_;
v___y_3463_ = v___x_3472_;
goto v___jp_3461_;
}
else
{
lean_object* v___x_3473_; lean_object* v___x_3474_; 
v___x_3473_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___closed__0));
lean_inc(v___y_3469_);
lean_inc(v_sp_3446_);
v___x_3474_ = l_Lean_SearchPath_findWithExt(v_sp_3446_, v___x_3473_, v___y_3469_);
if (lean_obj_tag(v___x_3474_) == 0)
{
lean_object* v_a_3475_; 
v_a_3475_ = lean_ctor_get(v___x_3474_, 0);
lean_inc(v_a_3475_);
lean_dec_ref_known(v___x_3474_, 1);
if (lean_obj_tag(v_a_3475_) == 0)
{
lean_object* v___x_3476_; lean_object* v___x_3477_; lean_object* v___x_3478_; 
v___x_3476_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__3));
lean_inc(v___y_3469_);
v___x_3477_ = l_Lean_modToFilePath(v___x_3476_, v___y_3469_, v___x_3473_);
v___x_3478_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0(v_key_3453_, v_value_3454_, v___x_3477_, v___y_3449_, v___y_3450_);
v___y_3462_ = v___y_3469_;
v___y_3463_ = v___x_3478_;
goto v___jp_3461_;
}
else
{
lean_object* v_val_3479_; lean_object* v___x_3480_; 
v_val_3479_ = lean_ctor_get(v_a_3475_, 0);
lean_inc(v_val_3479_);
lean_dec_ref_known(v_a_3475_, 1);
v___x_3480_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___lam__0(v_key_3453_, v_value_3454_, v_val_3479_, v___y_3449_, v___y_3450_);
v___y_3462_ = v___y_3469_;
v___y_3463_ = v___x_3480_;
goto v___jp_3461_;
}
}
else
{
lean_object* v_a_3481_; lean_object* v___x_3483_; uint8_t v_isShared_3484_; uint8_t v_isSharedCheck_3493_; 
lean_dec(v___y_3469_);
lean_dec(v_tail_3455_);
lean_dec(v_value_3454_);
lean_dec(v_key_3453_);
lean_dec_ref(v_x_3447_);
lean_dec(v_sp_3446_);
v_a_3481_ = lean_ctor_get(v___x_3474_, 0);
v_isSharedCheck_3493_ = !lean_is_exclusive(v___x_3474_);
if (v_isSharedCheck_3493_ == 0)
{
v___x_3483_ = v___x_3474_;
v_isShared_3484_ = v_isSharedCheck_3493_;
goto v_resetjp_3482_;
}
else
{
lean_inc(v_a_3481_);
lean_dec(v___x_3474_);
v___x_3483_ = lean_box(0);
v_isShared_3484_ = v_isSharedCheck_3493_;
goto v_resetjp_3482_;
}
v_resetjp_3482_:
{
lean_object* v_ref_3485_; lean_object* v___x_3486_; lean_object* v___x_3487_; lean_object* v___x_3488_; lean_object* v___x_3489_; lean_object* v___x_3491_; 
v_ref_3485_ = lean_ctor_get(v___y_3449_, 5);
v___x_3486_ = lean_io_error_to_string(v_a_3481_);
v___x_3487_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3487_, 0, v___x_3486_);
v___x_3488_ = l_Lean_MessageData_ofFormat(v___x_3487_);
lean_inc(v_ref_3485_);
v___x_3489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3489_, 0, v_ref_3485_);
lean_ctor_set(v___x_3489_, 1, v___x_3488_);
if (v_isShared_3484_ == 0)
{
lean_ctor_set(v___x_3483_, 0, v___x_3489_);
v___x_3491_ = v___x_3483_;
goto v_reusejp_3490_;
}
else
{
lean_object* v_reuseFailAlloc_3492_; 
v_reuseFailAlloc_3492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3492_, 0, v___x_3489_);
v___x_3491_ = v_reuseFailAlloc_3492_;
goto v_reusejp_3490_;
}
v_reusejp_3490_:
{
return v___x_3491_;
}
}
}
}
}
else
{
lean_object* v_val_3494_; lean_object* v_fst_3495_; lean_object* v_snd_3496_; lean_object* v___x_3498_; uint8_t v_isShared_3499_; uint8_t v_isSharedCheck_3504_; 
v_val_3494_ = lean_ctor_get(v___x_3470_, 0);
lean_inc(v_val_3494_);
lean_dec_ref_known(v___x_3470_, 1);
v_fst_3495_ = lean_ctor_get(v_val_3494_, 0);
v_snd_3496_ = lean_ctor_get(v_val_3494_, 1);
v_isSharedCheck_3504_ = !lean_is_exclusive(v_val_3494_);
if (v_isSharedCheck_3504_ == 0)
{
v___x_3498_ = v_val_3494_;
v_isShared_3499_ = v_isSharedCheck_3504_;
goto v_resetjp_3497_;
}
else
{
lean_inc(v_snd_3496_);
lean_inc(v_fst_3495_);
lean_dec(v_val_3494_);
v___x_3498_ = lean_box(0);
v_isShared_3499_ = v_isSharedCheck_3504_;
goto v_resetjp_3497_;
}
v_resetjp_3497_:
{
lean_object* v___x_3500_; lean_object* v___x_3502_; 
v___x_3500_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(v_snd_3496_, v_key_3453_, v_value_3454_);
if (v_isShared_3499_ == 0)
{
lean_ctor_set(v___x_3498_, 1, v___x_3500_);
v___x_3502_ = v___x_3498_;
goto v_reusejp_3501_;
}
else
{
lean_object* v_reuseFailAlloc_3503_; 
v_reuseFailAlloc_3503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3503_, 0, v_fst_3495_);
lean_ctor_set(v_reuseFailAlloc_3503_, 1, v___x_3500_);
v___x_3502_ = v_reuseFailAlloc_3503_;
goto v_reusejp_3501_;
}
v_reusejp_3501_:
{
v___y_3457_ = v___y_3469_;
v_a_3458_ = v___x_3502_;
goto v___jp_3456_;
}
}
}
}
}
else
{
lean_object* v_a_3508_; lean_object* v___x_3510_; uint8_t v_isShared_3511_; uint8_t v_isSharedCheck_3515_; 
lean_dec(v_tail_3455_);
lean_dec(v_value_3454_);
lean_dec(v_key_3453_);
lean_dec_ref(v_x_3447_);
lean_dec(v_sp_3446_);
v_a_3508_ = lean_ctor_get(v___x_3465_, 0);
v_isSharedCheck_3515_ = !lean_is_exclusive(v___x_3465_);
if (v_isSharedCheck_3515_ == 0)
{
v___x_3510_ = v___x_3465_;
v_isShared_3511_ = v_isSharedCheck_3515_;
goto v_resetjp_3509_;
}
else
{
lean_inc(v_a_3508_);
lean_dec(v___x_3465_);
v___x_3510_ = lean_box(0);
v_isShared_3511_ = v_isSharedCheck_3515_;
goto v_resetjp_3509_;
}
v_resetjp_3509_:
{
lean_object* v___x_3513_; 
if (v_isShared_3511_ == 0)
{
v___x_3513_ = v___x_3510_;
goto v_reusejp_3512_;
}
else
{
lean_object* v_reuseFailAlloc_3514_; 
v_reuseFailAlloc_3514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3514_, 0, v_a_3508_);
v___x_3513_ = v_reuseFailAlloc_3514_;
goto v_reusejp_3512_;
}
v_reusejp_3512_:
{
return v___x_3513_;
}
}
}
v___jp_3456_:
{
lean_object* v___x_3459_; 
v___x_3459_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Batteries_Tactic_Lint_lintCore_spec__0___redArg(v_x_3447_, v___y_3457_, v_a_3458_);
v_x_3447_ = v___x_3459_;
v_x_3448_ = v_tail_3455_;
goto _start;
}
v___jp_3461_:
{
lean_object* v_a_3464_; 
v_a_3464_ = lean_ctor_get(v___y_3463_, 0);
lean_inc(v_a_3464_);
lean_dec_ref(v___y_3463_);
v___y_3457_ = v___y_3462_;
v_a_3458_ = v_a_3464_;
goto v___jp_3456_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6___boxed(lean_object* v_useErrorFormat_3516_, lean_object* v_sp_3517_, lean_object* v_x_3518_, lean_object* v_x_3519_, lean_object* v___y_3520_, lean_object* v___y_3521_, lean_object* v___y_3522_){
_start:
{
uint8_t v_useErrorFormat_boxed_3523_; lean_object* v_res_3524_; 
v_useErrorFormat_boxed_3523_ = lean_unbox(v_useErrorFormat_3516_);
v_res_3524_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6(v_useErrorFormat_boxed_3523_, v_sp_3517_, v_x_3518_, v_x_3519_, v___y_3520_, v___y_3521_);
lean_dec(v___y_3521_);
lean_dec_ref(v___y_3520_);
return v_res_3524_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7(uint8_t v_useErrorFormat_3525_, lean_object* v_sp_3526_, lean_object* v_as_3527_, size_t v_i_3528_, size_t v_stop_3529_, lean_object* v_b_3530_, lean_object* v___y_3531_, lean_object* v___y_3532_){
_start:
{
uint8_t v___x_3534_; 
v___x_3534_ = lean_usize_dec_eq(v_i_3528_, v_stop_3529_);
if (v___x_3534_ == 0)
{
lean_object* v___x_3535_; lean_object* v___x_3536_; 
v___x_3535_ = lean_array_uget_borrowed(v_as_3527_, v_i_3528_);
lean_inc(v___x_3535_);
lean_inc(v_sp_3526_);
v___x_3536_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_groupedByFilename_spec__6(v_useErrorFormat_3525_, v_sp_3526_, v_b_3530_, v___x_3535_, v___y_3531_, v___y_3532_);
if (lean_obj_tag(v___x_3536_) == 0)
{
lean_object* v_a_3537_; size_t v___x_3538_; size_t v___x_3539_; 
v_a_3537_ = lean_ctor_get(v___x_3536_, 0);
lean_inc(v_a_3537_);
lean_dec_ref_known(v___x_3536_, 1);
v___x_3538_ = ((size_t)1ULL);
v___x_3539_ = lean_usize_add(v_i_3528_, v___x_3538_);
v_i_3528_ = v___x_3539_;
v_b_3530_ = v_a_3537_;
goto _start;
}
else
{
lean_dec(v_sp_3526_);
return v___x_3536_;
}
}
else
{
lean_object* v___x_3541_; 
lean_dec(v_sp_3526_);
v___x_3541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3541_, 0, v_b_3530_);
return v___x_3541_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7___boxed(lean_object* v_useErrorFormat_3542_, lean_object* v_sp_3543_, lean_object* v_as_3544_, lean_object* v_i_3545_, lean_object* v_stop_3546_, lean_object* v_b_3547_, lean_object* v___y_3548_, lean_object* v___y_3549_, lean_object* v___y_3550_){
_start:
{
uint8_t v_useErrorFormat_boxed_3551_; size_t v_i_boxed_3552_; size_t v_stop_boxed_3553_; lean_object* v_res_3554_; 
v_useErrorFormat_boxed_3551_ = lean_unbox(v_useErrorFormat_3542_);
v_i_boxed_3552_ = lean_unbox_usize(v_i_3545_);
lean_dec(v_i_3545_);
v_stop_boxed_3553_ = lean_unbox_usize(v_stop_3546_);
lean_dec(v_stop_3546_);
v_res_3554_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7(v_useErrorFormat_boxed_3551_, v_sp_3543_, v_as_3544_, v_i_boxed_3552_, v_stop_boxed_3553_, v_b_3547_, v___y_3548_, v___y_3549_);
lean_dec(v___y_3549_);
lean_dec_ref(v___y_3548_);
lean_dec_ref(v_as_3544_);
return v_res_3554_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_groupedByFilename___closed__0(void){
_start:
{
lean_object* v___x_3555_; lean_object* v___x_3556_; 
v___x_3555_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0, &lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0);
v___x_3556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3556_, 0, v___x_3555_);
lean_ctor_set(v___x_3556_, 1, v___x_3555_);
return v___x_3556_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_groupedByFilename(lean_object* v_results_3557_, uint8_t v_useErrorFormat_3558_, lean_object* v_a_3559_, lean_object* v_a_3560_){
_start:
{
lean_object* v___y_3563_; lean_object* v___y_3564_; lean_object* v___y_3565_; lean_object* v___y_3588_; lean_object* v___y_3589_; lean_object* v___y_3590_; lean_object* v___y_3591_; lean_object* v___y_3592_; lean_object* v___y_3593_; lean_object* v___y_3596_; lean_object* v___y_3597_; lean_object* v___y_3598_; lean_object* v___y_3599_; lean_object* v___y_3600_; lean_object* v___y_3601_; lean_object* v___y_3604_; lean_object* v___y_3605_; lean_object* v___y_3606_; lean_object* v___y_3614_; lean_object* v___y_3615_; lean_object* v_size_3616_; lean_object* v_buckets_3617_; lean_object* v___y_3630_; lean_object* v___y_3631_; lean_object* v___y_3632_; lean_object* v_sp_3645_; lean_object* v___y_3646_; lean_object* v___y_3647_; 
if (v_useErrorFormat_3558_ == 0)
{
lean_object* v___x_3661_; 
v___x_3661_ = lean_box(0);
v_sp_3645_ = v___x_3661_;
v___y_3646_ = v_a_3559_;
v___y_3647_ = v_a_3560_;
goto v___jp_3644_;
}
else
{
lean_object* v___x_3662_; 
v___x_3662_ = l_Lean_getSrcSearchPath();
if (lean_obj_tag(v___x_3662_) == 0)
{
lean_object* v_a_3663_; 
v_a_3663_ = lean_ctor_get(v___x_3662_, 0);
lean_inc(v_a_3663_);
lean_dec_ref_known(v___x_3662_, 1);
v_sp_3645_ = v_a_3663_;
v___y_3646_ = v_a_3559_;
v___y_3647_ = v_a_3560_;
goto v___jp_3644_;
}
else
{
lean_object* v_a_3664_; lean_object* v___x_3666_; uint8_t v_isShared_3667_; uint8_t v_isSharedCheck_3676_; 
v_a_3664_ = lean_ctor_get(v___x_3662_, 0);
v_isSharedCheck_3676_ = !lean_is_exclusive(v___x_3662_);
if (v_isSharedCheck_3676_ == 0)
{
v___x_3666_ = v___x_3662_;
v_isShared_3667_ = v_isSharedCheck_3676_;
goto v_resetjp_3665_;
}
else
{
lean_inc(v_a_3664_);
lean_dec(v___x_3662_);
v___x_3666_ = lean_box(0);
v_isShared_3667_ = v_isSharedCheck_3676_;
goto v_resetjp_3665_;
}
v_resetjp_3665_:
{
lean_object* v_ref_3668_; lean_object* v___x_3669_; lean_object* v___x_3670_; lean_object* v___x_3671_; lean_object* v___x_3672_; lean_object* v___x_3674_; 
v_ref_3668_ = lean_ctor_get(v_a_3559_, 5);
v___x_3669_ = lean_io_error_to_string(v_a_3664_);
v___x_3670_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3670_, 0, v___x_3669_);
v___x_3671_ = l_Lean_MessageData_ofFormat(v___x_3670_);
lean_inc(v_ref_3668_);
v___x_3672_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3672_, 0, v_ref_3668_);
lean_ctor_set(v___x_3672_, 1, v___x_3671_);
if (v_isShared_3667_ == 0)
{
lean_ctor_set(v___x_3666_, 0, v___x_3672_);
v___x_3674_ = v___x_3666_;
goto v_reusejp_3673_;
}
else
{
lean_object* v_reuseFailAlloc_3675_; 
v_reuseFailAlloc_3675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3675_, 0, v___x_3672_);
v___x_3674_ = v_reuseFailAlloc_3675_;
goto v_reusejp_3673_;
}
v_reusejp_3673_:
{
return v___x_3674_;
}
}
}
}
v___jp_3562_:
{
lean_object* v___x_3566_; lean_object* v___x_3567_; lean_object* v___x_3568_; 
v___x_3566_ = lean_array_to_list(v___y_3565_);
v___x_3567_ = lean_box(0);
v___x_3568_ = lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0(v_useErrorFormat_3558_, v___x_3566_, v___x_3567_, v___y_3563_, v___y_3564_);
if (lean_obj_tag(v___x_3568_) == 0)
{
lean_object* v_a_3569_; lean_object* v___x_3571_; uint8_t v_isShared_3572_; uint8_t v_isSharedCheck_3578_; 
v_a_3569_ = lean_ctor_get(v___x_3568_, 0);
v_isSharedCheck_3578_ = !lean_is_exclusive(v___x_3568_);
if (v_isSharedCheck_3578_ == 0)
{
v___x_3571_ = v___x_3568_;
v_isShared_3572_ = v_isSharedCheck_3578_;
goto v_resetjp_3570_;
}
else
{
lean_inc(v_a_3569_);
lean_dec(v___x_3568_);
v___x_3571_ = lean_box(0);
v_isShared_3572_ = v_isSharedCheck_3578_;
goto v_resetjp_3570_;
}
v_resetjp_3570_:
{
lean_object* v___x_3573_; lean_object* v___x_3574_; lean_object* v___x_3576_; 
v___x_3573_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_groupedByFilename___closed__0, &lp_batteries_Batteries_Tactic_Lint_groupedByFilename___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_groupedByFilename___closed__0);
v___x_3574_ = l_Lean_MessageData_joinSep(v_a_3569_, v___x_3573_);
if (v_isShared_3572_ == 0)
{
lean_ctor_set(v___x_3571_, 0, v___x_3574_);
v___x_3576_ = v___x_3571_;
goto v_reusejp_3575_;
}
else
{
lean_object* v_reuseFailAlloc_3577_; 
v_reuseFailAlloc_3577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3577_, 0, v___x_3574_);
v___x_3576_ = v_reuseFailAlloc_3577_;
goto v_reusejp_3575_;
}
v_reusejp_3575_:
{
return v___x_3576_;
}
}
}
else
{
lean_object* v_a_3579_; lean_object* v___x_3581_; uint8_t v_isShared_3582_; uint8_t v_isSharedCheck_3586_; 
v_a_3579_ = lean_ctor_get(v___x_3568_, 0);
v_isSharedCheck_3586_ = !lean_is_exclusive(v___x_3568_);
if (v_isSharedCheck_3586_ == 0)
{
v___x_3581_ = v___x_3568_;
v_isShared_3582_ = v_isSharedCheck_3586_;
goto v_resetjp_3580_;
}
else
{
lean_inc(v_a_3579_);
lean_dec(v___x_3568_);
v___x_3581_ = lean_box(0);
v_isShared_3582_ = v_isSharedCheck_3586_;
goto v_resetjp_3580_;
}
v_resetjp_3580_:
{
lean_object* v___x_3584_; 
if (v_isShared_3582_ == 0)
{
v___x_3584_ = v___x_3581_;
goto v_reusejp_3583_;
}
else
{
lean_object* v_reuseFailAlloc_3585_; 
v_reuseFailAlloc_3585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3585_, 0, v_a_3579_);
v___x_3584_ = v_reuseFailAlloc_3585_;
goto v_reusejp_3583_;
}
v_reusejp_3583_:
{
return v___x_3584_;
}
}
}
}
v___jp_3587_:
{
lean_object* v___x_3594_; 
v___x_3594_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg(v___y_3591_, v___y_3590_, v___y_3592_, v___y_3593_);
lean_dec(v___y_3593_);
lean_dec(v___y_3591_);
v___y_3563_ = v___y_3588_;
v___y_3564_ = v___y_3589_;
v___y_3565_ = v___x_3594_;
goto v___jp_3562_;
}
v___jp_3595_:
{
uint8_t v___x_3602_; 
v___x_3602_ = lean_nat_dec_le(v___y_3601_, v___y_3599_);
if (v___x_3602_ == 0)
{
lean_dec(v___y_3599_);
lean_inc(v___y_3601_);
v___y_3588_ = v___y_3596_;
v___y_3589_ = v___y_3597_;
v___y_3590_ = v___y_3598_;
v___y_3591_ = v___y_3600_;
v___y_3592_ = v___y_3601_;
v___y_3593_ = v___y_3601_;
goto v___jp_3587_;
}
else
{
v___y_3588_ = v___y_3596_;
v___y_3589_ = v___y_3597_;
v___y_3590_ = v___y_3598_;
v___y_3591_ = v___y_3600_;
v___y_3592_ = v___y_3601_;
v___y_3593_ = v___y_3599_;
goto v___jp_3587_;
}
}
v___jp_3603_:
{
lean_object* v___x_3607_; lean_object* v___x_3608_; uint8_t v___x_3609_; 
v___x_3607_ = lean_array_get_size(v___y_3606_);
v___x_3608_ = lean_unsigned_to_nat(0u);
v___x_3609_ = lean_nat_dec_eq(v___x_3607_, v___x_3608_);
if (v___x_3609_ == 0)
{
lean_object* v___x_3610_; lean_object* v___x_3611_; uint8_t v___x_3612_; 
v___x_3610_ = lean_unsigned_to_nat(1u);
v___x_3611_ = lean_nat_sub(v___x_3607_, v___x_3610_);
v___x_3612_ = lean_nat_dec_le(v___x_3608_, v___x_3611_);
if (v___x_3612_ == 0)
{
lean_inc(v___x_3611_);
v___y_3596_ = v___y_3604_;
v___y_3597_ = v___y_3605_;
v___y_3598_ = v___y_3606_;
v___y_3599_ = v___x_3611_;
v___y_3600_ = v___x_3607_;
v___y_3601_ = v___x_3611_;
goto v___jp_3595_;
}
else
{
v___y_3596_ = v___y_3604_;
v___y_3597_ = v___y_3605_;
v___y_3598_ = v___y_3606_;
v___y_3599_ = v___x_3611_;
v___y_3600_ = v___x_3607_;
v___y_3601_ = v___x_3608_;
goto v___jp_3595_;
}
}
else
{
v___y_3563_ = v___y_3604_;
v___y_3564_ = v___y_3605_;
v___y_3565_ = v___y_3606_;
goto v___jp_3562_;
}
}
v___jp_3613_:
{
lean_object* v___x_3618_; lean_object* v___x_3619_; lean_object* v___x_3620_; uint8_t v___x_3621_; 
v___x_3618_ = lean_mk_empty_array_with_capacity(v_size_3616_);
lean_dec(v_size_3616_);
v___x_3619_ = lean_unsigned_to_nat(0u);
v___x_3620_ = lean_array_get_size(v_buckets_3617_);
v___x_3621_ = lean_nat_dec_lt(v___x_3619_, v___x_3620_);
if (v___x_3621_ == 0)
{
lean_dec_ref(v_buckets_3617_);
v___y_3604_ = v___y_3614_;
v___y_3605_ = v___y_3615_;
v___y_3606_ = v___x_3618_;
goto v___jp_3603_;
}
else
{
uint8_t v___x_3622_; 
v___x_3622_ = lean_nat_dec_le(v___x_3620_, v___x_3620_);
if (v___x_3622_ == 0)
{
if (v___x_3621_ == 0)
{
lean_dec_ref(v_buckets_3617_);
v___y_3604_ = v___y_3614_;
v___y_3605_ = v___y_3615_;
v___y_3606_ = v___x_3618_;
goto v___jp_3603_;
}
else
{
size_t v___x_3623_; size_t v___x_3624_; lean_object* v___x_3625_; 
v___x_3623_ = ((size_t)0ULL);
v___x_3624_ = lean_usize_of_nat(v___x_3620_);
v___x_3625_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3(v_buckets_3617_, v___x_3623_, v___x_3624_, v___x_3618_);
lean_dec_ref(v_buckets_3617_);
v___y_3604_ = v___y_3614_;
v___y_3605_ = v___y_3615_;
v___y_3606_ = v___x_3625_;
goto v___jp_3603_;
}
}
else
{
size_t v___x_3626_; size_t v___x_3627_; lean_object* v___x_3628_; 
v___x_3626_ = ((size_t)0ULL);
v___x_3627_ = lean_usize_of_nat(v___x_3620_);
v___x_3628_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__3(v_buckets_3617_, v___x_3626_, v___x_3627_, v___x_3618_);
lean_dec_ref(v_buckets_3617_);
v___y_3604_ = v___y_3614_;
v___y_3605_ = v___y_3615_;
v___y_3606_ = v___x_3628_;
goto v___jp_3603_;
}
}
}
v___jp_3629_:
{
if (lean_obj_tag(v___y_3632_) == 0)
{
lean_object* v_a_3633_; lean_object* v_size_3634_; lean_object* v_buckets_3635_; 
v_a_3633_ = lean_ctor_get(v___y_3632_, 0);
lean_inc(v_a_3633_);
lean_dec_ref_known(v___y_3632_, 1);
v_size_3634_ = lean_ctor_get(v_a_3633_, 0);
lean_inc(v_size_3634_);
v_buckets_3635_ = lean_ctor_get(v_a_3633_, 1);
lean_inc_ref(v_buckets_3635_);
lean_dec(v_a_3633_);
v___y_3614_ = v___y_3630_;
v___y_3615_ = v___y_3631_;
v_size_3616_ = v_size_3634_;
v_buckets_3617_ = v_buckets_3635_;
goto v___jp_3613_;
}
else
{
lean_object* v_a_3636_; lean_object* v___x_3638_; uint8_t v_isShared_3639_; uint8_t v_isSharedCheck_3643_; 
v_a_3636_ = lean_ctor_get(v___y_3632_, 0);
v_isSharedCheck_3643_ = !lean_is_exclusive(v___y_3632_);
if (v_isSharedCheck_3643_ == 0)
{
v___x_3638_ = v___y_3632_;
v_isShared_3639_ = v_isSharedCheck_3643_;
goto v_resetjp_3637_;
}
else
{
lean_inc(v_a_3636_);
lean_dec(v___y_3632_);
v___x_3638_ = lean_box(0);
v_isShared_3639_ = v_isSharedCheck_3643_;
goto v_resetjp_3637_;
}
v_resetjp_3637_:
{
lean_object* v___x_3641_; 
if (v_isShared_3639_ == 0)
{
v___x_3641_ = v___x_3638_;
goto v_reusejp_3640_;
}
else
{
lean_object* v_reuseFailAlloc_3642_; 
v_reuseFailAlloc_3642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3642_, 0, v_a_3636_);
v___x_3641_ = v_reuseFailAlloc_3642_;
goto v_reusejp_3640_;
}
v_reusejp_3640_:
{
return v___x_3641_;
}
}
}
}
v___jp_3644_:
{
lean_object* v_buckets_3648_; lean_object* v___x_3649_; lean_object* v___x_3650_; lean_object* v___x_3651_; uint8_t v___x_3652_; 
v_buckets_3648_ = lean_ctor_get(v_results_3557_, 1);
v___x_3649_ = lean_unsigned_to_nat(0u);
v___x_3650_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__6);
v___x_3651_ = lean_array_get_size(v_buckets_3648_);
v___x_3652_ = lean_nat_dec_lt(v___x_3649_, v___x_3651_);
if (v___x_3652_ == 0)
{
lean_dec(v_sp_3645_);
v___y_3614_ = v___y_3646_;
v___y_3615_ = v___y_3647_;
v_size_3616_ = v___x_3649_;
v_buckets_3617_ = v___x_3650_;
goto v___jp_3613_;
}
else
{
lean_object* v___x_3653_; uint8_t v___x_3654_; 
v___x_3653_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__10___closed__7);
v___x_3654_ = lean_nat_dec_le(v___x_3651_, v___x_3651_);
if (v___x_3654_ == 0)
{
if (v___x_3652_ == 0)
{
lean_dec(v_sp_3645_);
v___y_3614_ = v___y_3646_;
v___y_3615_ = v___y_3647_;
v_size_3616_ = v___x_3649_;
v_buckets_3617_ = v___x_3650_;
goto v___jp_3613_;
}
else
{
size_t v___x_3655_; size_t v___x_3656_; lean_object* v___x_3657_; 
v___x_3655_ = ((size_t)0ULL);
v___x_3656_ = lean_usize_of_nat(v___x_3651_);
v___x_3657_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7(v_useErrorFormat_3558_, v_sp_3645_, v_buckets_3648_, v___x_3655_, v___x_3656_, v___x_3653_, v___y_3646_, v___y_3647_);
v___y_3630_ = v___y_3646_;
v___y_3631_ = v___y_3647_;
v___y_3632_ = v___x_3657_;
goto v___jp_3629_;
}
}
else
{
size_t v___x_3658_; size_t v___x_3659_; lean_object* v___x_3660_; 
v___x_3658_ = ((size_t)0ULL);
v___x_3659_ = lean_usize_of_nat(v___x_3651_);
v___x_3660_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_groupedByFilename_spec__7(v_useErrorFormat_3558_, v_sp_3645_, v_buckets_3648_, v___x_3658_, v___x_3659_, v___x_3653_, v___y_3646_, v___y_3647_);
v___y_3630_ = v___y_3646_;
v___y_3631_ = v___y_3647_;
v___y_3632_ = v___x_3660_;
goto v___jp_3629_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_groupedByFilename___boxed(lean_object* v_results_3677_, lean_object* v_useErrorFormat_3678_, lean_object* v_a_3679_, lean_object* v_a_3680_, lean_object* v_a_3681_){
_start:
{
uint8_t v_useErrorFormat_boxed_3682_; lean_object* v_res_3683_; 
v_useErrorFormat_boxed_3682_ = lean_unbox(v_useErrorFormat_3678_);
v_res_3683_ = lp_batteries_Batteries_Tactic_Lint_groupedByFilename(v_results_3677_, v_useErrorFormat_boxed_3682_, v_a_3679_, v_a_3680_);
lean_dec(v_a_3680_);
lean_dec_ref(v_a_3679_);
lean_dec_ref(v_results_3677_);
return v_res_3683_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1(lean_object* v_n_3684_, lean_object* v_as_3685_, lean_object* v_lo_3686_, lean_object* v_hi_3687_, lean_object* v_w_3688_, lean_object* v_hlo_3689_, lean_object* v_hhi_3690_){
_start:
{
lean_object* v___x_3691_; 
v___x_3691_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___redArg(v_n_3684_, v_as_3685_, v_lo_3686_, v_hi_3687_);
return v___x_3691_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1___boxed(lean_object* v_n_3692_, lean_object* v_as_3693_, lean_object* v_lo_3694_, lean_object* v_hi_3695_, lean_object* v_w_3696_, lean_object* v_hlo_3697_, lean_object* v_hhi_3698_){
_start:
{
lean_object* v_res_3699_; 
v_res_3699_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1(v_n_3692_, v_as_3693_, v_lo_3694_, v_hi_3695_, v_w_3696_, v_hlo_3697_, v_hhi_3698_);
lean_dec(v_hi_3695_);
lean_dec(v_n_3692_);
return v_res_3699_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5(lean_object* v_00_u03b2_3700_, lean_object* v_m_3701_, lean_object* v_a_3702_){
_start:
{
lean_object* v___x_3703_; 
v___x_3703_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg(v_m_3701_, v_a_3702_);
return v___x_3703_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___boxed(lean_object* v_00_u03b2_3704_, lean_object* v_m_3705_, lean_object* v_a_3706_){
_start:
{
lean_object* v_res_3707_; 
v_res_3707_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5(v_00_u03b2_3704_, v_m_3705_, v_a_3706_);
lean_dec(v_a_3706_);
lean_dec_ref(v_m_3705_);
return v_res_3707_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1(lean_object* v_n_3708_, lean_object* v_lo_3709_, lean_object* v_hi_3710_, lean_object* v_hhi_3711_, lean_object* v_pivot_3712_, lean_object* v_as_3713_, lean_object* v_i_3714_, lean_object* v_k_3715_, lean_object* v_ilo_3716_, lean_object* v_ik_3717_, lean_object* v_w_3718_){
_start:
{
lean_object* v___x_3719_; 
v___x_3719_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___redArg(v_hi_3710_, v_pivot_3712_, v_as_3713_, v_i_3714_, v_k_3715_);
return v___x_3719_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1___boxed(lean_object* v_n_3720_, lean_object* v_lo_3721_, lean_object* v_hi_3722_, lean_object* v_hhi_3723_, lean_object* v_pivot_3724_, lean_object* v_as_3725_, lean_object* v_i_3726_, lean_object* v_k_3727_, lean_object* v_ilo_3728_, lean_object* v_ik_3729_, lean_object* v_w_3730_){
_start:
{
lean_object* v_res_3731_; 
v_res_3731_ = lp_batteries___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Batteries_Tactic_Lint_groupedByFilename_spec__1_spec__1(v_n_3720_, v_lo_3721_, v_hi_3722_, v_hhi_3723_, v_pivot_3724_, v_as_3725_, v_i_3726_, v_k_3727_, v_ilo_3728_, v_ik_3729_, v_w_3730_);
lean_dec(v_hi_3722_);
lean_dec(v_lo_3721_);
lean_dec(v_n_3720_);
return v_res_3731_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7(lean_object* v_00_u03b2_3732_, lean_object* v_a_3733_, lean_object* v_x_3734_){
_start:
{
lean_object* v___x_3735_; 
v___x_3735_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___redArg(v_a_3733_, v_x_3734_);
return v___x_3735_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7___boxed(lean_object* v_00_u03b2_3736_, lean_object* v_a_3737_, lean_object* v_x_3738_){
_start:
{
lean_object* v_res_3739_; 
v_res_3739_ = lp_batteries_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5_spec__7(v_00_u03b2_3736_, v_a_3737_, v_x_3738_);
lean_dec(v_x_3738_);
lean_dec(v_a_3737_);
return v_res_3739_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2(lean_object* v_as_3740_, size_t v_i_3741_, size_t v_stop_3742_, lean_object* v_b_3743_){
_start:
{
uint8_t v___x_3744_; 
v___x_3744_ = lean_usize_dec_eq(v_i_3741_, v_stop_3742_);
if (v___x_3744_ == 0)
{
lean_object* v___x_3745_; lean_object* v___x_3746_; size_t v___x_3747_; size_t v___x_3748_; 
v___x_3745_ = lean_array_uget_borrowed(v_as_3740_, v_i_3741_);
v___x_3746_ = lean_nat_add(v_b_3743_, v___x_3745_);
lean_dec(v_b_3743_);
v___x_3747_ = ((size_t)1ULL);
v___x_3748_ = lean_usize_add(v_i_3741_, v___x_3747_);
v_i_3741_ = v___x_3748_;
v_b_3743_ = v___x_3746_;
goto _start;
}
else
{
return v_b_3743_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2___boxed(lean_object* v_as_3750_, lean_object* v_i_3751_, lean_object* v_stop_3752_, lean_object* v_b_3753_){
_start:
{
size_t v_i_boxed_3754_; size_t v_stop_boxed_3755_; lean_object* v_res_3756_; 
v_i_boxed_3754_ = lean_unbox_usize(v_i_3751_);
lean_dec(v_i_3751_);
v_stop_boxed_3755_ = lean_unbox_usize(v_stop_3752_);
lean_dec(v_stop_3752_);
v_res_3756_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2(v_as_3750_, v_i_boxed_3754_, v_stop_boxed_3755_, v_b_3753_);
lean_dec_ref(v_as_3750_);
return v_res_3756_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_3758_; lean_object* v___x_3759_; 
v___x_3758_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__0));
v___x_3759_ = l_Lean_stringToMessageData(v___x_3758_);
return v___x_3759_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_3761_; lean_object* v___x_3762_; 
v___x_3761_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__2));
v___x_3762_ = l_Lean_stringToMessageData(v___x_3761_);
return v___x_3762_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_3764_; lean_object* v___x_3765_; 
v___x_3764_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__4));
v___x_3765_ = l_Lean_stringToMessageData(v___x_3764_);
return v___x_3765_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__7(void){
_start:
{
lean_object* v___x_3767_; lean_object* v___x_3768_; 
v___x_3767_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__6));
v___x_3768_ = l_Lean_stringToMessageData(v___x_3767_);
return v___x_3768_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__9(void){
_start:
{
lean_object* v___x_3770_; lean_object* v___x_3771_; 
v___x_3770_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__8));
v___x_3771_ = l_Lean_stringToMessageData(v___x_3770_);
return v___x_3771_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0(uint8_t v_useErrorFormat_3772_, uint8_t v_groupByFilename_3773_, uint8_t v_verbose_3774_, lean_object* v_as_3775_, size_t v_i_3776_, size_t v_stop_3777_, lean_object* v_b_3778_, lean_object* v___y_3779_, lean_object* v___y_3780_){
_start:
{
lean_object* v_a_3783_; lean_object* v_val_3788_; uint8_t v___x_3790_; 
v___x_3790_ = lean_usize_dec_eq(v_i_3776_, v_stop_3777_);
if (v___x_3790_ == 0)
{
lean_object* v___x_3791_; lean_object* v_fst_3792_; lean_object* v_snd_3793_; lean_object* v___x_3795_; uint8_t v_isShared_3796_; uint8_t v_isSharedCheck_3859_; 
v___x_3791_ = lean_array_uget(v_as_3775_, v_i_3776_);
v_fst_3792_ = lean_ctor_get(v___x_3791_, 0);
v_snd_3793_ = lean_ctor_get(v___x_3791_, 1);
v_isSharedCheck_3859_ = !lean_is_exclusive(v___x_3791_);
if (v_isSharedCheck_3859_ == 0)
{
v___x_3795_ = v___x_3791_;
v_isShared_3796_ = v_isSharedCheck_3859_;
goto v_resetjp_3794_;
}
else
{
lean_inc(v_snd_3793_);
lean_inc(v_fst_3792_);
lean_dec(v___x_3791_);
v___x_3795_ = lean_box(0);
v_isShared_3796_ = v_isSharedCheck_3859_;
goto v_resetjp_3794_;
}
v_resetjp_3794_:
{
lean_object* v_warnings_3798_; lean_object* v_size_3829_; lean_object* v___x_3830_; uint8_t v___x_3831_; 
v_size_3829_ = lean_ctor_get(v_snd_3793_, 0);
v___x_3830_ = lean_unsigned_to_nat(0u);
v___x_3831_ = lean_nat_dec_eq(v_size_3829_, v___x_3830_);
if (v___x_3831_ == 0)
{
if (v_groupByFilename_3773_ == 0)
{
if (v_useErrorFormat_3772_ == 0)
{
lean_object* v___x_3832_; lean_object* v___x_3833_; 
v___x_3832_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___x_3833_ = lp_batteries_Batteries_Tactic_Lint_printWarnings(v_snd_3793_, v___x_3832_, v_useErrorFormat_3772_, v___y_3779_, v___y_3780_);
lean_dec(v_snd_3793_);
if (lean_obj_tag(v___x_3833_) == 0)
{
lean_object* v_a_3834_; 
v_a_3834_ = lean_ctor_get(v___x_3833_, 0);
lean_inc(v_a_3834_);
lean_dec_ref_known(v___x_3833_, 1);
v_warnings_3798_ = v_a_3834_;
goto v___jp_3797_;
}
else
{
lean_object* v_a_3835_; lean_object* v___x_3837_; uint8_t v_isShared_3838_; uint8_t v_isSharedCheck_3842_; 
lean_del_object(v___x_3795_);
lean_dec(v_fst_3792_);
lean_dec_ref(v_b_3778_);
v_a_3835_ = lean_ctor_get(v___x_3833_, 0);
v_isSharedCheck_3842_ = !lean_is_exclusive(v___x_3833_);
if (v_isSharedCheck_3842_ == 0)
{
v___x_3837_ = v___x_3833_;
v_isShared_3838_ = v_isSharedCheck_3842_;
goto v_resetjp_3836_;
}
else
{
lean_inc(v_a_3835_);
lean_dec(v___x_3833_);
v___x_3837_ = lean_box(0);
v_isShared_3838_ = v_isSharedCheck_3842_;
goto v_resetjp_3836_;
}
v_resetjp_3836_:
{
lean_object* v___x_3840_; 
if (v_isShared_3838_ == 0)
{
v___x_3840_ = v___x_3837_;
goto v_reusejp_3839_;
}
else
{
lean_object* v_reuseFailAlloc_3841_; 
v_reuseFailAlloc_3841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3841_, 0, v_a_3835_);
v___x_3840_ = v_reuseFailAlloc_3841_;
goto v_reusejp_3839_;
}
v_reusejp_3839_:
{
return v___x_3840_;
}
}
}
}
else
{
goto v___jp_3818_;
}
}
else
{
goto v___jp_3818_;
}
}
else
{
lean_object* v___x_3844_; uint8_t v_isShared_3845_; uint8_t v_isSharedCheck_3856_; 
lean_del_object(v___x_3795_);
v_isSharedCheck_3856_ = !lean_is_exclusive(v_snd_3793_);
if (v_isSharedCheck_3856_ == 0)
{
lean_object* v_unused_3857_; lean_object* v_unused_3858_; 
v_unused_3857_ = lean_ctor_get(v_snd_3793_, 1);
lean_dec(v_unused_3857_);
v_unused_3858_ = lean_ctor_get(v_snd_3793_, 0);
lean_dec(v_unused_3858_);
v___x_3844_ = v_snd_3793_;
v_isShared_3845_ = v_isSharedCheck_3856_;
goto v_resetjp_3843_;
}
else
{
lean_dec(v_snd_3793_);
v___x_3844_ = lean_box(0);
v_isShared_3845_ = v_isSharedCheck_3856_;
goto v_resetjp_3843_;
}
v_resetjp_3843_:
{
uint8_t v___x_3846_; uint8_t v___x_3847_; 
v___x_3846_ = 2;
v___x_3847_ = lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity(v_verbose_3774_, v___x_3846_);
if (v___x_3847_ == 0)
{
lean_del_object(v___x_3844_);
lean_dec(v_fst_3792_);
v_a_3783_ = v_b_3778_;
goto v___jp_3782_;
}
else
{
lean_object* v_toLinter_3848_; lean_object* v_noErrorsFound_3849_; lean_object* v___x_3850_; lean_object* v___x_3852_; 
v_toLinter_3848_ = lean_ctor_get(v_fst_3792_, 0);
lean_inc_ref(v_toLinter_3848_);
lean_dec(v_fst_3792_);
v_noErrorsFound_3849_ = lean_ctor_get(v_toLinter_3848_, 1);
lean_inc_ref(v_noErrorsFound_3849_);
lean_dec_ref(v_toLinter_3848_);
v___x_3850_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__9, &lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__9_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__9);
if (v_isShared_3845_ == 0)
{
lean_ctor_set_tag(v___x_3844_, 7);
lean_ctor_set(v___x_3844_, 1, v_noErrorsFound_3849_);
lean_ctor_set(v___x_3844_, 0, v___x_3850_);
v___x_3852_ = v___x_3844_;
goto v_reusejp_3851_;
}
else
{
lean_object* v_reuseFailAlloc_3855_; 
v_reuseFailAlloc_3855_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3855_, 0, v___x_3850_);
lean_ctor_set(v_reuseFailAlloc_3855_, 1, v_noErrorsFound_3849_);
v___x_3852_ = v_reuseFailAlloc_3855_;
goto v_reusejp_3851_;
}
v_reusejp_3851_:
{
lean_object* v___x_3853_; lean_object* v___x_3854_; 
v___x_3853_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5, &lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarning___closed__5);
v___x_3854_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3854_, 0, v___x_3852_);
lean_ctor_set(v___x_3854_, 1, v___x_3853_);
v_val_3788_ = v___x_3854_;
goto v___jp_3787_;
}
}
}
}
v___jp_3797_:
{
lean_object* v_toLinter_3799_; lean_object* v_name_3800_; lean_object* v_errorsFound_3801_; lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3805_; 
v_toLinter_3799_ = lean_ctor_get(v_fst_3792_, 0);
lean_inc_ref(v_toLinter_3799_);
v_name_3800_ = lean_ctor_get(v_fst_3792_, 1);
lean_inc(v_name_3800_);
lean_dec(v_fst_3792_);
v_errorsFound_3801_ = lean_ctor_get(v_toLinter_3799_, 2);
lean_inc_ref(v_errorsFound_3801_);
lean_dec_ref(v_toLinter_3799_);
v___x_3802_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__1);
v___x_3803_ = l_Lean_MessageData_ofName(v_name_3800_);
lean_inc_ref(v___x_3803_);
if (v_isShared_3796_ == 0)
{
lean_ctor_set_tag(v___x_3795_, 7);
lean_ctor_set(v___x_3795_, 1, v___x_3803_);
lean_ctor_set(v___x_3795_, 0, v___x_3802_);
v___x_3805_ = v___x_3795_;
goto v_reusejp_3804_;
}
else
{
lean_object* v_reuseFailAlloc_3817_; 
v_reuseFailAlloc_3817_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3817_, 0, v___x_3802_);
lean_ctor_set(v_reuseFailAlloc_3817_, 1, v___x_3803_);
v___x_3805_ = v_reuseFailAlloc_3817_;
goto v_reusejp_3804_;
}
v_reusejp_3804_:
{
lean_object* v___x_3806_; lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; lean_object* v___x_3811_; lean_object* v___x_3812_; lean_object* v___x_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; 
v___x_3806_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__3);
v___x_3807_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3807_, 0, v___x_3805_);
lean_ctor_set(v___x_3807_, 1, v___x_3806_);
v___x_3808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3808_, 0, v___x_3807_);
lean_ctor_set(v___x_3808_, 1, v_errorsFound_3801_);
v___x_3809_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__5);
v___x_3810_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3810_, 0, v___x_3808_);
lean_ctor_set(v___x_3810_, 1, v___x_3809_);
v___x_3811_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3811_, 0, v___x_3810_);
lean_ctor_set(v___x_3811_, 1, v___x_3803_);
v___x_3812_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___closed__7);
v___x_3813_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3813_, 0, v___x_3811_);
lean_ctor_set(v___x_3813_, 1, v___x_3812_);
v___x_3814_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3814_, 0, v___x_3813_);
lean_ctor_set(v___x_3814_, 1, v_warnings_3798_);
v___x_3815_ = lean_obj_once(&lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3, &lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3_once, _init_lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3);
v___x_3816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3816_, 0, v___x_3814_);
lean_ctor_set(v___x_3816_, 1, v___x_3815_);
v_val_3788_ = v___x_3816_;
goto v___jp_3787_;
}
}
v___jp_3818_:
{
lean_object* v___x_3819_; 
v___x_3819_ = lp_batteries_Batteries_Tactic_Lint_groupedByFilename(v_snd_3793_, v_useErrorFormat_3772_, v___y_3779_, v___y_3780_);
lean_dec(v_snd_3793_);
if (lean_obj_tag(v___x_3819_) == 0)
{
lean_object* v_a_3820_; 
v_a_3820_ = lean_ctor_get(v___x_3819_, 0);
lean_inc(v_a_3820_);
lean_dec_ref_known(v___x_3819_, 1);
v_warnings_3798_ = v_a_3820_;
goto v___jp_3797_;
}
else
{
lean_object* v_a_3821_; lean_object* v___x_3823_; uint8_t v_isShared_3824_; uint8_t v_isSharedCheck_3828_; 
lean_del_object(v___x_3795_);
lean_dec(v_fst_3792_);
lean_dec_ref(v_b_3778_);
v_a_3821_ = lean_ctor_get(v___x_3819_, 0);
v_isSharedCheck_3828_ = !lean_is_exclusive(v___x_3819_);
if (v_isSharedCheck_3828_ == 0)
{
v___x_3823_ = v___x_3819_;
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
else
{
lean_inc(v_a_3821_);
lean_dec(v___x_3819_);
v___x_3823_ = lean_box(0);
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
v_resetjp_3822_:
{
lean_object* v___x_3826_; 
if (v_isShared_3824_ == 0)
{
v___x_3826_ = v___x_3823_;
goto v_reusejp_3825_;
}
else
{
lean_object* v_reuseFailAlloc_3827_; 
v_reuseFailAlloc_3827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3827_, 0, v_a_3821_);
v___x_3826_ = v_reuseFailAlloc_3827_;
goto v_reusejp_3825_;
}
v_reusejp_3825_:
{
return v___x_3826_;
}
}
}
}
}
}
else
{
lean_object* v___x_3860_; 
v___x_3860_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3860_, 0, v_b_3778_);
return v___x_3860_;
}
v___jp_3782_:
{
size_t v___x_3784_; size_t v___x_3785_; 
v___x_3784_ = ((size_t)1ULL);
v___x_3785_ = lean_usize_add(v_i_3776_, v___x_3784_);
v_i_3776_ = v___x_3785_;
v_b_3778_ = v_a_3783_;
goto _start;
}
v___jp_3787_:
{
lean_object* v___x_3789_; 
v___x_3789_ = lean_array_push(v_b_3778_, v_val_3788_);
v_a_3783_ = v___x_3789_;
goto v___jp_3782_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0___boxed(lean_object* v_useErrorFormat_3861_, lean_object* v_groupByFilename_3862_, lean_object* v_verbose_3863_, lean_object* v_as_3864_, lean_object* v_i_3865_, lean_object* v_stop_3866_, lean_object* v_b_3867_, lean_object* v___y_3868_, lean_object* v___y_3869_, lean_object* v___y_3870_){
_start:
{
uint8_t v_useErrorFormat_boxed_3871_; uint8_t v_groupByFilename_boxed_3872_; uint8_t v_verbose_boxed_3873_; size_t v_i_boxed_3874_; size_t v_stop_boxed_3875_; lean_object* v_res_3876_; 
v_useErrorFormat_boxed_3871_ = lean_unbox(v_useErrorFormat_3861_);
v_groupByFilename_boxed_3872_ = lean_unbox(v_groupByFilename_3862_);
v_verbose_boxed_3873_ = lean_unbox(v_verbose_3863_);
v_i_boxed_3874_ = lean_unbox_usize(v_i_3865_);
lean_dec(v_i_3865_);
v_stop_boxed_3875_ = lean_unbox_usize(v_stop_3866_);
lean_dec(v_stop_3866_);
v_res_3876_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0(v_useErrorFormat_boxed_3871_, v_groupByFilename_boxed_3872_, v_verbose_boxed_3873_, v_as_3864_, v_i_boxed_3874_, v_stop_boxed_3875_, v_b_3867_, v___y_3868_, v___y_3869_);
lean_dec(v___y_3869_);
lean_dec_ref(v___y_3868_);
lean_dec_ref(v_as_3864_);
return v_res_3876_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0(uint8_t v_useErrorFormat_3877_, uint8_t v_groupByFilename_3878_, uint8_t v_verbose_3879_, lean_object* v_as_3880_, lean_object* v_start_3881_, lean_object* v_stop_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_){
_start:
{
lean_object* v___x_3886_; uint8_t v___x_3887_; 
v___x_3886_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__2));
v___x_3887_ = lean_nat_dec_lt(v_start_3881_, v_stop_3882_);
if (v___x_3887_ == 0)
{
lean_object* v___x_3888_; 
v___x_3888_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3888_, 0, v___x_3886_);
return v___x_3888_;
}
else
{
lean_object* v___x_3889_; uint8_t v___x_3890_; 
v___x_3889_ = lean_array_get_size(v_as_3880_);
v___x_3890_ = lean_nat_dec_le(v_stop_3882_, v___x_3889_);
if (v___x_3890_ == 0)
{
uint8_t v___x_3891_; 
v___x_3891_ = lean_nat_dec_lt(v_start_3881_, v___x_3889_);
if (v___x_3891_ == 0)
{
lean_object* v___x_3892_; 
v___x_3892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3892_, 0, v___x_3886_);
return v___x_3892_;
}
else
{
size_t v___x_3893_; size_t v___x_3894_; lean_object* v___x_3895_; 
v___x_3893_ = lean_usize_of_nat(v_start_3881_);
v___x_3894_ = lean_usize_of_nat(v___x_3889_);
v___x_3895_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0(v_useErrorFormat_3877_, v_groupByFilename_3878_, v_verbose_3879_, v_as_3880_, v___x_3893_, v___x_3894_, v___x_3886_, v___y_3883_, v___y_3884_);
return v___x_3895_;
}
}
else
{
size_t v___x_3896_; size_t v___x_3897_; lean_object* v___x_3898_; 
v___x_3896_ = lean_usize_of_nat(v_start_3881_);
v___x_3897_ = lean_usize_of_nat(v_stop_3882_);
v___x_3898_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0_spec__0(v_useErrorFormat_3877_, v_groupByFilename_3878_, v_verbose_3879_, v_as_3880_, v___x_3896_, v___x_3897_, v___x_3886_, v___y_3883_, v___y_3884_);
return v___x_3898_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0___boxed(lean_object* v_useErrorFormat_3899_, lean_object* v_groupByFilename_3900_, lean_object* v_verbose_3901_, lean_object* v_as_3902_, lean_object* v_start_3903_, lean_object* v_stop_3904_, lean_object* v___y_3905_, lean_object* v___y_3906_, lean_object* v___y_3907_){
_start:
{
uint8_t v_useErrorFormat_boxed_3908_; uint8_t v_groupByFilename_boxed_3909_; uint8_t v_verbose_boxed_3910_; lean_object* v_res_3911_; 
v_useErrorFormat_boxed_3908_ = lean_unbox(v_useErrorFormat_3899_);
v_groupByFilename_boxed_3909_ = lean_unbox(v_groupByFilename_3900_);
v_verbose_boxed_3910_ = lean_unbox(v_verbose_3901_);
v_res_3911_ = lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0(v_useErrorFormat_boxed_3908_, v_groupByFilename_boxed_3909_, v_verbose_boxed_3910_, v_as_3902_, v_start_3903_, v_stop_3904_, v___y_3905_, v___y_3906_);
lean_dec(v___y_3906_);
lean_dec_ref(v___y_3905_);
lean_dec(v_stop_3904_);
lean_dec(v_start_3903_);
lean_dec_ref(v_as_3902_);
return v_res_3911_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_formatLinterResults_spec__1(size_t v_sz_3912_, size_t v_i_3913_, lean_object* v_bs_3914_){
_start:
{
uint8_t v___x_3915_; 
v___x_3915_ = lean_usize_dec_lt(v_i_3913_, v_sz_3912_);
if (v___x_3915_ == 0)
{
return v_bs_3914_;
}
else
{
lean_object* v_v_3916_; lean_object* v_snd_3917_; lean_object* v_size_3918_; lean_object* v___x_3919_; lean_object* v_bs_x27_3920_; size_t v___x_3921_; size_t v___x_3922_; lean_object* v___x_3923_; 
v_v_3916_ = lean_array_uget_borrowed(v_bs_3914_, v_i_3913_);
v_snd_3917_ = lean_ctor_get(v_v_3916_, 1);
v_size_3918_ = lean_ctor_get(v_snd_3917_, 0);
lean_inc(v_size_3918_);
v___x_3919_ = lean_unsigned_to_nat(0u);
v_bs_x27_3920_ = lean_array_uset(v_bs_3914_, v_i_3913_, v___x_3919_);
v___x_3921_ = ((size_t)1ULL);
v___x_3922_ = lean_usize_add(v_i_3913_, v___x_3921_);
v___x_3923_ = lean_array_uset(v_bs_x27_3920_, v_i_3913_, v_size_3918_);
v_i_3913_ = v___x_3922_;
v_bs_3914_ = v___x_3923_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_formatLinterResults_spec__1___boxed(lean_object* v_sz_3925_, lean_object* v_i_3926_, lean_object* v_bs_3927_){
_start:
{
size_t v_sz_boxed_3928_; size_t v_i_boxed_3929_; lean_object* v_res_3930_; 
v_sz_boxed_3928_ = lean_unbox_usize(v_sz_3925_);
lean_dec(v_sz_3925_);
v_i_boxed_3929_ = lean_unbox_usize(v_i_3926_);
lean_dec(v_i_3926_);
v_res_3930_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_formatLinterResults_spec__1(v_sz_boxed_3928_, v_i_boxed_3929_, v_bs_3927_);
return v_res_3930_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg(lean_object* v_as_3931_, size_t v_i_3932_, size_t v_stop_3933_, lean_object* v_b_3934_, lean_object* v___y_3935_){
_start:
{
lean_object* v_a_3938_; uint8_t v___x_3942_; 
v___x_3942_ = lean_usize_dec_eq(v_i_3932_, v_stop_3933_);
if (v___x_3942_ == 0)
{
lean_object* v___x_3943_; lean_object* v_env_3944_; lean_object* v___x_3945_; uint8_t v___x_3946_; 
v___x_3943_ = lean_st_ref_get(v___y_3935_);
v_env_3944_ = lean_ctor_get(v___x_3943_, 0);
lean_inc_ref(v_env_3944_);
lean_dec(v___x_3943_);
v___x_3945_ = lean_array_uget_borrowed(v_as_3931_, v_i_3932_);
lean_inc(v___x_3945_);
v___x_3946_ = lp_batteries_Lean_Environment_isAutoDecl(v_env_3944_, v___x_3945_);
if (v___x_3946_ == 0)
{
v_a_3938_ = v_b_3934_;
goto v___jp_3937_;
}
else
{
lean_object* v___x_3947_; 
lean_inc(v___x_3945_);
v___x_3947_ = lean_array_push(v_b_3934_, v___x_3945_);
v_a_3938_ = v___x_3947_;
goto v___jp_3937_;
}
}
else
{
lean_object* v___x_3948_; 
v___x_3948_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3948_, 0, v_b_3934_);
return v___x_3948_;
}
v___jp_3937_:
{
size_t v___x_3939_; size_t v___x_3940_; 
v___x_3939_ = ((size_t)1ULL);
v___x_3940_ = lean_usize_add(v_i_3932_, v___x_3939_);
v_i_3932_ = v___x_3940_;
v_b_3934_ = v_a_3938_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg___boxed(lean_object* v_as_3949_, lean_object* v_i_3950_, lean_object* v_stop_3951_, lean_object* v_b_3952_, lean_object* v___y_3953_, lean_object* v___y_3954_){
_start:
{
size_t v_i_boxed_3955_; size_t v_stop_boxed_3956_; lean_object* v_res_3957_; 
v_i_boxed_3955_ = lean_unbox_usize(v_i_3950_);
lean_dec(v_i_3950_);
v_stop_boxed_3956_ = lean_unbox_usize(v_stop_3951_);
lean_dec(v_stop_3951_);
v_res_3957_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg(v_as_3949_, v_i_boxed_3955_, v_stop_boxed_3956_, v_b_3952_, v___y_3953_);
lean_dec(v___y_3953_);
lean_dec_ref(v_as_3949_);
return v_res_3957_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__1(void){
_start:
{
lean_object* v___x_3959_; lean_object* v___x_3960_; 
v___x_3959_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__0));
v___x_3960_ = l_Lean_stringToMessageData(v___x_3959_);
return v___x_3960_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__3(void){
_start:
{
lean_object* v___x_3962_; lean_object* v___x_3963_; 
v___x_3962_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__2));
v___x_3963_ = l_Lean_stringToMessageData(v___x_3962_);
return v___x_3963_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__5(void){
_start:
{
lean_object* v___x_3965_; lean_object* v___x_3966_; 
v___x_3965_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__4));
v___x_3966_ = l_Lean_stringToMessageData(v___x_3965_);
return v___x_3966_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__7(void){
_start:
{
lean_object* v___x_3968_; lean_object* v___x_3969_; 
v___x_3968_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__6));
v___x_3969_ = l_Lean_stringToMessageData(v___x_3968_);
return v___x_3969_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__9(void){
_start:
{
lean_object* v___x_3971_; lean_object* v___x_3972_; 
v___x_3971_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__8));
v___x_3972_ = l_Lean_stringToMessageData(v___x_3971_);
return v___x_3972_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__11(void){
_start:
{
lean_object* v___x_3974_; lean_object* v___x_3975_; 
v___x_3974_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__10));
v___x_3975_ = l_Lean_stringToMessageData(v___x_3974_);
return v___x_3975_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__13(void){
_start:
{
lean_object* v___x_3977_; lean_object* v___x_3978_; 
v___x_3977_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__12));
v___x_3978_ = l_Lean_stringToMessageData(v___x_3977_);
return v___x_3978_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__15(void){
_start:
{
lean_object* v___x_3980_; lean_object* v___x_3981_; 
v___x_3980_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__14));
v___x_3981_ = l_Lean_stringToMessageData(v___x_3980_);
return v___x_3981_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults(lean_object* v_results_3983_, lean_object* v_decls_3984_, uint8_t v_groupByFilename_3985_, lean_object* v_whereDesc_3986_, uint8_t v_runSlowLinters_3987_, uint8_t v_verbose_3988_, lean_object* v_numLinters_3989_, uint8_t v_useErrorFormat_3990_, lean_object* v_a_3991_, lean_object* v_a_3992_){
_start:
{
lean_object* v_s_3995_; lean_object* v___x_4000_; lean_object* v___x_4001_; lean_object* v___x_4002_; 
v___x_4000_ = lean_unsigned_to_nat(0u);
v___x_4001_ = lean_array_get_size(v_results_3983_);
v___x_4002_ = lp_batteries_Array_filterMapM___at___00Batteries_Tactic_Lint_formatLinterResults_spec__0(v_useErrorFormat_3990_, v_groupByFilename_3985_, v_verbose_3988_, v_results_3983_, v___x_4000_, v___x_4001_, v_a_3991_, v_a_3992_);
if (lean_obj_tag(v___x_4002_) == 0)
{
lean_object* v_a_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___y_4009_; lean_object* v___y_4010_; lean_object* v___y_4011_; lean_object* v___y_4041_; lean_object* v___y_4042_; lean_object* v_a_4055_; lean_object* v___y_4068_; lean_object* v___x_4078_; uint8_t v___x_4079_; 
v_a_4003_ = lean_ctor_get(v___x_4002_, 0);
lean_inc(v_a_4003_);
lean_dec_ref_known(v___x_4002_, 1);
v___x_4004_ = lean_array_to_list(v_a_4003_);
v___x_4005_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0, &lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0_once, _init_lp_batteries_Batteries_Tactic_Lint_printWarnings___closed__0);
v___x_4006_ = l_Lean_MessageData_joinSep(v___x_4004_, v___x_4005_);
v___x_4007_ = lean_array_get_size(v_decls_3984_);
v___x_4078_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1));
v___x_4079_ = lean_nat_dec_lt(v___x_4000_, v___x_4007_);
if (v___x_4079_ == 0)
{
v_a_4055_ = v___x_4078_;
goto v___jp_4054_;
}
else
{
uint8_t v___x_4080_; 
v___x_4080_ = lean_nat_dec_le(v___x_4007_, v___x_4007_);
if (v___x_4080_ == 0)
{
if (v___x_4079_ == 0)
{
v_a_4055_ = v___x_4078_;
goto v___jp_4054_;
}
else
{
size_t v___x_4081_; size_t v___x_4082_; lean_object* v___x_4083_; 
v___x_4081_ = ((size_t)0ULL);
v___x_4082_ = lean_usize_of_nat(v___x_4007_);
v___x_4083_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg(v_decls_3984_, v___x_4081_, v___x_4082_, v___x_4078_, v_a_3992_);
v___y_4068_ = v___x_4083_;
goto v___jp_4067_;
}
}
else
{
size_t v___x_4084_; size_t v___x_4085_; lean_object* v___x_4086_; 
v___x_4084_ = ((size_t)0ULL);
v___x_4085_ = lean_usize_of_nat(v___x_4007_);
v___x_4086_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg(v_decls_3984_, v___x_4084_, v___x_4085_, v___x_4078_, v_a_3992_);
v___y_4068_ = v___x_4086_;
goto v___jp_4067_;
}
}
v___jp_4008_:
{
lean_object* v___x_4012_; lean_object* v___x_4013_; lean_object* v___x_4014_; lean_object* v___x_4015_; lean_object* v___x_4016_; lean_object* v___x_4017_; lean_object* v___x_4018_; lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; lean_object* v___x_4022_; lean_object* v___x_4023_; lean_object* v___x_4024_; lean_object* v___x_4025_; lean_object* v___x_4026_; lean_object* v___x_4027_; lean_object* v___x_4028_; lean_object* v___x_4029_; lean_object* v___x_4030_; lean_object* v___x_4031_; lean_object* v___x_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; lean_object* v___x_4036_; lean_object* v___x_4037_; lean_object* v___x_4038_; lean_object* v___x_4039_; 
lean_inc_ref(v___y_4011_);
v___x_4012_ = l_Lean_stringToMessageData(v___y_4011_);
v___x_4013_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4013_, 0, v___y_4009_);
lean_ctor_set(v___x_4013_, 1, v___x_4012_);
v___x_4014_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__3, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__3_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__3);
v___x_4015_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4015_, 0, v___x_4013_);
lean_ctor_set(v___x_4015_, 1, v___x_4014_);
v___x_4016_ = lean_nat_sub(v___x_4007_, v___y_4010_);
v___x_4017_ = l_Nat_reprFast(v___x_4016_);
v___x_4018_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4018_, 0, v___x_4017_);
v___x_4019_ = l_Lean_MessageData_ofFormat(v___x_4018_);
v___x_4020_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4020_, 0, v___x_4015_);
lean_ctor_set(v___x_4020_, 1, v___x_4019_);
v___x_4021_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__5, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__5_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__5);
v___x_4022_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4022_, 0, v___x_4020_);
lean_ctor_set(v___x_4022_, 1, v___x_4021_);
v___x_4023_ = l_Nat_reprFast(v___y_4010_);
v___x_4024_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4024_, 0, v___x_4023_);
v___x_4025_ = l_Lean_MessageData_ofFormat(v___x_4024_);
v___x_4026_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4026_, 0, v___x_4022_);
lean_ctor_set(v___x_4026_, 1, v___x_4025_);
v___x_4027_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__7, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__7_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__7);
v___x_4028_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4028_, 0, v___x_4026_);
lean_ctor_set(v___x_4028_, 1, v___x_4027_);
v___x_4029_ = l_Lean_stringToMessageData(v_whereDesc_3986_);
v___x_4030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4030_, 0, v___x_4028_);
lean_ctor_set(v___x_4030_, 1, v___x_4029_);
v___x_4031_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__9, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__9_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__9);
v___x_4032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4032_, 0, v___x_4030_);
lean_ctor_set(v___x_4032_, 1, v___x_4031_);
v___x_4033_ = l_Nat_reprFast(v_numLinters_3989_);
v___x_4034_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4034_, 0, v___x_4033_);
v___x_4035_ = l_Lean_MessageData_ofFormat(v___x_4034_);
v___x_4036_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4036_, 0, v___x_4032_);
lean_ctor_set(v___x_4036_, 1, v___x_4035_);
v___x_4037_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__11, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__11_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__11);
v___x_4038_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4038_, 0, v___x_4036_);
lean_ctor_set(v___x_4038_, 1, v___x_4037_);
v___x_4039_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4039_, 0, v___x_4038_);
lean_ctor_set(v___x_4039_, 1, v___x_4006_);
v_s_3995_ = v___x_4039_;
goto v___jp_3994_;
}
v___jp_4040_:
{
if (v_verbose_3988_ == 0)
{
lean_dec(v___y_4042_);
lean_dec(v___y_4041_);
lean_dec(v_numLinters_3989_);
lean_dec_ref(v_whereDesc_3986_);
v_s_3995_ = v___x_4006_;
goto v___jp_3994_;
}
else
{
lean_object* v___x_4043_; lean_object* v___x_4044_; lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; uint8_t v___x_4051_; 
v___x_4043_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__13, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__13_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__13);
lean_inc(v___y_4042_);
v___x_4044_ = l_Nat_reprFast(v___y_4042_);
v___x_4045_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4045_, 0, v___x_4044_);
v___x_4046_ = l_Lean_MessageData_ofFormat(v___x_4045_);
v___x_4047_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4047_, 0, v___x_4043_);
lean_ctor_set(v___x_4047_, 1, v___x_4046_);
v___x_4048_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__15, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__15_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__15);
v___x_4049_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4049_, 0, v___x_4047_);
lean_ctor_set(v___x_4049_, 1, v___x_4048_);
v___x_4050_ = lean_unsigned_to_nat(1u);
v___x_4051_ = lean_nat_dec_eq(v___y_4042_, v___x_4050_);
lean_dec(v___y_4042_);
if (v___x_4051_ == 0)
{
lean_object* v___x_4052_; 
v___x_4052_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__16));
v___y_4009_ = v___x_4049_;
v___y_4010_ = v___y_4041_;
v___y_4011_ = v___x_4052_;
goto v___jp_4008_;
}
else
{
lean_object* v___x_4053_; 
v___x_4053_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_4009_ = v___x_4049_;
v___y_4010_ = v___y_4041_;
v___y_4011_ = v___x_4053_;
goto v___jp_4008_;
}
}
}
v___jp_4054_:
{
lean_object* v___x_4056_; size_t v_sz_4057_; size_t v___x_4058_; lean_object* v___x_4059_; lean_object* v___x_4060_; uint8_t v___x_4061_; 
v___x_4056_ = lean_array_get_size(v_a_4055_);
lean_dec_ref(v_a_4055_);
v_sz_4057_ = lean_array_size(v_results_3983_);
v___x_4058_ = ((size_t)0ULL);
v___x_4059_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_formatLinterResults_spec__1(v_sz_4057_, v___x_4058_, v_results_3983_);
v___x_4060_ = lean_array_get_size(v___x_4059_);
v___x_4061_ = lean_nat_dec_lt(v___x_4000_, v___x_4060_);
if (v___x_4061_ == 0)
{
lean_dec_ref(v___x_4059_);
v___y_4041_ = v___x_4056_;
v___y_4042_ = v___x_4000_;
goto v___jp_4040_;
}
else
{
uint8_t v___x_4062_; 
v___x_4062_ = lean_nat_dec_le(v___x_4060_, v___x_4060_);
if (v___x_4062_ == 0)
{
if (v___x_4061_ == 0)
{
lean_dec_ref(v___x_4059_);
v___y_4041_ = v___x_4056_;
v___y_4042_ = v___x_4000_;
goto v___jp_4040_;
}
else
{
size_t v___x_4063_; lean_object* v___x_4064_; 
v___x_4063_ = lean_usize_of_nat(v___x_4060_);
v___x_4064_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2(v___x_4059_, v___x_4058_, v___x_4063_, v___x_4000_);
lean_dec_ref(v___x_4059_);
v___y_4041_ = v___x_4056_;
v___y_4042_ = v___x_4064_;
goto v___jp_4040_;
}
}
else
{
size_t v___x_4065_; lean_object* v___x_4066_; 
v___x_4065_ = lean_usize_of_nat(v___x_4060_);
v___x_4066_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__2(v___x_4059_, v___x_4058_, v___x_4065_, v___x_4000_);
lean_dec_ref(v___x_4059_);
v___y_4041_ = v___x_4056_;
v___y_4042_ = v___x_4066_;
goto v___jp_4040_;
}
}
}
v___jp_4067_:
{
if (lean_obj_tag(v___y_4068_) == 0)
{
lean_object* v_a_4069_; 
v_a_4069_ = lean_ctor_get(v___y_4068_, 0);
lean_inc(v_a_4069_);
lean_dec_ref_known(v___y_4068_, 1);
v_a_4055_ = v_a_4069_;
goto v___jp_4054_;
}
else
{
lean_object* v_a_4070_; lean_object* v___x_4072_; uint8_t v_isShared_4073_; uint8_t v_isSharedCheck_4077_; 
lean_dec_ref(v___x_4006_);
lean_dec(v_numLinters_3989_);
lean_dec_ref(v_whereDesc_3986_);
lean_dec_ref(v_results_3983_);
v_a_4070_ = lean_ctor_get(v___y_4068_, 0);
v_isSharedCheck_4077_ = !lean_is_exclusive(v___y_4068_);
if (v_isSharedCheck_4077_ == 0)
{
v___x_4072_ = v___y_4068_;
v_isShared_4073_ = v_isSharedCheck_4077_;
goto v_resetjp_4071_;
}
else
{
lean_inc(v_a_4070_);
lean_dec(v___y_4068_);
v___x_4072_ = lean_box(0);
v_isShared_4073_ = v_isSharedCheck_4077_;
goto v_resetjp_4071_;
}
v_resetjp_4071_:
{
lean_object* v___x_4075_; 
if (v_isShared_4073_ == 0)
{
v___x_4075_ = v___x_4072_;
goto v_reusejp_4074_;
}
else
{
lean_object* v_reuseFailAlloc_4076_; 
v_reuseFailAlloc_4076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4076_, 0, v_a_4070_);
v___x_4075_ = v_reuseFailAlloc_4076_;
goto v_reusejp_4074_;
}
v_reusejp_4074_:
{
return v___x_4075_;
}
}
}
}
}
else
{
lean_object* v_a_4087_; lean_object* v___x_4089_; uint8_t v_isShared_4090_; uint8_t v_isSharedCheck_4094_; 
lean_dec(v_numLinters_3989_);
lean_dec_ref(v_whereDesc_3986_);
lean_dec_ref(v_results_3983_);
v_a_4087_ = lean_ctor_get(v___x_4002_, 0);
v_isSharedCheck_4094_ = !lean_is_exclusive(v___x_4002_);
if (v_isSharedCheck_4094_ == 0)
{
v___x_4089_ = v___x_4002_;
v_isShared_4090_ = v_isSharedCheck_4094_;
goto v_resetjp_4088_;
}
else
{
lean_inc(v_a_4087_);
lean_dec(v___x_4002_);
v___x_4089_ = lean_box(0);
v_isShared_4090_ = v_isSharedCheck_4094_;
goto v_resetjp_4088_;
}
v_resetjp_4088_:
{
lean_object* v___x_4092_; 
if (v_isShared_4090_ == 0)
{
v___x_4092_ = v___x_4089_;
goto v_reusejp_4091_;
}
else
{
lean_object* v_reuseFailAlloc_4093_; 
v_reuseFailAlloc_4093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4093_, 0, v_a_4087_);
v___x_4092_ = v_reuseFailAlloc_4093_;
goto v_reusejp_4091_;
}
v_reusejp_4091_:
{
return v___x_4092_;
}
}
}
v___jp_3994_:
{
if (v_runSlowLinters_3987_ == 0)
{
lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; 
v___x_3996_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__1, &lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint_formatLinterResults___closed__1);
v___x_3997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3997_, 0, v_s_3995_);
lean_ctor_set(v___x_3997_, 1, v___x_3996_);
v___x_3998_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3998_, 0, v___x_3997_);
return v___x_3998_;
}
else
{
lean_object* v___x_3999_; 
v___x_3999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3999_, 0, v_s_3995_);
return v___x_3999_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_formatLinterResults___boxed(lean_object* v_results_4095_, lean_object* v_decls_4096_, lean_object* v_groupByFilename_4097_, lean_object* v_whereDesc_4098_, lean_object* v_runSlowLinters_4099_, lean_object* v_verbose_4100_, lean_object* v_numLinters_4101_, lean_object* v_useErrorFormat_4102_, lean_object* v_a_4103_, lean_object* v_a_4104_, lean_object* v_a_4105_){
_start:
{
uint8_t v_groupByFilename_boxed_4106_; uint8_t v_runSlowLinters_boxed_4107_; uint8_t v_verbose_boxed_4108_; uint8_t v_useErrorFormat_boxed_4109_; lean_object* v_res_4110_; 
v_groupByFilename_boxed_4106_ = lean_unbox(v_groupByFilename_4097_);
v_runSlowLinters_boxed_4107_ = lean_unbox(v_runSlowLinters_4099_);
v_verbose_boxed_4108_ = lean_unbox(v_verbose_4100_);
v_useErrorFormat_boxed_4109_ = lean_unbox(v_useErrorFormat_4102_);
v_res_4110_ = lp_batteries_Batteries_Tactic_Lint_formatLinterResults(v_results_4095_, v_decls_4096_, v_groupByFilename_boxed_4106_, v_whereDesc_4098_, v_runSlowLinters_boxed_4107_, v_verbose_boxed_4108_, v_numLinters_4101_, v_useErrorFormat_boxed_4109_, v_a_4103_, v_a_4104_);
lean_dec(v_a_4104_);
lean_dec_ref(v_a_4103_);
lean_dec_ref(v_decls_4096_);
return v_res_4110_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3(lean_object* v_as_4111_, size_t v_i_4112_, size_t v_stop_4113_, lean_object* v_b_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_){
_start:
{
lean_object* v___x_4118_; 
v___x_4118_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___redArg(v_as_4111_, v_i_4112_, v_stop_4113_, v_b_4114_, v___y_4116_);
return v___x_4118_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3___boxed(lean_object* v_as_4119_, lean_object* v_i_4120_, lean_object* v_stop_4121_, lean_object* v_b_4122_, lean_object* v___y_4123_, lean_object* v___y_4124_, lean_object* v___y_4125_){
_start:
{
size_t v_i_boxed_4126_; size_t v_stop_boxed_4127_; lean_object* v_res_4128_; 
v_i_boxed_4126_ = lean_unbox_usize(v_i_4120_);
lean_dec(v_i_4120_);
v_stop_boxed_4127_ = lean_unbox_usize(v_stop_4121_);
lean_dec(v_stop_4121_);
v_res_4128_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_formatLinterResults_spec__3(v_as_4119_, v_i_boxed_4126_, v_stop_boxed_4127_, v_b_4122_, v___y_4123_, v___y_4124_);
lean_dec(v___y_4124_);
lean_dec_ref(v___y_4123_);
lean_dec_ref(v_as_4119_);
return v_res_4128_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___lam__0(lean_object* v_r_4129_, lean_object* v_k_4130_, lean_object* v_x_4131_){
_start:
{
lean_object* v___x_4132_; 
v___x_4132_ = lean_array_push(v_r_4129_, v_k_4130_);
return v___x_4132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___lam__0___boxed(lean_object* v_r_4133_, lean_object* v_k_4134_, lean_object* v_x_4135_){
_start:
{
lean_object* v_res_4136_; 
v_res_4136_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___lam__0(v_r_4133_, v_k_4134_, v_x_4135_);
lean_dec_ref(v_x_4135_);
return v_res_4136_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_f_4137_, lean_object* v_keys_4138_, lean_object* v_vals_4139_, lean_object* v_i_4140_, lean_object* v_acc_4141_){
_start:
{
lean_object* v___x_4142_; uint8_t v___x_4143_; 
v___x_4142_ = lean_array_get_size(v_keys_4138_);
v___x_4143_ = lean_nat_dec_lt(v_i_4140_, v___x_4142_);
if (v___x_4143_ == 0)
{
lean_dec(v_i_4140_);
lean_dec(v_f_4137_);
return v_acc_4141_;
}
else
{
lean_object* v_k_4144_; lean_object* v_v_4145_; lean_object* v___x_4146_; lean_object* v___x_4147_; lean_object* v___x_4148_; 
v_k_4144_ = lean_array_fget_borrowed(v_keys_4138_, v_i_4140_);
v_v_4145_ = lean_array_fget_borrowed(v_vals_4139_, v_i_4140_);
lean_inc(v_f_4137_);
lean_inc(v_v_4145_);
lean_inc(v_k_4144_);
v___x_4146_ = lean_apply_3(v_f_4137_, v_acc_4141_, v_k_4144_, v_v_4145_);
v___x_4147_ = lean_unsigned_to_nat(1u);
v___x_4148_ = lean_nat_add(v_i_4140_, v___x_4147_);
lean_dec(v_i_4140_);
v_i_4140_ = v___x_4148_;
v_acc_4141_ = v___x_4146_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_f_4150_, lean_object* v_keys_4151_, lean_object* v_vals_4152_, lean_object* v_i_4153_, lean_object* v_acc_4154_){
_start:
{
lean_object* v_res_4155_; 
v_res_4155_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg(v_f_4150_, v_keys_4151_, v_vals_4152_, v_i_4153_, v_acc_4154_);
lean_dec_ref(v_vals_4152_);
lean_dec_ref(v_keys_4151_);
return v_res_4155_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(lean_object* v_f_4156_, lean_object* v_x_4157_, lean_object* v_x_4158_){
_start:
{
if (lean_obj_tag(v_x_4157_) == 0)
{
lean_object* v_es_4159_; lean_object* v___x_4160_; lean_object* v___x_4161_; uint8_t v___x_4162_; 
v_es_4159_ = lean_ctor_get(v_x_4157_, 0);
v___x_4160_ = lean_unsigned_to_nat(0u);
v___x_4161_ = lean_array_get_size(v_es_4159_);
v___x_4162_ = lean_nat_dec_lt(v___x_4160_, v___x_4161_);
if (v___x_4162_ == 0)
{
lean_dec(v_f_4156_);
return v_x_4158_;
}
else
{
uint8_t v___x_4163_; 
v___x_4163_ = lean_nat_dec_le(v___x_4161_, v___x_4161_);
if (v___x_4163_ == 0)
{
if (v___x_4162_ == 0)
{
lean_dec(v_f_4156_);
return v_x_4158_;
}
else
{
size_t v___x_4164_; size_t v___x_4165_; lean_object* v___x_4166_; 
v___x_4164_ = ((size_t)0ULL);
v___x_4165_ = lean_usize_of_nat(v___x_4161_);
v___x_4166_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg(v_f_4156_, v_es_4159_, v___x_4164_, v___x_4165_, v_x_4158_);
return v___x_4166_;
}
}
else
{
size_t v___x_4167_; size_t v___x_4168_; lean_object* v___x_4169_; 
v___x_4167_ = ((size_t)0ULL);
v___x_4168_ = lean_usize_of_nat(v___x_4161_);
v___x_4169_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg(v_f_4156_, v_es_4159_, v___x_4167_, v___x_4168_, v_x_4158_);
return v___x_4169_;
}
}
}
else
{
lean_object* v_ks_4170_; lean_object* v_vs_4171_; lean_object* v___x_4172_; lean_object* v___x_4173_; 
v_ks_4170_ = lean_ctor_get(v_x_4157_, 0);
v_vs_4171_ = lean_ctor_get(v_x_4157_, 1);
v___x_4172_ = lean_unsigned_to_nat(0u);
v___x_4173_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg(v_f_4156_, v_ks_4170_, v_vs_4171_, v___x_4172_, v_x_4158_);
return v___x_4173_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_f_4174_, lean_object* v_as_4175_, size_t v_i_4176_, size_t v_stop_4177_, lean_object* v_b_4178_){
_start:
{
lean_object* v___y_4180_; uint8_t v___x_4184_; 
v___x_4184_ = lean_usize_dec_eq(v_i_4176_, v_stop_4177_);
if (v___x_4184_ == 0)
{
lean_object* v___x_4185_; 
v___x_4185_ = lean_array_uget_borrowed(v_as_4175_, v_i_4176_);
switch(lean_obj_tag(v___x_4185_))
{
case 0:
{
lean_object* v_key_4186_; lean_object* v_val_4187_; lean_object* v___x_4188_; 
v_key_4186_ = lean_ctor_get(v___x_4185_, 0);
v_val_4187_ = lean_ctor_get(v___x_4185_, 1);
lean_inc(v_f_4174_);
lean_inc(v_val_4187_);
lean_inc(v_key_4186_);
v___x_4188_ = lean_apply_3(v_f_4174_, v_b_4178_, v_key_4186_, v_val_4187_);
v___y_4180_ = v___x_4188_;
goto v___jp_4179_;
}
case 1:
{
lean_object* v_node_4189_; lean_object* v___x_4190_; 
v_node_4189_ = lean_ctor_get(v___x_4185_, 0);
lean_inc(v_f_4174_);
v___x_4190_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(v_f_4174_, v_node_4189_, v_b_4178_);
v___y_4180_ = v___x_4190_;
goto v___jp_4179_;
}
default: 
{
v___y_4180_ = v_b_4178_;
goto v___jp_4179_;
}
}
}
else
{
lean_dec(v_f_4174_);
return v_b_4178_;
}
v___jp_4179_:
{
size_t v___x_4181_; size_t v___x_4182_; 
v___x_4181_ = ((size_t)1ULL);
v___x_4182_ = lean_usize_add(v_i_4176_, v___x_4181_);
v_i_4176_ = v___x_4182_;
v_b_4178_ = v___y_4180_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_f_4191_, lean_object* v_as_4192_, lean_object* v_i_4193_, lean_object* v_stop_4194_, lean_object* v_b_4195_){
_start:
{
size_t v_i_boxed_4196_; size_t v_stop_boxed_4197_; lean_object* v_res_4198_; 
v_i_boxed_4196_ = lean_unbox_usize(v_i_4193_);
lean_dec(v_i_4193_);
v_stop_boxed_4197_ = lean_unbox_usize(v_stop_4194_);
lean_dec(v_stop_4194_);
v_res_4198_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg(v_f_4191_, v_as_4192_, v_i_boxed_4196_, v_stop_boxed_4197_, v_b_4195_);
lean_dec_ref(v_as_4192_);
return v_res_4198_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_f_4199_, lean_object* v_x_4200_, lean_object* v_x_4201_){
_start:
{
lean_object* v_res_4202_; 
v_res_4202_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(v_f_4199_, v_x_4200_, v_x_4201_);
lean_dec_ref(v_x_4200_);
return v_res_4202_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg___lam__0(lean_object* v_f_4203_, lean_object* v_x1_4204_, lean_object* v_x2_4205_, lean_object* v_x3_4206_){
_start:
{
lean_object* v___x_4207_; 
v___x_4207_ = lean_apply_3(v_f_4203_, v_x1_4204_, v_x2_4205_, v_x3_4206_);
return v___x_4207_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg(lean_object* v_map_4208_, lean_object* v_f_4209_, lean_object* v_init_4210_){
_start:
{
lean_object* v___f_4211_; lean_object* v___x_4212_; 
v___f_4211_ = lean_alloc_closure((void*)(lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg___lam__0), 4, 1);
lean_closure_set(v___f_4211_, 0, v_f_4209_);
v___x_4212_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(v___f_4211_, v_map_4208_, v_init_4210_);
return v___x_4212_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg___boxed(lean_object* v_map_4213_, lean_object* v_f_4214_, lean_object* v_init_4215_){
_start:
{
lean_object* v_res_4216_; 
v_res_4216_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg(v_map_4213_, v_f_4214_, v_init_4215_);
lean_dec_ref(v_map_4213_);
return v_res_4216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg(lean_object* v_a_4218_){
_start:
{
lean_object* v___x_4220_; lean_object* v_env_4221_; lean_object* v___x_4222_; lean_object* v_map_u2082_4223_; lean_object* v___f_4224_; lean_object* v___x_4225_; lean_object* v___x_4226_; lean_object* v___x_4227_; 
v___x_4220_ = lean_st_ref_get(v_a_4218_);
v_env_4221_ = lean_ctor_get(v___x_4220_, 0);
lean_inc_ref(v_env_4221_);
lean_dec(v___x_4220_);
v___x_4222_ = l_Lean_Environment_constants(v_env_4221_);
v_map_u2082_4223_ = lean_ctor_get(v___x_4222_, 1);
lean_inc_ref(v_map_u2082_4223_);
lean_dec_ref(v___x_4222_);
v___f_4224_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___closed__0));
v___x_4225_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_shouldBeLinted___at___00Batteries_Tactic_Lint_lintCore_spec__7___redArg___closed__1));
v___x_4226_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg(v_map_u2082_4223_, v___f_4224_, v___x_4225_);
lean_dec_ref(v_map_u2082_4223_);
v___x_4227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4227_, 0, v___x_4226_);
return v___x_4227_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg___boxed(lean_object* v_a_4228_, lean_object* v_a_4229_){
_start:
{
lean_object* v_res_4230_; 
v_res_4230_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg(v_a_4228_);
lean_dec(v_a_4228_);
return v_res_4230_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule(lean_object* v_a_4231_, lean_object* v_a_4232_){
_start:
{
lean_object* v___x_4234_; 
v___x_4234_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg(v_a_4232_);
return v___x_4234_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___boxed(lean_object* v_a_4235_, lean_object* v_a_4236_, lean_object* v_a_4237_){
_start:
{
lean_object* v_res_4238_; 
v_res_4238_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule(v_a_4235_, v_a_4236_);
lean_dec(v_a_4236_);
lean_dec_ref(v_a_4235_);
return v_res_4238_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0(lean_object* v_00_u03c3_4239_, lean_object* v_00_u03b2_4240_, lean_object* v_map_4241_, lean_object* v_f_4242_, lean_object* v_init_4243_){
_start:
{
lean_object* v___x_4244_; 
v___x_4244_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___redArg(v_map_4241_, v_f_4242_, v_init_4243_);
return v___x_4244_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0___boxed(lean_object* v_00_u03c3_4245_, lean_object* v_00_u03b2_4246_, lean_object* v_map_4247_, lean_object* v_f_4248_, lean_object* v_init_4249_){
_start:
{
lean_object* v_res_4250_; 
v_res_4250_ = lp_batteries_Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0(v_00_u03c3_4245_, v_00_u03b2_4246_, v_map_4247_, v_f_4248_, v_init_4249_);
lean_dec_ref(v_map_4247_);
return v_res_4250_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___redArg(lean_object* v_map_4251_, lean_object* v_f_4252_, lean_object* v_init_4253_){
_start:
{
lean_object* v___x_4254_; 
v___x_4254_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(v_f_4252_, v_map_4251_, v_init_4253_);
return v___x_4254_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___redArg___boxed(lean_object* v_map_4255_, lean_object* v_f_4256_, lean_object* v_init_4257_){
_start:
{
lean_object* v_res_4258_; 
v_res_4258_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___redArg(v_map_4255_, v_f_4256_, v_init_4257_);
lean_dec_ref(v_map_4255_);
return v_res_4258_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0(lean_object* v_00_u03c3_4259_, lean_object* v_00_u03b2_4260_, lean_object* v_map_4261_, lean_object* v_f_4262_, lean_object* v_init_4263_){
_start:
{
lean_object* v___x_4264_; 
v___x_4264_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(v_f_4262_, v_map_4261_, v_init_4263_);
return v___x_4264_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0___boxed(lean_object* v_00_u03c3_4265_, lean_object* v_00_u03b2_4266_, lean_object* v_map_4267_, lean_object* v_f_4268_, lean_object* v_init_4269_){
_start:
{
lean_object* v_res_4270_; 
v_res_4270_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0(v_00_u03c3_4265_, v_00_u03b2_4266_, v_map_4267_, v_f_4268_, v_init_4269_);
lean_dec_ref(v_map_4267_);
return v_res_4270_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1(lean_object* v_00_u03c3_4271_, lean_object* v_00_u03b1_4272_, lean_object* v_00_u03b2_4273_, lean_object* v_f_4274_, lean_object* v_x_4275_, lean_object* v_x_4276_){
_start:
{
lean_object* v___x_4277_; 
v___x_4277_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___redArg(v_f_4274_, v_x_4275_, v_x_4276_);
return v___x_4277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03c3_4278_, lean_object* v_00_u03b1_4279_, lean_object* v_00_u03b2_4280_, lean_object* v_f_4281_, lean_object* v_x_4282_, lean_object* v_x_4283_){
_start:
{
lean_object* v_res_4284_; 
v_res_4284_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1(v_00_u03c3_4278_, v_00_u03b1_4279_, v_00_u03b2_4280_, v_f_4281_, v_x_4282_, v_x_4283_);
lean_dec_ref(v_x_4282_);
return v_res_4284_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_4285_, lean_object* v_00_u03b2_4286_, lean_object* v_00_u03c3_4287_, lean_object* v_f_4288_, lean_object* v_as_4289_, size_t v_i_4290_, size_t v_stop_4291_, lean_object* v_b_4292_){
_start:
{
lean_object* v___x_4293_; 
v___x_4293_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___redArg(v_f_4288_, v_as_4289_, v_i_4290_, v_stop_4291_, v_b_4292_);
return v___x_4293_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_4294_, lean_object* v_00_u03b2_4295_, lean_object* v_00_u03c3_4296_, lean_object* v_f_4297_, lean_object* v_as_4298_, lean_object* v_i_4299_, lean_object* v_stop_4300_, lean_object* v_b_4301_){
_start:
{
size_t v_i_boxed_4302_; size_t v_stop_boxed_4303_; lean_object* v_res_4304_; 
v_i_boxed_4302_ = lean_unbox_usize(v_i_4299_);
lean_dec(v_i_4299_);
v_stop_boxed_4303_ = lean_unbox_usize(v_stop_4300_);
lean_dec(v_stop_4300_);
v_res_4304_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_4294_, v_00_u03b2_4295_, v_00_u03c3_4296_, v_f_4297_, v_as_4298_, v_i_boxed_4302_, v_stop_boxed_4303_, v_b_4301_);
lean_dec_ref(v_as_4298_);
return v_res_4304_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03c3_4305_, lean_object* v_00_u03b1_4306_, lean_object* v_00_u03b2_4307_, lean_object* v_f_4308_, lean_object* v_keys_4309_, lean_object* v_vals_4310_, lean_object* v_heq_4311_, lean_object* v_i_4312_, lean_object* v_acc_4313_){
_start:
{
lean_object* v___x_4314_; 
v___x_4314_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___redArg(v_f_4308_, v_keys_4309_, v_vals_4310_, v_i_4312_, v_acc_4313_);
return v___x_4314_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03c3_4315_, lean_object* v_00_u03b1_4316_, lean_object* v_00_u03b2_4317_, lean_object* v_f_4318_, lean_object* v_keys_4319_, lean_object* v_vals_4320_, lean_object* v_heq_4321_, lean_object* v_i_4322_, lean_object* v_acc_4323_){
_start:
{
lean_object* v_res_4324_; 
v_res_4324_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Batteries_Tactic_Lint_getDeclsInCurrModule_spec__0_spec__0_spec__1_spec__3(v_00_u03c3_4315_, v_00_u03b1_4316_, v_00_u03b2_4317_, v_f_4318_, v_keys_4319_, v_vals_4320_, v_heq_4321_, v_i_4322_, v_acc_4323_);
lean_dec_ref(v_vals_4320_);
lean_dec_ref(v_keys_4319_);
return v_res_4324_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getAllDecls_spec__0(lean_object* v_x_4325_, lean_object* v_x_4326_){
_start:
{
if (lean_obj_tag(v_x_4326_) == 0)
{
return v_x_4325_;
}
else
{
lean_object* v_key_4327_; lean_object* v_tail_4328_; lean_object* v___x_4329_; 
v_key_4327_ = lean_ctor_get(v_x_4326_, 0);
lean_inc(v_key_4327_);
v_tail_4328_ = lean_ctor_get(v_x_4326_, 2);
lean_inc(v_tail_4328_);
lean_dec_ref_known(v_x_4326_, 3);
v___x_4329_ = lean_array_push(v_x_4325_, v_key_4327_);
v_x_4325_ = v___x_4329_;
v_x_4326_ = v_tail_4328_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1(lean_object* v_as_4331_, size_t v_i_4332_, size_t v_stop_4333_, lean_object* v_b_4334_){
_start:
{
uint8_t v___x_4335_; 
v___x_4335_ = lean_usize_dec_eq(v_i_4332_, v_stop_4333_);
if (v___x_4335_ == 0)
{
lean_object* v___x_4336_; lean_object* v___x_4337_; size_t v___x_4338_; size_t v___x_4339_; 
v___x_4336_ = lean_array_uget_borrowed(v_as_4331_, v_i_4332_);
lean_inc(v___x_4336_);
v___x_4337_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getAllDecls_spec__0(v_b_4334_, v___x_4336_);
v___x_4338_ = ((size_t)1ULL);
v___x_4339_ = lean_usize_add(v_i_4332_, v___x_4338_);
v_i_4332_ = v___x_4339_;
v_b_4334_ = v___x_4337_;
goto _start;
}
else
{
return v_b_4334_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1___boxed(lean_object* v_as_4341_, lean_object* v_i_4342_, lean_object* v_stop_4343_, lean_object* v_b_4344_){
_start:
{
size_t v_i_boxed_4345_; size_t v_stop_boxed_4346_; lean_object* v_res_4347_; 
v_i_boxed_4345_ = lean_unbox_usize(v_i_4342_);
lean_dec(v_i_4342_);
v_stop_boxed_4346_ = lean_unbox_usize(v_stop_4343_);
lean_dec(v_stop_4343_);
v_res_4347_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1(v_as_4341_, v_i_boxed_4345_, v_stop_boxed_4346_, v_b_4344_);
lean_dec_ref(v_as_4341_);
return v_res_4347_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls___redArg(lean_object* v_a_4348_){
_start:
{
lean_object* v___x_4350_; lean_object* v___x_4351_; lean_object* v_a_4352_; lean_object* v_env_4353_; lean_object* v___x_4354_; lean_object* v_map_u2081_4355_; lean_object* v_buckets_4356_; lean_object* v___x_4357_; lean_object* v___x_4358_; uint8_t v___x_4359_; 
v___x_4350_ = lean_st_ref_get(v_a_4348_);
v___x_4351_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg(v_a_4348_);
v_a_4352_ = lean_ctor_get(v___x_4351_, 0);
lean_inc(v_a_4352_);
v_env_4353_ = lean_ctor_get(v___x_4350_, 0);
lean_inc_ref(v_env_4353_);
lean_dec(v___x_4350_);
v___x_4354_ = l_Lean_Environment_constants(v_env_4353_);
v_map_u2081_4355_ = lean_ctor_get(v___x_4354_, 0);
lean_inc_ref(v_map_u2081_4355_);
lean_dec_ref(v___x_4354_);
v_buckets_4356_ = lean_ctor_get(v_map_u2081_4355_, 1);
lean_inc_ref(v_buckets_4356_);
lean_dec_ref(v_map_u2081_4355_);
v___x_4357_ = lean_unsigned_to_nat(0u);
v___x_4358_ = lean_array_get_size(v_buckets_4356_);
v___x_4359_ = lean_nat_dec_lt(v___x_4357_, v___x_4358_);
if (v___x_4359_ == 0)
{
lean_dec_ref(v_buckets_4356_);
lean_dec(v_a_4352_);
return v___x_4351_;
}
else
{
uint8_t v___x_4360_; 
v___x_4360_ = lean_nat_dec_le(v___x_4358_, v___x_4358_);
if (v___x_4360_ == 0)
{
if (v___x_4359_ == 0)
{
lean_dec_ref(v_buckets_4356_);
lean_dec(v_a_4352_);
return v___x_4351_;
}
else
{
lean_object* v___x_4362_; uint8_t v_isShared_4363_; uint8_t v_isSharedCheck_4370_; 
v_isSharedCheck_4370_ = !lean_is_exclusive(v___x_4351_);
if (v_isSharedCheck_4370_ == 0)
{
lean_object* v_unused_4371_; 
v_unused_4371_ = lean_ctor_get(v___x_4351_, 0);
lean_dec(v_unused_4371_);
v___x_4362_ = v___x_4351_;
v_isShared_4363_ = v_isSharedCheck_4370_;
goto v_resetjp_4361_;
}
else
{
lean_dec(v___x_4351_);
v___x_4362_ = lean_box(0);
v_isShared_4363_ = v_isSharedCheck_4370_;
goto v_resetjp_4361_;
}
v_resetjp_4361_:
{
size_t v___x_4364_; size_t v___x_4365_; lean_object* v___x_4366_; lean_object* v___x_4368_; 
v___x_4364_ = ((size_t)0ULL);
v___x_4365_ = lean_usize_of_nat(v___x_4358_);
v___x_4366_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1(v_buckets_4356_, v___x_4364_, v___x_4365_, v_a_4352_);
lean_dec_ref(v_buckets_4356_);
if (v_isShared_4363_ == 0)
{
lean_ctor_set(v___x_4362_, 0, v___x_4366_);
v___x_4368_ = v___x_4362_;
goto v_reusejp_4367_;
}
else
{
lean_object* v_reuseFailAlloc_4369_; 
v_reuseFailAlloc_4369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4369_, 0, v___x_4366_);
v___x_4368_ = v_reuseFailAlloc_4369_;
goto v_reusejp_4367_;
}
v_reusejp_4367_:
{
return v___x_4368_;
}
}
}
}
else
{
lean_object* v___x_4373_; uint8_t v_isShared_4374_; uint8_t v_isSharedCheck_4381_; 
v_isSharedCheck_4381_ = !lean_is_exclusive(v___x_4351_);
if (v_isSharedCheck_4381_ == 0)
{
lean_object* v_unused_4382_; 
v_unused_4382_ = lean_ctor_get(v___x_4351_, 0);
lean_dec(v_unused_4382_);
v___x_4373_ = v___x_4351_;
v_isShared_4374_ = v_isSharedCheck_4381_;
goto v_resetjp_4372_;
}
else
{
lean_dec(v___x_4351_);
v___x_4373_ = lean_box(0);
v_isShared_4374_ = v_isSharedCheck_4381_;
goto v_resetjp_4372_;
}
v_resetjp_4372_:
{
size_t v___x_4375_; size_t v___x_4376_; lean_object* v___x_4377_; lean_object* v___x_4379_; 
v___x_4375_ = ((size_t)0ULL);
v___x_4376_ = lean_usize_of_nat(v___x_4358_);
v___x_4377_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getAllDecls_spec__1(v_buckets_4356_, v___x_4375_, v___x_4376_, v_a_4352_);
lean_dec_ref(v_buckets_4356_);
if (v_isShared_4374_ == 0)
{
lean_ctor_set(v___x_4373_, 0, v___x_4377_);
v___x_4379_ = v___x_4373_;
goto v_reusejp_4378_;
}
else
{
lean_object* v_reuseFailAlloc_4380_; 
v_reuseFailAlloc_4380_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4380_, 0, v___x_4377_);
v___x_4379_ = v_reuseFailAlloc_4380_;
goto v_reusejp_4378_;
}
v_reusejp_4378_:
{
return v___x_4379_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls___redArg___boxed(lean_object* v_a_4383_, lean_object* v_a_4384_){
_start:
{
lean_object* v_res_4385_; 
v_res_4385_ = lp_batteries_Batteries_Tactic_Lint_getAllDecls___redArg(v_a_4383_);
lean_dec(v_a_4383_);
return v_res_4385_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls(lean_object* v_a_4386_, lean_object* v_a_4387_){
_start:
{
lean_object* v___x_4389_; 
v___x_4389_ = lp_batteries_Batteries_Tactic_Lint_getAllDecls___redArg(v_a_4387_);
return v___x_4389_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getAllDecls___boxed(lean_object* v_a_4390_, lean_object* v_a_4391_, lean_object* v_a_4392_){
_start:
{
lean_object* v_res_4393_; 
v_res_4393_ = lp_batteries_Batteries_Tactic_Lint_getAllDecls(v_a_4390_, v_a_4391_);
lean_dec(v_a_4391_);
lean_dec_ref(v_a_4390_);
return v_res_4393_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__1(lean_object* v_msg_4394_){
_start:
{
lean_object* v___x_4395_; lean_object* v___x_4396_; 
v___x_4395_ = lean_unsigned_to_nat(0u);
v___x_4396_ = lean_panic_fn_borrowed(v___x_4395_, v_msg_4394_);
return v___x_4396_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__0(lean_object* v_pkg_4397_, size_t v_sz_4398_, size_t v_i_4399_, lean_object* v_bs_4400_){
_start:
{
uint8_t v___x_4401_; 
v___x_4401_ = lean_usize_dec_lt(v_i_4399_, v_sz_4398_);
if (v___x_4401_ == 0)
{
return v_bs_4400_;
}
else
{
lean_object* v_v_4402_; lean_object* v___x_4403_; lean_object* v_bs_x27_4404_; uint8_t v___x_4405_; size_t v___x_4406_; size_t v___x_4407_; lean_object* v___x_4408_; lean_object* v___x_4409_; 
v_v_4402_ = lean_array_uget(v_bs_4400_, v_i_4399_);
v___x_4403_ = lean_unsigned_to_nat(0u);
v_bs_x27_4404_ = lean_array_uset(v_bs_4400_, v_i_4399_, v___x_4403_);
v___x_4405_ = l_Lean_Name_isPrefixOf(v_pkg_4397_, v_v_4402_);
lean_dec(v_v_4402_);
v___x_4406_ = ((size_t)1ULL);
v___x_4407_ = lean_usize_add(v_i_4399_, v___x_4406_);
v___x_4408_ = lean_box(v___x_4405_);
v___x_4409_ = lean_array_uset(v_bs_x27_4404_, v_i_4399_, v___x_4408_);
v_i_4399_ = v___x_4407_;
v_bs_4400_ = v___x_4409_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__0___boxed(lean_object* v_pkg_4411_, lean_object* v_sz_4412_, lean_object* v_i_4413_, lean_object* v_bs_4414_){
_start:
{
size_t v_sz_boxed_4415_; size_t v_i_boxed_4416_; lean_object* v_res_4417_; 
v_sz_boxed_4415_ = lean_unbox_usize(v_sz_4412_);
lean_dec(v_sz_4412_);
v_i_boxed_4416_ = lean_unbox_usize(v_i_4413_);
lean_dec(v_i_4413_);
v_res_4417_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__0(v_pkg_4411_, v_sz_boxed_4415_, v_i_boxed_4416_, v_bs_4414_);
lean_dec(v_pkg_4411_);
return v_res_4417_;
}
}
static lean_object* _init_lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__3(void){
_start:
{
lean_object* v___x_4421_; lean_object* v___x_4422_; lean_object* v___x_4423_; lean_object* v___x_4424_; lean_object* v___x_4425_; lean_object* v___x_4426_; 
v___x_4421_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__2));
v___x_4422_ = lean_unsigned_to_nat(14u);
v___x_4423_ = lean_unsigned_to_nat(22u);
v___x_4424_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__1));
v___x_4425_ = ((lean_object*)(lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__0));
v___x_4426_ = l_mkPanicMessageWithDecl(v___x_4425_, v___x_4424_, v___x_4423_, v___x_4422_, v___x_4421_);
return v___x_4426_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2(lean_object* v___x_4427_, lean_object* v___x_4428_, lean_object* v_x_4429_, lean_object* v_x_4430_){
_start:
{
if (lean_obj_tag(v_x_4430_) == 0)
{
lean_dec_ref(v___x_4428_);
return v_x_4429_;
}
else
{
lean_object* v_key_4431_; lean_object* v_tail_4432_; uint8_t v___x_4433_; lean_object* v___y_4435_; lean_object* v___x_4442_; lean_object* v___x_4443_; 
v_key_4431_ = lean_ctor_get(v_x_4430_, 0);
lean_inc(v_key_4431_);
v_tail_4432_ = lean_ctor_get(v_x_4430_, 2);
lean_inc(v_tail_4432_);
lean_dec_ref_known(v_x_4430_, 3);
v___x_4433_ = 0;
lean_inc_ref(v___x_4428_);
v___x_4442_ = l_Lean_Environment_const2ModIdx(v___x_4428_);
v___x_4443_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Batteries_Tactic_Lint_groupedByFilename_spec__5___redArg(v___x_4442_, v_key_4431_);
lean_dec_ref(v___x_4442_);
if (lean_obj_tag(v___x_4443_) == 0)
{
lean_object* v___x_4444_; lean_object* v___x_4445_; 
v___x_4444_ = lean_obj_once(&lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__3, &lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__3_once, _init_lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___closed__3);
v___x_4445_ = lp_batteries_panic___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__1(v___x_4444_);
v___y_4435_ = v___x_4445_;
goto v___jp_4434_;
}
else
{
lean_object* v_val_4446_; 
v_val_4446_ = lean_ctor_get(v___x_4443_, 0);
lean_inc(v_val_4446_);
lean_dec_ref_known(v___x_4443_, 1);
v___y_4435_ = v_val_4446_;
goto v___jp_4434_;
}
v___jp_4434_:
{
lean_object* v___x_4436_; lean_object* v___x_4437_; uint8_t v___x_4438_; 
v___x_4436_ = lean_box(v___x_4433_);
v___x_4437_ = lean_array_get(v___x_4436_, v___x_4427_, v___y_4435_);
lean_dec(v___y_4435_);
lean_dec(v___x_4436_);
v___x_4438_ = lean_unbox(v___x_4437_);
lean_dec(v___x_4437_);
if (v___x_4438_ == 0)
{
lean_dec(v_key_4431_);
v_x_4430_ = v_tail_4432_;
goto _start;
}
else
{
lean_object* v___x_4440_; 
v___x_4440_ = lean_array_push(v_x_4429_, v_key_4431_);
v_x_4429_ = v___x_4440_;
v_x_4430_ = v_tail_4432_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2___boxed(lean_object* v___x_4447_, lean_object* v___x_4448_, lean_object* v_x_4449_, lean_object* v_x_4450_){
_start:
{
lean_object* v_res_4451_; 
v_res_4451_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2(v___x_4447_, v___x_4448_, v_x_4449_, v_x_4450_);
lean_dec_ref(v___x_4447_);
return v_res_4451_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3(lean_object* v___x_4452_, lean_object* v___x_4453_, lean_object* v_as_4454_, size_t v_i_4455_, size_t v_stop_4456_, lean_object* v_b_4457_){
_start:
{
uint8_t v___x_4458_; 
v___x_4458_ = lean_usize_dec_eq(v_i_4455_, v_stop_4456_);
if (v___x_4458_ == 0)
{
lean_object* v___x_4459_; lean_object* v___x_4460_; size_t v___x_4461_; size_t v___x_4462_; 
v___x_4459_ = lean_array_uget_borrowed(v_as_4454_, v_i_4455_);
lean_inc(v___x_4459_);
lean_inc_ref(v___x_4453_);
v___x_4460_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__2(v___x_4452_, v___x_4453_, v_b_4457_, v___x_4459_);
v___x_4461_ = ((size_t)1ULL);
v___x_4462_ = lean_usize_add(v_i_4455_, v___x_4461_);
v_i_4455_ = v___x_4462_;
v_b_4457_ = v___x_4460_;
goto _start;
}
else
{
lean_dec_ref(v___x_4453_);
return v_b_4457_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3___boxed(lean_object* v___x_4464_, lean_object* v___x_4465_, lean_object* v_as_4466_, lean_object* v_i_4467_, lean_object* v_stop_4468_, lean_object* v_b_4469_){
_start:
{
size_t v_i_boxed_4470_; size_t v_stop_boxed_4471_; lean_object* v_res_4472_; 
v_i_boxed_4470_ = lean_unbox_usize(v_i_4467_);
lean_dec(v_i_4467_);
v_stop_boxed_4471_ = lean_unbox_usize(v_stop_4468_);
lean_dec(v_stop_4468_);
v_res_4472_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3(v___x_4464_, v___x_4465_, v_as_4466_, v_i_boxed_4470_, v_stop_boxed_4471_, v_b_4469_);
lean_dec_ref(v_as_4466_);
lean_dec_ref(v___x_4464_);
return v_res_4472_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg(lean_object* v_pkg_4473_, lean_object* v_a_4474_){
_start:
{
lean_object* v___x_4476_; lean_object* v___x_4477_; lean_object* v_a_4478_; lean_object* v_env_4479_; lean_object* v___x_4480_; lean_object* v___x_4481_; lean_object* v_map_u2081_4482_; lean_object* v_buckets_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; uint8_t v___x_4486_; 
v___x_4476_ = lean_st_ref_get(v_a_4474_);
v___x_4477_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___redArg(v_a_4474_);
v_a_4478_ = lean_ctor_get(v___x_4477_, 0);
lean_inc(v_a_4478_);
v_env_4479_ = lean_ctor_get(v___x_4476_, 0);
lean_inc_ref_n(v_env_4479_, 2);
lean_dec(v___x_4476_);
v___x_4480_ = l_Lean_Environment_header(v_env_4479_);
v___x_4481_ = l_Lean_Environment_constants(v_env_4479_);
v_map_u2081_4482_ = lean_ctor_get(v___x_4481_, 0);
lean_inc_ref(v_map_u2081_4482_);
lean_dec_ref(v___x_4481_);
v_buckets_4483_ = lean_ctor_get(v_map_u2081_4482_, 1);
lean_inc_ref(v_buckets_4483_);
lean_dec_ref(v_map_u2081_4482_);
v___x_4484_ = lean_unsigned_to_nat(0u);
v___x_4485_ = lean_array_get_size(v_buckets_4483_);
v___x_4486_ = lean_nat_dec_lt(v___x_4484_, v___x_4485_);
if (v___x_4486_ == 0)
{
lean_dec_ref(v_buckets_4483_);
lean_dec_ref(v___x_4480_);
lean_dec_ref(v_env_4479_);
lean_dec(v_a_4478_);
return v___x_4477_;
}
else
{
lean_object* v___x_4487_; size_t v_sz_4488_; size_t v___x_4489_; lean_object* v___x_4490_; uint8_t v___x_4491_; 
v___x_4487_ = l_Lean_EnvironmentHeader_moduleNames(v___x_4480_);
v_sz_4488_ = lean_array_size(v___x_4487_);
v___x_4489_ = ((size_t)0ULL);
v___x_4490_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__0(v_pkg_4473_, v_sz_4488_, v___x_4489_, v___x_4487_);
v___x_4491_ = lean_nat_dec_le(v___x_4485_, v___x_4485_);
if (v___x_4491_ == 0)
{
if (v___x_4486_ == 0)
{
lean_dec_ref(v___x_4490_);
lean_dec_ref(v_buckets_4483_);
lean_dec_ref(v_env_4479_);
lean_dec(v_a_4478_);
return v___x_4477_;
}
else
{
lean_object* v___x_4493_; uint8_t v_isShared_4494_; uint8_t v_isSharedCheck_4500_; 
v_isSharedCheck_4500_ = !lean_is_exclusive(v___x_4477_);
if (v_isSharedCheck_4500_ == 0)
{
lean_object* v_unused_4501_; 
v_unused_4501_ = lean_ctor_get(v___x_4477_, 0);
lean_dec(v_unused_4501_);
v___x_4493_ = v___x_4477_;
v_isShared_4494_ = v_isSharedCheck_4500_;
goto v_resetjp_4492_;
}
else
{
lean_dec(v___x_4477_);
v___x_4493_ = lean_box(0);
v_isShared_4494_ = v_isSharedCheck_4500_;
goto v_resetjp_4492_;
}
v_resetjp_4492_:
{
size_t v___x_4495_; lean_object* v___x_4496_; lean_object* v___x_4498_; 
v___x_4495_ = lean_usize_of_nat(v___x_4485_);
v___x_4496_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3(v___x_4490_, v_env_4479_, v_buckets_4483_, v___x_4489_, v___x_4495_, v_a_4478_);
lean_dec_ref(v_buckets_4483_);
lean_dec_ref(v___x_4490_);
if (v_isShared_4494_ == 0)
{
lean_ctor_set(v___x_4493_, 0, v___x_4496_);
v___x_4498_ = v___x_4493_;
goto v_reusejp_4497_;
}
else
{
lean_object* v_reuseFailAlloc_4499_; 
v_reuseFailAlloc_4499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4499_, 0, v___x_4496_);
v___x_4498_ = v_reuseFailAlloc_4499_;
goto v_reusejp_4497_;
}
v_reusejp_4497_:
{
return v___x_4498_;
}
}
}
}
else
{
lean_object* v___x_4503_; uint8_t v_isShared_4504_; uint8_t v_isSharedCheck_4510_; 
v_isSharedCheck_4510_ = !lean_is_exclusive(v___x_4477_);
if (v_isSharedCheck_4510_ == 0)
{
lean_object* v_unused_4511_; 
v_unused_4511_ = lean_ctor_get(v___x_4477_, 0);
lean_dec(v_unused_4511_);
v___x_4503_ = v___x_4477_;
v_isShared_4504_ = v_isSharedCheck_4510_;
goto v_resetjp_4502_;
}
else
{
lean_dec(v___x_4477_);
v___x_4503_ = lean_box(0);
v_isShared_4504_ = v_isSharedCheck_4510_;
goto v_resetjp_4502_;
}
v_resetjp_4502_:
{
size_t v___x_4505_; lean_object* v___x_4506_; lean_object* v___x_4508_; 
v___x_4505_ = lean_usize_of_nat(v___x_4485_);
v___x_4506_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_Lint_getDeclsInPackage_spec__3(v___x_4490_, v_env_4479_, v_buckets_4483_, v___x_4489_, v___x_4505_, v_a_4478_);
lean_dec_ref(v_buckets_4483_);
lean_dec_ref(v___x_4490_);
if (v_isShared_4504_ == 0)
{
lean_ctor_set(v___x_4503_, 0, v___x_4506_);
v___x_4508_ = v___x_4503_;
goto v_reusejp_4507_;
}
else
{
lean_object* v_reuseFailAlloc_4509_; 
v_reuseFailAlloc_4509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4509_, 0, v___x_4506_);
v___x_4508_ = v_reuseFailAlloc_4509_;
goto v_reusejp_4507_;
}
v_reusejp_4507_:
{
return v___x_4508_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg___boxed(lean_object* v_pkg_4512_, lean_object* v_a_4513_, lean_object* v_a_4514_){
_start:
{
lean_object* v_res_4515_; 
v_res_4515_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg(v_pkg_4512_, v_a_4513_);
lean_dec(v_a_4513_);
lean_dec(v_pkg_4512_);
return v_res_4515_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage(lean_object* v_pkg_4516_, lean_object* v_a_4517_, lean_object* v_a_4518_){
_start:
{
lean_object* v___x_4520_; 
v___x_4520_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___redArg(v_pkg_4516_, v_a_4518_);
return v___x_4520_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___boxed(lean_object* v_pkg_4521_, lean_object* v_a_4522_, lean_object* v_a_4523_, lean_object* v_a_4524_){
_start:
{
lean_object* v_res_4525_; 
v_res_4525_ = lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage(v_pkg_4521_, v_a_4522_, v_a_4523_);
lean_dec(v_a_4523_);
lean_dec_ref(v_a_4522_);
lean_dec(v_pkg_4521_);
return v_res_4525_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_4651_; lean_object* v___x_4652_; lean_object* v___x_4653_; 
v___x_4651_ = lean_box(0);
v___x_4652_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_4653_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4653_, 0, v___x_4652_);
lean_ctor_set(v___x_4653_, 1, v___x_4651_);
return v___x_4653_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg(){
_start:
{
lean_object* v___x_4655_; lean_object* v___x_4656_; 
v___x_4655_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___closed__0);
v___x_4656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4656_, 0, v___x_4655_);
return v___x_4656_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg___boxed(lean_object* v___y_4657_){
_start:
{
lean_object* v_res_4658_; 
v_res_4658_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
return v_res_4658_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0(lean_object* v_00_u03b1_4659_, lean_object* v___y_4660_, lean_object* v___y_4661_){
_start:
{
lean_object* v___x_4663_; 
v___x_4663_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
return v___x_4663_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___boxed(lean_object* v_00_u03b1_4664_, lean_object* v___y_4665_, lean_object* v___y_4666_, lean_object* v___y_4667_){
_start:
{
lean_object* v_res_4668_; 
v_res_4668_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0(v_00_u03b1_4664_, v___y_4665_, v___y_4666_);
lean_dec(v___y_4666_);
lean_dec_ref(v___y_4665_);
return v_res_4668_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg(lean_object* v_t_4669_, lean_object* v___y_4670_){
_start:
{
lean_object* v___x_4672_; lean_object* v_infoState_4673_; uint8_t v_enabled_4674_; 
v___x_4672_ = lean_st_ref_get(v___y_4670_);
v_infoState_4673_ = lean_ctor_get(v___x_4672_, 7);
lean_inc_ref(v_infoState_4673_);
lean_dec(v___x_4672_);
v_enabled_4674_ = lean_ctor_get_uint8(v_infoState_4673_, sizeof(void*)*3);
lean_dec_ref(v_infoState_4673_);
if (v_enabled_4674_ == 0)
{
lean_object* v___x_4675_; lean_object* v___x_4676_; 
lean_dec_ref(v_t_4669_);
v___x_4675_ = lean_box(0);
v___x_4676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4676_, 0, v___x_4675_);
return v___x_4676_;
}
else
{
lean_object* v___x_4677_; lean_object* v_infoState_4678_; lean_object* v_env_4679_; lean_object* v_nextMacroScope_4680_; lean_object* v_ngen_4681_; lean_object* v_auxDeclNGen_4682_; lean_object* v_traceState_4683_; lean_object* v_cache_4684_; lean_object* v_messages_4685_; lean_object* v_snapshotTasks_4686_; lean_object* v___x_4688_; uint8_t v_isShared_4689_; uint8_t v_isSharedCheck_4708_; 
v___x_4677_ = lean_st_ref_take(v___y_4670_);
v_infoState_4678_ = lean_ctor_get(v___x_4677_, 7);
v_env_4679_ = lean_ctor_get(v___x_4677_, 0);
v_nextMacroScope_4680_ = lean_ctor_get(v___x_4677_, 1);
v_ngen_4681_ = lean_ctor_get(v___x_4677_, 2);
v_auxDeclNGen_4682_ = lean_ctor_get(v___x_4677_, 3);
v_traceState_4683_ = lean_ctor_get(v___x_4677_, 4);
v_cache_4684_ = lean_ctor_get(v___x_4677_, 5);
v_messages_4685_ = lean_ctor_get(v___x_4677_, 6);
v_snapshotTasks_4686_ = lean_ctor_get(v___x_4677_, 8);
v_isSharedCheck_4708_ = !lean_is_exclusive(v___x_4677_);
if (v_isSharedCheck_4708_ == 0)
{
v___x_4688_ = v___x_4677_;
v_isShared_4689_ = v_isSharedCheck_4708_;
goto v_resetjp_4687_;
}
else
{
lean_inc(v_snapshotTasks_4686_);
lean_inc(v_infoState_4678_);
lean_inc(v_messages_4685_);
lean_inc(v_cache_4684_);
lean_inc(v_traceState_4683_);
lean_inc(v_auxDeclNGen_4682_);
lean_inc(v_ngen_4681_);
lean_inc(v_nextMacroScope_4680_);
lean_inc(v_env_4679_);
lean_dec(v___x_4677_);
v___x_4688_ = lean_box(0);
v_isShared_4689_ = v_isSharedCheck_4708_;
goto v_resetjp_4687_;
}
v_resetjp_4687_:
{
uint8_t v_enabled_4690_; lean_object* v_assignment_4691_; lean_object* v_lazyAssignment_4692_; lean_object* v_trees_4693_; lean_object* v___x_4695_; uint8_t v_isShared_4696_; uint8_t v_isSharedCheck_4707_; 
v_enabled_4690_ = lean_ctor_get_uint8(v_infoState_4678_, sizeof(void*)*3);
v_assignment_4691_ = lean_ctor_get(v_infoState_4678_, 0);
v_lazyAssignment_4692_ = lean_ctor_get(v_infoState_4678_, 1);
v_trees_4693_ = lean_ctor_get(v_infoState_4678_, 2);
v_isSharedCheck_4707_ = !lean_is_exclusive(v_infoState_4678_);
if (v_isSharedCheck_4707_ == 0)
{
v___x_4695_ = v_infoState_4678_;
v_isShared_4696_ = v_isSharedCheck_4707_;
goto v_resetjp_4694_;
}
else
{
lean_inc(v_trees_4693_);
lean_inc(v_lazyAssignment_4692_);
lean_inc(v_assignment_4691_);
lean_dec(v_infoState_4678_);
v___x_4695_ = lean_box(0);
v_isShared_4696_ = v_isSharedCheck_4707_;
goto v_resetjp_4694_;
}
v_resetjp_4694_:
{
lean_object* v___x_4697_; lean_object* v___x_4699_; 
v___x_4697_ = l_Lean_PersistentArray_push___redArg(v_trees_4693_, v_t_4669_);
if (v_isShared_4696_ == 0)
{
lean_ctor_set(v___x_4695_, 2, v___x_4697_);
v___x_4699_ = v___x_4695_;
goto v_reusejp_4698_;
}
else
{
lean_object* v_reuseFailAlloc_4706_; 
v_reuseFailAlloc_4706_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_4706_, 0, v_assignment_4691_);
lean_ctor_set(v_reuseFailAlloc_4706_, 1, v_lazyAssignment_4692_);
lean_ctor_set(v_reuseFailAlloc_4706_, 2, v___x_4697_);
lean_ctor_set_uint8(v_reuseFailAlloc_4706_, sizeof(void*)*3, v_enabled_4690_);
v___x_4699_ = v_reuseFailAlloc_4706_;
goto v_reusejp_4698_;
}
v_reusejp_4698_:
{
lean_object* v___x_4701_; 
if (v_isShared_4689_ == 0)
{
lean_ctor_set(v___x_4688_, 7, v___x_4699_);
v___x_4701_ = v___x_4688_;
goto v_reusejp_4700_;
}
else
{
lean_object* v_reuseFailAlloc_4705_; 
v_reuseFailAlloc_4705_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4705_, 0, v_env_4679_);
lean_ctor_set(v_reuseFailAlloc_4705_, 1, v_nextMacroScope_4680_);
lean_ctor_set(v_reuseFailAlloc_4705_, 2, v_ngen_4681_);
lean_ctor_set(v_reuseFailAlloc_4705_, 3, v_auxDeclNGen_4682_);
lean_ctor_set(v_reuseFailAlloc_4705_, 4, v_traceState_4683_);
lean_ctor_set(v_reuseFailAlloc_4705_, 5, v_cache_4684_);
lean_ctor_set(v_reuseFailAlloc_4705_, 6, v_messages_4685_);
lean_ctor_set(v_reuseFailAlloc_4705_, 7, v___x_4699_);
lean_ctor_set(v_reuseFailAlloc_4705_, 8, v_snapshotTasks_4686_);
v___x_4701_ = v_reuseFailAlloc_4705_;
goto v_reusejp_4700_;
}
v_reusejp_4700_:
{
lean_object* v___x_4702_; lean_object* v___x_4703_; lean_object* v___x_4704_; 
v___x_4702_ = lean_st_ref_set(v___y_4670_, v___x_4701_);
v___x_4703_ = lean_box(0);
v___x_4704_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4704_, 0, v___x_4703_);
return v___x_4704_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_t_4709_, lean_object* v___y_4710_, lean_object* v___y_4711_){
_start:
{
lean_object* v_res_4712_; 
v_res_4712_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg(v_t_4709_, v___y_4710_);
lean_dec(v___y_4710_);
return v_res_4712_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__0(void){
_start:
{
lean_object* v___x_4713_; lean_object* v___x_4714_; lean_object* v___x_4715_; 
v___x_4713_ = lean_unsigned_to_nat(32u);
v___x_4714_ = lean_mk_empty_array_with_capacity(v___x_4713_);
v___x_4715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4715_, 0, v___x_4714_);
return v___x_4715_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__1(void){
_start:
{
size_t v___x_4716_; lean_object* v___x_4717_; lean_object* v___x_4718_; lean_object* v___x_4719_; lean_object* v___x_4720_; lean_object* v___x_4721_; 
v___x_4716_ = ((size_t)5ULL);
v___x_4717_ = lean_unsigned_to_nat(0u);
v___x_4718_ = lean_unsigned_to_nat(32u);
v___x_4719_ = lean_mk_empty_array_with_capacity(v___x_4718_);
v___x_4720_ = lean_obj_once(&lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__0, &lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__0_once, _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__0);
v___x_4721_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_4721_, 0, v___x_4720_);
lean_ctor_set(v___x_4721_, 1, v___x_4719_);
lean_ctor_set(v___x_4721_, 2, v___x_4717_);
lean_ctor_set(v___x_4721_, 3, v___x_4717_);
lean_ctor_set_usize(v___x_4721_, 4, v___x_4716_);
return v___x_4721_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1(lean_object* v_t_4722_, lean_object* v___y_4723_, lean_object* v___y_4724_){
_start:
{
lean_object* v___x_4726_; lean_object* v_infoState_4727_; uint8_t v_enabled_4728_; 
v___x_4726_ = lean_st_ref_get(v___y_4724_);
v_infoState_4727_ = lean_ctor_get(v___x_4726_, 7);
lean_inc_ref(v_infoState_4727_);
lean_dec(v___x_4726_);
v_enabled_4728_ = lean_ctor_get_uint8(v_infoState_4727_, sizeof(void*)*3);
lean_dec_ref(v_infoState_4727_);
if (v_enabled_4728_ == 0)
{
lean_object* v___x_4729_; lean_object* v___x_4730_; 
lean_dec_ref(v_t_4722_);
v___x_4729_ = lean_box(0);
v___x_4730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4730_, 0, v___x_4729_);
return v___x_4730_;
}
else
{
lean_object* v___x_4731_; lean_object* v___x_4732_; lean_object* v___x_4733_; 
v___x_4731_ = lean_obj_once(&lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__1, &lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__1_once, _init_lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___closed__1);
v___x_4732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4732_, 0, v_t_4722_);
lean_ctor_set(v___x_4732_, 1, v___x_4731_);
v___x_4733_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg(v___x_4732_, v___y_4724_);
return v___x_4733_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1___boxed(lean_object* v_t_4734_, lean_object* v___y_4735_, lean_object* v___y_4736_, lean_object* v___y_4737_){
_start:
{
lean_object* v_res_4738_; 
v_res_4738_ = lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1(v_t_4734_, v___y_4735_, v___y_4736_);
lean_dec(v___y_4736_);
lean_dec_ref(v___y_4735_);
return v_res_4738_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1(lean_object* v_stx_4739_, lean_object* v_n_4740_, lean_object* v_expectedType_x3f_4741_, lean_object* v___y_4742_, lean_object* v___y_4743_){
_start:
{
lean_object* v___x_4745_; 
v___x_4745_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0(v_n_4740_, v___y_4742_, v___y_4743_);
if (lean_obj_tag(v___x_4745_) == 0)
{
lean_object* v_a_4746_; lean_object* v___x_4747_; lean_object* v___x_4748_; lean_object* v___x_4749_; uint8_t v___x_4750_; lean_object* v___x_4751_; lean_object* v___x_4752_; lean_object* v___x_4753_; 
v_a_4746_ = lean_ctor_get(v___x_4745_, 0);
lean_inc(v_a_4746_);
lean_dec_ref_known(v___x_4745_, 1);
v___x_4747_ = lean_box(0);
v___x_4748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4748_, 0, v___x_4747_);
lean_ctor_set(v___x_4748_, 1, v_stx_4739_);
v___x_4749_ = l_Lean_LocalContext_empty;
v___x_4750_ = 0;
v___x_4751_ = lean_alloc_ctor(0, 4, 2);
lean_ctor_set(v___x_4751_, 0, v___x_4748_);
lean_ctor_set(v___x_4751_, 1, v___x_4749_);
lean_ctor_set(v___x_4751_, 2, v_expectedType_x3f_4741_);
lean_ctor_set(v___x_4751_, 3, v_a_4746_);
lean_ctor_set_uint8(v___x_4751_, sizeof(void*)*4, v___x_4750_);
lean_ctor_set_uint8(v___x_4751_, sizeof(void*)*4 + 1, v___x_4750_);
v___x_4752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4752_, 0, v___x_4751_);
v___x_4753_ = lp_batteries_Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1(v___x_4752_, v___y_4742_, v___y_4743_);
return v___x_4753_;
}
else
{
lean_object* v_a_4754_; lean_object* v___x_4756_; uint8_t v_isShared_4757_; uint8_t v_isSharedCheck_4761_; 
lean_dec(v_expectedType_x3f_4741_);
lean_dec(v_stx_4739_);
v_a_4754_ = lean_ctor_get(v___x_4745_, 0);
v_isSharedCheck_4761_ = !lean_is_exclusive(v___x_4745_);
if (v_isSharedCheck_4761_ == 0)
{
v___x_4756_ = v___x_4745_;
v_isShared_4757_ = v_isSharedCheck_4761_;
goto v_resetjp_4755_;
}
else
{
lean_inc(v_a_4754_);
lean_dec(v___x_4745_);
v___x_4756_ = lean_box(0);
v_isShared_4757_ = v_isSharedCheck_4761_;
goto v_resetjp_4755_;
}
v_resetjp_4755_:
{
lean_object* v___x_4759_; 
if (v_isShared_4757_ == 0)
{
v___x_4759_ = v___x_4756_;
goto v_reusejp_4758_;
}
else
{
lean_object* v_reuseFailAlloc_4760_; 
v_reuseFailAlloc_4760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4760_, 0, v_a_4754_);
v___x_4759_ = v_reuseFailAlloc_4760_;
goto v_reusejp_4758_;
}
v_reusejp_4758_:
{
return v___x_4759_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1___boxed(lean_object* v_stx_4762_, lean_object* v_n_4763_, lean_object* v_expectedType_x3f_4764_, lean_object* v___y_4765_, lean_object* v___y_4766_, lean_object* v___y_4767_){
_start:
{
lean_object* v_res_4768_; 
v_res_4768_ = lp_batteries_Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1(v_stx_4762_, v_n_4763_, v_expectedType_x3f_4764_, v___y_4765_, v___y_4766_);
lean_dec(v___y_4766_);
lean_dec_ref(v___y_4765_);
return v_res_4768_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_4770_; lean_object* v___x_4771_; 
v___x_4770_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__0));
v___x_4771_ = l_Lean_stringToMessageData(v___x_4770_);
return v___x_4771_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2(lean_object* v___x_4772_, lean_object* v_as_4773_, size_t v_sz_4774_, size_t v_i_4775_, lean_object* v_b_4776_, lean_object* v___y_4777_, lean_object* v___y_4778_){
_start:
{
lean_object* v_a_4781_; uint8_t v___x_4785_; 
v___x_4785_ = lean_usize_dec_lt(v_i_4775_, v_sz_4774_);
if (v___x_4785_ == 0)
{
lean_object* v___x_4786_; 
v___x_4786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4786_, 0, v_b_4776_);
return v___x_4786_;
}
else
{
lean_object* v_a_4787_; lean_object* v___x_4788_; lean_object* v___x_4789_; lean_object* v___x_4790_; 
v_a_4787_ = lean_array_uget_borrowed(v_as_4773_, v_i_4775_);
v___x_4788_ = l_Lean_TSyntax_getId(v_a_4787_);
v___x_4789_ = l_Lean_Name_eraseMacroScopes(v___x_4788_);
lean_dec(v___x_4788_);
v___x_4790_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___x_4772_, v___x_4789_);
if (lean_obj_tag(v___x_4790_) == 1)
{
lean_object* v_val_4791_; lean_object* v_fst_4792_; lean_object* v___x_4793_; lean_object* v___x_4794_; 
v_val_4791_ = lean_ctor_get(v___x_4790_, 0);
lean_inc(v_val_4791_);
lean_dec_ref_known(v___x_4790_, 1);
v_fst_4792_ = lean_ctor_get(v_val_4791_, 0);
lean_inc_n(v_fst_4792_, 2);
lean_dec(v_val_4791_);
v___x_4793_ = lean_box(0);
lean_inc(v_a_4787_);
v___x_4794_ = lp_batteries_Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1(v_a_4787_, v_fst_4792_, v___x_4793_, v___y_4777_, v___y_4778_);
if (lean_obj_tag(v___x_4794_) == 0)
{
lean_object* v___x_4795_; 
lean_dec_ref_known(v___x_4794_, 1);
v___x_4795_ = lp_batteries_Batteries_Tactic_Lint_getLinter(v___x_4789_, v_fst_4792_, v___y_4777_, v___y_4778_);
if (lean_obj_tag(v___x_4795_) == 0)
{
lean_object* v_a_4796_; lean_object* v___x_4797_; 
v_a_4796_ = lean_ctor_get(v___x_4795_, 0);
lean_inc_n(v_a_4796_, 2);
lean_dec_ref_known(v___x_4795_, 1);
v___x_4797_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint_getChecks_spec__0(v_a_4796_, v_b_4776_, v_a_4796_);
lean_dec(v_a_4796_);
v_a_4781_ = v___x_4797_;
goto v___jp_4780_;
}
else
{
lean_object* v_a_4798_; lean_object* v___x_4800_; uint8_t v_isShared_4801_; uint8_t v_isSharedCheck_4805_; 
lean_dec_ref(v_b_4776_);
v_a_4798_ = lean_ctor_get(v___x_4795_, 0);
v_isSharedCheck_4805_ = !lean_is_exclusive(v___x_4795_);
if (v_isSharedCheck_4805_ == 0)
{
v___x_4800_ = v___x_4795_;
v_isShared_4801_ = v_isSharedCheck_4805_;
goto v_resetjp_4799_;
}
else
{
lean_inc(v_a_4798_);
lean_dec(v___x_4795_);
v___x_4800_ = lean_box(0);
v_isShared_4801_ = v_isSharedCheck_4805_;
goto v_resetjp_4799_;
}
v_resetjp_4799_:
{
lean_object* v___x_4803_; 
if (v_isShared_4801_ == 0)
{
v___x_4803_ = v___x_4800_;
goto v_reusejp_4802_;
}
else
{
lean_object* v_reuseFailAlloc_4804_; 
v_reuseFailAlloc_4804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4804_, 0, v_a_4798_);
v___x_4803_ = v_reuseFailAlloc_4804_;
goto v_reusejp_4802_;
}
v_reusejp_4802_:
{
return v___x_4803_;
}
}
}
}
else
{
lean_object* v_a_4806_; lean_object* v___x_4808_; uint8_t v_isShared_4809_; uint8_t v_isSharedCheck_4813_; 
lean_dec(v_fst_4792_);
lean_dec(v___x_4789_);
lean_dec_ref(v_b_4776_);
v_a_4806_ = lean_ctor_get(v___x_4794_, 0);
v_isSharedCheck_4813_ = !lean_is_exclusive(v___x_4794_);
if (v_isSharedCheck_4813_ == 0)
{
v___x_4808_ = v___x_4794_;
v_isShared_4809_ = v_isSharedCheck_4813_;
goto v_resetjp_4807_;
}
else
{
lean_inc(v_a_4806_);
lean_dec(v___x_4794_);
v___x_4808_ = lean_box(0);
v_isShared_4809_ = v_isSharedCheck_4813_;
goto v_resetjp_4807_;
}
v_resetjp_4807_:
{
lean_object* v___x_4811_; 
if (v_isShared_4809_ == 0)
{
v___x_4811_ = v___x_4808_;
goto v_reusejp_4810_;
}
else
{
lean_object* v_reuseFailAlloc_4812_; 
v_reuseFailAlloc_4812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4812_, 0, v_a_4806_);
v___x_4811_ = v_reuseFailAlloc_4812_;
goto v_reusejp_4810_;
}
v_reusejp_4810_:
{
return v___x_4811_;
}
}
}
}
else
{
lean_object* v___x_4814_; lean_object* v___x_4815_; lean_object* v___x_4816_; lean_object* v___x_4817_; 
lean_dec(v___x_4790_);
v___x_4814_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___closed__1);
v___x_4815_ = l_Lean_MessageData_ofName(v___x_4789_);
v___x_4816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4816_, 0, v___x_4814_);
lean_ctor_set(v___x_4816_, 1, v___x_4815_);
v___x_4817_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic_Lint_printWarning_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_a_4787_, v___x_4816_, v___y_4777_, v___y_4778_);
if (lean_obj_tag(v___x_4817_) == 0)
{
lean_dec_ref_known(v___x_4817_, 1);
v_a_4781_ = v_b_4776_;
goto v___jp_4780_;
}
else
{
lean_object* v_a_4818_; lean_object* v___x_4820_; uint8_t v_isShared_4821_; uint8_t v_isSharedCheck_4825_; 
lean_dec_ref(v_b_4776_);
v_a_4818_ = lean_ctor_get(v___x_4817_, 0);
v_isSharedCheck_4825_ = !lean_is_exclusive(v___x_4817_);
if (v_isSharedCheck_4825_ == 0)
{
v___x_4820_ = v___x_4817_;
v_isShared_4821_ = v_isSharedCheck_4825_;
goto v_resetjp_4819_;
}
else
{
lean_inc(v_a_4818_);
lean_dec(v___x_4817_);
v___x_4820_ = lean_box(0);
v_isShared_4821_ = v_isSharedCheck_4825_;
goto v_resetjp_4819_;
}
v_resetjp_4819_:
{
lean_object* v___x_4823_; 
if (v_isShared_4821_ == 0)
{
v___x_4823_ = v___x_4820_;
goto v_reusejp_4822_;
}
else
{
lean_object* v_reuseFailAlloc_4824_; 
v_reuseFailAlloc_4824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4824_, 0, v_a_4818_);
v___x_4823_ = v_reuseFailAlloc_4824_;
goto v_reusejp_4822_;
}
v_reusejp_4822_:
{
return v___x_4823_;
}
}
}
}
}
v___jp_4780_:
{
size_t v___x_4782_; size_t v___x_4783_; 
v___x_4782_ = ((size_t)1ULL);
v___x_4783_ = lean_usize_add(v_i_4775_, v___x_4782_);
v_i_4775_ = v___x_4783_;
v_b_4776_ = v_a_4781_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2___boxed(lean_object* v___x_4826_, lean_object* v_as_4827_, lean_object* v_sz_4828_, lean_object* v_i_4829_, lean_object* v_b_4830_, lean_object* v___y_4831_, lean_object* v___y_4832_, lean_object* v___y_4833_){
_start:
{
size_t v_sz_boxed_4834_; size_t v_i_boxed_4835_; lean_object* v_res_4836_; 
v_sz_boxed_4834_ = lean_unbox_usize(v_sz_4828_);
lean_dec(v_sz_4828_);
v_i_boxed_4835_ = lean_unbox_usize(v_i_4829_);
lean_dec(v_i_4829_);
v_res_4836_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2(v___x_4826_, v_as_4827_, v_sz_boxed_4834_, v_i_boxed_4835_, v_b_4830_, v___y_4831_, v___y_4832_);
lean_dec(v___y_4832_);
lean_dec_ref(v___y_4831_);
lean_dec_ref(v_as_4827_);
lean_dec(v___x_4826_);
return v_res_4836_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___lam__0(uint8_t v___y_4837_, lean_object* v___y_4838_, lean_object* v___x_4839_, lean_object* v_linters_4840_, lean_object* v___y_4841_, lean_object* v___y_4842_){
_start:
{
lean_object* v___x_4844_; 
v___x_4844_ = lp_batteries_Batteries_Tactic_Lint_getChecks(v___y_4837_, v___y_4838_, v___x_4839_, v___y_4841_, v___y_4842_);
if (lean_obj_tag(v___x_4844_) == 0)
{
lean_object* v_a_4845_; lean_object* v___x_4846_; lean_object* v_env_4847_; lean_object* v___x_4848_; lean_object* v_toEnvExtension_4849_; lean_object* v_asyncMode_4850_; lean_object* v___x_4851_; lean_object* v___x_4852_; lean_object* v___x_4853_; size_t v_sz_4854_; size_t v___x_4855_; lean_object* v___x_4856_; 
v_a_4845_ = lean_ctor_get(v___x_4844_, 0);
lean_inc(v_a_4845_);
lean_dec_ref_known(v___x_4844_, 1);
v___x_4846_ = lean_st_ref_get(v___y_4842_);
v_env_4847_ = lean_ctor_get(v___x_4846_, 0);
lean_inc_ref(v_env_4847_);
lean_dec(v___x_4846_);
v___x_4848_ = lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
v_toEnvExtension_4849_ = lean_ctor_get(v___x_4848_, 0);
v_asyncMode_4850_ = lean_ctor_get(v_toEnvExtension_4849_, 2);
v___x_4851_ = lean_box(1);
v___x_4852_ = lean_box(0);
v___x_4853_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_4851_, v___x_4848_, v_env_4847_, v_asyncMode_4850_, v___x_4852_);
v_sz_4854_ = lean_array_size(v_linters_4840_);
v___x_4855_ = ((size_t)0ULL);
v___x_4856_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__2(v___x_4853_, v_linters_4840_, v_sz_4854_, v___x_4855_, v_a_4845_, v___y_4841_, v___y_4842_);
lean_dec(v___x_4853_);
return v___x_4856_;
}
else
{
return v___x_4844_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___lam__0___boxed(lean_object* v___y_4857_, lean_object* v___y_4858_, lean_object* v___x_4859_, lean_object* v_linters_4860_, lean_object* v___y_4861_, lean_object* v___y_4862_, lean_object* v___y_4863_){
_start:
{
uint8_t v___y_10101__boxed_4864_; lean_object* v_res_4865_; 
v___y_10101__boxed_4864_ = lean_unbox(v___y_4857_);
v_res_4865_ = lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___lam__0(v___y_10101__boxed_4864_, v___y_4858_, v___x_4859_, v_linters_4860_, v___y_4861_, v___y_4862_);
lean_dec(v___y_4862_);
lean_dec_ref(v___y_4861_);
lean_dec_ref(v_linters_4860_);
lean_dec(v___x_4859_);
lean_dec(v___y_4858_);
return v_res_4865_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__7(lean_object* v_opts_4866_, lean_object* v_opt_4867_){
_start:
{
lean_object* v_name_4868_; lean_object* v_defValue_4869_; lean_object* v_map_4870_; lean_object* v___x_4871_; 
v_name_4868_ = lean_ctor_get(v_opt_4867_, 0);
v_defValue_4869_ = lean_ctor_get(v_opt_4867_, 1);
v_map_4870_ = lean_ctor_get(v_opts_4866_, 0);
v___x_4871_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_4870_, v_name_4868_);
if (lean_obj_tag(v___x_4871_) == 0)
{
uint8_t v___x_4872_; 
v___x_4872_ = lean_unbox(v_defValue_4869_);
return v___x_4872_;
}
else
{
lean_object* v_val_4873_; 
v_val_4873_ = lean_ctor_get(v___x_4871_, 0);
lean_inc(v_val_4873_);
lean_dec_ref_known(v___x_4871_, 1);
if (lean_obj_tag(v_val_4873_) == 1)
{
uint8_t v_v_4874_; 
v_v_4874_ = lean_ctor_get_uint8(v_val_4873_, 0);
lean_dec_ref_known(v_val_4873_, 0);
return v_v_4874_;
}
else
{
uint8_t v___x_4875_; 
lean_dec(v_val_4873_);
v___x_4875_ = lean_unbox(v_defValue_4869_);
return v___x_4875_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__7___boxed(lean_object* v_opts_4876_, lean_object* v_opt_4877_){
_start:
{
uint8_t v_res_4878_; lean_object* v_r_4879_; 
v_res_4878_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__7(v_opts_4876_, v_opt_4877_);
lean_dec_ref(v_opt_4877_);
lean_dec_ref(v_opts_4876_);
v_r_4879_ = lean_box(v_res_4878_);
return v_r_4879_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___lam__0(uint8_t v___y_4880_, uint8_t v_suppressElabErrors_4881_, lean_object* v_x_4882_){
_start:
{
if (lean_obj_tag(v_x_4882_) == 1)
{
lean_object* v_pre_4883_; 
v_pre_4883_ = lean_ctor_get(v_x_4882_, 0);
if (lean_obj_tag(v_pre_4883_) == 0)
{
lean_object* v_str_4884_; lean_object* v___x_4885_; uint8_t v___x_4886_; 
v_str_4884_ = lean_ctor_get(v_x_4882_, 1);
v___x_4885_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__3));
v___x_4886_ = lean_string_dec_eq(v_str_4884_, v___x_4885_);
if (v___x_4886_ == 0)
{
return v___y_4880_;
}
else
{
return v_suppressElabErrors_4881_;
}
}
else
{
return v___y_4880_;
}
}
else
{
return v___y_4880_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___lam__0___boxed(lean_object* v___y_4887_, lean_object* v_suppressElabErrors_4888_, lean_object* v_x_4889_){
_start:
{
uint8_t v___y_10149__boxed_4890_; uint8_t v_suppressElabErrors_boxed_4891_; uint8_t v_res_4892_; lean_object* v_r_4893_; 
v___y_10149__boxed_4890_ = lean_unbox(v___y_4887_);
v_suppressElabErrors_boxed_4891_ = lean_unbox(v_suppressElabErrors_4888_);
v_res_4892_ = lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___lam__0(v___y_10149__boxed_4890_, v_suppressElabErrors_boxed_4891_, v_x_4889_);
lean_dec(v_x_4889_);
v_r_4893_ = lean_box(v_res_4892_);
return v_r_4893_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg(lean_object* v_msgData_4894_, lean_object* v___y_4895_){
_start:
{
lean_object* v___x_4897_; lean_object* v_env_4898_; lean_object* v___x_4899_; lean_object* v_scopes_4900_; lean_object* v___x_4901_; lean_object* v___x_4902_; lean_object* v_opts_4903_; lean_object* v___x_4904_; lean_object* v___x_4905_; lean_object* v___x_4906_; lean_object* v___x_4907_; lean_object* v___x_4908_; lean_object* v___x_4909_; lean_object* v___x_4910_; 
v___x_4897_ = lean_st_ref_get(v___y_4895_);
v_env_4898_ = lean_ctor_get(v___x_4897_, 0);
lean_inc_ref(v_env_4898_);
lean_dec(v___x_4897_);
v___x_4899_ = lean_st_ref_get(v___y_4895_);
v_scopes_4900_ = lean_ctor_get(v___x_4899_, 2);
lean_inc(v_scopes_4900_);
lean_dec(v___x_4899_);
v___x_4901_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_4902_ = l_List_head_x21___redArg(v___x_4901_, v_scopes_4900_);
lean_dec(v_scopes_4900_);
v_opts_4903_ = lean_ctor_get(v___x_4902_, 1);
lean_inc_ref(v_opts_4903_);
lean_dec(v___x_4902_);
v___x_4904_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__2);
v___x_4905_ = lean_unsigned_to_nat(32u);
v___x_4906_ = lean_mk_empty_array_with_capacity(v___x_4905_);
lean_dec_ref(v___x_4906_);
v___x_4907_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2_spec__5___closed__5);
v___x_4908_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4908_, 0, v_env_4898_);
lean_ctor_set(v___x_4908_, 1, v___x_4904_);
lean_ctor_set(v___x_4908_, 2, v___x_4907_);
lean_ctor_set(v___x_4908_, 3, v_opts_4903_);
v___x_4909_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_4909_, 0, v___x_4908_);
lean_ctor_set(v___x_4909_, 1, v_msgData_4894_);
v___x_4910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4910_, 0, v___x_4909_);
return v___x_4910_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg___boxed(lean_object* v_msgData_4911_, lean_object* v___y_4912_, lean_object* v___y_4913_){
_start:
{
lean_object* v_res_4914_; 
v_res_4914_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg(v_msgData_4911_, v___y_4912_);
lean_dec(v___y_4912_);
return v_res_4914_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4(lean_object* v_ref_4915_, lean_object* v_msgData_4916_, uint8_t v_severity_4917_, uint8_t v_isSilent_4918_, lean_object* v___y_4919_, lean_object* v___y_4920_){
_start:
{
lean_object* v___y_4923_; lean_object* v___y_4924_; uint8_t v___y_4925_; lean_object* v___y_4926_; uint8_t v___y_4927_; lean_object* v___y_4928_; lean_object* v___y_4929_; lean_object* v___y_4930_; uint8_t v___y_4987_; uint8_t v___y_4988_; uint8_t v___y_4989_; lean_object* v___y_4990_; lean_object* v___y_4991_; uint8_t v___y_5015_; lean_object* v___y_5016_; uint8_t v___y_5017_; uint8_t v___y_5018_; lean_object* v___y_5019_; uint8_t v___y_5023_; uint8_t v___y_5024_; uint8_t v___y_5025_; uint8_t v___x_5040_; uint8_t v___y_5042_; uint8_t v___y_5043_; uint8_t v___y_5044_; uint8_t v___y_5046_; uint8_t v___x_5058_; 
v___x_5040_ = 2;
v___x_5058_ = l_Lean_instBEqMessageSeverity_beq(v_severity_4917_, v___x_5040_);
if (v___x_5058_ == 0)
{
v___y_5046_ = v___x_5058_;
goto v___jp_5045_;
}
else
{
uint8_t v___x_5059_; 
lean_inc_ref(v_msgData_4916_);
v___x_5059_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_4916_);
v___y_5046_ = v___x_5059_;
goto v___jp_5045_;
}
v___jp_4922_:
{
lean_object* v___x_4931_; 
v___x_4931_ = l_Lean_Elab_Command_getScope___redArg(v___y_4930_);
if (lean_obj_tag(v___x_4931_) == 0)
{
lean_object* v_a_4932_; lean_object* v___x_4933_; 
v_a_4932_ = lean_ctor_get(v___x_4931_, 0);
lean_inc(v_a_4932_);
lean_dec_ref_known(v___x_4931_, 1);
v___x_4933_ = l_Lean_Elab_Command_getScope___redArg(v___y_4930_);
if (lean_obj_tag(v___x_4933_) == 0)
{
lean_object* v_a_4934_; lean_object* v___x_4936_; uint8_t v_isShared_4937_; uint8_t v_isSharedCheck_4969_; 
v_a_4934_ = lean_ctor_get(v___x_4933_, 0);
v_isSharedCheck_4969_ = !lean_is_exclusive(v___x_4933_);
if (v_isSharedCheck_4969_ == 0)
{
v___x_4936_ = v___x_4933_;
v_isShared_4937_ = v_isSharedCheck_4969_;
goto v_resetjp_4935_;
}
else
{
lean_inc(v_a_4934_);
lean_dec(v___x_4933_);
v___x_4936_ = lean_box(0);
v_isShared_4937_ = v_isSharedCheck_4969_;
goto v_resetjp_4935_;
}
v_resetjp_4935_:
{
lean_object* v___x_4938_; lean_object* v_currNamespace_4939_; lean_object* v_openDecls_4940_; lean_object* v_env_4941_; lean_object* v_messages_4942_; lean_object* v_scopes_4943_; lean_object* v_usedQuotCtxts_4944_; lean_object* v_nextMacroScope_4945_; lean_object* v_maxRecDepth_4946_; lean_object* v_ngen_4947_; lean_object* v_auxDeclNGen_4948_; lean_object* v_infoState_4949_; lean_object* v_traceState_4950_; lean_object* v_snapshotTasks_4951_; lean_object* v_prevLinterStates_4952_; lean_object* v___x_4954_; uint8_t v_isShared_4955_; uint8_t v_isSharedCheck_4968_; 
v___x_4938_ = lean_st_ref_take(v___y_4930_);
v_currNamespace_4939_ = lean_ctor_get(v_a_4932_, 2);
lean_inc(v_currNamespace_4939_);
lean_dec(v_a_4932_);
v_openDecls_4940_ = lean_ctor_get(v_a_4934_, 3);
lean_inc(v_openDecls_4940_);
lean_dec(v_a_4934_);
v_env_4941_ = lean_ctor_get(v___x_4938_, 0);
v_messages_4942_ = lean_ctor_get(v___x_4938_, 1);
v_scopes_4943_ = lean_ctor_get(v___x_4938_, 2);
v_usedQuotCtxts_4944_ = lean_ctor_get(v___x_4938_, 3);
v_nextMacroScope_4945_ = lean_ctor_get(v___x_4938_, 4);
v_maxRecDepth_4946_ = lean_ctor_get(v___x_4938_, 5);
v_ngen_4947_ = lean_ctor_get(v___x_4938_, 6);
v_auxDeclNGen_4948_ = lean_ctor_get(v___x_4938_, 7);
v_infoState_4949_ = lean_ctor_get(v___x_4938_, 8);
v_traceState_4950_ = lean_ctor_get(v___x_4938_, 9);
v_snapshotTasks_4951_ = lean_ctor_get(v___x_4938_, 10);
v_prevLinterStates_4952_ = lean_ctor_get(v___x_4938_, 11);
v_isSharedCheck_4968_ = !lean_is_exclusive(v___x_4938_);
if (v_isSharedCheck_4968_ == 0)
{
v___x_4954_ = v___x_4938_;
v_isShared_4955_ = v_isSharedCheck_4968_;
goto v_resetjp_4953_;
}
else
{
lean_inc(v_prevLinterStates_4952_);
lean_inc(v_snapshotTasks_4951_);
lean_inc(v_traceState_4950_);
lean_inc(v_infoState_4949_);
lean_inc(v_auxDeclNGen_4948_);
lean_inc(v_ngen_4947_);
lean_inc(v_maxRecDepth_4946_);
lean_inc(v_nextMacroScope_4945_);
lean_inc(v_usedQuotCtxts_4944_);
lean_inc(v_scopes_4943_);
lean_inc(v_messages_4942_);
lean_inc(v_env_4941_);
lean_dec(v___x_4938_);
v___x_4954_ = lean_box(0);
v_isShared_4955_ = v_isSharedCheck_4968_;
goto v_resetjp_4953_;
}
v_resetjp_4953_:
{
lean_object* v___x_4956_; lean_object* v___x_4957_; lean_object* v___x_4958_; lean_object* v___x_4959_; lean_object* v___x_4961_; 
v___x_4956_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4956_, 0, v_currNamespace_4939_);
lean_ctor_set(v___x_4956_, 1, v_openDecls_4940_);
v___x_4957_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_4957_, 0, v___x_4956_);
lean_ctor_set(v___x_4957_, 1, v___y_4923_);
lean_inc_ref(v___y_4926_);
lean_inc_ref(v___y_4924_);
v___x_4958_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_4958_, 0, v___y_4924_);
lean_ctor_set(v___x_4958_, 1, v___y_4929_);
lean_ctor_set(v___x_4958_, 2, v___y_4928_);
lean_ctor_set(v___x_4958_, 3, v___y_4926_);
lean_ctor_set(v___x_4958_, 4, v___x_4957_);
lean_ctor_set_uint8(v___x_4958_, sizeof(void*)*5, v___y_4925_);
lean_ctor_set_uint8(v___x_4958_, sizeof(void*)*5 + 1, v___y_4927_);
lean_ctor_set_uint8(v___x_4958_, sizeof(void*)*5 + 2, v_isSilent_4918_);
v___x_4959_ = l_Lean_MessageLog_add(v___x_4958_, v_messages_4942_);
if (v_isShared_4955_ == 0)
{
lean_ctor_set(v___x_4954_, 1, v___x_4959_);
v___x_4961_ = v___x_4954_;
goto v_reusejp_4960_;
}
else
{
lean_object* v_reuseFailAlloc_4967_; 
v_reuseFailAlloc_4967_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_4967_, 0, v_env_4941_);
lean_ctor_set(v_reuseFailAlloc_4967_, 1, v___x_4959_);
lean_ctor_set(v_reuseFailAlloc_4967_, 2, v_scopes_4943_);
lean_ctor_set(v_reuseFailAlloc_4967_, 3, v_usedQuotCtxts_4944_);
lean_ctor_set(v_reuseFailAlloc_4967_, 4, v_nextMacroScope_4945_);
lean_ctor_set(v_reuseFailAlloc_4967_, 5, v_maxRecDepth_4946_);
lean_ctor_set(v_reuseFailAlloc_4967_, 6, v_ngen_4947_);
lean_ctor_set(v_reuseFailAlloc_4967_, 7, v_auxDeclNGen_4948_);
lean_ctor_set(v_reuseFailAlloc_4967_, 8, v_infoState_4949_);
lean_ctor_set(v_reuseFailAlloc_4967_, 9, v_traceState_4950_);
lean_ctor_set(v_reuseFailAlloc_4967_, 10, v_snapshotTasks_4951_);
lean_ctor_set(v_reuseFailAlloc_4967_, 11, v_prevLinterStates_4952_);
v___x_4961_ = v_reuseFailAlloc_4967_;
goto v_reusejp_4960_;
}
v_reusejp_4960_:
{
lean_object* v___x_4962_; lean_object* v___x_4963_; lean_object* v___x_4965_; 
v___x_4962_ = lean_st_ref_set(v___y_4930_, v___x_4961_);
v___x_4963_ = lean_box(0);
if (v_isShared_4937_ == 0)
{
lean_ctor_set(v___x_4936_, 0, v___x_4963_);
v___x_4965_ = v___x_4936_;
goto v_reusejp_4964_;
}
else
{
lean_object* v_reuseFailAlloc_4966_; 
v_reuseFailAlloc_4966_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4966_, 0, v___x_4963_);
v___x_4965_ = v_reuseFailAlloc_4966_;
goto v_reusejp_4964_;
}
v_reusejp_4964_:
{
return v___x_4965_;
}
}
}
}
}
else
{
lean_object* v_a_4970_; lean_object* v___x_4972_; uint8_t v_isShared_4973_; uint8_t v_isSharedCheck_4977_; 
lean_dec(v_a_4932_);
lean_dec_ref(v___y_4929_);
lean_dec(v___y_4928_);
lean_dec_ref(v___y_4923_);
v_a_4970_ = lean_ctor_get(v___x_4933_, 0);
v_isSharedCheck_4977_ = !lean_is_exclusive(v___x_4933_);
if (v_isSharedCheck_4977_ == 0)
{
v___x_4972_ = v___x_4933_;
v_isShared_4973_ = v_isSharedCheck_4977_;
goto v_resetjp_4971_;
}
else
{
lean_inc(v_a_4970_);
lean_dec(v___x_4933_);
v___x_4972_ = lean_box(0);
v_isShared_4973_ = v_isSharedCheck_4977_;
goto v_resetjp_4971_;
}
v_resetjp_4971_:
{
lean_object* v___x_4975_; 
if (v_isShared_4973_ == 0)
{
v___x_4975_ = v___x_4972_;
goto v_reusejp_4974_;
}
else
{
lean_object* v_reuseFailAlloc_4976_; 
v_reuseFailAlloc_4976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4976_, 0, v_a_4970_);
v___x_4975_ = v_reuseFailAlloc_4976_;
goto v_reusejp_4974_;
}
v_reusejp_4974_:
{
return v___x_4975_;
}
}
}
}
else
{
lean_object* v_a_4978_; lean_object* v___x_4980_; uint8_t v_isShared_4981_; uint8_t v_isSharedCheck_4985_; 
lean_dec_ref(v___y_4929_);
lean_dec(v___y_4928_);
lean_dec_ref(v___y_4923_);
v_a_4978_ = lean_ctor_get(v___x_4931_, 0);
v_isSharedCheck_4985_ = !lean_is_exclusive(v___x_4931_);
if (v_isSharedCheck_4985_ == 0)
{
v___x_4980_ = v___x_4931_;
v_isShared_4981_ = v_isSharedCheck_4985_;
goto v_resetjp_4979_;
}
else
{
lean_inc(v_a_4978_);
lean_dec(v___x_4931_);
v___x_4980_ = lean_box(0);
v_isShared_4981_ = v_isSharedCheck_4985_;
goto v_resetjp_4979_;
}
v_resetjp_4979_:
{
lean_object* v___x_4983_; 
if (v_isShared_4981_ == 0)
{
v___x_4983_ = v___x_4980_;
goto v_reusejp_4982_;
}
else
{
lean_object* v_reuseFailAlloc_4984_; 
v_reuseFailAlloc_4984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4984_, 0, v_a_4978_);
v___x_4983_ = v_reuseFailAlloc_4984_;
goto v_reusejp_4982_;
}
v_reusejp_4982_:
{
return v___x_4983_;
}
}
}
}
v___jp_4986_:
{
lean_object* v_fileName_4992_; lean_object* v_fileMap_4993_; uint8_t v_suppressElabErrors_4994_; lean_object* v___x_4995_; lean_object* v___x_4996_; lean_object* v_a_4997_; lean_object* v___x_4999_; uint8_t v_isShared_5000_; uint8_t v_isSharedCheck_5013_; 
v_fileName_4992_ = lean_ctor_get(v___y_4919_, 0);
v_fileMap_4993_ = lean_ctor_get(v___y_4919_, 1);
v_suppressElabErrors_4994_ = lean_ctor_get_uint8(v___y_4919_, sizeof(void*)*10);
v___x_4995_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_4916_);
v___x_4996_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg(v___x_4995_, v___y_4920_);
v_a_4997_ = lean_ctor_get(v___x_4996_, 0);
v_isSharedCheck_5013_ = !lean_is_exclusive(v___x_4996_);
if (v_isSharedCheck_5013_ == 0)
{
v___x_4999_ = v___x_4996_;
v_isShared_5000_ = v_isSharedCheck_5013_;
goto v_resetjp_4998_;
}
else
{
lean_inc(v_a_4997_);
lean_dec(v___x_4996_);
v___x_4999_ = lean_box(0);
v_isShared_5000_ = v_isSharedCheck_5013_;
goto v_resetjp_4998_;
}
v_resetjp_4998_:
{
lean_object* v___x_5001_; lean_object* v___x_5002_; lean_object* v___x_5003_; lean_object* v___x_5004_; 
lean_inc_ref_n(v_fileMap_4993_, 2);
v___x_5001_ = l_Lean_FileMap_toPosition(v_fileMap_4993_, v___y_4990_);
lean_dec(v___y_4990_);
v___x_5002_ = l_Lean_FileMap_toPosition(v_fileMap_4993_, v___y_4991_);
lean_dec(v___y_4991_);
v___x_5003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5003_, 0, v___x_5002_);
v___x_5004_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
if (v_suppressElabErrors_4994_ == 0)
{
lean_del_object(v___x_4999_);
v___y_4923_ = v_a_4997_;
v___y_4924_ = v_fileName_4992_;
v___y_4925_ = v___y_4988_;
v___y_4926_ = v___x_5004_;
v___y_4927_ = v___y_4989_;
v___y_4928_ = v___x_5003_;
v___y_4929_ = v___x_5001_;
v___y_4930_ = v___y_4920_;
goto v___jp_4922_;
}
else
{
lean_object* v___x_5005_; lean_object* v___x_5006_; lean_object* v___f_5007_; uint8_t v___x_5008_; 
v___x_5005_ = lean_box(v___y_4987_);
v___x_5006_ = lean_box(v_suppressElabErrors_4994_);
v___f_5007_ = lean_alloc_closure((void*)(lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___lam__0___boxed), 3, 2);
lean_closure_set(v___f_5007_, 0, v___x_5005_);
lean_closure_set(v___f_5007_, 1, v___x_5006_);
lean_inc(v_a_4997_);
v___x_5008_ = l_Lean_MessageData_hasTag(v___f_5007_, v_a_4997_);
if (v___x_5008_ == 0)
{
lean_object* v___x_5009_; lean_object* v___x_5011_; 
lean_dec_ref_known(v___x_5003_, 1);
lean_dec_ref(v___x_5001_);
lean_dec(v_a_4997_);
v___x_5009_ = lean_box(0);
if (v_isShared_5000_ == 0)
{
lean_ctor_set(v___x_4999_, 0, v___x_5009_);
v___x_5011_ = v___x_4999_;
goto v_reusejp_5010_;
}
else
{
lean_object* v_reuseFailAlloc_5012_; 
v_reuseFailAlloc_5012_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5012_, 0, v___x_5009_);
v___x_5011_ = v_reuseFailAlloc_5012_;
goto v_reusejp_5010_;
}
v_reusejp_5010_:
{
return v___x_5011_;
}
}
else
{
lean_del_object(v___x_4999_);
v___y_4923_ = v_a_4997_;
v___y_4924_ = v_fileName_4992_;
v___y_4925_ = v___y_4988_;
v___y_4926_ = v___x_5004_;
v___y_4927_ = v___y_4989_;
v___y_4928_ = v___x_5003_;
v___y_4929_ = v___x_5001_;
v___y_4930_ = v___y_4920_;
goto v___jp_4922_;
}
}
}
}
v___jp_5014_:
{
lean_object* v___x_5020_; 
v___x_5020_ = l_Lean_Syntax_getTailPos_x3f(v___y_5016_, v___y_5017_);
lean_dec(v___y_5016_);
if (lean_obj_tag(v___x_5020_) == 0)
{
lean_inc(v___y_5019_);
v___y_4987_ = v___y_5015_;
v___y_4988_ = v___y_5017_;
v___y_4989_ = v___y_5018_;
v___y_4990_ = v___y_5019_;
v___y_4991_ = v___y_5019_;
goto v___jp_4986_;
}
else
{
lean_object* v_val_5021_; 
v_val_5021_ = lean_ctor_get(v___x_5020_, 0);
lean_inc(v_val_5021_);
lean_dec_ref_known(v___x_5020_, 1);
v___y_4987_ = v___y_5015_;
v___y_4988_ = v___y_5017_;
v___y_4989_ = v___y_5018_;
v___y_4990_ = v___y_5019_;
v___y_4991_ = v_val_5021_;
goto v___jp_4986_;
}
}
v___jp_5022_:
{
lean_object* v___x_5026_; 
v___x_5026_ = l_Lean_Elab_Command_getRef___redArg(v___y_4919_);
if (lean_obj_tag(v___x_5026_) == 0)
{
lean_object* v_a_5027_; lean_object* v_ref_5028_; lean_object* v___x_5029_; 
v_a_5027_ = lean_ctor_get(v___x_5026_, 0);
lean_inc(v_a_5027_);
lean_dec_ref_known(v___x_5026_, 1);
v_ref_5028_ = l_Lean_replaceRef(v_ref_4915_, v_a_5027_);
lean_dec(v_a_5027_);
v___x_5029_ = l_Lean_Syntax_getPos_x3f(v_ref_5028_, v___y_5024_);
if (lean_obj_tag(v___x_5029_) == 0)
{
lean_object* v___x_5030_; 
v___x_5030_ = lean_unsigned_to_nat(0u);
v___y_5015_ = v___y_5023_;
v___y_5016_ = v_ref_5028_;
v___y_5017_ = v___y_5024_;
v___y_5018_ = v___y_5025_;
v___y_5019_ = v___x_5030_;
goto v___jp_5014_;
}
else
{
lean_object* v_val_5031_; 
v_val_5031_ = lean_ctor_get(v___x_5029_, 0);
lean_inc(v_val_5031_);
lean_dec_ref_known(v___x_5029_, 1);
v___y_5015_ = v___y_5023_;
v___y_5016_ = v_ref_5028_;
v___y_5017_ = v___y_5024_;
v___y_5018_ = v___y_5025_;
v___y_5019_ = v_val_5031_;
goto v___jp_5014_;
}
}
else
{
lean_object* v_a_5032_; lean_object* v___x_5034_; uint8_t v_isShared_5035_; uint8_t v_isSharedCheck_5039_; 
lean_dec_ref(v_msgData_4916_);
v_a_5032_ = lean_ctor_get(v___x_5026_, 0);
v_isSharedCheck_5039_ = !lean_is_exclusive(v___x_5026_);
if (v_isSharedCheck_5039_ == 0)
{
v___x_5034_ = v___x_5026_;
v_isShared_5035_ = v_isSharedCheck_5039_;
goto v_resetjp_5033_;
}
else
{
lean_inc(v_a_5032_);
lean_dec(v___x_5026_);
v___x_5034_ = lean_box(0);
v_isShared_5035_ = v_isSharedCheck_5039_;
goto v_resetjp_5033_;
}
v_resetjp_5033_:
{
lean_object* v___x_5037_; 
if (v_isShared_5035_ == 0)
{
v___x_5037_ = v___x_5034_;
goto v_reusejp_5036_;
}
else
{
lean_object* v_reuseFailAlloc_5038_; 
v_reuseFailAlloc_5038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5038_, 0, v_a_5032_);
v___x_5037_ = v_reuseFailAlloc_5038_;
goto v_reusejp_5036_;
}
v_reusejp_5036_:
{
return v___x_5037_;
}
}
}
}
v___jp_5041_:
{
if (v___y_5044_ == 0)
{
v___y_5023_ = v___y_5042_;
v___y_5024_ = v___y_5043_;
v___y_5025_ = v_severity_4917_;
goto v___jp_5022_;
}
else
{
v___y_5023_ = v___y_5042_;
v___y_5024_ = v___y_5043_;
v___y_5025_ = v___x_5040_;
goto v___jp_5022_;
}
}
v___jp_5045_:
{
if (v___y_5046_ == 0)
{
lean_object* v___x_5047_; lean_object* v_scopes_5048_; lean_object* v___x_5049_; lean_object* v___x_5050_; lean_object* v_opts_5051_; uint8_t v___x_5052_; uint8_t v___x_5053_; 
v___x_5047_ = lean_st_ref_get(v___y_4920_);
v_scopes_5048_ = lean_ctor_get(v___x_5047_, 2);
lean_inc(v_scopes_5048_);
lean_dec(v___x_5047_);
v___x_5049_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_5050_ = l_List_head_x21___redArg(v___x_5049_, v_scopes_5048_);
lean_dec(v_scopes_5048_);
v_opts_5051_ = lean_ctor_get(v___x_5050_, 1);
lean_inc_ref(v_opts_5051_);
lean_dec(v___x_5050_);
v___x_5052_ = 1;
v___x_5053_ = l_Lean_instBEqMessageSeverity_beq(v_severity_4917_, v___x_5052_);
if (v___x_5053_ == 0)
{
lean_dec_ref(v_opts_5051_);
v___y_5042_ = v___y_5046_;
v___y_5043_ = v___y_5046_;
v___y_5044_ = v___x_5053_;
goto v___jp_5041_;
}
else
{
lean_object* v___x_5054_; uint8_t v___x_5055_; 
v___x_5054_ = l_Lean_warningAsError;
v___x_5055_ = lp_batteries_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__7(v_opts_5051_, v___x_5054_);
lean_dec_ref(v_opts_5051_);
v___y_5042_ = v___y_5046_;
v___y_5043_ = v___y_5046_;
v___y_5044_ = v___x_5055_;
goto v___jp_5041_;
}
}
else
{
lean_object* v___x_5056_; lean_object* v___x_5057_; 
lean_dec_ref(v_msgData_4916_);
v___x_5056_ = lean_box(0);
v___x_5057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5057_, 0, v___x_5056_);
return v___x_5057_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4___boxed(lean_object* v_ref_5060_, lean_object* v_msgData_5061_, lean_object* v_severity_5062_, lean_object* v_isSilent_5063_, lean_object* v___y_5064_, lean_object* v___y_5065_, lean_object* v___y_5066_){
_start:
{
uint8_t v_severity_boxed_5067_; uint8_t v_isSilent_boxed_5068_; lean_object* v_res_5069_; 
v_severity_boxed_5067_ = lean_unbox(v_severity_5062_);
v_isSilent_boxed_5068_ = lean_unbox(v_isSilent_5063_);
v_res_5069_ = lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4(v_ref_5060_, v_msgData_5061_, v_severity_boxed_5067_, v_isSilent_boxed_5068_, v___y_5064_, v___y_5065_);
lean_dec(v___y_5065_);
lean_dec_ref(v___y_5064_);
lean_dec(v_ref_5060_);
return v_res_5069_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3(lean_object* v_ref_5070_, lean_object* v_msgData_5071_, lean_object* v___y_5072_, lean_object* v___y_5073_){
_start:
{
uint8_t v___x_5075_; uint8_t v___x_5076_; lean_object* v___x_5077_; 
v___x_5075_ = 0;
v___x_5076_ = 0;
v___x_5077_ = lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4(v_ref_5070_, v_msgData_5071_, v___x_5075_, v___x_5076_, v___y_5072_, v___y_5073_);
return v___x_5077_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3___boxed(lean_object* v_ref_5078_, lean_object* v_msgData_5079_, lean_object* v___y_5080_, lean_object* v___y_5081_, lean_object* v___y_5082_){
_start:
{
lean_object* v_res_5083_; 
v_res_5083_ = lp_batteries_Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3(v_ref_5078_, v_msgData_5079_, v___y_5080_, v___y_5081_);
lean_dec(v___y_5081_);
lean_dec_ref(v___y_5080_);
lean_dec(v_ref_5078_);
return v_res_5083_;
}
}
LEAN_EXPORT uint8_t lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__5(uint8_t v___x_5084_, lean_object* v_as_5085_, size_t v_i_5086_, size_t v_stop_5087_){
_start:
{
uint8_t v___x_5088_; 
v___x_5088_ = lean_usize_dec_eq(v_i_5086_, v_stop_5087_);
if (v___x_5088_ == 0)
{
lean_object* v___x_5089_; lean_object* v_snd_5090_; lean_object* v_size_5091_; uint8_t v___x_5092_; uint8_t v___y_5094_; lean_object* v___x_5098_; uint8_t v___x_5099_; 
v___x_5089_ = lean_array_uget_borrowed(v_as_5085_, v_i_5086_);
v_snd_5090_ = lean_ctor_get(v___x_5089_, 1);
v_size_5091_ = lean_ctor_get(v_snd_5090_, 0);
v___x_5092_ = 1;
v___x_5098_ = lean_unsigned_to_nat(0u);
v___x_5099_ = lean_nat_dec_eq(v_size_5091_, v___x_5098_);
if (v___x_5099_ == 0)
{
v___y_5094_ = v___x_5084_;
goto v___jp_5093_;
}
else
{
v___y_5094_ = v___x_5088_;
goto v___jp_5093_;
}
v___jp_5093_:
{
if (v___y_5094_ == 0)
{
size_t v___x_5095_; size_t v___x_5096_; 
v___x_5095_ = ((size_t)1ULL);
v___x_5096_ = lean_usize_add(v_i_5086_, v___x_5095_);
v_i_5086_ = v___x_5096_;
goto _start;
}
else
{
return v___x_5092_;
}
}
}
else
{
uint8_t v___x_5100_; 
v___x_5100_ = 0;
return v___x_5100_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__5___boxed(lean_object* v___x_5101_, lean_object* v_as_5102_, lean_object* v_i_5103_, lean_object* v_stop_5104_){
_start:
{
uint8_t v___x_10492__boxed_5105_; size_t v_i_boxed_5106_; size_t v_stop_boxed_5107_; uint8_t v_res_5108_; lean_object* v_r_5109_; 
v___x_10492__boxed_5105_ = lean_unbox(v___x_5101_);
v_i_boxed_5106_ = lean_unbox_usize(v_i_5103_);
lean_dec(v_i_5103_);
v_stop_boxed_5107_ = lean_unbox_usize(v_stop_5104_);
lean_dec(v_stop_5104_);
v_res_5108_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__5(v___x_10492__boxed_5105_, v_as_5102_, v_i_boxed_5106_, v_stop_boxed_5107_);
lean_dec_ref(v_as_5102_);
v_r_5109_ = lean_box(v_res_5108_);
return v_r_5109_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__6(size_t v_sz_5110_, size_t v_i_5111_, lean_object* v_bs_5112_){
_start:
{
uint8_t v___x_5113_; 
v___x_5113_ = lean_usize_dec_lt(v_i_5111_, v_sz_5110_);
if (v___x_5113_ == 0)
{
return v_bs_5112_;
}
else
{
lean_object* v_v_5114_; lean_object* v___x_5115_; lean_object* v_bs_x27_5116_; lean_object* v___x_5117_; size_t v___x_5118_; size_t v___x_5119_; lean_object* v___x_5120_; 
v_v_5114_ = lean_array_uget(v_bs_5112_, v_i_5111_);
v___x_5115_ = lean_unsigned_to_nat(0u);
v_bs_x27_5116_ = lean_array_uset(v_bs_5112_, v_i_5111_, v___x_5115_);
v___x_5117_ = l_Lean_TSyntax_getId(v_v_5114_);
lean_dec(v_v_5114_);
v___x_5118_ = ((size_t)1ULL);
v___x_5119_ = lean_usize_add(v_i_5111_, v___x_5118_);
v___x_5120_ = lean_array_uset(v_bs_x27_5116_, v_i_5111_, v___x_5117_);
v_i_5111_ = v___x_5119_;
v_bs_5112_ = v___x_5120_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__6___boxed(lean_object* v_sz_5122_, lean_object* v_i_5123_, lean_object* v_bs_5124_){
_start:
{
size_t v_sz_boxed_5125_; size_t v_i_boxed_5126_; lean_object* v_res_5127_; 
v_sz_boxed_5125_ = lean_unbox_usize(v_sz_5122_);
lean_dec(v_sz_5122_);
v_i_boxed_5126_ = lean_unbox_usize(v_i_5123_);
lean_dec(v_i_5123_);
v_res_5127_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__6(v_sz_boxed_5125_, v_i_boxed_5126_, v_bs_5124_);
return v_res_5127_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6(lean_object* v_msgData_5128_, uint8_t v_severity_5129_, uint8_t v_isSilent_5130_, lean_object* v___y_5131_, lean_object* v___y_5132_){
_start:
{
lean_object* v___x_5134_; 
v___x_5134_ = l_Lean_Elab_Command_getRef___redArg(v___y_5131_);
if (lean_obj_tag(v___x_5134_) == 0)
{
lean_object* v_a_5135_; lean_object* v___x_5136_; 
v_a_5135_ = lean_ctor_get(v___x_5134_, 0);
lean_inc(v_a_5135_);
lean_dec_ref_known(v___x_5134_, 1);
v___x_5136_ = lp_batteries_Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4(v_a_5135_, v_msgData_5128_, v_severity_5129_, v_isSilent_5130_, v___y_5131_, v___y_5132_);
lean_dec(v_a_5135_);
return v___x_5136_;
}
else
{
lean_object* v_a_5137_; lean_object* v___x_5139_; uint8_t v_isShared_5140_; uint8_t v_isSharedCheck_5144_; 
lean_dec_ref(v_msgData_5128_);
v_a_5137_ = lean_ctor_get(v___x_5134_, 0);
v_isSharedCheck_5144_ = !lean_is_exclusive(v___x_5134_);
if (v_isSharedCheck_5144_ == 0)
{
v___x_5139_ = v___x_5134_;
v_isShared_5140_ = v_isSharedCheck_5144_;
goto v_resetjp_5138_;
}
else
{
lean_inc(v_a_5137_);
lean_dec(v___x_5134_);
v___x_5139_ = lean_box(0);
v_isShared_5140_ = v_isSharedCheck_5144_;
goto v_resetjp_5138_;
}
v_resetjp_5138_:
{
lean_object* v___x_5142_; 
if (v_isShared_5140_ == 0)
{
v___x_5142_ = v___x_5139_;
goto v_reusejp_5141_;
}
else
{
lean_object* v_reuseFailAlloc_5143_; 
v_reuseFailAlloc_5143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5143_, 0, v_a_5137_);
v___x_5142_ = v_reuseFailAlloc_5143_;
goto v_reusejp_5141_;
}
v_reusejp_5141_:
{
return v___x_5142_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6___boxed(lean_object* v_msgData_5145_, lean_object* v_severity_5146_, lean_object* v_isSilent_5147_, lean_object* v___y_5148_, lean_object* v___y_5149_, lean_object* v___y_5150_){
_start:
{
uint8_t v_severity_boxed_5151_; uint8_t v_isSilent_boxed_5152_; lean_object* v_res_5153_; 
v_severity_boxed_5151_ = lean_unbox(v_severity_5146_);
v_isSilent_boxed_5152_ = lean_unbox(v_isSilent_5147_);
v_res_5153_ = lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6(v_msgData_5145_, v_severity_boxed_5151_, v_isSilent_boxed_5152_, v___y_5148_, v___y_5149_);
lean_dec(v___y_5149_);
lean_dec_ref(v___y_5148_);
return v_res_5153_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4(lean_object* v_msgData_5154_, lean_object* v___y_5155_, lean_object* v___y_5156_){
_start:
{
uint8_t v___x_5158_; uint8_t v___x_5159_; lean_object* v___x_5160_; 
v___x_5158_ = 2;
v___x_5159_ = 0;
v___x_5160_ = lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6(v_msgData_5154_, v___x_5158_, v___x_5159_, v___y_5155_, v___y_5156_);
return v___x_5160_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4___boxed(lean_object* v_msgData_5161_, lean_object* v___y_5162_, lean_object* v___y_5163_, lean_object* v___y_5164_){
_start:
{
lean_object* v_res_5165_; 
v_res_5165_ = lp_batteries_Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4(v_msgData_5161_, v___y_5162_, v___y_5163_);
lean_dec(v___y_5163_);
lean_dec_ref(v___y_5162_);
return v_res_5165_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__1(void){
_start:
{
lean_object* v___x_5167_; lean_object* v___x_5168_; 
v___x_5167_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__0));
v___x_5168_ = l_Lean_stringToMessageData(v___x_5167_);
return v___x_5168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1(lean_object* v_x_5175_, lean_object* v_a_5176_, lean_object* v_a_5177_){
_start:
{
lean_object* v___x_5182_; uint8_t v___x_5183_; 
v___x_5182_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__1));
lean_inc(v_x_5175_);
v___x_5183_ = l_Lean_Syntax_isOfKind(v_x_5175_, v___x_5182_);
if (v___x_5183_ == 0)
{
lean_object* v___x_5184_; 
lean_dec(v_x_5175_);
v___x_5184_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
return v___x_5184_;
}
else
{
lean_object* v___x_5185_; lean_object* v_tk_5186_; lean_object* v___y_5188_; uint8_t v___y_5189_; lean_object* v___y_5190_; uint8_t v___y_5191_; lean_object* v___y_5192_; lean_object* v___y_5193_; lean_object* v___y_5194_; uint8_t v___y_5195_; uint8_t v___y_5196_; lean_object* v___y_5197_; uint8_t v___y_5198_; lean_object* v___y_5223_; uint8_t v___y_5224_; lean_object* v___y_5225_; lean_object* v___y_5226_; lean_object* v___y_5227_; lean_object* v___y_5228_; lean_object* v___y_5229_; uint8_t v___y_5230_; uint8_t v___y_5231_; lean_object* v___y_5264_; lean_object* v___y_5265_; lean_object* v___y_5266_; uint8_t v___y_5267_; uint8_t v___y_5268_; lean_object* v___y_5269_; lean_object* v___y_5270_; uint8_t v___y_5271_; lean_object* v___y_5272_; lean_object* v___y_5275_; lean_object* v___y_5276_; uint8_t v___y_5277_; lean_object* v___y_5278_; uint8_t v___y_5279_; lean_object* v___y_5280_; uint8_t v___y_5281_; lean_object* v___y_5282_; lean_object* v___y_5285_; lean_object* v___y_5286_; uint8_t v___y_5287_; lean_object* v___y_5288_; lean_object* v___y_5289_; uint8_t v___y_5290_; lean_object* v___y_5291_; lean_object* v___y_5292_; lean_object* v___y_5293_; uint8_t v___y_5294_; lean_object* v___y_5308_; lean_object* v___y_5309_; lean_object* v___y_5310_; uint8_t v___y_5311_; lean_object* v___y_5312_; lean_object* v___y_5313_; lean_object* v___y_5314_; uint8_t v_verbosity_5315_; lean_object* v___y_5316_; lean_object* v___y_5317_; lean_object* v___y_5320_; lean_object* v___y_5321_; uint8_t v___y_5322_; lean_object* v___y_5323_; lean_object* v___y_5324_; lean_object* v___y_5325_; lean_object* v___y_5326_; lean_object* v___y_5327_; lean_object* v___y_5328_; lean_object* v___y_5339_; lean_object* v___y_5340_; lean_object* v___y_5341_; lean_object* v___y_5342_; lean_object* v___y_5343_; lean_object* v_fst_5344_; lean_object* v_fst_5345_; uint8_t v_snd_5346_; lean_object* v___y_5347_; lean_object* v___y_5348_; lean_object* v___x_5364_; lean_object* v___y_5366_; lean_object* v___y_5367_; lean_object* v___y_5368_; lean_object* v___y_5369_; lean_object* v___y_5370_; lean_object* v___y_5371_; lean_object* v___x_5428_; lean_object* v___y_5430_; lean_object* v___y_5431_; lean_object* v___y_5432_; lean_object* v___y_5433_; lean_object* v___y_5434_; lean_object* v___x_5445_; lean_object* v___x_5446_; lean_object* v___y_5448_; lean_object* v___y_5449_; lean_object* v___y_5450_; lean_object* v___x_5461_; lean_object* v___x_5462_; lean_object* v___x_5463_; lean_object* v___x_5464_; lean_object* v___y_5466_; lean_object* v___x_5478_; lean_object* v___x_5479_; lean_object* v___x_5480_; 
v___x_5185_ = lean_unsigned_to_nat(0u);
v_tk_5186_ = l_Lean_Syntax_getArg(v_x_5175_, v___x_5185_);
v___x_5364_ = lean_unsigned_to_nat(1u);
v___x_5428_ = l_Lean_Syntax_getArg(v_x_5175_, v___x_5364_);
v___x_5445_ = lean_unsigned_to_nat(2u);
v___x_5446_ = l_Lean_Syntax_getArg(v_x_5175_, v___x_5445_);
v___x_5461_ = lean_unsigned_to_nat(3u);
v___x_5462_ = l_Lean_Syntax_getArg(v_x_5175_, v___x_5461_);
v___x_5463_ = lean_unsigned_to_nat(4u);
v___x_5464_ = l_Lean_Syntax_getArg(v_x_5175_, v___x_5463_);
v___x_5478_ = lean_unsigned_to_nat(5u);
v___x_5479_ = l_Lean_Syntax_getArg(v_x_5175_, v___x_5478_);
lean_dec(v_x_5175_);
v___x_5480_ = l_Lean_Syntax_getOptional_x3f(v___x_5479_);
lean_dec(v___x_5479_);
if (lean_obj_tag(v___x_5480_) == 0)
{
lean_object* v___x_5481_; 
v___x_5481_ = lean_box(0);
v___y_5466_ = v___x_5481_;
goto v___jp_5465_;
}
else
{
lean_object* v_val_5482_; lean_object* v___x_5484_; uint8_t v_isShared_5485_; uint8_t v_isSharedCheck_5489_; 
v_val_5482_ = lean_ctor_get(v___x_5480_, 0);
v_isSharedCheck_5489_ = !lean_is_exclusive(v___x_5480_);
if (v_isSharedCheck_5489_ == 0)
{
v___x_5484_ = v___x_5480_;
v_isShared_5485_ = v_isSharedCheck_5489_;
goto v_resetjp_5483_;
}
else
{
lean_inc(v_val_5482_);
lean_dec(v___x_5480_);
v___x_5484_ = lean_box(0);
v_isShared_5485_ = v_isSharedCheck_5489_;
goto v_resetjp_5483_;
}
v_resetjp_5483_:
{
lean_object* v___x_5487_; 
if (v_isShared_5485_ == 0)
{
v___x_5487_ = v___x_5484_;
goto v_reusejp_5486_;
}
else
{
lean_object* v_reuseFailAlloc_5488_; 
v_reuseFailAlloc_5488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5488_, 0, v_val_5482_);
v___x_5487_ = v_reuseFailAlloc_5488_;
goto v_reusejp_5486_;
}
v_reusejp_5486_:
{
v___y_5466_ = v___x_5487_;
goto v___jp_5465_;
}
}
}
v___jp_5187_:
{
lean_object* v___x_5199_; lean_object* v___x_5200_; lean_object* v___x_5201_; lean_object* v___x_5202_; lean_object* v___x_5203_; lean_object* v___x_5204_; lean_object* v___x_5205_; 
v___x_5199_ = lean_array_get_size(v___y_5197_);
lean_dec_ref(v___y_5197_);
v___x_5200_ = lean_box(v___y_5195_);
v___x_5201_ = lean_box(v___y_5191_);
v___x_5202_ = lean_box(v___y_5189_);
v___x_5203_ = lean_box(v___y_5196_);
v___x_5204_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_formatLinterResults___boxed), 11, 8);
lean_closure_set(v___x_5204_, 0, v___y_5192_);
lean_closure_set(v___x_5204_, 1, v___y_5190_);
lean_closure_set(v___x_5204_, 2, v___x_5200_);
lean_closure_set(v___x_5204_, 3, v___y_5194_);
lean_closure_set(v___x_5204_, 4, v___x_5201_);
lean_closure_set(v___x_5204_, 5, v___x_5202_);
lean_closure_set(v___x_5204_, 6, v___x_5199_);
lean_closure_set(v___x_5204_, 7, v___x_5203_);
v___x_5205_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_5204_, v___y_5193_, v___y_5188_);
if (lean_obj_tag(v___x_5205_) == 0)
{
if (v___y_5198_ == 0)
{
lean_object* v_a_5206_; uint8_t v___x_5207_; uint8_t v___x_5208_; 
v_a_5206_ = lean_ctor_get(v___x_5205_, 0);
lean_inc(v_a_5206_);
lean_dec_ref_known(v___x_5205_, 1);
v___x_5207_ = 0;
v___x_5208_ = lp_batteries_Batteries_Tactic_Lint_instDecidableEqLintVerbosity(v___y_5189_, v___x_5207_);
if (v___x_5208_ == 0)
{
if (v___x_5183_ == 0)
{
lean_dec(v_a_5206_);
lean_dec(v_tk_5186_);
goto v___jp_5179_;
}
else
{
lean_object* v___x_5209_; lean_object* v___x_5210_; lean_object* v___x_5211_; 
v___x_5209_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__1, &lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__1_once, _init_lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__1);
v___x_5210_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5210_, 0, v_a_5206_);
lean_ctor_set(v___x_5210_, 1, v___x_5209_);
v___x_5211_ = lp_batteries_Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3(v_tk_5186_, v___x_5210_, v___y_5193_, v___y_5188_);
lean_dec(v_tk_5186_);
return v___x_5211_;
}
}
else
{
lean_dec(v_a_5206_);
lean_dec(v_tk_5186_);
goto v___jp_5179_;
}
}
else
{
lean_object* v_a_5212_; lean_object* v___x_5213_; 
lean_dec(v_tk_5186_);
v_a_5212_ = lean_ctor_get(v___x_5205_, 0);
lean_inc(v_a_5212_);
lean_dec_ref_known(v___x_5205_, 1);
v___x_5213_ = lp_batteries_Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4(v_a_5212_, v___y_5193_, v___y_5188_);
return v___x_5213_;
}
}
else
{
lean_object* v_a_5214_; lean_object* v___x_5216_; uint8_t v_isShared_5217_; uint8_t v_isSharedCheck_5221_; 
lean_dec(v_tk_5186_);
v_a_5214_ = lean_ctor_get(v___x_5205_, 0);
v_isSharedCheck_5221_ = !lean_is_exclusive(v___x_5205_);
if (v_isSharedCheck_5221_ == 0)
{
v___x_5216_ = v___x_5205_;
v_isShared_5217_ = v_isSharedCheck_5221_;
goto v_resetjp_5215_;
}
else
{
lean_inc(v_a_5214_);
lean_dec(v___x_5205_);
v___x_5216_ = lean_box(0);
v_isShared_5217_ = v_isSharedCheck_5221_;
goto v_resetjp_5215_;
}
v_resetjp_5215_:
{
lean_object* v___x_5219_; 
if (v_isShared_5217_ == 0)
{
v___x_5219_ = v___x_5216_;
goto v_reusejp_5218_;
}
else
{
lean_object* v_reuseFailAlloc_5220_; 
v_reuseFailAlloc_5220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5220_, 0, v_a_5214_);
v___x_5219_ = v_reuseFailAlloc_5220_;
goto v_reusejp_5218_;
}
v_reusejp_5218_:
{
return v___x_5219_;
}
}
}
}
v___jp_5222_:
{
lean_object* v___x_5232_; lean_object* v___x_5233_; lean_object* v___f_5234_; lean_object* v___x_5235_; 
v___x_5232_ = lean_box(0);
v___x_5233_ = lean_box(v___y_5231_);
v___f_5234_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___lam__0___boxed), 7, 4);
lean_closure_set(v___f_5234_, 0, v___x_5233_);
lean_closure_set(v___f_5234_, 1, v___y_5226_);
lean_closure_set(v___f_5234_, 2, v___x_5232_);
lean_closure_set(v___f_5234_, 3, v___y_5228_);
v___x_5235_ = l_Lean_Elab_Command_liftCoreM___redArg(v___f_5234_, v___y_5225_, v___y_5223_);
if (lean_obj_tag(v___x_5235_) == 0)
{
lean_object* v_a_5236_; uint8_t v___x_5237_; lean_object* v___x_5238_; lean_object* v___x_5239_; lean_object* v___x_5240_; 
v_a_5236_ = lean_ctor_get(v___x_5235_, 0);
lean_inc_n(v_a_5236_, 2);
lean_dec_ref_known(v___x_5235_, 1);
v___x_5237_ = 0;
v___x_5238_ = lean_box(v___x_5237_);
lean_inc_ref(v___y_5227_);
v___x_5239_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_lintCore___boxed), 7, 4);
lean_closure_set(v___x_5239_, 0, v___y_5227_);
lean_closure_set(v___x_5239_, 1, v_a_5236_);
lean_closure_set(v___x_5239_, 2, v___x_5232_);
lean_closure_set(v___x_5239_, 3, v___x_5238_);
v___x_5240_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_5239_, v___y_5225_, v___y_5223_);
if (lean_obj_tag(v___x_5240_) == 0)
{
lean_object* v_a_5241_; lean_object* v___x_5242_; uint8_t v___x_5243_; 
v_a_5241_ = lean_ctor_get(v___x_5240_, 0);
lean_inc(v_a_5241_);
lean_dec_ref_known(v___x_5240_, 1);
v___x_5242_ = lean_array_get_size(v_a_5241_);
v___x_5243_ = lean_nat_dec_lt(v___x_5185_, v___x_5242_);
if (v___x_5243_ == 0)
{
v___y_5188_ = v___y_5223_;
v___y_5189_ = v___y_5224_;
v___y_5190_ = v___y_5227_;
v___y_5191_ = v___y_5231_;
v___y_5192_ = v_a_5241_;
v___y_5193_ = v___y_5225_;
v___y_5194_ = v___y_5229_;
v___y_5195_ = v___y_5230_;
v___y_5196_ = v___x_5237_;
v___y_5197_ = v_a_5236_;
v___y_5198_ = v___x_5237_;
goto v___jp_5187_;
}
else
{
if (v___x_5243_ == 0)
{
v___y_5188_ = v___y_5223_;
v___y_5189_ = v___y_5224_;
v___y_5190_ = v___y_5227_;
v___y_5191_ = v___y_5231_;
v___y_5192_ = v_a_5241_;
v___y_5193_ = v___y_5225_;
v___y_5194_ = v___y_5229_;
v___y_5195_ = v___y_5230_;
v___y_5196_ = v___x_5237_;
v___y_5197_ = v_a_5236_;
v___y_5198_ = v___x_5237_;
goto v___jp_5187_;
}
else
{
size_t v___x_5244_; size_t v___x_5245_; uint8_t v___x_5246_; 
v___x_5244_ = ((size_t)0ULL);
v___x_5245_ = lean_usize_of_nat(v___x_5242_);
v___x_5246_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__5(v___x_5183_, v_a_5241_, v___x_5244_, v___x_5245_);
v___y_5188_ = v___y_5223_;
v___y_5189_ = v___y_5224_;
v___y_5190_ = v___y_5227_;
v___y_5191_ = v___y_5231_;
v___y_5192_ = v_a_5241_;
v___y_5193_ = v___y_5225_;
v___y_5194_ = v___y_5229_;
v___y_5195_ = v___y_5230_;
v___y_5196_ = v___x_5237_;
v___y_5197_ = v_a_5236_;
v___y_5198_ = v___x_5246_;
goto v___jp_5187_;
}
}
}
else
{
lean_object* v_a_5247_; lean_object* v___x_5249_; uint8_t v_isShared_5250_; uint8_t v_isSharedCheck_5254_; 
lean_dec(v_a_5236_);
lean_dec_ref(v___y_5229_);
lean_dec_ref(v___y_5227_);
lean_dec(v_tk_5186_);
v_a_5247_ = lean_ctor_get(v___x_5240_, 0);
v_isSharedCheck_5254_ = !lean_is_exclusive(v___x_5240_);
if (v_isSharedCheck_5254_ == 0)
{
v___x_5249_ = v___x_5240_;
v_isShared_5250_ = v_isSharedCheck_5254_;
goto v_resetjp_5248_;
}
else
{
lean_inc(v_a_5247_);
lean_dec(v___x_5240_);
v___x_5249_ = lean_box(0);
v_isShared_5250_ = v_isSharedCheck_5254_;
goto v_resetjp_5248_;
}
v_resetjp_5248_:
{
lean_object* v___x_5252_; 
if (v_isShared_5250_ == 0)
{
v___x_5252_ = v___x_5249_;
goto v_reusejp_5251_;
}
else
{
lean_object* v_reuseFailAlloc_5253_; 
v_reuseFailAlloc_5253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5253_, 0, v_a_5247_);
v___x_5252_ = v_reuseFailAlloc_5253_;
goto v_reusejp_5251_;
}
v_reusejp_5251_:
{
return v___x_5252_;
}
}
}
}
else
{
lean_object* v_a_5255_; lean_object* v___x_5257_; uint8_t v_isShared_5258_; uint8_t v_isSharedCheck_5262_; 
lean_dec_ref(v___y_5229_);
lean_dec_ref(v___y_5227_);
lean_dec(v_tk_5186_);
v_a_5255_ = lean_ctor_get(v___x_5235_, 0);
v_isSharedCheck_5262_ = !lean_is_exclusive(v___x_5235_);
if (v_isSharedCheck_5262_ == 0)
{
v___x_5257_ = v___x_5235_;
v_isShared_5258_ = v_isSharedCheck_5262_;
goto v_resetjp_5256_;
}
else
{
lean_inc(v_a_5255_);
lean_dec(v___x_5235_);
v___x_5257_ = lean_box(0);
v_isShared_5258_ = v_isSharedCheck_5262_;
goto v_resetjp_5256_;
}
v_resetjp_5256_:
{
lean_object* v___x_5260_; 
if (v_isShared_5258_ == 0)
{
v___x_5260_ = v___x_5257_;
goto v_reusejp_5259_;
}
else
{
lean_object* v_reuseFailAlloc_5261_; 
v_reuseFailAlloc_5261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5261_, 0, v_a_5255_);
v___x_5260_ = v_reuseFailAlloc_5261_;
goto v_reusejp_5259_;
}
v_reusejp_5259_:
{
return v___x_5260_;
}
}
}
}
v___jp_5263_:
{
if (v___y_5268_ == 0)
{
v___y_5223_ = v___y_5265_;
v___y_5224_ = v___y_5267_;
v___y_5225_ = v___y_5269_;
v___y_5226_ = v___y_5272_;
v___y_5227_ = v___y_5266_;
v___y_5228_ = v___y_5264_;
v___y_5229_ = v___y_5270_;
v___y_5230_ = v___y_5271_;
v___y_5231_ = v___x_5183_;
goto v___jp_5222_;
}
else
{
uint8_t v___x_5273_; 
v___x_5273_ = 0;
v___y_5223_ = v___y_5265_;
v___y_5224_ = v___y_5267_;
v___y_5225_ = v___y_5269_;
v___y_5226_ = v___y_5272_;
v___y_5227_ = v___y_5266_;
v___y_5228_ = v___y_5264_;
v___y_5229_ = v___y_5270_;
v___y_5230_ = v___y_5271_;
v___y_5231_ = v___x_5273_;
goto v___jp_5222_;
}
}
v___jp_5274_:
{
lean_object* v___x_5283_; 
v___x_5283_ = lean_box(0);
v___y_5264_ = v___y_5275_;
v___y_5265_ = v___y_5276_;
v___y_5266_ = v___y_5278_;
v___y_5267_ = v___y_5277_;
v___y_5268_ = v___y_5279_;
v___y_5269_ = v___y_5280_;
v___y_5270_ = v___y_5282_;
v___y_5271_ = v___y_5281_;
v___y_5272_ = v___x_5283_;
goto v___jp_5263_;
}
v___jp_5284_:
{
if (lean_obj_tag(v___y_5293_) == 0)
{
lean_dec_ref(v___y_5292_);
v___y_5275_ = v___y_5285_;
v___y_5276_ = v___y_5286_;
v___y_5277_ = v___y_5287_;
v___y_5278_ = v___y_5288_;
v___y_5279_ = v___y_5294_;
v___y_5280_ = v___y_5289_;
v___y_5281_ = v___y_5290_;
v___y_5282_ = v___y_5291_;
goto v___jp_5274_;
}
else
{
lean_object* v___x_5296_; uint8_t v_isShared_5297_; uint8_t v_isSharedCheck_5305_; 
v_isSharedCheck_5305_ = !lean_is_exclusive(v___y_5293_);
if (v_isSharedCheck_5305_ == 0)
{
lean_object* v_unused_5306_; 
v_unused_5306_ = lean_ctor_get(v___y_5293_, 0);
lean_dec(v_unused_5306_);
v___x_5296_ = v___y_5293_;
v_isShared_5297_ = v_isSharedCheck_5305_;
goto v_resetjp_5295_;
}
else
{
lean_dec(v___y_5293_);
v___x_5296_ = lean_box(0);
v_isShared_5297_ = v_isSharedCheck_5305_;
goto v_resetjp_5295_;
}
v_resetjp_5295_:
{
if (v___x_5183_ == 0)
{
lean_del_object(v___x_5296_);
lean_dec_ref(v___y_5292_);
v___y_5275_ = v___y_5285_;
v___y_5276_ = v___y_5286_;
v___y_5277_ = v___y_5287_;
v___y_5278_ = v___y_5288_;
v___y_5279_ = v___y_5294_;
v___y_5280_ = v___y_5289_;
v___y_5281_ = v___y_5290_;
v___y_5282_ = v___y_5291_;
goto v___jp_5274_;
}
else
{
size_t v_sz_5298_; size_t v___x_5299_; lean_object* v___x_5300_; lean_object* v___x_5301_; lean_object* v___x_5303_; 
v_sz_5298_ = lean_array_size(v___y_5292_);
v___x_5299_ = ((size_t)0ULL);
v___x_5300_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__6(v_sz_5298_, v___x_5299_, v___y_5292_);
v___x_5301_ = lean_array_to_list(v___x_5300_);
if (v_isShared_5297_ == 0)
{
lean_ctor_set(v___x_5296_, 0, v___x_5301_);
v___x_5303_ = v___x_5296_;
goto v_reusejp_5302_;
}
else
{
lean_object* v_reuseFailAlloc_5304_; 
v_reuseFailAlloc_5304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5304_, 0, v___x_5301_);
v___x_5303_ = v_reuseFailAlloc_5304_;
goto v_reusejp_5302_;
}
v_reusejp_5302_:
{
v___y_5264_ = v___y_5285_;
v___y_5265_ = v___y_5286_;
v___y_5266_ = v___y_5288_;
v___y_5267_ = v___y_5287_;
v___y_5268_ = v___y_5294_;
v___y_5269_ = v___y_5289_;
v___y_5270_ = v___y_5291_;
v___y_5271_ = v___y_5290_;
v___y_5272_ = v___x_5303_;
goto v___jp_5263_;
}
}
}
}
}
v___jp_5307_:
{
if (lean_obj_tag(v___y_5312_) == 0)
{
uint8_t v___x_5318_; 
v___x_5318_ = 0;
v___y_5285_ = v___y_5308_;
v___y_5286_ = v___y_5317_;
v___y_5287_ = v_verbosity_5315_;
v___y_5288_ = v___y_5309_;
v___y_5289_ = v___y_5316_;
v___y_5290_ = v___y_5311_;
v___y_5291_ = v___y_5310_;
v___y_5292_ = v___y_5313_;
v___y_5293_ = v___y_5314_;
v___y_5294_ = v___x_5318_;
goto v___jp_5284_;
}
else
{
lean_dec_ref_known(v___y_5312_, 1);
v___y_5285_ = v___y_5308_;
v___y_5286_ = v___y_5317_;
v___y_5287_ = v_verbosity_5315_;
v___y_5288_ = v___y_5309_;
v___y_5289_ = v___y_5316_;
v___y_5290_ = v___y_5311_;
v___y_5291_ = v___y_5310_;
v___y_5292_ = v___y_5313_;
v___y_5293_ = v___y_5314_;
v___y_5294_ = v___x_5183_;
goto v___jp_5284_;
}
}
v___jp_5319_:
{
lean_object* v___x_5329_; lean_object* v_a_5330_; lean_object* v___x_5332_; uint8_t v_isShared_5333_; uint8_t v_isSharedCheck_5337_; 
lean_dec(v___y_5326_);
lean_dec_ref(v___y_5325_);
lean_dec(v___y_5324_);
lean_dec_ref(v___y_5323_);
lean_dec_ref(v___y_5321_);
lean_dec_ref(v___y_5320_);
v___x_5329_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
v_a_5330_ = lean_ctor_get(v___x_5329_, 0);
v_isSharedCheck_5337_ = !lean_is_exclusive(v___x_5329_);
if (v_isSharedCheck_5337_ == 0)
{
v___x_5332_ = v___x_5329_;
v_isShared_5333_ = v_isSharedCheck_5337_;
goto v_resetjp_5331_;
}
else
{
lean_inc(v_a_5330_);
lean_dec(v___x_5329_);
v___x_5332_ = lean_box(0);
v_isShared_5333_ = v_isSharedCheck_5337_;
goto v_resetjp_5331_;
}
v_resetjp_5331_:
{
lean_object* v___x_5335_; 
if (v_isShared_5333_ == 0)
{
v___x_5335_ = v___x_5332_;
goto v_reusejp_5334_;
}
else
{
lean_object* v_reuseFailAlloc_5336_; 
v_reuseFailAlloc_5336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5336_, 0, v_a_5330_);
v___x_5335_ = v_reuseFailAlloc_5336_;
goto v_reusejp_5334_;
}
v_reusejp_5334_:
{
return v___x_5335_;
}
}
}
v___jp_5338_:
{
if (lean_obj_tag(v___y_5340_) == 0)
{
uint8_t v___x_5349_; 
v___x_5349_ = 1;
v___y_5308_ = v___y_5339_;
v___y_5309_ = v_fst_5344_;
v___y_5310_ = v_fst_5345_;
v___y_5311_ = v_snd_5346_;
v___y_5312_ = v___y_5341_;
v___y_5313_ = v___y_5342_;
v___y_5314_ = v___y_5343_;
v_verbosity_5315_ = v___x_5349_;
v___y_5316_ = v___y_5347_;
v___y_5317_ = v___y_5348_;
goto v___jp_5307_;
}
else
{
lean_object* v_val_5350_; 
v_val_5350_ = lean_ctor_get(v___y_5340_, 0);
lean_inc(v_val_5350_);
lean_dec_ref_known(v___y_5340_, 1);
if (lean_obj_tag(v_val_5350_) == 1)
{
lean_object* v_kind_5351_; 
v_kind_5351_ = lean_ctor_get(v_val_5350_, 1);
lean_inc(v_kind_5351_);
lean_dec_ref_known(v_val_5350_, 3);
if (lean_obj_tag(v_kind_5351_) == 1)
{
lean_object* v_pre_5352_; 
v_pre_5352_ = lean_ctor_get(v_kind_5351_, 0);
lean_inc(v_pre_5352_);
if (lean_obj_tag(v_pre_5352_) == 1)
{
lean_object* v_pre_5353_; 
v_pre_5353_ = lean_ctor_get(v_pre_5352_, 0);
if (lean_obj_tag(v_pre_5353_) == 0)
{
lean_object* v_str_5354_; lean_object* v_str_5355_; lean_object* v___x_5356_; uint8_t v___x_5357_; 
v_str_5354_ = lean_ctor_get(v_kind_5351_, 1);
lean_inc_ref(v_str_5354_);
lean_dec_ref_known(v_kind_5351_, 2);
v_str_5355_ = lean_ctor_get(v_pre_5352_, 1);
lean_inc_ref(v_str_5355_);
lean_dec_ref_known(v_pre_5352_, 2);
v___x_5356_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__9));
v___x_5357_ = lean_string_dec_eq(v_str_5355_, v___x_5356_);
lean_dec_ref(v_str_5355_);
if (v___x_5357_ == 0)
{
lean_dec_ref(v_str_5354_);
lean_dec(v_tk_5186_);
v___y_5320_ = v___y_5339_;
v___y_5321_ = v_fst_5344_;
v___y_5322_ = v_snd_5346_;
v___y_5323_ = v_fst_5345_;
v___y_5324_ = v___y_5341_;
v___y_5325_ = v___y_5342_;
v___y_5326_ = v___y_5343_;
v___y_5327_ = v___y_5347_;
v___y_5328_ = v___y_5348_;
goto v___jp_5319_;
}
else
{
lean_object* v___x_5358_; uint8_t v___x_5359_; 
v___x_5358_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__8));
v___x_5359_ = lean_string_dec_eq(v_str_5354_, v___x_5358_);
if (v___x_5359_ == 0)
{
lean_object* v___x_5360_; uint8_t v___x_5361_; 
v___x_5360_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_command_x23lint_x2b_x2d_x2aOnly_______00__closed__13));
v___x_5361_ = lean_string_dec_eq(v_str_5354_, v___x_5360_);
lean_dec_ref(v_str_5354_);
if (v___x_5361_ == 0)
{
lean_dec(v_tk_5186_);
v___y_5320_ = v___y_5339_;
v___y_5321_ = v_fst_5344_;
v___y_5322_ = v_snd_5346_;
v___y_5323_ = v_fst_5345_;
v___y_5324_ = v___y_5341_;
v___y_5325_ = v___y_5342_;
v___y_5326_ = v___y_5343_;
v___y_5327_ = v___y_5347_;
v___y_5328_ = v___y_5348_;
goto v___jp_5319_;
}
else
{
uint8_t v___x_5362_; 
v___x_5362_ = 0;
v___y_5308_ = v___y_5339_;
v___y_5309_ = v_fst_5344_;
v___y_5310_ = v_fst_5345_;
v___y_5311_ = v_snd_5346_;
v___y_5312_ = v___y_5341_;
v___y_5313_ = v___y_5342_;
v___y_5314_ = v___y_5343_;
v_verbosity_5315_ = v___x_5362_;
v___y_5316_ = v___y_5347_;
v___y_5317_ = v___y_5348_;
goto v___jp_5307_;
}
}
else
{
uint8_t v___x_5363_; 
lean_dec_ref(v_str_5354_);
v___x_5363_ = 2;
v___y_5308_ = v___y_5339_;
v___y_5309_ = v_fst_5344_;
v___y_5310_ = v_fst_5345_;
v___y_5311_ = v_snd_5346_;
v___y_5312_ = v___y_5341_;
v___y_5313_ = v___y_5342_;
v___y_5314_ = v___y_5343_;
v_verbosity_5315_ = v___x_5363_;
v___y_5316_ = v___y_5347_;
v___y_5317_ = v___y_5348_;
goto v___jp_5307_;
}
}
}
else
{
lean_dec_ref_known(v_pre_5352_, 2);
lean_dec_ref_known(v_kind_5351_, 2);
lean_dec(v_tk_5186_);
v___y_5320_ = v___y_5339_;
v___y_5321_ = v_fst_5344_;
v___y_5322_ = v_snd_5346_;
v___y_5323_ = v_fst_5345_;
v___y_5324_ = v___y_5341_;
v___y_5325_ = v___y_5342_;
v___y_5326_ = v___y_5343_;
v___y_5327_ = v___y_5347_;
v___y_5328_ = v___y_5348_;
goto v___jp_5319_;
}
}
else
{
lean_dec(v_pre_5352_);
lean_dec_ref_known(v_kind_5351_, 2);
lean_dec(v_tk_5186_);
v___y_5320_ = v___y_5339_;
v___y_5321_ = v_fst_5344_;
v___y_5322_ = v_snd_5346_;
v___y_5323_ = v_fst_5345_;
v___y_5324_ = v___y_5341_;
v___y_5325_ = v___y_5342_;
v___y_5326_ = v___y_5343_;
v___y_5327_ = v___y_5347_;
v___y_5328_ = v___y_5348_;
goto v___jp_5319_;
}
}
else
{
lean_dec(v_kind_5351_);
lean_dec(v_tk_5186_);
v___y_5320_ = v___y_5339_;
v___y_5321_ = v_fst_5344_;
v___y_5322_ = v_snd_5346_;
v___y_5323_ = v_fst_5345_;
v___y_5324_ = v___y_5341_;
v___y_5325_ = v___y_5342_;
v___y_5326_ = v___y_5343_;
v___y_5327_ = v___y_5347_;
v___y_5328_ = v___y_5348_;
goto v___jp_5319_;
}
}
else
{
lean_dec(v_val_5350_);
lean_dec(v_tk_5186_);
v___y_5320_ = v___y_5339_;
v___y_5321_ = v_fst_5344_;
v___y_5322_ = v_snd_5346_;
v___y_5323_ = v_fst_5345_;
v___y_5324_ = v___y_5341_;
v___y_5325_ = v___y_5342_;
v___y_5326_ = v___y_5343_;
v___y_5327_ = v___y_5347_;
v___y_5328_ = v___y_5348_;
goto v___jp_5319_;
}
}
}
v___jp_5365_:
{
if (lean_obj_tag(v___y_5367_) == 0)
{
lean_object* v___x_5372_; lean_object* v___x_5373_; 
v___x_5372_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_getDeclsInCurrModule___boxed), 3, 0);
v___x_5373_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_5372_, v_a_5176_, v_a_5177_);
if (lean_obj_tag(v___x_5373_) == 0)
{
lean_object* v_a_5374_; lean_object* v___x_5375_; uint8_t v___x_5376_; 
v_a_5374_ = lean_ctor_get(v___x_5373_, 0);
lean_inc(v_a_5374_);
lean_dec_ref_known(v___x_5373_, 1);
v___x_5375_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__2));
v___x_5376_ = 0;
v___y_5339_ = v___y_5366_;
v___y_5340_ = v___y_5371_;
v___y_5341_ = v___y_5368_;
v___y_5342_ = v___y_5369_;
v___y_5343_ = v___y_5370_;
v_fst_5344_ = v_a_5374_;
v_fst_5345_ = v___x_5375_;
v_snd_5346_ = v___x_5376_;
v___y_5347_ = v_a_5176_;
v___y_5348_ = v_a_5177_;
goto v___jp_5338_;
}
else
{
lean_object* v_a_5377_; lean_object* v___x_5379_; uint8_t v_isShared_5380_; uint8_t v_isSharedCheck_5384_; 
lean_dec(v___y_5371_);
lean_dec(v___y_5370_);
lean_dec_ref(v___y_5369_);
lean_dec(v___y_5368_);
lean_dec_ref(v___y_5366_);
lean_dec(v_tk_5186_);
v_a_5377_ = lean_ctor_get(v___x_5373_, 0);
v_isSharedCheck_5384_ = !lean_is_exclusive(v___x_5373_);
if (v_isSharedCheck_5384_ == 0)
{
v___x_5379_ = v___x_5373_;
v_isShared_5380_ = v_isSharedCheck_5384_;
goto v_resetjp_5378_;
}
else
{
lean_inc(v_a_5377_);
lean_dec(v___x_5373_);
v___x_5379_ = lean_box(0);
v_isShared_5380_ = v_isSharedCheck_5384_;
goto v_resetjp_5378_;
}
v_resetjp_5378_:
{
lean_object* v___x_5382_; 
if (v_isShared_5380_ == 0)
{
v___x_5382_ = v___x_5379_;
goto v_reusejp_5381_;
}
else
{
lean_object* v_reuseFailAlloc_5383_; 
v_reuseFailAlloc_5383_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5383_, 0, v_a_5377_);
v___x_5382_ = v_reuseFailAlloc_5383_;
goto v_reusejp_5381_;
}
v_reusejp_5381_:
{
return v___x_5382_;
}
}
}
}
else
{
lean_object* v_val_5385_; lean_object* v___x_5386_; uint8_t v___x_5387_; 
v_val_5385_ = lean_ctor_get(v___y_5367_, 0);
lean_inc_n(v_val_5385_, 2);
lean_dec_ref_known(v___y_5367_, 1);
v___x_5386_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_inProject___closed__2));
v___x_5387_ = l_Lean_Syntax_isOfKind(v_val_5385_, v___x_5386_);
if (v___x_5387_ == 0)
{
lean_object* v___x_5388_; lean_object* v_a_5389_; lean_object* v___x_5391_; uint8_t v_isShared_5392_; uint8_t v_isSharedCheck_5396_; 
lean_dec(v_val_5385_);
lean_dec(v___y_5371_);
lean_dec(v___y_5370_);
lean_dec_ref(v___y_5369_);
lean_dec(v___y_5368_);
lean_dec_ref(v___y_5366_);
lean_dec(v_tk_5186_);
v___x_5388_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
v_a_5389_ = lean_ctor_get(v___x_5388_, 0);
v_isSharedCheck_5396_ = !lean_is_exclusive(v___x_5388_);
if (v_isSharedCheck_5396_ == 0)
{
v___x_5391_ = v___x_5388_;
v_isShared_5392_ = v_isSharedCheck_5396_;
goto v_resetjp_5390_;
}
else
{
lean_inc(v_a_5389_);
lean_dec(v___x_5388_);
v___x_5391_ = lean_box(0);
v_isShared_5392_ = v_isSharedCheck_5396_;
goto v_resetjp_5390_;
}
v_resetjp_5390_:
{
lean_object* v___x_5394_; 
if (v_isShared_5392_ == 0)
{
v___x_5394_ = v___x_5391_;
goto v_reusejp_5393_;
}
else
{
lean_object* v_reuseFailAlloc_5395_; 
v_reuseFailAlloc_5395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5395_, 0, v_a_5389_);
v___x_5394_ = v_reuseFailAlloc_5395_;
goto v_reusejp_5393_;
}
v_reusejp_5393_:
{
return v___x_5394_;
}
}
}
else
{
lean_object* v_id_5397_; lean_object* v___x_5398_; lean_object* v_id_5399_; lean_object* v___x_5400_; uint8_t v___x_5401_; 
v_id_5397_ = l_Lean_Syntax_getArg(v_val_5385_, v___x_5364_);
lean_dec(v_val_5385_);
v___x_5398_ = l_Lean_TSyntax_getId(v_id_5397_);
lean_dec(v_id_5397_);
v_id_5399_ = l_Lean_Name_eraseMacroScopes(v___x_5398_);
lean_dec(v___x_5398_);
v___x_5400_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__4));
v___x_5401_ = lean_name_eq(v_id_5399_, v___x_5400_);
if (v___x_5401_ == 0)
{
lean_object* v___x_5402_; lean_object* v___x_5403_; 
lean_inc(v_id_5399_);
v___x_5402_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_getDeclsInPackage___boxed), 4, 1);
lean_closure_set(v___x_5402_, 0, v_id_5399_);
v___x_5403_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_5402_, v_a_5176_, v_a_5177_);
if (lean_obj_tag(v___x_5403_) == 0)
{
lean_object* v_a_5404_; lean_object* v___x_5405_; lean_object* v___x_5406_; lean_object* v___x_5407_; 
v_a_5404_ = lean_ctor_get(v___x_5403_, 0);
lean_inc(v_a_5404_);
lean_dec_ref_known(v___x_5403_, 1);
v___x_5405_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__5));
v___x_5406_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_id_5399_, v___x_5387_);
v___x_5407_ = lean_string_append(v___x_5405_, v___x_5406_);
lean_dec_ref(v___x_5406_);
v___y_5339_ = v___y_5366_;
v___y_5340_ = v___y_5371_;
v___y_5341_ = v___y_5368_;
v___y_5342_ = v___y_5369_;
v___y_5343_ = v___y_5370_;
v_fst_5344_ = v_a_5404_;
v_fst_5345_ = v___x_5407_;
v_snd_5346_ = v___x_5183_;
v___y_5347_ = v_a_5176_;
v___y_5348_ = v_a_5177_;
goto v___jp_5338_;
}
else
{
lean_object* v_a_5408_; lean_object* v___x_5410_; uint8_t v_isShared_5411_; uint8_t v_isSharedCheck_5415_; 
lean_dec(v_id_5399_);
lean_dec(v___y_5371_);
lean_dec(v___y_5370_);
lean_dec_ref(v___y_5369_);
lean_dec(v___y_5368_);
lean_dec_ref(v___y_5366_);
lean_dec(v_tk_5186_);
v_a_5408_ = lean_ctor_get(v___x_5403_, 0);
v_isSharedCheck_5415_ = !lean_is_exclusive(v___x_5403_);
if (v_isSharedCheck_5415_ == 0)
{
v___x_5410_ = v___x_5403_;
v_isShared_5411_ = v_isSharedCheck_5415_;
goto v_resetjp_5409_;
}
else
{
lean_inc(v_a_5408_);
lean_dec(v___x_5403_);
v___x_5410_ = lean_box(0);
v_isShared_5411_ = v_isSharedCheck_5415_;
goto v_resetjp_5409_;
}
v_resetjp_5409_:
{
lean_object* v___x_5413_; 
if (v_isShared_5411_ == 0)
{
v___x_5413_ = v___x_5410_;
goto v_reusejp_5412_;
}
else
{
lean_object* v_reuseFailAlloc_5414_; 
v_reuseFailAlloc_5414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5414_, 0, v_a_5408_);
v___x_5413_ = v_reuseFailAlloc_5414_;
goto v_reusejp_5412_;
}
v_reusejp_5412_:
{
return v___x_5413_;
}
}
}
}
else
{
lean_object* v___x_5416_; lean_object* v___x_5417_; 
lean_dec(v_id_5399_);
v___x_5416_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Lint_getAllDecls___boxed), 3, 0);
v___x_5417_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_5416_, v_a_5176_, v_a_5177_);
if (lean_obj_tag(v___x_5417_) == 0)
{
lean_object* v_a_5418_; lean_object* v___x_5419_; 
v_a_5418_ = lean_ctor_get(v___x_5417_, 0);
lean_inc(v_a_5418_);
lean_dec_ref_known(v___x_5417_, 1);
v___x_5419_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___closed__6));
v___y_5339_ = v___y_5366_;
v___y_5340_ = v___y_5371_;
v___y_5341_ = v___y_5368_;
v___y_5342_ = v___y_5369_;
v___y_5343_ = v___y_5370_;
v_fst_5344_ = v_a_5418_;
v_fst_5345_ = v___x_5419_;
v_snd_5346_ = v___x_5183_;
v___y_5347_ = v_a_5176_;
v___y_5348_ = v_a_5177_;
goto v___jp_5338_;
}
else
{
lean_object* v_a_5420_; lean_object* v___x_5422_; uint8_t v_isShared_5423_; uint8_t v_isSharedCheck_5427_; 
lean_dec(v___y_5371_);
lean_dec(v___y_5370_);
lean_dec_ref(v___y_5369_);
lean_dec(v___y_5368_);
lean_dec_ref(v___y_5366_);
lean_dec(v_tk_5186_);
v_a_5420_ = lean_ctor_get(v___x_5417_, 0);
v_isSharedCheck_5427_ = !lean_is_exclusive(v___x_5417_);
if (v_isSharedCheck_5427_ == 0)
{
v___x_5422_ = v___x_5417_;
v_isShared_5423_ = v_isSharedCheck_5427_;
goto v_resetjp_5421_;
}
else
{
lean_inc(v_a_5420_);
lean_dec(v___x_5417_);
v___x_5422_ = lean_box(0);
v_isShared_5423_ = v_isSharedCheck_5427_;
goto v_resetjp_5421_;
}
v_resetjp_5421_:
{
lean_object* v___x_5425_; 
if (v_isShared_5423_ == 0)
{
v___x_5425_ = v___x_5422_;
goto v_reusejp_5424_;
}
else
{
lean_object* v_reuseFailAlloc_5426_; 
v_reuseFailAlloc_5426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5426_, 0, v_a_5420_);
v___x_5425_ = v_reuseFailAlloc_5426_;
goto v_reusejp_5424_;
}
v_reusejp_5424_:
{
return v___x_5425_;
}
}
}
}
}
}
}
v___jp_5429_:
{
lean_object* v___x_5435_; 
v___x_5435_ = l_Lean_Syntax_getOptional_x3f(v___x_5428_);
lean_dec(v___x_5428_);
if (lean_obj_tag(v___x_5435_) == 0)
{
lean_object* v___x_5436_; 
v___x_5436_ = lean_box(0);
v___y_5366_ = v___y_5430_;
v___y_5367_ = v___y_5431_;
v___y_5368_ = v___y_5434_;
v___y_5369_ = v___y_5432_;
v___y_5370_ = v___y_5433_;
v___y_5371_ = v___x_5436_;
goto v___jp_5365_;
}
else
{
lean_object* v_val_5437_; lean_object* v___x_5439_; uint8_t v_isShared_5440_; uint8_t v_isSharedCheck_5444_; 
v_val_5437_ = lean_ctor_get(v___x_5435_, 0);
v_isSharedCheck_5444_ = !lean_is_exclusive(v___x_5435_);
if (v_isSharedCheck_5444_ == 0)
{
v___x_5439_ = v___x_5435_;
v_isShared_5440_ = v_isSharedCheck_5444_;
goto v_resetjp_5438_;
}
else
{
lean_inc(v_val_5437_);
lean_dec(v___x_5435_);
v___x_5439_ = lean_box(0);
v_isShared_5440_ = v_isSharedCheck_5444_;
goto v_resetjp_5438_;
}
v_resetjp_5438_:
{
lean_object* v___x_5442_; 
if (v_isShared_5440_ == 0)
{
v___x_5442_ = v___x_5439_;
goto v_reusejp_5441_;
}
else
{
lean_object* v_reuseFailAlloc_5443_; 
v_reuseFailAlloc_5443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5443_, 0, v_val_5437_);
v___x_5442_ = v_reuseFailAlloc_5443_;
goto v_reusejp_5441_;
}
v_reusejp_5441_:
{
v___y_5366_ = v___y_5430_;
v___y_5367_ = v___y_5431_;
v___y_5368_ = v___y_5434_;
v___y_5369_ = v___y_5432_;
v___y_5370_ = v___y_5433_;
v___y_5371_ = v___x_5442_;
goto v___jp_5365_;
}
}
}
}
v___jp_5447_:
{
lean_object* v___x_5451_; 
v___x_5451_ = l_Lean_Syntax_getOptional_x3f(v___x_5446_);
lean_dec(v___x_5446_);
if (lean_obj_tag(v___x_5451_) == 0)
{
lean_object* v___x_5452_; 
v___x_5452_ = lean_box(0);
lean_inc_ref(v___y_5449_);
v___y_5430_ = v___y_5449_;
v___y_5431_ = v___y_5448_;
v___y_5432_ = v___y_5449_;
v___y_5433_ = v___y_5450_;
v___y_5434_ = v___x_5452_;
goto v___jp_5429_;
}
else
{
lean_object* v_val_5453_; lean_object* v___x_5455_; uint8_t v_isShared_5456_; uint8_t v_isSharedCheck_5460_; 
v_val_5453_ = lean_ctor_get(v___x_5451_, 0);
v_isSharedCheck_5460_ = !lean_is_exclusive(v___x_5451_);
if (v_isSharedCheck_5460_ == 0)
{
v___x_5455_ = v___x_5451_;
v_isShared_5456_ = v_isSharedCheck_5460_;
goto v_resetjp_5454_;
}
else
{
lean_inc(v_val_5453_);
lean_dec(v___x_5451_);
v___x_5455_ = lean_box(0);
v_isShared_5456_ = v_isSharedCheck_5460_;
goto v_resetjp_5454_;
}
v_resetjp_5454_:
{
lean_object* v___x_5458_; 
if (v_isShared_5456_ == 0)
{
v___x_5458_ = v___x_5455_;
goto v_reusejp_5457_;
}
else
{
lean_object* v_reuseFailAlloc_5459_; 
v_reuseFailAlloc_5459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5459_, 0, v_val_5453_);
v___x_5458_ = v_reuseFailAlloc_5459_;
goto v_reusejp_5457_;
}
v_reusejp_5457_:
{
lean_inc_ref(v___y_5449_);
v___y_5430_ = v___y_5449_;
v___y_5431_ = v___y_5448_;
v___y_5432_ = v___y_5449_;
v___y_5433_ = v___y_5450_;
v___y_5434_ = v___x_5458_;
goto v___jp_5429_;
}
}
}
}
v___jp_5465_:
{
lean_object* v_linters_5467_; lean_object* v___x_5468_; 
v_linters_5467_ = l_Lean_Syntax_getArgs(v___x_5464_);
lean_dec(v___x_5464_);
v___x_5468_ = l_Lean_Syntax_getOptional_x3f(v___x_5462_);
lean_dec(v___x_5462_);
if (lean_obj_tag(v___x_5468_) == 0)
{
lean_object* v___x_5469_; 
v___x_5469_ = lean_box(0);
v___y_5448_ = v___y_5466_;
v___y_5449_ = v_linters_5467_;
v___y_5450_ = v___x_5469_;
goto v___jp_5447_;
}
else
{
lean_object* v_val_5470_; lean_object* v___x_5472_; uint8_t v_isShared_5473_; uint8_t v_isSharedCheck_5477_; 
v_val_5470_ = lean_ctor_get(v___x_5468_, 0);
v_isSharedCheck_5477_ = !lean_is_exclusive(v___x_5468_);
if (v_isSharedCheck_5477_ == 0)
{
v___x_5472_ = v___x_5468_;
v_isShared_5473_ = v_isSharedCheck_5477_;
goto v_resetjp_5471_;
}
else
{
lean_inc(v_val_5470_);
lean_dec(v___x_5468_);
v___x_5472_ = lean_box(0);
v_isShared_5473_ = v_isSharedCheck_5477_;
goto v_resetjp_5471_;
}
v_resetjp_5471_:
{
lean_object* v___x_5475_; 
if (v_isShared_5473_ == 0)
{
v___x_5475_ = v___x_5472_;
goto v_reusejp_5474_;
}
else
{
lean_object* v_reuseFailAlloc_5476_; 
v_reuseFailAlloc_5476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5476_, 0, v_val_5470_);
v___x_5475_ = v_reuseFailAlloc_5476_;
goto v_reusejp_5474_;
}
v_reusejp_5474_:
{
v___y_5448_ = v___y_5466_;
v___y_5449_ = v_linters_5467_;
v___y_5450_ = v___x_5475_;
goto v___jp_5447_;
}
}
}
}
}
v___jp_5179_:
{
lean_object* v___x_5180_; lean_object* v___x_5181_; 
v___x_5180_ = lean_box(0);
v___x_5181_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5181_, 0, v___x_5180_);
return v___x_5181_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1___boxed(lean_object* v_x_5490_, lean_object* v_a_5491_, lean_object* v_a_5492_, lean_object* v_a_5493_){
_start:
{
lean_object* v_res_5494_; 
v_res_5494_ = lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1(v_x_5490_, v_a_5491_, v_a_5492_);
lean_dec(v_a_5492_);
lean_dec_ref(v_a_5491_);
return v_res_5494_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2(lean_object* v_t_5495_, lean_object* v___y_5496_, lean_object* v___y_5497_){
_start:
{
lean_object* v___x_5499_; 
v___x_5499_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___redArg(v_t_5495_, v___y_5497_);
return v___x_5499_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2___boxed(lean_object* v_t_5500_, lean_object* v___y_5501_, lean_object* v___y_5502_, lean_object* v___y_5503_){
_start:
{
lean_object* v_res_5504_; 
v_res_5504_ = lp_batteries_Lean_Elab_pushInfoTree___at___00Lean_Elab_pushInfoLeaf___at___00Lean_Elab_addConstInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__1_spec__1_spec__2(v_t_5500_, v___y_5501_, v___y_5502_);
lean_dec(v___y_5502_);
lean_dec_ref(v___y_5501_);
return v_res_5504_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6(lean_object* v_msgData_5505_, lean_object* v___y_5506_, lean_object* v___y_5507_){
_start:
{
lean_object* v___x_5509_; 
v___x_5509_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___redArg(v_msgData_5505_, v___y_5507_);
return v___x_5509_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6___boxed(lean_object* v_msgData_5510_, lean_object* v___y_5511_, lean_object* v___y_5512_, lean_object* v___y_5513_){
_start:
{
lean_object* v_res_5514_; 
v_res_5514_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__3_spec__4_spec__6(v_msgData_5510_, v___y_5511_, v___y_5512_);
lean_dec(v___y_5512_);
lean_dec_ref(v___y_5511_);
return v_res_5514_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg(lean_object* v_as_5530_, size_t v_sz_5531_, size_t v_i_5532_, lean_object* v_b_5533_){
_start:
{
uint8_t v___x_5535_; 
v___x_5535_ = lean_usize_dec_lt(v_i_5532_, v_sz_5531_);
if (v___x_5535_ == 0)
{
lean_object* v___x_5536_; 
v___x_5536_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5536_, 0, v_b_5533_);
return v___x_5536_;
}
else
{
lean_object* v_a_5537_; lean_object* v_fst_5538_; lean_object* v_snd_5539_; lean_object* v___x_5541_; uint8_t v_isShared_5542_; uint8_t v_isSharedCheck_5559_; 
v_a_5537_ = lean_array_uget(v_as_5530_, v_i_5532_);
v_fst_5538_ = lean_ctor_get(v_a_5537_, 0);
v_snd_5539_ = lean_ctor_get(v_a_5537_, 1);
v_isSharedCheck_5559_ = !lean_is_exclusive(v_a_5537_);
if (v_isSharedCheck_5559_ == 0)
{
v___x_5541_ = v_a_5537_;
v_isShared_5542_ = v_isSharedCheck_5559_;
goto v_resetjp_5540_;
}
else
{
lean_inc(v_snd_5539_);
lean_inc(v_fst_5538_);
lean_dec(v_a_5537_);
v___x_5541_ = lean_box(0);
v_isShared_5542_ = v_isSharedCheck_5559_;
goto v_resetjp_5540_;
}
v_resetjp_5540_:
{
lean_object* v___x_5543_; lean_object* v___x_5544_; lean_object* v___x_5546_; 
v___x_5543_ = lean_obj_once(&lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3, &lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3_once, _init_lp_batteries_List_mapM_loop___at___00Batteries_Tactic_Lint_groupedByFilename_spec__0___closed__3);
v___x_5544_ = l_Lean_MessageData_ofName(v_fst_5538_);
if (v_isShared_5542_ == 0)
{
lean_ctor_set_tag(v___x_5541_, 7);
lean_ctor_set(v___x_5541_, 1, v___x_5544_);
lean_ctor_set(v___x_5541_, 0, v___x_5543_);
v___x_5546_ = v___x_5541_;
goto v_reusejp_5545_;
}
else
{
lean_object* v_reuseFailAlloc_5558_; 
v_reuseFailAlloc_5558_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5558_, 0, v___x_5543_);
lean_ctor_set(v_reuseFailAlloc_5558_, 1, v___x_5544_);
v___x_5546_ = v_reuseFailAlloc_5558_;
goto v_reusejp_5545_;
}
v_reusejp_5545_:
{
lean_object* v___y_5548_; uint8_t v___x_5555_; 
v___x_5555_ = lean_unbox(v_snd_5539_);
lean_dec(v_snd_5539_);
if (v___x_5555_ == 0)
{
lean_object* v___x_5556_; 
v___x_5556_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic_Lint_lintCore_spec__2___closed__1));
v___y_5548_ = v___x_5556_;
goto v___jp_5547_;
}
else
{
lean_object* v___x_5557_; 
v___x_5557_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg___closed__0));
v___y_5548_ = v___x_5557_;
goto v___jp_5547_;
}
v___jp_5547_:
{
lean_object* v___x_5549_; lean_object* v___x_5550_; lean_object* v___x_5551_; size_t v___x_5552_; size_t v___x_5553_; 
lean_inc_ref(v___y_5548_);
v___x_5549_ = l_Lean_stringToMessageData(v___y_5548_);
v___x_5550_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5550_, 0, v___x_5546_);
lean_ctor_set(v___x_5550_, 1, v___x_5549_);
v___x_5551_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5551_, 0, v_b_5533_);
lean_ctor_set(v___x_5551_, 1, v___x_5550_);
v___x_5552_ = ((size_t)1ULL);
v___x_5553_ = lean_usize_add(v_i_5532_, v___x_5552_);
v_i_5532_ = v___x_5553_;
v_b_5533_ = v___x_5551_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg___boxed(lean_object* v_as_5560_, lean_object* v_sz_5561_, lean_object* v_i_5562_, lean_object* v_b_5563_, lean_object* v___y_5564_){
_start:
{
size_t v_sz_boxed_5565_; size_t v_i_boxed_5566_; lean_object* v_res_5567_; 
v_sz_boxed_5565_ = lean_unbox_usize(v_sz_5561_);
lean_dec(v_sz_5561_);
v_i_boxed_5566_ = lean_unbox_usize(v_i_5562_);
lean_dec(v_i_5562_);
v_res_5567_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg(v_as_5560_, v_sz_boxed_5565_, v_i_boxed_5566_, v_b_5563_);
lean_dec_ref(v_as_5560_);
return v_res_5567_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(lean_object* v_x1_5568_, lean_object* v_x2_5569_){
_start:
{
lean_object* v_fst_5570_; lean_object* v_fst_5571_; uint8_t v___x_5572_; 
v_fst_5570_ = lean_ctor_get(v_x1_5568_, 0);
v_fst_5571_ = lean_ctor_get(v_x2_5569_, 0);
v___x_5572_ = l_Lean_Name_lt(v_fst_5570_, v_fst_5571_);
return v___x_5572_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0___boxed(lean_object* v_x1_5573_, lean_object* v_x2_5574_){
_start:
{
uint8_t v_res_5575_; lean_object* v_r_5576_; 
v_res_5575_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v_x1_5573_, v_x2_5574_);
lean_dec_ref(v_x2_5574_);
lean_dec_ref(v_x1_5573_);
v_r_5576_ = lean_box(v_res_5575_);
return v_r_5576_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg(lean_object* v___x_5577_, lean_object* v_as_5578_, lean_object* v_k_5579_, lean_object* v_x_5580_, lean_object* v_x_5581_){
_start:
{
lean_object* v___x_5582_; lean_object* v___x_5583_; lean_object* v_mid_5584_; lean_object* v_midVal_5585_; uint8_t v___x_5586_; 
v___x_5582_ = lean_nat_add(v_x_5580_, v_x_5581_);
v___x_5583_ = lean_unsigned_to_nat(1u);
v_mid_5584_ = lean_nat_shiftr(v___x_5582_, v___x_5583_);
lean_dec(v___x_5582_);
v_midVal_5585_ = lean_array_fget_borrowed(v_as_5578_, v_mid_5584_);
v___x_5586_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v_midVal_5585_, v_k_5579_);
if (v___x_5586_ == 0)
{
uint8_t v___x_5587_; 
lean_dec(v_x_5581_);
v___x_5587_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v_k_5579_, v_midVal_5585_);
if (v___x_5587_ == 0)
{
lean_object* v___x_5588_; uint8_t v___x_5589_; 
lean_dec(v_x_5580_);
v___x_5588_ = lean_array_get_size(v_as_5578_);
v___x_5589_ = lean_nat_dec_lt(v_mid_5584_, v___x_5588_);
if (v___x_5589_ == 0)
{
lean_dec(v_mid_5584_);
lean_dec_ref(v___x_5577_);
return v_as_5578_;
}
else
{
lean_object* v___x_5590_; lean_object* v_xs_x27_5591_; lean_object* v___x_5592_; 
v___x_5590_ = lean_box(0);
v_xs_x27_5591_ = lean_array_fset(v_as_5578_, v_mid_5584_, v___x_5590_);
v___x_5592_ = lean_array_fset(v_xs_x27_5591_, v_mid_5584_, v___x_5577_);
lean_dec(v_mid_5584_);
return v___x_5592_;
}
}
else
{
v_x_5581_ = v_mid_5584_;
goto _start;
}
}
else
{
uint8_t v___x_5594_; 
v___x_5594_ = lean_nat_dec_eq(v_mid_5584_, v_x_5580_);
if (v___x_5594_ == 0)
{
lean_dec(v_x_5580_);
v_x_5580_ = v_mid_5584_;
goto _start;
}
else
{
lean_object* v___x_5596_; lean_object* v_j_5597_; lean_object* v_as_5598_; lean_object* v___x_5599_; 
lean_dec(v_mid_5584_);
lean_dec(v_x_5581_);
v___x_5596_ = lean_nat_add(v_x_5580_, v___x_5583_);
lean_dec(v_x_5580_);
v_j_5597_ = lean_array_get_size(v_as_5578_);
v_as_5598_ = lean_array_push(v_as_5578_, v___x_5577_);
v___x_5599_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_5596_, v_as_5598_, v_j_5597_);
lean_dec(v___x_5596_);
return v___x_5599_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg___boxed(lean_object* v___x_5600_, lean_object* v_as_5601_, lean_object* v_k_5602_, lean_object* v_x_5603_, lean_object* v_x_5604_){
_start:
{
lean_object* v_res_5605_; 
v_res_5605_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg(v___x_5600_, v_as_5601_, v_k_5602_, v_x_5603_, v_x_5604_);
lean_dec_ref(v_k_5602_);
return v_res_5605_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0(lean_object* v___x_5606_, lean_object* v_as_5607_, lean_object* v_k_5608_){
_start:
{
lean_object* v___x_5609_; lean_object* v___x_5610_; uint8_t v___x_5611_; 
v___x_5609_ = lean_array_get_size(v_as_5607_);
v___x_5610_ = lean_unsigned_to_nat(0u);
v___x_5611_ = lean_nat_dec_eq(v___x_5609_, v___x_5610_);
if (v___x_5611_ == 0)
{
lean_object* v___x_5612_; uint8_t v___x_5613_; 
v___x_5612_ = lean_array_fget_borrowed(v_as_5607_, v___x_5610_);
v___x_5613_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v_k_5608_, v___x_5612_);
if (v___x_5613_ == 0)
{
uint8_t v___x_5614_; 
v___x_5614_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v___x_5612_, v_k_5608_);
if (v___x_5614_ == 0)
{
uint8_t v___x_5615_; 
v___x_5615_ = lean_nat_dec_lt(v___x_5610_, v___x_5609_);
if (v___x_5615_ == 0)
{
lean_dec_ref(v___x_5606_);
return v_as_5607_;
}
else
{
lean_object* v___x_5616_; lean_object* v_xs_x27_5617_; lean_object* v___x_5618_; 
v___x_5616_ = lean_box(0);
v_xs_x27_5617_ = lean_array_fset(v_as_5607_, v___x_5610_, v___x_5616_);
v___x_5618_ = lean_array_fset(v_xs_x27_5617_, v___x_5610_, v___x_5606_);
return v___x_5618_;
}
}
else
{
lean_object* v___x_5619_; lean_object* v___x_5620_; lean_object* v___x_5621_; uint8_t v___x_5622_; 
v___x_5619_ = lean_unsigned_to_nat(1u);
v___x_5620_ = lean_nat_sub(v___x_5609_, v___x_5619_);
v___x_5621_ = lean_array_fget_borrowed(v_as_5607_, v___x_5620_);
v___x_5622_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v___x_5621_, v_k_5608_);
if (v___x_5622_ == 0)
{
uint8_t v___x_5623_; 
v___x_5623_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___lam__0(v_k_5608_, v___x_5621_);
if (v___x_5623_ == 0)
{
uint8_t v___x_5624_; 
v___x_5624_ = lean_nat_dec_lt(v___x_5620_, v___x_5609_);
if (v___x_5624_ == 0)
{
lean_dec(v___x_5620_);
lean_dec_ref(v___x_5606_);
return v_as_5607_;
}
else
{
lean_object* v___x_5625_; lean_object* v_xs_x27_5626_; lean_object* v___x_5627_; 
v___x_5625_ = lean_box(0);
v_xs_x27_5626_ = lean_array_fset(v_as_5607_, v___x_5620_, v___x_5625_);
v___x_5627_ = lean_array_fset(v_xs_x27_5626_, v___x_5620_, v___x_5606_);
lean_dec(v___x_5620_);
return v___x_5627_;
}
}
else
{
lean_object* v___x_5628_; 
v___x_5628_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg(v___x_5606_, v_as_5607_, v_k_5608_, v___x_5610_, v___x_5620_);
return v___x_5628_;
}
}
else
{
lean_object* v___x_5629_; 
lean_dec(v___x_5620_);
v___x_5629_ = lean_array_push(v_as_5607_, v___x_5606_);
return v___x_5629_;
}
}
}
else
{
lean_object* v_as_5630_; lean_object* v___x_5631_; 
v_as_5630_ = lean_array_push(v_as_5607_, v___x_5606_);
v___x_5631_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_5610_, v_as_5630_, v___x_5609_);
return v___x_5631_;
}
}
else
{
lean_object* v___x_5632_; 
v___x_5632_ = lean_array_push(v_as_5607_, v___x_5606_);
return v___x_5632_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0___boxed(lean_object* v___x_5633_, lean_object* v_as_5634_, lean_object* v_k_5635_){
_start:
{
lean_object* v_res_5636_; 
v_res_5636_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0(v___x_5633_, v_as_5634_, v_k_5635_);
lean_dec_ref(v_k_5635_);
return v_res_5636_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg(lean_object* v_init_5637_, lean_object* v_x_5638_){
_start:
{
if (lean_obj_tag(v_x_5638_) == 0)
{
lean_object* v_k_5640_; lean_object* v_v_5641_; lean_object* v_l_5642_; lean_object* v_r_5643_; lean_object* v___x_5644_; lean_object* v_a_5645_; lean_object* v_a_5646_; lean_object* v_snd_5647_; lean_object* v___x_5649_; uint8_t v_isShared_5650_; uint8_t v_isSharedCheck_5656_; 
v_k_5640_ = lean_ctor_get(v_x_5638_, 1);
lean_inc(v_k_5640_);
v_v_5641_ = lean_ctor_get(v_x_5638_, 2);
lean_inc(v_v_5641_);
v_l_5642_ = lean_ctor_get(v_x_5638_, 3);
lean_inc(v_l_5642_);
v_r_5643_ = lean_ctor_get(v_x_5638_, 4);
lean_inc(v_r_5643_);
lean_dec_ref_known(v_x_5638_, 5);
v___x_5644_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg(v_init_5637_, v_l_5642_);
v_a_5645_ = lean_ctor_get(v___x_5644_, 0);
lean_inc(v_a_5645_);
lean_dec_ref(v___x_5644_);
v_a_5646_ = lean_ctor_get(v_a_5645_, 0);
lean_inc(v_a_5646_);
lean_dec(v_a_5645_);
v_snd_5647_ = lean_ctor_get(v_v_5641_, 1);
v_isSharedCheck_5656_ = !lean_is_exclusive(v_v_5641_);
if (v_isSharedCheck_5656_ == 0)
{
lean_object* v_unused_5657_; 
v_unused_5657_ = lean_ctor_get(v_v_5641_, 0);
lean_dec(v_unused_5657_);
v___x_5649_ = v_v_5641_;
v_isShared_5650_ = v_isSharedCheck_5656_;
goto v_resetjp_5648_;
}
else
{
lean_inc(v_snd_5647_);
lean_dec(v_v_5641_);
v___x_5649_ = lean_box(0);
v_isShared_5650_ = v_isSharedCheck_5656_;
goto v_resetjp_5648_;
}
v_resetjp_5648_:
{
lean_object* v___x_5652_; 
if (v_isShared_5650_ == 0)
{
lean_ctor_set(v___x_5649_, 0, v_k_5640_);
v___x_5652_ = v___x_5649_;
goto v_reusejp_5651_;
}
else
{
lean_object* v_reuseFailAlloc_5655_; 
v_reuseFailAlloc_5655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5655_, 0, v_k_5640_);
lean_ctor_set(v_reuseFailAlloc_5655_, 1, v_snd_5647_);
v___x_5652_ = v_reuseFailAlloc_5655_;
goto v_reusejp_5651_;
}
v_reusejp_5651_:
{
lean_object* v___x_5653_; 
lean_inc_ref(v___x_5652_);
v___x_5653_ = lp_batteries_Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0(v___x_5652_, v_a_5646_, v___x_5652_);
lean_dec_ref(v___x_5652_);
v_init_5637_ = v___x_5653_;
v_x_5638_ = v_r_5643_;
goto _start;
}
}
}
else
{
lean_object* v___x_5658_; lean_object* v___x_5659_; 
v___x_5658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5658_, 0, v_init_5637_);
v___x_5659_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5659_, 0, v___x_5658_);
return v___x_5659_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg___boxed(lean_object* v_init_5660_, lean_object* v_x_5661_, lean_object* v___y_5662_){
_start:
{
lean_object* v_res_5663_; 
v_res_5663_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg(v_init_5660_, v_x_5661_);
return v_res_5663_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__2(lean_object* v_msgData_5664_, lean_object* v___y_5665_, lean_object* v___y_5666_){
_start:
{
uint8_t v___x_5668_; uint8_t v___x_5669_; lean_object* v___x_5670_; 
v___x_5668_ = 0;
v___x_5669_ = 0;
v___x_5670_ = lp_batteries_Lean_log___at___00Lean_logError___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__4_spec__6(v_msgData_5664_, v___x_5668_, v___x_5669_, v___y_5665_, v___y_5666_);
return v___x_5670_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_logInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__2___boxed(lean_object* v_msgData_5671_, lean_object* v___y_5672_, lean_object* v___y_5673_, lean_object* v___y_5674_){
_start:
{
lean_object* v_res_5675_; 
v_res_5675_ = lp_batteries_Lean_logInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__2(v_msgData_5671_, v___y_5672_, v___y_5673_);
lean_dec(v___y_5673_);
lean_dec_ref(v___y_5672_);
return v_res_5675_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__2(void){
_start:
{
lean_object* v___x_5679_; lean_object* v___x_5680_; 
v___x_5679_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__1));
v___x_5680_ = l_Lean_stringToMessageData(v___x_5679_);
return v___x_5680_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1(lean_object* v_x_5681_, lean_object* v_a_5682_, lean_object* v_a_5683_){
_start:
{
lean_object* v___x_5685_; uint8_t v___x_5686_; 
v___x_5685_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint_command_x23list__linters___closed__1));
v___x_5686_ = l_Lean_Syntax_isOfKind(v_x_5681_, v___x_5685_);
if (v___x_5686_ == 0)
{
lean_object* v___x_5687_; 
v___x_5687_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23lint_x2b_x2d_x2aOnly________1_spec__0___redArg();
return v___x_5687_;
}
else
{
lean_object* v___x_5688_; lean_object* v_env_5689_; lean_object* v___x_5690_; lean_object* v_toEnvExtension_5691_; lean_object* v_asyncMode_5692_; lean_object* v_result_5693_; lean_object* v___x_5694_; lean_object* v___x_5695_; lean_object* v___x_5696_; lean_object* v___x_5697_; lean_object* v_a_5698_; lean_object* v_a_5700_; lean_object* v_a_5715_; 
v___x_5688_ = lean_st_ref_get(v_a_5683_);
v_env_5689_ = lean_ctor_get(v___x_5688_, 0);
lean_inc_ref(v_env_5689_);
lean_dec(v___x_5688_);
v___x_5690_ = lp_batteries_Batteries_Tactic_Lint_batteriesLinterExt;
v_toEnvExtension_5691_ = lean_ctor_get(v___x_5690_, 0);
v_asyncMode_5692_ = lean_ctor_get(v_toEnvExtension_5691_, 2);
v_result_5693_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__0));
v___x_5694_ = lean_box(1);
v___x_5695_ = lean_box(0);
v___x_5696_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_5694_, v___x_5690_, v_env_5689_, v_asyncMode_5692_, v___x_5695_);
v___x_5697_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg(v_result_5693_, v___x_5696_);
v_a_5698_ = lean_ctor_get(v___x_5697_, 0);
lean_inc(v_a_5698_);
lean_dec_ref(v___x_5697_);
v_a_5715_ = lean_ctor_get(v_a_5698_, 0);
lean_inc(v_a_5715_);
lean_dec(v_a_5698_);
v_a_5700_ = v_a_5715_;
goto v___jp_5699_;
v___jp_5699_:
{
lean_object* v___x_5701_; size_t v_sz_5702_; size_t v___x_5703_; lean_object* v___x_5704_; 
v___x_5701_ = lean_obj_once(&lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__2, &lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__2_once, _init_lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___closed__2);
v_sz_5702_ = lean_array_size(v_a_5700_);
v___x_5703_ = ((size_t)0ULL);
v___x_5704_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg(v_a_5700_, v_sz_5702_, v___x_5703_, v___x_5701_);
lean_dec_ref(v_a_5700_);
if (lean_obj_tag(v___x_5704_) == 0)
{
lean_object* v_a_5705_; lean_object* v___x_5706_; 
v_a_5705_ = lean_ctor_get(v___x_5704_, 0);
lean_inc(v_a_5705_);
lean_dec_ref_known(v___x_5704_, 1);
v___x_5706_ = lp_batteries_Lean_logInfo___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__2(v_a_5705_, v_a_5682_, v_a_5683_);
return v___x_5706_;
}
else
{
lean_object* v_a_5707_; lean_object* v___x_5709_; uint8_t v_isShared_5710_; uint8_t v_isSharedCheck_5714_; 
v_a_5707_ = lean_ctor_get(v___x_5704_, 0);
v_isSharedCheck_5714_ = !lean_is_exclusive(v___x_5704_);
if (v_isSharedCheck_5714_ == 0)
{
v___x_5709_ = v___x_5704_;
v_isShared_5710_ = v_isSharedCheck_5714_;
goto v_resetjp_5708_;
}
else
{
lean_inc(v_a_5707_);
lean_dec(v___x_5704_);
v___x_5709_ = lean_box(0);
v_isShared_5710_ = v_isSharedCheck_5714_;
goto v_resetjp_5708_;
}
v_resetjp_5708_:
{
lean_object* v___x_5712_; 
if (v_isShared_5710_ == 0)
{
v___x_5712_ = v___x_5709_;
goto v_reusejp_5711_;
}
else
{
lean_object* v_reuseFailAlloc_5713_; 
v_reuseFailAlloc_5713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5713_, 0, v_a_5707_);
v___x_5712_ = v_reuseFailAlloc_5713_;
goto v_reusejp_5711_;
}
v_reusejp_5711_:
{
return v___x_5712_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1___boxed(lean_object* v_x_5716_, lean_object* v_a_5717_, lean_object* v_a_5718_, lean_object* v_a_5719_){
_start:
{
lean_object* v_res_5720_; 
v_res_5720_ = lp_batteries_Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1(v_x_5716_, v_a_5717_, v_a_5718_);
lean_dec(v_a_5718_);
lean_dec_ref(v_a_5717_);
return v_res_5720_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1(lean_object* v_as_5721_, size_t v_sz_5722_, size_t v_i_5723_, lean_object* v_b_5724_, lean_object* v___y_5725_, lean_object* v___y_5726_){
_start:
{
lean_object* v___x_5728_; 
v___x_5728_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___redArg(v_as_5721_, v_sz_5722_, v_i_5723_, v_b_5724_);
return v___x_5728_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1___boxed(lean_object* v_as_5729_, lean_object* v_sz_5730_, lean_object* v_i_5731_, lean_object* v_b_5732_, lean_object* v___y_5733_, lean_object* v___y_5734_, lean_object* v___y_5735_){
_start:
{
size_t v_sz_boxed_5736_; size_t v_i_boxed_5737_; lean_object* v_res_5738_; 
v_sz_boxed_5736_ = lean_unbox_usize(v_sz_5730_);
lean_dec(v_sz_5730_);
v_i_boxed_5737_ = lean_unbox_usize(v_i_5731_);
lean_dec(v_i_5731_);
v_res_5738_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__1(v_as_5729_, v_sz_boxed_5736_, v_i_boxed_5737_, v_b_5732_, v___y_5733_, v___y_5734_);
lean_dec(v___y_5734_);
lean_dec_ref(v___y_5733_);
lean_dec_ref(v_as_5729_);
return v_res_5738_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3(lean_object* v_init_5739_, lean_object* v_x_5740_, lean_object* v___y_5741_, lean_object* v___y_5742_){
_start:
{
lean_object* v___x_5744_; 
v___x_5744_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___redArg(v_init_5739_, v_x_5740_);
return v___x_5744_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3___boxed(lean_object* v_init_5745_, lean_object* v_x_5746_, lean_object* v___y_5747_, lean_object* v___y_5748_, lean_object* v___y_5749_){
_start:
{
lean_object* v_res_5750_; 
v_res_5750_ = lp_batteries_Std_DTreeMap_Internal_Impl_forInStep___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__3(v_init_5745_, v_x_5746_, v___y_5747_, v___y_5748_);
lean_dec(v___y_5748_);
lean_dec_ref(v___y_5747_);
return v_res_5750_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0(lean_object* v___x_5751_, lean_object* v_as_5752_, lean_object* v_k_5753_, lean_object* v_x_5754_, lean_object* v_x_5755_, lean_object* v_x_5756_, lean_object* v_x_5757_){
_start:
{
lean_object* v___x_5758_; 
v___x_5758_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___redArg(v___x_5751_, v_as_5752_, v_k_5753_, v_x_5754_, v_x_5755_);
return v___x_5758_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0___boxed(lean_object* v___x_5759_, lean_object* v_as_5760_, lean_object* v_k_5761_, lean_object* v_x_5762_, lean_object* v_x_5763_, lean_object* v_x_5764_, lean_object* v_x_5765_){
_start:
{
lean_object* v_res_5766_; 
v_res_5766_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00Batteries_Tactic_Lint___aux__Batteries__Tactic__Lint__Frontend______elabRules__Batteries__Tactic__Lint__command_x23list__linters__1_spec__0_spec__0(v___x_5759_, v_as_5760_, v_k_5761_, v_x_5762_, v_x_5763_, v_x_5764_, v_x_5765_);
lean_dec_ref(v_k_5761_);
return v_res_5766_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_5831_; uint8_t v___x_5832_; lean_object* v___x_5833_; lean_object* v___x_5834_; 
v___x_5831_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic_Lint_lintCore_spec__9___closed__2));
v___x_5832_ = 0;
v___x_5833_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn___closed__24_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_));
v___x_5834_ = l_Lean_registerTraceClass(v___x_5831_, v___x_5832_, v___x_5833_);
return v___x_5834_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2____boxed(lean_object* v_a_5835_){
_start:
{
lean_object* v_res_5836_; 
v_res_5836_ = lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_();
return v_res_5836_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Frontend(uint8_t builtin) {
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
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Lint_Frontend(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity_default = _init_lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity_default();
lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity = _init_lp_batteries_Batteries_Tactic_Lint_instInhabitedLintVerbosity();
res = lp_batteries___private_Batteries_Tactic_Lint_Frontend_0__Batteries_Tactic_Lint_initFn_00___x40_Batteries_Tactic_Lint_Frontend_971841226____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Lint_Frontend(uint8_t builtin) {
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
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Frontend(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Lint_Frontend(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Lint_Frontend(builtin);
}
#ifdef __cplusplus
}
#endif
